#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from hydra.utils import instantiate

from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.evaluation import EvaluationRunner
from habitat_llm.tools.motor_skills.motor_skill_tool import MotorSkillTool


# Evaluation runner, will go over episodes, run planners and store necessary data
class CentralizedEvaluationRunner(EvaluationRunner):
    def __init__(self, evaluation_runner_config_arg, env_arg: EnvironmentInterface):
        # Call EvaluationRunnerclass constructor
        super().__init__(evaluation_runner_config_arg, env_arg)

    def get_low_level_actions(self, instruction, observations, world_graph):
        """
        Given a set of observations, gets a vector of low level actions, an info dictionary and a boolean indicating that the run should end.
        """
        low_level_actions, planner_info, should_end = self.planner.get_next_action(
            instruction, observations, world_graph
        )
        return low_level_actions, planner_info, should_end

    def reset_planners(self):
        self.planner.reset()

    def _initialize_planners(self):
        planner_conf = self.evaluation_runner_config.planner
        planner = instantiate(planner_conf)
        self.planner = planner(env_interface=self.env_interface)

        # Set both agents to the planner
        self.planner.agents = [
            self.agents[agent_id] for agent_id in sorted(self.agents.keys())
        ]

        if (
            "planning_mode" in planner_conf.plan_config
            and planner_conf.plan_config.planning_mode == "st"
        ):
            for agent in self.planner.agents:
                for tool in agent.tools.values():
                    if isinstance(tool, MotorSkillTool):
                        tool.error_mode = "st"
