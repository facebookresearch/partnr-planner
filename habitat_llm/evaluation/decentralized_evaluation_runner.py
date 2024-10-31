#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from hydra.utils import instantiate

from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.evaluation import EvaluationRunner
from habitat_llm.tools.motor_skills.motor_skill_tool import MotorSkillTool


# Evaluation runner, will go over episodes, run planners and store necessary data
class DecentralizedEvaluationRunner(EvaluationRunner):
    def __init__(self, evaluation_runner_config_arg, env_arg: EnvironmentInterface):
        # Call EvaluationRunner class constructor
        super().__init__(evaluation_runner_config_arg, env_arg)

    def _initialize_planners(self):
        self.planner = {}

        # Set an agent to each planner
        for agent_conf in self.evaluation_runner_config.agents.values():
            planner_conf = agent_conf.planner
            planner = instantiate(planner_conf)
            planner = planner(env_interface=self.env_interface)
            planner.agents = [self.agents[agent_conf.uid]]
            self.planner[agent_conf.uid] = planner
            if (
                "planning_mode" in planner_conf.plan_config
                and planner_conf.plan_config.planning_mode == "st"
            ):
                for v in self.planner.values():
                    for agent in v.agents:
                        for tool in agent.tools.values():
                            if isinstance(tool, MotorSkillTool):
                                tool.error_mode = "st"

    # Method to print the object
    def __str__(self):
        """
        Return string with state of the evaluator
        """
        planner_str = " ".join(
            [
                f"{planner_id}:{type(planner_val)}"
                for planner_id, planner_val in self.planner.items()
            ]
        )
        out = f"Decentralized Planner: {planner_str}\n"
        out += f"Number of Agents: {len(self.agents)}"
        return out

    def reset_planners(self):
        """
        Method to reset planner parameters.
        Usually called after finishing one episode.
        """
        for planner in self.planner.values():
            planner.reset()

    def get_low_level_actions(self, instruction, observations, world_graph):
        """
        Given a set of observations, gets a vector of low level actions,
        an info dictionary and a boolean indicating that the run should end.
        """
        # Declare container to store planned low level actions
        # from all planners
        low_level_actions = {}

        # Declare container to store planning info from all planners
        planner_info = {}

        # Marks the end of all planners
        all_planners_are_done = True

        # Loop through all available planners
        for planner in self.planner.values():
            # Get next action for this planner
            (
                this_planner_low_level_actions,
                this_planner_info,
                this_planner_is_done,
            ) = planner.get_next_action(instruction, observations, world_graph)
            # Update the output dictionary with planned low level actions
            low_level_actions.update(this_planner_low_level_actions)

            # Merges this_planner_info from all planners
            for key, val in this_planner_info.items():
                if type(val) == dict:
                    if key not in planner_info:
                        planner_info[key] = {}
                    planner_info[key].update(val)
                elif type(val) == str:
                    if key not in planner_info:
                        planner_info[key] = ""
                    planner_info[key] += val
                else:
                    raise ValueError(
                        "Logging entity can only be a dictionary or string!"
                    )

            all_planners_are_done = this_planner_is_done and all_planners_are_done

        return low_level_actions, planner_info, all_planners_are_done
