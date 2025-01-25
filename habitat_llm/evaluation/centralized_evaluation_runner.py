#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains the CentralizedEvaluationRunner class, which implements a centralized evaluation strategy
where a single planner coordinates actions across multiple agents. It extends the base EvaluationRunner class
to provide functionality for running multi-agent evaluations with centralized control.
"""

from typing import TYPE_CHECKING, Any, Dict, Tuple

from hydra.utils import instantiate

from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.evaluation import EvaluationRunner
from habitat_llm.planner.planner import Planner
from habitat_llm.tools.motor_skills.motor_skill_tool import MotorSkillTool

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.world_model import WorldGraph


# Evaluation runner, will go over episodes, run planners and store necessary data
class CentralizedEvaluationRunner(EvaluationRunner):
    """
    Evaluation runner that implements a centralized control strategy, where a single planner
    coordinates actions across multiple agents. This class handles episode execution,
    planner initialization, and action generation for all agents in a centralized manner.
    """

    def __init__(
        self,
        evaluation_runner_config_arg: "DictConfig",  # Hydra config type
        env_arg: EnvironmentInterface,
    ) -> None:
        """
        Initialize the CentralizedEvaluationRunner.

        :param evaluation_runner_config_arg: Configuration object containing evaluation parameters
                                           including agent and planner configurations
        :param env_arg: Environment interface instance that provides access to the simulation
        """
        super().__init__(evaluation_runner_config_arg, env_arg)

    def get_low_level_actions(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Given a set of observations, gets a vector of low level actions for all agents.

        :param instruction: Natural language instruction describing the task to execute
        :param observations: Dictionary of observations from the environment
        :param world_graph: Dictionary mapping agent IDs to their world graph states

        :return: Tuple containing:
            - List of low level action dictionaries for each agent
            - Dictionary with planner info about high level actions
            - Boolean indicating whether execution should end
        """
        low_level_actions, planner_info, should_end = self.planner.get_next_action(
            instruction, observations, world_graph
        )
        return low_level_actions, planner_info, should_end

    def reset_planners(self) -> None:
        """Reset the centralized planner to prepare for a new episode."""
        assert isinstance(self.planner, Planner)
        self.planner.reset()

    def _initialize_planners(self) -> None:
        """
        Initialize the centralized planner based on the evaluation runner configuration.
        Sets up the planner with appropriate agents and configures special planning modes
        if specified.
        """
        planner_conf = self.evaluation_runner_config.planner
        planner = instantiate(planner_conf)
        self.planner: Planner = planner(env_interface=self.env_interface)

        # Set both agents to the planner
        self.planner.agents = [
            self.agents[agent_id] for agent_id in sorted(self.agents.keys())
        ]
        if (
            "plan_config" in planner_conf
            and "planning_mode" in planner_conf.plan_config
            and planner_conf.plan_config.planning_mode == "st"
        ):
            for agent in self.planner.agents:
                for tool in agent.tools.values():
                    if isinstance(tool, MotorSkillTool):
                        tool.error_mode = "st"
