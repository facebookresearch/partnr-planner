#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
"""
This module contains the DecentralizedEvaluationRunner class, which implements a decentralized evaluation strategy
where multiple planners independently control different agents. It extends the base EvaluationRunner class
to provide functionality for running multi-agent evaluations with decentralized control.
"""

from typing import TYPE_CHECKING, Any, Dict, Tuple

from hydra.utils import instantiate

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from habitat_llm.world_model.world_graph import WorldGraph

from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.evaluation import EvaluationRunner
from habitat_llm.tools.motor_skills.motor_skill_tool import MotorSkillTool


# Evaluation runner, will go over episodes, run planners and store necessary data
class DecentralizedEvaluationRunner(EvaluationRunner):
    """
    Evaluation runner that implements a decentralized control strategy where separate planners
    independently control different agents. This class handles episode execution, planner initialization,
    and action generation with each agent having its own dedicated planner.
    """

    def __init__(
        self,
        evaluation_runner_config_arg: "DictConfig",  # Hydra config type
        env_arg: EnvironmentInterface,
    ) -> None:
        # Call EvaluationRunner class constructor
        super().__init__(evaluation_runner_config_arg, env_arg)

    def _initialize_planners(self) -> None:
        """
        Initialize separate planners for each agent based on the evaluation runner configuration.
        Sets up each planner with its corresponding agent and configures special planning modes
        if specified.
        """
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
    def __str__(self) -> str:
        """
        Return string with state of the evaluator
        """
        assert isinstance(self.planner, dict)
        planner_str = " ".join(
            [
                f"{planner_id}:{type(planner_val)}"
                for planner_id, planner_val in self.planner.items()
            ]
        )
        out = f"Decentralized Planner: {planner_str}\n"
        out += f"Number of Agents: {len(self.agents)}"
        return out

    def reset_planners(self) -> None:
        """
        Method to reset planner parameters.
        Usually called after finishing one episode.
        """
        assert isinstance(self.planner, dict)
        for planner in self.planner.values():
            planner.reset()

    def get_low_level_actions(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Given a set of observations, gets a vector of low level actions,
        an info dictionary and a boolean indicating that the run should end.

        :param instruction: Natural language instruction describing the task to execute
        :param observations: Dictionary of observations from the environment
        :param world_graph: Dictionary mapping agent IDs to their world graph states

        :return: Tuple containing:
            - Dictionary mapping agent IDs to their low level actions
            - Dictionary with planner info about high level actions
            - Boolean indicating whether all planners are done
        """
        # Declare container to store planned low level actions
        # from all planners
        low_level_actions: Dict[int, Any] = {}

        # Declare container to store planning info from all planners
        planner_info: Dict[str, Any] = {}

        # Marks the end of all planners
        all_planners_are_done = True

        assert isinstance(self.planner, dict)
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
