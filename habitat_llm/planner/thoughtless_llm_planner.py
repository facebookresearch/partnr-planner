#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import functools
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from habitat_llm.llm.instruct.utils import build_single_step_prompt
from habitat_llm.planner import LLMPlanner
from habitat_llm.tools.tool import PerceptionTool

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent import Agent
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph


class ThoughtlessLLMPlanner(LLMPlanner):
    """
    This class builds the prompt for the single step, thoughtless (i.e. no chain of thought prompting) format
    used for LLMs finetuned on  habitat-llm data.
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ):
        """
        Initialize the ThoughtlessLLMPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        super().__init__(plan_config, env_interface)
        self.stopword: str = "<end_act>"
        self.end_expression: str = "Done"

        # Cache the actions parser function, this will be partially applied later
        self._actions_parser: Callable[
            [int, List[Agent], str, Optional[Dict[str, Any]]],
            Dict[int, Tuple[str, str, str]],
        ] = self.actions_parser

        self.prompt_history: List[str] = []
        self.prompt_header: str = "Solve the given multi-agent planning problem as best as you can. The task assigned to you will be situated in a house and will generally involve navigating to objects, picking and placing them on different receptacles to achieve rearrangement. Below is the detailed description of the actions you can use for solving the task. You can assign them to Agent_0 and/or Agent_1 as required."

    def reset(self) -> None:
        """
        Reset the planner state.
        """
        self.prompt_history = []
        return super().reset()

    # Ideally the result of this should be cached on reset but the first reset
    # is called before the agents are initialized so there's
    # no convenient hook where we can cache this at the correct time.
    def _get_perception_tool_names(self, agent: "Agent") -> List[str]:
        """
        Get the names of perception tools for the given agent.

        :param agent: The agent to get perception tool names for.
        :return: A list of perception tool names.
        """
        perception_tools: List[str] = []
        for tool_name, tool in agent.tools.items():
            if isinstance(tool, PerceptionTool):
                perception_tools.append(tool_name)
        return perception_tools

    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        :return: A string representing the grammar for valid responses.
        """
        tool_rules: str = self.build_tool_grammar(world_graph)
        root_role: str = f'root ::= tool_call "{self.stopword}"'
        return "\n".join([root_role, tool_rules])

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        verbose: bool = False,
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Get the next low-level action to execute.

        :param instruction: The instruction for the task.
        :param observations: The current observations.
        :param world_graph: The world graph for each agent.
        :param verbose: Whether to print verbose output.
        :return: A tuple containing:
                 - The low-level actions for each agent
                 - Planner information
                 - Whether the planner is done
        """
        assert len(self.agents) == 1
        agent = self.agents[0]
        prompt_string: str = build_single_step_prompt(
            instruction,
            world_graph[agent.uid],
            str(agent.uid),
            self.env_interface.agent_action_history,
            tools_to_skip=self._get_perception_tool_names(agent),
        )
        self.curr_prompt = prompt_string
        # provide the agent id to the actions parse which is required for this type of planner
        self.actions_parser = functools.partial(self._actions_parser, agent.uid)
        previous_replanning_count = self.replanning_count

        low_level_actions, planner_info, is_done = super().get_next_action(
            instruction, observations, world_graph, verbose
        )
        # Detect that replanning occurred
        if self.replanning_count > previous_replanning_count:
            self.prompt_history.append(self.curr_prompt)
        if "responses" in planner_info and planner_info["responses"][agent.uid] != "":
            # This isn't really part of the prompting but this "prompts" key is what is written out to the log on disk
            # and it is useful to have the agent's response in the log
            self.prompt_history.append(
                "Observation: " + planner_info["responses"][agent.uid]
            )
        # Stack all prompts together for logging because future prompts do not necessarily contain the previous prompt
        planner_info["prompts"] = {
            agent.uid: "\n-----------------\n".join(self.prompt_history)
        }
        return low_level_actions, planner_info, is_done
