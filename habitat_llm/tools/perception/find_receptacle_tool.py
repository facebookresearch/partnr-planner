#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
This module contains FindReceptacleTool, a PerceptionTool, used by tool-based LLM to
find exact identifier for a Furniture node in the world-graph given some natural
language description as input
"""

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM
    from habitat_llm.agent.env import EnvironmentInterface

from habitat_llm.tools import PerceptionTool, get_prompt
from habitat_llm.utils.grammar import FREE_TEXT


class FindReceptacleTool(PerceptionTool):
    """
    PerceptionTool used by tool-based LLM planner to get the exact identifier for a
    furniture node given natural language description of the receptacle/furniture
    """

    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.llm = None
        self.env_interface = None
        self.skill_config = skill_config
        self.prompt_maker = None

    def set_environment(self, env_interface: "EnvironmentInterface") -> None:
        """
        Sets the tool's environment_interface var

        :param env: EnvironmentInterface instance associated with episode
        """
        self.env_interface = env_interface

    def set_llm(self, llm: "BaseLLM"):
        """
        Sets the tool's LLM interface object

        :param llm: LLM object to be used for generating responses
        """
        self.llm = llm
        self.prompt_maker = get_prompt(self.skill_config.prompt, self.llm.llm_conf)

    @property
    def description(self) -> str:
        """
        Property to return the description of this tool as provided in configuration

        :return: tool description
        """
        return self.skill_config.description

    def _get_receptacles_list(self) -> str:
        """
        Helper function to extract furniture-name list from the world-graph stored in self.env_interface
        """
        # Right now we are modeling receptacles as injective with furniture, so we can use the furniture list
        grouped_furniture = self.env_interface.world_graph[
            self.agent_uid
        ].group_furniture_by_type()
        fur_to_room_map = self.env_interface.world_graph[
            self.agent_uid
        ].get_furniture_to_room_map()
        # Combine the information
        combined_info = ""
        for furniture_type, furniture_node_list in grouped_furniture.items():
            combined_info += furniture_type.capitalize() + " : "
            for fur in furniture_node_list:
                component_string = ""
                if len(fur.properties.get("components", [])) > 0:
                    component_string = " with components: " + ", ".join(
                        fur.properties["components"]
                    )
                combined_info += (
                    fur.name
                    + " in "
                    + fur_to_room_map[fur].properties["type"]
                    + component_string
                    + ", "
                )
            combined_info = combined_info[:-2] + "\n"
        return combined_info

    def process_high_level_action(self, input_query, observations) -> Tuple[None, str]:
        """
        Main entry-point, takes the natural language furniture query as input and the latest
        observation. Returns the exact name of the Furniture node matching query.
        :param input_query: Natural language description of furniture of interest
        :param observations: Dict of agent's observations

        :return: Tuple[None, str], where 2nd element is the exact name of the node matching
        input-quer, or a message explaining such furniture was not found
        """
        super().process_high_level_action(input_query, observations)

        if not self.llm:
            raise ValueError(f"LLM not set in the {self.__class__.__name__}")

        # Extract receptacles from environment
        receptacles = self._get_receptacles_list()

        # Create prompt
        prompt = self.prompt_maker(input_query, receptacles)

        # Execute llm query
        answer = self.llm.generate(prompt, stop="<Done>", max_length=100)

        # Handle the edge case where answer is empty or only spaces
        if answer == "" or answer.isspace():
            answer = f"Could not find any receptacles in world for the query '{input_query}'."

        return None, answer

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the FindObjectTool.

        :return: List of argument types.
        """
        return [FREE_TEXT]
