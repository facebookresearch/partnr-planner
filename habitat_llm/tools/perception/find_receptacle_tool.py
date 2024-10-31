#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from habitat_llm.tools import PerceptionTool, get_prompt
from habitat_llm.utils.grammar import FREE_TEXT


class FindReceptacleTool(PerceptionTool):
    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.llm = None
        self.env_interface = None
        self.skill_config = skill_config
        self.prompt_maker = None

    def set_environment(self, env_interface):
        self.env_interface = env_interface

    def set_llm(self, llm):
        self.llm = llm
        self.prompt_maker = get_prompt(self.skill_config.prompt, self.llm.llm_conf)

    @property
    def description(self) -> str:
        return self.skill_config.description

    def _get_receptacles_list(self):
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

    def process_high_level_action(self, input_query, observations):
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
    def argument_types(self):
        """
        Returns the types of arguments required for the FindObjectTool.

        :return: List of argument types.
        """
        return [FREE_TEXT]
