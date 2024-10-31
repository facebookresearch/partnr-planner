#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from habitat_llm.tools import PerceptionTool, get_prompt
from habitat_llm.utils.grammar import FREE_TEXT


class FindRoomTool(PerceptionTool):
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

    def process_high_level_action(self, input_query, observations):
        super().process_high_level_action(input_query, observations)

        if not self.llm:
            raise ValueError(f"LLM not set in the {self.__class__.__name__}")

        # get room-list
        room_list = self.env_interface.world_graph[self.agent_uid].get_all_rooms()

        # create prompt
        room_string = "".join([f"- {room.name}\n" for room in room_list])
        # print("FindRoomTool-->", room_string)

        # Handle the case of input_query is None
        if input_query is None:
            response = "I could not find any room matching the query since input_query is not given."
            return None, response

        prompt = self.prompt_maker(room_string, input_query)

        # execute llm query
        response = self.llm.generate(prompt, stop="<Done>", max_length=250)
        # print("FindRoomTool-->", response)

        # handle the edge case where answer is empty or only spaces
        if not response.strip():
            response = "I could not find any room matching the query."

        return None, response

    @property
    def argument_types(self):
        """
        Returns the types of arguments required for the FindObjectTool.

        :return: List of argument types.
        """
        return [FREE_TEXT]
