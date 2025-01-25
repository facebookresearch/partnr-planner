#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
This module contains FindRoomTool, a PerceptionTool, used by tool-based LLM to
find exact identifier for a Room node in the world-graph given some natural
language description as input
"""

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface
    from habitat_llm.llm.base_llm import BaseLLM

from habitat_llm.tools import PerceptionTool, get_prompt
from habitat_llm.utils.grammar import FREE_TEXT


class FindRoomTool(PerceptionTool):
    """
    PerceptionTool used by tool-based LLM planner to get the exact identifier for a
    Room node given natural language description of the receptacle/furniture
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
        property to return the description of this tool as provided in configuration
        :return: tool description
        """
        return self.skill_config.description

    def process_high_level_action(self, input_query, observations) -> Tuple[None, str]:
        """
        Main entry-point, takes the natural language room query as input and the latest
        observation. Returns the exact name of the Room node matching query.

        :param input_query: Natural language description of room of interest
        :param observations: Dict of agent's observations

        :return: Tuple[None, str] where 2nd element is the exact name of the node matching
        input-query or message explaining such room was not found
        """
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
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the FindRoomTool.

        :return: List of argument types.
        """
        return [FREE_TEXT]
