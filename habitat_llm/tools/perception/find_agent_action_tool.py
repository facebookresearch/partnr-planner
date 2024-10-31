#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import List

from habitat_llm.tools import PerceptionTool, get_prompt


class FindAgentActionTool(PerceptionTool):
    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.llm = None
        self.env_interface = None
        self.skill_config = skill_config
        self.prompt_maker = None
        self.wait_count = 0

    def set_environment(self, env):
        self.env_interface = env

    def set_llm(self, llm):
        self.llm = llm
        self.prompt_maker = get_prompt(self.skill_config.prompt, self.llm.llm_conf)

    @property
    def description(self) -> str:
        return self.skill_config.description

    def _get_state_history(self):
        """Method to get state history of the other agent"""

        # Set other agent id - assumes there are only two agents named 0 and 1
        other_agent_id = 1 - self.agent_uid

        if len(self.env_interface.agent_state_history[other_agent_id]) == 0:
            return None

        history_elements = self.env_interface.agent_state_history[other_agent_id]
        states = [el.state for el in history_elements]
        # Construct the state history
        return ", ".join(states)

    def process_high_level_action(self, input_query, observation):
        if not self.env_interface:
            raise ValueError("Environment Interface not set, use set_environment")

        # Wait for a few steps to give the other agent
        # chance to process the recently assigned action
        if self.wait_count < 10:
            self.wait_count += 1
            return None, ""

        self.wait_count = 0

        # Extract state history from environment
        state_history = self._get_state_history()

        if state_history == None:
            return (
                None,
                "Information about the states of other agent is not available. Try again after sometime.",
            )

        # Create prompt
        prompt = self.prompt_maker(state_history, verbose=False)

        # Execute llm query
        answer = self.llm.generate(prompt, stop="<Done>", max_length=250)

        # Handle the edge case where answer is empty or only spaces
        if answer == "" or answer.isspace():
            answer = "Could not find any state history for the other agent."

        return None, answer

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the FindObjectTool.

        :return: List of argument types.
        """
        return []
