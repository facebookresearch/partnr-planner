#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""Implements FindAgentActionTool which is used by tool-based LLM planner
 to gather information about other agent's actions during planning """

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface
    from habitat_llm.llm.base_llm import BaseLLM

from habitat_llm.tools import PerceptionTool, get_prompt


class FindAgentActionTool(PerceptionTool):
    """
    A PerceptionTool designed to be used by an agent's LLM planner to gather information
    about other agent's actions during the episode
    """

    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.llm = None
        self.env_interface = None
        self.skill_config = skill_config
        self.prompt_maker = None
        self.wait_count = 0

    def set_environment(self, env: "EnvironmentInterface") -> None:
        """
        Sets the tool's environment_interface var

        :param env: EnvironmentInterface instance associated with episode
        """
        self.env_interface = env

    def set_llm(self, llm: "BaseLLM") -> None:
        """
        Sets the tool's LLM interface object

        :param llm: LLM object to be used for generating responses
        """
        self.llm = llm
        self.prompt_maker = get_prompt(self.skill_config.prompt, self.llm.llm_conf)

    @property
    def description(self) -> str:
        """
        Property to return the description of this tool as provided in configuration.

        :return: The description of this tool provided in figuration.
        """
        return self.skill_config.description

    def _get_state_history(self) -> str:
        """Method to get state history of the other agent"""

        # Set other agent id - assumes there are only two agents named 0 and 1
        other_agent_id = 1 - self.agent_uid

        if len(self.env_interface.agent_state_history[other_agent_id]) == 0:
            return None

        history_elements = self.env_interface.agent_state_history[other_agent_id]
        states = [el.state for el in history_elements]
        # Construct the state history
        return ", ".join(states)

    def process_high_level_action(
        self, input_query: str, observations: dict
    ) -> Tuple[None, str]:
        """
        Main entry-point method, containing logic to gather relevant information from
        episode-state, create a prompt for LLM, generate and parse response to return
        the summarized history of actions taken by the other agent

        :param input_query: Action inputs
        :param observation: Observation dict

        :return: The summarized history of actions taken by the other agent
        """

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
        Returns the types of arguments required for this tool

        :return: List of argument types.
        """
        return []  # NOTE: Empty as this tool is state-based, not input-based
