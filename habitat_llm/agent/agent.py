#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Class containing agent definition, which is in charge of inferring which actions
to take for a given task. Most agents operate by calling an LLM
to infer the next action.
"""

from typing import TYPE_CHECKING, Dict, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from habitat_llm.tools import PerceptionTool

if TYPE_CHECKING:
    import torch

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.llm import BaseLLM
    from habitat_llm.tools import Tool


class Agent:
    """
    This class represents an agent, which decides which action should be called
    at every time step. The agent has access to tools, which will convert high
    level actions into low-level control commands.
    """

    def __init__(
        self,
        uid: int,
        agent_conf: DictConfig,
        env_interface: Optional["EnvironmentInterface"] = None,
    ) -> None:
        """
        Initialize Agent
        :param uid: integer representing a unique id of the agent.
        :param agent_conf: configuration for the agent.
        :param env_interface: the environment where this agent operates
        """
        # Assign the unique ID
        self.uid = uid

        self.agent_conf = agent_conf
        self.env_interface = env_interface
        self._dry_run = False

        # Initialize tools
        self.tools = self.__init_tools()

        # Log last used tool
        self.last_used_tool: Optional["Tool"] = None

    # Hashing Operator
    def __hash__(self) -> int:
        return hash(self.uid)

    # Equality operator
    def __eq__(self, other) -> bool:
        #  Make sure that the other is of type Tool
        if not isinstance(other, Agent):
            return False

        # Return the equivalence based on the uid
        return self.uid == other.uid

    def reset(self) -> None:
        """Resets state variables of the agent"""

        # Reset state variables
        self.last_used_tool = None
        self._dry_run = False

        # Reset skills owned by this agent
        for tool_name in self.tools:
            if hasattr(self.tools[tool_name], "skill"):
                # Reset the skill's internal variables such as hidden states
                # and flag for tracking when the skill is entered for the first time
                self.tools[tool_name].skill.reset([0])  # type: ignore
                # print(f"Reset {tool_name} skill for agent {self.uid}...")

    @property
    def agent_description(self) -> str:
        """Returns a string listing the agent's descriptions"""

        out = ""
        out += f"Agent ID: {self.uid}\n"
        for tool_name in sorted(self.tools.keys()):
            out += f"- {tool_name}: {self.tools[tool_name].description}\n"

        out += "\n"

        return out

    @property
    def tool_list(self) -> str:
        """Returns a string listing the agent's tools"""
        tool_names = sorted(self.tools.keys())
        return str(tool_names)

    @property
    def tool_descriptions(self):
        """Returns a string listing the agent's tools with its descriptions"""
        tool_names = sorted(self.tools.keys())
        return "\n".join(
            [
                f"- {tool_name}: {self.tools[tool_name].description}"
                for tool_name in tool_names
            ]
        )

    def __init_tools(self) -> Dict[str, "Tool"]:
        # Declare as set to ensure uniqueness
        tools = {}

        for tool_category in self.agent_conf.tools:
            for tool_name in self.agent_conf.tools[tool_category]:
                print(f'processing tool: "{tool_name}"')
                tool_config = self.agent_conf.tools[tool_category][tool_name]
                tool = instantiate(tool_config)
                tool.agent_uid = self.uid

                # Motor skills require access to the environment and the device
                if "motor_skills" in tool_category:
                    tool.set_environment(self.env_interface)
                    tool.to(self.env_interface.device)

                # Perception requires access to the environment
                if "perception" in tool_category:
                    tool.set_environment(self.env_interface)

                # Make sure tool is not already added in the set
                if tool in tools:
                    print(f'tools already contains an instance of "{tool_name}"')
                    raise ValueError(
                        f'tools already contains an instance of "{tool_name}"'
                    )

                # Add tool to the set
                tools[tool.name] = tool

        return tools

    def pass_llm_to_tools(self, llm: "BaseLLM"):
        """
        This method passes a given instance of LLM into the agent tools.
        Some tools require LLMs for their operation. However, maintaining copies of LLMs in memory
        is expensive, so we need to share the instance of planner LLM across the tools.
        :param llm: The llm that drives this agent.
        """
        for tool in self.tools.values():
            if hasattr(tool, "llm"):
                tool.set_llm(llm)  # type: ignore

        return

    def get_tool_from_name(self, tool_name: str) -> "Tool":
        """
        Get a tool with a given name.
        :param tool_name: the name of the tool
        """
        if tool_name in self.tools:
            return self.tools[tool_name]
        raise ValueError(f'Tool "{tool_name}" not found')

    def get_last_state_description(self) -> str:
        """
        Obtain the last tool that the agent used.
        """
        if self.last_used_tool == None:
            return "Idle"

        return self.last_used_tool.get_state_description()  # type: ignore

    def process_high_level_action(
        self, action: str, action_input: str, observations: Dict[str, "torch.Tensor"]
    ) -> tuple["torch.Tensor", str]:
        """
        This method will consume high level actions to generate
        either a text response or a low level action.
        For every empty text response, this method should return a non empty low-level action.
        :param action: The name of the high level action (e.g. Pick, Place)
        :param action_input: The argument of the action
        :param observations: current agent observations
        """
        # Fetch tool corresponding to the action
        try:
            tool = self.get_tool_from_name(action)
        except ValueError:
            return None, f'Tool "{action}" not found'

        # Process the high level action
        if self._dry_run and not isinstance(tool, PerceptionTool):
            return None, f"{action} was a success"
        low_level_action, response = tool.process_high_level_action(
            action_input, observations
        )

        # Set last used tool
        self.last_used_tool = tool

        return low_level_action, response
