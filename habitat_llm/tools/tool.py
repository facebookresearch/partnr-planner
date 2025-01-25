#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
An abstract class for tools. A tool is a callable object that takes a dictionary
of inputs and returns a dictionary of outputs. The inputs and outputs are specified
by the attributes "input_keys" and "output_keys", respectively. The tool also has
a description and a name.

Examples of tools include low-level actions to the environment, perception
tools, and language generation tools. The requirement is that the inputs and
outputs are in LANGUAGE so that it can be sent to the LLM.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class Tool(ABC):
    # Parameterized Constructor
    def __init__(self, name_arg, agent_uid_arg=0):
        self.name = name_arg
        self.agent_uid = agent_uid_arg

    # Hashing Operator
    def __hash__(self):
        return hash(self.name)

    # Equality operator
    def __eq__(self, other):
        #  Make sure that the other is of type Tool
        if not isinstance(other, Tool):
            return False

        # Make sure that the name is same
        if self.name != other.name:
            return False

        # Make sure that the agent uid is same
        if self.agent_uid != other.agent_uid:
            return False

        return True

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def process_high_level_action(self, input_query, observations):
        pass

    @property
    @abstractmethod
    def argument_types(self) -> List[str]:
        raise NotImplementedError()

    def grammar(self):
        argument_string = ' "," WS '.join(self.argument_types)
        return f'"{self.name}[" { argument_string } "]"'


class PerceptionTool(Tool):
    """
    Creating tool-specific types to discriminate between motor skills and perception
    skills. Mainly supports dry_run use-case, where user wants planner to only run
    through the perception-tools without moving agents or objects around in the world.

    Example of when to use dry-run: When your WM is not integrated with skills, or
    trying to test the perception tools in isolation or when you want to test the
    full planner cycles without moving the agents around.
    """

    def __init__(self, name_arg, agent_uid_arg=0):
        super().__init__(name_arg, agent_uid_arg)
        self.env_interface = None

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def process_high_level_action(self, input_query, observations) -> Tuple[None, str]:
        if not self.env_interface:
            raise ValueError(
                f"Environment interface not set in the {self.__class__.__name__}"
            )
        return None, ""

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        room_node = self.env_interface.world_graph[self.agent_uid].get_room_for_entity(
            f"agent_{self.agent_uid}"
        )
        return f"Standing in {room_node.name}"
