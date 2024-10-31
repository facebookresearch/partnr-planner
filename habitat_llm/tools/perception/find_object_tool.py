#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import numpy as np

from habitat_llm.tools import PerceptionTool, get_prompt
from habitat_llm.utils.grammar import FREE_TEXT
from habitat_llm.world_model import Furniture, Human, Receptacle, Room, SpotRobot


class FindObjectTool(PerceptionTool):
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

    def _get_object_list(self, add_state_info=True):
        output = ""

        # Get articulated agent
        articulated_agent = self.env_interface.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # Get position of agent
        agent_pos = np.array(list(articulated_agent.base_pos))

        # Get all object nodes from the world
        objects = self.env_interface.world_graph[self.agent_uid].get_all_objects()

        for obj in objects:
            # For readability
            obj_position = np.array(obj.properties["translation"])

            # Calculate distance from the robot
            distance = round(np.linalg.norm(obj_position - agent_pos), 2)
            distance = "{:.2f}".format(distance)

            # find out if this object has a container attached to it
            container = self.env_interface.world_graph[
                self.agent_uid
            ].get_neighbors_of_type(obj, (Furniture, SpotRobot, Human, Receptacle))

            rooms_path = self.env_interface.world_graph[self.agent_uid].find_path(
                root_node=obj, end_node_types=[Room]
            )
            if rooms_path is None:
                room_name = "an unknown room"
            else:
                rooms = [x for x in rooms_path if isinstance(x, Room)]
                if len(rooms) == 0:
                    room_name = "an unknown room"
                else:
                    if len(rooms) > 1:
                        raise ValueError(
                            f"Multiple rooms detected for object {obj.name}"
                        )
                    room_name = rooms[0].name
            state_string = ""
            if (add_state_info) and ("states" in obj.properties):
                states = obj.properties["states"]
                if len(states) > 0:
                    state_string = " It has the following states: "
                    state_string += ", ".join(
                        [f"{state}: {value}" for state, value in states.items()]
                    )
            # Update string
            if len(container) == 0:
                output += f"{obj.name} is {distance} meters away from the agent in {room_name}.{state_string}\n"
            else:
                if isinstance(container[0], Receptacle):
                    container[0] = self.env_interface.world_graph[
                        self.agent_uid
                    ].find_furniture_for_receptacle(container[0])
                output += f"{obj.name} is in/on {container[0].name} and {distance} meters away from the agent in {room_name}.{state_string}\n"

        if len(objects) == 0:
            output = "No objects found yet."
        return output

    def process_high_level_action(self, input_query, observations):
        super().process_high_level_action(input_query, observations)
        if not self.llm:
            raise ValueError(f"LLM not set in the {self.__class__.__name__}")

        # Extract objects from environment
        objects = self._get_object_list()

        # Create prompt
        prompt = self.prompt_maker(input_query, objects, verbose=False)

        # Execute llm query
        answer = self.llm.generate(prompt, stop="<Done>", max_length=250)

        # Handle the edge case where answer is empty or only spaces
        if answer == "" or answer.isspace():
            answer = (
                f"Could not find any object in world for the query '{input_query}'."
            )

        return None, answer

    @property
    def argument_types(self):
        """
        Returns the types of arguments required for the FindObjectTool.

        :return: List of argument types.
        """
        return [FREE_TEXT]
