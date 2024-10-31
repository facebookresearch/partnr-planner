#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import TYPE_CHECKING, Any, Dict, Tuple

from habitat_llm.llm.instruct.utils import get_world_descr
from habitat_llm.planner.llm_planner import LLMPlanner

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph


class CentralizedLLMPlanner(LLMPlanner):
    def __init__(
        self, plan_config: Dict[str, Any], env_interface: "EnvironmentInterface"
    ):
        """
        Initialize the CentralizedLLMPlanner.

        :param plan_config: Configuration for the planner.
        :param env_interface: Interface to the environment.
        :return: None
        """
        super().__init__(plan_config, env_interface)

    def prepare_prompt(
        self, instruction: str, world_graph: "WorldGraph"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the prompt for the LLM by adding the input and agent descriptions.

        :param instruction: The instruction to be evaluated.
        :param world_graph: The world graph object containing environment information.
        :return: A tuple containing the formatted prompt and the parameters used.
        :raises ValueError: If world_graph is not a WorldGraph object.
        """
        if isinstance(world_graph, Dict):
            raise ValueError(
                f"Expected world_graph to be a WorldGraph object, not a Dict. Received: {world_graph}"
            )

        params = {
            "input": instruction,
            "tool_list": self.tool_list,
            "world_graph": world_graph,
        }

        ## house description
        furn_room = world_graph.group_furniture_by_room()
        house_info = ""
        for k, v in furn_room.items():
            furn_names = [furn.name for furn in v]
            all_furn = ", ".join(furn_names)
            house_info += k + ": " + all_furn + "\n"

        ## objects in the house
        objs_info = ""
        all_objs = world_graph.get_all_objects()
        for obj in all_objs:
            fur_object = world_graph.find_furniture_for_object(obj).name
            objs_info += obj.name + ": " + fur_object + "\n"

        if "{tool_descriptions}" in self.prompt:
            params["tool_descriptions"] = self.agents[0].tool_descriptions
        if "{agent_descriptions}" in self.prompt:
            params["agent_descriptions"] = self.agent_descriptions
        if "{tool_list}" in self.prompt:
            params["tool_list"] = self.tool_list
        if "{house_description}" in self.prompt:
            params["house_description"] = house_info
        if "{all_objects}" in self.prompt:
            params["all_objects"] = objs_info
        if "{world_description}" in self.prompt:
            world_description = get_world_descr(
                world_graph,
                agent_uid=self.agents[0].uid,
                add_state_info=True,
                include_room_name=True,
                centralized=True,
            )
            params["world_description"] = world_description

        if "{system_tag}" in self.prompt:
            params["system_tag"] = self.planner_config.llm.system_tag
        if "{user_tag}" in self.prompt:
            params["user_tag"] = self.planner_config.llm.user_tag
        if "{assistant_tag}" in self.prompt:
            params["assistant_tag"] = self.planner_config.llm.assistant_tag
        if "{eot_tag}" in self.prompt:
            params["eot_tag"] = self.planner_config.llm.eot_tag

        return self.prompt.format(**params), params
