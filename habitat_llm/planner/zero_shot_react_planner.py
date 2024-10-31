# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import TYPE_CHECKING, Dict

from habitat_llm.llm.instruct.utils import get_objects_descr
from habitat_llm.planner import LLMPlanner
from habitat_llm.utils.grammar import FREE_TEXT

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph


class ZeroShotReactPlanner(LLMPlanner):
    """
    This class builds the prompt for the, zero shot llm react planner format.
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ) -> None:
        """
        Initialize the ZeroShotReactPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        super().__init__(plan_config, env_interface)

    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        :return: The response grammar.
        """
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)

        root_rule = (
            f'root ::= {FREE_TEXT} "{delimiter}" tool_call "{delimiter}Assigned!"'
        )

        return "\n".join([root_rule, tool_rules])

    def _add_responses_to_prompt(self, responses: Dict[int, str]) -> str:
        """
        Add skill responses to the prompt optionally including object descriptions (depending on the config).

        :param responses: A dictionary of agent responses.
        :return: The updated print string.
        """
        if self.planner_config.objects_response:
            assert len(self.agents) == 1
            agent = self.agents[0]
            result = ""
            world_graph = self.env_interface.world_graph[agent.uid]
            if responses[agent.uid] != "":
                response_format = (
                    "{user_tag}Result: {result}\nObjects: {objects}{eot_tag}"
                )
                objects = get_objects_descr(
                    world_graph,
                    agent.uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                )
                result = response_format.format(
                    result=responses[agent.uid],
                    objects=objects,
                    user_tag=self.planner_config.llm.user_tag,
                    eot_tag=self.planner_config.llm.eot_tag,
                )
                self.curr_prompt += result + self.planner_config.llm.assistant_tag
                print(result + self.planner_config.llm.assistant_tag, end="")
                self.trace += result + self.planner_config.llm.assistant_tag
        else:
            result = super()._add_responses_to_prompt(responses)
        return result
