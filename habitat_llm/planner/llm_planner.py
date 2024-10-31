#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from habitat.tasks.rearrange.utils import coll_name_matches
from hydra.utils import instantiate

from habitat_llm.llm.instruct.utils import (
    get_objects_descr,
    get_rearranged_objects_descr,
    get_world_descr,
)
from habitat_llm.planner.planner import Planner
from habitat_llm.utils.grammar import (
    FREE_TEXT,
    FURNITURE,
    NAV_TARGET,
    OBJECT,
    OBJECT_OR_FURNITURE,
    ROOM,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.agent import Agent
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.planner.rag import RAG
    from habitat_llm.world_model.world_graph import WorldGraph


class LLMPlanner(Planner):
    """
    High level planner policy used by agents to decide high level actions
    given task description and state of the world
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ):
        """
        Initialize the LLMPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        # Set the planner config
        super().__init__(plan_config, env_interface)
        # Initialize LLM
        self.__initialize_llm()

        # Initialize a variable to indicate if replanning is required
        self.replan_required: bool = True

        # Initialize a variable to count number of llm calls
        self.replanning_count: int = 0

        # Initialize container to store entire prompt and current object states
        self.curr_prompt: str = ""
        self.curr_obj_states: str = ""

        # Initialize container to store rollout without
        # any other material in prompt
        self.trace: str = ""
        self.rag: Optional["RAG"] = None

        self.reset()

        # Build RAG dataset if we want to use RAG
        if self.enable_rag:
            from habitat_llm.planner.rag import RAG

            self.rag = RAG(
                plan_config.example_type,
                plan_config.rag_dataset_dir,
                plan_config.rag_data_source_name,
                plan_config.llm,
            )

    def reset(self):
        """
        Reset the planner state.
        """
        self.last_high_level_actions: Dict[int, Tuple[str, str, str]] = {}
        self.replan_required: bool = True
        self.replanning_count: int = 0
        self.is_done: bool = False

        # save agent observations to get feedback on skill execution
        self.latest_agent_response: Dict[int, str] = {}
        self.curr_prompt: str = ""
        self.trace: str = ""
        self.curr_obj_states: str = ""
        self.params: Dict[str, Any] = {}

        # Reset agents
        for agent in self._agents:
            agent.reset()

    def build_tool_grammar(self, world_graph: "WorldGraph") -> str:
        """
        This method builds a grammar that accepts all valid tool calls based a world graph
        The grammar is specified in the EBNF grammar description format
        see https://github.com/epfl-dlab/transformers-CFG for details and examples

        :param world_graph: The world graph.
        """
        tool_grammar = {}
        objects = world_graph.get_all_objects()
        rules = []
        # we cannot include rules which have objects when there are no objects
        if len(objects) != 0:
            object_expansion = " | ".join(
                (f'"{x.name}"' for x in world_graph.get_all_objects())
            )
            object_rule = f"{OBJECT} ::= " + object_expansion
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM} | {OBJECT})"
            object_of_furniture_rule = (
                f"{OBJECT_OR_FURNITURE} ::= ({FURNITURE} | {OBJECT})"
            )
            rules.append(nav_target_rule)
            rules.append(object_rule)
            rules.append(object_of_furniture_rule)
        else:
            object_of_furniture_rule = f"{OBJECT_OR_FURNITURE} ::= {FURNITURE}"
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM})"
            rules.append(nav_target_rule)
            rules.append(object_of_furniture_rule)
        for agent in self.agents:
            for tool_name, tool in agent.tools.items():
                if tool_name not in tool_grammar:
                    # skip tools that require objects when there are no objects
                    if OBJECT in tool.argument_types and len(objects) == 0:
                        continue
                    tool_grammar[tool_name] = tool.grammar()
        tool_grammar["Done"] = '"Done[]"'
        grammar_str = "tool_call ::= " + " | ".join(tool_grammar.keys()) + "\n"
        for tool_name, tool_grammar_str in tool_grammar.items():
            grammar_str += f"{tool_name} ::= {tool_grammar_str}\n"

        # build rules for each of the argument types
        furniture_rule = f"{FURNITURE} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_furnitures())
        )
        room_rule = f"{ROOM} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_rooms())
        )
        spatial_constraint_rule = f'{SPATIAL_CONSTRAINT} ::= "next_to"'
        spatial_relation_rule = f'{SPATIAL_RELATION} ::= "on" | "within"'
        free_text_rule = f"{FREE_TEXT} ::= [ \"'.:,!a-zA-Z_0-9]*"
        white_space_rule = "WS ::= [ ]*"
        rules.append(furniture_rule)
        rules.append(room_rule)
        rules.append(spatial_constraint_rule)
        rules.append(spatial_relation_rule)
        rules.append(free_text_rule)
        rules.append(white_space_rule)
        grammar_str += "\n".join(rules)
        return grammar_str

    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        """
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)

        action_rules = []
        for i, agent in enumerate(self.agents):
            agent_id = agent.uid
            action_rule = f'action_{i} ::= "Agent_{agent_id}_Action: " tool_call'
            action_rules.append(action_rule)

        combined_action_rule = (
            "action ::= "
            + f' "{delimiter}" '.join(f"action_{i}" for i in range(len(self.agents)))
            + f' "{delimiter}Assigned!"'
        )
        termination_rule = 'termination ::= "Final Thought: Exit!"'
        root_role = f'root ::= {FREE_TEXT} "{delimiter}" (action | termination)'

        return "\n".join(
            [root_role, combined_action_rule]
            + action_rules
            + [termination_rule, tool_rules]
        )

    def __initialize_llm(self):
        """
        This method instantiates LLM as defined in the config
        """
        # Instantiate LLM from the Hydra config
        llm_conf = self.planner_config.llm
        self.llm = instantiate(llm_conf.llm)
        self.llm = self.llm(llm_conf)

        # Setup the LLM parameters
        # self.instruct = self.planner_config.llm.instruct
        self.instruct = self.planner_config.instruct
        self.prompt = self.instruct.prompt
        self.stopword = self.instruct.stopword
        self.end_expression = self.instruct.end_expression
        self.actions_parser = instantiate(self.instruct.actions_parser)

        # save agent observations to get feedback on skill execution
        self.latest_agent_response = {}

    def prepare_prompt(
        self, input_instruction: str, world_graph: "WorldGraph"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the prompt for the LLM.

        :param input_instruction: The input instruction.
        :param world_graph: The world graph.
        :return: The prepared prompt and parameters.
        """
        params = {
            "input": input_instruction,
            "tool_list": self.tool_list,
            "world_graph": world_graph,
            "id": self.agents[0].uid,
        }

        # We modify the prompt if we want to use RAG and the prompt has not
        # been modified
        if "{rag_examples}" in self.prompt:
            if self.rag is not None:
                _, index = self.rag.retrieve_top_k_given_query(
                    input_instruction, top_k=1, agent_id=self._agents[0].uid
                )
                index = index[0]

                example_str = (
                    f"{self.planner_config.llm.user_tag}Below are some example solutions from different settings:\nExample 1:\n"
                    + self.rag.data_dict[index]["trace"]
                    + "\n"
                )
                params["rag_examples"] = example_str
            else:
                params["rag_examples"] = ""
        if "{tool_descriptions}" in self.prompt:
            params["tool_descriptions"] = self.agents[0].tool_descriptions
        if "{agent_descriptions}" in self.prompt:
            params["agent_descriptions"] = self.agent_descriptions
        if "{tool_list}" in self.prompt:
            params["tool_list"] = self.tool_list
        if "{system_tag}" in self.prompt:
            params["system_tag"] = self.planner_config.llm.system_tag
        if "{user_tag}" in self.prompt:
            params["user_tag"] = self.planner_config.llm.user_tag
        if "{assistant_tag}" in self.prompt:
            params["assistant_tag"] = self.planner_config.llm.assistant_tag
        if "{eot_tag}" in self.prompt:
            params["eot_tag"] = self.planner_config.llm.eot_tag
        if "{agent_role_description}" in self.prompt:
            # only support agent role description when planning for a single agent
            assert len(self.agents) == 1
            if str(self.agents[0].uid) == "1":
                agent_role_description = '\nYou are playing the role of the task giver. This means if the instruction says something like "You should move the object and I will wash it", then the other agent should be moving the object, and you should washing the it.\n'
            else:
                agent_role_description = '\nYou are playing the role of the task receiver. This means if the instruction says something like "You should move the object and I will wash it", then you should move the object and the other agent should wash it.\n'
            params["agent_role_description"] = agent_role_description
        if "{world_description}" in self.prompt:
            # only designed for the decentralized setting
            assert len(self.agents) == 1
            world_description = get_world_descr(
                world_graph,
                agent_uid=self.agents[0].uid,
                add_state_info=self.planner_config.objects_response_include_states,
                include_room_name=True,
                centralized=self.planner_config.centralized,
            )
            params["world_description"] = world_description

        return self.prompt.format(**params), params

    @property
    def tool_list(self) -> List[str]:
        """
        Returns a string listing the agents tools
        :return: A sorted list of tool names.
        """
        tool_set = set()
        for agent in self.agents:
            for tool_name in agent.tools:
                tool_set.add(tool_name)

        return sorted(tool_set)

    @property
    def agents(self) -> List["Agent"]:
        """
        Get the list of agents associated with this planner.

        :return: A list of Agent objects.
        """
        return self._agents

    @agents.setter
    def agents(self, agents: List["Agent"]) -> None:
        """
        Set the list of agents for this planner.

        :param agents: A list of Agent objects to be associated with this planner.
        """
        self._agents = agents
        # Pass on respective LLM instance into agent tools
        for agent in self._agents:
            agent.pass_llm_to_tools(self.llm)

    def get_last_agent_states(self) -> Dict[int, str]:
        """
        Get the last state descriptions for all agents.

        :return: A dictionary mapping agent UIDs to their last state descriptions.
        """
        # Container to store state descriptions
        agent_states = {}

        # Loop through the agents and populate state descriptions
        for agent in self._agents:
            agent_states[agent.uid] = agent.get_last_state_description()

        return agent_states

    # TODO: @zephirefaith implement agent's room affiliations in the world graph
    # and edit this function to read from it
    def get_last_agent_positions(self) -> Dict[str, Any]:
        """
        Get the last positions for all agents.

        :return: A dictionary mapping agent names to their positions.
        """
        # Container to store agent positions
        agent_positions = {}

        # get agent nodes
        agents = self.env_interface.full_world_graph.get_agents()

        # Loop through the agents and populate nodes
        for agent in agents:
            agent_positions[agent.name] = agent.get_property("translation")

        return agent_positions

    def get_agent_collisions(self) -> Dict[int, bool]:
        """
        Check if the agents are colliding.

        :return: A dictionary mapping agent UIDs to collision status.
        """
        # set collision to false
        collision = False

        # Get list of agent ids
        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in self.env_interface.sim.agents_mgr.articulated_agents_iter
        ]

        # Return false if only one agent is in the scene
        if len(agent_ids) == 2:
            # Perform collision check
            self.env_interface.sim.perform_discrete_collision_detection()
            contact_points = self.env_interface.sim.get_physics_contact_points()

            for cp in contact_points:
                if coll_name_matches(cp, agent_ids[0]) and coll_name_matches(
                    cp, agent_ids[1]
                ):
                    collision = True

        # Declare output container
        out = {}

        # update the output
        for agent in self._agents:
            out[agent.uid] = collision

        return out

    def format_response(
        self, response: str, end_expression: Union[str, List[str]]
    ) -> str:
        """
        Format the LLM response by trimming it up to the first appearance of end_expression.

        :param response: The LLM response to format.
        :param end_expression: The end expression(s) to look for.
        :return: The formatted response.
        """
        response = response.rstrip("\n")
        if type(end_expression) == str:
            index = response.find(end_expression)
            target_end_expression = end_expression
        else:
            # end_expression is a list of string
            index = -1
            target_end_expression = ""
            for _end_expression in end_expression:
                _index = response.find(_end_expression)
                if _index < index or index == -1:
                    index = _index
                    target_end_expression = _end_expression
        return (
            response[: index + len(target_end_expression)] if index != -1 else response
        )

    def parse_thought(self, input_string: str) -> str:
        """
        Extract thought from the LLM response.

        :param input_string: The input string to parse.
        :return: The extracted thought.
        """
        # Define the patterns for Agent actions
        pattern = r"\n|Final Thought"

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        if match:
            # Extract the text before the pattern
            return input_string[: match.start()].strip()
        else:
            # If no pattern is found, return the whole string
            return ""

    def _add_responses_to_prompt(self, responses: Dict[int, str]) -> str:
        """
        Add agent responses to the prompt.

        :param responses: A dictionary of agent responses.
        :return: The updated print string.
        """
        print_str = ""
        prompt_addition = ""
        add_object_update = False
        for agent_uid in sorted(responses.keys()):
            # If the response for a given agent is valid, add to the prompt and printout
            if responses[agent_uid]:
                # Update print string
                print_str += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # Update the prompt
                prompt_addition += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # self.curr_prompt += prompt_addition
                self.trace += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )

            # If the response is empty then indicate the action is still in progress
            # only when replanning was required
            elif self.replan_required:
                responses[
                    agent_uid
                ] = f"Action {self.last_high_level_actions[agent_uid][0]}[{self.last_high_level_actions[agent_uid][1]}] is still in progress."

                # Update print string
                print_str += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )

                # Update the prompt
                prompt_addition += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # self.curr_prompt += prompt_addition
                self.trace += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                add_object_update = True

            # save agent observations to get feedback on skill execution
            self.latest_agent_response[agent_uid] = responses[agent_uid]

        if prompt_addition != "":
            self.curr_prompt += self.planner_config.llm.user_tag + prompt_addition
            if (
                self.planner_config.objects_response
                and add_object_update
                and self.planner_config.centralized
            ):
                world_graph = self.env_interface.world_graph[agent_uid]
                objects = get_objects_descr(
                    world_graph,
                    agent_uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                    centralized=self.planner_config.centralized,
                )
                if self.planner_config.prompt_w_updatedobjects_only:
                    # add details on what changed in the world.
                    # TODO: this currently assumes symmetric world graph,
                    # extend for decentralized/asymmetric WG
                    updated_objects = get_rearranged_objects_descr(
                        obj_descr_t=objects, obj_descr_t_1=self.curr_obj_states
                    )
                    self.curr_obj_states = objects
                    if updated_objects != "":
                        result = f"Newly found objects/updates on known objects: {updated_objects}\n"
                    else:
                        result = (
                            "No new objects or updates on known objects were found.\n"
                        )
                else:
                    result = f"Objects: {objects}\n"
                self.curr_obj_states = objects
                self.curr_prompt += result
                self.trace += result
                print_str += result
            self.curr_prompt += self.planner_config.llm.eot_tag
            print(self.curr_prompt)

        # Force add thought after every observation
        if self.planner_config.planning_mode.lower() == "cot":
            for agent_uid in sorted(responses.keys()):
                if responses[agent_uid]:
                    print_str += "Thought:"
                    prompt_addition = f"{self.planner_config.llm.assistant_tag}Thought:"
                    self.curr_prompt += prompt_addition
                    self.trace += "Thought:"
                    break
        return print_str

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        verbose: bool = False,
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Get the next low-level action to execute.

        :param instruction: The instruction for the task.
        :param observations: The current observations.
        :param world_graph: The world graph for each agent.
        :param verbose: Whether to print verbose output. Defaults to False.
        :return: A tuple containing:
                 - The low-level actions for each agent
                 - Planner information
                 - Whether the planner is done
        """
        planner_info: Dict[str, Union[Any, str]] = {}
        # Early return if planner is already done
        if self.is_done:
            planner_info = {
                "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                "traces": {agent.uid: self.trace for agent in self.agents},
                "replanning_count": {
                    agent.uid: self.replanning_count for agent in self.agents
                },
                "replanned": {agent.uid: False for agent in self.agents},
                "replan_required": {
                    agent.uid: self.replan_required for agent in self.agents
                },
                "is_done": {agent.uid: self.is_done for agent in self.agents},
            }
            return {}, planner_info, self.is_done

        if self.curr_prompt == "":
            # Prepare prompts
            self.curr_prompt, self.params = self.prepare_prompt(
                instruction, world_graph[self._agents[0].uid]
            )
            self.curr_obj_states = get_objects_descr(
                world_graph[self._agents[0].uid],
                self._agents[0].uid,
                include_room_name=True,
                add_state_info=self.planner_config.objects_response_include_states,
                centralized=self.planner_config.centralized,
            )

        if self.trace == "":
            self.trace += f"Task: {instruction}\nThought: "

        print_str = ""
        self.is_done = False

        if self.replan_required:
            planner_info["replanned"] = {agent.uid: True for agent in self.agents}
            if verbose:
                # calculate the total time of response generation
                start_time = time.time()
            # Generate response
            if self.planner_config.get("constrained_generation", False):
                llm_response = self.llm.generate(
                    self.curr_prompt,
                    self.stopword,
                    generation_args={
                        "grammar_definition": self.build_response_grammar(
                            world_graph[self._agents[0].uid]
                        )
                    },
                )
            else:
                llm_response = self.llm.generate(self.curr_prompt, self.stopword)

            # Format the response
            # This removes extra text followed by end expression when needed.
            llm_response = self.format_response(llm_response, self.end_expression)

            # parse thought from the response
            thought = self.parse_thought(llm_response)

            if verbose:
                total_time = time.time() - start_time
                print(
                    f"Time taken for LLM response generation: {total_time}; replanning_count: {self.replanning_count}"
                )

            # Update prompt with the first response
            print_str += f"""{llm_response}\n{self.stopword}\n"""
            prompt_addition = (
                f"""{llm_response}\n{self.stopword}{self.planner_config.llm.eot_tag}"""
            )
            self.curr_prompt += prompt_addition
            self.trace += prompt_addition

            # Check if the planner should stop
            # Stop if the replanning count exceed a certain threshold
            # or end expression is found in llm response
            # This is helpful to break infinite planning loop.
            self.is_done = (self.check_if_agent_done(llm_response)) or (
                self.replanning_count == self.planner_config.replanning_threshold
            )
            # Increment the llm call counter on every replan
            # doesn't get incremented before comparison as first "replan" is technically
            # the first required plan
            self.replanning_count += 1

            # Early return if stop is required
            if self.is_done:
                planner_info = {
                    "print": print_str,
                    # "print_no_tags": print_str_no_tags,
                    "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                    "traces": {agent.uid: self.trace for agent in self.agents},
                    "replanning_count": {
                        agent.uid: self.replanning_count for agent in self.agents
                    },
                    "replan_required": {
                        agent.uid: self.replan_required for agent in self.agents
                    },
                    "replanned": {agent.uid: True for agent in self.agents},
                    "is_done": {agent.uid: self.is_done for agent in self.agents},
                    "thought": {agent.uid: thought for agent in self.agents},
                    "high_level_actions": {
                        agent.uid: ("Done", None, None) for agent in self.agents
                    },
                }
                return {}, planner_info, self.is_done

            # Parse high level action directives from llm response
            high_level_actions = self.actions_parser(
                self.agents, llm_response, self.params
            )

            print(f"\n\n[DEBUG] Now Executing: {high_level_actions}\n\n")

            # Get low level actions and/or responses
            low_level_actions, responses = self.process_high_level_actions(
                high_level_actions, observations
            )

            # Store last executed high level action
            self.last_high_level_actions = high_level_actions
        else:
            planner_info["replanned"] = {agent.uid: False for agent in self.agents}
            # Set thought to None
            thought = None

            # Get low level actions and/or responses using last high level actions
            low_level_actions, responses = self.process_high_level_actions(
                self.last_high_level_actions, observations
            )

        # Log if replanning was done or not before overwriting the value
        planner_info["replan_required"] = {
            agent.uid: self.replan_required for agent in self.agents
        }

        # Check if replanning is required
        # Replanning is required when any of the actions being executed
        # have a response indicating success or failure (and the reason)
        self.replan_required = any(responses.values())
        print_str += self._add_responses_to_prompt(responses)

        # Update planner info
        planner_info["responses"] = responses
        planner_info["thought"] = {agent.uid: thought for agent in self.agents}
        planner_info["is_done"] = {agent.uid: self.is_done for agent in self.agents}
        planner_info["print"] = print_str
        # planner_info["print_no_tags"] = print_str_no_tags
        planner_info["high_level_actions"] = self.last_high_level_actions
        planner_info["prompts"] = {agent.uid: self.curr_prompt for agent in self.agents}
        planner_info["traces"] = {agent.uid: self.trace for agent in self.agents}
        planner_info["replanning_count"] = {
            agent.uid: self.replanning_count for agent in self.agents
        }
        planner_info["agent_states"] = self.get_last_agent_states()
        planner_info["agent_positions"] = self.get_last_agent_positions()
        planner_info["agent_collisions"] = self.get_agent_collisions()
        return low_level_actions, planner_info, self.is_done

    def check_if_agent_done(self, llm_response: str) -> bool:
        """
        Check if the agent is done based on the LLM response.

        :param llm_response: The LLM response to check.
        :return: True if the agent is done, False otherwise.
        """
        return self.end_expression in llm_response
