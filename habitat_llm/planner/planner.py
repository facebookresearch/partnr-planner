#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from omegaconf import DictConfig

if TYPE_CHECKING:
    from habitat_llm.agent import Agent
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph


# This class represents an abstract class for any planner.
class Planner:
    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ) -> None:
        """
        Initialize the Planner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        # Set the planner config
        self.planner_config: "DictConfig" = plan_config
        self.env_interface: "EnvironmentInterface" = env_interface
        self._agents: List["Agent"] = []
        self.is_done: bool = False
        self.enable_rag: bool = (
            plan_config.get("enable_rag", False)
            if isinstance(plan_config, DictConfig)
            else False
        )
        self.swap_instruction: bool = True
        self.last_high_level_actions: Dict[int, Tuple[str, str, str]] = {}

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Gives the next low level action to execute.

        :param instruction: The instruction for the task.
        :param observations: The current observations.
        :param world_graph: The world graph for each agent.
        :return: A tuple containing:
                 - The low-level actions for each agent
                 - Planner information
                 - Whether the planner is done
        """
        raise NotImplementedError

    @property
    def agent_indices(self) -> List[int]:
        """
        Get the indices (uids) of the agents that this planner controls.

        :return: A list of agent indices.
        """
        return [agent.uid for agent in self._agents]

    def reset(self) -> None:
        """
        Reset the planner state.
        """
        raise NotImplementedError

    @property
    def agents(self) -> List["Agent"]:
        """
        Get the list of agents controlled by this planner.

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

    @property
    def agent_descriptions(self) -> str:
        """
        Get a string listing the descriptions of all agents.

        :return: A string containing agent descriptions.
        """
        out = ""
        for agent in self.agents:
            out += agent.agent_description
        return out

    def get_agent_from_uid(self, agent_uid: int) -> "Agent":
        """
        Get an agent object given its UID.

        :param agent_uid: The unique identifier of the agent.
        :return: The Agent object corresponding to the given UID.
        :raises ValueError: If no agent with the given UID is found.
        """
        for agent in self.agents:
            if agent.uid == agent_uid:
                return agent
        raise ValueError(f'Agent with uid "{agent_uid}" not found')

    def filter_obs_space(self, batch: Dict[str, Any], agent_uid: int) -> Dict[str, Any]:
        """
        Filter observations to return only those belonging to the specified agent.

        :param batch: A dictionary of observations for all agents.
        :param agent_uid: The unique identifier of the agent to filter observations for.
        :return: A dictionary of filtered observations for the specified agent.
        """
        if self.env_interface._single_agent_mode:
            return batch
        agent_name = f"agent_{agent_uid}"
        agent_name_bar = f"{agent_name}_"
        output_batch = {
            obs_name.replace(agent_name_bar, ""): obs_value
            for obs_name, obs_value in batch.items()
            if agent_name in obs_name
        }
        return output_batch

    def process_high_level_actions(
        self, hl_actions: Dict[int, Tuple[str, str, str]], observations: Dict[str, Any]
    ) -> Tuple[Dict[int, Any], Dict[int, str]]:
        """
        Process high-level actions and generate low-level actions and responses.

        :param hl_actions: A dictionary of high-level actions for each agent.
        :param observations: The current observations.
        :return: A tuple containing:
                 - A dictionary of low-level actions for each agent
                 - A dictionary of responses for each agent
        """
        # Make sure that the high level actions are not empty
        agent_indices = self.agent_indices
        if not hl_actions:
            response = "No actions were assigned. Please assign action to this agent."
            responses = {agent_ind: response for agent_ind in agent_indices}
            return {}, responses

        # Declare containers for responses and low level actions
        low_level_actions = {}
        responses = {}

        # Iterate through all agents
        for agent in self.agents:
            agent_uid = agent.uid

            if agent_uid in hl_actions:
                # For readability
                hl_action_name = hl_actions[agent_uid][0]
                hl_action_input = hl_actions[agent_uid][1]
                hl_error_message = hl_actions[agent_uid][2]

                # Handle error message
                if hl_error_message:
                    responses[agent_uid] = hl_error_message
                    continue

                # Fetch agent specific observations
                filtered_observations = self.filter_obs_space(observations, agent_uid)
                # Get response and/or low level actions
                low_level_action, response = agent.process_high_level_action(
                    hl_action_name, hl_action_input, filtered_observations
                )

                # Insert to the output
                if low_level_action is not None:
                    low_level_actions[agent_uid] = low_level_action
                responses[agent_uid] = response.rstrip("\n")

        # update world based on actions
        self.update_world(responses)

        return low_level_actions, responses

    def update_world(self, responses: Dict[int, str]) -> None:
        """
        Update the world graph with the latest observations and actions. Notes this is
        only required for partial-observability case, this function does NOTHING under
        full observability.

        Full observability condition does not need an update due to actions.
        Action-based-updates were necessary because in partial-obs the object is not
        visible while being carried so is dropped from "agent-is-holding" relation.

        :param responses: A dictionary of responses for each agent.
        """
        if self.env_interface.partial_obs:
            self._partial_obs_update(responses)

    def _partial_obs_update(self, responses: Dict[int, str]) -> None:
        """
        Update each agent's graph with respect to other agent's actions for both CG and
        GT conditions under partial-observable setting.

        :param responses: A dictionary of responses for each agent.
        """
        composite_action_response = self.env_interface._composite_action_response
        for agent_uid in self.last_high_level_actions:
            action_and_args = None
            action_results = None
            int_agent_uid = int(agent_uid)
            if agent_uid in responses or int_agent_uid in composite_action_response:
                if int_agent_uid in composite_action_response:
                    action_and_args = composite_action_response[int_agent_uid]
                    action_results = action_and_args[2]
                    # reset to empty out this variable
                    self.env_interface.reset_composite_action_response()
                elif agent_uid in responses:
                    action_and_args = self.last_high_level_actions[agent_uid]
                    action_results = responses[agent_uid]
                int_other_agent_uid = 1 - int_agent_uid
                # update own and other's world-graph
                # -----------------------
                if (
                    self.env_interface.conf.world_model.type == "concept_graph"
                    and agent_uid == self.env_interface.robot_agent_uid
                ):
                    self.env_interface.world_graph[
                        int_agent_uid
                    ].update_non_privileged_graph_by_action(
                        agent_uid,
                        action_and_args,
                        action_results,
                    )
                else:
                    self.env_interface.world_graph[int_agent_uid].update_by_action(
                        agent_uid,
                        action_and_args,
                        action_results,
                    )

                # update other agent's graph with current agent's actions
                # NOTE: this is a separate function since two agents may refer to the
                # same entity using different descriptions. This function call handles
                # that ambiguity
                if (
                    self.env_interface.conf.agent_asymmetry
                    and int_agent_uid == self.env_interface.human_agent_uid
                ) or (not self.env_interface.conf.agent_asymmetry):
                    # only update robot's WG with other agent's actions
                    # OR
                    # add action based updates irrespective of agent types
                    if (
                        self.env_interface.conf.world_model.type == "concept_graph"
                        and int_other_agent_uid == self.env_interface.robot_agent_uid
                    ):
                        self.env_interface.world_graph[
                            int_other_agent_uid
                        ].update_non_privileged_graph_by_other_agent_action(
                            agent_uid,
                            action_and_args,
                            action_results,
                        )
                    else:
                        self.env_interface.world_graph[
                            int_other_agent_uid
                        ].update_by_other_agent_action(
                            agent_uid,
                            action_and_args,
                            action_results,
                        )
                # -----------------------
