# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import FURNITURE
from habitat_llm.utils.sim import (
    ee_distance_to_object,
    get_ao_and_joint_idx,
    get_parent_ao_and_joint_idx,
    get_receptacle_index,
)
from habitat_llm.world_model import Furniture, Receptacle


class OracleOpenCloseSkill(SkillPolicy):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(config, action_space, batch_size, True, agent_uid=agent_uid)
        self.env = env
        self.wait_time_for_rep_to_open = 0
        self.was_successful = False
        self.max_dis_interact_art = config.max_dis_interact_art

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # NOTE: the following are initialized in the subclasses
        # Fetch the action range
        self.action_range: Tuple[int, int] = None
        # Fetch the index of open flag (first element in the action space)
        self.open_close_flag_index: int = None
        # Fetch the index of object id (second element in the action space)
        self.object_index: int = None
        # Fetch the index of is_surface_flag
        self.surface_flag_index: int = None
        # Fetch the index of surface_index
        self.surface_index: int = None

        # NOTE: this allows open or close skill to pivot. Must be one of ["Open", "Close"]
        self.skill_subtype = None

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        was_successful = self.was_successful
        return (torch.ones(1) * was_successful).to(masks.device)

    def reset(
        self,
        batch_idxs,
    ):
        super().reset(
            batch_idxs,
        )
        self.was_successful = False
        self.action_assigned = False

        return

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        # Following the try/except pattern used in other skills
        try:
            target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
                self.target_handle
            )
            return f"{self.skill_subtype} {target_node.name}"
        except Exception as e:
            print(
                f"WARNING: cannot get {self.target_handle} in graph due to {e}. Agent's state is standing"
            )
            return "Standing"

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # must have a subtype set
        if self.skill_subtype not in ["Open", "Close"]:
            raise ValueError(
                f"Provided skill subtype '{self.skill_subtype}' is not supported. Should be 'Open' or 'Close'."
            )

        # Prepare action tensor and target name
        action = torch.zeros_like(prev_actions)

        # Since we dont have a sensor observations indicating whether
        # the open action was successful or not, we assume that this
        # oracle skill succeeds when its step through. The logic below
        # marks this skill as successful after first sim step and during
        # second execution. TODO: Need to rethink this logic later.

        if self._is_action_issued[-1]:
            self.was_successful = True
            self._is_action_issued[cur_batch_idx] = False
            return action, None

        # Get target_entity
        target_entity = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
            self.target_handle
        )

        # Check if the target is a receptacle or surface
        target_is_furniture = False
        target_is_receptacle = False
        if isinstance(target_entity, Furniture):
            target_is_furniture = True
        elif isinstance(target_entity, Receptacle):
            target_is_receptacle = True
        else:
            self.failed = True
            self.termination_message = f"Failed to {self.skill_subtype}! You can't {self.skill_subtype} a {target_entity.name}."
            return action, None

        # Make sure that the target is articulated object
        if not target_entity.is_articulated():
            self.failed = True
            self.termination_message = f"Failed to {self.skill_subtype}! {target_entity.name} is not articulated - and cannot be {self.skill_subtype}ed."
            return action, None

        # Get the furniture (or parent furniture if target is a receptacle)
        # and corresponding joint index
        articulated_object = None
        if target_is_furniture:
            articulated_object, _ = get_ao_and_joint_idx(target_entity, self.env)
        elif target_is_receptacle:
            articulated_object, _ = get_parent_ao_and_joint_idx(target_entity, self.env)

        # verify horizontal L2 distance to receptacle
        ee_dist_to_furniture = ee_distance_to_object(
            self.env.sim,
            self.env.sim.agents_mgr,
            self.agent_uid,
            articulated_object.handle,
            max_distance=self.max_dis_interact_art,
        )
        if (
            ee_dist_to_furniture is None
            or ee_dist_to_furniture > self.max_dis_interact_art
        ):
            self.failed = True
            self.termination_message = f"Failed to {self.skill_subtype}! {target_entity.name} is occluded or too far from agent to {self.skill_subtype}."
            return action, None

        # Get the object index
        aom = self.env.sim.get_articulated_object_manager()
        obj_idx = aom.get_object_id_by_handle(articulated_object.handle)

        # Set surface index
        surface_idx = 0
        if target_is_receptacle:
            surface_idx = get_receptacle_index(self.target_handle, self.env.receptacles)

        # Populate the action tensor
        action[cur_batch_idx, self.open_close_flag_index] = 1
        action[cur_batch_idx, self.object_index] = obj_idx
        action[cur_batch_idx, self.surface_flag_index] = target_is_receptacle
        action[cur_batch_idx, self.surface_index] = surface_idx

        # Set action assigned to true to indicate that the action was assigned once
        # Because this is oracle skill, the action will be assumed to be successful next time its run
        # self.action_assigned = True
        self._is_action_issued[cur_batch_idx] = True

        return action, None


class OracleOpenSkill(OracleOpenCloseSkill):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config, observation_space, action_space, batch_size, env, agent_uid
        )

        # Fetch the action range
        self.action_range = find_action_range(
            self.action_space, f"agent_{self.agent_uid}_oracle_open_action"
        )
        # Fetch the index of open flag (first element in the action space)
        self.open_close_flag_index = self.action_range[0]
        # Fetch the index of object id (second element in the action space)
        self.object_index = self.action_range[0] + 1
        # Fetch the index of is_surface_flag
        self.surface_flag_index = self.action_range[0] + 2
        # Fetch the index of surface_index
        self.surface_index = self.action_range[0] + 3

        # NOTE: this allows open or close skill to pivot. Must be one of ["Open", "Close"]
        self.skill_subtype = "Open"

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleOpenSkill.

        :return: List of argument types.
        """
        return [FURNITURE]


class OracleCloseSkill(OracleOpenCloseSkill):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config, observation_space, action_space, batch_size, env, agent_uid
        )

        # Fetch the action range
        self.action_range = find_action_range(
            self.action_space, f"agent_{self.agent_uid}_oracle_close_action"
        )
        # Fetch the index of open flag (first element in the action space)
        self.open_close_flag_index = self.action_range[0]
        # Fetch the index of object id (second element in the action space)
        self.object_index = self.action_range[0] + 1
        # Fetch the index of is_surface_flag
        self.surface_flag_index = self.action_range[0] + 2
        # Fetch the index of surface_index
        self.surface_index = self.action_range[0] + 3

        # NOTE: this allows open or close skill to pivot. Must be one of ["Open", "Close"]
        self.skill_subtype = "Close"

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleCloseSkill.

        :return: List of argument types.
        """
        return [FURNITURE]
