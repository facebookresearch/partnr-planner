# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills import PickSkillPolicy
from habitat_llm.utils.sim import (
    find_receptacles,
    get_ao_and_joint_idx,
    get_parent_ao_and_joint_idx,
    get_receptacle_index,
)
from habitat_llm.world_model import Furniture, Receptacle


class OpenSkillPolicy(PickSkillPolicy):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        batch_size,
        env=None,
        agent_uid=0,
    ):
        super().__init__(
            config,
            observation_space,
            action_space,
            batch_size,
            env=env,
            agent_uid=agent_uid,
        )

        # The following are the parameters for oracle open skill
        self.wait_time_for_rep_to_open = 0
        self.was_successful = False
        self.max_dis_interact_art = config.max_dis_interact_art

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # Fetch the action range
        self.action_range = find_action_range(
            self.action_space, f"agent_{self.agent_uid}_oracle_open_action"
        )
        # Fetch the index of open flag (first element in the action space)
        self.open_flag_index = self.action_range[0]
        # Fetch the index of object id (second element in the action space)
        self.object_index = self.action_range[0] + 1
        # Fetch the index of is_surface_flag
        self.surface_flag_index = self.action_range[0] + 2
        # Fetch the index of surface_index
        self.surface_index = self.action_range[0] + 3
        # Get manager
        self._rom = self.env.sim.get_rigid_object_manager()
        self._aom = self.env.sim.get_articulated_object_manager()
        # Set the threshold
        self._ee_to_art_obj_threshold = config.ee_to_art_obj_threshold
        # Set the mode
        self.mode = "open"
        # Get the grasping mgr
        self.grasp_mgr = self.env.sim.agents_mgr[self.agent_uid].grasp_mgr

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        current_joint_pos = observations["joint"].cpu().numpy()
        rel_resting_pos = torch.norm(
            torch.tensor(self._joint_rest_state) - current_joint_pos, keepdim=True
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold
        was_successful = self.was_successful
        return (was_successful * is_within_thresh).to(masks.device)

    def reset(
        self,
        batch_idxs,
    ):
        super().reset(
            batch_idxs,
        )
        self.was_successful = False
        return

    def set_target(self, target_name, env):
        super().set_target(target_name, env)
        # The following is from oracle open skill, only activates when the arm is near
        furniture, _, _ = self._get_art_obj_info()
        self.env.sim.dynamic_target = furniture.translation

    def extract_receptacle(self, receptacle_name, is_surface=False):
        if receptacle_name in self._rom.get_object_handles():
            return self._rom.get_object_by_handle(receptacle_name)
        elif receptacle_name in self._aom.get_object_handles():
            return self._aom.get_object_by_handle(receptacle_name)
        else:
            raise Exception(f"Receptacle {receptacle_name} not found.")

    def _get_receptacle_and_joint_idx(self, receptacle_name):
        """
        This method fetches the receptacle and joint indices provided the receptacle name. "receptacle_name" can be parent object (Fridge) or any child surface (fridge_shelf_2). This method checks if the object referenced by "receptacle_name" is the parent object or a child surface and returns the joint indices accordingly.
        """

        for r in find_receptacles(self.env.sim):
            if receptacle_name == r.parent_object_handle:
                # If "receptacle_name" is parent
                rec = self.extract_receptacle(receptacle_name)
                joint_idx = list(range(len(rec.joint_positions)))
                return rec, joint_idx
            elif receptacle_name == r.name:
                # If "receptacle_name" is surface
                rec = self.extract_receptacle(r.parent_object_handle)
                joint_idx = [rec.get_link_joint_pos_offset(r.parent_link)]
                return rec, joint_idx
        return None, None

    def _does_arm_reach_handle(self, observations, cur_batch_idx):
        """Check if the arm reaches the art obj"""
        return (
            np.linalg.norm(np.array(observations["obj_start_sensor"]))
            < self._ee_to_art_obj_threshold
        )

    def _get_art_obj_info(self):
        """Get the articulated object info"""
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
            if self.mode == "open":
                self.termination_message = (
                    f"Failed to Open! You can't open a {target_entity.name}."
                )
            else:
                self.termination_message = f"Failed to Close! {target_entity.name} is not articulated - and cannot be closed."
            return None, None

        # Make sure that the target is articulated object
        if not target_entity.is_articulated():
            self.failed = True
            if self.mode == "open":
                self.termination_message = f"Failed to Open! {target_entity.name} is not articulated - and cannot be opened."
            else:
                self.termination_message = f"Failed to Close! {target_entity.name} is not articulated - and cannot be closed."
            return None, None

        # Get the furniture (or parent furniture if target is a receptacle)
        # and corresponding joint index
        if target_is_furniture:
            fur, joint_idx = get_ao_and_joint_idx(target_entity, self.env)
        elif target_is_receptacle:
            fur, joint_idx = get_parent_ao_and_joint_idx(target_entity, self.env)

        return fur, joint_idx, target_is_receptacle

    def _get_oracle_skill_param(
        self, fur, joint_idx, target_is_receptacle, enable_distance_check=True
    ):
        """Get the oracle action"""
        if fur is None or joint_idx is None:
            return None, None

        # Get target_entity
        target_entity = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
            self.target_handle
        )

        # Verify if the agent is holding something
        if self.grasp_mgr.is_grasped:
            self.failed = True
            if self.mode == "open":
                self.termination_message = f"Failed to Open! You cannot open {target_entity.name} since you are holding something. Please place the object on the floor first and then open. Pick up the object again afterward."
            else:
                self.termination_message = f"Failed to close! You cannot close {target_entity.name} since you are holding something. Please place the object on the floor first and then close. Pick up the object again afterward."

        # Verify distance to receptacle
        robot_pos = np.array(self.articulated_agent.base_pos)
        dist = np.linalg.norm((robot_pos - np.array(fur.translation))[[0, 2]])
        if dist > self.max_dis_interact_art and enable_distance_check:
            self.failed = True
            if self.mode == "open":
                self.termination_message = (
                    f"Failed to Open! {target_entity.name} is too far to open."
                )
            else:
                self.termination_message = (
                    f"Failed to close! {target_entity.name} is too far to close."
                )
            return None, None

        # Get the object index
        aom = self.env.sim.get_articulated_object_manager()
        obj_idx = aom.get_object_id_by_handle(fur.handle)

        # Set surface index
        surface_idx = 0
        if target_is_receptacle:
            surface_idx = get_receptacle_index(self.target_handle, self.env.receptacles)

        return obj_idx, surface_idx

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        # Following the try/except pattern used in other skills
        try:
            target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
                self.target_handle
            )
            return f"Opening {target_node.name}"
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
        for_place_skill=False,
    ):
        action, rnn_hidden_states = super(PickSkillPolicy, self)._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        # Check if the agent's ee is near the art obj
        if not self.was_successful:
            self.was_successful = self._does_arm_reach_handle(
                observations, cur_batch_idx
            )

        # The following is from oracle open skill, only activates when the arm is near
        furniture, joint_idx, target_is_receptacle = self._get_art_obj_info()

        # Get the oracle action
        obj_idx, surface_idx = self._get_oracle_skill_param(
            furniture, joint_idx, target_is_receptacle
        )

        if obj_idx is None or surface_idx is None:
            return torch.zeros_like(prev_actions), rnn_hidden_states

        if self.was_successful:
            # If the agent reaches the handle, then we retract the arm
            # Retract the arm if the agent is holding something
            action = self._retract_arm_action(observations, action)

            # Populate the action tensor
            action[cur_batch_idx, self.open_flag_index] = 1
            action[cur_batch_idx, self.object_index] = obj_idx
            action[cur_batch_idx, self.surface_flag_index] = target_is_receptacle
            action[cur_batch_idx, self.surface_index] = surface_idx

        return action, rnn_hidden_states
