# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import habitat.sims.habitat_simulator.sim_utilities as sutils
import numpy as np
import torch

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import OBJECT
from habitat_llm.utils.sim import (
    check_if_gripper_is_full,
    check_if_the_object_is_held_by_agent,
    check_if_the_object_is_inside_furniture,
    check_if_the_object_is_moveable,
)


class OraclePickSkill(SkillPolicy):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=False,
            agent_uid=agent_uid,
        )
        self.env = env
        self.steps = 0
        self.thresh_for_art_state = config.thresh_for_art_state

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # Get grasp manager
        self.grasp_mgr = self.env.sim.agents_mgr[self.agent_uid].grasp_mgr

        # Get indices for linear and angular velocities in the action tensor
        self.action_range = find_action_range(
            self.action_space, f"agent_{self.agent_uid}_oracle_pick_action"
        )
        self.grip_index = self.action_range[0]
        self.object_index = self.action_range[1] - 1

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._is_action_issued = torch.zeros(self._batch_size)
        self.steps = 0

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        try:
            target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
                self.target_handle
            )
            return f"Picking {target_node.name}"
        except Exception as e:
            print(
                f"WARNING: cannot get {self.target_handle} in graph due to {e}. Agent's state is standing"
            )
            return "Standing"

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        # For this skill to be done, two events need to happen
        # 1. The pick action should be issued
        # 2. The pick action should be executed

        # Check if the pick action was issued
        was_pick_action_issued = self._is_action_issued[batch_idx] > 0.0

        # Check if the gripper holds the target.
        was_pick_successful = torch.tensor([False])
        if self.grasp_mgr.is_grasped:
            rom = self.env.sim.get_rigid_object_manager()
            grasped_obj_handle = rom.get_object_handle_by_id(self.grasp_mgr.snap_idx)
            if grasped_obj_handle == self.target_handle:
                was_pick_successful = torch.tensor([True])

        is_done = was_pick_action_issued and was_pick_successful

        # Earlier logic used observations[IsHoldingSensor.cls_uuid].view(-1)
        # along with the logic above. Later figure out why this observation did not work
        # It would be best to rely on observations to decide if the skill is done rather
        # than using internal privileged information.

        return (is_done).to(masks.device)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # Increase the step count of this skill
        self.steps += 1

        # Declare container for storing action values
        action = torch.zeros(prev_actions.shape, device=masks.device)
        if self.failed:
            return action, None

        # Early return if the gripper is already full.
        action, self.termination_message, self.failed = check_if_gripper_is_full(
            self.env, action, self.grasp_mgr, self.target_handle
        )
        if self.failed:
            return action, None

        # Early exit if the object is not a movable object.
        action, self.termination_message, self.failed = check_if_the_object_is_moveable(
            self.env, action, self.target_handle
        )
        if self.failed:
            return action, None

        # Early exit if the object is being held by the other agent.
        (
            action,
            self.termination_message,
            self.failed,
        ) = check_if_the_object_is_held_by_agent(
            self.env, action, self.target_handle, self.agent_uid
        )
        if self.failed:
            return action, None

        # Early exit if the object is inside closed furniture.
        (
            action,
            self.termination_message,
            self.failed,
        ) = check_if_the_object_is_inside_furniture(
            self.env,
            action,
            self.target_handle,
            self.thresh_for_art_state,
        )
        if self.failed:
            return action, None

        # Get the object index.
        obj = sutils.get_obj_from_handle(self.env.sim, self.target_handle)
        if obj is None or obj.is_articulated:
            raise ValueError(
                f"Cannot find rigid object with name {self.target_handle} for picking."
            )
        # Early exit if the object is out of reach.
        ee_pos = np.array(self.articulated_agent.ee_transform().translation)
        target_pos = obj.translation
        # TODO: picking through walls is possible here without occlusion check
        ee_dist_to_target = np.linalg.norm(ee_pos - target_pos)
        if ee_dist_to_target > self._config.grasping_distance:
            self.failed = True
            self.termination_message = "Failed to pick! Not close enough to the object."
            return action, None

        # If all preconditions are met, populate the action vector
        action[cur_batch_idx, self.grip_index] = 1
        action[cur_batch_idx, self.object_index] = obj.object_id

        # Set the flag indicating success of grasp to true
        self._is_action_issued[cur_batch_idx] = True

        return action, None

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OraclePickSkill.

        :return: List of argument types.
        """
        return [OBJECT]
