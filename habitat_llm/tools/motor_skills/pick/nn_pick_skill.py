# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np
import torch

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills.nn_skill import NnSkillPolicy
from habitat_llm.utils.grammar import OBJECT
from habitat_llm.utils.sim import (
    check_if_gripper_is_full,
    check_if_the_object_is_held_by_agent,
    check_if_the_object_is_inside_furniture,
    check_if_the_object_is_moveable,
)


class PickSkillPolicy(NnSkillPolicy):
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
            should_keep_hold_state=True,
            env=env,
            agent_uid=agent_uid,
        )

        # Initialize the grasping manager
        self.grasp_mgr = self.env.sim.agents_mgr[self.agent_uid].grasp_mgr

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # Parameters for resetting the arm
        for k in action_space:
            if k == f"agent_{self.agent_uid}_arm_action":
                self.arm_start_id, self.arm_len = find_action_range(action_space, k)

        # Get indices for release flag and target position
        self.action_range_pick = find_action_range(
            action_space, f"agent_{self.agent_uid}_oracle_pick_action"
        )
        self.pick_flag_index = self.action_range_pick[0]
        self.pick_object_index = self.action_range_pick[1] - 1

        self._joint_rest_state = np.array(config.joint_rest_state)
        self.target_handle = None
        # A flag to check if the skill has failed. The reason for doing this is that
        # neural network skill needs to reset its arm after moving the arm.
        # We want to ensure that the skill has time to retract the arm.
        # This also reflects what we do in the real world.
        # Cache the information of failure
        self._has_failed = False
        # Cache the information of the termination msg
        self._termination_message_when_failing = ""

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OraclePickSkill.

        :return: List of argument types.
        """
        return [OBJECT]

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        # The pick skill is done only when the agent holding the object and
        # the end-effector at the resting position
        current_joint_pos = observations["joint"].cpu().numpy()
        rel_resting_pos = torch.norm(
            torch.tensor(self._joint_rest_state) - current_joint_pos, keepdim=True
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold
        is_holding = self.grasp_mgr.is_grasped
        return (is_holding * is_within_thresh).to(masks.device)

    def _check_grasping(self, action, observations, masks, cur_batch_idx):
        """Check if the agent can grasp the object with the given action"""

        # Early return if the gripper is already full.
        (
            action,
            self._termination_message_when_failing,
            self._has_failed,
        ) = check_if_gripper_is_full(
            self.env, action, self.grasp_mgr, self.target_handle
        )
        if self._has_failed:
            print("check_if_gripper_is_full")
            return action

        # Early exit if the object is not a movable object.
        (
            action,
            self._termination_message_when_failing,
            self._has_failed,
        ) = check_if_the_object_is_moveable(self.env, action, self.target_handle)
        if self._has_failed:
            print("check_if_the_object_is_moveable")
            return action

        # Early exit if the object is inside closed furniture.
        (
            action,
            self._termination_message_when_failing,
            self._has_failed,
        ) = check_if_the_object_is_inside_furniture(
            self.env,
            action,
            self.target_handle,
            self.config.threshold_for_ao_state,
        )
        if self._has_failed:
            print("check_if_the_object_is_inside_furniture")
            return action

        # Early exit if the object is being held by the other agent.
        (
            action,
            self._termination_message_when_failing,
            self._has_failed,
        ) = check_if_the_object_is_held_by_agent(
            self.env, action, self.target_handle, self.agent_uid
        )
        if self._has_failed:
            print("check_if_the_object_is_held_by_other_agent")
            return action

        return action

    def _try_to_pick(self, action, observations, masks, cur_batch_idx):
        """Pick the object using oracle Pick actions"""
        ee_pos = np.array(self.articulated_agent.ee_transform().translation)
        target_pos = self.target_pos
        ee_dist_to_target = np.linalg.norm(ee_pos - target_pos)

        # Get the object index.
        rom = self.env.sim.get_rigid_object_manager()
        obj_idx = rom.get_object_id_by_handle(self.target_handle)

        if ee_dist_to_target < self._config.pick_distance:
            # If all conditions are met, return action with True release flag and target position
            action[cur_batch_idx, self.pick_flag_index] = 1
            action[cur_batch_idx, self.pick_object_index] = obj_idx
            return action
        else:
            return action

    def _retract_arm_action(self, observations, action):
        """Retract the arm"""
        action = torch.zeros(action.shape, device=action.device)
        current_joint_pos = observations["joint"].cpu().numpy()
        # Compute the joint delta between the current joint state and the resting state
        joint_delta = self._joint_rest_state - current_joint_pos
        action[
            :, self.arm_start_id : self.arm_start_id + self.arm_len - 1
        ] = torch.from_numpy(joint_delta).to(device=action.device, dtype=action.dtype)
        return action

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

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        # Reset the internal flag
        self._has_failed = False
        self._termination_message_when_failing = ""

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
        for_place_skill=False,
        use_privileged_information: bool = True,
    ):
        action, rnn_hidden_states = super()._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        if not self.grasp_mgr.is_grasped and not for_place_skill:
            action = self._try_to_pick(action, observations, masks, cur_batch_idx)

        # Check if the arm is retracted
        current_joint_pos = observations["joint"].cpu().numpy()
        rel_resting_pos = torch.norm(
            torch.tensor(self._joint_rest_state) - current_joint_pos, keepdim=True
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold

        # Get the info to see if the agent wants to start retracting the arm to avoid skill time out.
        # Or the arm needs to do retraction if the skill has failed.
        # We leave 100 steps to let agent retract the arm.
        should_start_retract_arm = False
        if (
            self._config.max_skill_steps - self._cur_skill_step
        ) < 100 or self._has_failed:
            should_start_retract_arm = True

        # This means that the arm has been retracted, and it is because of skill
        # being not able to find the target or other failures
        if is_within_thresh and should_start_retract_arm:
            if self._termination_message_when_failing == "":
                self.termination_message = (
                    "Failed to pick! It took too long to execute pick skill."
                )
            else:
                # Use the cache information
                self.termination_message = self._termination_message_when_failing
            self.failed = True
            return torch.zeros(prev_actions.shape, device=masks.device), None

        # Check if the agent is grasping or do early arm retraction
        if (
            self.grasp_mgr.is_grasped or should_start_retract_arm
        ) and not for_place_skill:
            # Retract the arm if the agent is holding something
            action = self._retract_arm_action(observations, action)

        if (
            not self.grasp_mgr.is_grasped
            and use_privileged_information
            and not for_place_skill
            and not self._has_failed
        ):
            _ = self._check_grasping(action, observations, masks, cur_batch_idx)

        return action, rnn_hidden_states
