# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills import OpenSkillPolicy


class CloseSkillPolicy(OpenSkillPolicy):
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

        # The following are the parameters for oracle close skill
        # Fetch the action range
        self.action_range = find_action_range(
            self.action_space, f"agent_{self.agent_uid}_oracle_close_action"
        )
        # Fetch the index of close flag (first element in the action space)
        self.close_flag_index = self.action_range[0]
        # Fetch the index of object id (second element in the action space)
        self.object_index = self.action_range[0] + 1
        # Fetch the index of is_surface_flag
        self.surface_flag_index = self.action_range[0] + 2
        # Fetch the index of surface_index
        self.surface_index = self.action_range[0] + 3
        # Set the mode
        self.mode = "close"

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        # Following the try/except pattern used in other skills
        try:
            target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
                self.target_handle
            )
            return f"Closing {target_node.name}"
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
        action, rnn_hidden_states = super()._internal_act(
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

        # The following is from oracle close skill, only activates when the arm is near
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
            action[cur_batch_idx, self.close_flag_index] = 1
            action[cur_batch_idx, self.object_index] = obj_idx
            action[cur_batch_idx, self.surface_flag_index] = target_is_receptacle
            action[cur_batch_idx, self.surface_index] = surface_idx

        return action, rnn_hidden_states
