# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import gym.spaces as spaces
import numpy as np
import torch

# Habitat
from habitat_baselines.utils.common import get_num_actions

# Local
from habitat_llm.tools.motor_skills.nn_skill import SkillPolicy


class ResetArmSkill(SkillPolicy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, True)
        self._target = np.array([float(x) for x in config.reset_joint_state])

        self._ac_start = 0
        for k, space in action_space.items():
            if k != "arm_action":
                self._ac_start += get_num_actions(space)
            else:
                break

    def reset(
        self,
        batch_idxs,
    ):
        super().reset(
            batch_idxs,
        )

        # SID: I THINK THIS CLASS IS DEPRECATED AND NOT USED ANYWHERE
        # REMOVE IN FUTURE IF FOUND TRUE
        # self._initial_delta = self._target - observations["joint"].cpu().numpy()

        return

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ):
        current_joint_pos = observations["joint"].cpu().numpy()

        return (
            torch.as_tensor(
                np.abs(current_joint_pos - self._target).max(-1),
                device=rnn_hidden_states.device,
                dtype=torch.float32,
            )
            < 5e-2
        )

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        current_joint_pos = observations["joint"].cpu().numpy()
        delta = self._target - current_joint_pos

        # Dividing by max initial delta means that the action will
        # always in [-1,1] and has the benefit of reducing the delta
        # amount as we converge to the target.
        delta = delta / np.maximum(self._initial_delta.max(-1, keepdims=True), 1e-5)

        action = torch.zeros_like(prev_actions)

        action[..., self._ac_start : self._ac_start + 7] = torch.from_numpy(delta).to(
            device=action.device, dtype=action.dtype
        )

        return action, rnn_hidden_states
