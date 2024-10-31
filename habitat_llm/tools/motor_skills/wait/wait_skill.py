# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from habitat_llm.tools.motor_skills.skill import SkillPolicy


class WaitSkill(SkillPolicy):
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
        self.steps_elapsed = torch.zeros(self._batch_size)
        # TODO: Revert from hardcoded thresh to specific in future.
        # self.step_threshold = int(self._config.sim_freq) * self._config.default_wait_time * 60
        self.step_threshold = int(self._config.sim_freq) * 5

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self.steps = 0
        self.steps_elapsed = torch.zeros(self._batch_size)
        # TODO: Revert from hardcoded thresh to specific in future.
        # self.step_threshold = int(self._config.sim_freq) * self._config.default_wait_time * 60
        self.step_threshold = int(self._config.sim_freq) * 5
        return

    def set_target(self, wait_time, env):
        # TODO: Revert from hardcoded thresh to specific in future.
        # self.step_threshold = int(self._config.sim_freq) * self.get_number(wait_time) * 60
        self.step_threshold = int(self._config.sim_freq) * 5
        return

    def get_number(self, string):
        num_str = ""

        for char in string:
            if char.isdigit():
                num_str += char
            elif num_str:
                break

        if num_str:
            return int(num_str)
        else:
            raise ValueError(
                "Input to Wait action needs to be a number indicating wait time"
            )

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        return (self.steps_elapsed[batch_idx] > self.step_threshold).to(masks.device)

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        return "Waiting"

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
        self.steps_elapsed[cur_batch_idx] += 1

        # Declare container for storing action values
        action = torch.zeros(prev_actions.shape, device=masks.device)

        return action, None

    @property
    def argument_types(self) -> List[str]:
        return []
