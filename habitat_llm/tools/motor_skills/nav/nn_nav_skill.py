# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

# Local
from habitat_llm.tools.motor_skills.nn_skill import NnSkillPolicy
from habitat_llm.utils.grammar import NAV_TARGET


class NavSkillPolicy(NnSkillPolicy):
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

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        # We use this sensor to check the distance between the agent and the goal
        distance = observations["goal_to_agent_gps_compass"][batch_idx, 0]
        # We also need to check if the agent has used the skill at least one step
        if (
            distance < self.config.nav_success_dis_threshold
            and self._cur_skill_step > 0
        ):
            self._did_want_done[batch_idx] = 1.0
        return (self._did_want_done[batch_idx] > 0.0).to(masks.device)

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        return "Walking"

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleNavSkill.

        Right now only allowing objects, furniture and rooms as this is all the
        planner can reason about.

        :return: List of argument types.
        """
        return [NAV_TARGET]
