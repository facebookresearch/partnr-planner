# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import numpy as np
import torch

from habitat_llm.tools.motor_skills.explore.oracle_explore_skill import (
    OracleExploreSkill,
)
from habitat_llm.tools.motor_skills.nav.nn_nav_skill import NavSkillPolicy
from habitat_llm.tools.motor_skills.skill import SkillPolicy


class ExploreSkillPolicy(OracleExploreSkill):
    """
    This skill uses nav skill to navigate objects to random furnitures
    in the world in order to search for an instance of the queried type.
    Whenever an instance of the queried type appears in the world, the
    skill exists with success.
    """

    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        SkillPolicy.__init__(
            self,
            config,
            action_space,
            batch_size,
            should_keep_hold_state=True,
            agent_uid=agent_uid,
        )
        self.env = env

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # Construct nav skill
        self.nav_skill = NavSkillPolicy(
            config.nav_skill_config,
            observation_space,
            action_space,
            batch_size,
            env,
            agent_uid,
        )

        self.visited_node_count = 0
        self.target_node_reached = False

        # Variable to indicate end of exploration
        self._is_exploration_done = torch.zeros(self._batch_size)

        # Get agent pose
        agent_pos = np.array(self.articulated_agent.base_pos)

        # Set target node pose
        self.target_node_pose = [
            math.floor(agent_pos[0]),
            math.floor(agent_pos[1]),
            agent_pos[2],
        ]

        self.object_found = False
        self.target_object = None
        self.target_handle = None
        self.target_room_name: str = None
        self.fur_queue = []
        self.target_fur_name = None

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # Throw error if nav skill is none
        if self.nav_skill == None:
            raise ValueError("Navigation skill cannot be None in the NNExploreSkill")

        # Initialize the actions and hidden states to zeros and None
        action = torch.zeros(prev_actions.shape, device=masks.device)
        hxs = None

        # Check if the target node has been reached
        self.target_node_reached = (
            self.nav_skill._is_skill_done(
                observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx
            ).sum()
            > 0
        )

        # Increase the step counter for the nav skill
        self.nav_skill._cur_skill_step[cur_batch_idx] += 1

        # Check if exploration is done
        if self.target_node_reached and len(self.fur_queue) == 0:
            self._is_exploration_done[cur_batch_idx] = True

            # Fetch termination message from the skill
            self.termination_message = self.nav_skill.termination_message
            self.failed = self.nav_skill.failed

            return action, hxs

        # Check if there are no furniture in the room to start with
        if self.target_fur_name is None and len(self.fur_queue) == 0:
            self._is_exploration_done[cur_batch_idx] = True

            self.failed = False

            # Fetch termination message from the skill
            self.termination_message = f"There is no furniture in {self.target_room_name} room. Explore another room."

            return action, hxs

        # Check if the target node has been reached
        # or if target was never set
        if self.target_fur_name is None or self.target_node_reached:
            # Select next furniture to visit
            fur_node = self.fur_queue.pop(0)
            self.target_fur_name = fur_node.name

            # Increase node counter
            self.visited_node_count += 1
            # print(f"\nself.visited_node_count {self.visited_node_count}")

            # Reset the skill so that the agent can set a new target continuously
            self.nav_skill.reset(cur_batch_idx)

            # Reset the step counter of the nav
            self.nav_skill._cur_skill_step[cur_batch_idx] = 0

        # Set nav skill target
        self.nav_skill.set_target(self.target_fur_name, self.env)

        # Get action for the navigation skill
        action, hxs = self.nav_skill._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        # Fetch termination message from the skill
        self.termination_message = self.nav_skill.termination_message
        self.failed = self.nav_skill.failed

        return action, hxs
