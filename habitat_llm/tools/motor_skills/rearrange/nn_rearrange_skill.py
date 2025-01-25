# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from habitat.core.logging import logger

from habitat_llm.tools.motor_skills.nav.nn_nav_skill import NavSkillPolicy
from habitat_llm.tools.motor_skills.pick.nn_pick_skill import PickSkillPolicy
from habitat_llm.tools.motor_skills.place.nn_place_skill import PlaceSkillPolicy
from habitat_llm.tools.motor_skills.rearrange.oracle_rearrange_skill import (
    AtomicSkills,
    OracleRearrangeSkill,
)
from habitat_llm.tools.motor_skills.skill import SkillPolicy

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


class RearrangeSkillPolicy(OracleRearrangeSkill):
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
        self.env: "EnvironmentInterface" = env

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # Container to represent the active skill type
        self.active_skill = AtomicSkills.NAV_OBJ

        # Make a dictionary for easy use
        self.skills = {
            AtomicSkills.NAV_OBJ: NavSkillPolicy(
                config.nav_skill_config,
                observation_space,
                action_space,
                batch_size,
                env,
                agent_uid,
            ),
            AtomicSkills.PICK: PickSkillPolicy(
                config.pick_skill_config,
                observation_space,
                action_space,
                batch_size,
                env,
                agent_uid,
            ),
            AtomicSkills.NAV_REC: NavSkillPolicy(
                config.nav_skill_config,
                observation_space,
                action_space,
                batch_size,
                env,
                agent_uid,
            ),
            AtomicSkills.PLACE: PlaceSkillPolicy(
                config.place_skill_config,
                observation_space,
                action_space,
                batch_size,
                env,
                agent_uid,
            ),
        }

        # Variable to indicate end of nav-pick-nav-place sequence
        self._is_rearrange_done = torch.zeros(self._batch_size)

    def set_target(self, arg_string, env):
        # We want to call this set_target again and again because the target here is dynamic for NN skills
        # Do not add a check for self.target_is_set here. Its intentionally omitted

        # We follow the format:
        # Rearrange[<object_to_be_moved>, <spatial_relation>, <furniture_to_be_placed>, <spatial_constraint>, <reference_object>]

        # Split the comma separated values
        values = [value.strip() for value in arg_string.split(",")]

        # Make sure that there are 5 arguments
        if len(values) != 5:
            raise ValueError(
                "Wrong use of API for rearrange tool, Please rearrange[<object_to_be_moved>, <spatial_relation>, <furniture_to_be_placed>, <spatial_constraint>, <reference_object>]"
            )

        # Parse the arguments
        (
            object_to_be_moved,
            spatial_relation,
            place_receptacle,
            spatial_constraint,
            reference_object,
        ) = values

        # Only set targets for the active skill. Otherwise the dynamics target
        # will be overwritten by the next skill's target
        if self.active_skill in [AtomicSkills.NAV_OBJ, AtomicSkills.PICK]:
            self.skills[self.active_skill].set_target(object_to_be_moved, env)
            self._last_args = object_to_be_moved
        elif self.active_skill == AtomicSkills.PLACE:
            self.skills[self.active_skill].set_target(arg_string, env)
            self._last_args = arg_string
        else:
            self.skills[self.active_skill].set_target(place_receptacle, env)
            self._last_args = place_receptacle
        return

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # Throw error if active skill is none
        if self.active_skill == None:
            raise ValueError("Active skill cannot be None in the RearrangeSkillPolicy")

        # Get action for the active skill
        action, hxs = self.skills[self.active_skill]._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        # Check if the current skill is done
        skill_is_done = self.skills[self.active_skill]._is_skill_done(
            observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx
        )

        # Increase step counter for the active skill. Otherwise the composite
        # skills will not increase the step counter
        self.skills[self.active_skill]._cur_skill_step[cur_batch_idx] += 1

        # Change active skill state
        if skill_is_done.sum() > 0:
            skill_termination_message = self.skills[
                self.active_skill
            ].termination_message
            if skill_termination_message == None or skill_termination_message == "":
                skill_termination_message = "success"
            # communicate end of previous skill
            self.env._composite_action_response = {
                self.agent_uid: (
                    str(self.active_skill),
                    self._last_args,
                    skill_termination_message,
                ),
            }
            logger.debug(
                f"[RearrangeSkillPolicy] Skill {self.active_skill} is done. Response: {self.env._composite_action_response}"
            )

            # Mark end of rearrangement
            if (self.active_skill.value + 1) == 4:
                self._is_rearrange_done[cur_batch_idx] = True

            # Reset the skill if it is done
            self.skills[self.active_skill].reset(cur_batch_idx)

            # Get the next skill
            new_skill_value = (self.active_skill.value + 1) % 4
            self.active_skill = AtomicSkills(new_skill_value)

        # Fetch termination message from the skill
        self.termination_message = self.skills[self.active_skill].termination_message
        self.failed = self.skills[self.active_skill].failed

        return action, hxs
