# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List

import torch
from habitat.core.logging import logger

from habitat_llm.tools.motor_skills.nav.oracle_nav_skill import OracleNavSkill
from habitat_llm.tools.motor_skills.pick.oracle_pick_skill import OraclePickSkill
from habitat_llm.tools.motor_skills.place.oracle_place_skill import OraclePlaceSkill
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import (
    FURNITURE,
    OBJECT,
    OBJECT_OR_FURNITURE,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION,
)

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


# Declare enumeration to
class AtomicSkills(Enum):
    NAV_OBJ = 0
    PICK = 1
    NAV_REC = 2
    PLACE = 3


class OracleRearrangeSkill(SkillPolicy):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
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
            AtomicSkills.NAV_OBJ: OracleNavSkill(
                config.nav_skill_config,
                observation_space,
                action_space,
                batch_size,
                env,
                agent_uid,
            ),
            AtomicSkills.PICK: OraclePickSkill(
                config.pick_skill_config,
                observation_space,
                action_space,
                batch_size,
                env,
                agent_uid,
            ),
            AtomicSkills.NAV_REC: OracleNavSkill(
                config.nav_skill_config,
                observation_space,
                action_space,
                batch_size,
                env,
                agent_uid,
            ),
            AtomicSkills.PLACE: OraclePlaceSkill(
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

        # variable to store the args passed to each skill
        self._skill_args = {}

    @property
    def current_skill_name(self):
        return self.skills[self.active_skill].current_skill_name

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleRearrangeSkill.

        :return: List of argument types.
        """
        none = '("none" | "None")'
        optional_constraint = f'(({SPATIAL_CONSTRAINT} "," WS {OBJECT_OR_FURNITURE} )| ({none} WS "," WS {none}))'
        return [OBJECT, SPATIAL_RELATION, FURNITURE, optional_constraint]

    def set_target(self, arg_string, env):
        # We follow the format:
        # Rearrange[<object_to_be_moved>, <spatial_relation>, <furniture_to_be_placed>, <spatial_constraint>, <reference_object>]

        # Early return if the target is already set
        if self.target_is_set:
            return

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

        # Set targets for individual names
        self.skills[AtomicSkills.NAV_OBJ].set_target(object_to_be_moved, env)
        self._skill_args[AtomicSkills.NAV_OBJ] = object_to_be_moved
        self.skills[AtomicSkills.PICK].set_target(object_to_be_moved, env)
        self._skill_args[AtomicSkills.PICK] = object_to_be_moved
        self.skills[AtomicSkills.NAV_REC].set_target(place_receptacle, env)
        self._skill_args[AtomicSkills.NAV_REC] = place_receptacle
        self.skills[AtomicSkills.PLACE].set_target(arg_string, env)
        self._skill_args[AtomicSkills.PLACE] = arg_string

        # Set flag to true
        self.target_is_set = True

        return

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self.skills[AtomicSkills.NAV_OBJ].reset(batch_idxs)
        self.skills[AtomicSkills.PICK].reset(batch_idxs)
        self.skills[AtomicSkills.NAV_REC].reset(batch_idxs)
        self.skills[AtomicSkills.PLACE].reset(batch_idxs)
        self._is_rearrange_done[batch_idxs] = False
        self.active_skill = AtomicSkills.NAV_OBJ
        return

    def get_state_description(self):
        """Method to get a string describing current atomic action for this tool"""
        return self.skills[self.active_skill].get_state_description()

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
            raise ValueError("Active skill cannot be None in the OracleRearrangeSkill")
        # Check if the current skill is done
        skill_is_done = self.skills[self.active_skill]._is_skill_done(
            observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx
        )

        # Change active skill state
        if skill_is_done.sum() > 0:
            # communicate end of previous skill
            skill_termination_message = self.skills[
                self.active_skill
            ].termination_message
            if skill_termination_message == None or skill_termination_message == "":
                skill_termination_message = "success"
            self.env._composite_action_response = {
                self.agent_uid: (
                    str(self.active_skill),
                    self._skill_args[self.active_skill],
                    skill_termination_message,
                ),
            }
            logger.debug(
                f"[OracleRearrangeSkill] Skill {self.active_skill} is done. Response: {self.env._composite_action_response}"
            )
            # Mark end of rearrangement
            # TODO: maybe we can remove this
            if (self.active_skill.value + 1) == 4:
                self._is_rearrange_done[cur_batch_idx] = True
            new_skill_value = (self.active_skill.value + 1) % 4
            self.active_skill = AtomicSkills(new_skill_value)

        if self._is_rearrange_done[cur_batch_idx]:
            action = torch.zeros(prev_actions.shape)
            hxs = rnn_hidden_states
        else:
            # Get action for the active skill
            action, hxs = self.skills[self.active_skill]._internal_act(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                cur_batch_idx,
                deterministic,
            )

        # Fetch termination message from the skill
        self.termination_message = self.skills[self.active_skill].termination_message
        self.failed = self.skills[self.active_skill].failed

        return action, hxs

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        if self.active_skill == AtomicSkills.PLACE:
            # Check if the current skill is done
            skill_is_done = self.skills[self.active_skill]._is_skill_done(
                observations, rnn_hidden_states, prev_actions, masks, batch_idx
            )
            if skill_is_done[batch_idx]:
                self._is_rearrange_done[batch_idx] = True

        return (self._is_rearrange_done[batch_idx] > 0.0).to(masks.device)
