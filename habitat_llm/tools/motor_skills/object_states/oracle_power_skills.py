# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List

from habitat_llm.tools.motor_skills.compound_skill import CompoundSkill
from habitat_llm.tools.motor_skills.nav.oracle_nav_skill import OracleNavSkill
from habitat_llm.tools.motor_skills.object_states.oracle_object_state_skill import (
    ObjectStateSkillResult,
    OracleObjectStateInPlaceSkill,
)
from habitat_llm.utils.grammar import OBJECT_OR_FURNITURE

if TYPE_CHECKING:
    from habitat_llm.sims.collaboration_sim import CollaborationSim
    from habitat_llm.sims.metadata_interface import MetadataInterface


class OraclePowerInPlaceSkill(OracleObjectStateInPlaceSkill):
    """
    power_on is a boolean indicating whether the skill is for powering on or off
    maximum_distance indicates the maximum distance the agent can be from the object to power on or off
    """

    def __init__(
        self,
        config,
        observation_space,
        action_space,
        batch_size,
        env,
        agent_uid,
        power_on=True,
        maximum_distance=1.5,
    ):
        self.polarity = "on" if power_on else "off"
        super().__init__(
            f"oracle_power_{self.polarity}_action",
            config,
            observation_space,
            action_space,
            batch_size,
            env,
            agent_uid,
            maximum_distance,
        )

    def state_name(self):
        return "is_powered_on"

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        if self.target_handle is None:
            return "Standing"
        target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
            self.target_handle
        )
        return f"Powering {self.polarity} {target_node.name}"

    @staticmethod
    def can_modify_state_impl(
        sim: "CollaborationSim",
        agent_uid: int,
        target_handle: str,
        state_name: str,
        maximum_distance: float,
        metadata_interface: "MetadataInterface",
    ) -> ObjectStateSkillResult:
        result = OracleObjectStateInPlaceSkill.can_modify_state_impl(
            sim,
            agent_uid,
            target_handle,
            state_name,
            maximum_distance,
            metadata_interface,
        )
        if not result.succeeded:
            return result

        # Check distance to object
        dist = OracleObjectStateInPlaceSkill.distance_to_object_impl(
            sim, target_handle, agent_uid, maximum_distance
        )
        if dist is None or dist > maximum_distance:
            return ObjectStateSkillResult(
                False,
                error_message_llm=f"Agent too far from object to power $polarity: Distance {dist}",
                error_message_user="The object is out of reach.",
            )
        return ObjectStateSkillResult.success()

    def can_modify_state(self) -> ObjectStateSkillResult:
        result = self.can_modify_state_impl(
            self.env.sim,
            self.agent_uid,
            self.target_handle,
            self.state_name(),
            self.maximum_distance,
            self.env.perception.metadata_interface,
        )
        if not result.succeeded:
            # Hack: The polarity depends on the skill's state, which is unavailable to the stateless `can_modify_state_impl`.
            #       For now, we simply replace the `$polarity` string in the stateful version of the function.
            result.error_message_llm = result.error_message_llm.replace(
                "$polarity", f"{self.polarity}"
            )
        return result

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleCleanSkill.

        :return: List of argument types.
        """
        return [OBJECT_OR_FURNITURE]


class OraclePowerOnInPlaceSkill(OraclePowerInPlaceSkill):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config,
            observation_space,
            action_space,
            batch_size,
            env,
            agent_uid=agent_uid,
            power_on=True,
        )


class OraclePowerOffInPlaceSkill(OraclePowerInPlaceSkill):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config,
            observation_space,
            action_space,
            batch_size,
            env,
            agent_uid=agent_uid,
            power_on=False,
        )


class OraclePowerOnSkill(CompoundSkill):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        nav_skill = OracleNavSkill(
            config.nav_skill_config,
            observation_space,
            action_space,
            batch_size,
            env=env,
            agent_uid=agent_uid,
        )
        power_on_skill = OraclePowerOnInPlaceSkill(
            config.oracle_power_on_in_place_skill_config,
            observation_space,
            action_space,
            batch_size,
            env=env,
            agent_uid=agent_uid,
        )
        super().__init__(config, [nav_skill, power_on_skill])

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OraclePowerOnSkill.

        :return: List of argument types.
        """
        return [OBJECT_OR_FURNITURE]


class OraclePowerOffSkill(CompoundSkill):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        nav_skill = OracleNavSkill(
            config.nav_skill_config,
            observation_space,
            action_space,
            batch_size,
            env=env,
            agent_uid=agent_uid,
        )
        power_off_skill = OraclePowerOffInPlaceSkill(
            config.oracle_power_off_in_place_skill_config,
            observation_space,
            action_space,
            batch_size,
            env=env,
            agent_uid=agent_uid,
        )
        super().__init__(config, [nav_skill, power_off_skill])

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OraclePowerOffSkill.

        :return: List of argument types.
        """
        return [OBJECT_OR_FURNITURE]
