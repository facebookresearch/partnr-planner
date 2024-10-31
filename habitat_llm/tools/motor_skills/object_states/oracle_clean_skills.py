# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, List

from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle

from habitat_llm.tools.motor_skills.compound_skill import CompoundSkill
from habitat_llm.tools.motor_skills.nav.oracle_nav_skill import OracleNavSkill
from habitat_llm.tools.motor_skills.object_states.oracle_object_state_skill import (
    ObjectStateSkillResult,
    OracleObjectStateInPlaceSkill,
)
from habitat_llm.utils.grammar import OBJECT_OR_FURNITURE
from habitat_llm.world_model.object_states import ObjectIsClean

if TYPE_CHECKING:
    from habitat_llm.sims.collaboration_sim import CollaborationSim
    from habitat_llm.sims.metadata_interface import MetadataInterface


class OracleCleanInPlaceSkill(OracleObjectStateInPlaceSkill):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        batch_size,
        env,
        agent_uid,
        maximum_distance=1.5,
    ):
        super().__init__(
            "oracle_clean_action",
            config,
            observation_space,
            action_space,
            batch_size,
            env,
            agent_uid,
            maximum_distance,
        )

    def state_name(self):
        return "is_clean"

    def get_state_description(self):
        if self.target_handle is None:
            return "Standing"
        target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
            self.target_handle
        )
        return f"Cleaning {target_node.name}"

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

        obj = get_obj_from_handle(sim, target_handle)
        obj_type = metadata_interface.get_object_instance_category(obj)
        clean_state = [
            x
            for x in sim.object_state_machine.active_states
            if isinstance(x, ObjectIsClean)
        ][0]
        if (
            obj_type in clean_state.requires_faucet_semantic_classes
            and not OracleObjectStateInPlaceSkill.is_near_faucet_impl(
                sim, agent_uid, maximum_distance
            )
        ):
            return ObjectStateSkillResult(
                False,
                error_message_llm=f"Object {obj.object_id} requires faucet to clean, but agent is not near faucet.",
                error_message_user="The object requires a nearby faucet to clean.",
            )

        # Check distance to object
        dist = OracleObjectStateInPlaceSkill.distance_to_object_impl(
            sim, target_handle, agent_uid, maximum_distance
        )
        if dist is None or dist > maximum_distance:
            return ObjectStateSkillResult(
                False,
                error_message_llm=f"Agent too far from object to clean: Distance {dist}",
                error_message_user="The object is out of reach.",
            )
        return ObjectStateSkillResult.success()

    def can_modify_state(self) -> ObjectStateSkillResult:
        return self.can_modify_state_impl(
            self.env.sim,
            self.agent_uid,
            self.target_handle,
            self.state_name(),
            self.maximum_distance,
            self.env.perception.metadata_interface,
        )

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleCleanInPlaceSkill.

        :return: List of argument types.
        """
        return [OBJECT_OR_FURNITURE]


class OracleCleanSkill(CompoundSkill):
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
        clean_skill = OracleCleanInPlaceSkill(
            config.oracle_clean_in_place_skill_config,
            observation_space,
            action_space,
            batch_size,
            env=env,
            agent_uid=agent_uid,
        )
        super().__init__(config, [nav_skill, clean_skill])

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleCleanSkill.

        :return: List of argument types.
        """
        return [OBJECT_OR_FURNITURE]
