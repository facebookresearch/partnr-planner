# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List

from habitat.sims.habitat_simulator.object_state_machine import get_state_of_obj
from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_id

from habitat_llm.tools.motor_skills.compound_skill import CompoundSkill
from habitat_llm.tools.motor_skills.nav.oracle_nav_skill import OracleNavSkill
from habitat_llm.tools.motor_skills.object_states.oracle_object_state_skill import (
    ObjectStateSkillResult,
    OracleObjectStateInPlaceSkill,
)
from habitat_llm.utils.grammar import OBJECT

if TYPE_CHECKING:
    from habitat_llm.sims.collaboration_sim import CollaborationSim
    from habitat_llm.sims.metadata_interface import MetadataInterface


class OraclePourInPlaceSkill(OracleObjectStateInPlaceSkill):
    """
    A skill to fill one container from another
    """

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
            "oracle_pour_action",
            config,
            observation_space,
            action_space,
            batch_size,
            env,
            agent_uid,
            maximum_distance,
        )

    def state_name(self):
        return "is_filled"

    def get_state_description(self):
        if self.target_handle is None:
            return "Standing"
        target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
            self.target_handle
        )
        return f"Pouring into {target_node.name}"

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

        # Check if the agent is holding an object.
        grasp_mgr = sim.agents_mgr[agent_uid].grasp_mgr
        if not grasp_mgr.is_grasped:
            return ObjectStateSkillResult(
                False,
                error_message_llm="Unable to pour: Agent is not holding an object",
                error_message_user="The object must be held to pour.",
            )

        # Check if the held object is full.
        obj = get_obj_from_id(sim, grasp_mgr.snap_idx)
        if not get_state_of_obj(obj, "is_filled"):
            return ObjectStateSkillResult(
                False,
                error_message_llm="Unable to pour: The held object is not filled",
                error_message_user="The object is empty.",
            )

        # Check distance to object
        dist = OracleObjectStateInPlaceSkill.distance_to_object_impl(
            sim, target_handle, agent_uid, maximum_distance
        )
        if dist is None or dist > maximum_distance:
            return ObjectStateSkillResult(
                False,
                error_message_llm=f"Agent too far from object to pour: Distance {dist}",
                error_message_user="The object is out of reach.",
            )
        return ObjectStateSkillResult.success()

    def can_modify_state(self) -> ObjectStateSkillResult:
        """
        Checks if the agent can modify the state of the environment for the pour action.

        This method checks if the agent is holding an object, if the held object is filled,
        and if the agent is within a maximum distance to the object.
        :return: Success or failure boolean and response string
        """
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
        Returns the types of arguments required for the OraclePourInPlaceSkill.

        :return: List of argument types.
        """
        return [OBJECT]


class OraclePourSkill(CompoundSkill):
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
        pour_skill = OraclePourInPlaceSkill(
            config.oracle_pour_in_place_skill_config,
            observation_space,
            action_space,
            batch_size,
            env=env,
            agent_uid=agent_uid,
        )
        super().__init__(config, [nav_skill, pour_skill])

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OraclePourSkill.

        :return: List of argument types.
        """
        return [OBJECT]
