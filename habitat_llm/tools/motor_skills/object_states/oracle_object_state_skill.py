# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.sim import ee_distance_to_object, get_faucet_points

if TYPE_CHECKING:
    from habitat_llm.sims.collaboration_sim import CollaborationSim
    from habitat_llm.sims.metadata_interface import MetadataInterface


@dataclass
class ObjectStateSkillResult:
    """Result of an object state modification check."""

    succeeded: bool
    """Whether the object state can be changed."""

    error_message_llm: Optional[str]
    """Detailed error message for LLM agent feedback."""

    error_message_user: Optional[str]
    """User-facing error message for GUI applications."""

    @staticmethod
    def success() -> ObjectStateSkillResult:
        """Create a `ObjectStateSkillResult` that signal success."""
        return ObjectStateSkillResult(
            True, error_message_llm=None, error_message_user=None
        )


class OracleObjectStateInPlaceSkill(SkillPolicy):
    def __init__(
        self,
        action_name: str,
        config,
        observation_space,
        action_space,
        batch_size,
        env,
        agent_uid: int,
        maximum_distance: float,
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=False,
            agent_uid=agent_uid,
        )
        self.env = env
        self.steps = 0
        self.action_name = action_name
        self.maximum_distance = maximum_distance

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        self.action_range = find_action_range(
            self.action_space,
            f"agent_{self.agent_uid}_{self.action_name}",
        )
        # WARNING: this and all similar pieces of code in other skills
        # is relying on the alphabetical order of the action names in the action definition
        # these index assignments are all hardcoding the order based on the indices of the actions after sorting
        self.clean_index = self.action_range[0]
        self.object_index = self.action_range[1] - 1

        if "oracle_skill_duration_range" in config:
            # Duration range specifying how long the skill should take in steps, top value is exclusive
            self.duration_range: Tuple[int, int] = config.oracle_skill_duration_range
        else:
            # Default duration range to 1 step
            self.duration_range = (1, 2)

        # This has to be computed here because reset isn't actually called before the first step
        self.current_duration = np.random.randint(
            self.duration_range[0], self.duration_range[1]
        )

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._is_action_issued = torch.zeros(self._batch_size)
        self.steps = 0
        self.current_duration = np.random.randint(
            self.duration_range[0], self.duration_range[1]
        )

    def get_state_description(self):
        raise not NotImplementedError()

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        was_action_issued = self._is_action_issued[batch_idx] > 0.0
        is_done = was_action_issued
        return (is_done).to(masks.device)

    @staticmethod
    def distance_to_object_impl(
        sim: "CollaborationSim", target_handle: str, agent_uid: int, max_distance: float
    ) -> Optional[float]:
        """
        Returns the distance between the specified agent and object.
        Returns None if the object is occluded or beyond the specified maximum distance.

        :param: sim: CollaborationSim simulator.
        :param: agent_uid: Unique identifier of the agent to test.
        :param: target_handle: Handle of the object to test.
        :param: max_distance: Maximum distance. Returns None if the object is beyond this distance.

        This function purposefully avoids depending on the gym environment so that it can be called externally.
        TODO: Move this logic out of skills.
        """
        return ee_distance_to_object(
            sim, sim.agents_mgr, agent_uid, target_handle, max_distance
        )

    def distance_to_object(self):
        """
        Returns the distance from the agent's end effector to the target object, or None if the object is occluded.
        """
        return self.distance_to_object_impl(
            self.env.sim, self.agent_uid, self.target_handle, self.maximum_distance
        )

    @staticmethod
    def is_near_faucet_impl(
        sim: "CollaborationSim",
        agent_uid: int,
        maximum_distance: float,
    ) -> bool:
        """
        Returns true if the specified agent is near a faucet.
        :param: sim: CollaborationSim simulator.
        :param: agent_uid: Unique identifier of the agent to test.
        :param: maximum_distance: Maximum reach of the agent.

        This function purposefully avoids depending on the gym environment so that it can be called externally.
        TODO: Move this logic out of skills.
        """
        faucets = get_faucet_points(sim)
        for handle in faucets:
            # Note: This solution will compute the distance to the nearest surface of the object containing the faucet
            # not the faucet itself. This will allow the agent to use facets on one side of a large object even if,
            # it on the other side. Right now the navigation policy cannot navigate to a specific part of an object so
            # we are using this approximation for now.
            # TODO: When the navigation policy supports navigating to specific parts of objects, we should update this
            # check to be more restrictive.
            faucet_distance = ee_distance_to_object(
                sim, sim.agents_mgr, agent_uid, handle, maximum_distance
            )
            if faucet_distance is not None and faucet_distance <= maximum_distance:
                return True
        return False

    def is_near_faucet(self):
        """
        Determines if the agent is near any faucet within the maximum allowed distance.

        :return: True if the agent is within the maximum allowed distance of any faucet, False otherwise.
        """
        return self.is_near_faucet_impl(
            self.env.sim, self.agent_uid, self.maximum_distance
        )

    @staticmethod
    def target_affords_state(
        sim: "CollaborationSim", target_handle: str, state_name: str
    ):
        allowed_states = sim.object_state_machine.objects_with_states[target_handle]
        return state_name in [state.name for state in allowed_states]

    @staticmethod
    def can_modify_state_impl(
        sim: "CollaborationSim",
        agent_uid: int,
        target_handle: str,
        state_name: str,
        maximum_distance: float,
        metadata_interface: "MetadataInterface",
    ) -> ObjectStateSkillResult:
        """
        Returns true if the state can be modified.
        :param: sim: CollaborationSim simulator.
        :param: agent_uid: Unique identifier of the agent that attempts to modify the state.
        :param: target_handle: Handle of the object to modify.
        :param: maximum_distance: Maximum reach of the agent.
        :param: metadata_interface: MetadataInterface object.

        This function purposefully avoids depending on the gym environment so that it can be called externally.
        TODO: Move this logic out of skills.
        """
        if OracleObjectStateInPlaceSkill.target_affords_state(
            sim, target_handle, state_name
        ):
            return ObjectStateSkillResult.success()
        else:
            return ObjectStateSkillResult(
                False,
                error_message_llm=f"The targeted object does not afford the state: {state_name}",
                error_message_user="The state cannot be changed for this object.",
            )

    def can_modify_state(self) -> ObjectStateSkillResult:
        """
        Should return boolean indicating if the state can be modified by this skill and a string
        If the state can't be modified string should contain a message indicating why the state cannot be modified,
        otherwise the string is unused
        """
        return self.can_modify_state_impl(
            self.env.sim,
            self.agent_uid,
            self.target_handle,
            self.state_name(),
            self.maximum_distance,
            self.env.perception.metadata_interface,
        )

    def state_name(self):
        raise NotImplementedError()

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
        self.steps += 1

        # Declare container for storing action values
        action = torch.zeros(prev_actions.shape, device=masks.device)

        # Don't issue the action until the current duration has been reached
        if self.steps < self.current_duration:
            return action, None

        obj = get_obj_from_handle(self.env.sim, self.target_handle)
        obj_idx = obj.object_id

        # If all preconditions are met, populate the action vector
        result = self.can_modify_state()
        if result.succeeded:
            action[cur_batch_idx, self.clean_index] = 1
            action[cur_batch_idx, self.object_index] = obj_idx
        else:
            self.failed = True
            self.termination_message = result.error_message_llm
        self._is_action_issued[cur_batch_idx] = True
        return action, None
