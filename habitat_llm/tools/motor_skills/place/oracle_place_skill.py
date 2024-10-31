# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Tuple

import magnum as mn
import torch

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import (
    FURNITURE,
    OBJECT,
    OBJECT_OR_FURNITURE,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION,
)
from habitat_llm.utils.sim import ee_distance_to_object
from habitat_llm.world_model import Floor, Furniture

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


class OraclePlaceSkill(SkillPolicy):
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
        self.steps = 0
        self.thresh_for_art_state = config.thresh_for_art_state
        self.wait_time_for_obj_to_place = config.wait_time_for_obj_to_place

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        # Get grasp manager
        self.grasp_mgr = self.env.sim.agents_mgr[self.agent_uid].grasp_mgr

        # Get indices for release flag and target position
        self.action_range = find_action_range(
            self.action_space, f"agent_{self.agent_uid}_oracle_place_action"
        )
        # WARNING: this and all similar pieces of code in other skills
        # is relying on the alphabetical order of the action names in the action definition
        # these index assignments are all hardcoding the order based on the indices of the actions after sorting
        self.release_flag_index = self.action_range[0]
        self.target_position_start_index = self.action_range[1] - 3

        # Define the placement variable
        self.target_is_set = False
        self.object_to_be_moved = None
        self.spatial_relation = None
        self.place_entity = None
        self.spatial_constraint = None
        self.reference_object = None

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._is_action_issued = torch.zeros(self._batch_size)
        self.steps = 0
        self.target_is_set = False

        # Reset
        self.object_to_be_moved = None
        self.spatial_relation = None
        self.place_entity = None
        self.spatial_constraint = None
        self.reference_object = None

    def set_target(
        self,
        arg_string: str,
        env: "EnvironmentInterface",
    ) -> None:
        """The function to get furniture object, place location, place description.

        :param target_string: a furniture name with id, e.g., table_0
        :param env: an env
        :return: do not return anything, but set the variables
        """
        # Early return if the target is already set
        if self.target_is_set:
            return

        # Declare error message for incorrect api usage
        incorrect_api_usage_str = "Wrong use of API for place or rearrange. Please use [<object_to_be_moved>, <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint> (Optional - None otherwise), <reference_object> (Optional - None otherwise)]"

        # Early return if the input is empty
        if not arg_string:
            raise ValueError(incorrect_api_usage_str)

        # Separate the input string into values
        values = [value.strip() for value in arg_string.split(",")]

        # Early exit if 5 arguments are not provided
        if len(values) != 5:
            raise ValueError(incorrect_api_usage_str)

        # Early exit of any of the values are empty
        if any(v == "" for v in values):
            raise ValueError(incorrect_api_usage_str)

        # Separate the arguments
        (
            object_to_be_moved_name,
            spatial_relation,
            place_entity_name,
            spatial_constraint,
            reference_object_name,
        ) = values

        # Early exit if object_to_be_moved or place_entity is "None"
        if (
            object_to_be_moved_name.lower() == "none"
            or place_entity_name.lower() == "none"
        ):
            raise ValueError(incorrect_api_usage_str)

        # Get nodes from world graph for object to be moved and place receptacle
        self.object_to_be_moved = env.world_graph[self.agent_uid].get_node_from_name(
            object_to_be_moved_name
        )

        self.place_entity = env.world_graph[self.agent_uid].get_node_from_name(
            place_entity_name
        )

        # Early return if spatial_relation is "none"
        if spatial_relation.lower() == "none":
            raise ValueError(incorrect_api_usage_str)

        # Set spatial relation
        self.spatial_relation = spatial_relation.lower()

        # Handle case where spatial constraint is none and reference object is not
        if (
            spatial_constraint.lower() == "none"
            and reference_object_name.lower() != "none"
        ):
            raise ValueError(
                "Incorrect syntax for place/rearrange skill. reference_object was valid, but corresponding spatial_constraint was none"
            )

        # Handle case where spatial constraint is not none but the reference object is None
        if (
            spatial_constraint.lower() != "none"
            and reference_object_name.lower() == "none"
        ):
            raise ValueError(
                "Incorrect syntax for place/rearrange skill. Spatial_constraint was valid, but corresponding reference_object was none. Ensure that the spatial constraint is required for rearranging this object, else try without it. Alternatively, the reference_object entity might be erroneous."
            )

        # Handle case where both spatial constraint and reference object are valid
        if (
            spatial_constraint.lower() != "none"
            and reference_object_name.lower() != "none"
        ):
            self.spatial_constraint = spatial_constraint.lower()
            self.reference_object = env.world_graph[self.agent_uid].get_node_from_name(
                reference_object_name
            )

        # Set flag to true
        self.target_is_set = True

        return

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""

        # Return standing if any of these are found to be None
        # This can happen if the set target function fails due to incorrect syntax and
        # get_state_description is called
        if (
            self.object_to_be_moved is None
            or self.place_entity is None
            or self.spatial_relation is None
        ):
            return "Standing"

        # Do not involve reference_object and spatial_constraint
        if self.reference_object is None or self.spatial_constraint is None:
            return f"Placing {self.object_to_be_moved.name} {self.spatial_relation} {self.place_entity.name}"
        else:
            return f"Placing {self.object_to_be_moved.name} {self.spatial_relation} {self.place_entity.name} {self.spatial_constraint} {self.reference_object.name}"

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        # For this skill to be done, two events need to happen
        # 1. The place action should be issued
        # 2. The place action should be executed

        # Check if the pick action was issued
        was_place_action_issued = self._is_action_issued[batch_idx] > 0.0

        # Check if the gripper holds the target.
        was_place_successful = torch.tensor([False])
        if not self.grasp_mgr.is_grasped:
            was_place_successful = torch.tensor([True])

        is_done = was_place_action_issued and was_place_successful

        # Earlier logic used observations[IsHoldingSensor.cls_uuid].view(-1)
        # along with the logic above. Later figure out why this observation did not work
        # It would be best to rely on observations to decide if the skill is done rather
        # than using internal privileged information.

        return (is_done).to(masks.device)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # Increase the step count of this action
        self.steps += 1

        # Declare container for actions
        action = torch.zeros(prev_actions.shape, device=masks.device)

        # Early return if the arm doesn't contain any object
        if not self.grasp_mgr.is_grasped:
            self.termination_message = (
                "Failed to place! The agent is not holding any object."
            )
            self.failed = True
            return action, None

        # Early exit if the place_receptacle is not furniture or floor
        # Check for floor as well
        if not isinstance(self.place_entity, Furniture):
            self.failed = True
            self.termination_message = (
                "Failed to place! Place receptacle is not furniture or floor."
            )
            return action, None

        # Early return if the furniture not within reach or occluded
        max_distance = self._config.placing_distance
        if isinstance(self.place_entity, Floor):
            # get distance to the target location
            cur_agent = self.env.sim.agents_mgr[self.agent_uid].articulated_agent
            cur_agent_ee_pos = cur_agent.ee_transform().translation
            ee_dist_to_target = (
                mn.Vector3(cur_agent_ee_pos)
                - mn.Vector3(self.place_entity.get_property("translation"))
            ).length()
        else:
            # Furniture non-floor object
            ee_dist_to_target = ee_distance_to_object(
                self.env.sim,
                self.env.sim.agents_mgr,
                self.agent_uid,
                self.place_entity.sim_handle,
                max_distance=max_distance,
            )

        if ee_dist_to_target is None or ee_dist_to_target > max_distance:
            self.failed = True
            self.termination_message = f"Failed to place! Not close enough to {self.place_entity.name} or occluded."
            return action, None

        # Sample place location
        # The return of target_pos is a list
        try:
            target_poses: List[
                Tuple[mn.Vector3, mn.Quaternion]
            ] = self.place_entity.sample_place_location(
                self.spatial_relation,
                self.spatial_constraint,
                self.reference_object,
                self.env,
                self.articulated_agent,
                self.grasp_mgr,
            )
        except Exception as e:
            # Return failure if the agent fails sampling
            self.failed = True
            self.termination_message = (
                f"No valid placements found for entity due to {e}"
            )
            return action, None

        # Return with failure if no place location was found for that surface
        if len(target_poses) == 0:
            self.failed = True
            if self.reference_object != None and self.spatial_constraint != None:
                # Failed with spatial constraint (e.g. "next_to")
                # NOTE: we can try sampling again without the spatial constraint to better establish causality.
                target_poses_no_spatial_constraint: List[
                    Tuple[mn.Vector3, mn.Quaternion]
                ] = self.place_entity.sample_place_location(
                    self.spatial_relation,
                    None,
                    None,
                    self.env,
                    self.articulated_agent,
                    self.grasp_mgr,
                )
                if len(target_poses_no_spatial_constraint) > 0:
                    # very likely the next_to was the problem
                    self.termination_message = f"No valid placements found for entity {self.place_entity.name}. It looks like the spatial constraint {self.spatial_constraint} is not feasible because the reference object {self.reference_object.name} either does not exist or has not yet been placed on the {self.place_entity.name}. Try placing the grasped object {self.object_to_be_moved.name} on {self.place_entity.name} without the spatial constraint {self.spatial_constraint}."
                    return action, None

            self.termination_message = (
                f"No valid placements found for entity {self.place_entity.name}."
            )
            return action, None

        # use the first pose in the list (closest to the agent)
        target_pos, target_rot = target_poses[0]

        # Convert the target pos into torch tensor
        target_pos = torch.tensor(target_pos).to(action.device)

        # If all conditions are met, return action with True release flag and target position
        action[cur_batch_idx, self.release_flag_index] = 1
        action[
            cur_batch_idx,
            self.target_position_start_index + 0 : self.target_position_start_index + 3,
        ] = target_pos

        # Set the flag indicating the object has been placed
        self._is_action_issued[cur_batch_idx] = True

        # NOTE: we set the object's orientation from the sampled state immediately because the place action doesn't take or modify the orientation
        self.grasp_mgr.snap_rigid_obj.rotation = target_rot

        return action, None

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OraclePlaceSkill.

        :return: List of argument types.
        """
        none = '("none" | "None")'
        optional_constraint = f'(({SPATIAL_CONSTRAINT} "," WS {OBJECT_OR_FURNITURE} )| ({none} WS "," WS {none}))'
        return [OBJECT, SPATIAL_RELATION, FURNITURE, optional_constraint]
