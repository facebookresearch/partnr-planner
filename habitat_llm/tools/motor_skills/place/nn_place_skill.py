# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, List

import magnum as mn
import numpy as np
import torch

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills import PickSkillPolicy
from habitat_llm.utils.grammar import (
    FURNITURE,
    OBJECT,
    OBJECT_OR_FURNITURE,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION,
)
from habitat_llm.utils.sim import is_open
from habitat_llm.world_model import Furniture

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


class PlaceSkillPolicy(PickSkillPolicy):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config,
            observation_space,
            action_space,
            batch_size,
            env=env,
            agent_uid=0,
        )

        # Get indices for release flag and target position
        self.action_range = find_action_range(
            self.action_space, f"agent_{self.agent_uid}_oracle_place_action"
        )
        self.release_flag_index = self.action_range[0]
        self.target_position_start_index = self.action_range[1] - 3
        self.thresh_for_art_state = config.thresh_for_art_state

        # Define the placement variable
        self.first_enter_skill = True
        self.target_is_set = False
        self.target_pos = None
        self.object_to_be_moved = None
        self.spatial_relation = None
        self.place_entity = None
        self.spatial_constraint = None
        self.reference_object = None

    def reset(self, batch_idxs: List[int]):
        super().reset(batch_idxs)
        self.first_enter_skill = True
        self.target_is_set = False
        self.target_pos = None

        # Reset
        self.object_to_be_moved = None
        self.spatial_relation = None
        self.place_entity = None
        self.spatial_constraint = None
        self.reference_object = None
        return

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
        incorrect_api_usage_str = "Wrong use of API for place or rearrange. Please us [<object_to_be_moved>, <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint> (Optional - None otherwise), <reference_object> (Optional - None otherwise)]"

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
                "Incorrect syntax for place/rearrange skill. spatial_constraint was valid, but corresponding reference_object was none"
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
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        # The place skill is done only when the agent is not holding the object and
        # the end-effector at the resting position
        current_joint_pos = observations["joint"].cpu().numpy()
        rel_resting_pos = torch.norm(
            torch.tensor(self._joint_rest_state) - current_joint_pos, keepdim=True
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold
        is_not_holding = not self.grasp_mgr.is_grasped
        return (is_not_holding * is_within_thresh).to(masks.device)

    def _try_to_place(self, action, observations, masks, cur_batch_idx):
        """Place the object using oracle place actions"""

        # Get the ee location of the robot
        ee_pos = np.array(self.articulated_agent.ee_transform().translation)
        # Get the target placement location
        target_pos = self.target_pos
        # Compute the distance
        ee_dist_to_target = np.linalg.norm(ee_pos - target_pos)

        # You can only place the object is the distance is small
        if ee_dist_to_target < self._config.placing_distance:
            # Need to reset the action so that the arm does not move
            action = torch.zeros(action.shape, device=masks.device)
            # Convert the target pos into torch tensor
            target_pos = torch.tensor(target_pos).to(action.device)
            # If all conditions are met, return action with True release flag and target position
            action[cur_batch_idx, self.release_flag_index] = 1
            action[
                cur_batch_idx,
                self.target_position_start_index
                + 0 : self.target_position_start_index
                + 3,
            ] = target_pos
            return action
        else:
            # Just the arm action from neural network
            return action

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # Declare container for actions
        action = torch.zeros(prev_actions.shape, device=masks.device)

        # Early return if the arm does not contain any object.
        # This is only checked once when entering the skill at the first time
        # since when the robot is retracting the arm, there is no object holding
        # and we do not want to output error.
        if not self.grasp_mgr.is_grasped and self.first_enter_skill:
            self.termination_message = (
                "Failed to place! The agent is not holding any object."
            )
            self.failed = True
            return action, None

        # Early exit if the target is not a furniture.
        if not isinstance(self.place_entity, Furniture):
            self.failed = True
            self.termination_message = (
                " Failed to place! Place receptacle is not furniture or floor."
            )
            return action, None

        # If the target is articulated, make sure that its open
        if (
            isinstance(self.place_entity, Furniture)
            and self.place_entity.is_articulated()
            and (
                not is_open(
                    self.place_entity,
                    self.env,
                    self.thresh_for_art_state,
                )
            )
            and (self.spatial_relation == "inside" or self.spatial_relation == "within")
        ):
            self.failed = True
            self.termination_message = (
                "Failed to place! Furniture is closed, you need to open it first."
            )
            return action, None

        # We only sample placement location when the skill is called
        # at the first time
        if self.first_enter_skill:
            # Change the flag
            self.first_enter_skill = False
            # Sample place location,
            # The return of target_pos is a list
            try:
                target_poses = self.place_entity.sample_place_location(
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
                self.termination_message = (
                    f"No valid placements found for entity {self.place_entity.name}."
                )
                return action, None
            else:
                # Use the first one as the target placement location
                self.target_pos = target_poses[0][0]
                # TODO: this skill is not using the validated orientation from the sampled target pose
                # Set the sensor variable
                self.env.sim.dynamic_target = mn.Vector3(self.target_pos)

        # Check if the arm is retracted
        current_joint_pos = observations["joint"].cpu().numpy()
        rel_resting_pos = torch.norm(
            torch.tensor(self._joint_rest_state) - current_joint_pos, keepdim=True
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold

        # Get the info to see if the agent wants to start retracting the arm to avoid skill time out.
        # We leave 100 steps to let agent retract the arm
        should_start_retract_arm = False
        if (self._config.max_skill_steps - self._cur_skill_step) < 100:
            should_start_retract_arm = True

        # This means that the arm has been retracted, and it is because of skill
        # being not able to find the target
        if is_within_thresh and should_start_retract_arm:
            self.termination_message = (
                "Failed to place! It took too long to execute place skill."
            )
            self.failed = True
            return action, None

        # Overwrite the is holding sensor to be false
        observations["is_holding"] = torch.zeros_like(observations["is_holding"])

        # Get the action
        action, rnn_hidden_states = super()._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
            for_place_skill=True,
        )

        # Check if the agent is grasping the object currently or do early arm retraction
        if not self.grasp_mgr.is_grasped or should_start_retract_arm:
            # Retract the arm if the agent is not holding anything
            action = self._retract_arm_action(observations, action)
        else:
            # keep trying to place the object using mobile place
            action = self._try_to_place(action, observations, masks, cur_batch_idx)
        return action, rnn_hidden_states

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OraclePlaceSkill.

        :return: List of argument types.
        """
        none = '("none" | "None")'
        optional_constraint = f'(({SPATIAL_CONSTRAINT} "," WS {OBJECT_OR_FURNITURE} )| ({none} WS "," WS {none}))'
        return [OBJECT, SPATIAL_RELATION, FURNITURE, optional_constraint]
