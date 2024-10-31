# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.world_model import Furniture, Object, Room

if hasattr(torch, "inference_mode"):
    inference_mode = torch.inference_mode
else:
    inference_mode = torch.no_grad


class SkillPolicy(Policy):
    def __init__(
        self,
        config,
        action_space,
        batch_size,
        should_keep_hold_state: bool = False,
        env=None,
        agent_uid=0,
    ):
        self._config = config
        self._batch_size = batch_size
        self.action_space = action_space
        self.needs_target = False
        self._should_keep_hold_state = should_keep_hold_state
        self._cur_skill_step = torch.zeros(self._batch_size)
        self.device = self._cur_skill_step.device

        self.action_assigned = False
        self._is_action_issued = torch.zeros(self._batch_size)

        self.agent_uid = agent_uid

        # This is the index of the stop action in the action space
        # This is commented because this is only used on nn skills
        # self._stop_action_idx, _ = find_action_range(action_space,
        # "rearrange_stop")

        # Flag to indicate if contains arm_action
        self._contains_arm_action = "arm_action" in action_space

        # This is the index of the grip action in the action space
        if self._contains_arm_action:
            _, self.grip_action_index = find_action_range(action_space, "arm_action")
            self.grip_action_index -= 1
            baselines_logger.debug("grip_action_index " + str(self.grip_action_index))

        # Declare hidden states, previous actions and done masks
        self.recurrent_hidden_states = None
        self.prev_actions = None
        self.not_done_masks = None

        # Variables related to state of the skill
        self.termination_message = ""
        self.finished = False
        self.failed = False

        # Variable to keep track of whether target was set
        self.target_is_set = False

        # Variables related to target
        self.target_pos = None
        self.target_handle = None
        self._logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self._logger.setLevel(logging.DEBUG)

    @property
    def current_skill_name(self):
        return self.__class__.__name__

    def to(self, device):
        self.device = device
        self._cur_skill_step = self._cur_skill_step.to(device)

    def _keep_holding_state(
        self, full_action: torch.Tensor, observations
    ) -> torch.Tensor:
        """
        Makes the action so it does not result in dropping or picking up an
        object. Used in navigation and other skills which are not supposed to
        interact through the gripper.
        """

        # Keep the same grip state as the previous action.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)

        # If it is not holding (0) want to keep releasing -> output -1.
        # If it is holding (1) want to keep grasping -> output +1.
        full_action[:, self.grip_action_index] = is_holding + (is_holding - 1.0)
        return full_action

    def _internal_log(self, s, observations=None):
        baselines_logger.debug(
            f"Skill {self._config.name} @ step {self._cur_skill_step}: {s}"
        )

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        return torch.zeros(observations.shape[0], dtype=torch.bool).to(self.device)

    def get_low_level_action(self, observations, deterministic=False):
        """
        This method processes sensor observation from a given agent to
        output a low level action and/or a text response if the skill breaks / finishes
        """

        # Initialize an empty container for actions and response
        actions = torch.zeros(
            self.prev_actions.shape, device=self.not_done_masks.device
        )
        response = ""

        # Finished : Did the skill reach the termination condition?
        self.finished = self._is_skill_done(
            observations,
            self.recurrent_hidden_states,
            self.prev_actions,
            self.not_done_masks,
            batch_idx=[0],
        )

        # Return if the skill has finished
        if self.finished:
            response = "Successful execution!"

            # Add action to previous actions
            self.prev_actions.copy_(actions)

            # Clip the actions tensor based on the bounds of actions space
            actions = self._clip_actions(actions)

            # Reset the skill if its finished
            self.reset([0])

            return actions[0], response

        # If the skill is not finished, perform act
        # Get the low level action and renewed hidden states
        with inference_mode():
            actions, self.recurrent_hidden_states = self.act(
                observations,
                self.recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                cur_batch_idx=[0],
                deterministic=deterministic,
            )

        # Check if the skill failed
        if self.act_failed():
            response = f"Unexpected failure! - {self.termination_message}"

            # Reset the skill if it failed
            self.reset([0])

        # Add inferred actions to previous actions
        self.prev_actions.copy_(actions)

        # Clip the actions tensor based on the bounds of actions space
        actions = self._clip_actions(actions)

        return actions[0], response

    def act_failed(self) -> torch.BoolTensor:
        # Check if the allowed step count exceeded
        step_count_exceeded = False

        if (self._config.max_skill_steps > 0) and (self._config.force_end_on_timeout):
            step_count_exceeded = self._cur_skill_step > self._config.max_skill_steps
            if step_count_exceeded:
                self.termination_message = "Skill took too long to finish."
                self.failed = True

        return self.failed

    def reset(self, batch_idxs: List[int]):
        """
        This method resets the critical members of the skill
        """

        self.recurrent_hidden_states = torch.zeros(
            self.env.conf.habitat_baselines.num_environments,
            self.env.conf.habitat_baselines.rl.ddppo.num_recurrent_layers
            * 2,  # TODO why 2?
            self.env.ppo_cfg.hidden_size,
            device=self.device,
        )

        self.prev_actions = torch.zeros(
            self.env.conf.habitat_baselines.num_environments,
            *(get_num_actions(self.env.action_space),),
            device=self.env.device,
            dtype=torch.float,
        )

        self.not_done_masks = torch.ones(
            self.env.conf.habitat_baselines.num_environments,
            1,
            device=self.env.device,
            dtype=torch.bool,
        )

        self._cur_skill_step[batch_idxs] = 0
        self._did_leave_start_zone = torch.zeros(self._batch_size, device=self.device)

        # Reset basic target properties of the skill
        self.target_handle = None
        self.target_pos = None
        self.target_is_set = False

        # Set basic state representing variables of the skill
        self.failed = False
        self.finished = False
        self.termination_message = ""

        return

    def set_target(self, target_name: str, env):
        """
        Set the target (receptacle, object) of the skill.

        :param target_name: The name of the target Entity.
        """

        # Early return if the target is already set
        if self.target_is_set:
            return

        # Get entity based on the target
        entity = self.env.world_graph[self.agent_uid].get_node_from_name(target_name)

        # non-privileged graph entities do not have any sim-handle; this logic handles assigning the
        # sim-handle of closest sim-object/furniture entity to non-privileged object/furniture entity respectively
        if entity.sim_handle is None:
            # find closest entity to assign as proxy sim-handle
            # TODO: @zephirefaith :BE: is there a way to make following less brittle
            if type(self).__name__ == "PlaceSkillPolicy":
                all_gt_entities = self.env.perception.gt_graph.get_all_nodes_of_type(
                    Furniture
                )
                # only keep furniture with placeable receptacle
                all_gt_entities = [
                    ent
                    for ent in all_gt_entities
                    if ent.sim_handle in self.env.perception.fur_to_rec
                ]
            elif type(self).__name__ == "PickSkillPolicy":
                all_gt_entities = self.env.perception.gt_graph.get_all_nodes_of_type(
                    Object
                )
            elif type(self).__name__ == "NavSkillPolicy":
                all_gt_entities = (
                    self.env.perception.gt_graph.get_all_nodes_of_type(Furniture)
                    + self.env.perception.gt_graph.get_all_nodes_of_type(Object)
                    + self.env.perception.gt_graph.get_all_nodes_of_type(Room)
                )
            # only keep entities that have a translation property
            all_gt_entities = [
                ent for ent in all_gt_entities if "translation" in ent.properties
            ]

            # find the closest entity to given target
            entity_positions = np.array(
                [
                    np.array(entity.properties["translation"])
                    for entity in all_gt_entities
                ]
            )
            entity_distance = np.linalg.norm(
                entity_positions - np.array(entity.properties["translation"]),
                axis=1,
            )
            closest_entity_idx = np.argmin(entity_distance)
            entity.sim_handle = all_gt_entities[closest_entity_idx].sim_handle
            self._logger.debug(
                f"Detected non-sim object. Matched {all_gt_entities[closest_entity_idx].name} with non-sim object {entity.name}"
            )

        # Get sim handle of the target
        self.target_handle = entity.sim_handle
        if hasattr(self, "_target_name"):
            self._target_name = target_name

        # Set target pos
        self.target_pos = entity.get_property("translation")

        # Set flag to True to avoid resetting the target
        self.target_is_set = True

        return

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        """
        :returns: Predicted action and next rnn hidden state.
        """
        self._cur_skill_step[cur_batch_idx] += 1
        action, hxs = self._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        # Call only when arm action (NN) is in use
        # Oracle pick or place actions do not require this step
        if self._should_keep_hold_state and self._contains_arm_action:
            action = self._keep_holding_state(action, observations)

        return action, hxs

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def _clip_actions(self, actions):
        # Get the clipping range
        clip_range = self.env.env.action_space

        # Clip the actions anc convert to np array
        actions = [
            np.clip(a.numpy(), clip_range.low, clip_range.high) for a in actions.cpu()
        ]

        return actions

    @property
    @abstractmethod
    def argument_types(self) -> List[str]:
        """
        :returns: A list of argument types that the skill expects.
        """

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        return cls(config, action_space, batch_size)
