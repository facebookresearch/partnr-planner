# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import gym.spaces as spaces
import magnum as mn
import torch

# Habitat
from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.utils.common import get_num_actions

from habitat_llm.agent.env.actions import find_action_range

# Local
from habitat_llm.tools.motor_skills.skill import SkillPolicy


class NnSkillPolicy(SkillPolicy):
    """
    Defines a skill to be used in the TP+SRL baseline.
    """

    def __init__(
        self,
        config,
        observation_space,
        action_space,
        batch_size,
        should_keep_hold_state: bool = False,
        env=None,
        agent_uid=0,
    ):
        """1
        :param action_space: The overall action space of the entire task, not task specific.
        """

        assert hasattr(
            config, "load_ckpt_file"
        ), "Use NN skill but do not provide checkpoint path"

        self._use_torchscript = "torchscript" in config.load_ckpt_file
        self._agent_uid = agent_uid
        self.config = config

        # This is the torchscript system to load the model
        if self._use_torchscript:
            # Load the model
            self.actor_critic = torch.jit.load(
                config.load_ckpt_file,
                map_location=torch.device("cpu"),
            )
            super().__init__(config, action_space, batch_size, should_keep_hold_state)
            # Setup the env and meta data
            self._did_want_done = torch.zeros(batch_size)
            self.env = env
            # Define the observation space
            expected_obs_space = [
                f"agent_{agent_uid}_{obs}" for obs in config.obs_space
            ]
            filtered_obs_space = spaces.Dict(
                {k: observation_space.spaces[k] for k in expected_obs_space}
            )
            self._filtered_obs_space = filtered_obs_space

            # Define the action space
            # expected_action_space is a list of strings that are the names of the actions
            expected_action_space = [
                f"agent_{agent_uid}_{act}" for act in config.action_space
            ]
            # filtered_action_space is a spaces dict that contains the actions we want
            filtered_action_space = spaces.Dict(
                {k: action_space.spaces[k] for k in expected_action_space}
            )
            self._filtered_action_space = filtered_action_space
            # Find the action ranges
            self._ac_start = []
            self._ac_len = []
            # Loop over all the possible action spaces
            for _action_space in expected_action_space:
                # Find the action range
                action_range = find_action_range(self.action_space, _action_space)
                self._ac_start.append(action_range[0])
                self._ac_len.append(action_range[1] - action_range[0])
        else:
            # This is normal way to load the model.
            try:
                ckpt_dict = torch.load(config.load_ckpt_file, map_location="cpu")
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "Could not load neural network weights for skill."
                ) from e

            policy_cfg = ckpt_dict["config"]
            policy = baseline_registry.get_policy(config.policy)

            expected_obs_space = policy_cfg.habitat.gym.obs_keys
            excepted_action_space = policy_cfg.habitat.task.actions.keys()

            filtered_obs_space = spaces.Dict(
                {k: observation_space.spaces[k] for k in expected_obs_space}
            )

            baselines_logger.debug(
                f"Loaded obs space {filtered_obs_space} for skill {config.name}",
            )

            baselines_logger.debug("Expected obs space: " + str(expected_obs_space))

            filtered_action_space = ActionSpace(
                OrderedDict((k, action_space[k]) for k in excepted_action_space)
            )

            if "arm_action" in filtered_action_space.spaces and (
                policy_cfg.habitat.task.actions.arm_action.grip_controller is None
            ):
                filtered_action_space["arm_action"] = spaces.Dict(
                    {
                        k: v
                        for k, v in filtered_action_space["arm_action"].items()
                        if k != "grip_action"
                    }
                )

            baselines_logger.debug(
                f"Loaded action space {filtered_action_space} for skill {config.name}",
            )
            baselines_logger.debug("=" * 80)

            actor_critic = policy.from_config(
                policy_cfg, filtered_obs_space, filtered_action_space
            )

            try:
                actor_critic.load_state_dict(
                    {  # type: ignore
                        k[len("actor_critic.") :]: v
                        for k, v in ckpt_dict["state_dict"].items()
                    }
                )

            except Exception as e:
                raise ValueError(
                    f"Could not load checkpoint for skill {config.name} from {config.load_ckpt_file}"
                ) from e

            super().__init__(config, action_space, batch_size, should_keep_hold_state)
            self.actor_critic = actor_critic
            self._filtered_obs_space = filtered_obs_space
            self._filtered_action_space = filtered_action_space
            self._ac_start = [0]
            self._ac_len = [get_num_actions(filtered_action_space)]
            self._did_want_done = torch.zeros(self._batch_size)

            self._internal_log(
                f"Skill {config.name}: action offset {self._ac_start}, action length {self._ac_len}"
            )

    def parameters(self):
        if self.actor_critic is not None:
            return self.actor_critic.parameters()
        else:
            return []

    @property
    def num_recurrent_layers(self):
        if self.actor_critic is not None:
            return self.actor_critic.net.num_recurrent_layers
        else:
            return 0

    def to(self, device):
        if not self._use_torchscript:
            super().to(device)
        self._did_want_done = self._did_want_done.to(device)
        if self.actor_critic is not None:
            self.actor_critic.to(device)

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._did_want_done *= 0.0

    def _get_filtered_obs(self, observations, cur_batch_idx) -> TensorDict:
        return TensorDict(
            {
                k.replace(f"agent_{self._agent_uid}_", ""): observations[
                    k.replace(f"agent_{self._agent_uid}_", "")
                ]
                for k in self._filtered_obs_space.spaces
            }
        )

    def set_target(self, target_name, env):
        super().set_target(target_name, env)
        entity = self.env.world_graph[self.agent_uid].get_node_from_name(target_name)
        # Set the navigation target for the point nav skill
        self.env.sim.dynamic_target = entity.properties["translation"]
        if not isinstance(self.env.sim.dynamic_target, mn.Vector3):
            self.env.sim.dynamic_target = mn.Vector3(self.env.sim.dynamic_target)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        filtered_obs = self._get_filtered_obs(observations, cur_batch_idx)
        input_prev_actions = []
        for i in range(len(self._ac_start)):
            input_prev_actions.append(
                prev_actions[:, self._ac_start[i] : self._ac_start[i] + self._ac_len[i]]
            )
        input_prev_actions = torch.cat(input_prev_actions, 1)

        # Get the actions
        if self._use_torchscript:
            action, rnn_hidden_states, _ = self.actor_critic(
                filtered_obs, rnn_hidden_states, input_prev_actions, masks
            )
        else:
            _, action, _, rnn_hidden_states = self.actor_critic.act(
                filtered_obs,
                rnn_hidden_states,
                input_prev_actions,
                masks,
                deterministic,
            )

        # Assign the action to the correct place
        full_action = torch.zeros(prev_actions.shape, device=masks.device)
        cur_len = 0
        # Loop over all the possible action spaces
        for i in range(len(self._ac_start)):
            full_action[
                :, self._ac_start[i] : self._ac_start[i] + self._ac_len[i]
            ] = action[:, cur_len : cur_len + self._ac_len[i]]
            cur_len += self._ac_len[i]

        if not self._use_torchscript:
            self._did_want_done[cur_batch_idx] = full_action[
                cur_batch_idx, self._stop_action_idx
            ]
            full_action[cur_batch_idx, self._stop_action_idx] = 0.0

        return full_action, rnn_hidden_states

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        return cls(config, observation_space, action_space, batch_size)
