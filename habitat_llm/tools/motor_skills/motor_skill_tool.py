#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import random
from typing import List

import torch
from habitat_baselines.utils.common import get_num_actions
from hydra.utils import instantiate
from torch import nn

from habitat_llm.tools import Tool

# from habitat_llm.tools.motor_skills.reset_arm.reset_arm_skill import ResetArmSkill


class MotorSkillTool(Tool):
    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.skill_config = skill_config
        self.env = None
        self.reset = None
        self._use_torchscript = False
        # Flag to determine whether to use torchscript (non-oracle skills) based on the name of file
        if hasattr(self.skill_config, "load_ckpt_file"):
            self._use_torchscript = "torchscript" in skill_config.load_ckpt_file

        ## Error messages from motor skill, depending on our prompting architecture
        self.error_mode = "cot"

    @property
    def description(self) -> str:
        return self.skill_config.description

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""
        return self.skill.get_state_description()

    def set_environment(self, env):
        self.env = env
        self.habitat_baselines_conf = env.conf.habitat_baselines
        self.skill = self.__init_skill()

    def to(self, device):
        self.skill.to(device)

    def __init_skill(self):
        # TODO: What is happening here??
        # Does instantiate call the constructor?
        # Does line skill() call constructor again?
        skill = instantiate(self.skill_config.skill)
        skill = skill(
            self.skill_config,
            self.env.internal_observation_space,
            self.env.orig_action_space,
            1,
            self.env,
            self.agent_uid,
        )

        # If skill is nn - freeze visual encoder and reset critic
        if hasattr(skill, "actor_critic") and not self._use_torchscript:
            skill.actor_critic.eval()
            for param in skill.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

            if self.habitat_baselines_conf.rl.ddppo.reset_critic:
                nn.init.orthogonal_(skill.actor_critic.critic.fc.weight)
                nn.init.constant_(skill.actor_critic.critic.fc.bias, 0)

        # # Some skills require an arm reset before execution (pick, place)
        # if "reset_arm" in self.skill_config:
        #     self.reset = ResetArmSkill(
        #         self.skill_config.reset_arm, self.env.orig_action_space, 1
        #     )
        #     self.reset.to(self.env.device)

        # Instantiate hidden states
        skill.recurrent_hidden_states = torch.zeros(
            self.env.conf.habitat_baselines.num_environments,
            self.env.conf.habitat_baselines.rl.ddppo.num_recurrent_layers
            * 2,  # TODO why 2?
            self.env.ppo_cfg.hidden_size,
            device=self.env.device,
        )

        # Instantiate container for storing previous actions
        skill.prev_actions = torch.zeros(
            self.env.conf.habitat_baselines.num_environments,
            *(get_num_actions(self.env.action_space),),
            device=self.env.device,
            dtype=torch.float,
        )

        # Instantiate not done masks
        # When executing skills, the skill is not done at the beginning.
        # So we set the not done mask to be true.
        skill.not_done_masks = torch.ones(
            self.env.conf.habitat_baselines.num_environments,
            1,
            device=self.env.device,
            dtype=torch.bool,
        )

        return skill

    def process_high_level_action(self, target_string, observations):
        error_messages = [
            "Use the appropriate tool to get a valid name.",
            "This may not be the correct node name, try using appropriate tool to get the exact name. If that doesnt work, this node may just not exist yet, explore the house to discover.",
        ]
        # Make sure that the environment is set
        if self.env is None:
            raise Exception("No environment set, use set_environment(env)")

        # Set the target for this skill
        try:
            self.skill.set_target(target_string, self.env)
        except ValueError as e:
            # Reset the skill if set target fails
            self.skill.reset([0])

            # Return with appropriate error message
            if self.error_mode == "cot":
                return (
                    None,
                    f"{e}. {random.choice(error_messages)}",
                )
            elif self.error_mode == "st":
                return (
                    None,
                    f"{e}. The entity name may be wrong or the entity may not exist in the house. Use entity names that match with the names of rooms and furniture provided in the house description. If the house description has no entity matching the furniture or object, consider exploring the rooms that are likely to have those objects or furniture in the house.",
                )
            else:
                return (
                    None,
                    f"{e}. The entity name may be wrong or the entity may not exist in the house.",
                )

        # Get low level action from the skill
        low_level_action, msg = self.skill.get_low_level_action(observations)

        return low_level_action, msg

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the MotorSkillTool.

        :return: List of argument types.
        """
        return self.skill.argument_types
