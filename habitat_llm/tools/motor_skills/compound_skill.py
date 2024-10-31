# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from habitat_llm.tools.motor_skills.skill import SkillPolicy


class CompoundSkill(SkillPolicy):
    """
    Initializes a CompoundSkill instance. This is a skill that executes a sequence of skills in order.

    :param config: A dictionary containing configuration settings.
    :param skills: A list of SkillPolicy instances to be combined.
    """

    def __init__(self, config, skills):
        assert len(skills) > 0, "At least one skill should be provided"
        first_skill = skills[0]
        action_space = first_skill.action_space
        batch_size = first_skill._batch_size
        self.env = first_skill.env
        agent_uid = first_skill.agent_uid
        self.skills = skills
        super().__init__(
            config,
            action_space,
            batch_size,
            env=self.env,
            should_keep_hold_state=True,
            agent_uid=agent_uid,
        )

        # Variable to indicate then end of the skill sequence
        self._did_finish_skill = torch.zeros(self._batch_size)

        self.active_skill = 0
        self.reset([0])

    def get_low_level_action(self, observations, deterministic=False):
        """
        Gets the low-level action for the compound skill.

        :param observations: The current observations.
        :param deterministic: Passed through to sub-skills.
        :return: A tuple containing the low-level action and a response message indicating success or failure.
        """
        action, resp = self.skills[self.active_skill].get_low_level_action(
            observations, deterministic
        )

        # Right now skills terminate when there is any non-empty response
        # hacky solution which should be replayed with something like response codes
        skill_success = "success" in resp.lower()
        if skill_success:
            if self.active_skill == self.num_skills - 1:
                self._did_finish_skill[[0]] = True
                self.finished = True
                resp = "Successful execution!"
                # Note: Following the pattern from SkillPolicy, we must reset skills immediately after success or failure
                self.reset([0])
            else:
                resp = ""
                self.active_skill = min(self.active_skill + 1, self.num_skills - 1)
        # Non empty response means skill failed if not success
        elif len(resp) > 0:
            self.termination_message = self.skills[
                self.active_skill
            ].termination_message
            self.failed = True
            # Note: Following the pattern from SkillPolicy, we must reset skills immediately after success or failure
            self.reset([0])
        return action, resp

    @property
    def num_skills(self):
        """
        :return: the number of skills in the CompoundSkill instance.
        """
        return len(self.skills)

    def set_target(self, target, env):
        """
        Sets the target for all skills in the CompoundSkill instance.

        :param target: The target to set.
        :param env: The environment to set the target in.
        """
        for skill in self.skills:
            skill.set_target(target, env)

    def reset(self, batch_idxs):
        """
        This method is executed each time this skill
        is called for the first time by the high level planner.
        Resets the critical members of the class.
        """

        super().reset(batch_idxs)

        for skill in self.skills:
            skill.reset(batch_idxs)

        self.active_skill = 0
        self._did_finish_skill = torch.zeros(self._batch_size)

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        """
        Unused arguments exist to match the signature of the parent class.

        :return: Returns a boolean tensor indicating if the skill is done.
        """
        return (self._did_finish_skill[batch_idx] > 0.0).to(masks.device)

    def get_state_description(self):
        """Method to get a string describing the state for this tool
        :return: A string describing the state for this tool
        """
        return self.skills[self.active_skill].get_state_description()
