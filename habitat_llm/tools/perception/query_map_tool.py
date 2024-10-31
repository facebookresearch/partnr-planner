#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from hydra.utils import instantiate

from habitat_llm.tools import Tool


class QueryMapTool(Tool):
    def __init__(self, llm_config, skill_config):
        super().__init__(skill_config.name)
        self.llm = instantiate(llm_config.llm)(llm_config)
        self.env = None
        self.skill_config = skill_config

    def set_environment(self, env):
        self.env = env

    @property
    def description(self) -> str:
        return self.skill_config.description

    def process_high_level_action(self, query, observations):
        if not self.env:
            raise ValueError("Environment not set, use set_env to set the environment")
        return None, self.env.map.query_map(input)
