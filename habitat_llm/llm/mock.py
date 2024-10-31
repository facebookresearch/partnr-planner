#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Optional

from habitat_llm.llm.base_llm import BaseLLM


class MockLLM(BaseLLM):
    """
    A Mock LLM to test integration. Will generate a predefined response
    independently of the input.
    """

    def __init__(self, conf):
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        self.responses = self.generation_params.responses
        self._index = 0

    def generate(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
    ):
        """
        Generate an output. It ignores the prompt and loops over a set of predefined responses.
        All the parameters will be ignored
        :param prompt: Input prompt
        :param stop: stop word
        :param max_length: Max number of tokens to generate
        """
        self._index += 1
        return self.responses[(self._index - 1) % len(self.responses)]
