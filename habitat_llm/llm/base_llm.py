# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Optional

from omegaconf import DictConfig


class BaseLLM:
    """
    Base LLM Class
    """

    def __init__(self, conf: DictConfig):
        """
        Initialize the HF Language Model
        :param conf: The Language Model config
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params

    def generate(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
    ):
        """
        Generate a response autoregressively.
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate
        """
        raise NotImplementedError
