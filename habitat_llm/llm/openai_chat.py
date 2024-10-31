#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from openai import AzureOpenAI

from habitat_llm.llm.base_llm import BaseLLM


class OpenAIChat(BaseLLM):
    def __init__(self, conf: DictConfig):
        """
        Initialize the chat model.
        :param conf: the configuration of the language model
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            assert len(api_key) > 0, ValueError("No OPENAI API keys provided")
        except Exception:
            raise ValueError("No OPENAI API keys provided")
        try:
            endpoint = os.getenv("OPENAI_ENDPOINT")
            assert len(endpoint) > 0, ValueError("No OPENAI endpoint keys provided")
        except Exception:
            raise ValueError("No OPENAI endpoint keys provided")
        self.client = AzureOpenAI(
            api_version="2024-06-01",
            api_key=api_key,
            azure_endpoint=f"https://{endpoint}",
        )
        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.verbose = True
        self.message_history: List[Dict] = []
        self.keep_message_history = self.llm_conf.keep_message_history

    def _validate_conf(self):
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")

    # @retry(Timeout, tries=3)
    def generate(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
    ):
        """
        Generate a response autoregressively.
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate.
        :param request_timeout: maximum time before timeout.
        :param generation_args: contains arguments like the grammar definition. We don't use this here
        """

        params = OmegaConf.to_object(self.generation_params)

        # Override stop if provided
        if stop is None:
            stop = self.generation_params.stop
        params["stop"] = stop

        # Override max_length if provided
        if max_length is not None:
            params["max_tokens"] = max_length

        messages = self.message_history.copy()
        # Add system message if no messages
        if len(messages) == 0:
            messages.append({"role": "system", "content": self.llm_conf.system_message})

        # Add current message
        messages.append({"role": "user", "content": prompt})
        params["messages"] = messages

        params["request_timeout"] = request_timeout

        try:
            response = self.client.chat.completions.create(
                model=params["model"], messages=params["messages"]
            )
            text_response = response.choices[0].message.content
            self.response = text_response
        except BaseException:
            print("call to openAI API failed")

        # Update message history
        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        if stop is not None:
            text_response = text_response.split(stop)[0]
        return text_response
