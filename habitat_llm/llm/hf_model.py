#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import subprocess
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

try:
    from rlm.llm import RemoteLanguageModel

    from habitat_llm.llm.rlm_lock import RemotePoolLanguageModel
except ImportError:
    RemoteLanguageModel = None
from transformers import AutoModelForCausalLM, AutoTokenizer

from habitat_llm.llm.base_llm import BaseLLM, Prompt


class VLMHFModel(BaseLLM):
    def __init__(self, conf: DictConfig):
        """
        Initialize the HF Language Model
        :param conf: The Language Model config
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        self.max_tokens = self.generation_params.max_tokens
        self.inference_mode = self.llm_conf.inference_mode
        if self.inference_mode == "hf":
            self.init_local_model()
        elif self.inference_mode == "rlm":
            self.init_remote_model()
        else:
            print("HFModel does not support this inference mode")
            raise NotImplementedError

    def init_remote_model(self):
        if self.llm_conf.serverdir == "":
            host, port = self.llm_conf.host, self.llm_conf.port
            RLM_API_ADDRESS = f"http://{host}:{port}"
            self.model = RemoteLanguageModel(RLM_API_ADDRESS)
        else:
            serverdir = self.llm_conf.serverdir
            self.model = RemotePoolLanguageModel(serverdir)

    def init_local_model(self):
        raise NotImplementedError

    def show_gpu(self, msg):
        """
        A helpful function to show GPUs usage for the debugging purpose
        ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
        """

        def query(field):
            return subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={field}", "--format=csv,nounits,noheader"],
                encoding="utf-8",
            )

        def to_int(result):
            return int(result.strip().split("\n")[0])

        used = to_int(query("memory.used"))
        total = to_int(query("memory.total"))
        pct = used / total
        print("\n" + msg, f"{100*pct:2.1f}% ({used} out of {total})")

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        **kwargs,
    ):
        """
        Generate the response autoregressively.
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate
        """
        # Prepare the max_length and the stop word
        max_length = max_length if max_length is not None else self.max_tokens
        self.response: Optional[str] = None
        self.batch_response: Optional[List[str]] = None
        if stop is None:
            stop = self.generation_params.stop

        if self.inference_mode == "hf":
            self.generate_hf(prompt, stop, max_length, generation_args=generation_args)
        elif self.inference_mode == "rlm":
            self.generate_rlm(prompt, stop, max_length, generation_args=generation_args)
        else:
            raise Exception
        # Clean up the GPU cuda memory,
        # otherwise out-of-memory issues when calling multiple times
        # We also keep track of GPU usages here
        # self.show_gpu("after generation   :")
        torch.cuda.empty_cache()
        # self.show_gpu("after empty_cache():")
        if self.generation_params.batch_response:
            return self.batch_response
        else:
            return self.response

    def generate_hf(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output using hf
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation. If none, will use the value from the config
        :max_length: The max number of tokens to generate. If none, will use the value from the config
        """
        if isinstance(prompt, str):
            return self.generate_hf_llm(prompt, stop, max_length, generation_args)
        elif isinstance(prompt, list):
            return self.generate_hf_vlm(prompt, stop, max_length, generation_args)
        return None

    def generate_hf_llm(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    def generate_hf_vlm(
        self,
        prompt: List[Tuple[str, str]],
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError

    def generate_rlm(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate the instruction using a remote language model.
        This feature is disabled for now.
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :max_length: The max number of tokens to generate
        """
        # Generate the response
        if self.generation_params.temperature == 0:
            # This is to solve a warning/crash that doesn't allow
            # a model to have temperature 0. When temp is 0, we
            # set the do_sample parameter to False and set an arbitrary
            # temp > 0.
            self.generation_params.temperature = 0.1
            self.generation_params.do_sample = False

        self.response_raw: Dict[str, Any] = {}
        if self.generation_params.batch_response:
            self.batch_response = []
            # Repeat generation
            for _ in range(self.generation_params.n):
                self.response_raw = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=max_length,
                    temperature=self.generation_params.temperature,
                    sampling=self.generation_params.do_sample,
                    generation_args=generation_args,
                )
                generation = self.response_raw["generation"]
                # Split using stop word or multiple stop words from a list.
                if isinstance(stop, str):
                    generation = generation.split(stop)[0]
                else:
                    for s in stop:
                        if s in generation:
                            generation = generation.split(s)[0]
                            break
                # remove white spaces at the end
                generation = generation.rstrip()
                self.batch_response.append(generation)
        else:
            self.response_raw = self.model.generate(
                prompt=prompt,
                max_new_tokens=max_length,
                temperature=self.generation_params.temperature,
                sampling=self.generation_params.do_sample,
                generation_args=generation_args,
            )
            generation = self.response_raw["generation"]
            # Split using stop word or multiple stop words from a list.
            if isinstance(stop, str):
                generation = generation.split(stop)[0]
            else:
                for s in stop:
                    if s in generation:
                        generation = generation.split(s)[0]
                        break
            # remove white spaces at the end
            generation = generation.rstrip()
            self.response = generation

    def get_logprobs(self):
        """Get the log probability of the generated text"""
        if self.generation_params.batch_response and self.inference_mode == "hf":
            return [
                float(i.cpu().detach().numpy())
                for i in self.response_raw["sequences_scores"]
            ]
        elif self.generation_params.batch_response and self.inference_mode == "rlm":
            return [choice["mean_prob"] for choice in self.batch_response]
        else:
            return []


class HFModel(VLMHFModel):
    """Specific instance of HFModel that uses only language"""

    def init_local_model(self):
        """
        Initialize the HF Language Model
        :param conf: The Language Model config
        """

        # Load the model using model-tokenizer approach
        # Setting device_map "auto" allows the model to use multiple GPUs
        # Setting load_in_4bit False allows for fast inference
        # You might encounter an issue of tensor being not at the same cuda GPUs
        # when using transformers package from HF. The solution is to
        # go to site-packages/transformers/models/llama/modeling_llama.py line 821,
        # and do logits =
        # [F.linear(hidden_states.to("cuda"), lm_head_slices[i].to("cuda"))
        # for i in range(self.pretraining_tp)],
        # forcing hidden_states and lm_head_slices[i] are in the same GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.generation_params.engine,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.generation_params.engine, use_fast=False
        )
