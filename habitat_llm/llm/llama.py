#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from transformers.generation import GenerationConfig
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint

from habitat_llm.llm.hf_model import HFModel


class Llama(HFModel):
    """Load llama using Hugging Face (HF)"""

    def __init__(self, conf):
        super().__init__(conf)

    def generate_hf(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output using hf
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :max_length: The max number of tokens to generate
        """
        # Prepare the model input from prompt
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Set the generating parameters
        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        gen_cfg.max_new_tokens = max_length
        gen_cfg.do_sample = self.generation_params.do_sample
        gen_cfg.num_return_sequences = self.generation_params.n
        gen_cfg.num_beams = self.generation_params.best_of
        gen_cfg.temperature = self.generation_params.temperature
        gen_cfg.repetition_penalty = self.generation_params.repetition_penalty
        gen_cfg.top_k = self.generation_params.top_k
        gen_cfg.top_p = self.generation_params.top_p
        gen_cfg.output_scores = True
        gen_cfg.return_dict_in_generate = True

        if stop is None:
            stop = self.generation_params.stop
        if max_length is None:
            max_length = self.generation_params.max_tokens

        extra_generation_args = {}
        if generation_args is not None and "grammar_definition" in generation_args:
            # These add constrained generation to the hugging face
            # inference as a logits processor.
            # IncrementalGrammarConstraint contains the core code for running
            # the grammar based constraint
            # and GrammarConstrainedLogitsProcessor is mainly an interface
            # between Hugging Face and the grammar constraint.
            grammar = IncrementalGrammarConstraint(
                generation_args["grammar_definition"], "root", self.tokenizer
            )
            grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
            extra_generation_args["logits_processor"] = [grammar_processor]

        # Generate the response
        self.response_raw = self.model.generate(
            **model_inputs,
            generation_config=gen_cfg,
            pad_token_id=self.tokenizer.eos_token_id,
            **extra_generation_args
        )

        # Only decode the response, not including the prompt
        input_prompt_len = model_inputs.input_ids.shape[1]
        # Decode the response
        decode_text = self.tokenizer.batch_decode(
            self.response_raw["sequences"][:, input_prompt_len:],
            skip_special_tokens=True,
        )

        # Process the response
        if self.generation_params.batch_response:
            # Return a list of response
            if isinstance(stop, str):
                self.batch_response = [
                    res.split(stop)[0].rstrip() for res in decode_text
                ]
            else:
                found_stop = False
                for s in stop:
                    if found_stop:
                        break
                    for res in decode_text:
                        if s in res:
                            self.batch_response = [res.split(s)[0].rstrip()]
                            found_stop = True
        else:
            # Return a single response
            response = ""
            if isinstance(stop, str):
                response = decode_text[0].split(stop)[0]
            else:
                for s in stop:
                    if s in decode_text:
                        response = decode_text[0].split(s)[0]
                        break
            self.response: str = response.rstrip()

    def generate_rlm(
        self,
        prompt: str,
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
        # Generation time
        if self.generation_params.temperature == 0:
            # This is to solve a warning/crash that doesn't allow
            # a model to have temperature 0. When temp is 0, we
            # set the do_sample parameter to False and set an arbitrary
            # temp > 0.
            self.generation_params.temperature = 0.1
            self.generation_params.do_sample = False

        if self.generation_params.batch_response:
            self.response_raw = []
            self.batch_response = []
            # Repeat generation
            for _ in range(self.generation_params.n):
                self.response_raw.append(
                    self.model.generate(
                        prompt=prompt,
                        max_new_tokens=max_length,
                        temperature=self.generation_params.temperature,
                        sampling=self.generation_params.do_sample,
                        generation_args=generation_args,
                    )
                )
                generation = self.response_raw[-1]["generation"]
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
