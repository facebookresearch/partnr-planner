#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from transformers.generation import GenerationConfig
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint

from habitat_llm.llm.hf_model import VLMHFModel
from habitat_llm.llm.instruct.utils import image_data_to_pil


class MultiModalLlama(VLMHFModel):
    """Load llama using Hugging Face (HF)"""

    def init_local_model(self):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.generation_params.engine,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        self.processor = AutoProcessor.from_pretrained(self.generation_params.engine)
        self.tokenizer = self.processor.tokenizer

    def generate_hf_vlm(
        self,
        prompt: List[Tuple[str, str]],
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
        images = [img for type_prompt, img in prompt if type_prompt == "image"]
        text = [txt for type_prompt, txt in prompt if type_prompt == "text"]
        assert len(images) == 1, "This model should have one image"
        assert len(text) == 1, "This model should have one item of text"
        image_prompt = images[0]
        text_prompt = text[0]
        image_prompt = image_data_to_pil(image_prompt)

        model_inputs = self.processor(
            image_prompt, text_prompt, return_tensors="pt"
        ).to(self.model.device)
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
            **extra_generation_args,
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
