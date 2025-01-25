#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

from habitat_llm.llm.hf_model import VLMHFModel
from habitat_llm.llm.instruct.utils import image_data_to_pil

try:
    from third_party.RoboPoint.robopoint.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from third_party.RoboPoint.robopoint.conversation import conv_templates
    from third_party.RoboPoint.robopoint.mm_utils import (
        process_images,
        tokenizer_image_token,
    )
    from third_party.RoboPoint.robopoint.model.builder import load_pretrained_model
except BaseException:
    load_pretrained_model, process_images, tokenizer_image_token = None, None, None
    DEFAULT_IM_END_TOKEN = ""
    DEFAULT_IM_START_TOKEN = ""
    DEFAULT_IMAGE_TOKEN = ""
    IMAGE_TOKEN_INDEX = ""
    conv_templates = None

ADDED_TEXT = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."


class RoboPoint(VLMHFModel):
    """Load llama using Hugging Face (HF)"""

    def init_local_model(self):
        use_flash_attn = False
        load_8bit = True
        load_4bit = False
        model_path = self.llm_conf.generation_params.engine
        model_base = None
        model_paths = model_path.split("/")
        if model_paths[-1].startswith("checkpoint-"):
            model_name = model_paths[-2] + "_" + model_paths[-1]
        else:
            model_name = model_paths[-1]

        device = "cuda"
        (
            self.tokenizer,
            self.model,
            self.processor,
            self.context_len,
        ) = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device=device,
            use_flash_attn=use_flash_attn,
        )

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
        if generation_args is None or len(generation_args) == 0:
            generation_args = self.llm_conf.generation_params

        # Prepare the model input from prompt
        images = [img for type_prompt, img in prompt if type_prompt == "image"]
        text: List[str] = [txt for type_prompt, txt in prompt if type_prompt == "text"]
        assert len(images) == 1, "This model should have one image"
        assert len(text) == 1, "This model should have one item of text"
        image_prompt = images[0]
        image_prompt_pil = image_data_to_pil(image_prompt)
        image_processed = process_images(
            [image_prompt_pil], self.processor, self.model.config
        )
        images = image_processed.to(self.model.device, dtype=torch.float16)

        # From RoboPoint https://github.com/wentaoyuan/RoboPoint/blob/master/robopoint/eval/model_vqa.py#L44
        qs = text[0]
        if DEFAULT_IMAGE_TOKEN not in qs:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        else:
            qs.split("\n", 1)[1]
        qs += f" {ADDED_TEXT}"
        conv = conv_templates[self.llm_conf.generation_params.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        text_prompt = conv.get_prompt()

        image_sizes = [image_prompt_pil.size]
        image_args = {"images": images, "image_sizes": image_sizes}
        input_ids = (
            tokenizer_image_token(
                text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        res = self.model.generate(
            inputs=input_ids,
            do_sample=generation_args["do_sample"],
            top_p=generation_args["top_p"],
            max_new_tokens=max_length,
            streamer=None,
            use_cache=True,
            **image_args,
        )
        res_str = self.tokenizer.decode(res[0])
        decode_text = res_str
        self.response = decode_text.strip()
