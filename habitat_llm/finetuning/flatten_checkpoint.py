# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelFlattener:
    def __init__(self, model_name: str, checkpoint_dir: str, device: str) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer, self.model = self.init_model(model_name)
        self.model = self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir

    def init_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        return tokenizer, model

    def load_and_flatten_checkpoint(self):
        print("loading from: ", self.checkpoint_dir)
        short_model_name = self.model_name.split("/")[-1]
        merged_dir = os.path.join(self.checkpoint_dir, short_model_name)
        print(merged_dir)
        os.makedirs(merged_dir, exist_ok=True)
        shutil.copy(
            f"{self.model_name}/original/tokenizer.model",
            f"{merged_dir}/tokenizer.model",
        )
        peft_model = PeftModel.from_pretrained(
            self.model,
            self.checkpoint_dir,
            torch_dtype=torch.float16,
        )
        merged_model = peft_model.merge_and_unload()
        print("Checkpoint loaded and LoRA weights flattened into the model.")
        merged_model.save_pretrained(merged_dir)
        self.tokenizer.save_pretrained(merged_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten LoRA weights into the model.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Path to the model."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory.",
    )

    args = parser.parse_args()

    flattener = ModelFlattener(args.model_name, args.checkpoint_dir, "cuda")
    flattener.load_and_flatten_checkpoint()
