# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import select
import sys
from typing import Dict, List, Optional

import torch
import torch._dynamo.config
import torch._inductor.config

from rlm_utils.gpt_fast.tp import _get_rank

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

if hasattr(torch._dynamo.config, "max_loop_unroll_nodes"):
    torch._dynamo.config.max_loop_unroll_nodes = 7500

import json
import os

import typer

from rlm_utils.gpt_fast.gpt_fast_model import FastGPT

RLM_RESPONSE = "RLM_RESPONSE"

import logging

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(__name__)


def run_interactive(model, input_type="REST"):
    is_rank_0 = int(os.environ["LOCAL_RANK"]) == 0
    world_size = int(os.environ["WORLD_SIZE"])

    if input_type == "REST" and is_rank_0:
        status = json.dumps({"status": "ready"})
        print(f"{RLM_RESPONSE}{status}")
    num_requests = 0

    while True:
        keep_alive = False
        if is_rank_0:
            if input_type == "REST":
                try:
                    # wait for 5 min seconds for input othewise run keepalive request
                    rlist, _, _ = select.select([sys.stdin], [], [], 300)
                    if rlist:
                        request = json.loads(input())
                    else:
                        keep_alive = True
                        print("Runnning keepalive request")
                        request = {
                            "prompt": "keepalive",
                            "max_new_tokens": 10,
                            "temperature": 0.0,
                            "sampling": False,
                        }

                except json.decoder.JSONDecodeError:
                    response = json.dumps(
                        {"status": "error", "message": "invalid json"}
                    )
                    print(f"{RLM_RESPONSE}{response}")
                    continue
            else:
                # input_text = input("prompt: ")
                with open("input_file.txt", "r") as file:
                    input_text = file.read().strip()
                request = {
                    "prompt": input_text,
                    "max_new_tokens": 200,
                    "temperature": 0.0,
                    "sampling": False,
                    "grammar_definition": 'root ::= "test" | "hello"',
                }
            if is_rank_0 and not keep_alive:
                print(f"request: {request}")
            prompt = request["prompt"]
            max_new_tokens = request["max_new_tokens"]
            temperature = request["temperature"]
            grammar_definition = request.get("grammar_definition", None)
            sampling = request.get("sampling", False)
            data: List[Optional[str]] = [prompt] * world_size
            shared_kwargs: List[Optional[Dict]] = [
                {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "remove_prompts": True,
                    "grammar_definition": grammar_definition,
                    "sampling": sampling,
                }
            ] * world_size

        else:
            data = [None] * world_size
            shared_kwargs = [None] * world_size

        prompt_list = [None]
        kwargs_list = [None]
        if world_size > 1:
            logger.debug(f"rank: {_get_rank()}, waiting for scatter_prompt")
            torch.distributed.scatter_object_list(prompt_list, data, src=0)
            logger.debug(f"rank: {_get_rank()}, waiting for scatter_kwargs")
            torch.distributed.scatter_object_list(kwargs_list, shared_kwargs, src=0)
        else:
            kwargs_list = [shared_kwargs[0]]
            prompt_list = [data[0]]

        kwargs = kwargs_list[0]
        generated_text = model.generate_batch(
            prompt_list,  # type: ignore
            max_new_tokens=kwargs["max_new_tokens"],
            temperature=kwargs["temperature"],
            grammar_definition=kwargs["grammar_definition"],
            sampling=kwargs["sampling"],
        )

        if is_rank_0:
            response = json.dumps(generated_text[0])
            if not keep_alive:
                num_requests += 1
                print(f"{RLM_RESPONSE}{response}")
                print(f"num_requests: {num_requests}")


def main(
    checkpoint_dir: str,
    max_context_length: int = 2048,
    max_new_tokens: int = 200,
    seed: int = 1,
    add_bos: bool = False,
    early_stop: bool = True,
    temperature: float = 1.0,
    input_type: str = "REST",
):
    if input_type not in ["REST", "KB"]:
        print("Error: input type should either be REST (for web) or KB (for keyboard)")
    print("Starting process...")
    # sys.settrace(gpu_profile)
    model = FastGPT(
        checkpoint_dir,
        seed=seed,
        max_context_length=max_context_length,
        bos=add_bos,
        early_stop=early_stop,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    run_interactive(model, input_type=input_type)


if __name__ == "__main__":
    typer.run(main)
