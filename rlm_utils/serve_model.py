# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import submitit
import torch
from omegaconf import OmegaConf
from rlm.llm import RemoteLanguageModel
from rlm.transformers_llm import TransformersLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlm_utils.gpt_fast.grammar_sampler import GrammarSampler

RLM_RESPONSE = "RLM_RESPONSE"


class FastGPTLanguageModel(TransformersLanguageModel):
    """
    Uses a precompiled language model to speed up inference. Requires transforming the checkpoint
    first. For more information on compiling that checkpoint, check out: https://github.com/pytorch-labs/gpt-fast
    """

    def __init__(
        self,
        name_or_path: str,
        max_context_length=4000,
        max_new_tokens=200,
        temperature=1.0,
        ngpus=2,
        add_bos=False,
        early_stop=True,
    ) -> None:
        seed = 0
        early_stop_str = "early-stop" if early_stop else "no-early-stop"
        add_bos_str = "add-bos" if add_bos else "no-add-bos"
        self.proc = subprocess.Popen(
            [
                "env",
                "ENABLE_INTRA_NODE_COMM=1",
                "torchrun",
                "--standalone",
                f"--nproc_per_node={ngpus}",
                "-m",
                "rlm_utils.interactive_genfast_gpt",
                f"--max-context-length={max_context_length}",
                f"--max-new-tokens={max_new_tokens}",
                f"--{add_bos_str}",
                f"--{early_stop_str}",
                f"--seed={seed}",
                f"--temperature={temperature}",
                name_or_path,
            ],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        response = json.loads(self.readline())
        if response["status"] != "ready":
            raise RuntimeError(f"Unexpected response from subprocess: {response}")

    def readline(self, timeout=None):
        time.time()
        while True:
            response = self.proc.stdout.readline()
            if len(response) > 0:
                print(response)
            if response.startswith(RLM_RESPONSE):
                return response[len(RLM_RESPONSE) :]
            # if timeout is not None and time.time() - start_time > timeout:
            #     raise TimeoutError("Timeout while waiting for response")

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 1.0,
        sampling: bool = False,
        generation_args: Optional[Dict] = None,
    ) -> List[Dict]:
        results = []
        for prompt in prompts:
            results.append(
                self.generate(
                    prompt, max_new_tokens, temperature, sampling, generation_args
                )
            )
        return results

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1,
        sampling: bool = False,
        generation_args: Optional[Dict] = None,
    ) -> Dict:
        print(len(prompt))

        request_args = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "sampling": sampling,
        }
        if generation_args is not None:
            request_args["grammar_definition"] = generation_args.get(
                "grammar_definition", None
            )
        request = json.dumps(request_args)
        t1 = time.time()
        self.proc.stdin.write(f"{request}\n")
        resp = json.loads(self.readline(timeout=60))
        time_e = time.time() - t1
        all_num_tokens = resp["num_tokens"]
        speed = all_num_tokens / time_e
        prompt_tokens = resp["num_tokens_prompt"]
        print(
            f"Time ellapsed: {time_e:.2f}s, PromptTokens: {prompt_tokens}, OutputTokens: {all_num_tokens}, Speed: {speed:.2f} tokens/s."
        )
        return resp


class AutoTransformersLanguageModel(TransformersLanguageModel):
    def __init__(self, name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        self.device = self.model.device
        self.grammar_sampler = GrammarSampler('root ::= "temp"', self.tokenizer)

    def modified_super_batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 1.0,
        sampling: bool = False,
        device: Optional[str] = None,
        generation_args: Optional[Dict] = None,
    ) -> List[Dict]:
        # Copying the super function to avoid branching RLM

        tokenizer = self.tokenizer
        model = self.model
        if device is None:
            device = self.device
        if tokenizer.pad_token is None:
            logging.info("Pad token not set")
            if tokenizer.eos_token is None or tokenizer.eos_token == "":
                logging.info(
                    "Could not find good alternate token, using token at position 0 instead"
                )
                tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
            else:
                logging.info("Setting pad token to eos token")
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        if sampling:
            raise NotImplementedError("Sampling not implemented yet")

        do_sample = True
        if temperature == 0.0:
            do_sample = False

        extra_generation_args = {}
        if generation_args is not None and "grammar_definition" in generation_args:
            self.grammar_sampler.set_grammar(generation_args["grammar_definition"])
            extra_generation_args["logits_processor"] = [
                self.grammar_sampler.grammar_processor
            ]

        output = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=do_sample,
            **extra_generation_args,
        )
        assert len(inputs.input_ids.shape) == 2
        # Cut out the prompt
        n_prompt_tokens = inputs.input_ids.shape[1]
        text = tokenizer.batch_decode(
            output.sequences[:, n_prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        batched_out = []

        num_scores = len(output.scores)
        for batch_i in range(output.sequences.shape[0]):
            prompt = prompts[batch_i]
            generated_token_ids = output.sequences[batch_i, n_prompt_tokens:]
            generated_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids)
            # There are only scores for predicted tokens
            assert len(generated_token_ids) == len(generated_tokens) == num_scores

            n_output = len(generated_tokens)
            top_probs = []
            for i in range(n_output):
                token_id = generated_token_ids[i]
                token = generated_tokens[i]
                score_tensor = output.scores[i][batch_i]

                token_text = tokenizer.convert_tokens_to_string([token])
                if token_text == "\n":
                    if i == 0:
                        continue
                    else:  # noqa: RET507
                        break
                else:
                    assert len(score_tensor.shape) == 1
                    # TODO: Add temperature here
                    prob = torch.softmax(score_tensor, 0)[token_id]
                    top_probs.append(prob.cpu())

            batched_out.append(
                {
                    "generation": text[batch_i].strip(),
                    "prompt": prompt,
                    "mean_prob": float(np.mean(top_probs)),
                    "num_tokens": n_output,
                    "num_tokens_prompt": n_prompt_tokens,
                }
            )
        return batched_out

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 1.0,
        sampling: bool = False,
        device: Optional[str] = None,
        generation_args: Optional[Dict] = None,
    ) -> List[Dict]:
        t1 = time.time()
        print("Calling LLM...")
        batch_out = self.modified_super_batch_generate(
            prompts,
            max_new_tokens,
            temperature,
            sampling,
            self.model.device,
            generation_args=generation_args,
        )
        all_num_tokens = 0
        prompt_tokens = 0
        for batch in batch_out:
            # We don't need to store the prompt
            batch["prompt"] = ""
            if np.isnan(batch["mean_prob"]):
                batch["mean_prob"] = -1
            all_num_tokens += batch["num_tokens"]
            prompt_tokens += batch["num_tokens_prompt"]
        time_e = time.time() - t1
        speed = all_num_tokens / time_e
        print("LLM processed.")
        print(
            f"Time ellapsed: {time_e:.2f}s, PromptTokens: {prompt_tokens}, OutputTokens: {all_num_tokens}, Speed: {speed:.2f} tokens/s."
        )
        return batch_out


def setup_llm(config):
    engine_name = config.engine_name
    print(engine_name)
    print(f"Model Folder: {engine_name}")
    print("Loading model, this make take a bit...")
    if config.gptfast:
        # engine_name = f"{engine_name}/model.pth"
        # engine_name = "/home/xavierpuig/models/Meta-Llama-3-8B"
        engine_name = f"{engine_name}/model.pth"
    if config.gptfast:
        print(engine_name)
        model = FastGPTLanguageModel(
            engine_name,
            temperature=config.temperature,
            ngpus=config.ngpus,
            add_bos=config.add_bos,
            max_context_length=config.max_context_length,
        )
    else:
        model = AutoTransformersLanguageModel(engine_name)

    model.serve(config.ip, config.port)


def setup_llm_slurm(*, config):
    hostname = socket.gethostname()
    sock = socket.socket()
    sock.bind(("", 0))
    random_open_port = sock.getsockname()[1]
    sock.close()

    engine_name = config.engine_name
    print(engine_name)
    print(f"Model Folder: {engine_name}")
    print("Loading model, this make take a bit...")
    if config.gptfast:
        engine_name = f"{engine_name}/model.pth"
    if config.gptfast:
        print(engine_name)
        model = FastGPTLanguageModel(
            engine_name,
            temperature=config.temperature,
            ngpus=config.ngpus,
            add_bos=config.add_bos,
            max_context_length=config.max_context_length,
        )
    else:
        model = AutoTransformersLanguageModel(engine_name)
    out_dir = f"rlm/slurm/{config.exp_name}/server_list"
    address_file = Path(out_dir) / f"{hostname}:{random_open_port}"
    address_file.touch()
    model.serve(config.ip, random_open_port)


def keep_alive(*, config):
    socket.gethostname()
    sock = socket.socket()
    sock.bind(("", 0))
    sock.getsockname()[1]
    sock.close()

    out_dir = f"rlm/slurm/{config.exp_name}/server_list"
    log_dir = f"rlm/slurm/{config.exp_name}/log"

    server_ips = []
    models = {}
    while True:
        # Find the server ips
        if len(server_ips) < config.n_server:
            print("trying to find servers...")
            # Get the list of the name of the file
            try:
                dir_list = os.listdir(log_dir)
            except BaseException:
                dir_list = []
            # while loop to search a100
            for name in dir_list:
                name_ip = name
                if "a100" in name_ip and name_ip not in server_ips:
                    server_ips.append(name_ip)
                    # Create symlink
                    try:
                        os.symlink(f"{log_dir}/{name}", f"{out_dir}/{name}")
                    except Exception as e:
                        print(f"Cannot create symlink due to {e}")
            print(f"server ips collected {server_ips}")

        # Create the model based on the server ip
        for ip in server_ips:
            if ip not in models:
                models[ip] = RemoteLanguageModel(f"http://{ip}")

        # Keep model alive
        for model_ip in models:
            try:
                models[model_ip].generate(
                    "keep_alive", 5, temperature=config.temperature
                )
                print(f"{datetime.now()}: {model_ip} is running")
            except BaseException:
                print(f"{datetime.now()}: {model_ip} is not running")

        # We take a break if we have run the models
        if len(models) == config.n_server:
            time.sleep(60)
        elif len(models) != 0:
            time.sleep(15)


def call_rest_api():
    # Not implemented yet, will connect to rest api and read from user input.
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch LLM.")
    parser.add_argument("--port", default=4449, type=int, help="port number")
    parser.add_argument("--temperature", default=0.0, type=float, help="temperature")
    parser.add_argument("--ip", default="0.0.0.0", type=str, help="IP address")
    parser.add_argument("--gptfast", action="store_true")
    parser.add_argument("--add_bos", action="store_true")
    parser.add_argument("--ngpus", type=int, default=2)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sbatch", action="store_true")
    parser.add_argument("--n_server", default=1, type=int, help="the number of server")
    parser.add_argument("--days", default=1, type=int, help="the number of days")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="keep slurm jobs alive after closing this process",
    )
    parser.add_argument(
        "--exp_name", default="fastgptllama", type=str, help="name of fast gpt"
    )
    parser.add_argument(
        "--engine-name",
        type=str,
        help="Hugging face model name or path to the model",
    )
    parser.add_argument(
        "--max_context_length", default=4000, type=int, help="maximum context length"
    )
    args = parser.parse_args()
    config_name = f"rlm/slurm/{args.exp_name}/config.yaml"
    args_dict = vars(args)
    args_conf = OmegaConf.create(args_dict)
    if args.test:
        call_rest_api()
    else:
        if args.sbatch:
            log_dir = f"rlm/slurm/{args.exp_name}/log"
            if args.gptfast:
                keep_alive_log_dir = f"rlm/slurm/{args.exp_name}/keepalive"
            out_dir = f"rlm/slurm/{args.exp_name}/server_list"

            if os.path.exists(log_dir) or os.path.exists(out_dir):
                print(
                    f"The exp_name {args.exp_name} you specify is duplicated! Use other name. Terminate the program..."
                )
                exit()

            Path(log_dir).mkdir(exist_ok=True, parents=True)
            with open(config_name, "w+") as f:
                f.write(OmegaConf.to_yaml(args_conf))
            if args.gptfast:
                Path(keep_alive_log_dir).mkdir(exist_ok=True, parents=True)
                prepend_str = "fastgpt-api-"
            else:
                prepend_str = "hf-llm-api-"
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            executor = submitit.AutoExecutor(folder=log_dir, cluster=None)

            executor.update_parameters(
                cpus_per_task=10,
                slurm_ntasks_per_node=1,
                slurm_gres=f"gpu:{args.ngpus}",
                slurm_time=int(24 * 60 * args.days),
                slurm_job_name=f"{prepend_str}{args.exp_name}",
                slurm_account="siro",
            )

            try:
                if args.gptfast:
                    executor_light = submitit.AutoExecutor(
                        folder=keep_alive_log_dir, cluster=None
                    )

                    executor_light.update_parameters(
                        cpus_per_task=10,
                        slurm_ntasks_per_node=1,
                        slurm_gres=f"gpu:{1}",
                        slurm_time=int(24 * 60 * args.days),
                        slurm_job_name="keep_alive",
                        slurm_account="siro",
                    )
                # try:
                with executor.batch():
                    for _ in range(args.n_server):
                        job = executor.submit(
                            setup_llm_slurm,
                            config=args,
                        )

                if args.gptfast:
                    with executor_light.batch():
                        job = executor_light.submit(
                            keep_alive,
                            config=args,
                        )

                if shutil.which("squeue") is not None:
                    print("Waiting 5 seconds to print: squeue --me")
                    time.sleep(5)
                    subprocess.run(
                        "squeue --me --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R'",
                        shell=True,
                    )
                print(
                    f"Slurm nodes launched, ls into {log_dir} to check the servers available"
                )
                job.results()
                print("Done")
            finally:
                if not args.daemon:
                    base_out_dir = f"rlm/slurm/{args.exp_name}"
                    # Make sure we delete all files
                    shutil.rmtree(base_out_dir)
        else:
            setup_llm(args)
