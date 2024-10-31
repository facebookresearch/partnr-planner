# mypy: ignore-errors
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config

from rlm_utils.gpt_fast.tokenizer import get_tokenizer
from rlm_utils.gpt_fast.tp import _get_rank, _get_world_size, maybe_init_dist

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


import logging

from transformers import AutoTokenizer

from rlm_utils.gpt_fast.grammar_sampler import GrammarSampler
from rlm_utils.gpt_fast.model import Transformer

logger = logging.getLogger(__name__)


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


def disable_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print_f(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print_f


class FastGPT:
    def __init__(
        self,
        checkpoint_dir: str,
        seed: int = 1,
        max_context_length: int = 1000,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        bos: bool = False,
        early_stop: bool = False,
        device="cuda",
    ):
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.add_bos = bos
        self.early_stop = early_stop
        assert (
            max_new_tokens < max_context_length
        ), "Error the number of tokens should be smaller than the max context length,"
        self.rank = maybe_init_dist()
        self.use_tp = self.rank is not None
        # if self.use_tp:
        #     disable_print(self.rank == 0)

        checkpoint_path = Path(checkpoint_dir)
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        self.device = device
        precision = torch.bfloat16

        self.compile = True
        self.compile_prefill = False

        self.model = self._load_model(checkpoint_path, device, precision, self.use_tp)
        device_sync(device=device)  # MKG
        self.tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
        # initialize with a temporary gramar to build the tokenizer trie
        self.grammar_sampler = GrammarSampler('root ::= "tmp"', self.hf_tokenizer)
        if self.compile:
            # type: ignore
            print("Compiling...")
            self.compute_logits = torch.compile(
                self.compute_logits, mode="reduce-overhead", fullgraph=True
            )

            # Uncomment to squeeze more perf out of prefill
            # if self.compile_prefill:
            #     self.prefill = torch.compile(self.prefill, fullgraph=True, dynamic=True)

        if self.rank == 0:
            print("Model loaded. Applying warmup...")
        self.warmup()

    def warmup(self):
        # TODO: move this to some precompile
        max_prompt_len = self.max_context_length - self.max_new_tokens
        warm_input = ""
        while len(self.tokenizer.encode(warm_input.strip())) + 1 < max_prompt_len:
            warm_input += "[PAD]"

        print("warmup")
        _ = self.generate_batch(
            [warm_input],
            max_new_tokens=self.max_new_tokens,
            temperature=0,
            device=self.device,
        )

        # warm up with non pad tokens
        res = self.generate_batch(
            ["generated json: "],
            max_new_tokens=self.max_new_tokens,
            temperature=1,
            device=self.device,
        )
        print("warmup_result: ", res)
        print("done")

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 300,
        temperature: float = 1.0,
        device: Optional[str] = "cuda",
        sampling: bool = True,
        grammar_definition: Optional[str] = None,
    ) -> List[Dict]:
        draft_model = None

        if not sampling:
            temperature = 0
        speculate_k = 5
        num_samples = 1
        interactive = False
        callback = lambda x: x
        top_k = 50
        early_stop = self.early_stop

        profile = None
        batched_out = []

        for i, prompt in enumerate(prompts):
            encoded = self.encode_tokens(
                self.tokenizer, prompt, bos=self.add_bos, device=device
            )
            device_sync(device=device)  # MKG

            prompt_length = encoded.size(0)
            if self.rank == 0:
                print("Input prompt length:", prompt_length)
            if (prompt_length + max_new_tokens) > self.max_context_length:
                # We will trim the first tokens...
                trim = prompt_length - self.max_context_length + max_new_tokens
                encoded = encoded[trim:]
                prompt_length = encoded.size(0)

            if self.rank == 0:
                print("Final Prompt Length:", prompt_length)
                print("T:", temperature)

            callback = lambda x: x

            import contextlib

            if (i != num_samples - 1 or not profile) or (
                self.use_tp and self.rank != 0
            ):
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            with prof:
                y, metrics = self.generate(
                    self.model,
                    encoded,
                    max_new_tokens,
                    draft_model=draft_model,
                    speculate_k=speculate_k,
                    interactive=interactive,
                    early_stop=early_stop,
                    callback=callback,
                    temperature=temperature,
                    top_k=top_k,
                    grammar_definition=grammar_definition,
                )
            device_sync(device=device)  # MKG
            tokens_generated = y.size(0) - prompt_length
            text = [self.tokenizer.decode(y[prompt_length:].tolist())]
            batched_out.append(
                {
                    "generation": text[0].strip(),
                    "prompt": prompt,
                    "mean_prob": 0,
                    "num_tokens": tokens_generated,
                    "num_tokens_prompt": encoded.shape[0],
                }
            )
        return batched_out

    @torch.no_grad()
    def generate(
        self,
        model: Transformer,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        interactive: bool,
        draft_model: Transformer,
        speculate_k: Optional[int] = 8,
        early_stop: bool = False,
        callback=lambda x: x,
        grammar_definition: Optional[str] = None,
        **sampling_kwargs,
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """
        if grammar_definition is not None:
            self.grammar_sampler.set_grammar(grammar_definition)
            sampling_kwargs["grammar_sampler"] = self.grammar_sampler

        is_speculative = draft_model is not None
        # create an empty tensor of the expected final shape and fill in the current tokens
        T = prompt.size(0)
        T_new = T + max_new_tokens
        max_seq_length = 350 if interactive else min(T_new, model.config.block_size)

        device, dtype = prompt.device, prompt.dtype
        max_seq_length = (
            max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
        )
        with torch.device(device):
            model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
            if is_speculative and draft_model is not model:
                draft_model.setup_caches(
                    max_batch_size=1, max_seq_length=max_seq_length
                )

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        next_token = self.prefill(
            model, prompt.view(1, -1), input_pos, **sampling_kwargs
        )
        logger.debug(f"rank: {_get_rank()}, after prefill: {next_token}")
        if _get_world_size() > 1:
            torch.distributed.broadcast(next_token, src=0)
        logger.debug(f"rank: {_get_rank()}, after sync: {next_token}")

        if is_speculative:
            self.prefill(draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
        seq[T] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        accept_counts = [0] * (speculate_k + 1)

        if is_speculative:
            if grammar_definition is not None:
                raise NotImplementedError(
                    "Grammar speculative decoding not implemented"
                )
            input_pos = (
                input_pos.item()
            )  # for speculative decoding easier to keep on host
            while input_pos < T_new - 1:
                cur_token = next_token.view(())

                next_tokens = self.speculative_decode(
                    model,
                    draft_model,
                    cur_token,
                    input_pos,
                    speculate_k,
                    **sampling_kwargs,
                )

                accept_counts[len(next_tokens) - 1] += 1
                num_added = min(T_new - input_pos - 1, len(next_tokens))
                seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[:num_added]
                for i in next_tokens[:num_added,]:
                    callback(i)
                input_pos = input_pos + num_added
                next_token = next_tokens[-1]
        else:
            logger.debug(f"rank: {_get_rank()}, before decode: {next_token.item()}")
            generated_tokens, _ = self.decode_n_tokens(
                model,
                next_token.view(1, -1),
                input_pos,
                max_new_tokens - 1,
                early_stop=early_stop,
                callback=callback,
                **sampling_kwargs,
            )

            valid_first_token = 1
            if early_stop and next_token.item() == self.tokenizer.eos_id():
                valid_first_token = 0
            out_size = T + valid_first_token + len(generated_tokens)
            if len(generated_tokens) > 0:
                seq[T + 1 : out_size] = torch.tensor(generated_tokens)
            seq = seq[:out_size]
        generate_stats = {"accept_counts": accept_counts}
        return seq, generate_stats

    def _load_model(self, checkpoint_path, device, precision, use_tp):
        with torch.device("meta"):
            model = Transformer.from_name(checkpoint_path.parent.name)

        if "int8" in str(checkpoint_path):
            print("Using int8 weight-only quantization!")
            from quantize import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        if "int4" in str(checkpoint_path):
            print("Using int4 quantization!")
            path_comps = checkpoint_path.name.split(".")
            assert path_comps[-2].startswith("g")
            groupsize = int(path_comps[-2][1:])
            from quantize import WeightOnlyInt4QuantHandler

            simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
            model = simple_quantizer.convert_for_runtime()

        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)

        if use_tp:
            from rlm_utils.gpt_fast.tp import apply_tp

            print("Applying tensor parallel to model ...")
            apply_tp(model)

        model = model.to(device=device, dtype=precision)
        return model.eval()

    def encode_tokens(self, tokenizer, string, bos=True, device="cuda"):
        tokens = tokenizer.encode(string)
        if bos:
            tokens = [tokenizer.bos_id()] + tokens
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def compute_logits(self, model, x, input_pos):
        return model(x, input_pos)

    def decode_one_token(
        self,
        model: Transformer,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        grammar_sampler: GrammarSampler = None,
        **sampling_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.compute_logits(model, x, input_pos)
        partial_sample = partial(self.sample, **sampling_kwargs)
        logger.debug(f"rank: {_get_rank()}, before sample")
        if grammar_sampler:
            assert logits.shape[:2] == (1, 1)
            return grammar_sampler.constrained_sample_from_logits(
                logits[0], partial_sample
            )
        else:
            return self.sample(logits, **sampling_kwargs)
        # # input_pos: [B, 1]
        # assert input_pos.shape[-1] == 1
        # logits = model(x, input_pos)
        # return self.sample(logits, **sampling_kwargs)
        # 17818
        # 17856
        # 17867 with sampling from grammar

    def sample(self, logits, temperature: float = 1.0, top_k: Optional[int] = None):
        probs = self.logits_to_probs(logits[0, -1], temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs, temperature)
        return idx_next, probs

    def speculative_decode(
        self,
        model: Transformer,
        draft_model: Transformer,
        cur_token: torch.Tensor,
        input_pos: int,
        speculate_k: int,
        **sampling_kwargs,
    ) -> torch.Tensor:
        temperature = sampling_kwargs.get("temperature", 1.0)
        # draft model inference sequentially
        device = cur_token.device
        orig_input_pos = torch.tensor(
            [input_pos], dtype=torch.int64, device=cur_token.device
        )
        draft_tokens, draft_probs = self.decode_n_tokens(
            draft_model,
            cur_token.view(1, -1),
            orig_input_pos.clone(),
            speculate_k,
            **sampling_kwargs,
        )

        draft_tokens = torch.cat(draft_tokens)
        # parallel inference on target model using draft tokens
        target_logits = self.model_forward(
            model,
            torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
            torch.arange(
                input_pos, input_pos + speculate_k + 1, device=cur_token.device
            ),
        )
        target_probs = self.logits_to_probs(target_logits[0], **sampling_kwargs)
        draft_probs = torch.stack(draft_probs)
        # q: target prob, p: draft prob
        # q >= p: always accept draft token
        # q < p: q/p prob to accept draft token
        p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
        rejected_locations = (
            torch.rand_like(accept_draft_prob) > accept_draft_prob
        ).nonzero()

        if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
            accept_length = speculate_k + 1
            last_token = self.multinomial_sample_one_no_sync(
                target_probs[-1], temperature
            )
            # fill last token into draft model
            self.model_forward(
                draft_model,
                draft_tokens[-1].view(1, -1),
                orig_input_pos + speculate_k,
            )
            return torch.cat([draft_tokens, last_token])
        else:
            accept_length = rejected_locations[0].item()
            p = draft_probs[accept_length]
            q = target_probs[accept_length]
            new = q - p
            new = torch.where(new > 0, new, 0.0)
            new = new / new.sum()
            next_token = self.multinomial_sample_one_no_sync(new, temperature)
            return torch.cat([draft_tokens[:accept_length], next_token])

    def logits_to_probs(
        self, logits, temperature: float = 1.0, top_k: Optional[int] = None
    ):
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def model_forward(self, model, x, input_pos):
        return model(x, input_pos)

    def decode_n_tokens(
        self,
        model: Transformer,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        early_stop: bool = False,
        callback=lambda _: _,
        **sampling_kwargs,
    ):
        # pass grammar state down if grammar is defined
        new_tokens, new_probs = [], []
        if early_stop and cur_token.item() == self.tokenizer.eos_id():
            return new_tokens, new_probs
        for step in range(num_new_tokens):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):  # Actually better for Inductor to codegen attention here
                logger.debug(
                    f"rank: {_get_rank()}, decode_step: {step}, before decode: {cur_token}"
                )
                next_token, next_prob = self.decode_one_token(
                    model, cur_token, input_pos, **sampling_kwargs
                )
                logger.debug(
                    f"rank: {_get_rank()}, decode_step: {step}, after decode: {next_token}"
                )
                if _get_world_size() > 1:
                    torch.distributed.broadcast(next_token, src=0)
                    logger.debug(
                        f"rank: {_get_rank()}, decode_step: {step}, broadcast next token"
                    )
                    # somehow second broadcast was causing a deadlock, luckily it's not needed for correctness
                    # torch.distributed.broadcast(next_prob, src=0)
                logger.debug(
                    f"rank: {_get_rank()}, decode_step: {step}, after decode sync: {next_token}"
                )
                input_pos += 1
                if early_stop and next_token.item() == self.tokenizer.eos_id():
                    break
                new_tokens.append(next_token.item())
                callback(new_tokens[-1])
                new_probs.append(next_prob.tolist())
                cur_token = next_token.view(1, -1)

        return new_tokens, new_probs

    def prefill(
        self,
        model: Transformer,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        grammar_sampler: GrammarSampler = None,
        **sampling_kwargs,
    ) -> torch.Tensor:
        # input_pos: [B, S]
        logits = model(x, input_pos)
        partial_sample = partial(self.sample, **sampling_kwargs)
        if grammar_sampler:
            result = grammar_sampler.constrained_sample_from_logits(
                logits[0, -1:], partial_sample
            )
            return result[0]
        else:
            return self.sample(logits, **sampling_kwargs)[0]

    def multinomial_sample_one_no_sync(
        self, probs_sort, temperature
    ):  # Does multinomial sampling without a cuda synchronization
        if temperature > 0.0:
            q = torch.empty_like(probs_sort).exponential_(1)
        else:
            q = torch.ones_like(probs_sort)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)
