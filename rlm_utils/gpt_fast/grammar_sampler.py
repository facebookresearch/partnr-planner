#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import logging

from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint

from rlm_utils.gpt_fast.tp import _get_rank

logger = logging.getLogger(__name__)


class GrammarSampler:
    def __init__(self, grammar_string, tokenizer):
        self.tokenizer = tokenizer
        self.grammar = IncrementalGrammarConstraint(grammar_string, "root", tokenizer)
        self.grammar_processor = GrammarConstrainedLogitsProcessor(self.grammar)
        self.sampled_tokens = [[]]

    def reset(self):
        # The grammar procesor must reset its internal state before each generation
        self.grammar.last_size = None
        self.grammar_processor.batch_parsing_states = None
        self.grammar_processor.last_size = None
        self.sampled_tokens = [[]]

    def set_grammar(self, grammar_string):
        # del self.grammar
        # del self.grammar_processor
        # make new instances but reuse the trie from the old grammar to avoid recomputing
        self.grammar = IncrementalGrammarConstraint(
            grammar_string,
            "root",
            self.grammar.tokenizer,
            trie=self.grammar.byte_trie,
            homomorphism=self.grammar.homomorphism,
        )
        # self.grammar = IncrementalGrammarConstraint(
        #     grammar_string,
        #     "root",
        #     self.tokenizer,
        # )
        self.grammar_processor = GrammarConstrainedLogitsProcessor(self.grammar)
        self.reset()

    def constrained_sample_from_logits(self, logits, sampling_function):
        logger.debug(f"rank: {_get_rank()}, before process logits")
        masked_logits = self.grammar_processor.process_logits(
            self.sampled_tokens, logits, "cpu"
        )
        logger.debug(f"rank: {_get_rank()}, before process sampling")
        tokens, probs = sampling_function(masked_logits.unsqueeze(0))
        # only support batch size 1 for now
        assert tokens.shape == (1,)
        # the grammar processor needs to keep track of the sampled tokens
        self.sampled_tokens[0].append(tokens.item())
        # print('sampled_tokens', self.sampled_tokens)
        return tokens, probs
