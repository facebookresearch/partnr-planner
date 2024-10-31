#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
from typing import Any, Dict

import spacy
from spacy.language import Language


def resolve_coreferences(text: str, nlp: Language) -> str:
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    return doc._.resolved_text


def resolve_coreferences_for_dataset(
    dataset: Dict[str, Any], device: str = "cpu"
) -> Dict[str, str]:
    """Returns a mapping of episode ID -> resolved instruction"""
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
        "fastcoref",
        config={
            "model_architecture": "LingMessCoref",
            "model_path": "biu-nlp/lingmess-coref",
            "device": device,
        },
    )
    resolved_instructions = {}
    for ep in dataset["episodes"]:
        resolved_instructions[ep["episode_id"]] = resolve_coreferences(
            ep["instruction"], nlp
        )
    return resolved_instructions


def main():
    """
    Resolves all coreferences in a provided dataset:
        "Move x to y, and place it on z" -> "Move x to y, and place x on the z"

    The result is saved as a mapping from episode ID to the coref-resolved task
    instruction. This uses LingMess through spacy fastcoref.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save-to", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    args, _ = parser.parse_known_args()

    with gzip.open(args.dataset, "rt") as f:
        dataset = json.load(f)
        resolved_instructions = resolve_coreferences_for_dataset(dataset, args.device)
    with open(args.save_to, "w") as f:
        json.dump(resolved_instructions, f)


if __name__ == "__main__":
    main()
