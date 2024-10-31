#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import Any, Dict

import spacy
from spacy.language import Language


def resolve_coreferences(text: str, nlp: Language) -> str:
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    return doc._.resolved_text


def eid_from_episode(episode) -> int:
    """Extracts the episode ID integer from an episode object."""
    eid = episode["info"]["extra_info"]["episode_id"]
    if isinstance(eid, int):
        return eid
    if eid.isdigit():
        return int(eid)
    return int(eid.split("|")[-1].split(".")[0])


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
    resolved_instructions: Dict[str, str] = {}
    for ep in dataset["episodes"]:
        eid = str(eid_from_episode(ep))
        resolved_instructions[eid] = resolve_coreferences(
            ep["info"]["extra_info"]["instruction"], nlp
        )
    return resolved_instructions


def main():
    """
    The eval gen source directory must contain:
        [scene-id]/dataset.json.gz
        [scene-id]/scene_info.json

    After this script:
        [scene-id]/dataset.json.gz
        [scene-id]/scene_info.json
        [scene-id]/resolved_coref.json

    For each scene ID, this script produces [scene-id]/resolved_coref.json, which maps
    episode ID to a modified instruction in which the instruction's coreferences have
    been resolved using LingMess (https://arxiv.org/abs/2205.12644).

    to run:
    >>> python dataset_generation/benchmark_generation/evaluation_generation/coreference_resolution/resolve_src_dir.py \
        --src [path to source directory] --device [cuda/cpu]
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        required=False,
        default="data/datasets/dev/2024_07_25_train_os_scaled/src",
    )
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    args, _ = parser.parse_known_args()

    sids = sorted(os.listdir(args.src), key=lambda s: int(s.split("_")[0]))
    for sid in sids:
        in_f = os.path.join(args.src, sid, "dataset.json.gz")
        out_f = os.path.join(args.src, sid, "resolved_coref.json")
        if os.path.exists(out_f):
            continue

        with gzip.open(in_f, "rt") as f:
            dataset = json.load(f)
        resolved_instructions = resolve_coreferences_for_dataset(dataset, args.device)
        with open(out_f, "w") as f:
            json.dump(resolved_instructions, f)


if __name__ == "__main__":
    main()
