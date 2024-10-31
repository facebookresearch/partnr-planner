#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os

import hydra
import omegaconf

from dataset_generation.benchmark_generation.evaluation_generation.utils import (
    extract_template_task_number,
    set_config_scene_ids,
)


@hydra.main(
    version_base=None,
    config_path="../../../habitat_llm/conf/benchmark_gen",
    config_name="evaluation_gen.yaml",
)
def main(config: omegaconf.DictConfig) -> None:
    """
    Manually verify that the link to template episodes is correct (tasks are similar
    between templates and LLM-scale episodes). Displays the task instructions from the
    dataset and the task instruction of the linked template episode.

    Invoke with the following config overrides:

    python dataset_generation/benchmark_generation/evaluation_generation/check_templates_match.py \
        eval_gen.template_dataset=... \
        eval_gen.scene_index=... \
        eval_gen.scene_ids=... \
        eval_gen.path_to_dataset_in=...
    """
    config = set_config_scene_ids(config)

    with gzip.open(config.eval_gen.template_dataset, "rt") as f:
        template_episodes = json.load(f)["episodes"]

    ttn_to_instruction = {
        i: ep["instruction"] for i, ep in enumerate(template_episodes)
    }

    sidx = max(config.eval_gen.scene_index, 0)
    fname = os.path.join(
        config.eval_gen.path_to_dataset_in,
        config.eval_gen.scene_ids[sidx],
        "dataset.json.gz",
    )
    with gzip.open(fname, "rt") as f:
        dataset = json.load(f)

    displayed = 0
    for ep in dataset["episodes"]:
        eid = ep["info"]["extra_info"]["episode_id"]
        instruction = ep["info"]["extra_info"]["instruction"]
        ttn = extract_template_task_number(ep)
        sid = dataset["episodes"][0]["scene_id"]
        print(f"--- EID: {eid}  SID: {sid}  TTN: {ttn} ---")
        print(f"template instruction: {ttn_to_instruction[ttn]}")
        print(f" LLM new instruction: {instruction}\n")
        displayed += 1
        if displayed == 10:
            break


if __name__ == "__main__":
    main()
