#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set

import omegaconf

from habitat_llm.agent.env.evaluation.evaluation_functions import EvaluationProposition


def scene_dirs_from_dir(d: str, full_path: bool = True) -> List[str]:
    """
    Returns a sorted list of scene IDs in a directory. Scene IDs are identified as
    being directories with an integer first character.
    """
    scene_ids = [
        sid
        for sid in os.listdir(d)
        if os.path.isdir(os.path.join(d, sid)) and sid[0].isdigit()
    ]
    scene_ids = sorted(scene_ids, key=lambda sid: tuple(int(s) for s in sid.split("_")))
    if not full_path:
        return scene_ids
    return [os.path.join(d, sid) for sid in scene_ids]


def set_config_scene_ids(config: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """If scene_ids is empty, then search the path_to_dataset_in for scenes."""
    if len(config.eval_gen.scene_ids):
        return config
    config.eval_gen.scene_ids = scene_dirs_from_dir(
        config.eval_gen.path_to_dataset_in, full_path=False
    )
    return config


def display_packing_stats(source_path: str, n_eps: int) -> None:
    """
    Summarizes the stats of packing failure modes as contained in each scene directory.
    n_eps: the nubmer of episodes that were successfully packed.
    """

    def perc(n: int, m: int) -> str:
        return f"{round(100 * n / m, 2)}"

    failure_summary: DefaultDict[str, int] = defaultdict(int)
    for scene_dir in scene_dirs_from_dir(source_path):
        scene_packing_failures = os.path.join(scene_dir, "packing_failures.json")
        if not os.path.exists(scene_packing_failures):
            continue

        with open(scene_packing_failures, "r") as f:
            packing_failures = json.load(f)
            if "failure_modes" in packing_failures:
                packing_failures = packing_failures["failure_modes"]

            for k, v in packing_failures.items():
                failure_summary[k] += len(v)

    tot_failures = sum(failure_summary.values())
    tot_eps = n_eps + tot_failures

    with open(os.path.join(source_path, "packing_summary.json"), "w") as f:
        summary = {
            "episodes_before": tot_eps,
            "episodes_after": n_eps,
            "episodes_failed": tot_failures,
            "packed_percentage": float(perc(n_eps, tot_eps)),
            "failure_modes": failure_summary,
        }
        json.dump(summary, f, indent=2)

    print()
    print(" ------ Packing failures summary ------")
    print(f"Episodes after/before: {n_eps}/{tot_eps} ({perc(n_eps, tot_eps)}%)")
    print("failure modes:")
    failures = [(k, v, perc(v, tot_failures)) for k, v in failure_summary.items()]
    for k, v, p in sorted(failures, key=lambda x: x[2], reverse=True):
        print(k, v, f"({p}%)")
    print()


def extract_template_task_number(episode) -> int:
    """
    The template task number is an ID linking an episode to the index of the
    template episode used as basis during LLM generation. It has been stored in
    a couple inconsistent places, thus the additional logic here to extract it.
    """

    def int_convert(ttn):
        if isinstance(ttn, list):
            return int(ttn[-1])
        return int(ttn)

    extra_info = episode["info"]["extra_info"]
    if "template_task_number" in extra_info:
        return int_convert(extra_info["template_task_number"])
    if "template_task_number" in episode["info"]["extra_info"]["initial_state"][-1]:
        return int_convert(extra_info["initial_state"][-1]["template_task_number"])
    raise ValueError("Cannot find `template_task_number` in episode object.")


def get_scene_to_within_receps(filter_file_dir: str) -> Dict[str, Set[str]]:
    """
    Returns a mapping from the scene ID to the set of receptacle handles in that scene
    that are labeled as "within" according to the HSSD receptacle filter file.
    """
    scene_to_within_receps = {}
    for fname in os.listdir(filter_file_dir):
        if ".rec_filter.json" not in fname:
            continue
        sid = fname.split(".")[0]
        with open(os.path.join(filter_file_dir, fname)) as f:
            scene_to_within_receps[sid] = set(json.load(f)["within_set"])
    return scene_to_within_receps


def object_initializations_from_name_to_recep(
    name_to_recep: Dict[str, str],
    scene_id: str,
    scene_to_within_receps: Dict[str, Set[str]],
) -> Dict[str, str]:
    """
    Create a mapping of all object handles (keys in name_to_recep) to their associated
    initialization type (one of `within`, `ontop`, `floor`).
    """
    init_info = {}
    for k, v in name_to_recep.items():
        if v == "floor":
            init_info[k] = "floor"
        elif v in scene_to_within_receps[scene_id]:
            init_info[k] = "within"
        else:
            init_info[k] = "ontop"
    return init_info


def self_next_to_self_in_proposition(proposition: EvaluationProposition) -> bool:
    """
    next_to(x, x) is an invalid proposition construction; an object cannot be next to
    itself. Returns True if this occurs in any spatial proposition.
    """

    if proposition.function_name == "is_next_to":
        a = set(proposition.args["entity_handles_a"])
        b = set(proposition.args["entity_handles_b"])
        return len(a & b) > 0

    if proposition.function_name == "is_clustered":
        argument_lists = [set(arg_lst) for arg_lst in proposition.args["*args"]]
        for i in range(len(argument_lists)):
            for j in range(i, len(argument_lists)):  # noqa: SIM110
                if len(argument_lists[i] & argument_lists[j]) > 0:
                    return True
        return False

    return False
