#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import json
import os
from typing import Set

from habitat_llm.agent.env.dataset import CollaborationDatasetV0


def get_active_within_furniture(rec_filter_dir: str, scene_id: str) -> Set[str]:
    """
    Return a set of furniture handles that have at least one associated "within" receptacle.
    """
    with open(os.path.join(rec_filter_dir, f"{scene_id}.rec_filter.json")) as f:
        recep_filter_data = json.load(f)
    active_furn = {recep.split("|")[0] for recep in recep_filter_data["active"]}
    furn_with_within_recep = {
        recep.split("|")[0] for recep in recep_filter_data["within_set"]
    }
    return active_furn & furn_with_within_recep


def verify_and_correct_within_set_propositions(
    dataset: CollaborationDatasetV0, rec_filter_dir: str
) -> CollaborationDatasetV0:
    """
    For each "is_inside" proposition, check that there exists at least one receptacle
    classified as a "within" receptacle. If not, change the proposition type to
    "is_on_top" to make the episode possible.
    """
    sids = {ep.scene_id for ep in dataset.episodes}
    sid_to_active_within_furniture = {}
    for sid in sids:
        sid_to_active_within_furniture[sid] = get_active_within_furniture(
            rec_filter_dir, sid
        )

    count, total = 0, 0
    for ep in dataset.episodes:
        for prop in ep.evaluation_propositions:
            total += 1
            if prop.function_name == "is_inside":
                count += 1
    print("--- Before Rec Filter Check --- ")
    print(f"within propositions: {count}")
    print(f" total propositions: {total}")
    print(f"         percentage: {100*count/(total+1e-5):.2f}%")
    print()

    changed = 0
    for ep in dataset.episodes:
        sid_to_active_within_furniture[ep.scene_id]
        for prop in ep.evaluation_propositions:
            if prop.function_name == "is_inside" and not any(
                h in sid_to_active_within_furniture[ep.scene_id]
                for h in prop.args["receptacle_handles"]
            ):
                prop.function_name = "is_on_top"
                changed += 1

    count, total = 0, 0
    for ep in dataset.episodes:
        for prop in ep.evaluation_propositions:
            total += 1
            if prop.function_name == "is_inside":
                count += 1
    print("--- After Rec Filter Check --- ")
    print(f"propositions changed: {changed}")
    print(f" within propositions: {count}")
    print(f"  total propositions: {total}")
    print(f"          percentage: {100*count/(total+1e-5):.2f}%")
    return dataset


def main():
    """
    Check that all furniture referenced by `is_inside` propositions have a `within`
    receptacle. If not, change the proposition to `is_on_top`.

    To run:
        >>> python within_set_verification.py \
            --dataset-path [path-to-dataset] \
            --save-to [path-to-dataset-out]
    """
    desc = (
        "Check that all furniture referenced by `is_inside` propositions have a `within`"
        "receptacle. If not, change the proposition to `is_on_top`."
    )
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--dataset-path",
        default="data/datasets/partnr_episodes/v0_0/val.json.gz",
        type=str,
        help="Path to the collaboration dataset",
    )
    parser.add_argument(
        "--save-to",
        default="val_verified.json.gz",
        type=str,
        help="Path to where the resulting dataset should be saved",
    )
    parser.add_argument(
        "--rec-filter-dir",
        required=False,
        default="data/hssd-hab/scene_filter_files",
    )
    args = parser.parse_args()
    with gzip.open(args.dataset_path, "rt") as f:
        dataset_json = json.load(f)

    dataset = CollaborationDatasetV0()
    dataset.from_json(json.dumps(dataset_json))

    dataset = verify_and_correct_within_set_propositions(dataset, args.rec_filter_dir)
    with gzip.open(args.save_to, "wt") as f:
        f.write(dataset.to_json())


if __name__ == "__main__":
    main()
