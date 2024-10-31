#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import json
import os
from collections import defaultdict
from itertools import chain, combinations
from typing import Any, Dict

import matplotlib
from matplotlib import pyplot as plt


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def ep_has_prop_from_set(episode, prop_set):
    return any(
        prop["function_name"] in prop_set for prop in episode["evaluation_propositions"]
    )


def ep_is_rearrange(episode: Dict[str, Any]) -> bool:
    return ep_has_prop_from_set(
        episode, {"is_on_top", "is_in_room", "is_inside", "is_on_floor"}
    )


def ep_is_spatial(episode: Dict[str, Any]) -> bool:
    return ep_has_prop_from_set(episode, {"is_next_to"})


def ep_is_heterogeneous(episode: Dict[str, Any]) -> bool:
    return ep_has_prop_from_set(
        episode,
        {
            "is_clean",
            "is_dirty",
            "is_powered_on",
            "is_powered_off",
            "is_filled",
            "is_empty",
        },
    )


def ep_is_temporal(episode: Dict[str, Any]) -> bool:
    """A temporal episode is one in which the temporal constraint has at least one DAG edge."""
    for c in episode["evaluation_constraints"]:
        if c["type"] == "TemporalConstraint":
            return bool(len(c["args"]["dag_edges"]))
    raise AssertionError("no temporal constraint")


def generate_upset_figs(dataset_f: str, keep_empty: bool, save_dir: str) -> None:
    """Produces the top figure and side figure of the upset diagram and saves them separately."""
    with gzip.open(dataset_f, "rt") as f:
        episodes = json.load(f)["episodes"]

    os.makedirs(save_dir, exist_ok=True)

    upset_dict = load_upset_data(episodes)

    generate_upset_side(upset_dict, save_dir)

    if not keep_empty:
        # remove combinations with low/no percentage
        upset_dict = {k: v for k, v in upset_dict.items() if v > 0.01}

    generate_upset_top(upset_dict, save_dir)

    print("UpSet data (L to R):")
    for k, v in upset_dict.items():
        print("\t", k, v)


def load_upset_data(episodes):
    upset_data = defaultdict(int)
    for ep in episodes:
        ks = []
        for b, k_type in [
            (ep_is_rearrange(ep), "R"),
            (ep_is_spatial(ep), "S"),
            (ep_is_temporal(ep), "T"),
            (ep_is_heterogeneous(ep), "O"),
        ]:
            if b:
                ks.append(k_type)
        upset_data[tuple(ks)] += 1

    return {k: v / len(episodes) for k, v in upset_data.items()}


def generate_upset_top(upset_dict, save_dir):
    matplotlib.rc("font", **{"size": 14})

    fig, ax = plt.subplots()

    fig.set_size_inches(6, 2)
    ax.grid(visible=True, zorder=-1000)
    ax.set_aspect("auto")

    xs = []
    ys = []
    key_order = [
        ("R",),
        ("S",),
        ("T",),
        ("O",),
        ("R", "S"),
        ("R", "T"),
        ("R", "O"),
        ("R", "S", "T"),
        ("R", "S", "O"),
        ("R", "T", "O"),
        ("R", "S", "T", "O"),
    ]
    for x in key_order:
        if x not in upset_dict:
            continue
        xs.append(str(x))
        ys.append(upset_dict[x] * 100)

    ax.bar(xs, ys, width=0.5, zorder=10, color="tab:gray")
    ax.get_xaxis().set_visible(False)
    ax.set_yticks(
        [i * 5 for i in range(9)], labels=[i * 5 for i in range(9)], fontsize=12
    )
    ax.set_ylim([0, 30])
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel("Episodes (%)")

    save_to = os.path.join(save_dir, "upset_top.pdf")
    plt.savefig(
        save_to,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    print(f"saved to `{save_to}`")
    save_to = os.path.join(save_dir, "upset_top.png")
    plt.savefig(
        save_to,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    print(f"saved to `{save_to}`")
    plt.clf()
    plt.close()


def generate_upset_side(upset_dict, save_dir):
    matplotlib.rc("font", **{"size": 14})

    ttd = {"R": 0, "S": 0, "T": 0, "O": 0}
    for k, v in upset_dict.items():
        for k2 in ttd:
            if k2 in k:
                ttd[k2] += v * 100

    ys = [ttd["O"], ttd["T"], ttd["S"], ttd["R"]]
    xs = ["O", "T", "S", "R"]

    fig, ax = plt.subplots()

    fig.set_size_inches(5, 2.5)
    ax.grid(visible=True, zorder=-1000, axis="x")
    ax.set_aspect("auto")
    color = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    ax.barh(xs, ys, height=0.5, zorder=10, color=color)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks(
        [i * 20 for i in range(6)], labels=[i * 20 for i in range(6)], fontsize=14
    )
    ax.set_xlim([0, 110])
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.set_ylabel("Episodes With (%)", fontsize=14)
    ax.invert_xaxis()

    save_to = os.path.join(save_dir, "upset_side.pdf")
    plt.savefig(
        save_to,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    print(f"saved to `{save_to}`")
    save_to = os.path.join(save_dir, "upset_side.png")
    plt.savefig(
        save_to,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    print(f"saved to `{save_to}`")
    plt.clf()
    plt.close()


def main():
    """
    https://en.wikipedia.org/wiki/UpSet_plot
    Generates pdf images of the side and top plots of an UpSet plot.
    The dot combinations can be curated in PowerPoint.

    Alt option: https://upsetplot.readthedocs.io/en/stable/
    We used matplotlib + powerpoint to have more control over appearance.

    python dataset_generation/benchmark_generation/analysis/task_type_upset_plot.py
    """
    parser = argparse.ArgumentParser(
        description="Generates an UpSet figure for a given dataset."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/datasets/partnr_episodes/v0_0/val.json.gz",
        type=str,
        help="Path to the collaboration dataset",
    )
    parser.add_argument(
        "--save-dir",
        default="data/dataset_analysis",
        type=str,
        help="Path to where the upset plot images should be saved",
    )
    parser.add_argument(
        "--keep-empty",
        action=argparse.BooleanOptionalAction,
        help="if keep-empty, display all combinations, even those with zero mass",
    )

    args = parser.parse_args()
    generate_upset_figs(args.dataset_path, args.keep_empty, args.save_dir)


if __name__ == "__main__":
    main()
