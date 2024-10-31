#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple, TypedDict

import networkx as nx

from matplotlib import pyplot as plt  # isort: skip
import pandas as pd  # isort: skip


REARRANGE_PREDICATES = {"is_on_top", "is_inside", "is_in_room", "is_on_floor"}
HETEROGENEOUS_PREDICATES = {
    "is_clean",
    "is_dirty",
    "is_filled",
    "is_empty",
    "is_powered_on",
    "is_powered_off",
}


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    if ".pickle" in dataset_path:
        raise ValueError("only .json.gz supported.")
    with gzip.open(dataset_path, "rt") as f:
        dataset = json.load(f)
    return dataset


def plot_bar(
    data: Dict[Any, int],
    fname: str,
    ylabel: str,
    title: str,
    fig_size: Tuple[int, int] = (8, 4),
    sort_by_key: bool = False,
    rotate: bool = True,
    show_mean: bool = False,
) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size[0], fig_size[1])

    i = 0 if sort_by_key else 1
    sorted_data = sorted(data.items(), key=lambda x: x[i], reverse=True)

    labels = [x[0] for x in sorted_data]
    counts = [x[1] for x in sorted_data]
    ax.set_aspect("auto")
    ax.bar(labels, counts)

    if show_mean:
        tot = 0
        for l, c in zip(labels, counts):
            tot += l * c
        m = tot / sum(counts)
        ax.text(
            m + 0.25,
            int(0.8 * max(counts)),
            f"Mean: {round(m, 2)}",
            ha="left",
            va="center",
            fontsize=10,
        )
        ax.axvline(x=m, color="black", linestyle="dashed")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for label in ax.get_xticklabels():
        if rotate:
            label.set_rotation(45)
        label.set_ha("right")
    plt.savefig(fname, dpi=300, transparent=False, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_pie(
    sizes: List[int],
    labels: List[str],
    fname: str,
    title: str,
    fig_size: Tuple[int, int] = (8, 4),
    float_digits: int = 0,
) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size[0], fig_size[1])

    ax.pie(sizes, labels=labels, autopct=f"%1.{float_digits}f%%")
    ax.set_title(title)
    plt.savefig(fname, dpi=300, transparent=False, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_ambiguity(
    yes_no_distribution: Dict[str, int],
    soln_space_distribution: List[int],
    y_label: str,
    fname: str,
    title: str,
    fig_size: Tuple[int, int] = (8, 4),
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, width_ratios=[0.3, 0.7], constrained_layout=True
    )
    fig.set_size_inches(fig_size[0], fig_size[1])
    fig.suptitle(title)

    ax1.pie(
        list(yes_no_distribution.values()),
        labels=list(yes_no_distribution.keys()),
        autopct="%1.0f%%",
    )

    labels = list(range(2, max(soln_space_distribution) + 1))
    counts = {l: 0 for l in labels}
    for i in soln_space_distribution:
        counts[i] += 1
    sorted_counts = [x[1] for x in sorted(counts.items(), key=lambda x: x[0])]
    ax2.set_aspect("auto")
    ax2.bar([str(l) for l in labels], sorted_counts)
    ax2.set_ylabel(y_label)
    ax2.set_xlabel("# of allowed entities")

    plt.savefig(fname, dpi=300, transparent=False, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_temporal_groups(
    yes_no_distribution: Dict[str, int],
    soln_space_distribution: List[int],
    y_label: str,
    fname: str,
    title: str,
    fig_size: Tuple[int, int] = (8, 4),
):
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, width_ratios=[0.3, 0.7], constrained_layout=True
    )
    fig.set_size_inches(fig_size[0], fig_size[1])
    fig.suptitle(title)

    ax1.pie(
        list(yes_no_distribution.values()),
        labels=list(yes_no_distribution.keys()),
        autopct="%1.0f%%",
    )
    labels = list(range(1, max(soln_space_distribution) + 1))
    counts = {l: 0 for l in labels}
    for i in soln_space_distribution:
        counts[i] += 1
    sorted_counts = [x[1] for x in sorted(counts.items(), key=lambda x: x[0])]
    ax2.set_aspect("auto")
    ax2.bar([str(l) for l in labels], sorted_counts)
    ax2.set_ylabel(y_label)
    ax2.set_xlabel("# Rearrange Sub-Tasks")

    plt.savefig(fname, dpi=300, transparent=False, bbox_inches="tight")
    plt.clf()
    plt.close()


def is_single_agent_episode(episode) -> bool:
    """
    Here, a single-agent episode is defined as an episode in which there is no possible
    sub-task parallelism, i.e., there exist no two rearrangements that are allowed to
    be performed in any order.
    """

    def get_temporal_groups(episode):
        for c in episode["evaluation_constraints"]:
            if c["type"] == "TemporalConstraint":
                if len(c["args"]["dag_edges"]):
                    return list(
                        nx.topological_generations(nx.DiGraph(c["args"]["dag_edges"]))
                    )
                return [list(range(len(episode["evaluation_propositions"])))]
        raise AssertionError("Temporal Constraint not found.")

    def has_multiple_rearranges(props):
        prop_names = [prop["function_name"] for prop in props]

        # room and floor count as one
        n_room_props = sum(n == "is_in_room" for n in prop_names)
        n_floor_props = sum(n == "is_on_floor" for n in prop_names)
        n_to_ignore = min(n_floor_props, n_room_props)

        n_rearrange_props = len([n in REARRANGE_PREDICATES for n in prop_names])
        n_props = n_rearrange_props - n_to_ignore
        return n_props > 1

    def has_unrelated_heterogenous_and_rearrange(props):
        prop_names = [prop["function_name"] for prop in props]
        if not any(n in HETEROGENEOUS_PREDICATES for n in prop_names):
            return False

        rearrange_props = [
            p for p in props if p["function_name"] in REARRANGE_PREDICATES
        ]
        heterogeneous_props = [
            p for p in props if p["function_name"] in HETEROGENEOUS_PREDICATES
        ]
        for p1 in rearrange_props:
            h1 = set(p1["args"]["object_handles"])
            for p2 in heterogeneous_props:
                h2 = set(p2["args"]["object_handles"])
                if len(h1 & h2) == 0:
                    return True
        return False

    def group_is_single_agent(episode, temporal_group: List[int]):
        props = [episode["evaluation_propositions"][i] for i in temporal_group]
        if has_multiple_rearranges(props):
            return False
        return not has_unrelated_heterogenous_and_rearrange(props)

    return all(group_is_single_agent(episode, g) for g in get_temporal_groups(episode))


def get_has_temporal(episode: Dict[str, Any]) -> Tuple[bool, int, List[int]]:
    """returns is_temporal, num_temporal_steps, temporal_step_sizes."""

    def get_num_rearranges(temporal_group: List[int]) -> int:
        """next_to only counts as a rearrange if not accompanied by a furniture/room placement."""
        props = [episode["evaluation_propositions"][i] for i in temporal_group]
        if "is_next_to" not in [p["function_name"] for p in props]:
            return len(temporal_group)
        objects_explicitly_placed = set()
        num_rearranges = 0
        for prop in props:
            if prop["function_name"] == "is_next_to":
                continue
            if "object_handles" not in prop["args"]:
                continue
            num_rearranges += 1
            objects_explicitly_placed |= set(prop["args"]["object_handles"])

        for prop in props:
            if prop["function_name"] != "is_next_to":
                continue
            for entity in prop["args"]["entity_handles_a"]:
                if entity not in objects_explicitly_placed:
                    num_rearranges += 1
                    break
        return num_rearranges

    for c in episode["evaluation_constraints"]:
        if c["type"] == "TemporalConstraint":
            if len(c["args"]["dag_edges"]):
                temporal_groups = list(
                    nx.topological_generations(nx.DiGraph(c["args"]["dag_edges"]))
                )
                return (
                    True,
                    len(temporal_groups),
                    [get_num_rearranges(x) for x in temporal_groups],
                )
            return False, 1, [len(episode["evaluation_propositions"])]
    raise AssertionError("no temporal constraint")


def ambiguity_analysis(dataset: Dict[str, Any], save_dir: str) -> None:
    """
    Plots ambiguity w.r.t objects, furniture, and room specifications. Here, ambiguity
    refers to multiple entities being capable of satisfying the instruction.
    """
    resolvable_ambiguity_distribution = {
        "Object": {"With": 0, "Without": 0},
        "Furniture": {"With": 0, "Without": 0},
        "Room": {"With": 0, "Without": 0},
    }
    resolvable_ambiguity_counts_distribution: Dict[str, List[int]] = {
        "Object": [],
        "Furniture": [],
        "Room": [],
    }

    for ep in dataset["episodes"]:
        for prop in ep["evaluation_propositions"]:
            if "object_handles" in prop["args"]:
                n = len(prop["args"]["object_handles"])
                has_amb = n > 1 and prop["args"]["number"] != n
                if has_amb:
                    resolvable_ambiguity_distribution["Object"]["With"] += 1
                    resolvable_ambiguity_counts_distribution["Object"].append(n)
                else:
                    resolvable_ambiguity_distribution["Object"]["Without"] += 1

            if "receptacle_handles" in prop["args"]:
                n = len(prop["args"]["receptacle_handles"])
                if n > 1:
                    resolvable_ambiguity_distribution["Furniture"]["With"] += 1
                    resolvable_ambiguity_counts_distribution["Furniture"].append(n)
                else:
                    resolvable_ambiguity_distribution["Furniture"]["Without"] += 1

            if "room_ids" in prop["args"]:
                n = len(prop["args"]["room_ids"])
                if n > 1:
                    resolvable_ambiguity_distribution["Room"]["With"] += 1
                    resolvable_ambiguity_counts_distribution["Room"].append(n)
                else:
                    resolvable_ambiguity_distribution["Room"]["Without"] += 1

    plot_ambiguity(
        resolvable_ambiguity_distribution["Object"],
        resolvable_ambiguity_counts_distribution["Object"],
        "n propositions",
        f"{save_dir}/ambiguity_object.png",
        "Episodes With Resolvable Object Ambiguity",
    )
    plot_ambiguity(
        resolvable_ambiguity_distribution["Furniture"],
        resolvable_ambiguity_counts_distribution["Furniture"],
        "n propositions",
        f"{save_dir}/ambiguity_furniture.png",
        "Episodes With Resolvable Furniture Ambiguity",
    )
    plot_ambiguity(
        resolvable_ambiguity_distribution["Room"],
        resolvable_ambiguity_counts_distribution["Room"],
        "n propositions",
        f"{save_dir}/ambiguity_room.png",
        "Episodes With Resolvable Room Ambiguity",
    )


def temporal_analysis(dataset: Dict[str, Any], save_dir: str) -> None:
    """
    Generates plots for:
    - multi vs single agent task percentage
    - distribution of the number of rearranges in temporal groups
    Does this for all episodes and just the subset of episodes that contain more than one temporal group.
    """
    temporal_steps: DefaultDict[int, int] = defaultdict(int)
    temporal_step_sizes: List[int] = []
    temporal_step_size_only_one = {"single-agent tasks": 0, "multi-agent tasks": 0}
    temporal_step_sizes_t_only: List[int] = []
    temporal_step_size_only_one_t_only = {
        "single-agent tasks": 0,
        "multi-agent tasks": 0,
    }

    for ep in dataset["episodes"]:
        has_temporal, temp_steps, temp_step_sizes = get_has_temporal(ep)
        if is_single_agent_episode(ep):
            temporal_step_size_only_one["single-agent tasks"] += 1
            if has_temporal:
                temporal_step_size_only_one_t_only["single-agent tasks"] += 1
        else:
            temporal_step_size_only_one["multi-agent tasks"] += 1
            if has_temporal:
                temporal_step_size_only_one_t_only["multi-agent tasks"] += 1

        temporal_steps[temp_steps] += 1
        temporal_step_sizes.extend(temp_step_sizes)
        if has_temporal:
            temporal_step_sizes_t_only.extend(temp_step_sizes)

    plot_bar(
        temporal_steps,
        fname=f"{save_dir}/temporal_steps.png",
        ylabel="n episodes",
        title="Temporal Steps",
        fig_size=(6, 4),
        sort_by_key=True,
        show_mean=True,
    )
    plot_temporal_groups(
        temporal_step_size_only_one,
        temporal_step_sizes,
        "# Temporal Task Groups",
        fname=f"{save_dir}/temporal_group_sizes.png",
        title="Temporal Group Sizes",
    )
    plot_temporal_groups(
        temporal_step_size_only_one_t_only,
        temporal_step_sizes_t_only,
        "# Temporal Task Groups",
        fname=f"{save_dir}/temporal_group_sizes_temporal_only.png",
        title="Temporal Group Sizes (Temporal Tasks Only)",
    )


def dependent_rearrange_analysis(dataset: Dict[str, Any], save_dir: str) -> None:
    same_rearranges = []
    diff_rearranges = []

    for ep in dataset["episodes"]:
        for c in ep["evaluation_constraints"]:
            if c["type"] == "SameArgConstraint":
                same_rearranges.append(1)
                break
        else:
            same_rearranges.append(0)
        for c in ep["evaluation_constraints"]:
            if c["type"] == "DifferentArgConstraint":
                diff_rearranges.append(1)
                break
        else:
            diff_rearranges.append(0)

    dependent_rearrange_sizes = {"None": 0, "Same": 0, "Different": 0}
    for s, d in zip(same_rearranges, diff_rearranges):
        if s:
            dependent_rearrange_sizes["Same"] += 1
        if d:
            dependent_rearrange_sizes["Different"] += 1
        if not s and not d:
            dependent_rearrange_sizes["None"] += 1

    plot_pie(
        list(dependent_rearrange_sizes.values()),
        list(dependent_rearrange_sizes.keys()),
        fname=f"{save_dir}/dependent_rearranges.png",
        title="Tasks With Dependent Rearranges",
        float_digits=1,
    )


def subset_count_analysis(dataset: Dict[str, Any], save_dir: str) -> None:
    """
    subset counts: n objects of a class k must be used to satisfy a proposition
    Example: "bring two plates" when there are >2 in the scene.
    """

    class SubsetCounts(TypedDict):
        has: int
        data: List[Tuple[int, int]]

    subset_counts: SubsetCounts = {"has": 0, "data": []}

    for ep in dataset["episodes"]:
        has_subset_counts = False
        for prop in ep["evaluation_propositions"]:
            if "number" not in prop["args"]:
                continue
            n = int(prop["args"]["number"])
            if n == 1:
                continue
            k = (
                "object_handles"
                if "object_handles" in prop["args"]
                else "entity_handles_a"
            )
            if n == len(prop["args"][k]):
                continue
            has_subset_counts = True
            subset_counts["data"].append((len(prop["args"][k]), n))

        subset_counts["has"] += int(has_subset_counts)

    # plot the subsets as both pie and scatter.

    fig, (ax1, ax2) = plt.subplots(
        ncols=2, width_ratios=[0.3, 0.7], constrained_layout=True
    )
    fig.set_size_inches(8, 4)
    fig.suptitle("Episodes With Subset Counts")

    labels = ["With", "Without"]
    ax1.pie(
        [subset_counts["has"], len(dataset["episodes"]) - subset_counts["has"]],
        labels=labels,
        autopct="%1.1f%%",
    )

    ax2.set_aspect("auto")
    ax2.scatter(
        [x[1] for x in subset_counts["data"]],
        [x[0] for x in subset_counts["data"]],
        zorder=200,
    )
    if len(subset_counts["data"]):
        max_val = max(x[0] for x in subset_counts["data"])
    else:
        max_val = 1
    ax2.set_xticks(list(range(2 + max_val)))
    ax2.set_yticks(list(range(2 + max_val)))
    ax2.grid(visible=True, zorder=100)
    ax2.set_ylabel("Set Size")
    ax2.set_xlabel("Subset Target")
    ax2.set_aspect(1)

    plt.savefig(
        f"{save_dir}/subset_counts.png", dpi=300, transparent=False, bbox_inches="tight"
    )
    plt.clf()
    plt.close()


def multi_step_analysis(dataset: Dict[str, Any], save_dir):
    """
    multi-step: the same object is involved in 2+ rearrangements in the same episode
    """
    multistep = []
    for ep in dataset["episodes"]:
        props = ep["evaluation_propositions"]
        for c in ep["evaluation_constraints"]:
            if c["type"] == "TerminalSatisfactionConstraint":
                multistep.append(
                    int(len(c["args"]["proposition_indices"]) != len(props))
                )
                break

    multistep_sizes = {0: 0, 1: 1}
    for m in multistep:
        multistep_sizes[m] += 1
    multistep_sizes_named = {"With": multistep_sizes[1], "Without": multistep_sizes[0]}
    plot_pie(
        list(multistep_sizes_named.values()),
        [str(k) for k in multistep_sizes_named],
        fname=f"{save_dir}/multistep_rearrange.png",
        title="Multi-step Rearrange Tasks",
        float_digits=1,
    )


def task_type_analysis(dataset: Dict[str, Any], save_dir: str) -> None:
    """
    Task types are rearrange-only (R), spatial (S), temporal (T), or object states (O).
    See task_type_upset_plot.py for a more detailed task type analysis.
    """
    task_type_distribution = {"R": 0, "RT": 0, "RS": 0, "RTS": 0}

    for ep in dataset["episodes"]:
        has_temporal, _, _ = get_has_temporal(ep)
        has_spatial = any(
            prop["function_name"] == "is_next_to"
            for prop in ep["evaluation_propositions"]
        )
        if has_temporal and has_spatial:
            task_type_distribution["RTS"] += 1
        elif has_temporal:
            task_type_distribution["RT"] += 1
        elif has_spatial:
            task_type_distribution["RS"] += 1
        else:
            task_type_distribution["R"] += 1

    task_type_distribution = {
        k: int(100 * v / len(dataset["episodes"]))
        for k, v in task_type_distribution.items()
    }
    task_type_distribution = {
        "Rearrange-Only": task_type_distribution["R"],
        "Temporal": task_type_distribution["RT"],
        "Spatial": task_type_distribution["RS"],
        "Temporal+Spatial": task_type_distribution["RTS"],
    }
    plot_pie(
        list(task_type_distribution.values()),
        list(task_type_distribution.keys()),
        fname=f"{save_dir}/task_types.png",
        title="Task Type Distribution",
    )


def object_furniture_room_analysis(
    dataset: Dict[str, Any], metadata: Dict[str, str], save_dir: str
) -> None:
    """
    Distribution of objects, rooms, and furniture referenced by the evaluation functions.
    """
    obj_cat_distribution: Dict[str, int] = defaultdict(int)
    recep_cat_distribution: Dict[str, int] = {
        k: 0 for k in metadata["receptacle_classes"]
    }
    room_cat_distribution: Dict[str, int] = defaultdict(int)

    for ep in dataset["episodes"]:
        for prop in ep["evaluation_propositions"]:
            if "object_handles" in prop["args"]:
                for obj_handle in prop["args"]["object_handles"]:
                    obj_handle = obj_handle.split(":")[0].rstrip("_")
                    obj_cat = metadata["hash_to_cat"][obj_handle]
                    obj_cat_distribution[obj_cat] += 1
            if "receptacle_handles" in prop["args"]:
                for recep_handle in prop["args"]["receptacle_handles"]:
                    recep_handle = recep_handle.split(":")[0].rstrip("_")
                    hash_to_cat = metadata["hash_to_cat"]
                    try:
                        recep_cat = hash_to_cat[recep_handle]
                    except KeyError:
                        continue
                    if recep_cat in recep_cat_distribution:
                        recep_cat_distribution[recep_cat] += 1
            if "room_ids" in prop["args"]:
                for room_id in prop["args"]["room_ids"]:
                    room_cat = room_id.split(".")[0]
                    room_cat_distribution[room_cat] += 1

    plot_bar(
        room_cat_distribution,
        fname=f"{save_dir}/room_distribution.png",
        ylabel="n rooms",
        title="Distribution of Rooms",
        fig_size=(8, 4),
    )
    plot_bar(
        recep_cat_distribution,
        fname=f"{save_dir}/furniture_distribution.png",
        ylabel="n furniture",
        title="Distribution of Furniture",
        fig_size=(8, 4),
    )
    plot_bar(
        obj_cat_distribution,
        fname=f"{save_dir}/object_distribution.png",
        ylabel="n objects",
        title="Distribution of Objects",
        fig_size=(36, 4),
    )


def predicate_analysis(dataset: Dict[str, Any], save_dir: str) -> None:
    """
    Plots the distribution of the number of propositions in an episode as well as
    the distribution of the predicate functions (is_inside, is_in_room, etc).
    """
    n_props_distribution: DefaultDict[int, int] = defaultdict(int)
    predicate_distribution: DefaultDict[str, int] = defaultdict(int)
    total_props = 0
    rearranges_distribution: DefaultDict[int, int] = defaultdict(int)

    for ep in dataset["episodes"]:
        _, _, gs = get_has_temporal(ep)
        for t_group_size in gs:
            rearranges_distribution[t_group_size] += 1

        props = ep["evaluation_propositions"]
        n_props_distribution[len(props)] += 1

        for prop in props:
            total_props += 1
            predicate_distribution[prop["function_name"]] += 1

    total_predicates = sum(predicate_distribution.values())
    predicate_dist_percents = {
        k: int(100 * v / sum(predicate_distribution.values()))
        for k, v in predicate_distribution.items()
    }
    plot_bar(
        rearranges_distribution,
        fname=f"{save_dir}/n_rearranges_distribution.png",
        ylabel="n episodes",
        title="Rearranges Per Episode",
        fig_size=(6, 4),
        sort_by_key=True,
        show_mean=True,
    )
    plot_bar(
        n_props_distribution,
        fname=f"{save_dir}/n_props_distribution.png",
        ylabel="n episodes",
        title="Propositions Per Episode",
        fig_size=(6, 4),
        sort_by_key=True,
        show_mean=True,
    )
    plot_pie(
        list(predicate_dist_percents.values()),
        list(predicate_dist_percents.keys()),
        fname=f"{save_dir}/predicate_distribution.png",
        title=f"Distribution of Predicates (n={total_predicates})",
    )


def analyze(dataset: Dict[str, Any], metadata: Dict[str, str], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    object_furniture_room_analysis(dataset, metadata, save_dir)
    predicate_analysis(dataset, save_dir)
    task_type_analysis(dataset, save_dir)
    ambiguity_analysis(dataset, save_dir)
    temporal_analysis(dataset, save_dir)
    dependent_rearrange_analysis(dataset, save_dir)
    subset_count_analysis(dataset, save_dir)
    multi_step_analysis(dataset, save_dir)


def load_metadata(annotations_dir: str) -> Dict[str, Any]:
    """modified from MetadataInterface to extract object classes and receptacle classes"""

    df_static_objects = pd.read_csv(
        os.path.join(annotations_dir, "fpmodels-with-decomposed.csv")
    )
    df_objects = pd.read_csv(
        os.path.join(annotations_dir, "object_categories_filtered.csv")
    )

    df1 = df_static_objects.rename(columns={"id": "handle", "main_category": "type"})

    # get recep cat set
    df1_sets = df1[["handle", "type", "notes"]]
    receptacle_classes = set()
    for index in range(df1_sets.shape[0]):
        cat = df1_sets.at[index, "type"]
        note = df1_sets.at[index, "notes"]
        if note == "receptacle":
            receptacle_classes.add(cat)

    df2 = df_objects.rename(columns={"id": "handle", "clean_category": "type"})
    df1 = df1[["handle", "type"]]
    df2 = df2[["handle", "type"]]
    metadata = pd.concat([df1, df2], ignore_index=True)

    hash_to_cat: Dict[str, str] = {}
    for index in range(metadata.shape[0]):
        cat = metadata.at[index, "type"]
        hash_to_cat[metadata.at[index, "handle"]] = cat

    # object cat set
    object_classes = (
        set(c for c in hash_to_cat.values() if isinstance(c, str)) - receptacle_classes
    )

    return {
        "hash_to_cat": hash_to_cat,
        "object_classes": object_classes,
        "receptacle_classes": receptacle_classes,
    }


def main():
    """
    This dataset analysis includes plots of the following distributions:
        -  propositions per episode
        -  rearrangements per episode
        -  objects
        -  predicates
        -  furniture classes
        -  task types
        -  tasks with (resolvable) object/furniture/room ambiguity
        -  tasks with subset counts
        -  tasks with multi-step object rearrangements (terminal constraint false)
        -  tasks with dependent rearranges (SameArg, DifferentArg constraints)
        -  number of temporal task "steps"
        -  temporal group sizes
        -  tasks with only group size == 1 (single-agent tasks)

    To run:
        >>> python dataset_generation/benchmark_generation/analysis/run_dataset_analysis.py
    For more info:
        >>> python dataset_generation/benchmark_generation/analysis/run_dataset_analysis.py --help
    """
    parser = argparse.ArgumentParser(
        description="Dataset analysis script that saves image plots."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/datasets/partnr_episodes/v0_0/val.json.gz",
        type=str,
        help="Path to the collaboration dataset",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/hssd-hab/metadata",
        type=str,
        help="Path to the metadata annotations directory",
    )
    parser.add_argument(
        "--save-dir",
        default="data/dataset_analysis",
        type=str,
        help="Path to where the analysis images should be saved",
    )
    args = parser.parse_args()
    dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    analyze(
        load_dataset(args.dataset_path),
        load_metadata(args.metadata_dir),
        os.path.join(args.save_dir, dataset_name),
    )


if __name__ == "__main__":
    main()
