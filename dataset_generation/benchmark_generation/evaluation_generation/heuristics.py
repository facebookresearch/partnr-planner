#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Callable, List, Set, Union

from dataset_generation.benchmark_generation.evaluation_generation.parsing import (
    TemporalParser,
)
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    DifferentArgConstraint,
    EvaluationProposition,
    SameArgConstraint,
    TemporalConstraint,
)


def filter_generated_ties_heuristic(
    eid: int,
    ties: List[Union[SameArgConstraint, DifferentArgConstraint]],
    propositions: List[EvaluationProposition],
    log_func: Callable,
) -> List[Union[SameArgConstraint, DifferentArgConstraint]]:
    """
    Keep tie constraints that apply to 2+ propositions w/ more than one satisfying value.
    """
    valid_ties = []
    for tie in ties:
        if len(tie.proposition_indices) < 2:
            continue

        for prop_idx, arg_name in zip(tie.proposition_indices, tie.args["arg_names"]):
            try:
                prop = propositions[prop_idx]
                matched_arg = prop.args[arg_name]
            except (IndexError, KeyError) as e:
                log_func(eid, f"[tie call] Improper indices generated. Error: {str(e)}")
                continue

            if not isinstance(matched_arg, list):
                continue
            # if a matched arg has more than one possible value, keep this tie.
            if len(matched_arg) > 1:
                valid_ties.append(tie)
                break

    return valid_ties


def nearest_connected_placement_group(
    entities: Set[str],
    group_idx: int,
    propositions: List[EvaluationProposition],
    temporal_groups: List[List[int]],
):
    """
    For a given temporal group index, find the nearest temporal group index that contains
    a placement proposition with at least one of the provided entities appearing in its
    object_handles argument.
    """
    placement_props = {"is_on_top", "is_on_floor", "is_inside", "is_in_room"}
    groups_to_check = list(range(len(temporal_groups)))
    # sort by nearest group to the group_idx. if equal distance, take earliest group first.
    groups_to_check = sorted(groups_to_check, key=lambda x: (abs(x - group_idx), x))

    for group_idx in groups_to_check:
        for prop_idx in temporal_groups[group_idx]:
            prop = propositions[prop_idx]
            if prop.function_name not in placement_props:
                continue
            if not len(set(prop.args["object_handles"]) & entities):
                continue
            return group_idx

    # no connected placement group
    return -1


def spatial_temporal_correction_heuristic(
    tc: TemporalConstraint, propositions: List[EvaluationProposition]
):
    """
    Next-to propositions are often predicted to be in separate temporal groups than
    their respective placement propositions. This heuristic corrects for this failure
    mode. Solving this problem via prompting is non-trivial because the proposition
    generation is inconsistent about proposition order.
    """
    temporal_groups = TemporalParser.groups_from_constraint(tc)
    if len(temporal_groups) < 2:
        # no temporal dependency
        return tc

    to_check = []  # (prop_idx, entities, temporal_group_idx)
    for i, prop in enumerate(propositions):
        if prop.function_name == "is_next_to":
            entities = set(
                prop.args["entity_handles_a"] + prop.args["entity_handles_b"]
            )
            for temporal_group_idx, g in enumerate(temporal_groups):  # noqa: B007
                if i in g:
                    break
            else:
                # index of this proposition is missing; add it to the last group.
                temporal_groups[-1].append(i)
            to_check.append((i, entities, temporal_group_idx))

    new_temporal_groups = copy.deepcopy(temporal_groups)
    for next_to_idx, next_to_entities, next_to_group_idx in to_check:
        # find nearest connected placement proposition.
        g_idx = nearest_connected_placement_group(
            next_to_entities, next_to_group_idx, propositions, temporal_groups
        )
        if g_idx == -1 or g_idx == next_to_group_idx:
            continue
        # move next_to_idx to the temporal group of that prop.
        new_temporal_groups[g_idx].append(next_to_idx)
        new_temporal_groups[next_to_group_idx].remove(next_to_idx)

    # remove empty temporal groups
    new_temporal_groups = [g for g in new_temporal_groups if len(g)]

    return TemporalParser.constraint_from_groups(
        new_temporal_groups, n_props=len(propositions)
    )
