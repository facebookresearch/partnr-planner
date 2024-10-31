#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import pytest

from dataset_generation.benchmark_generation.evaluation_generation.heuristics import (
    spatial_temporal_correction_heuristic,
)
from dataset_generation.benchmark_generation.evaluation_generation.parsing import (
    TemporalParser,
)
from habitat_llm.agent.env.evaluation.evaluation_functions import EvaluationProposition

propositions_a = [
    EvaluationProposition(
        "is_on_top", {"object_handles": ["o-x"], "receptacle_handles": ["r-x"]}
    ),
    EvaluationProposition(
        "is_on_top", {"object_handles": ["o-a"], "receptacle_handles": ["r-a"]}
    ),
    EvaluationProposition(
        "is_on_top", {"object_handles": ["o-b"], "receptacle_handles": ["r-a"]}
    ),
    EvaluationProposition(
        "is_next_to", {"entity_handles_a": ["o-b"], "entity_handles_b": ["o-a"]}
    ),
    EvaluationProposition(
        "is_on_top", {"object_handles": ["o-b"], "receptacle_handles": ["r-b"]}
    ),
]
propositions_b = [
    EvaluationProposition(
        "is_on_top",
        {
            "object_handles": ["can_0"],
            "receptacle_handles": ["table_0", "table_6", "table_8", "table_9"],
        },
    ),
    EvaluationProposition(
        "is_on_top",
        {
            "object_handles": ["spoon_0"],
            "receptacle_handles": ["table_0", "table_6", "table_8", "table_9"],
        },
    ),
    EvaluationProposition(
        "is_on_top",
        {
            "object_handles": ["bowl_0"],
            "receptacle_handles": ["table_0", "table_6", "table_8", "table_9"],
        },
    ),
    EvaluationProposition(
        "is_next_to", {"entity_handles_a": ["can_0"], "entity_handles_b": ["spoon_0"]}
    ),
    EvaluationProposition(
        "is_next_to", {"entity_handles_a": ["spoon_0"], "entity_handles_b": ["bowl_0"]}
    ),
]


@pytest.mark.parametrize(
    "src_groups,expected_groups,propositions",
    [
        (
            [[0], [1, 2], [3], [4]],
            [[0], [1, 2, 3], [4]],
            propositions_a,
        ),
        (
            [[0], [1, 2, 3], [4]],
            [[0], [1, 2, 3], [4]],
            propositions_a,
        ),
        (
            [[0, 1, 2, 3, 4]],
            [[0, 1, 2, 3, 4]],
            propositions_a,
        ),
        (
            [[0], [1], [2], [3], [4]],
            [[0], [1], [2, 3], [4]],
            propositions_a,
        ),
        (
            [[0], [1], [2], [3, 4]],
            [[0], [1, 3], [2, 4]],
            propositions_b,
        ),
        (
            [[0, 1, 2], [3, 4]],
            [[0, 1, 2, 3, 4]],
            propositions_b,
        ),
    ],
)
def test_spatial_temporal_heuristic(
    src_groups: List[List[int]],
    expected_groups: List[List[int]],
    propositions: List[EvaluationProposition],
):
    tc = TemporalParser.constraint_from_groups(src_groups, len(propositions))
    tc_corrected = spatial_temporal_correction_heuristic(tc, propositions)
    actual = TemporalParser.groups_from_constraint(tc_corrected)
    assert actual == expected_groups
