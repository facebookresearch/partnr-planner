#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import itertools
import os
from copy import deepcopy
from typing import List

import habitat
import numpy as np
import pytest
from habitat.sims.habitat_simulator import sim_utilities
from habitat.sims.habitat_simulator.object_state_machine import set_state_of_obj
from hydra import compose, initialize

from habitat_llm.agent.env import register_sensors  # noqa
from habitat_llm.agent.env import register_measures
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    DifferentArgConstraint,
    EvaluationConstraint,
    EvaluationProposition,
    EvaluationPropositionDependency,
    SameArgConstraint,
    TemporalConstraint,
    TerminalSatisfactionConstraint,
    apply_constraint_satisfaction,
    compute_percent_complete,
    determine_propositions_to_evaluate,
    unroll_propositions_with_number,
)
from habitat_llm.agent.env.evaluation.failure_explanations import (
    derive_evaluation_explanation,
)
from habitat_llm.agent.env.evaluation.predicate_wrappers import (
    PropositionResult,
    SimBasedPredicates,
)
from habitat_llm.sims.metadata_interface import MetadataInterface, default_metadata_dict
from habitat_llm.utils import setup_config


def init_env(cfg):
    os.environ["GLOG_minloglevel"] = "3"  # noqa: SIM112
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.agents_order = sorted(cfg.habitat.simulator.agents.keys())

    register_sensors(cfg)
    register_measures(cfg)
    env = habitat.Env(config=cfg)
    env.sim.dynamic_target = np.zeros(3)
    env.reset()
    return env


def test_sim_predicates():
    """Queries all predicates against the first CI episode."""

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="benchmark_gen/evaluation_validation.yaml",
            overrides=[
                "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
                "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
                "habitat.dataset.data_path='data/datasets/partnr_episodes/v0_0/ci.json.gz'",
            ],
        )

    if not CollaborationDatasetV0.check_config_paths_exist(cfg.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    cfg = setup_config(cfg)

    env = init_env(cfg)
    sim = env.sim

    ao_link_map = sim_utilities.get_ao_link_id_map(sim)

    # ensure scene has objects and receptacles we test with
    ep = env.current_episode
    assert ep.scene_id == "102817140", "expected scene `102817140`."
    oh1 = "pillow_9_:0000"
    oh2 = "CREATIVE_BLOCKS_35_MM_:0000"
    recep_handle_1 = "925f48efff312862677c9517a8ec0faba4909570_part_2_:0000"
    room_1 = "bedroom.001"
    room_2 = "living room"
    SimBasedPredicates.sim_instance_from_handle(sim, oh1)
    SimBasedPredicates.sim_instance_from_handle(sim, oh2)
    SimBasedPredicates.sim_instance_from_handle(sim, recep_handle_1)
    SimBasedPredicates.sim_region_from_id(sim, room_1)
    SimBasedPredicates.sim_region_from_id(sim, room_2)

    # run each sim predicate
    q = SimBasedPredicates.is_inside(sim, [oh1], [recep_handle_1])
    assert not q.is_satisfied
    q = SimBasedPredicates.is_on_top(sim, [oh1], [recep_handle_1])
    assert not q.is_satisfied
    q = SimBasedPredicates.is_on_floor(sim, [oh1], ao_link_map=ao_link_map)
    assert not q.is_satisfied
    q = SimBasedPredicates.is_in_room(sim, [oh2], [room_1], ao_link_map=ao_link_map)
    assert q.is_satisfied
    q = SimBasedPredicates.is_in_room(sim, [oh1], [room_2], ao_link_map=ao_link_map)
    assert not q.is_satisfied
    q = SimBasedPredicates.is_next_to(sim, [oh1], [oh2])
    assert not q.is_satisfied
    q = SimBasedPredicates.is_next_to(sim, [oh1], [recep_handle_1], l2_threshold=20.0)
    assert q.is_satisfied
    q = SimBasedPredicates.is_inside(
        sim, [oh1, oh2], [recep_handle_1], number=2, is_same_receptacle=True
    )
    assert not q.is_satisfied
    q = SimBasedPredicates.is_inside(sim, [oh1, oh2], [recep_handle_1])
    assert not q.is_satisfied

    q = SimBasedPredicates.is_clustered(
        [oh1],
        [oh2],
        sim=sim,
        number=[1, 1],
        l2_threshold=0.5,
        ao_link_map=ao_link_map,
    )
    assert not q.is_satisfied

    # test object state predicates
    # first: apply all affordances to the object classes we test with
    for state in sim.object_state_machine.active_states:
        state.accepted_semantic_classes += ["cushion", "toy_construction_set"]
    sim.object_state_machine.initialize_object_state_map(sim)

    handle = "pillow_9_:0000"
    # test default initial state predicates
    assert not SimBasedPredicates.is_clean(sim, [handle]).is_satisfied
    assert not SimBasedPredicates.is_filled(sim, [handle]).is_satisfied
    assert not SimBasedPredicates.is_powered_on(sim, [handle]).is_satisfied
    assert SimBasedPredicates.is_dirty(sim, [handle]).is_satisfied
    assert SimBasedPredicates.is_empty(sim, [handle]).is_satisfied
    assert SimBasedPredicates.is_powered_off(sim, [handle]).is_satisfied

    # flip the values, check again.
    obj = SimBasedPredicates.sim_instance_from_handle(sim, handle)
    set_state_of_obj(obj, "is_clean", True)
    set_state_of_obj(obj, "is_filled", True)
    set_state_of_obj(obj, "is_powered_on", True)

    assert SimBasedPredicates.is_clean(sim, [handle]).is_satisfied
    assert SimBasedPredicates.is_filled(sim, [handle]).is_satisfied
    assert SimBasedPredicates.is_powered_on(sim, [handle]).is_satisfied
    assert not SimBasedPredicates.is_dirty(sim, [handle]).is_satisfied
    assert not SimBasedPredicates.is_empty(sim, [handle]).is_satisfied
    assert not SimBasedPredicates.is_powered_off(sim, [handle]).is_satisfied

    handle = "CREATIVE_BLOCKS_35_MM_:0000"
    # check different initial states for the other object
    assert SimBasedPredicates.is_clean(sim, [handle]).is_satisfied
    assert not SimBasedPredicates.is_dirty(sim, [handle]).is_satisfied

    # test that object state negation predicates handle OR correctly
    cases = ((False, True, True), (True, True, False), (False, False, True))
    oh1, oh2 = "pillow_9_:0000", "CREATIVE_BLOCKS_35_MM_:0000"
    for sat1, sat2, expected in cases:
        set_state_of_obj(
            SimBasedPredicates.sim_instance_from_handle(sim, oh1), "is_clean", sat1
        )
        set_state_of_obj(
            SimBasedPredicates.sim_instance_from_handle(sim, oh2), "is_clean", sat2
        )
        assert SimBasedPredicates.is_dirty(sim, [oh1, oh2]).is_satisfied == expected

    env.close()
    del env


@pytest.mark.parametrize(
    "relation,dependency_mode",
    list(
        itertools.product(
            [
                "while_satisfied",
                "after_satisfied",
                "after_unsatisfied",
                "before_satisfied",
            ],
            ["all", "any"],
        )
    ),
)
def test_proposition_dependencies_any_mode(relation: str, dependency_mode: str):
    """
    Proposition dependencies default to requiring all depending propositions
    to be satisfied (mode: all). This function tests (mode: any).
    """
    state_sequence = [
        [PropositionResult(False), PropositionResult(False), PropositionResult(False)],
        [PropositionResult(True), PropositionResult(False), PropositionResult(False)],
    ]
    propositions = [
        EvaluationProposition("a", {}),
        EvaluationProposition("b", {}),
        EvaluationProposition("c", {}),
    ]
    expected_result = {
        "all": {
            "while_satisfied": {0, 1},
            "after_satisfied": {0, 1},
            "after_unsatisfied": {0, 1},
            "before_satisfied": {0, 1},
        },
        "any": {
            "while_satisfied": {0, 1, 2},
            "after_satisfied": {0, 1, 2},
            "after_unsatisfied": {0, 1},
            "before_satisfied": {0, 1, 2},
        },
    }[dependency_mode][relation]

    dependencies = [
        EvaluationPropositionDependency(
            proposition_indices=[2],
            depends_on=[0, 1],
            relation_type=relation,
            dependency_mode=dependency_mode,
        )
    ]
    propositions_to_evaluate = determine_propositions_to_evaluate(
        state_sequence, propositions, dependencies
    )
    assert propositions_to_evaluate == expected_result


@pytest.mark.parametrize(
    "relation",
    ["while_satisfied", "after_satisfied", "after_unsatisfied", "before_satisfied"],
)
def test_proposition_dependencies(relation: str):
    """
    tests that the `propositions_to_evaluate` are correct given at each point in time of
    adding a new state to the state sequence. Verifies the indices are correct for each
    type of dependency relation.
    """
    states_to_add = [
        [PropositionResult(False), PropositionResult(False)],
        [PropositionResult(True), PropositionResult(False)],
        [PropositionResult(False), PropositionResult(True)],
    ]
    propositions = [
        EvaluationProposition(function_name="a", args={}),
        EvaluationProposition(function_name="b", args={}),
    ]
    dependencies = [
        EvaluationPropositionDependency(
            proposition_indices=[1], depends_on=[0], relation_type=relation
        )
    ]
    expected_results = [
        {
            "while_satisfied": {0},
            "after_satisfied": {0},
            "after_unsatisfied": {0},
            "before_satisfied": {0, 1},
        },
        {
            "while_satisfied": {0, 1},
            "after_satisfied": {0, 1},
            "after_unsatisfied": {0},
            "before_satisfied": {0},
        },
        {
            "while_satisfied": {0},
            "after_satisfied": {0, 1},
            "after_unsatisfied": {0, 1},
            "before_satisfied": {0},
        },
    ]

    state_sequence = []
    for state, expected_result in zip(states_to_add, expected_results):
        state_sequence.append(state)
        propositions_to_evaluate = determine_propositions_to_evaluate(
            state_sequence, propositions, dependencies
        )
        assert propositions_to_evaluate == expected_result[relation]


def test_proposition_dependencies_multi():
    """
    tests that `propositions_to_evaluate` are correct in the multi-dependency case.
    """
    states_to_add = [
        [PropositionResult(False), PropositionResult(False), PropositionResult(False)],
        [PropositionResult(True), PropositionResult(False), PropositionResult(False)],
        [PropositionResult(True), PropositionResult(True), PropositionResult(True)],
    ]
    propositions = [
        EvaluationProposition(function_name="a", args={}),
        EvaluationProposition(function_name="b", args={}),
        EvaluationProposition(function_name="c", args={}),
    ]
    dependencies = [
        EvaluationPropositionDependency(
            proposition_indices=[2], depends_on=[0], relation_type="after_satisfied"
        ),
        EvaluationPropositionDependency(
            proposition_indices=[2], depends_on=[1], relation_type="while_satisfied"
        ),
    ]
    expected_results = [
        {0, 1},
        {0, 1},
        {0, 1, 2},
    ]

    state_sequence = []
    for state, expected_result in zip(states_to_add, expected_results):
        state_sequence.append(state)
        propositions_to_evaluate = determine_propositions_to_evaluate(
            state_sequence, propositions, dependencies
        )
        assert propositions_to_evaluate == expected_result


def test_evaluation_constraints():
    """Queries each evaluation constraint"""

    # test: no temporal constraint
    tc = TemporalConstraint(dag_edges=[], n_propositions=2)
    assert tc(state_sequence=None, proposition_satisfied_at=[1, 1]) == [True, True]
    assert tc(state_sequence=None, proposition_satisfied_at=[2, 1]) == [True, True]
    assert tc(state_sequence=None, proposition_satisfied_at=[1, 2]) == [True, True]

    # test: temporal constraint
    tc = TemporalConstraint(dag_edges=[(0, 1)], n_propositions=2)
    assert tc(state_sequence=None, proposition_satisfied_at=[1, 1]) == [True, False]
    assert tc(state_sequence=None, proposition_satisfied_at=[2, 1]) == [True, False]
    assert tc(state_sequence=None, proposition_satisfied_at=[1, 2]) == [True, True]

    ss_1 = [
        [
            PropositionResult(True, {"x": "xval"}),
            PropositionResult(False, {}),
            PropositionResult(False, {}),
        ],
        [
            PropositionResult(True, {"x": "xval"}),
            PropositionResult(True, {"y": "xval"}),
            PropositionResult(True, {"y": "diff"}),
        ],
    ]
    ss_2 = [
        [
            PropositionResult(True, {"x": "xval"}),
            PropositionResult(False, {}),
            PropositionResult(False, {}),
        ],
        [
            PropositionResult(True, {"x": "xval"}),
            PropositionResult(True, {"y": "diff"}),
            PropositionResult(True, {"y": "diff"}),
        ],
    ]
    TTT = [True, True, True]
    TFT = [True, False, True]

    # test: same arg constraint
    sac = SameArgConstraint(
        proposition_indices=[0, 1], arg_names=["x", "y"], n_propositions=3
    )
    assert sac(state_sequence=ss_1, proposition_satisfied_at=[0, 1, 1]) == TTT
    assert sac(state_sequence=ss_2, proposition_satisfied_at=[0, 1, 1]) == TFT

    # test: different arg constraint
    dac = DifferentArgConstraint(
        proposition_indices=[0, 1], arg_names=["x", "y"], n_propositions=3
    )
    assert dac(state_sequence=ss_1, proposition_satisfied_at=[0, 1, 1]) == TFT
    assert dac(state_sequence=ss_2, proposition_satisfied_at=[0, 1, 1]) == TTT

    # test: terminal satisfaction constraint
    tsc = TerminalSatisfactionConstraint(
        proposition_indices=[0, 1, 2], n_propositions=3
    )
    assert tsc(state_sequence=ss_1, proposition_satisfied_at=[0, 1, 1]) == TTT
    tsc = TerminalSatisfactionConstraint(proposition_indices=[0], n_propositions=1)
    ss_terminal = [[PropositionResult(True)], [PropositionResult(False)]]
    assert tsc(state_sequence=ss_terminal, proposition_satisfied_at=[0]) == [False]

    # test: unroll a proposition
    propositions = [
        EvaluationProposition(
            function_name="is_inside", args={"x": ["a"], "y": ["b", "c"]}
        ),
        EvaluationProposition(
            function_name="is_on_top",
            args={
                "x": ["d", "e", "f"],
                "y": ["b", "c"],
                "number": 2,
            },
        ),
        EvaluationProposition(
            function_name="is_inside", args={"x": ["g"], "y": ["b", "c"]}
        ),
    ]
    constraints: List[EvaluationConstraint] = [
        TemporalConstraint(dag_edges=[(0, 2), (1, 2)], n_propositions=3),
        SameArgConstraint(
            proposition_indices=[0, 1], arg_names=["x", "x"], n_propositions=3
        ),
        DifferentArgConstraint(
            proposition_indices=[1, 2], arg_names=["x", "x"], n_propositions=3
        ),
        TerminalSatisfactionConstraint(proposition_indices=[0, 1, 2], n_propositions=3),
    ]
    for i in range(1, propositions[1].args["number"]):
        new_prop = deepcopy(propositions[1])
        new_prop.args["number"] = i
        propositions.append(new_prop)
        new_prop_idx = len(propositions) - 1
        for j in range(len(constraints)):
            new_constraint = constraints[j].update_unrolled_proposition(
                propositions, idx_orig=1, idx_new=new_prop_idx
            )
            if new_constraint is not None:
                constraints.extend(new_constraint)

    assert len(propositions) == 4
    assert len(constraints) == 5  # additional DifferentArgConstraint added
    assert set(constraints[0].dag.edges) == {(0, 2), (1, 2), (3, 2)}  # dag gets 3->2
    assert set(constraints[1].proposition_indices) == {0, 1, 3}
    assert set(constraints[2].proposition_indices) == {1, 2}  # stays the same
    assert set(constraints[3].proposition_indices) == {0, 1, 2, 3}
    assert set(constraints[4].proposition_indices) == {3, 2}  # with new prop idx

    # test: apply constraint satisfaction
    ss_3 = [
        [PropositionResult(True, {"x": "val"}), PropositionResult(False, {})],
        [PropositionResult(True, {"x": "val"}), PropositionResult(True, {"x": "val"})],
    ]
    constraints = [
        TemporalConstraint(dag_edges=[(0, 1)], n_propositions=2),
        SameArgConstraint(
            proposition_indices=[0, 1], arg_names=["x", "x"], n_propositions=2
        ),
    ]
    prop_satisfied_at = [0, 1]
    expected = np.array([True, True])
    res = apply_constraint_satisfaction(constraints, ss_3, prop_satisfied_at)
    assert (res == expected).all()


def test_compute_percent_complete():
    cd1 = np.array(
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    cd2 = np.array(
        [
            [True, False, True, True],
            [True, False, True, True],
            [True, False, True, True],
        ]
    )
    assert compute_percent_complete([1, 2, 3, 4], cd1) == 1.0
    assert compute_percent_complete([1, 2, -1, -1], cd1) == 0.5
    assert compute_percent_complete([-1, -1, -1, -1], cd1) == 0.0
    assert compute_percent_complete([1, 2, 3, 4], cd2) == 0.75
    assert compute_percent_complete([1, 2, -1, -1], cd2) == 0.25
    assert compute_percent_complete([-1, -1, -1, -1], cd2) == 0.0


def test_evaluation_explanation():
    # set to True for debugging
    DISPLAY_FAILURES = False

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="habitat_conf/dataset/collaboration_hssd.yaml",
            overrides=["habitat.dataset.scenes_dir=data/hssd-partnr-ci"],
        )
        dataset_cfg = cfg.habitat.dataset

    if not CollaborationDatasetV0.check_config_paths_exist(dataset_cfg):
        pytest.skip("Test skipped as dataset files are missing.")

    md = default_metadata_dict
    md["metadata_folder"] = "data/hssd-partnr-ci/metadata/"
    metadata_interface = MetadataInterface(md)

    dataset = CollaborationDatasetV0(config=dataset_cfg)
    episode = dataset.episodes[0]

    propositions, _, constraints = unroll_propositions_with_number(
        episode.evaluation_propositions,
        episode.evaluation_proposition_dependencies,
        episode.evaluation_constraints,
    )

    # add extra propositions to test all the functions
    oh1 = "pillow_9_:0000"
    oh2 = "CREATIVE_BLOCKS_35_MM_:0000"
    rh1 = "925f48efff312862677c9517a8ec0faba4909570_part_2_:0002"
    args = {"object_handles": [oh1], "receptacle_handles": [rh1], "number": 1}
    propositions.append(EvaluationProposition("is_inside", args))

    args = {"object_handles": [oh1], "number": 1}
    propositions.append(EvaluationProposition("is_on_floor", args))

    args = {"object_handles": [oh1], "room_ids": ["bedroom.001"], "number": 1}
    propositions.append(EvaluationProposition("is_in_room", args))

    args = {
        "entity_handles_a": [oh1],
        "entity_handles_b": [oh2],
        "number": 1,
        "is_same_b": False,
        "l2_threshold": 0.5,
    }
    propositions.append(EvaluationProposition("is_next_to", args))

    args = {"*args": [[oh1, oh2], [oh2]], "number": [1, 1], "l2_threshold": 0.5}
    propositions.append(EvaluationProposition("is_clustered", args))

    args = {
        "object_handles": ["925f48efff312862677c9517a8ec0faba4909570_part_2_:0002"],
        "number": 1,
    }
    propositions.append(EvaluationProposition("is_clean", args))

    n_props = len(propositions)
    test_cases = [
        {
            "name": "case 0: test all predicates as failure",
            "state_sequence": [[PropositionResult(False) for _ in range(n_props)]],
            "constraints": [],
            "str_expected": "Missing steps",
        },
        {
            "name": "case 1: test a temporal constraint failure",
            "state_sequence": [[PropositionResult(True) for _ in range(n_props)]],
            "constraints": [TemporalConstraint([(0, 1), (2, 1)], len(propositions))],
            "str_expected": "Steps were completed out of order",
        },
        {
            "name": "case 2: test a terminal constraint failure",
            "state_sequence": [
                [PropositionResult(True) for _ in range(n_props)],
                [PropositionResult(False) for _ in range(n_props)],
            ],
            "constraints": [TerminalSatisfactionConstraint(list(range(n_props)))],
            "str_expected": "Completed steps were later undone",
        },
        {
            "name": "case 3: test a same arg constraint failure",
            "state_sequence": [
                [
                    PropositionResult(True, {"object_handles": "x"}),
                    PropositionResult(True, {"object_handles": "y"}),
                ]
                + [PropositionResult(True) for _ in range(n_props - 2)]
            ],
            "constraints": [
                SameArgConstraint([0, 1], ["object_handles", "object_handles"])
            ],
            "str_expected": "The same object should have been used for the following placements",
        },
        {
            "name": "case 4: test a different arg constraint failure",
            "state_sequence": [
                [
                    PropositionResult(True, {"object_handles": "x"}),
                    PropositionResult(True, {"object_handles": "x"}),
                ]
                + [PropositionResult(True) for _ in range(n_props - 2)]
            ],
            "constraints": [
                DifferentArgConstraint([0, 1], ["object_handles", "object_handles"])
            ],
            "str_expected": "Different objects should have been used for the following placements",
        },
    ]

    for test_case in test_cases:
        state_sequence = test_case["state_sequence"]
        constraints = test_case["constraints"]

        proposition_satisfied_at = [-1 for _ in range(len(propositions))]
        for i in range(len(propositions)):
            for t in range(len(state_sequence)):
                if proposition_satisfied_at[i] != -1:
                    continue
                if state_sequence[t][i].is_satisfied:
                    proposition_satisfied_at[i] = t

        constraint_satisfaction = apply_constraint_satisfaction(
            constraints, state_sequence, proposition_satisfied_at
        )

        explanation_str = derive_evaluation_explanation(
            propositions,
            constraints,
            proposition_satisfied_at,
            constraint_satisfaction,
            metadata_interface,
        )
        assert explanation_str != ""
        assert test_case["str_expected"] in explanation_str
        if DISPLAY_FAILURES:
            print(test_case["name"])
            print()
            print(explanation_str)
            print()
