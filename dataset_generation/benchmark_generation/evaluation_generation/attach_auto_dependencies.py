#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import json
from typing import Collection, List, Optional, Set

from habitat_llm.agent.env.dataset import CollaborationDatasetV0, CollaborationEpisode
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    EvaluationProposition,
    EvaluationPropositionDependency,
    TemporalConstraint,
    TerminalSatisfactionConstraint,
)

NAMES_TO_SEARCH_HANDLES = {
    "is_on_floor": ["object_handles"],
    "is_inside": ["object_handles"],
    "is_on_top": ["object_handles"],
    "is_in_room": ["object_handles"],
    "is_next_to": ["entity_handles_a", "entity_handles_b"],
    "is_clustered": ["*args"],
    "is_clean": ["object_handles"],
    "is_dirty": ["object_handles"],
    "is_filled": ["object_handles"],
    "is_empty": ["object_handles"],
    "is_powered_on": ["object_handles"],
    "is_powered_off": ["object_handles"],
}
ALL_PROPOSITIONS = set(NAMES_TO_SEARCH_HANDLES.keys())
PLACEMENT_PROPOSITIONS = {"is_on_floor", "is_inside", "is_on_top", "is_in_room"}
OBJECT_STATE_PROPOSITION_NEGATIONS = {
    "is_clean": "is_dirty",
    "is_dirty": "is_clean",
    "is_filled": "is_empty",
    "is_empty": "is_filled",
    "is_powered_on": "is_powered_off",
    "is_powered_off": "is_powered_on",
}
OBJECT_STATE_PROPOSITIONS = set(OBJECT_STATE_PROPOSITION_NEGATIONS.keys())


def get_proposition_groups_from_episode(
    episode: CollaborationEpisode,
) -> List[List[int]]:
    """Extract proposition groups from the episode's TemporalConstraint."""
    tc_constraints = [
        c for c in episode.evaluation_constraints if isinstance(c, TemporalConstraint)
    ]
    assert len(tc_constraints), "no temporal constraint found."

    prop_groups = tc_constraints[0].get_topological_generations()
    if len(prop_groups) == 0:
        return [list(range(len(episode.evaluation_propositions)))]
    return tc_constraints[0].get_topological_generations()


def get_mapping_prop_idx_to_temporal_group(temporal_groups: List[List[int]], n: int):
    prop_idx_to_temporal_group = {}
    for i in range(n):
        for grp in temporal_groups:
            if i in grp:
                prop_idx_to_temporal_group[i] = grp
                break
        else:
            raise ValueError("prop idx not found in temporal group")
    return prop_idx_to_temporal_group


def get_temporal_group_index_of_proposition(
    temporal_groups: List[List[int]], prop_idx: int
) -> int:
    for i in range(len(temporal_groups)):
        if prop_idx in temporal_groups[i]:
            return i
    else:
        raise ValueError("prop idx not found in temporal group")


def entity_in_proposition(entity: str, proposition: EvaluationProposition) -> bool:
    """
    searches for the name `entity`in the arguments of the
    proposition specified by NAMES_TO_SEARCH_HANDLES.
    """
    for n in NAMES_TO_SEARCH_HANDLES[proposition.function_name]:
        if isinstance(proposition.args[n], list):
            for e_list in proposition.args[n]:  # noqa: SIM110
                if entity in e_list:
                    return True
        else:
            if entity in proposition.args[n]:
                return True
    return False


def find_entity_in_propositions(
    entity_name: str,
    propositions: List[EvaluationProposition],
    proposition_group: Optional[Collection[int]] = None,
    exclude_idx: Optional[int] = None,
    prop_name_filter: Collection[str] = PLACEMENT_PROPOSITIONS,
) -> Set[int]:
    """returns the indices of propositions containing the entity entity_name"""
    if proposition_group is None:
        proposition_group = set(range(len(propositions)))

    if exclude_idx is None:
        exclude_idx = -1

    props_with_entity = set()
    for i in proposition_group:
        if i == exclude_idx:
            continue
        prop = propositions[i]
        if prop.function_name not in prop_name_filter:
            continue
        if not entity_in_proposition(entity_name, prop):
            continue
        props_with_entity.add(i)

    return props_with_entity


def flatten_nested_lists(x):
    """flatten a list of x where x can be either a list of x or an entity"""
    if isinstance(x, list):
        return [a for i in x for a in flatten_nested_lists(i)]
    else:
        return [x]


def infer_dependencies_next_to(
    episode: CollaborationEpisode,
) -> List[EvaluationPropositionDependency]:
    """
    Add a `while_satisfied` dependency if an `is_next_to` relation shares
    entities with a placement relation in the same temporal group.
    """
    dependencies: List[EvaluationPropositionDependency] = []

    # extract the temporal constraint
    temporal_groups = get_proposition_groups_from_episode(episode)
    try:
        prop_idx_to_temporal_group = get_mapping_prop_idx_to_temporal_group(
            temporal_groups, len(episode.evaluation_propositions)
        )
    except ValueError:
        print(
            "mismatch between temporal groups and propositions length. Episode ID:",
            episode.episode_id,
        )
        return dependencies

    for prop_idx, proposition in enumerate(episode.evaluation_propositions):
        if proposition.function_name not in ["is_next_to", "is_clustered"]:
            continue

        # extract entities in the proposition
        entity_lists = [
            proposition.args[k]
            for k in ["*args", "entity_handles_a", "entity_handles_b"]
            if k in proposition.args
        ]
        entities = set(flatten_nested_lists(entity_lists))

        # find propositions in the same group that have an explicit placement for the is_next_to entities
        depends_on = set()
        grp = prop_idx_to_temporal_group[prop_idx]
        for entity in entities:
            depends_on |= find_entity_in_propositions(
                entity, episode.evaluation_propositions, grp, exclude_idx=prop_idx
            )
        if len(depends_on):
            dependencies.append(
                EvaluationPropositionDependency(
                    proposition_indices=[prop_idx],
                    depends_on=sorted(depends_on),
                    relation_type="while_satisfied",
                    dependency_mode="any",
                )
            )

    return dependencies


def infer_dependencies_multistep(
    episode: CollaborationEpisode,
) -> List[EvaluationPropositionDependency]:
    """
    apply dependencies for multi-step placements (entities appear in earlier propositions)
    """
    temporal_groups = get_proposition_groups_from_episode(episode)
    dependencies: List[EvaluationPropositionDependency] = []
    for prop_idx, proposition in enumerate(episode.evaluation_propositions):
        if proposition.function_name not in PLACEMENT_PROPOSITIONS:
            continue
        if prop_idx in temporal_groups[0]:
            continue
        try:
            cur_temporal_grp_idx = get_temporal_group_index_of_proposition(
                temporal_groups, prop_idx
            )
        except ValueError:
            print(
                "mismatch between temporal groups and propositions length. Episode ID:",
                episode.episode_id,
            )
            return dependencies

        # for each earlier temporal group, check if entity exists.
        for entity in proposition.args["object_handles"]:
            depends_on = set()
            for i in range(cur_temporal_grp_idx - 1, -1, -1):
                depends_on |= find_entity_in_propositions(
                    entity,
                    episode.evaluation_propositions,
                    temporal_groups[i],
                    prop_name_filter=ALL_PROPOSITIONS - OBJECT_STATE_PROPOSITIONS,
                )
                if len(depends_on):
                    dependencies.append(
                        EvaluationPropositionDependency(
                            proposition_indices=[prop_idx],
                            depends_on=sorted(depends_on),
                            relation_type="after_satisfied",
                            dependency_mode="any",
                        )
                    )
                    # we only depend on the most recent placement of this entity.
                    break

    return dependencies


def infer_dependencies_floor_room(
    episode: CollaborationEpisode,
) -> List[EvaluationPropositionDependency]:
    """
    add while_satisfied to is_on_floor if is_in_room appears in the same temporal group.
    """
    temporal_groups = get_proposition_groups_from_episode(episode)
    dependencies = []

    for grp in temporal_groups:
        # get the is_floor props
        is_on_floor_props = []
        for i in grp:
            if episode.evaluation_propositions[i].function_name == "is_on_floor":
                is_on_floor_props.append((i, episode.evaluation_propositions[i]))

        for prop_idx, prop in is_on_floor_props:
            depends_on = set()

            for entity in prop.args["object_handles"]:
                # get associated is_in_room props
                depends_on |= find_entity_in_propositions(
                    entity,
                    episode.evaluation_propositions,
                    grp,
                    prop_name_filter=["is_in_room"],
                )

            if len(depends_on):
                dependencies.append(
                    EvaluationPropositionDependency(
                        proposition_indices=[prop_idx],
                        depends_on=sorted(depends_on),
                        relation_type="while_satisfied",
                    )
                )

    return dependencies


def infer_dependencies_object_states(
    episode: CollaborationEpisode,
) -> List[EvaluationPropositionDependency]:
    """
    add after_satisfied to an object state predicate if a contradictory predicate appears in an earlier temporal group.
    """
    temporal_groups = get_proposition_groups_from_episode(episode)
    dependencies: List[EvaluationPropositionDependency] = []
    for prop_idx, proposition in enumerate(episode.evaluation_propositions):
        if proposition.function_name not in OBJECT_STATE_PROPOSITIONS:
            continue
        if prop_idx in temporal_groups[0]:
            continue
        try:
            cur_temporal_grp_idx = get_temporal_group_index_of_proposition(
                temporal_groups, prop_idx
            )
        except ValueError:
            print(
                "mismatch between temporal groups and propositions length. Episode ID:",
                episode.episode_id,
            )
            return dependencies

        # for each earlier temporal group, check if a negation predicate exists for that object.
        for entity in proposition.args["object_handles"]:
            depends_on = set()
            for i in range(cur_temporal_grp_idx - 1, -1, -1):
                # work backward to get the most recent related object state change
                depends_on |= find_entity_in_propositions(
                    entity,
                    episode.evaluation_propositions,
                    temporal_groups[i],
                    prop_name_filter=OBJECT_STATE_PROPOSITION_NEGATIONS[
                        proposition.function_name
                    ],
                )
                if len(depends_on):
                    dependencies.append(
                        EvaluationPropositionDependency(
                            proposition_indices=[prop_idx],
                            depends_on=sorted(depends_on),
                            relation_type="after_satisfied",
                            dependency_mode="any",
                        )
                    )
                    # we only depend on the most recent state change of this entity.
                    break

    return dependencies


def infer_dependencies_for_episode(
    episode: CollaborationEpisode, override_existing: bool = False
) -> CollaborationEpisode:
    """ """
    if len(episode.evaluation_proposition_dependencies) and not override_existing:
        return episode

    dependencies = (
        infer_dependencies_next_to(episode)
        + infer_dependencies_multistep(episode)
        + infer_dependencies_floor_room(episode)
        + infer_dependencies_object_states(episode)
    )

    deduped_dependencies = []
    for dep in dependencies:
        if dep not in deduped_dependencies:
            deduped_dependencies.append(dep)

    episode.evaluation_proposition_dependencies = deduped_dependencies

    return episode


def update_terminal_satisfaction_constraint(
    episode: CollaborationEpisode,
) -> CollaborationEpisode:
    """
    TerminalSatisfactionConstraint logic:
    - any multistep prop that isnt the final one
    - if these props are joined with any 'while_satisfieds', remove those too.
    - if contradictory object state predicates exist, keep only the latest.
    """
    to_remove = set()

    # remove multi-step source steps and prior object state manipulations
    for dep in episode.evaluation_proposition_dependencies:
        if dep.relation_type == "after_satisfied":
            to_remove.update(dep.depends_on)

    # check if any steps are paired with a "while_satisfied". remove those too.
    for dep in episode.evaluation_proposition_dependencies:
        if dep.relation_type != "while_satisfied":
            continue
        if any(step in to_remove for step in dep.depends_on):
            to_remove.update(dep.proposition_indices)

    for _i, c in enumerate(episode.evaluation_constraints):
        if isinstance(c, TerminalSatisfactionConstraint):
            c.proposition_indices = [
                p for p in c.proposition_indices if p not in to_remove
            ]
            c.args["proposition_indices"] = list(c.proposition_indices)
            break
    else:
        n_props = len(episode.evaluation_propositions)
        prop_idxs = [i for i in range(n_props) if i not in to_remove]
        episode.evaluation_constraints.append(
            TerminalSatisfactionConstraint(prop_idxs, n_props)
        )

    return episode


def infer_and_attach_dependencies(
    dataset: CollaborationDatasetV0, override_existing: bool = True
) -> CollaborationDatasetV0:
    """
    Infers and attaches evaluation proposition dependencies for each episode. Then, these
    these dependencies are used to infer the terminal satisfaction constraint. if
    `override_existing` is false, dependencies + terminal constraints are inferred only
    for episodes that do not already have these elements.
    """
    eps_w_deps_before = sum(
        len(ep.evaluation_proposition_dependencies) > 0 for ep in dataset.episodes
    )

    dataset.episodes = [
        infer_dependencies_for_episode(ep, override_existing) for ep in dataset.episodes
    ]
    dataset.episodes = [
        update_terminal_satisfaction_constraint(ep) for ep in dataset.episodes
    ]

    attached_deps = sum(
        len(ep.evaluation_proposition_dependencies) > 0 for ep in dataset.episodes
    )
    if not override_existing:
        attached_deps -= eps_w_deps_before

    print(f"Attached dependencies to {attached_deps}/{len(dataset.episodes)} episodes.")
    return dataset


def main():
    """
    Automatically infers proposition dependencies and a terminal satisfaction constraint.
    Uses the list of evaluation propositions and the annotated TemporalConstraint.

    Limitation: does not reason explicitly about initial state.
    Can add this if necessary by deriving initial state propositions.

    To run:
        >>> python dataset_generation/benchmark_generation/attach_auto_dependencies.py \
            --dataset-path [path-to-dataset] --save-to [path-to-dataset-out]
    """
    parser = argparse.ArgumentParser(
        description="Add dependencies and terminal constraint to a dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the collaboration dataset",
    )
    parser.add_argument(
        "--save-to",
        default="dataset_w_deps.json.gz",
        type=str,
        help="Path to where the resulting dataset should be saved",
    )
    parser.add_argument("--override-existing", default=False, type=bool)
    args = parser.parse_args()
    with gzip.open(args.dataset_path, "rt") as f:
        dataset_json = json.load(f)

    dataset = CollaborationDatasetV0()
    dataset.from_json(json.dumps(dataset_json))

    dataset = infer_and_attach_dependencies(dataset, args.override_existing)
    with gzip.open(args.save_to, "wt") as f:
        f.write(dataset.to_json())
    with open("dataset_tst_view.json", "w") as f:
        json.dump(json.loads(dataset.to_json()), f, indent=2)


if __name__ == "__main__":
    main()
