#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

from collections import defaultdict
from typing import List


def get_arg_name_from_arg_name(arg_name: str) -> str:
    if arg_name == "room_ids":
        arg_name = "room_names"
    elif arg_name == "object_handles":
        arg_name = "object_names"
    elif arg_name == "receptacle_handles":
        arg_name = "receptacle_names"
    elif arg_name == "entity_handles_a":
        arg_name = "entity_handles_a_names_and_types"
    else:
        arg_name = "entity_handles_b_names_and_types"
    return arg_name


def update_proposition_given_constraints(
    evaluation_constraints: List[dict],
    evaluation_propositions: List[dict],
    local_idx: int,
    entities: List[str],
    prop_arg_name: str,
    local_to_global_idx: dict,
) -> None:
    # NOTE: Currently this function only works for Objects and Entities, not Rooms and Receptacles!
    if prop_arg_name == "entity_handles_a" or prop_arg_name == "entity_handles_b":
        entities = [entity[0] for entity in entities]

    for constraint in evaluation_constraints:
        if constraint["type"] == "SameArgConstraint":
            prop_indices = constraint["args"]["proposition_indices"]
            arg_names = constraint["args"]["arg_names"]
            if local_to_global_idx[local_idx] in prop_indices and (
                arg_names[prop_indices.index(local_to_global_idx[local_idx])]
                == prop_arg_name
            ):
                for idx, arg_name in zip(prop_indices, arg_names):
                    current_arg_name = get_arg_name_from_arg_name(arg_name)
                    if idx < len(evaluation_propositions):
                        if (
                            current_arg_name == "entity_handles_a_names_and_types"
                            or current_arg_name == "entity_handles_b_names_and_types"
                        ):
                            evaluation_propositions[idx]["args"][current_arg_name] = [
                                (entity, "object") for entity in entities
                            ]
                        else:
                            evaluation_propositions[idx]["args"][
                                current_arg_name
                            ] = entities

        elif constraint["type"] == "DiffArgConstraint":
            prop_indices = constraint["args"]["proposition_indices"]
            arg_names = constraint["args"]["arg_names"]
            if local_to_global_idx[local_idx] in prop_indices and (
                arg_names[prop_indices.index(local_to_global_idx[local_idx])]
                == prop_arg_name
            ):
                for idx, arg_name in zip(prop_indices, arg_names):
                    if idx < len(evaluation_propositions):
                        for entity in entities:
                            current_arg_name = get_arg_name_from_arg_name(arg_name)
                            if (
                                current_arg_name == "entity_handles_a_names_and_types"
                                or current_arg_name
                                == "entity_handles_b_names_and_types"
                            ):
                                if (entity, "object") in evaluation_propositions[idx][
                                    "args"
                                ][current_arg_name]:
                                    evaluation_propositions[idx]["args"][
                                        current_arg_name
                                    ].remove((entity, "object"))

                            else:
                                if (
                                    entity
                                    in evaluation_propositions[idx]["args"][
                                        current_arg_name
                                    ]
                                ):
                                    evaluation_propositions[idx]["args"][
                                        current_arg_name
                                    ].remove(entity)


def update_object_recep_and_room(
    initial_object_to_recep: dict[str, str],
    initial_object_to_room: dict[str, str],
    current_propositions: List[dict],
    evaluation_propositions: List[dict],
    evaluation_constraints: List[dict] = None,
    global_to_local_idx: dict[int, int] = None,
) -> tuple[dict[str, str], dict[str, str]]:
    # Initialize dictionaries to hold potential solutions
    potential_recep: defaultdict[str, set[str]] = defaultdict(set)
    potential_room: defaultdict[str, set[str]] = defaultdict(set)

    # Track processed objects to handle fallback
    processed_objects = set()

    local_to_global_idx = (
        {v: k for k, v in global_to_local_idx.items()} if global_to_local_idx else None
    )

    # Process constraints
    if evaluation_constraints:
        for constraint in evaluation_constraints:
            if constraint["type"] == "SameArgConstraint":
                prop_indices = constraint["args"]["proposition_indices"]
                arg_names = constraint["args"]["arg_names"]

                # Collect common intersecting values
                common_values: set[str] = set()
                for idx, arg_name in zip(prop_indices, arg_names):
                    arg_name = get_arg_name_from_arg_name(arg_name)
                    if idx in global_to_local_idx:
                        curr_idx = global_to_local_idx[idx]
                        if curr_idx < len(current_propositions):
                            prop = current_propositions[curr_idx]
                            values = set(prop["args"][arg_name])
                            if common_values:
                                common_values &= values
                            else:
                                common_values = values
                    if idx < len(evaluation_propositions):
                        prop = evaluation_propositions[idx]
                        values = set(prop["args"][arg_name])
                        if common_values:
                            common_values &= values
                        else:
                            common_values = values

                # Update propositions with intersecting values
                for idx, arg_name in zip(prop_indices, arg_names):
                    if idx in global_to_local_idx:
                        curr_idx = global_to_local_idx[idx]
                        if curr_idx < len(current_propositions):
                            current_propositions[curr_idx]["args"][arg_name] = list(
                                common_values
                            )
                for idx, arg_name in zip(prop_indices, arg_names):
                    if idx < len(evaluation_propositions):
                        evaluation_propositions[idx]["args"][arg_name] = list(
                            common_values
                        )

            elif constraint["type"] == "DiffArgConstraint":
                prop_indices = constraint["args"]["proposition_indices"]
                arg_names = constraint["args"]["arg_names"]

                # Collect common intersecting values
                common_values = set()
                for idx, arg_name in zip(prop_indices, arg_names):
                    arg_name = get_arg_name_from_arg_name(arg_name)
                    if idx in global_to_local_idx:
                        curr_idx = global_to_local_idx[idx]
                        if curr_idx < len(current_propositions):
                            prop = current_propositions[curr_idx]
                            values = set(prop["args"][arg_name])
                            common_values |= values
                    if idx < len(evaluation_propositions):
                        prop = evaluation_propositions[idx]
                        values = set(prop["args"][arg_name])
                        common_values |= values

                # Remove intersecting values from propositions
                for idx, arg_name in zip(prop_indices, arg_names):
                    if idx in global_to_local_idx:
                        curr_idx = global_to_local_idx[idx]
                        if curr_idx < len(current_propositions):
                            prop_values = set(
                                current_propositions[curr_idx]["args"][arg_name]
                            )
                            current_propositions[curr_idx]["args"][arg_name] = list(
                                prop_values - common_values
                            )
                for idx, arg_name in zip(prop_indices, arg_names):
                    if idx < len(evaluation_propositions):
                        prop_values = set(
                            evaluation_propositions[idx]["args"][arg_name]
                        )
                        evaluation_propositions[idx]["args"][arg_name] = list(
                            prop_values - common_values
                        )

    # Process each proposition
    for local_idx, proposition in enumerate(current_propositions):
        func_name = proposition["function_name"]
        args = proposition["args"]

        if func_name in ["is_on_top", "is_inside"]:
            number = args["number"]
            objects = args["object_names"][:number]
            update_proposition_given_constraints(
                evaluation_constraints,
                evaluation_propositions,
                local_idx,
                objects,
                "object_handles",
                local_to_global_idx,
            )
            receptacles = args["receptacle_names"]

            for obj_name in objects:
                for receptacle_name in receptacles:
                    potential_recep[obj_name].add(receptacle_name)
                processed_objects.add(obj_name)

        elif func_name == "is_next_to":
            number = args["number"]
            entities_a = args["entity_handles_a_names_and_types"][:number]
            update_proposition_given_constraints(
                evaluation_constraints,
                evaluation_propositions,
                local_idx,
                entities_a,
                "entity_handles_a",
                local_to_global_idx,
            )
            entities_b = args["entity_handles_b_names_and_types"]

            # NOTE: Below logic has changed because next to, with different initial receps, does not lead to a new receptacle.
            # This may not work correctly everytime
            for obj_name_a, obj_type_a in entities_a:
                if obj_type_a == "object":
                    for obj_name_b, obj_type_b in entities_b:
                        if obj_type_b == "object":
                            intersect_recep = potential_recep.get(
                                obj_name_a,
                                set(
                                    [initial_object_to_recep.get(obj_name_a)]
                                    if obj_name_a in initial_object_to_recep
                                    else []
                                ),
                            )
                            intersect_room = potential_room.get(
                                obj_name_a,
                                set(
                                    [initial_object_to_room.get(obj_name_a)]
                                    if obj_name_a in initial_object_to_room
                                    else []
                                ),
                            )
                            if intersect_recep or intersect_room:
                                potential_recep[obj_name_a] = intersect_recep
                                potential_recep[obj_name_b] = intersect_recep
                                potential_room[obj_name_a] = intersect_room
                                potential_room[obj_name_b] = intersect_room
                                processed_objects.add(obj_name_a)
                                processed_objects.add(obj_name_b)

        elif func_name == "is_in_room":
            number = args["number"]
            objects = args["object_names"][:number]
            update_proposition_given_constraints(
                evaluation_constraints,
                evaluation_propositions,
                local_idx,
                objects,
                "object_handles",
                local_to_global_idx,
            )
            rooms = args["room_names"]

            for obj_name in objects:
                for room_name in rooms:
                    potential_room[obj_name].add(room_name)
                processed_objects.add(obj_name)

        elif func_name == "is_on_floor":
            # NOTE: Have not handled `number` here so far.
            objects = args["object_names"]
            for obj_name in objects:
                potential_recep.pop(obj_name, None)
                potential_room[obj_name].clear()
                processed_objects.add(obj_name)

    # Update the object_to_recep and object_to_room dictionaries
    new_object_to_recep = {}
    new_object_to_room = {}

    # Process potential solutions
    for obj_name in set(initial_object_to_recep.keys()).union(potential_recep.keys()):
        if obj_name in processed_objects:
            if potential_recep[obj_name]:
                new_object_to_recep[obj_name] = next(iter(potential_recep[obj_name]))
            else:
                new_object_to_recep[obj_name] = "unknown"
        else:
            new_object_to_recep[obj_name] = initial_object_to_recep.get(
                obj_name, "unknown"
            )

    for obj_name in set(initial_object_to_room.keys()).union(potential_room.keys()):
        if obj_name in processed_objects:
            if potential_room[obj_name]:
                new_object_to_room[obj_name] = next(iter(potential_room[obj_name]))
            else:
                new_object_to_room[obj_name] = "unknown"
        else:
            new_object_to_room[obj_name] = initial_object_to_room.get(
                obj_name, "unknown"
            )

    # Remove entries with "unknown"
    new_object_to_recep = {
        k: v for k, v in new_object_to_recep.items() if v != "unknown"
    }
    new_object_to_room = {k: v for k, v in new_object_to_room.items() if v != "unknown"}

    return new_object_to_recep, new_object_to_room
