#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

from habitat_llm.agent.env.dataset import CollaborationEpisode


def generate_hash_to_text(
    metadata_csv: str,
    entity_name_to_handle: Dict,
) -> Dict[str, str]:
    """get a mapping from object/receptacle hash to text description."""
    description_map = defaultdict(str)
    description_map_culled = {}
    with open(metadata_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            description_map[row[0]] = row[1]
    for recep, handle in entity_name_to_handle.items():
        description_map_culled[recep] = description_map[handle.split("_:")[0]]
    return description_map_culled


def object_instance_info_from_episode(
    episode: CollaborationEpisode,
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    object_handles = list(episode["name_to_receptacle"].keys())

    objects = []
    object_to_room: Dict[str, str] = {}
    object_to_recep: Dict[str, str] = {}
    object_cat_to_count: DefaultDict[str, int] = defaultdict(int)
    for state_element in episode["info"]["extra_info"]["initial_state"]:
        if (
            "name" in state_element
            or "template_task_number" in state_element
            or len(state_element["object_classes"]) == 0
        ):  # skip clutter and template transfer state elements
            continue

        obj_name = state_element["object_classes"][0]
        for _ in range(state_element["number"]):
            o = f"{obj_name}_{object_cat_to_count[obj_name]}"
            object_cat_to_count[obj_name] += 1
            object_to_room[o] = state_element["allowed_regions"][0]
            object_to_recep[o] = state_element["furniture_names"][0]
            objects.append(o)

    # NOTE: this mapping is tenuous and relies on CPython dict order
    object_to_handle = {objects[i]: object_handles[i] for i in range(len(objects))}

    return (
        objects,
        object_to_handle,
        object_to_room,
        object_to_recep,
    )


def get_semantic_object_states(
    object_to_handle: Dict[str, str],
    receptacle_to_handle: Dict[str, str],
    object_states: Dict[str, Dict[str, bool]],
) -> Dict[str, Dict[str, bool]]:
    """
    Maps an object state dictionary of the form:
        {"[affordance]": {"[handle]": bool, ...}, ...}
    to semantic names of the form:
        {"[semantic_name]": {"[affordance]": bool, ...}, ...}
    """
    handle_to_obj = {v: k for k, v in (object_to_handle | receptacle_to_handle).items()}

    object_to_states: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for affordance, d in object_states.items():
        for handle, value in d.items():
            if handle not in handle_to_obj:
                continue
            object_to_states[handle_to_obj[handle]][affordance] = value
    return object_to_states


def generate_metadata_mappings(
    episode: CollaborationEpisode,
    scene_info_metadata: Dict[str, Any],
    recep_to_description: Dict[str, str],
) -> Dict[str, Any]:
    """
    Derives sorted metadata mappings using the contents of a scene info file
    and the episode data. Keys:
        {
            "objects",              # task-relevant objects (no clutter). semantic names.
            "rooms",                # semantic names
            "object_to_recep",      # semantic names
            "object_to_room",       # semantic names
            "recep_to_room",        # semantic names
            "recep_to_description", # semantic name to text description
            "object_to_handle",     # semantic name to sim handle
            "recep_to_handle",      # semantic name to sim handle
            "room_to_id",           # semantic name to sim id
            "instruction",          # original episode instruction
        }
    """

    def sort_k_single(entity_name: str):
        """
        Takes an entity name and returns a key that affords
        secondary sorting on the post index if it exists.
        """
        idx_str = entity_name.split("_")[-1]
        try:
            idx = int(idx_str)
            entity_name = "_".join(entity_name.split("_")[:-1])
        except ValueError:
            idx = 0
        return (entity_name, idx)

    def sorted_dict(d, key):
        return dict(sorted(d.items(), key=key))

    rooms = list(scene_info_metadata["room_to_id"].keys())
    recep_to_room = {}
    for _room, _receptacles in scene_info_metadata["furniture"].items():
        for receptacle in _receptacles:
            recep_to_room[receptacle] = _room

    (
        objects,
        object_to_handle,
        object_to_room,
        object_to_recep,
    ) = object_instance_info_from_episode(episode)

    receptacle_to_handle = scene_info_metadata["receptacle_to_handle"]
    room_to_id = scene_info_metadata["room_to_id"]

    object_to_states = {}
    if "object_states" in episode:
        object_to_states = get_semantic_object_states(
            object_to_handle, receptacle_to_handle, episode["object_states"]
        )

    # sort items for fast visual pathing
    objects = sorted(objects, key=sort_k_single)
    recep_to_description = sorted_dict(
        recep_to_description, key=lambda x: sort_k_single(x[0])
    )
    rooms = sorted(rooms, key=sort_k_single)
    object_to_recep = sorted_dict(
        object_to_recep, key=lambda x: (sort_k_single(x[1]), sort_k_single(x[0]))
    )
    object_to_room = sorted_dict(
        object_to_room, key=lambda x: (sort_k_single(x[1]), sort_k_single(x[0]))
    )
    object_to_states = sorted_dict(
        object_to_states, key=lambda x: (sort_k_single(x[0]))
    )
    receptacle_to_handle = sorted_dict(
        receptacle_to_handle, key=lambda x: (sort_k_single(x[0]))
    )
    recep_to_room = sorted_dict(
        recep_to_room, key=lambda x: (sort_k_single(x[1]), sort_k_single(x[0]))
    )
    room_to_id = sorted_dict(room_to_id, key=lambda x: (sort_k_single(x[0])))
    return {
        "objects": objects,
        "rooms": rooms,
        "object_to_recep": object_to_recep,
        "object_to_room": object_to_room,
        "recep_to_room": recep_to_room,
        "recep_to_description": recep_to_description,
        "object_to_states": object_to_states,
        "object_to_handle": object_to_handle,
        "recep_to_handle": receptacle_to_handle,
        "room_to_id": room_to_id,
        "instruction": episode["info"]["extra_info"]["instruction"],
    }
