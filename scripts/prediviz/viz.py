#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

import argparse
import gzip
import itertools
import json
import os
import random
import traceback
from collections import defaultdict

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import networkx as nx
from entities.constants import (
    CROPPED_RECEPTACLE_ICONS_PATH,
    FONTS_DIR_PATH,
    RECEPTACLE_ICONS_PATH,
)
from entities.object import Object
from entities.prediviz import PrediViz
from entities.receptacle import Receptacle
from entities.room import Room
from entities.scene import Scene
from omegaconf import OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")


def load_configuration():
    """
    Load configuration from config.yaml file.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "conf/config.yaml"
    )
    return OmegaConf.load(config_path)


def load_episode_data(metadata_dir, episode_id):
    """
    Load episode data from JSON file.
    """
    with open(os.path.join(metadata_dir, f"{episode_id}.json")) as f:
        return json.load(f)


def load_run_data(run_data, episode_id):
    """
    Load run data and retrieve episode data.
    """
    for episode in run_data["episodes"]:
        if episode["episode_id"] == str(episode_id):
            return episode
    return None


def plot_scene(
    config,
    episode_data,
    propositions,
    constraints,
    receptacle_icon_mapping,
    cropped_receptacle_icon_mapping,
    instruction=None,
    save_path=None,
    object_to_recep=None,
    object_to_room=None,
    object_to_states=None,
):
    objects = []
    # Initial Objects and States
    for obj_id in episode_data["object_to_room"]:
        new_obj = Object(config, obj_id)
        if object_to_states is not None and obj_id in object_to_states:
            new_obj.states = object_to_states[obj_id]
            new_obj.previous_states = object_to_states[obj_id].copy()
        objects.append(new_obj)

    rooms = []
    for room_id in episode_data["rooms"]:
        room_receptacles = []
        # Initial Receptacles and States
        for receptacle_id, r_room_id in episode_data["recep_to_room"].items():
            if r_room_id == room_id:
                icon_path = receptacle_icon_mapping.get(
                    receptacle_id, f"{RECEPTACLE_ICONS_PATH}/chair@2x.png"
                )
                new_recep = Receptacle(config, receptacle_id, icon_path)

                # NOTE: Receptacle also have states, but they MIGHT be present in the object_to_states
                if object_to_states is not None and receptacle_id in object_to_states:
                    new_recep.states = object_to_states[receptacle_id]
                    new_recep.previous_states = object_to_states[receptacle_id].copy()
                room_receptacles.append(new_recep)

        # Objects in the room
        room_objects = [
            obj
            for obj in objects
            if episode_data["object_to_room"][obj.object_id] == room_id
        ]
        room = Room(
            config,
            room_id,
            room_receptacles,
            room_objects,
            object_to_recep=object_to_recep,
        )
        rooms.append(room)

    scene = Scene(
        config,
        rooms,
        episode_data["instruction"] if instruction is None else instruction,
        object_to_recep,
        object_to_room,
    )
    prediviz = PrediViz(config, scene)
    result_fig_data = prediviz.plot(
        propositions,
        constraints,
        receptacle_icon_mapping,
        cropped_receptacle_icon_mapping,
        show_instruction=config.show_instruction,
    )
    step_id_to_path_mapping = {}
    for step_idx, (fig, ax, final_height, final_width) in enumerate(result_fig_data):
        width_inches = config.width_inches
        fig.set_size_inches(width_inches, (final_height / final_width) * width_inches)

        plt.sca(ax)
        if config.show_instruction:
            plt.subplots_adjust(right=0.98, left=0.02, bottom=0.02, top=0.95)
        else:
            # tight
            plt.subplots_adjust(right=0.99, left=0.01, bottom=0.01, top=0.99)
        if save_path:
            # Save each step as a separate image
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"step_{step_idx}.png"), dpi=300)
            step_id_to_path_mapping[step_idx] = os.path.join(
                save_path, f"step_{step_idx}.png"
            )
        else:
            fig.show()
        plt.close(fig)
    scene.cleanup()
    del scene
    return step_id_to_path_mapping


def get_episode_data_for_plot(metadata_dir, episode_id, loaded_run_data):
    episode_data = load_episode_data(metadata_dir, episode_id)
    handle_to_recep = {v: k for k, v in episode_data["recep_to_handle"].items()}
    handle_to_object = {v: k for k, v in episode_data["object_to_handle"].items()}
    id_to_room = {v: k for k, v in episode_data["room_to_id"].items()}
    for receptacle_id in episode_data["recep_to_description"]:
        if not os.path.exists(
            f'{RECEPTACLE_ICONS_PATH}/{"_".join(receptacle_id.split("_")[:-1])}@2x.png'
        ):
            raise NotImplementedError(
                f"Missing receptacle asset for receptacle ID: {receptacle_id}"
            )

    receptacle_icon_mapping = {
        receptacle_id: f'{RECEPTACLE_ICONS_PATH}/{"_".join(receptacle_id.split("_")[:-1])}@2x.png'
        for receptacle_id in episode_data["recep_to_description"]
    }
    cropped_receptacle_icon_mapping = {
        receptacle_id: f'{CROPPED_RECEPTACLE_ICONS_PATH}/{"_".join(receptacle_id.split("_")[:-1])}@2x.png'
        for receptacle_id in episode_data["recep_to_description"]
    }
    run_data = load_run_data(loaded_run_data, episode_id)

    propositions = run_data["evaluation_propositions"]
    for proposition in propositions:
        if proposition["function_name"] not in [
            "is_on_top",
            "is_inside",
            "is_on_floor",
            "is_in_room",
            "is_next_to",
            "is_filled",
            "is_powered_on",
            "is_powered_off",
            "is_clean",
        ]:
            raise NotImplementedError(
                f'Not implemented for function_name {proposition["function_name"]}'
            )
        if "object_handles" in proposition["args"]:
            if proposition["function_name"] in [
                "is_clean",
                "is_filled",
                "is_powered_on",
                "is_powered_off",
            ]:
                for handle in proposition["args"]["object_handles"]:
                    if handle in handle_to_recep:
                        if "receptacle_names" not in proposition["args"]:
                            proposition["args"]["receptacle_names"] = []
                        proposition["args"]["receptacle_names"].append(
                            handle_to_recep[handle]
                        )
                    else:
                        if "object_names" not in proposition["args"]:
                            proposition["args"]["object_names"] = []
                        proposition["args"]["object_names"].append(
                            handle_to_object[handle]
                        )
            else:
                proposition["args"]["object_names"] = []
                for object_handle in proposition["args"]["object_handles"]:
                    proposition["args"]["object_names"].append(
                        handle_to_object[object_handle]
                    )

        if "receptacle_handles" in proposition["args"]:
            proposition["args"]["receptacle_names"] = []
            for recep_handle in proposition["args"]["receptacle_handles"]:
                proposition["args"]["receptacle_names"].append(
                    handle_to_recep[recep_handle]
                )

        if "room_ids" in proposition["args"]:
            proposition["args"]["room_names"] = []
            for room_id in proposition["args"]["room_ids"]:
                proposition["args"]["room_names"].append(id_to_room[room_id])
        if "entity_handles_a" in proposition["args"]:
            for entity_index in ["a", "b"]:
                proposition["args"][
                    f"entity_handles_{entity_index}_names_and_types"
                ] = []
                for entity_handle in proposition["args"][
                    f"entity_handles_{entity_index}"
                ]:
                    if entity_handle in handle_to_object:
                        proposition["args"][
                            f"entity_handles_{entity_index}_names_and_types"
                        ].append((handle_to_object[entity_handle], "object"))
                    elif entity_handle in handle_to_recep:
                        proposition["args"][
                            f"entity_handles_{entity_index}_names_and_types"
                        ].append((handle_to_recep[entity_handle], "receptacle"))
                    else:
                        raise ValueError(
                            f"Unknown entity type for handle {entity_handle}. Should be either object or receptacle."
                        )

    # Handle Constraints
    constraints = run_data["evaluation_constraints"]
    for _idx, constraint in enumerate(constraints):
        if constraint["type"] == "TemporalConstraint":
            digraph = nx.DiGraph(constraint["args"]["dag_edges"])
            constraint["toposort"] = [
                sorted(generation) for generation in nx.topological_generations(digraph)
            ]
        elif constraint["type"] == "TerminalSatisfactionConstraint":
            continue
        elif constraint["type"] == "SameArgConstraint":
            same_args = []
            for proposition_index, arg_name in zip(
                constraint["args"]["proposition_indices"],
                constraint["args"]["arg_names"],
            ):
                if arg_name == "object_handles" or arg_name == "receptacle_handles":
                    if arg_name == "object_handles":
                        left_name = "object_names"
                        if (
                            "receptacle_names"
                            in propositions[proposition_index]["args"]
                        ):
                            right_name = "receptacle_names"
                        elif "room_names" in propositions[proposition_index]["args"]:
                            right_name = "room_names"
                        else:
                            raise NotImplementedError(
                                f"Not implemented for `arg_name`: {arg_name} and no receptacle or room names."
                            )
                    elif arg_name == "receptacle_handles":
                        left_name = "receptacle_names"
                        right_name = "object_names"

                    same_args.append(
                        {
                            "common_entities": [
                                (item, left_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    left_name
                                ]
                            ],
                            "corresponding_entities": [
                                (item, right_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    right_name
                                ]
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        "object_names"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                elif arg_name == "entity_handles_a" or arg_name == "entity_handles_b":
                    entity_index = arg_name.split("_")[-1]
                    opposite_entity_index = "b" if entity_index == "a" else "a"
                    same_args.append(
                        {
                            "common_entities": propositions[proposition_index]["args"][
                                f"entity_handles_{entity_index}_names_and_types"
                            ],
                            "corresponding_entities": propositions[proposition_index][
                                "args"
                            ][
                                f"entity_handles_{opposite_entity_index}_names_and_types"
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        f"entity_handles_{entity_index}_names_and_types"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                elif arg_name == "room_ids":
                    right_name = "object_names"
                    same_args.append(
                        {
                            "common_entities": [
                                (item, arg_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    arg_name
                                ]
                            ],
                            "corresponding_entities": [
                                (item, right_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    right_name
                                ]
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        "object_names"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                else:
                    raise NotImplementedError(
                        f"Not implemented SameArg for arg name: {arg_name}"
                    )
            constraint["same_args_data"] = {
                "proposition_indices": constraint["args"]["proposition_indices"],
                "data": same_args,
            }
        elif constraint["type"] == "DifferentArgConstraint":
            diff_args = []
            for proposition_index, arg_name in zip(
                constraint["args"]["proposition_indices"],
                constraint["args"]["arg_names"],
            ):
                if arg_name == "object_handles" or arg_name == "receptacle_handles":
                    if arg_name == "object_handles":
                        left_name = "object_names"
                        if (
                            "receptacle_names"
                            in propositions[proposition_index]["args"]
                        ):
                            right_name = "receptacle_names"
                        elif "room_names" in propositions[proposition_index]["args"]:
                            right_name = "room_names"
                        else:
                            raise NotImplementedError(
                                f"Not implemented for `arg_name`: {arg_name} and no receptacle or room names."
                            )
                    elif arg_name == "receptacle_handles":
                        left_name = "receptacle_names"
                        right_name = "object_names"

                    diff_args.append(
                        {
                            "different_entities": [
                                (item, left_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    left_name
                                ]
                            ],
                            "corresponding_entities": [
                                (item, right_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    right_name
                                ]
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        "object_names"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                elif arg_name == "entity_handles_a" or arg_name == "entity_handles_b":
                    entity_index = arg_name.split("_")[-1]
                    opposite_entity_index = "b" if entity_index == "a" else "b"
                    diff_args.append(
                        {
                            "different_entities": propositions[proposition_index][
                                "args"
                            ][f"entity_handles_{entity_index}_names_and_types"],
                            "corresponding_entities": propositions[proposition_index][
                                "args"
                            ][
                                f"entity_handles_{opposite_entity_index}_names_and_types"
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        f"entity_handles_{entity_index}_names_and_types"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                elif arg_name == "room_ids":
                    right_name = "object_names"
                    diff_args.append(
                        {
                            "different_entities": [
                                (item, arg_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    arg_name
                                ]
                            ],
                            "corresponding_entities": [
                                (item, right_name.split("_")[0])
                                for item in propositions[proposition_index]["args"][
                                    right_name
                                ]
                            ],
                            "line_style": (
                                "dotted"
                                if propositions[proposition_index]["args"]["number"]
                                < len(
                                    propositions[proposition_index]["args"][
                                        "object_names"
                                    ]
                                )
                                else "solid"
                            ),
                            "global_proposition_index": proposition_index,
                        }
                    )
                else:
                    raise NotImplementedError(
                        f"Not implemented SameArg for arg name: {arg_name}"
                    )
            constraint["diff_args_data"] = {
                "proposition_indices": constraint["args"]["proposition_indices"],
                "data": diff_args,
            }
        else:
            raise NotImplementedError(
                f"Constraint type {constraint['type']} is not handled currently."
            )
    return (
        episode_data,
        run_data,
        receptacle_icon_mapping,
        cropped_receptacle_icon_mapping,
        propositions,
        constraints,
    )


def sample_episodes(loaded_run_data, sample_size, metadata_dir):
    """
    Repeatedly sample an episode from each scene without replacement until reaching the
    desired sample size. Calls get_episode_data_for_plot() to validate that the episode
    can be visualized.
    """

    # Group episodes by scene_id
    grouped_episodes = defaultdict(list)
    for ep in loaded_run_data["episodes"]:
        grouped_episodes[ep["scene_id"]].append(ep)

    # Shuffle scene IDs to ensure random order
    scene_ids = list(grouped_episodes.keys())
    random.shuffle(scene_ids)
    shuffled_grouped_episodes = [grouped_episodes[sid] for sid in scene_ids]

    # merge scene episode lists
    shuffled_episodes = []
    for elements in itertools.zip_longest(*shuffled_grouped_episodes):
        shuffled_episodes.extend(filter(lambda x: x is not None, elements))

    # Sample one episode from each scene until reaching the desired sample size
    sampled_eids, idx = [], 0
    while len(sampled_eids) < sample_size:
        selected_episode = shuffled_episodes[idx]
        eid = selected_episode["episode_id"]
        sid = selected_episode["scene_id"]
        try:
            get_episode_data_for_plot(
                metadata_dir,
                eid,
                loaded_run_data,
            )
            sampled_eids.append(eid)
        except Exception as e:
            print(f"[Skipped] sid:{sid} eid:{eid} error: {e}")
            continue

        idx += 1
        if len(sampled_eids) >= sample_size or idx >= len(shuffled_episodes):
            break

    return sampled_eids


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot scene")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the dataset file (.json or .json.gz)",
    )
    parser.add_argument(
        "--metadata-dir",
        required=True,
        type=str,
        help="Directory containing the episode metadata JSON files",
    )
    parser.add_argument(
        "--save-path", required=True, type=str, help="Directory to save the figures"
    )
    parser.add_argument(
        "--episode-id",
        required=False,
        default=None,
        type=int,
        help="Index of episode",
    )
    parser.add_argument(
        "--sample-size",
        required=False,
        type=int,
        default=0,
        help="If only a random subset of all the episodes is to be visualized, the sample size.",
    )
    return parser.parse_args()


def main():
    """
    Main function to plot scenes based on provided arguments.
    """
    args = parse_arguments()
    config = load_configuration()
    font_files = font_manager.findSystemFonts(fontpaths=[FONTS_DIR_PATH])
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    plt.rcParams["font.family"] = "Inter"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["text.color"] = "white"

    if args.dataset.endswith(".gz"):
        with gzip.open(args.dataset, "rt") as f:
            loaded_run_data = json.load(f)
    else:
        with open(args.dataset, "r") as f:
            loaded_run_data = json.load(f)

    if args.episode_id is not None:
        eids = [args.episode_id]
    else:
        if args.sample_size:
            eids = sample_episodes(loaded_run_data, args.sample_size, args.metadata_dir)
        else:
            eids = sorted([int(ep["episode_id"]) for ep in loaded_run_data["episodes"]])

    # Create a dictionary to store run data for episodes with correct visualizations
    run_data_dict = {"config": None, "episodes": []}

    os.makedirs(args.save_path, exist_ok=True)
    for episode_id in tqdm(eids, dynamic_ncols=True):
        try:
            (
                episode_data,
                run_data,
                receptacle_icon_mapping,
                cropped_receptacle_icon_mapping,
                propositions,
                constraints,
            ) = get_episode_data_for_plot(
                args.metadata_dir, episode_id, loaded_run_data
            )

            # Save episode_data as JSON inside the folder
            ep_data_f = os.path.join(args.save_path, f"episode_data_{episode_id}.json")
            with open(ep_data_f, "w") as f:
                json.dump(episode_data, f, indent=4)

            step_id_to_path_mapping = plot_scene(
                config,
                episode_data,
                propositions,
                constraints,
                receptacle_icon_mapping,
                cropped_receptacle_icon_mapping,
                instruction=run_data["instruction"],
                save_path=os.path.join(args.save_path, f"viz_{episode_id}"),
                object_to_recep=episode_data["object_to_recep"],
                object_to_room=episode_data["object_to_room"],
                object_to_states=episode_data.get("object_to_states", None),
            )

            # Add run data for the current episode to the dictionary
            run_data["viz_paths"] = step_id_to_path_mapping
            run_data_dict["episodes"].append(run_data)

            # Save the run data dictionary to a JSON file
            with open(f"{args.save_path}/run_data.json", "w") as f:
                json.dump(run_data_dict, f, indent=4)

        except Exception:
            print(f"Episode ID: {episode_id}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
