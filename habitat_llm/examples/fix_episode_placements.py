#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Tuple, Dict, Any, Union


import omegaconf
import hydra
import magnum as mn
import numpy as np

from collections import defaultdict

from habitat_llm.utils import cprint, setup_config, fix_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.datasets.rearrange.samplers.receptacle import (
    get_scene_rec_filter_filepath,
    get_excluded_recs_from_filter_file,
    Receptacle,
)
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat_llm.utils.sim import find_receptacles
from habitat_sim.physics import ManagedRigidObject, ManagedArticulatedObject
from habitat_sim import Simulator
from habitat_llm.sims.collaboration_sim import CollaborationSim


def to_str_csv(data: Any) -> str:
    """
    Format some data element as a string for csv such that it fits nicely into a cell.
    Currently handles: str, int, float, list[str]

    :param data: Some python object which should be injected into csv.
    """

    if isinstance(data, str):
        # replace any commas with periods to avoid splitting strings into multiple columns
        return data.replace(",", ".")
    if isinstance(data, (int, float)):
        return str(data)
    if isinstance(data, list):
        # lists are separated by ';' and concatenated to fit into a cell.
        list_str = ""
        for elem in data:
            list_str += f"{to_str_csv(elem)};"
        return list_str

    raise NotImplementedError(
        f"Data type {type(data)} is not yet supported in to_str_csv."
    )


def export_results_csv(
    filepath: str, results_dict: Dict[str, Dict[str, Dict[str, Any]]]
) -> None:
    """
    Export the results dictionary from running the ep_validator util as a csv.

    Example hierarchy of labels:
    episode_id -> operation -> message class (e.g. "error") -> list(info strings)

    :param filepath: The output filepath. Must be a .csv
    :param results_dict: The results dictionary from the application.
    """

    assert filepath.endswith(".csv")

    with open(filepath, "w") as f:
        # first write the column labels
        f.write("episode id, operation, message class: message\n")

        # now a row for each scene
        for episode_id, operation_dict in results_dict.items():
            # write the scene column
            f.write(f"{episode_id},\n")
            for operation, message_class_dict in operation_dict.items():
                f.write(f",{operation},\n")
                for message_class, messages_dict in message_class_dict.items():
                    for message_tag, message in messages_dict.items():
                        f.write(
                            f",,{message_class}:{message_tag}:{to_str_csv(message)}\n"
                        )

    print(f"Wrote results csv to {filepath}")


def draw_receptacles(sim: Simulator, receptacles: List[Receptacle]) -> None:
    """
    Debug draw callback for dbv to render the provided Receptacles.

    :param sim: The Simulator instance.
    :param receptacles: A list of Receptacle objects to draw.
    """
    scene_filter_file = get_scene_rec_filter_filepath(
        sim.metadata_mediator, sim.curr_scene_name
    )
    filter_strings = get_excluded_recs_from_filter_file(scene_filter_file)
    for rec in receptacles:
        color = mn.Color4.green()
        if rec.unique_name in filter_strings:
            color = mn.Color4.red()
        rec.debug_draw(sim, color=color)


def dbv_with_relevant_recs(
    dbv: DebugVisualizer,
    obj: Union[ManagedRigidObject, ManagedArticulatedObject],
    sim: CollaborationSim,
    receptacles: Dict[str, Receptacle],
    intended_parent_handle: str,
    show: bool = True,
    save_to_dir: str = None,
) -> None:
    """
    Shortcut helper function to quickly render a dbv "peek" image of any object along with debug line representations of any Receptacles attached to a relevant parent Furniture.

    :param dbv: The DebugVisualizer instance.
    :param obj: The ManagedObject to peek.
    :param sim: The Simulator instance.
    :param receptacles: The available Receptacles provided as a dict mapping unique name to Receptacle object.
    :param intended_parent_handle: The handle string of the parent object for which to display the Receptacles.
    :param show: Whether or not to immediately open the image.
    :param save_to_dir: Optionally save the image to a provided filepath.
    """

    if not show and save_to_dir is None:
        # no-op
        return

    # get all Recepacles from the provided set which have the provided parent
    relevant_recs = [
        rec
        for rec in receptacles.values()
        if rec.parent_object_handle == intended_parent_handle
    ]
    # setup the callback
    dbv.dblr_callback = draw_receptacles
    dbv.dblr_callback_params = {
        "sim": sim,
        "receptacles": relevant_recs,
    }
    # do the peek
    dbo = dbv.peek(obj, peek_all_axis=True)
    # show or save the image
    if show:
        dbo.show()
    if save_to_dir is not None:
        dbo.save(
            output_path=save_to_dir,
            prefix=f"ep({sim.ep_info.episode_id})_obj({obj.handle})_",
        )


def try_resnap(
    sim: Simulator,
    obj: Union[ManagedRigidObject, ManagedArticulatedObject],
    target_rec: Receptacle,
    max_collision_depth: float = 0.01,
    y_offset: float = 0.08,
    new_samples: int = 0,
) -> Tuple[bool, float]:
    """
    Try to re-snap the object in-place by pushing it up a bit and then snapping it back down.
    NOTE: This could help to address thin object issues or cases where the lowest extent of the object is embedded slightly in a support surface.

    :param sim: The Simulator instance.
    :param obj: The ManagedObject instance.
    :param target_rec: The Receptacle onto which the object should be snapped.
    :param max_collision_depth: The maximum allowed collision depth between the object and the world.
    :param y_offset: The amount of vertical offset to apply to the object's current state before attempting to snap.
    :param new_samples: Maximum number of attempts to sample new candidate positions on the Receptacle, effectively searching for a new placement location.

    :return: binary success flag and L2 offset realized by the re-snap
    """

    original_translation = obj.translation
    if new_samples < 0:
        raise ValueError("Cannot provide a negative number of samples.")

    this_iter = 0
    # manually setting maximum 10 new sample attempts
    num_iter = 1 + new_samples
    # first push the object up
    obj.translation = obj.translation + mn.Vector3(0, y_offset, 0)
    support_obj_ids = target_rec.get_support_object_ids(sim)

    while this_iter < num_iter:
        if this_iter > 0:
            # sample a new position
            obj.translation = target_rec.sample_uniform_global(sim, 1) + mn.Vector3(
                0, y_offset, 0
            )
        # snap the object down
        snap_success = sutils.snap_down(
            sim, obj, support_obj_ids, max_collision_depth=max_collision_depth
        )
        resulting_offset = (obj.translation - original_translation).length()
        this_iter += 1
        if not snap_success:
            # snapping failed, so reset the object state
            obj.translation = original_translation
        else:
            # the object remains in the successfully snapped position
            break
    print(
        f"Re-snap {obj.handle}: {snap_success}, {resulting_offset}, support_surface_ids={support_obj_ids}"
    )
    return snap_success, resulting_offset


def update_obj_ep_state_and_save(
    sim: CollaborationSim,
    obj: Union[ManagedRigidObject, ManagedArticulatedObject],
    dataset: CollaborationDatasetV0,
    obj_initial_transform: mn.Matrix4,
    new_rec: str = None,
    dataset_output_filepath: str = "new_dataset.json.gz",
) -> None:
    """
    Update the given object's state in the current CollaborationEpisode within the CollaborationDataset and then saves the dataset.

    :param sim: The CollaborationSim instance. Requires access to ep_info.
    :param obj: The ManagedObject instance.
    :param dataset: The episode dataset to modify.
    :param obj_initial_transform: The initial transform of the object. Used to match the object to one of possibly multiple initial entries in the current Episode metadata.
    :param new_rec: Optionally provide a new receptacle correspondence for the object. If provided, overwrites the object->rec map entry.
    :param dataset_output_filepath: The output filepath to save the modified dataset.
    """

    if not dataset_output_filepath.endswith(".json.gz"):
        raise ValueError("Output filepath for dataset must be .json.gz.")

    obj_name = sutils.object_shortname_from_handle(obj.handle)

    # get index and rigid state info so we can inject the fix in the same order
    matching_inits = [
        (ix, tup)
        for (ix, tup) in enumerate(sim.ep_info.rigid_objs)
        if obj_name in tup[0]
    ]
    if len(matching_inits) == 0:
        print(f"Error: no match for object {obj.handle} in episode inits.")
        return
    # if multiple matches exist, find the closest one
    best_match: Tuple[Any, Any, int] = (None, None, None)
    for ix, candidate in matching_inits:
        transform = candidate[1]
        mn_transform = mn.Matrix4(
            [[transform[j][i] for j in range(4)] for i in range(4)]
        )
        candidate_dist = (
            obj_initial_transform.translation - mn_transform.translation
        ).length()
        print(f"candidate_dist = {candidate_dist}")
        if best_match[0] is None or candidate_dist < best_match[1]:
            best_match = candidate, candidate_dist, ix
            print(f"candidate = {candidate}, {ix}")
    if best_match[1] > 0.15:
        print(
            f"Error: best match candidate for {obj.handle} in episode inits {candidate} is far away from current object: {best_match[1]}."
        )

    # replace the candidate with the updated transform
    new_rigid_objs = sim.ep_info.rigid_objs[: best_match[2]]
    new_rigid_objs.append((best_match[0][0], np.array(obj.transformation)))
    if best_match[2] + 1 < len(sim.ep_info.rigid_objs):
        new_rigid_objs.extend(sim.ep_info.rigid_objs[best_match[2] + 1 :])

    if new_rec is not None:
        # modify the object->receptacle mapping
        sim.ep_info.name_to_receptacle[obj.handle] = new_rec

    # inject the modified episode into the dataset
    sim.ep_info.rigid_objs = new_rigid_objs
    for ix, ep in enumerate(dataset.episodes):
        if ep.episode_id == sim.ep_info.episode_id:
            new_episodes = dataset.episodes[:ix]
            new_episodes.append(sim.ep_info)
            if ix + 1 < len(dataset.episodes):
                new_episodes.extend(dataset.episodes[ix + 1 :])
            dataset.episodes = new_episodes
            print("found and replaced episode in dataset")
            break

    # serialize the dataset
    import gzip

    with gzip.open(dataset_output_filepath, "wt") as f:
        f.write(dataset.to_json())
    print(f"saved dataset to {dataset_output_filepath}")


def fix_placement(
    sim: CollaborationSim,
    obj: Union[ManagedRigidObject, ManagedArticulatedObject],
    active_set: List[Receptacle],
    match_set: List[Receptacle],
    dataset: CollaborationDatasetV0,
    dataset_output_filepath: str = "",
):
    """
    Utility function to attempt to "fix" poor or invalid initial Episode placement for an object.

    :param sim: The CollaborationSim instance. Requires access to ep_info.
    :param obj: The ManagedObject instance.
    :param active_set: The set of all active Receptacles.
    :param match_set: The set of all Receptacles (including inactive) which this object was matched to. Used to find related active Receptacles (e.g. on matched furniture) as alternative placement locations.
    :param dataset: The CollaborationDatasetV0 object which should be modified if the fix is successful.
    :param dataset_output_filepath: Optionally provide a custom output path for the dataset.
    """
    # cache the initial transformation
    obj_initial_transform = obj.transformation

    # we have active matches, so try to simply fix the placement on the currently matched Receptacles
    for active_option in active_set:
        # first try re-snapping the current position to the currently matched Receptacle
        success, dist = try_resnap(sim, obj, sim.receptacles[active_option])
        if success:
            cprint("PLACEMENT FIX", color="yellow")
            update_obj_ep_state_and_save(
                sim,
                obj,
                dataset,
                obj_initial_transform,
                active_option,
                dataset_output_filepath=dataset_output_filepath,
            )
            # dbv_with_relevant_recs(active_option.split("|")[0])
            return True
        else:
            # then try sampling a new placement on the currently matched Receptacle
            success, dist = try_resnap(
                sim,
                obj,
                sim.receptacles[active_option],
                new_samples=10,
            )
            if success:
                cprint("PLACEMENT FIX-RESAMPLE", color="yellow")
                update_obj_ep_state_and_save(
                    sim,
                    obj,
                    dataset,
                    obj_initial_transform,
                    active_option,
                    dataset_output_filepath=dataset_output_filepath,
                )
                # dbv_with_relevant_recs(active_option.split("|")[0])
                return True

    # reaching here there is no placement fix with active matches
    # if some inactive matches are provided
    if match_set is not None and len(match_set) > 0:
        # identify any alternative active Receptacles attached to the same parent Furniture as any Receptacle the object was matched to
        active_alternatives = []
        for match in match_set:
            print("Trying to other recs on match furniture.")
            parent_instance = match.split("|")[0]
            active_options = [
                mrec for mrec in sim.receptacles if parent_instance in mrec
            ]
            active_alternatives.extend(active_options)
        # deduplicate
        active_alternatives = list(set(active_alternatives))
        if len(active_alternatives) == 0:
            cprint(
                "!!NO PLACEMENT FIX - no active alternatives to inactive rec matches for the object!!",
                color="red",
            )
            return False
        # recursively try to snap onto the alternatives
        if fix_placement(
            sim,
            obj,
            active_alternatives,
            None,
            dataset,
        ):
            return True

        cprint(
            "!!NO PLACEMENT FIX - snapping found no valid placements for active alternatives to inactive rec matches for the object!!",
            color="red",
        )
        return False
    elif len(active_set) == 0:
        cprint("!!NO PLACEMENT FIX - no rec matches for the object!!", color="red")
        return False

    # nothing worked
    cprint("!!NO PLACEMENT FIX - nothing worked, manual fix!!", color="red")
    return False


def correct_obj_rec_inits(
    sim: CollaborationSim,
    dataset: CollaborationDatasetV0,
    validator_correction_level: int = 3,
    print_results: bool = True,
    output_dir: str = "",
    show_and_wait: bool = False,
):
    """
    Check that clutter object initializations in the episode are valid and optionally attempt to correct them:
    - receptacles exist
    - receptacles are active
    - AO receptacles are in the default_link

    :param sim: The CollaborationSim instance. Requires access to ep_info.
    :param dataset: The CollaborationDatasetV0 which should be modified and re-saved if corrections are made.
    :param validator_correction_level: determines how far we'll go to seek a valid state (0 - no corrections, only validate), (1 - only in-place corrections, no re-associate to new Receptacles), (2 - limit re-association to same parent Furniture), (3 - all re-association allowed)
    :param print_results: Optionally print the results of each Episode iteration.
    :param output_dir: Optionally provide a custom output directory.
    :param show_and_wait: If true, show debug images when corrections fail and enter a breakpoint.
    """

    if validator_correction_level not in [0, 1, 2, 3]:
        raise ValueError("validator_correction_level must be in the range [0,3].")

    os.makedirs(output_dir, exist_ok=True)
    dataset_output_filepath = os.path.join(output_dir, "new_dataset.json.gz")

    # cache a dictionary with issue details
    # NOTE: top level key is an issue category string
    issue_info: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(lambda: [])
    )
    issue_info["info"]
    issue_info["warning"]
    issue_info["error"]

    # initialize a dbv instance
    dbv = DebugVisualizer(sim)

    # get all Receptacles in the scene including those filtered out for any reason
    unfiltered_receptacles = {
        rec.unique_name: rec for rec in find_receptacles(sim, filter_receptacles=False)
    }

    # for each object->rec match in the episode
    for (
        obj_handle,
        rec_unique_name,
    ) in sim.ep_info.name_to_receptacle.items():
        if print_results:
            cprint(f"   Checking {obj_handle} -> {rec_unique_name}", color="blue")
        # first check the objects and Receptacles exist
        obj = sutils.get_obj_from_handle(sim, obj_handle)
        if obj is None:
            # abort because the provided object handle doesn't exist
            issue_info["error"]["missing_objects"].append(obj_handle)
            continue

        # find the best rec matches on the full unfiltered set
        matched_rec_names, confidence, info = sutils.get_obj_receptacle_and_confidence(
            sim,
            obj,
            candidate_receptacles=unfiltered_receptacles,
            island_index=sim._largest_indoor_island_idx,
        )
        # limit matches to active receptacles
        active_matches = [mrec for mrec in matched_rec_names if mrec in sim.receptacles]

        if rec_unique_name.startswith("floor"):
            # floor Receptacles don't really exist, they are placeholders in the CollaborationEpisode
            issue_info["info"]["floor_recs"].append(f"{obj_handle}->{rec_unique_name}")
            if len(matched_rec_names) > 0 and "floor" in matched_rec_names[0]:
                # this object is correctly matched to a floor Receptacle
                # NOTE: this check is needed because exact string name of floor Receptacles may not be the same between get_obj_receptacle_and_confidence and CollaborationEpisode metadata
                issue_info["warning"]["floor_rec_match"].append(
                    f"'{obj_handle}' matched to '{matched_rec_names[0]}' vs. target '{rec_unique_name}'"
                )
            else:
                # should be matched to floor but isn't, record a failure, no fix
                issue_info["error"]["matching_receptacles"].append(
                    f"('{obj_handle}' matched to '{matched_rec_names}'). Should be '{rec_unique_name}'"
                )
            continue

        target_rec = sim.receptacles.get(rec_unique_name, None)
        if target_rec is None:
            # Receptacle referenced doesn't exist in active set
            if rec_unique_name in unfiltered_receptacles:
                # Receptacle referenced is filtered, we'll try to fix it later
                issue_info["error"]["filtered_init_recs"].append(rec_unique_name)
            else:
                # Receptacle referenced does not exist anywhere, we'll try to match to something else
                issue_info["error"]["missing_recs"].append(rec_unique_name)
        elif target_rec.unique_name in active_matches:
            # Correctly matched to the target which is active
            continue
        else:
            # record that the reference Receptacle was not matched
            issue_info["error"]["matching_receptacles"].append(
                f"('{obj_handle}' matched to '{matched_rec_names}'). Should be '{rec_unique_name}'"
            )
            if print_results:
                print(issue_info["error"]["matching_receptacles"][-1])

        if validator_correction_level == 0:
            # not correcting anything, so we're done
            continue

        # 1) the expected rec is active, but not matched. Try to resnap to it.
        if validator_correction_level >= 1 and target_rec is not None:
            obj_initial_transform = obj.transformation
            success, _dist = try_resnap(
                sim,
                obj,
                target_rec,
                new_samples=10,
            )
            if success:
                cprint("RESNAP-SUCCEEDED", color="yellow")
                # found a valid placement, save out the modified obj state in the dataset
                update_obj_ep_state_and_save(
                    sim,
                    obj,
                    dataset,
                    obj_initial_transform,
                    dataset_output_filepath=dataset_output_filepath,
                )
                continue

        intended_parent = rec_unique_name.split("|")[0]
        # find alternative Receptacles on the same Furniture as the reference Receptacle
        active_parent_alternatives = [
            recn for recn in sim.receptacles if intended_parent in recn
        ]

        # 2) try to use other receptacles on the intended parent
        if (
            validator_correction_level >= 2
            and len(active_parent_alternatives) > 0
            and fix_placement(
                sim,
                obj,
                active_parent_alternatives,
                active_parent_alternatives,
                dataset,
                dataset_output_filepath=dataset_output_filepath,
            )
        ):
            continue

        if len(matched_rec_names) == 0:
            # no matching receptacles are found, including the intended reference Receptacle
            issue_info["error"]["matching_receptacles"].append(
                f"('{obj_handle}'->'NONE'): {info}"
            )
            if print_results:
                cprint(issue_info["error"]["matching_receptacles"][-1], "red")

        # 3) try to re-associate to use any other active matches or active recs on matched furniture
        elif validator_correction_level == 3 and fix_placement(
            sim,
            obj,
            active_matches,
            matched_rec_names,
            dataset,
            dataset_output_filepath=dataset_output_filepath,
        ):
            continue

        # everything failed, save a debug image
        dbv_with_relevant_recs(
            dbv,
            obj,
            sim,
            receptacles=unfiltered_receptacles,
            intended_parent_handle=rec_unique_name.split("|")[0],
            show=show_and_wait,
            save_to_dir=output_dir,
        )
        if show_and_wait:
            breakpoint()

    # clean up the dbv
    dbv.remove_dbv_agent()

    # optionally print the full issue_info dict for the Episode after validation is complete
    if print_results:
        cprint("    -- Check object receptacle inits results --", color="blue")
        # print the issue_info dict as each iteration is complete
        for key, info in issue_info.items():
            text_color = "gray"
            if key == "warning":
                text_color = "yellow"
            elif key == "error":
                text_color = "red"

            for key2, val in info.items():
                cprint(f"   {key}:{key2}", color=text_color)
                for thing in val:
                    cprint(f"   - {thing}", color=text_color)
        cprint("    -- End check object receptacle inits results --", color="blue")

    # return the issue_info dict for batched logging
    return issue_info


# Method to load agent planner from the config
@hydra.main(
    config_path="../conf", config_name="examples/skill_runner_default_config.yaml"
)
def fix_episode_placements(config: omegaconf.DictConfig) -> None:
    """
    The main function for executing the episode validator/fixer tool. A default config is provided.
    See the `main` function for example CLI command to run the tool.

    :param config: input is a habitat-llm config from Hydra. Can contain CLI overrides.
    """

    fix_config(config)
    # Setup a seed
    seed = 47668090
    # Setup some hardcoded config overrides (e.g. the metadata path)
    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict
    config = setup_config(config, seed)

    assert config.env == "habitat", "Only valid for Habitat skill testing."

    if not config.evaluation.save_video:
        remove_visual_sensors(config)

    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # We register custom measures
    register_measures(config)

    # create the dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")

    assert not (
        hasattr(config, "validator_episode_indices")
        and hasattr(config, "validator_episode_ids")
    ), "Episode selection options are mutually exclusive."

    # Collect any configured validator parameters
    ##################################################################

    # correction level determines how far we'll go to seek a valid state
    # 0 - no corrections, only validate
    # 1 - only in-place corrections, no re-associate to new Receptacles
    # 2 - limit re-association to same parent Furniture
    # 3 - all re-association allowed
    validator_correction_level = config.get("validator_correction_level", 3)

    # customize the output path for csv and images
    validator_output_path = config.get("validator_output_path", "episode_fixer_out/")

    # configure whether or not to show images and enter a breakpoint when something goes wrong
    show_and_wait = config.get("validator_show_and_wait", False)

    ##################################################################

    # Create a new dataset from the specified subset of episodes
    ##################################################################
    episode_subset = list(dataset.episodes)
    if hasattr(config, "validator_episode_indices"):
        # scrape from indices
        episode_indices = config.validator_episode_indices
        assert len(list(set(episode_indices))) == len(
            episode_indices
        ), "Duplicates detected in index list input, aborting."
        episode_subset = [
            ep for ix, ep in enumerate(dataset.episodes) if ix in episode_indices
        ]
        assert len(episode_subset) == len(
            episode_indices
        ), f"Could not find all requested indices, missing {list(set(episode_indices)-set(range(len(dataset.episodes))))}"
    elif hasattr(config, "validator_episode_ids"):
        # scrape from ids
        episode_ids = [str(eid) for eid in config.validator_episode_ids]
        assert len(list(set(episode_ids))) == len(
            episode_ids
        ), "Duplicates detected in id list input, aborting."
        episode_subset = [ep for ep in dataset.episodes if ep.episode_id in episode_ids]
        assert len(episode_subset) == len(
            episode_ids
        ), f"Could not find all requested indices, missing {list(set(episode_ids)-{ep.episode_id for ep in episode_subset})}"
    subset_dataset = CollaborationDatasetV0(
        config=config.habitat.dataset, episodes=episode_subset
    )
    print(f"Working on episodes {[ep.episode_id for ep in subset_dataset.episodes]}")
    ##################################################################

    # setup the validator operations
    ##################################################################
    assert hasattr(
        config, "validator_operations"
    ), "Must include a list of operations for validator."
    # NOTE: this is the list of implemented options
    available_operations = ["ep_obj_rec_inits"]
    active_operations = config.validator_operations
    for candidate_operation in active_operations:
        assert (
            candidate_operation in available_operations
        ), f"Requested validator operation {candidate_operation} is not valid. Must select operations from the implemented list: {available_operations}"
    ##################################################################

    # Initialize the environment interface for the agent
    env_interface = EnvironmentInterface(config, dataset=subset_dataset)
    sim: CollaborationSim = env_interface.sim

    # initialize the results log
    validation_results: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    # run the validation process for each episode
    no_error_ep_indices = []
    error_ep_indices = []
    for _epit in range(len(subset_dataset.episodes)):
        cprint("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", color="green")
        cprint(
            f"Evaluating Episode '{sim.ep_info.episode_id}' ({_epit}|{len(subset_dataset.episodes)})",
            color="green",
        )
        validation_results[sim.ep_info.episode_id] = {}
        # do validation checks here
        if "ep_obj_rec_inits" in active_operations:
            validation_results[sim.ep_info.episode_id][
                "ep_obj_rec_inits"
            ] = correct_obj_rec_inits(
                sim,
                dataset,
                validator_correction_level,
                output_dir=validator_output_path,
                show_and_wait=show_and_wait,
            )

        if (
            len(validation_results[sim.ep_info.episode_id]["ep_obj_rec_inits"]["error"])
            == 0
        ):
            no_error_ep_indices.append(_epit)
        else:
            error_ep_indices.append(_epit)

        cprint(
            f"Done evaluating Episode '{sim.ep_info.episode_id}' ({_epit}|{len(subset_dataset.episodes)})",
            color="green",
        )
        cprint("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", color="green")
        # initialize the next episode
        env_interface.reset_environment()

    # save the validation results in a csv file
    export_results_csv(
        os.path.join(validator_output_path, "validator_results.csv"), validation_results
    )

    # identify the original indices of all validated episodes with and without errors
    no_error_eo_indices_orig = []
    error_eo_indices_orig = []
    for ix in no_error_ep_indices:
        eid = subset_dataset.episodes[ix].episode_id
        orig_ix = [i for i, ep in enumerate(dataset.episodes) if ep.episode_id == eid][
            0
        ]
        no_error_eo_indices_orig.append(orig_ix)
    for ix in error_ep_indices:
        eid = subset_dataset.episodes[ix].episode_id
        orig_ix = [i for i, ep in enumerate(dataset.episodes) if ep.episode_id == eid][
            0
        ]
        error_eo_indices_orig.append(orig_ix)
    formatted_no_error_list_str = str(no_error_eo_indices_orig).replace(" ", "")
    formatted_error_list_str = str(error_eo_indices_orig).replace(" ", "")
    print("Finished all episodes.")
    print(f"Indices with no errors: {formatted_no_error_list_str}")
    # NOTE: use this list like "+validator_episode_indices=<paste this list>" to iterate only on the erroneous (and potentially fixed) episodes
    print(f"Indices with errors: {formatted_error_list_str}")


##########################################
# CLI Example:
# HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.fix_episode_placements hydra.run.dir="."
#
##########################################
# Script Specific CLI overrides:
#
# (mutually exclusive)
# - '+validator_episode_indices=[0]' - initialize the episode(s) with the specified indices within the dataset (indices are integers)
# - '+validator_episode_ids=[]' - initialize the episode(s) with the specified "ids" within the dataset (ids are strings)
#
# (validator operations)
# - '+validator_operations=[]' - the set of validation operations to execute on the episodes
#   - options are:
#     - 'ep_obj_rec_inits' - validate that episode object initializations are sound: receptacles exist, are active, are in default links
# - '+validator_show_and_wait=true' - show debug images at key events and enter a breakpoint for further analysis
# - '+validator_output_path="<desired output directory>"' - set an output directory. If not set, default = "ep_validator_output/".
# - '+validator_correction_level=0' - (default 3) Use this option to limit correction modes or only record failures. Options: (0="no corrections", 1="in-place corrections, no re-association to new Receptacles", 2="same furniture re-associations only", 3="full re-association").
##########################################
# Other useful CLI overrides:
#
# - 'habitat.dataset.data_path="<path to dataset .json.gz>"' - set the desired episode dataset
#
if __name__ == "__main__":
    cprint(
        "\nStart of the example program to validate and correct episode placements in a CollaborationDataset.",
        "blue",
    )

    # Run the skills
    fix_episode_placements()

    cprint(
        "\nEnd of the example program to validate and correct episode placements in a CollaborationDataset.",
        "blue",
    )
