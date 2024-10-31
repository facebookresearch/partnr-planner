#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import random
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
import magnum as mn
from habitat.core.logging import logger
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat.datasets.rearrange.samplers.object_sampler import ObjectSampler
from habitat.datasets.rearrange.samplers.scene_sampler import SingleSceneSampler
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import ManagedRigidObject

warnings.filterwarnings("ignore")

import numpy as np
from habitat.datasets.rearrange.rearrange_generator import RearrangeEpisodeGenerator
from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
from habitat.datasets.rearrange.samplers.receptacle import (
    Receptacle,
    ReceptacleSet,
    ReceptacleTracker,
)
from habitat.sims.habitat_simulator.debug_visualizer import (
    DebugObservation,
    DebugVisualizer,
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0, CollaborationEpisode
from habitat_llm.sims.collaboration_sim import initialize_object_state_machine
from habitat_llm.sims.metadata_interface import MetadataInterface, default_metadata_dict


def merge_dicts(dict1, dict2):
    """
    Recursively merges dict2 into dict1.
    :param dict1: The first dictionary to be merged.
    :param dict2: The second dictionary to be merged. Values from dict2 will overwrite values in dict1.
    :return: The merged dictionary.
    """
    for key in dict2:
        if (
            key in dict1
            and isinstance(dict1[key], dict)
            and isinstance(dict2[key], dict)
        ):
            merge_dicts(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]


class LLMRearrangeEpisodeGenerator(RearrangeEpisodeGenerator):
    """
    Extended RearrangeEpisodeGenerator which provides LLM accessible APIs for generating valid episodes from prompts.

    Leverages a Simulator
    Initialized from a JSON defining dataset and object asset paths.
    Exposes samplers, object sets, receptacle sets, scene sets to construction and manipulation from semantic information.
    Produces CollaborationEpisodes incrementally with interactive calls.
    """

    def __init__(
        self,
        cfg: Dict[Any, Any],
        metadata_interface: "MetadataInterface",
        debug_visualization: bool = False,
    ) -> None:
        """
        Initialize the generator object with a blank slate, loading only the scene metadata.

        :param cfg: The JSON config pre-parsed into a Dict.
        :param metadata_interface: The MetadataInterface object containing additional object metadata for mapping open language to specific asset instances.
        :param debug_visualization: Whether or not to generate debug images and videos.
        """

        # fill some fields of the generator config to minimally initialize the base class
        self.llm_cfg = cfg  # this is the JSON
        self.metadata_interface = metadata_interface
        self.cfg = (
            get_config_defaults()
        )  # this is the empty/default RearrangeEpisodeGeneratorConfig
        self.cfg.dataset_path = cfg["scene_dataset"]
        self.cfg.additional_object_paths = cfg["additional_object_paths"]
        if "enable_check_obj_stability" in cfg:
            self.cfg.enable_check_obj_stability = cfg["enable_check_obj_stability"]
        if "agent_max_slope" in cfg:
            self.cfg.agent_max_slope = cfg["agent_max_slope"]
        if "agent_max_climb" in cfg:
            self.cfg.agent_max_climb = cfg["agent_max_climb"]
        # NOTE: Any custom navmesh params should be set here
        self._limit_scene_set = None
        self.largest_indoor_island_id = -1
        self.ao_link_map = None

        super().__init__(self.cfg, debug_visualization=debug_visualization)

        # NOTE: later set self._scene_sampler.scene = scene_id
        self._scene_sampler = SingleSceneSampler("NONE")

        # NOTE: the following local state structures track content added to an in-progress episode generation
        # Call self.finalize_episode() to run post-process and retrieve a CollaborationEpisode
        self.ep_scene_handle = None
        self.ao_states: Dict[str, Dict[int, float]] = {}

        # target and goal receptacles are pre-determined locations where the target object will be initialized and rearranged to respectively
        # NOTE: these likely won't be used in this open-ended generator, but must be tracked for compatibility with the base class
        self.all_target_receptacles: List[Receptacle] = []
        self.all_goal_receptacles: List[Receptacle] = []

        self.episode_data: Dict[str, Dict[str, Any]] = {}
        self.object_to_containing_receptacle: Dict[str, Receptacle] = {}
        self.target_refs: Dict[
            str, str
        ] = {}  # maps target instance name to a unique id: "sampler_name|target_index"

        # dict to log details during a generation attempt in order to retro-actively analyze the results
        self.generation_details_log: Dict[str, Any] = {}
        # each generation pass is indexed
        self.phase = 0

    def reset_epsiode_state(self):
        """
        Reset the structures tracking content added to the current episode.
        """

        self.ep_scene_handle = None
        self.ao_states = {}
        self.ep_sampled_objects: List[
            habitat_sim.physics.ManagedRigidObject
        ] = []  # initialize in base class
        self.all_target_receptacles = []
        self.all_goal_receptacles = []
        self.episode_data: Dict[str, Dict[str, Any]] = {
            "sampled_objects": {},  # object sampler name -> sampled object instances
            "sampled_targets": {},  # (instance_handle: target state)
            "object_states": {},  # {state_name: {object_instance_handle: state_value}}
            "info": {},  # Dict[str,Dict[Any,Any]] - put what you want here
        }
        self.object_to_containing_receptacle = {}
        self.target_refs = {}
        self.generation_details_log = {}
        self.phase = 0

    def get_and_validate_config_sample_number(
        self, config: Dict[str, Any]
    ) -> Tuple[int, int, int]:
        """
        Parse and validate the "number" key in a phase config dict.
        The value of the "number" entry specifies how many items to sample from a set.

        :param config: The sample config entry which should contain the "number" entry to parse.

        :return: A Tuple containing the min, max, target for an integer range. Downstream, 'target' will be attempted, but any more than 'min' will still succeed. If fails, return (None, None, None), this should be handled correctly downstream.
        """

        #####################################
        # get number of objects to sample
        if "number" not in config:
            self.generation_details_log[
                "failure_mode"
            ] = "invalid config, must provide number of objects to sample"
            return None, None, None

        number_range: Tuple[int, int] = None
        target_number: int = None

        if isinstance(config["number"], list):
            assert (
                len(config["number"]) > 1 and len(config["number"]) < 4
            ), "If using a list for sample number, provide a range of 2 or 3 values: [min, max] or [min, target, max]."
            p_val = 1
            for val in config["number"]:
                assert isinstance(val, int), "Sample numbers must be integers."
                assert val >= p_val, "Number range list must be sorted ascending."
                p_val = val
            number_range = (config["number"][0], config["number"][-1])
            if len(config["number"]) == 3:
                target_number = config["number"][1]
            else:
                target_number = random.randint(number_range[0], number_range[1])
        else:
            if isinstance(config["number"], str):
                assert config[
                    "number"
                ].isnumeric(), "Single sample 'number' must be an integer."
                target_number = int(config["number"])
            elif isinstance(config["number"], int):
                target_number = config["number"]
            number_range = (target_number, target_number)

        if number_range is None or target_number is None:
            self.generation_details_log[
                "failure_mode"
            ] = "invalid config, 'number' is not correctly formatted."
            return None, None, None
        elif target_number <= 0:
            self.generation_details_log[
                "failure_mode"
            ] = "invalid config, 'number' must be greater than zero."
            return None, None, None

        # [arse succeeded, return the result
        return number_range[0], number_range[1], target_number

    def get_allowed_regions(
        self, region_config: List[str], phase_details: Dict[str, Any]
    ) -> Tuple[List[habitat_sim.scene.SemanticRegion], List[int]]:
        """
        Get the SemanticRegions matching a set of input string keys from a generator phase config.
        E.g. "allowed_regions": ["kitchen"], #only furniture in a kitchen
        E.g. "allowed_regions": ["kitchen_2"], #only furniture in kitchen_2

        :param region_config" The list of regions found in the "allowed_regions" dict value.
        :param phase_details" The 'phase_details' dict for tracking errors/warnings.

        :return: A Tuple of lists with matching regions 1) the SemanticRegions 2) the indices. If empty, something went wrong.
        """

        allowed_regions: List[habitat_sim.scene.SemanticRegion] = []
        allowed_region_ixs: List[int] = []
        found_regions = []
        # split region instances (end in "_<int>") from categories
        allowed_instances = [
            r_name for r_name in region_config if r_name.split("_")[-1].isnumeric()
        ]
        for r_name in allowed_instances:
            if r_name in self.metadata_interface.region_semname_to_id:
                allowed_region_ixs.append(
                    self.metadata_interface.region_semname_to_id[r_name]
                )
                allowed_regions.append(
                    self.sim.semantic_scene.regions[allowed_region_ixs[-1]]
                )
            else:
                print(
                    f"Unable to find region instance {r_name} specified in config. Could be hallucinated?"
                )
                if "unfound_regions" not in phase_details:
                    phase_details["unfound_regions"] = []
                phase_details["unfound_regions"].append(r_name)

        # handle remaining items as categories
        allowed_categories = list(set(region_config).difference(allowed_instances))
        for rix, region in enumerate(self.sim.semantic_scene.regions):
            found_regions.append(region.category.name())
            if region.category.name() in allowed_categories:
                allowed_regions.append(region)
                allowed_region_ixs.append(rix)

        # record categories which could not be found
        unfound_categories = list(set(allowed_categories).difference(found_regions))
        if len(unfound_categories) > 0:
            if "unfound_regions" not in phase_details:
                phase_details["unfound_regions"] = []
            phase_details["unfound_regions"].extend(unfound_categories)

        return allowed_regions, allowed_region_ixs

    def get_next_phase_details(self) -> Dict[str, Any]:
        """
        Increments the phase counter and returns the next 'phase_details' dictionary.
        Should be called only at the start of each new phase of generation.
        The 'phase_details' dict tracks events and occurrences of note during the phase execution. For example, when a non-terminal failure or warning is logged.

        :return: The 'phase_details' dictionary for the current phase. A reference to the dict entry at 'self.generation_details_log["phase_details"][self.phase]'.
        """

        self.phase += 1
        if "phase_details" not in self.generation_details_log:
            self.generation_details_log["phase_details"] = {}
        self.generation_details_log["phase_details"][self.phase] = {}
        phase_details = self.generation_details_log["phase_details"][self.phase]

        return phase_details

    def initialize_fresh_scene(self, scene_id: str):
        """
        Set the scene id and initialize a fresh Simulator instance with the specified scene.
        """

        self.reset_epsiode_state()
        self._reset_samplers()
        self._scene_sampler.scene = scene_id
        self.ep_scene_handle = self.generate_scene()
        self.phase = 0

        # generate the navmesh from the config parameters
        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = self.cfg.agent_radius
        navmesh_settings.agent_height = self.cfg.agent_height
        navmesh_settings.include_static_objects = True
        navmesh_settings.agent_max_climb = self.cfg.agent_max_climb
        navmesh_settings.agent_max_slope = self.cfg.agent_max_slope
        self.sim.recompute_navmesh(
            self.sim.pathfinder,
            navmesh_settings,
        )

        self.metadata_interface.refresh_scene_caches(self.sim)
        self.largest_indoor_island_id = get_largest_island_index(
            self.sim.pathfinder, self.sim, allow_outdoor=False
        )
        self.ao_link_map = sutils.get_ao_link_id_map(self.sim)

        self.generation_details_log["scene"] = scene_id

    def _record_sampling_results(
        self, new_obj_and_recs: List[Tuple[Any, Receptacle]], sampler_name: str
    ) -> None:
        """
        Record newly sampled objects and their receptacles into internal datastructures for CollaborationEpisode serialization.
        """

        # record results into internal structures
        new_objects = []

        for obj, rec in new_obj_and_recs:
            self.object_to_containing_receptacle[obj.handle] = rec
            new_objects.append(obj)

        if sampler_name not in self.episode_data["sampled_objects"]:
            self.episode_data["sampled_objects"][sampler_name] = new_objects
        else:
            # handle duplicate sampler names
            self.episode_data["sampled_objects"][sampler_name] += new_objects

        self.ep_sampled_objects += new_objects

    def sample_object_placement_on_floor(
        self,
        object_handle: str,
        alt_pathfinder: habitat_sim.nav.PathFinder = None,
        island_id: int = -1,
        allowed_regions: List[habitat_sim.scene.SemanticRegion] = None,
        max_samples: int = 100,
    ) -> ManagedRigidObject:
        """
        Attempt to sample a valid floor placement for an object.

        :param object_handle: The template handle for the object selected for placement.
        :param alt_pathfinder: The PathFinder instance to use for navmesh sampling. None defaults to sim.pathfinder.
        :param island_id: The navmesh island on which to sample placements. Default -1 samples the full navmesh. Warning: full navmesh may not exclusively be "floor".
        :param allowed_regions: Optionally constrain sampling to a set of pre-determined regions.
        :param max_samples: Maximum number of sampling attempts before reporting failure.

        :return: The sampled object or None if placement failed.
        """

        if alt_pathfinder is None:
            alt_pathfinder = self.sim.pathfinder
        assert alt_pathfinder.is_loaded

        rom = self.sim.get_rigid_object_manager()
        new_obj = rom.add_object_by_template_handle(object_handle)
        if new_obj is None:
            return None

        count = 0
        snap_success = False
        while count < max_samples and not snap_success:
            count += 1
            # sample a point
            nav_point = alt_pathfinder.get_random_navigable_point(
                island_index=island_id
            )
            # validate the region constraints
            if allowed_regions is not None:
                region_ok = False
                for region in allowed_regions:
                    if region.contains(nav_point):
                        region_ok = True
                        break
                if not region_ok:
                    continue
            # lift the object a bit to avoid possible collision margin error
            new_obj.translation = nav_point + mn.Vector3(0.0, 0.1, 0.0)
            # try the placement with snapdown on "stage"
            snap_success = sutils.snap_down(self.sim, new_obj)

        if not snap_success:
            # placement failed, remove the object
            rom.remove_object_by_handle(new_obj.handle)
            return None

        return new_obj

    def sample_objects(self, sample_config: Dict[str, Any], max_tries: int = 1000):
        """
        Sample objects according to the provided config and record the results in episode state structs.

        Example sample state dict

            # description: "sample one kettle on any table in the kitchen"
            {
                "number": "1", #number of objects to sample
                "object_classes": ["kettle"], #object class
                "location": "on", #controlling prepositions
                "furniture_names": ["table"], #list of receptacle classes
                "allowed_regions": ["kitchen"] #an optional list of allowed region names
            },

            # description: "sample two laptop objects on a specific chair"
            {
                "number": "2", #number of objects to sample
                "object_classes": ["laptop"], #object class
                "location": "on", #controlling prepositions
                "furniture_names": ["chair_1"], #list of receptacle furniture instance semantic names
            },

            # description: "sample four to six drinkware objects (target 5) on the floor in the living room and/or bedroom"
            {
                "number": [4, 5, 6], #number of objects to sample [min, target, max]
                "object_classes": ["drinkware"], #object class
                "location": "on", #controlling prepositions
                "furniture_names": ["floor"], #rec class
                "allowed_regions": ["living room", "bedroom"]
            },

        Current features:
            - Sampling multiple objects:
                - precise: (e.g. "number": "2") samples 2 objects with the given config.
                - range: (e.g. "number": [1, 5]) samples 1 to 5 objects.
                - range w/ precise: (e.g. "number": [1,3,5]) samples 1 to 5 targeting 3.
            - Sampling objects of a specified set of classes defined by "object_classes"
                - ("object_classes": ["kettle"]) samples any "kettle" object
            - Sampling objects of a specified set of instances defined by "object_instances".
            - Specifying object instances and classes to be excluded
                - ("excluded_object_classes": ["bowl"]) - cull all objects of the provided category for this sampler
                - ("excluded_object_instances": ["<object_template_handle>"]) - cull the specific objects listed.
                - ("exclude_existing_objects": True) - cull all object instances which were sampled by previous phases (e.g. to prevent clutter generation phase from adding duplicate objects which confound the intended task)
            - Sampling "common sense" objects given the region of the receptacle
                - ("common_sense_object_classes": True)
            - Sampling receptacles:
                - by class(es) (e.g. "furniture_classes": ["table"]) samples any "table"
                - by instances (via "furniture_instances": [...]")
                - specific Receptacle(s) (e.g. "furniture_names": ["table_1"]) looks for receptacle with semantic_name "table_1". See MetadataInterface.refresh_scene_caches().
                - floor as Receptacle (e.g. "furniture_names": ["floor"]) uses navmesh to sample valid navigable placements on the floor
                - "any" or "all" receptacles specified by empty list (e.g. "furniture_classes": [])
            - Sampling relationships:
                - (e.g. "location": "on") - we currently support on_top relationship by snap_down rejection sampling. This includes inside drawers.
            - Region constraints: optionally constraint sampling to a subset of regions
                - add (e.g. "allowed_regions": ["kitchen, "living room"]) to restrict sampling to the specified regions
            - Object States: optionally provide a set of specified object states to be applied to all sampled objects
                - "object_states": { "is_clean": False } - all objects will be marked as is_clean->False in the CollaborationEpisode
        """
        # increment the phase counter and get a dict to track details for this phase of generation
        phase_details = self.get_next_phase_details()

        if "sample_configs" not in self.episode_data["info"]:
            self.episode_data["info"]["sample_configs"] = {}
        self.episode_data["info"]["sample_configs"][
            len(self.episode_data["info"]["sample_configs"])
        ] = sample_config

        #####################################
        # get number of objects to sample
        min_num, max_num, target_number = self.get_and_validate_config_sample_number(
            sample_config
        )
        number_range = (min_num, max_num)

        ######################################
        # check for and validate object_states dict
        object_states = None
        if "object_states" in sample_config:
            object_states = sample_config["object_states"]
            for state_name, _state_val in object_states.items():
                if not isinstance(state_name, str):
                    self.generation_details_log[
                        "failure_mode"
                    ] = f"invalid config, 'object_states' must be a Dict keyed by state name strings. Got {object_states} where key '{state_name}' is not a string."
                    return []
                # NOTE: we don't validate the values because different states could have different types which wouldn't be known here

        ######################################
        # get matching objects
        config_object_classes = sample_config.get("object_classes", [])
        config_object_instances = sample_config.get("object_instances", [])
        config_obj_exclude_instances = sample_config.get(
            "excluded_object_instances", []
        )
        config_obj_exclude_cats = sample_config.get("excluded_object_classes", [])
        exclude_existing_objects = sample_config.get("exclude_existing_objects", False)

        # exclude previously sampled object instances
        if exclude_existing_objects:
            for obj in self.ep_sampled_objects:
                obj_hash = sutils.object_shortname_from_handle(obj.handle)
                config_obj_exclude_instances.append(obj_hash)
            config_obj_exclude_instances = list(set(config_obj_exclude_instances))

        use_common_sense_region_objects = sample_config.get(
            "common_sense_object_classes", False
        )

        mi = self.metadata_interface
        allowed_object_categories: List[str] = (
            config_object_classes
            if len(config_object_classes) > 0
            else self.metadata_interface.dynamic_lexicon.copy()
        )
        # exclude object classes
        allowed_object_categories = [
            allowed_cat
            for allowed_cat in allowed_object_categories
            if allowed_cat not in config_obj_exclude_cats
        ]

        # get objects from categories
        matching_objects: List[str] = []
        for object_class in allowed_object_categories:
            class_matching_objects = mi.get_template_handles_of_class(
                self.sim.metadata_mediator, object_class
            )

            # record failure to find a match for this object type
            if len(class_matching_objects) == 0:
                if "unfound_object_classes" not in phase_details:
                    phase_details["unfound_object_classes"] = []
                phase_details["unfound_object_classes"].append(object_class)
            matching_objects.extend(class_matching_objects)

        # get objects from template handles
        otm = self.sim.metadata_mediator.object_template_manager
        for obj_instance_hash in config_object_instances:
            potential_matches = otm.get_file_template_handles(obj_instance_hash)
            found_obj_instance_match = False
            if len(potential_matches) > 0:
                for potential_match in potential_matches:
                    obj_hash = sutils.object_shortname_from_handle(potential_match)
                    if (
                        obj_hash == obj_instance_hash
                        and mi.get_object_category(obj_hash)
                        in allowed_object_categories
                    ):
                        matching_objects.append(potential_match)
                        found_obj_instance_match = True
                        break
            if not found_obj_instance_match:
                # No match for the requested object could be found.
                # This suggests a typo, hallucination, or dataset inconsistency
                if "unmatched_object_handles" not in phase_details:
                    phase_details["unmatched_object_handles"] = []
                phase_details["unmatched_object_handles"].append(obj_instance_hash)

        # now remove excluded objects
        if len(config_obj_exclude_instances) > 0:
            matching_objects = [
                matching_obj
                for matching_obj in class_matching_objects
                if sutils.object_shortname_from_handle(matching_obj)
                not in config_obj_exclude_instances
            ]

        if len(matching_objects) == 0 and not use_common_sense_region_objects:
            print(
                f"Cannot satisfy sample config. No matching pickable objects found: {sample_config}."
            )
            self.generation_details_log["failure_mode"] = "no matching objects found"
            return []

        #######################################
        # get any allowed regions from metadata
        allowed_regions: List[habitat_sim.scene.SemanticRegion] = []
        if "allowed_regions" in sample_config:
            allowed_regions, _ = self.get_allowed_regions(
                sample_config["allowed_regions"], phase_details
            )

            # if we couldn't find anything, then there are no viable placements
            if len(allowed_regions) == 0:
                print(f"No regions found matching config, aborting. {sample_config}")
                self.generation_details_log[
                    "failure_mode"
                ] = "no matching regions found"
                return []

        # get candidate common sense objects for allowed regions (region -> list of object handles)
        allowed_region_objects: Dict[habitat_sim.scene.SemanticRegion, List[str]] = {}
        if use_common_sense_region_objects:
            total_available_objects = 0
            for rix, region in enumerate(self.sim.semantic_scene.regions):
                if (
                    len(allowed_regions) == 0 or region in allowed_regions
                ) and rix in mi.region_ix_to_room_key:
                    region_object_cats = mi.commonsense_room_objects[
                        mi.region_ix_to_room_key[rix]
                    ]
                    allowed_region_objects[region] = []
                    allowed_region_object_cats = [
                        cat
                        for cat in region_object_cats
                        if cat in allowed_object_categories
                    ]
                    for obj_cat in allowed_region_object_cats:
                        allowed_region_objects[region].extend(
                            mi.get_template_handles_of_class(
                                self.sim.metadata_mediator, obj_cat
                            )
                        )
                    # exclude specified instances
                    allowed_region_objects[region] = [
                        reg_obj
                        for reg_obj in allowed_region_objects[region]
                        if sutils.object_shortname_from_handle(reg_obj)
                        not in config_obj_exclude_instances
                    ]
                    total_available_objects += len(allowed_region_objects[region])

            if total_available_objects == 0:
                print(
                    "No common sense objects matching region, object class, and receptacle class constraints."
                )
                self.generation_details_log[
                    "failure_mode"
                ] = "No common sense objects matching region, object class, and receptacle class constraints."
                return []

        ##############################################
        # gather candidate receptacles from class(es)
        matching_recs: List[Receptacle] = []
        # track the floor option separately
        include_floor = False
        all_recs_allowed = False
        if "furniture_classes" in sample_config:
            # add all receptacles of the target classes to the list
            for furniture_class in sample_config["furniture_classes"]:
                if furniture_class == "floor":
                    include_floor = True
                    continue
                class_matches = mi.get_scene_recs_of_class(furniture_class)
                if len(class_matches) == 0:
                    if "unfound_furniture_classes" not in phase_details:
                        phase_details["unfound_furniture_classes"] = []
                    phase_details["unfound_furniture_classes"].append(furniture_class)
                matching_recs.extend(class_matches)
            if len(sample_config["furniture_classes"]) == 0:
                # empty list denotes wildcard, use all receptacles
                all_recs_allowed = True

        ########################################################
        # gather candidate receptacles from receptacle instances
        config_recep_instances = sample_config.get("furniture_instances", [])
        for recep_inst in config_recep_instances:
            matching_recs.extend(
                [
                    rec
                    for rec in mi.receptacles
                    if rec.parent_object_handle == recep_inst
                ]
            )

        ##############################################
        # gather candidate receptacles from semantic instance names
        if "furniture_names" in sample_config:
            for furniture_instance_name in sample_config["furniture_names"]:
                if furniture_instance_name == "floor":
                    include_floor = True
                    continue
                if furniture_instance_name in mi.recobj_semname_to_handle:
                    # NOTE: getting all Receptacles for a particular ReceptacleObject parent
                    matching_rec_obj_handle = mi.recobj_semname_to_handle[
                        furniture_instance_name
                    ]
                elif furniture_instance_name in mi.recobj_handle_to_semname:
                    matching_rec_obj_handle = furniture_instance_name
                else:
                    if "unfound_furniture_instances" not in phase_details:
                        phase_details["unfound_furniture_instances"] = []
                    phase_details["unfound_furniture_instances"].append(
                        furniture_instance_name
                    )
                    print(
                        f"No ReceptacleObject named {furniture_instance_name} found in the scene, possible hallucination."
                    )
                    continue
                matching_recs.extend(
                    [
                        rec
                        for rec in self.metadata_interface.receptacles
                        if rec.parent_object_handle == matching_rec_obj_handle
                    ]
                )
            if len(sample_config["furniture_names"]) == 0:
                # empty list denotes wildcard, use all receptacles
                all_recs_allowed = True

        # override constraints and allow all receptacles (before region culling)
        # NOTE: "floor" must still be specified somewhere to be allowed.
        if all_recs_allowed:
            matching_recs = self.metadata_interface.receptacles
            phase_details["all_recs_allowed"] = True

        if include_floor:
            phase_details["include_floor"] = True

        if len(matching_recs) == 0 and not include_floor:
            print(
                f"Cannot satisfy sample config. No matching Receptacle objects found: {sample_config}."
            )
            self.generation_details_log[
                "failure_mode"
            ] = "no matching receptacles found"
            return []

        ################################################################
        # cull the set of candidate Receptacles based on region containment.
        matching_recs_in_regions: Dict[
            habitat_sim.scene.SemanticRegion, List[Receptacle]
        ] = defaultdict(lambda: [])
        if len(allowed_regions) > 0 or use_common_sense_region_objects:
            obj_regions: Dict[str, List[habitat_sim.scene.SemanticRegion]] = {}
            for rec in matching_recs:
                # get all regions the Receptacle's parent object is inside of
                if rec.parent_object_handle not in obj_regions:
                    rec_parent_object = sutils.get_obj_from_handle(
                        self.sim, rec.parent_object_handle
                    )
                    obj_regions[rec.parent_object_handle] = [
                        self.sim.semantic_scene.regions[rec_region[0]]
                        for rec_region in sutils.get_object_regions(
                            self.sim,
                            rec_parent_object,
                            ao_link_map=self.ao_link_map,
                        )
                    ]
                for region in obj_regions[rec.parent_object_handle]:
                    matching_recs_in_regions[region].append(rec)

        # merge allowed region rec sets
        if len(allowed_regions) > 0:
            matching_region_recs = []
            for region in allowed_regions:
                matching_region_recs.extend(matching_recs_in_regions[region])
            matching_recs = matching_region_recs

        # remove duplicates
        matching_recs = list(set(matching_recs))

        # get the parent objects for active receptacles
        rec_objects = [rec.parent_object_handle for rec in matching_recs]
        rec_objects = list(set(rec_objects))

        if len(matching_recs) == 0 and not include_floor:
            print(
                f"Cannot satisfy sample config due to region exclusion: {sample_config}."
            )
            self.generation_details_log[
                "failure_mode"
            ] = f"region exclusion in {sample_config}"
            return []

        ##################################################
        # setup the shared ReceptacleSet and ObjectSampler

        # generate a unique name
        sampler_name = "sampler"
        if "name" in sample_config:
            sampler_name = sample_config["name"]
        count = 1
        new_sampler_name = sampler_name
        while new_sampler_name in self._obj_samplers:
            new_sampler_name = f"{sampler_name}_{count}"
            count += 1

        # this ReceptacleSet includes pre-determined Receptacle objects from above
        rec_set = ReceptacleSet(
            new_sampler_name,
            included_object_substrings=[""],
            excluded_object_substrings=[],
            included_receptacle_substrings=[rec.unique_name for rec in matching_recs],
            excluded_receptacle_substrings=[],
        )

        # create a ReceptacleTracker from the ReceptacleSet
        rec_tracker = ReceptacleTracker({}, {new_sampler_name: rec_set})
        # load the scene's filter config file to cull disabled Receptacles
        rec_tracker.init_scene_filters(
            mm=self.sim.metadata_mediator, scene_handle=self.ep_scene_handle
        )
        # register our custom ReceptacleSet for this sampling pass in the generator
        self._receptacle_sets[new_sampler_name] = rec_set

        # create the ObjectSampler
        obj_sampler = ObjectSampler(
            object_set=matching_objects,
            allowed_recep_set_names=[new_sampler_name],
            num_objects=number_range,
            orientation_sample="up",  # objects will only be rotated about Y axis during sampling
            constrain_to_largest_nav_island=True,  # sample placements must be navigable
            nav_to_min_distance=1.0,  # object must be reachable from navmesh
        )
        obj_sampler.receptacle_instances = self.metadata_interface.receptacles
        self._obj_samplers[new_sampler_name] = obj_sampler

        ##################################################
        # sample objects
        new_objs: List[Tuple[ManagedRigidObject, Optional[Receptacle]]] = []
        try_count = 0
        while len(new_objs) < target_number and try_count < max_tries:
            # if include floor as a possible receptacle to be sampled uniformly (among parent objects)
            if include_floor and random.randint(0, len(rec_objects)) == len(
                rec_objects
            ):
                # sample from the floor
                new_obj_handle: str = None
                obj_allowed_regions: List[habitat_sim.scene.SemanticRegion] = None
                if use_common_sense_region_objects:
                    # sample a region and matching object in advance
                    obj_allowed_regions = [
                        random.choice(list(allowed_region_objects.keys()))
                    ]
                    if len(allowed_region_objects[obj_allowed_regions[-1]]) == 0:
                        detail_tag = "no objects for region"
                        if detail_tag not in phase_details:
                            phase_details[detail_tag] = []
                        phase_details[detail_tag].append(obj_allowed_regions[-1].id)
                        try_count += 1
                        continue
                    new_obj_handle = random.choice(
                        allowed_region_objects[obj_allowed_regions[-1]]
                    )
                else:
                    # default behavior
                    new_obj_handle = random.choice(matching_objects)
                    obj_allowed_regions = allowed_regions

                new_obj = self.sample_object_placement_on_floor(
                    new_obj_handle,
                    island_id=self.largest_indoor_island_id,
                    allowed_regions=obj_allowed_regions,
                )
                if new_obj is not None:
                    new_objs.append((new_obj, None))

                # TODO: once on_floor is merged, uncomment this part
                # if not sutils.on_floor(
                #     self.sim,
                #     new_obj,
                #     island_index=self.largest_indoor_island_id,
                #     ao_link_map=self.ao_link_map,
                # ):
                # print("Failed a floor sample b/c placement not on floor...")
                # TODO: remove the object?

            else:
                if use_common_sense_region_objects:
                    # clear the cache
                    obj_sampler.receptacle_candidates = None
                    # modify the object and receptacle candidates to restrict generation to known allowed region
                    obj_allowed_region: habitat_sim.scene.SemanticRegion = (
                        random.choice(list(allowed_region_objects.keys()))
                    )
                    print(f" obj_allowed_region = {obj_allowed_region.id}")
                    obj_sampler.object_set = allowed_region_objects[obj_allowed_region]
                    cats = []
                    for obj in obj_sampler.object_set:
                        cats.append(
                            mi.get_object_category(
                                sutils.object_shortname_from_handle(obj)
                            )
                        )
                    cats = list(set(cats))
                    print("available obj cats for region:")
                    for cat in cats:
                        print(f"    {cat}")
                    if len(obj_sampler.object_set) == 0:
                        detail_tag = "no objects for region"
                        if detail_tag not in phase_details:
                            phase_details[detail_tag] = []
                        phase_details[detail_tag].append(obj_allowed_region.id)
                        try_count += 1
                        continue

                    rec_set.included_receptacle_substrings = [
                        rec.unique_name
                        for rec in matching_recs_in_regions[obj_allowed_region]
                    ]
                    if len(rec_set.included_receptacle_substrings) == 0:
                        detail_tag = "no receptacles for region"
                        if detail_tag not in phase_details:
                            phase_details[detail_tag] = []
                        phase_details[detail_tag].append(obj_allowed_region.id)
                        try_count += 1
                        continue

                    rec_tracker._receptacle_sets[sampler_name] = rec_set
                    print("available recs cats in region:")
                    rec_cats = []
                    for rec in matching_recs_in_regions[obj_allowed_region]:
                        rec_cats.append(
                            mi.get_object_category(
                                sutils.object_shortname_from_handle(
                                    rec.parent_object_handle
                                )
                            )
                        )
                    rec_cats = list(set(rec_cats))
                    for cat in rec_cats:
                        print(f"    {cat}")

                # do standard sampling
                try:
                    new_obj, receptacle = obj_sampler.single_sample(
                        self.sim,
                        rec_tracker,
                        snap_down=True,
                    )
                    if new_obj is not None:
                        logger.info(
                            f"sampled_object = {mi.get_object_category(sutils.object_shortname_from_handle(new_obj.handle))}"
                        )
                        logger.info(
                            f"sampled_rec = {mi.get_object_category(sutils.object_shortname_from_handle(receptacle.parent_object_handle))}"
                        )
                        rec_obj = sutils.get_obj_from_handle(
                            self.sim, receptacle.parent_object_handle
                        )
                        new_obj_regions: List[
                            Tuple[int, float]
                        ] = sutils.get_object_regions(self.sim, rec_obj)
                        obj_region_names = [
                            mi.region_ix_to_semname[rix] for rix, _ in new_obj_regions
                        ]
                        logger.info(f"region(s) = {obj_region_names}")

                        new_objs.append((new_obj, receptacle))
                except Exception as e:
                    if "ep gen failed: internal assert" not in phase_details:
                        phase_details["ep gen failed: internal assert"] = 0
                        phase_details["internal assertions"] = []
                    phase_details["ep gen failed: internal assert"] += 1
                    phase_details["internal assertions"].append(repr(e))
                    print(f"ep gen failed: internal assert: {repr(e)}")
            try_count += 1
        # end object generation

        if try_count >= max_tries and len(new_objs) < number_range[0]:
            print(
                f"Could not sample configured placements within iteration limit ({max_tries}): {sample_config}."
            )
            self.generation_details_log["failure_mode"] = "exceeded max iterations"
            return []

        #############################
        # record assignment of initial object states
        # generate a Dict mapping individual object instances to their initial object states
        if object_states is not None:
            for obj_state_name, obj_state_val in object_states.items():
                if obj_state_name not in self.episode_data["object_states"]:
                    self.episode_data["object_states"][obj_state_name] = {}
                # for each sampled object, set the state
                for obj, _ in new_objs:
                    self.episode_data["object_states"][obj_state_name][
                        obj.handle
                    ] = obj_state_val

        self._record_sampling_results(new_objs, sampler_name=new_sampler_name)

        return new_objs

    def apply_furniture_object_states(
        self, furniture_state_config: Dict[str, Any]
    ) -> bool:
        """
        Apply initial object states to furniture in the scene and cache the results in the episode info.

        In this case, Furniture objects are those which are already present in the scene instance before adding any additional objects.
        This function allows the initiate state of the Furniture in the scene to be modified before object sampling begins.

        :param furniture_state_config: A single config entry in the "initial furniture states" list. A Dict describing the modification to make. See below examples.

        :return: Success or failure of the phase. NOTE: The internal state of the generate is modified to track the changes.

        Example furniture state dicts:

            # description: sample a random 'chest of drawers' and set it to 'dirty' state
            {
                "name": "furniture_object_states_random",
                "furniture_classes": ["chest_of_drawers"],
                "number": "1", #any chest of drawers
                "object_states": {
                    "is_clean": false
                }
            }
            # description: set 'all' chest of drawer objects to 'dirty' state
            {
                "name": "furniture_object_states_all",
                "furniture_classes": ["chest_of_drawers"],
                #lack of "number" is interpreted as "all"
                "object_states": {
                    "is_clean": false
                }
            }
            #description: set a random 'oven' in the 'kitchen' region to 'on' state
            {
                "name": "furniture_object_states_region",
                "furniture_classes": ["oven"],
                "number": "1" #any oven
                "allowed_regions": ["kitchen"], #only furniture in the kitchen
                "object_states": {
                    "is_powered_on": false
                }
            }
            #description: set 'table_1' furniture instance to 'dirty' state
            {
                "name": "furniture_object_states_instance",
                "furniture_names": ["table_1"], #set state for `table_1`
                "object_states": {
                    "is_clean": false
                }
            }
        ]

        Current features:
            - Setting object states for Furniture:
                - explicit: (e.g. "is_clean": false) setting explicit object states
            - Specifying which Furniture's states to set:
                - sample from class(es): (e.g. "furniture_classes": ["table"]) samples any "table" type object
                    - precise: (e.g. "number": "2") sampled 2 Furniture from the provided "furniture_classes"
                    - range: (e.g. "number": [1, 5]) samples 1 to 5 Furniture from the class(es).
                    - all: (e.g. omit "number" key) sets the specified object states are all matching instances of the provided class(es)
                - specify instance(s): (e.g. "furniture_names": ["table_1"]) set the state for "table_1" instance
                - constrain to region: (e.g. "allowed_regions": ["living room", "bedroom"]) only allow setting for objects of the specified class which are also within the specified region(s)
        """

        # increment the phase counter and get a dict to track details for this phase of generation
        phase_details = self.get_next_phase_details()

        ######################################
        # check for and validate incoming  object_states dict
        object_states = None
        if "object_states" in furniture_state_config:
            object_states = furniture_state_config["object_states"]
            for state_name, _state_val in object_states.items():
                if not isinstance(state_name, str):
                    self.generation_details_log[
                        "failure_mode"
                    ] = f"invalid config, 'object_states' must be a Dict keyed by state name strings. Got {object_states} where key '{state_name}' is not a string."
                    return False
                # NOTE: we don't validate the values because different states could have different types which wouldn't be known here

        #####################################
        # get number of objects to sample
        target_number: int = None
        min_num: int = 0
        if "number" in furniture_state_config:
            (
                min_num,
                max_num,
                target_number,
            ) = self.get_and_validate_config_sample_number(furniture_state_config)
        else:
            # if no 'number' is specified, apply the specified initial state to all matching objects
            pass

        ######################################
        # get matching furniture objects
        mi = self.metadata_interface
        selected_furniture_objects = []
        config_object_instances = furniture_state_config.get("furniture_names", [])
        config_object_classes = furniture_state_config.get("furniture_classes", [])
        if len(config_object_classes) == 0:
            # empty list corresponds to all classes
            config_object_classes = mi.lexicon.copy()

        if "floor" in config_object_classes or "floor" in config_object_instances:
            self.generation_details_log[
                "failure_mode"
            ] = "'floor' class is not a currently implemented option for initial furniture state setter."
            raise NotImplementedError()

        # first look for instance matches
        for instance_obj_semname in config_object_instances:
            if instance_obj_semname in mi.recobj_semname_to_handle:
                selected_furniture_objects.append(
                    sutils.get_obj_from_handle(
                        self.sim, mi.recobj_semname_to_handle[instance_obj_semname]
                    )
                )
            else:
                # the specified furniture instance could not be found
                if "unfound_furniture_instances" not in phase_details:
                    phase_details["unfound_furniture_instances"] = []
                phase_details["unfound_furniture_instances"].append(
                    instance_obj_semname
                )

        # look for class matches
        for object_class in config_object_classes:
            class_matching_objects = mi.get_scene_objs_of_class(self.sim, object_class)

            # record failure to find a match for any one type
            if len(class_matching_objects) == 0:
                if "unfound_furniture_classes" not in phase_details:
                    phase_details["unfound_furniture_classes"] = []
                phase_details["unfound_furniture_classes"].append(object_class)
            selected_furniture_objects.extend(class_matching_objects)

        if len(selected_furniture_objects) == 0:
            self.generation_details_log[
                "failure_mode"
            ] = "no matching furniture instances found"
            return False

        #######################################
        # restrict results to the specified region

        # first get the matching regions
        allowed_regions: List[habitat_sim.scene.SemanticRegion] = []
        if "allowed_regions" in furniture_state_config:
            allowed_regions, allowed_region_ixs = self.get_allowed_regions(
                furniture_state_config["allowed_regions"], phase_details
            )

            # if we couldn't find anything, then there are no viable placements
            if len(allowed_regions) == 0:
                print(
                    f"No regions found matching config, aborting. {furniture_state_config}"
                )
                self.generation_details_log[
                    "failure_mode"
                ] = "no matching regions found"
                return False

            # then apply the regions as a constraint over matching objects
            region_culled_selected_objects = []
            for obj in selected_furniture_objects:
                obj_regions = sutils.get_object_regions(self.sim, obj, self.ao_link_map)
                for obj_reg_ix, _ in obj_regions:
                    if obj_reg_ix in allowed_region_ixs:
                        region_culled_selected_objects.append(obj)
            selected_furniture_objects = region_culled_selected_objects

            if len(selected_furniture_objects) == 0:
                self.generation_details_log[
                    "failure_mode"
                ] = "no matching furniture instances found after applying region constraints"
                return False

        #############################
        # apply the "number" to limit application of objects states within the matching set
        if target_number is not None:
            if min_num > len(selected_furniture_objects):
                self.generation_details_log[
                    "failure_mode"
                ] = "fewer than the specified minimum number of furniture instances matched the config constraints for object state application."
                return False
            target_number = min(target_number, len(selected_furniture_objects))
            # shuffle the list and then take a slice off the top
            random.shuffle(selected_furniture_objects)
            selected_furniture_objects = selected_furniture_objects[:target_number]

        #############################
        # record assignment of initial furniture object states
        # generate a Dict mapping individual object instances to their initial object states
        if object_states is not None:
            for obj_state_name, obj_state_val in object_states.items():
                if obj_state_name not in self.episode_data["object_states"]:
                    self.episode_data["object_states"][obj_state_name] = {}
                # for each sampled object, set the state
                for obj in selected_furniture_objects:
                    self.episode_data["object_states"][obj_state_name][
                        obj.handle
                    ] = obj_state_val

        # finished successfully
        return True

    def finalize_episode(self, extra_info) -> CollaborationEpisode:
        """
        Attempt to finalize the episode and distill it into a partial CollaborationEpisode object.
        NOTE: No evaluation info embedded into these episodes
        """

        if self.cfg.enable_check_obj_stability and not self.settle_sim(
            self.episode_data["sampled_targets"].keys()
        ):
            logger.info("Aborting episode generation due to unstable state.")
            self.generation_details_log["failure_mode"] = "unstable state"
            return None

        # collect final object states and serialize the episode
        # TODO: creating shortened names should be automated and embedded in the objects to be done in a uniform way
        sampled_rigid_object_states: List[
            Tuple[str, np.ndarray]
        ] = []  # [(handle, 4x4_transform)]
        for sampled_obj in self.ep_sampled_objects:
            creation_attrib = sampled_obj.creation_attributes
            file_handle = creation_attrib.handle.split(creation_attrib.file_directory)[
                -1
            ].split("/")[-1]
            sampled_rigid_object_states.append(
                (
                    file_handle,
                    np.array(sampled_obj.transformation),
                )
            )

        def extract_recep_info(recep):
            return (recep.parent_object_handle, recep.parent_link)

        save_target_receps = [
            extract_recep_info(x) for x in self.all_target_receptacles
        ]
        save_goal_receps = [extract_recep_info(x) for x in self.all_goal_receptacles]

        name_to_receptacle = {
            k: v.unique_name if v is not None else "floor"
            for k, v in self.object_to_containing_receptacle.items()
        }

        self.episode_data["info"]["object_labels"] = self.target_refs
        self.episode_data["info"]["extra_info"] = extra_info

        # NOTE: directly creating a CollaborationEpisode for object states, but embedding no evaluation info so these will be partial and need to be post-processed
        # NOTE: mypy doesn't like the keyword args for auto_attribs so ignored
        return CollaborationEpisode(  # type: ignore
            scene_dataset_config=self.cfg.dataset_path,
            additional_obj_config_paths=self.cfg.additional_object_paths,
            episode_id=str(self.num_ep_generated - 1),
            start_position=[0, 0, 0],
            start_rotation=[
                0,
                0,
                0,
                1,
            ],
            scene_id=self.ep_scene_handle,
            ao_states=self.ao_states,
            rigid_objs=sampled_rigid_object_states,
            targets=self.episode_data["sampled_targets"],
            target_receptacles=save_target_receps,
            goal_receptacles=save_goal_receps,
            markers=self.cfg.markers,
            name_to_receptacle=name_to_receptacle,
            info=self.episode_data["info"],
            object_states=self.episode_data["object_states"],
        )


def save_ep_dataset(eps: List[CollaborationEpisode], output_path: str) -> None:
    """
    Serialize the CollaborationEpisodes into a CollaborationDatasetV0
    """

    dataset = CollaborationDatasetV0()
    dataset.episodes = eps

    # serialize the dataset
    import gzip

    with gzip.open(output_path, "wt") as f:
        f.write(dataset.to_json())


def get_generator_state_debug_images(
    generator: LLMRearrangeEpisodeGenerator,
) -> List[DebugObservation]:
    """
    Get a list of DebugObservations capturing the current state of the generator.
    """
    dbv = DebugVisualizer(generator.sim)
    obs = [dbv.peek("scene")]
    for obj_handle, rec in generator.object_to_containing_receptacle.items():
        obs.append(dbv.peek(obj_handle, peek_all_axis=True))
        if rec is not None:  # could be "floor"
            obs.append(dbv.peek(rec.parent_object_handle, peek_all_axis=True))
    return obs


def get_generator_sem_state(
    generator: LLMRearrangeEpisodeGenerator,
) -> List[Tuple[str, str, List[str]]]:
    """
    Get a list of tuples, one for each sampled objects with: (object category, receptacle category, [regions])
    Use this to check/test output from a generator pass.
    """
    sem_state = []
    mi = generator.metadata_interface
    for obj_handle, rec in generator.object_to_containing_receptacle.items():
        obj_cat = mi.get_object_category(
            sutils.object_shortname_from_handle(obj_handle)
        )
        rec_cat = "floor"
        obj_regions = []
        if rec is not None:  # could be "floor"
            rec_cat = mi.get_object_category(
                sutils.object_shortname_from_handle(rec.parent_object_handle)
            )
            rec_obj = sutils.get_obj_from_handle(
                generator.sim, rec.parent_object_handle
            )
            obj_regions = sutils.get_object_regions(generator.sim, rec_obj)
        else:
            obj = sutils.get_obj_from_handle(generator.sim, obj_handle)
            obj_regions = sutils.get_object_regions(generator.sim, obj)

        obj_region_names = [mi.region_ix_to_semname[rix] for rix, _ in obj_regions]
        sem_state.append((obj_cat, rec_cat, obj_region_names))
    return sem_state


def get_generator_state_semantic_debug_info(
    generator: LLMRearrangeEpisodeGenerator,
) -> str:
    """
    Get a debug log containing line items for each sampled object and containing receptacles with semantic classes and regions annotated.
    """
    gen_sem_state = get_generator_sem_state(generator)
    log_str = "Generator State Semantic Debug Info:\n"
    log_str += " Object(category) | Receptacle(category) | region(s):\n"
    for obj_cat, rec_cat, regs in gen_sem_state:
        log_str += f"  - {obj_cat} on {rec_cat} in ({regs})\n"
    return log_str


def initialize_generator(
    gen_config: Dict[str, Any], metadata_dict: Dict[str, str]
) -> LLMRearrangeEpisodeGenerator:
    """
    Prepare the generator singleton by loading metadata and assets.

    :param gen_config: generator config dict with dataset, clutter object paths, and output path.
    :param metadata_dict: dict providing relative paths to metadata csvs with object semantics.

    :return: Initialized LLMRearrangeEpisodeGenerator instance.
    """
    # get a metadata interface for finding objects of a particular class
    mi = MetadataInterface(metadata_dict)
    ep_gen = LLMRearrangeEpisodeGenerator(cfg=gen_config, metadata_interface=mi)
    return ep_gen


def generate_episode(
    generator: LLMRearrangeEpisodeGenerator, initial_state_dict: Dict[str, Any]
) -> Tuple[Optional[CollaborationEpisode], Dict[str, Any]]:
    """
    Attempt to generate a single episode from an initial_state_dict defining the generator constraints.

    :param generator: The pre-initialized generator instance. See initialize_generator.
    :param initial_state_dict: dict of generator constraints. See below examples for details.

    :return: Tuple of generator results and a dict with details. Generator result is a CollaborationEpisode instance if generation was successful, or None.

    Initial state dict examples:
    {
        "scene_id": "102817140",
        "episode_id": "test|102817140|0",  # "<string> <scene_id_idx>",  # copy this over
        "initial_furniture_state":[
            {
                "name": "furniture_object_states_random",
                "furniture_classes": ["chest_of_drawers"],
                "number": "1", #any chest of drawers
                "object_states": {
                    "is_clean": false
                }
            }
            {
                "name": "furniture_object_states_all",
                "furniture_classes": ["chest_of_drawers"],
                #lack of "number" is interpreted as "all"
                "object_states": {
                    "is_clean": false
                }
            }
            {
                "name": "furniture_object_states_region",
                "furniture_classes": ["chest_of_drawers"],
                "number": "1" #any one chest of drawers
                "allowed_regions": ["kitchen"], #only furniture in the kitchen
                "object_states": {
                    "is_clean": false
                }
            }
            {
                "name": "furniture_object_states_instance",
                "furniture_names": ["table_1"], #set state for `table_1`
                "object_states": {
                    "is_clean": false
                }
            }
        ]
        "initial_state": [
            # example of articulated object query in any chest of drawers:
            {
                "name": "chest_sampler",
                "number": "2",
                "object_classes": ["drinkware"],
                "location": "on",
                "furniture_classes": ["chest_of_drawers"],
            },
            # example of floor query in some regions:
            {
                "name": "kitchen_floor",
                "number": "2",
                "object_classes": ["drinkware"],
                "location": "on",
                "furniture_names": ["floor"],
                "allowed_regions": ["kitchen"],
            },
            # sampling on a table in the living room
            {
                "name": "livingroom_table_kettle",
                "number": "2",
                "object_classes": ["kettle"],
                "location": "on",
                "furniture_classes": ["table"],
                "allowed_regions": ["living room"],
            },
            # sampling two kettles on a table in the living room with both object state "is_clean" = False
            {
                "name": "livingroom_table_kettle",
                "number": "2",
                "object_classes": ["kettle"],
                "location": "on",
                "furniture_classes": ["table"],
                "allowed_regions": ["living room"],
                "object_states": {
                    "is_clean": false
                }
            },
            # standard rec instance sampling:
            {
                "name": "table_1_kettle",
                "number": "1",
                "object_classes": ["kettle"],
                "location": "on",
                "furniture_names": ["table_1"],
            },
            # Combination sampler including floor, instance, and classes
            {
                "name": "combo_sampler",
                "number": "6",
                "object_classes": ["laptop", "kettle"],
                "location": "on",
                "furniture_names": ["table_1"],
                "furniture_classes": ["floor", "bed"],
            },
            {
                "number": "1",
                "object_classes": ["laptop"],
                "location": "on",
                "furniture_names": ["chair_1"],
            },
            # Example of allowing all receptacles and the floor by combining an empty furniture_classes list with furniture_names list
            {
                "number": "10",
                "object_classes": ["laptop"],
                "location": "on",
                "furniture_names": ["floor"], #"floor" must be explicitly requested somewhere to be used
                "furniture_classes": [], #this is wildcard, all receptacles allowed
                "allowed_regions": ["living room"], #we can still restrict generation to a(n) region(s)
            },
        ],
    }
    """
    # Initialize a fresh scene:
    generator.initialize_fresh_scene(str(initial_state_dict["scene_id"]))

    generator.episode_data["info"]["episode_id"] = initial_state_dict["episode_id"]
    generator.episode_data["info"][
        "is_dynamic"
    ] = generator.cfg.enable_check_obj_stability

    # apply furniture states based on incoming config
    success = True
    for initial_furniture_state in initial_state_dict.get(
        "initial_furniture_state", []
    ):
        success = generator.apply_furniture_object_states(initial_furniture_state)
        if not success:
            print(
                f"Failed to generate state for config {initial_furniture_state}, aborting."
            )
            break

    # generate the episode contents
    obj_info = {}
    if "initial_state" in initial_state_dict and success:
        for sample_config in initial_state_dict["initial_state"]:
            if "template_task_number" in sample_config:
                continue
            new_objs = generator.sample_objects(sample_config)
            if len(new_objs) == 0:
                success = False
                print(f"Failed to generate state for config {sample_config}, aborting.")
                break

            for obj in new_objs:
                obj_cat = generator.metadata_interface.get_object_instance_category(
                    obj[0]
                )
                obj_info[obj_cat] = obj[0].handle

    # finalize and return the episode object
    ep = None
    if success:
        # initialize default ObjectStateSpec values and merge them with assigned object states
        osm = initialize_object_state_machine(
            generator.sim, generator.metadata_interface
        )
        # Get default values
        object_states = osm.get_snapshot_dict(generator.sim)
        # merge in override values
        merge_dicts(object_states, generator.episode_data["object_states"])
        generator.episode_data["object_states"] = object_states

        extra_info = dict(
            filter(
                lambda x: x[0]
                in [
                    "instruction",
                    "scene_id",
                    "file_path",
                    "initial_state",
                    "episode_id",
                ],
                initial_state_dict.items(),
            )
        )
        extra_info.update({"obj_info": obj_info})
        if (
            "initial_state" in initial_state_dict
            and "template_task_number" in initial_state_dict["initial_state"][-1]
        ):
            extra_info["template_task_number"] = initial_state_dict["initial_state"][
                -1
            ]["template_task_number"]
        ep = generator.finalize_episode(extra_info)
    success = ep is not None

    return ep, generator.generation_details_log


def run_generation_over_proposals(gen_config, metadata_dict, init_state_dicts):
    # initialize the generator object
    ep_gen = initialize_generator(gen_config, metadata_dict)

    # store episodes as we compute them
    eps: List[CollaborationEpisode] = []
    # cache failure indices with detail logs
    aborted_inits: Dict[int, Dict] = {}
    # save which episodes are getting filtered out for easy viewing later
    invalid_init = []
    all_valid_inst = []
    for ix, init_state_dict in enumerate(init_state_dicts):
        all_valid_inst.append(init_state_dict["instruction"])
        ep = None
        max_ep_tries = 10
        ep_try = 0
        while ep is None and ep_try < max_ep_tries:
            ep, details_log = generate_episode(ep_gen, init_state_dict)
            if ep is None and details_log["failure_mode"] != "unstable state":
                # don't try again for unsatisfiable constraint failures
                aborted_inits[ix] = details_log
                break
            ep_try += 1

        if ep is not None:
            eps.append(ep)
            invalid_init.append("Episode generation succeeded!")
        elif ep_try == max_ep_tries:
            # record special case of iteration failure
            # TODO: this could be more elegant
            aborted_inits[ix] = {"exceeded_max_iterations": True}
            invalid_init.append("Invalid episode generation")
        else:
            invalid_init.append("Invalid episode generation")

    # save the datasets of episodes
    save_ep_dataset(eps, gen_config["ep_dataset_output"])
    print(
        f"Process complete, generated RearrangeDataset with {len(eps)} episodes out of {len(init_state_dicts)} requested and saved to {gen_config['ep_dataset_output']}"
    )
    if len(aborted_inits) > 0:
        failure_log_output = "generator_failure_log.json"
        if "failure_log_output" in gen_config:
            failure_log_output = gen_config["failure_log_output"]
        # save the failure info to a JSON for review
        with open(failure_log_output, "w") as f:
            json_dump = json.dumps(aborted_inits)
            f.write(json_dump)
        print(
            f"Failure logs for {len(aborted_inits)} aborted episodes saved to {failure_log_output}"
        )

    return len(eps)


default_gen_config = {
    "scene_dataset": "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json",
    "additional_object_paths": [
        "data/objects/ycb/configs/",
        "data/objects_ovmm/train_val/ai2thorhab/configs/objects",
        "data/objects_ovmm/train_val/amazon_berkeley/configs",
        "data/objects_ovmm/train_val/google_scanned/configs",
        "data/objects_ovmm/train_val/hssd/configs/objects",
    ],
    "ep_dataset_output": "test.json.gz",
    "failure_log_output": "generator_failure_log.json",
    # This flag determines whether Episodes are guaranteed dynamically stable or are kinematic-only.
    "enable_check_obj_stability": False,
    # these set the generator back to default values to match the task configs
    "agent_max_climb": 0.2,
    "agent_max_slope": 45.0,
}


if __name__ == "__main__":
    # Run python -m  dataset_generation.benchmark_generation.generate_episodes

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen-config",
        type=str,
        default=None,
        help="Relative path to LLMRearrangeEpisodeGenerator config JSON.",
    )
    parser.add_argument(
        "--metadata-dict",
        type=str,
        default=None,
        help="Relative path to metadata dict config JSON.",
    )
    parser.add_argument(
        "--init-state-dicts",
        type=str,
        default=None,
        help="Relative path to JSON with initial state dict configs for each episode.",
    )

    args, _ = parser.parse_known_args()

    # load the generator config
    gen_config = None
    if args.gen_config is not None:
        with open(args.gen_config, "r") as f:
            gen_config = json.load(f)
    else:
        # default
        gen_config = default_gen_config
    assert "ep_dataset_output" in gen_config

    # load the metadata dict
    metadata_dict = None
    if args.metadata_dict is not None:
        with open(args.metadata_dict, "r") as f:
            metadata_dict = json.load(f)
    else:
        # default
        metadata_dict = default_metadata_dict

    # load the initial state dicts for each episode
    init_state_dicts = []
    if args.init_state_dicts is not None:
        with open(args.init_state_dicts, "r") as f:
            # NOTE: expected JSON is a list of structures
            # {
            #   "initial_state_dicts": [{<init_state_dict},...]
            # }
            init_state_dicts = json.load(f)

    run_generation_over_proposals(gen_config, metadata_dict, init_state_dicts)
