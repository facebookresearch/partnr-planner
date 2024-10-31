#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle
import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
import omegaconf
import pandas as pd
from habitat.core.logging import logger
from habitat_sim.physics import ManagedArticulatedObject, ManagedRigidObject

from habitat_llm.utils.sim import find_receptacles

# known default paths to metadata files in HSSD SceneDataset
default_metadata_dict: Dict[str, str] = {
    "metadata_folder": "data/hssd-hab/metadata/",
    "obj_metadata": "object_categories_filtered.csv",
    "room_objects_json": "room_objects.json",
    "staticobj_metadata": "fpmodels-with-decomposed.csv",
    "object_affordances": "affordance_objects.csv",
}


def get_metadata_dict_from_config(
    dataset_config: omegaconf.DictConfig,
) -> Dict[str, str]:
    """
    Override the default_metadata_dict with values from a config and return the result.

    NOTE: overrides place these configs in 'habitat.dataset.metadata'

    :param config: The 'habitat.dataset' subconfig node possibly containing a 'metadata' node.

    :return: The metadata_dict containing the paths and files merged from default and config.
    """

    metadata_dict = default_metadata_dict.copy()

    if hasattr(dataset_config, "metadata"):
        metadata_config = dataset_config.metadata
        for metadata_config_key in [
            "metadata_folder",
            "obj_metadata",
            "staticobj_metadata",
            "room_objects_json",
            "object_affordances",
        ]:
            if hasattr(metadata_config, metadata_config_key):
                metadata_dict[metadata_config_key] = getattr(
                    metadata_config, metadata_config_key
                )

    return metadata_dict


class MetadataError(ValueError):
    def __init__(self, message="MetadataError"):
        super().__init__(message)


class MetadataInterface:
    """
    MetadataInterface provides a lightweight interface for managing the semantic metadata from csv files for HSSD.

    MetadataInterface loads and processes metadata about objects and receptacles.
    This class also offers several methods to work with the loaded metadata, such as queries for: common-sense mapping from regions to objects, object/furniture semantic categories, and searching for objects or receptacles of a specific semantic class.
    """

    def __init__(self, metadata_source_dict: Dict[str, str]) -> None:
        """
        Initialize the MetadataInterface, loading the metadata from provided paths.
        Does not fill internal ManagedObject handle mapping caches until `refresh_scene_caches` is called with a Simulator instance provided.

        :param metadata_source_dict: A dict containing paths to metadata files. See default_metadata_dict.
        """

        # digest the HSSD metadata
        self.metadata_source_dict = metadata_source_dict
        # common sense mappings from region categories to object categories typically found in the room
        self.commonsense_room_objects: Dict[str, List[str]] = {}
        # cache the source of each hash
        self.hash_to_source: Dict[str, str] = {}
        # cache the lexicon of available non-static (ovmm) objects
        self.dynamic_lexicon: List[str] = []
        # maps object states to semantic classes which can have those states. See affordance_objects.csv
        self.affordance_info: Dict[str, List[str]] = {}
        self.metadata = self.load_metadata(self.metadata_source_dict)

        self.receptacles: List[hab_receptacle.Receptacle] = None

        # generate a lexicon from the metadata
        self.hash_to_cat: Dict[str, str] = {}
        self.lexicon: List[str] = []  # all object classes annotated
        for index in range(self.metadata.shape[0]):
            cat = self.metadata.at[index, "type"]
            self.hash_to_cat[self.metadata.at[index, "handle"]] = cat
            self.lexicon.append(cat)
        # deduplicate lexicon
        self.lexicon = list(set(self.lexicon))
        # remove non strings (e.g. nan)
        self.lexicon = [entry for entry in self.lexicon if isinstance(entry, str)]

        # semantic naming for ReceptacleObjects (e.g. table_1)
        self.recobj_semname_to_handle: Dict[str, str] = {}
        self.recobj_handle_to_semname: Dict[str, str] = {}
        # consistent semantic naming for SemanticRegions (e.g. "living room" becomes "living_room_0")
        self.region_ix_to_semname: Dict[int, str] = {}
        self.region_semname_to_id: Dict[str, int] = {}
        # maps region index to key in commonsense_room_objects if a match was found
        self.region_ix_to_room_key: Dict[int, str] = {}

    def load_metadata(self, metadata_dict: Dict[str, str]) -> pd.DataFrame:
        """
        This method loads the metadata about objects and receptacles from csv and json files.
        This data will typically include tags associated with these objects such as semantic class, product descriptions, object state affordances, etc...

        Fills internal structures:
        - self.commonsense_room_objects
        - self.hash_to_source
        - self.dynamic_lexicon

        :param metadata_dict: A dict containing paths to metadata files. See default_metadata_dict.

        :return: The DataFrame object containing the object hash name to semantic classes map.
        """

        # Make sure that metadata_dict is not None
        if not metadata_dict:
            raise ValueError("Cannot load metadata from None")

        # Fetch relevant paths
        metadata_folder = metadata_dict["metadata_folder"]
        object_metadata_path = os.path.join(
            metadata_folder, metadata_dict["obj_metadata"]
        )
        static_object_metadata_path = os.path.join(
            metadata_folder, metadata_dict["staticobj_metadata"]
        )
        room_objects_json_path = os.path.join(
            metadata_folder, metadata_dict["room_objects_json"]
        )
        object_affordances_csv_path = os.path.join(
            metadata_folder, metadata_dict["object_affordances"]
        )

        # Make sure that the paths are valid
        if not os.path.exists(object_metadata_path):
            raise Exception(f"Object metadata file not found, {object_metadata_path}")
        if not os.path.exists(static_object_metadata_path):
            raise Exception(
                f"Receptacle metadata file not found, {static_object_metadata_path}"
            )
        if not os.path.exists(room_objects_json_path):
            raise Exception(
                f"Common sense region to object class json mapping file not found, {room_objects_json_path}"
            )

        # first load the json room->object categories map
        self.commonsense_room_objects = {}
        with open(room_objects_json_path, "r") as f:
            commonsense_room_objects = json.load(f)
            for room_name in commonsense_room_objects:
                # NOTE: removing upper case letters here
                self.commonsense_room_objects[
                    room_name.lower()
                ] = commonsense_room_objects[room_name]

        # load object affordances metadata
        self.affordance_info = {}
        with open(object_affordances_csv_path, "r", newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            for row in csvreader:
                state_type = row[0]
                allowed_classes = [
                    re.sub("[^a-zA-Z_]", "", r) for r in row[2:] if len(r) > 0
                ]
                self.affordance_info[state_type] = allowed_classes

        # Read the metadata files
        df_static_objects = pd.read_csv(static_object_metadata_path)
        df_objects = pd.read_csv(object_metadata_path)

        # Rename some columns
        df1 = df_static_objects.rename(
            columns={"id": "handle", "main_category": "type"}
        )
        df2 = df_objects.rename(columns={"id": "handle", "clean_category": "type"})

        # Drop the rest of the columns in both DataFrames
        df1 = df1[["handle", "type"]]
        df2 = df2[["handle", "type"]]

        # setup the hash to source mapping
        for index in range(df1.shape[0]):
            self.hash_to_source[df1.at[index, "handle"]] = "hssd"
        for index in range(df2.shape[0]):
            cat = df2.at[index, "type"]
            self.hash_to_source[df2.at[index, "handle"]] = "dynamic"
            self.dynamic_lexicon.append(cat)
        self.dynamic_lexicon = list(set(self.dynamic_lexicon))

        # Merge the two data frames
        union_df = pd.concat([df1, df2], ignore_index=True)

        return union_df

    def match_common_sense_region_objects_for_scene(
        self, sim: habitat_sim.Simulator
    ) -> None:
        """
        Attempts to match Simulator SemanticRegions to room name keys in self.commonsense_room_objects.
        Uses SemanticCategory.name() and attempts string regularization to match the keys in the map.

        :param sim: Simulator instance is necessary to extract active SemanticRegions.
        """

        self.region_ix_to_room_key = {}
        for rix, region in enumerate(sim.semantic_scene.regions):
            cat_names = [region.category.name()]
            cat_names.extend(
                cat_names[0].split("/")
            )  # sometimes annotations contain multiple synonyms split by "/"
            matching_room_name_keys = [
                cat_name
                for cat_name in cat_names
                # NOTE: by construction, only lower case in self.commonsense_room_objects
                if cat_name.lower() in self.commonsense_room_objects
            ]
            matching_room_name_keys = list(set(matching_room_name_keys))
            if len(matching_room_name_keys) == 0:
                logger.info(
                    f"Cannot find common sense object mapping for region '{region.id}' with category name set: {cat_names}."
                )
            else:
                self.region_ix_to_room_key[rix] = matching_room_name_keys[0]
                if len(matching_room_name_keys) > 1:
                    logger.info(
                        f"Multiple common sense object mappings found for region {region.id} with category name set: {cat_names}. Matches = {matching_room_name_keys}, only using {matching_room_name_keys[0]}."
                    )

    def refresh_scene_caches(
        self, sim: habitat_sim.Simulator, filter_receptacles: bool = True
    ) -> None:
        """
        When a new Simulator instance is initialized we need to refresh internal instance caches.

        Also populates a mapping of Entity::Receptacle semantic names used to specify generation.

        Semantic names are generated by collecting all Receptacle parent objects and enumerating them.

        :param sim: Simulator instance. Necessary to extract active ManagedObjects, Receptacles, and SemanticRegions.
        :param filter_receptacles: If true, apply the rec_filter_file for the scene during Receptacle parsing. Only accessible and valid receptacles (as annotated in the filter file) will be available through this MetadataInterface if this option is used.
        """

        # first parse the scene's active receptacles
        self.receptacles = find_receptacles(sim, filter_receptacles)

        self.recobj_semname_to_handle = {}
        self.recobj_handle_to_semname = {}

        sem_class_counter: Dict[str, int] = defaultdict(lambda: 0)

        for receptacle in self.receptacles:
            receptacle_object = sutils.get_obj_from_handle(
                sim, receptacle.parent_object_handle
            )
            if receptacle_object.handle in self.recobj_handle_to_semname:
                # this parent object is already registered
                continue
            sem_class = self.get_object_instance_category(receptacle_object)
            if sem_class is None:
                logger.debug(
                    f"Found Receptacle with unknown parent object '{receptacle.parent_object_handle}' type. Culling this object from available receptacles."
                )
                continue
            # computes the semantic name of the receptacle
            instance_sem_name = f"{sem_class}_{sem_class_counter[sem_class]}"
            sem_class_counter[sem_class] += 1
            self.recobj_semname_to_handle[instance_sem_name] = receptacle_object.handle
            self.recobj_handle_to_semname[receptacle_object.handle] = instance_sem_name

        # construct the region semantic name maps
        self.region_ix_to_semname = {}
        self.region_semname_to_id = {}
        for rix, region in enumerate(sim.semantic_scene.regions):
            std_name = region.category.name().replace(" ", "_")
            indexed_std_name = std_name + "_0"
            count = 1
            while indexed_std_name in self.region_semname_to_id:
                indexed_std_name = f"{std_name}_{count}"
                count += 1
            self.region_ix_to_semname[rix] = indexed_std_name
            self.region_semname_to_id[indexed_std_name] = rix

        self.match_common_sense_region_objects_for_scene(sim)

    def get_object_category(self, obj_hash: str) -> Optional[str]:
        """
        Get the semantic class lexicon entry corresponding to the object hash.

        :param obj_hash: The shortened name of the object created from a ManagedObject handle by stripping filepath prefix, file ending postfix, and instance handle index strings postfix.

        :return: The semantic class string or None if either the object has no annotated class or the hash cannot be matched to an object.
        """

        if obj_hash in self.hash_to_cat:
            obj_class = self.hash_to_cat[obj_hash]
            if isinstance(obj_class, str) and len(obj_class) > 0:
                return obj_class
        return None

    def get_object_instance_category(
        self, obj: Union[ManagedRigidObject, ManagedArticulatedObject]
    ) -> Optional[str]:
        """
        Get the semantic class lexicon entry corresponding to the ManagedObject instance's template hash.

        :param obj: The ManagedObject for which to query the semantic class.

        :return: The semantic class string or None if either the object has no annotated class or the object cannot be found in internal caches.
        """

        obj_hash = sutils.object_shortname_from_handle(obj.handle)
        obj_cat = self.get_object_category(obj_hash)
        return obj_cat

    def get_scene_lexicon(self, sim: habitat_sim.Simulator) -> List[str]:
        """
        Get the lexicon of the current scene contents by scraping the contents.

        :param sim: Get the lexicon of semantic classes for all active objects in the Simulator instance's active scene.

        :return: A list of all semantic classes in the currently active scene.
        """

        scene_lexicon = []
        rom = sim.get_rigid_object_manager()
        aom = sim.get_articulated_object_manager()
        # get all rigid and articulated object instances
        all_objs = list(rom.get_objects_by_handle_substring().values()) + list(
            aom.get_objects_by_handle_substring().values()
        )
        for obj in all_objs:
            obj_hash = sutils.object_shortname_from_handle(obj.handle)
            obj_cat = self.get_object_category(obj_hash)
            scene_lexicon.append(obj_cat)
        # de-dup
        scene_lexicon = list(set(scene_lexicon))
        # remove non-strings (e.g. None)
        scene_lexicon = [entry for entry in scene_lexicon if isinstance(entry, str)]
        return scene_lexicon

    def get_scene_objs_of_class(
        self, sim: habitat_sim.Simulator, sem_class: str
    ) -> List[Union[ManagedRigidObject, ManagedArticulatedObject]]:
        """
        Get all object instances of a given class in the currently instanced scene.

        :param sim: The Simulator instance.
        :param sem_class: The semantic class name.

        :return: The list of ManagedObject instances belonging to the desired semantic class.
        """

        objs_of_class: List[Union[ManagedRigidObject, ManagedArticulatedObject]] = []
        if sem_class not in self.lexicon:
            logger.error(
                f"get_scene_objs_of_class failed, sem_class '{sem_class}' not in lexicon."
            )
            return objs_of_class

        # get all rigid and articulated object instances
        all_objs = sutils.get_all_objects(sim)
        for obj in all_objs:
            obj_hash = sutils.object_shortname_from_handle(obj.handle)
            obj_cat = self.get_object_category(obj_hash)
            if obj_cat == sem_class:
                objs_of_class.append(obj)

        return objs_of_class

    def get_scene_recs_of_class(
        self,
        sem_class: str,
    ) -> List[hab_receptacle.Receptacle]:
        """
        Get all Receptacles of a given semantic class in the currently instanced scene.
        Concretely, searches for Receptacle's with parent objects belonging to the given class.
        NOTE: Must be called after 'self.refresh_scene_caches'.

        :param sem_class: The semantic class name.

        :return: The list of matching Receptacles.
        """

        if self.receptacles is None:
            raise ValueError(
                "self.receptacles is None. No receptacles have been scraped from the scene, call MetadataInterface.refresh_scene_caches() first."
            )

        class_recs: List[hab_receptacle.Receptacle] = []

        for rec in self.receptacles:
            parent_obj_hash = sutils.object_shortname_from_handle(
                rec.parent_object_handle
            )
            rec_cat = self.get_object_category(parent_obj_hash)
            if rec_cat == sem_class:
                class_recs.append(rec)

        return class_recs

    def get_template_handles_of_class(
        self,
        mm: habitat_sim.metadata.MetadataMediator,
        sem_class: str,
        dynamic_source: bool = True,
    ) -> List[str]:
        """
        Search the MetadataMediator for all objects of a given class and return a list of template handles.
        This search will cover the entirety of a SceneDataset, returning all matches, not limited to the instanced scene.

        :param mm: The MetadataMediator from which to query the handles. Should already have loaded all asset config templates.
        :param sem_class: The semantic category of the objects which should be returned.
        :param dynamic_source: Whether or not to limit the resulting handles to those enumerated in the external dynamic objects csv file. I.e., exclude Furniture objects.

        :return: The list of relevant object template handles.
        """

        otm = mm.object_template_manager
        class_handles: List[str] = []
        if (
            dynamic_source and sem_class not in self.dynamic_lexicon
        ) or sem_class not in self.lexicon:
            logger.error(
                f"'{sem_class}' not in {('dynamic' if dynamic_source else '')} lexicon."
            )
            return class_handles
        for handle in otm.get_file_template_handles():
            obj_short_name = sutils.object_shortname_from_handle(handle)
            if dynamic_source and (
                obj_short_name not in self.hash_to_source
                or self.hash_to_source[obj_short_name] != "dynamic"
            ):
                continue
            obj_cat = self.get_object_category(obj_short_name)
            if obj_cat == sem_class:
                class_handles.append(handle)
        return class_handles

    def get_region_rec_contents(
        self, sim: habitat_sim.Simulator
    ) -> Dict[str, List[str]]:
        """
        Get the current region set membership for the loaded scene.
        Returns a map of SemanticRegion semantic names (see self.region_ix_to_semname) to a list of Furniture object semantic names for all objects which have Receptacles annotated.

        :param sim: The Simulator instance.

        :return: The dict mapping region semantic names to lists of Furniture object handles.
        """

        if self.receptacles is None:
            raise ValueError(
                "No receptacles have been scraped from the scene, call 'refresh_scene_caches()' first."
            )

        region_recs: Dict[str, List[str]] = {}

        ao_link_map = sutils.get_ao_link_id_map(sim)

        # search directly in the semantic names map constructed during scene refresh
        for rec_obj_handle, obj_sem_name in self.recobj_handle_to_semname.items():
            parent_obj = sutils.get_obj_from_handle(sim, rec_obj_handle)
            obj_regions = sutils.get_object_regions(
                sim, parent_obj, ao_link_map=ao_link_map
            )
            for rix, _ratio in obj_regions:
                reg_name = self.region_ix_to_semname[rix]
                if reg_name not in region_recs:
                    region_recs[reg_name] = []
                region_recs[reg_name].append(obj_sem_name)

        return region_recs

    def get_object_property_from_metadata(
        self, handle: str, metadata_field: str
    ) -> Any:
        """
        This method returns the value of the requested property using the loaded metadata DataFrame.
        For example, this could be used to extract the semantic type of any object
        in HSSD. Not that the property should exist in the merged DataFrame object.
        See `load_metadata()`.

        NOTE: currently unused, but kept for potential usefulness.

        :param handle: The handle of the object for which the metadata entry should be queried.
        :param metadata_field: The metadata field to query for the object.

        :return: The value corresponding the requested 'metadata_field' for the passed object 'handle'.
        """

        # Declare default
        property_value = "unknown"

        # get hash from handle
        handle_hash = handle.rpartition("_")[0]

        # Use loc to locate the row with the specific key
        object_row = self.metadata.loc[self.metadata["handle"] == handle_hash]

        # Extract the value from the object_row
        if not object_row.empty:
            # Make sure the property value is not nan or empty
            if (
                object_row[metadata_field].notna().any()
                and (object_row[metadata_field] != "").any()
            ):
                property_value = object_row[metadata_field].values[0]
        else:
            raise MetadataError(f"Handle {handle} not found in the metadata.")

        return property_value
