# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import habitat_sim
from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
from habitat_sim import Simulator
from habitat_sim.nav import NavMeshSettings
from tqdm import tqdm

from dataset_generation.benchmark_generation.evaluation_generation.metadata_mapping import (
    generate_hash_to_text,
    get_semantic_object_states,
)
from habitat_llm.sims.metadata_interface import MetadataInterface, default_metadata_dict


class MetadataExtractor:
    """Extract scene and episode metadata for a dataset."""

    scene_metadata: Dict[str, Any]
    episode_metadata: Dict[str, Any]

    def __init__(
        self,
        dataset_path: str,
        metadata_dict: Dict[str, str],
        scene_metadata_cache: str = "",
    ) -> None:
        """
        Args:
            dataset_path (str): path to the PARTNR split to generate metadata for.
            metadata_dict (Dict[str, str]): Contains the dictionary used to instantiate
                the MetadataInterface.
            scene_metadata_cache (str): allows partial caches; loads scenes that already
                exist, computes scene metadata for those that don't. If empty string,
                skip caching altogether.
        """
        self.dataset = self._load_dataset(dataset_path)
        self.metadata_dict = metadata_dict
        self.scene_ids = {ep["scene_id"] for ep in self.dataset["episodes"]}
        self.scene_dataset_config = self.dataset["episodes"][0]["scene_dataset_config"]
        self.scene_metadata = {}
        self.episode_metadata = {}
        self.scene_metadata_cache = scene_metadata_cache

    def extract_scene_metadata(self) -> None:
        """
        Populates self.scene_metadata, which is needed to load the episode metadata. If
        set, loads and saves scene metadata from a cache dir (self.scene_metadata_cache).
        """
        self.load_scene_cache_if_exists()

        for sid in tqdm(self.scene_ids, desc="Scene Extraction", dynamic_ncols=True):
            if sid in self.scene_metadata:
                continue

            sim = self._initialize_fresh_scene(sid)
            mi = MetadataInterface(self.metadata_dict)
            mi.refresh_scene_caches(sim)
            self.scene_metadata[sid] = {
                "furniture": mi.get_region_rec_contents(sim),
                "receptacle_to_handle": mi.recobj_semname_to_handle,
                "room_to_id": {
                    k: sim.semantic_scene.regions[v].id
                    for k, v in mi.region_semname_to_id.items()
                },
            }
            sim.close()
            del sim
            self.save_scene_cache(sid)

    def load_scene_cache_if_exists(self) -> None:
        """Populates self.scene_metadata from self.scene_metadata_cache, if it exists"""
        if self.scene_metadata_cache == "" or not os.path.exists(
            self.scene_metadata_cache
        ):
            return

        for fname in os.listdir(self.scene_metadata_cache):
            sid = fname.split(".")[0]
            fpath = os.path.join(self.scene_metadata_cache, fname)
            with open(fpath) as f:
                self.scene_metadata[sid] = json.load(f)

    def save_scene_cache(self, scene_id: str) -> None:
        """Saves self.scene_metadata to self.scene_metadata_cache, if provided"""
        if self.scene_metadata_cache == "":
            return

        fpath = os.path.join(self.scene_metadata_cache, f"{scene_id}.json")
        if os.path.exists(fpath):
            return

        os.makedirs(self.scene_metadata_cache, exist_ok=True)
        with open(fpath, "wt") as f:
            json.dump(self.scene_metadata[scene_id], f, indent=2)

    def extract_episode_metadata(self) -> None:
        """
        Populates self.episode_metadata, a map from episode ID to a metadata dictionary.
        """
        if len(self.scene_metadata) == 0:
            raise ValueError("scene_metadata is empty.")

        for ep in tqdm(
            self.dataset["episodes"], desc="Episode Extraction", dynamic_ncols=True
        ):
            eid = ep["episode_id"]
            metadata = self.load_episode_metadata(ep)
            if self.is_inconsistent(metadata):
                print(
                    f"\n\n[EID: {eid}] WARNING: Inconsistent rooms:"
                    " obj-room is different than obj-recep-room.\n\n"
                )
            self.episode_metadata[eid] = metadata

    def save(self, save_dir: str) -> None:
        """
        Save episode metadata to disk. In `save_dir`, each episode's metadata is
        saved to the file [episode_id].json.
        """
        if len(self.episode_metadata) == 0:
            raise ValueError("episode_metadata is empty.")

        os.makedirs(save_dir, exist_ok=True)
        for eid, metadata in tqdm(
            self.episode_metadata.items(), desc="Saving", dynamic_ncols=True
        ):
            with open(os.path.join(save_dir, f"{eid}.json"), "w") as f:
                json.dump(metadata, f, indent=2)

    def _initialize_fresh_scene(self, scene_id: str) -> Simulator:
        """
        Set the scene id and initialize a fresh Simulator instance with the specified scene.
        """
        sim = self._initialize_sim(scene_id)
        cfg = get_config_defaults()

        # generate the navmesh from the config parameters
        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = cfg.agent_radius
        navmesh_settings.agent_height = cfg.agent_height
        navmesh_settings.include_static_objects = True
        navmesh_settings.agent_max_climb = cfg.agent_max_climb
        navmesh_settings.agent_max_slope = cfg.agent_max_slope
        sim.recompute_navmesh(
            sim.pathfinder,
            navmesh_settings,
        )
        return sim

    def _initialize_sim(self, scene_name: str) -> Simulator:
        """
        Initialize a new Simulator object with a selected scene and dataset.
        """
        camera_resolution = [540, 720]
        sensors = {
            "rgb": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
            }
        }

        additional_object_paths = self.dataset["episodes"][0]["additional_object_paths"]

        cfg = get_config_defaults()

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = self.scene_dataset_config
        backend_cfg.scene_id = scene_name
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = cfg.gpu_device_id

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(hab_cfg)

        object_attr_mgr = sim.get_object_template_manager()
        for object_path in additional_object_paths:
            object_attr_mgr.load_configs(os.path.abspath(object_path))

        return sim

    def load_episode_metadata(self, episode) -> Dict:
        """Loads the episode metadata. requires self.scene_metadata to be populated."""

        def sorted_dict(d, key):
            return dict(sorted(d.items(), key=key))

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

        scene_info = self.scene_metadata[episode["scene_id"]]
        rooms = scene_info["room_to_id"].keys()
        receptacle_to_handle = scene_info["receptacle_to_handle"]
        room_to_id = scene_info["room_to_id"]
        recep_to_description = generate_hash_to_text(
            "data/fphab/metadata/fpmodels-with-decomposed.csv", receptacle_to_handle
        )
        recep_to_room = {}
        for room, recep_list in scene_info["furniture"].items():
            for recep in recep_list:
                recep_to_room[recep] = room
        (
            objects,
            object_to_handle,
            object_to_room,
            object_to_recep,
        ) = self.object_instance_info_from_episode(episode, scene_info)

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
            "instruction": episode["instruction"],
        }

    @staticmethod
    def object_instance_info_from_episode(
        episode, scene_info: Dict
    ) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Extracts and returns objects, object_to_handle, object_to_room, and
        object_to_recep data.
        """
        handle_to_recep = {v: k for k, v in scene_info["receptacle_to_handle"].items()}
        handle_to_recep["floor"] = "floor"
        recep_to_room = {}
        for room, receps in scene_info["furniture"].items():
            for recep in receps:
                recep_to_room[recep] = room

        object_handles = list(episode["name_to_receptacle"].keys())

        objects = []
        object_cat_to_count: DefaultDict[str, int] = defaultdict(int)
        object_to_room: Dict[str, str] = {}
        for state_element in episode["info"]["initial_state"]:
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
                objects.append(o)

        # NOTE: this mapping is tenuous and relies on CPython dict order
        object_to_handle = {objects[i]: object_handles[i] for i in range(len(objects))}

        # get object to recep and object to room mappings
        object_to_recep: Dict[str, str] = {}
        for obj, handle in object_to_handle.items():
            recep_handle = episode["name_to_receptacle"][handle].split("|")[0]
            recep = handle_to_recep[recep_handle]
            object_to_recep[obj] = recep
            if recep == "floor":
                # accept the already-populated state_element room ID
                continue
            object_to_room[obj] = recep_to_room[recep]

        return (
            objects,
            object_to_handle,
            object_to_room,
            object_to_recep,
        )

    @staticmethod
    def _load_dataset(dataset_path: str) -> Dict[str, Any]:
        """Load the .json.gz PARTNR dataset"""
        with gzip.open(dataset_path, "rt") as f:
            dataset = json.load(f)
        return dataset

    @staticmethod
    def is_inconsistent(metadata):
        """
        Returns True if an object's room is different than the object's
        receptacle->room relation.
        """
        for obj, room in metadata["object_to_room"].items():
            recep = metadata["object_to_recep"][obj]
            if recep == "floor":
                continue
            recep_room = metadata["recep_to_room"][recep]
            if recep_room != room:
                print(obj, room, recep, recep_room)
                return True
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=str,
        help="Path to the PARTNR dataset to load in .json.gz format.",
    )
    parser.add_argument(
        "--scene-metadata-cache",
        required=False,
        type=str,
        default="",
        help="If provided, save scene info files to a cache directory. If the cache already exists, loads scene infos from it.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        type=str,
        help="Path to the directory to save the resulting metadata files.",
    )
    args, _ = parser.parse_known_args()

    me = MetadataExtractor(
        args.dataset_path,
        default_metadata_dict,
        scene_metadata_cache=args.scene_metadata_cache,
    )

    me.extract_scene_metadata()
    me.extract_episode_metadata()
    me.save(args.save_dir)
