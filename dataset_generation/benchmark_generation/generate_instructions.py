# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import warnings

import habitat_sim
import omegaconf
from habitat.sims.habitat_simulator.sim_utilities import object_shortname_from_handle

import habitat_llm
from habitat_llm.sims.metadata_interface import MetadataInterface, default_metadata_dict

warnings.filterwarnings("ignore")
import csv
import json

import hydra
import pandas as pd
from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
from habitat_sim.nav import NavMeshSettings
from hydra.utils import instantiate
from omegaconf import OmegaConf

try:
    from rlm.llm import RemoteLanguageModel
except ImportError:
    RemoteLanguageModel = None

from habitat_llm.utils.sim import find_receptacles


# Generate episodes given a json specification
class InstructionGeneratorHSSD:
    def __init__(self, config, **kwargs):
        # Here we should put the episode
        self.config = OmegaConf.create(config)
        self.sim = None
        self.cfg = get_config_defaults()

        # Init LLM
        habitat_llm_dir_path = os.path.dirname(habitat_llm.__file__)
        llm_config_path = f"{habitat_llm_dir_path}/conf/llm/{self.config.llm.name}.yaml"
        assert os.path.exists(
            llm_config_path
        ), f"LLM config file not found at {llm_config_path}"
        llm_config = OmegaConf.load(llm_config_path)
        llm_config.generation_params = OmegaConf.merge(
            llm_config.generation_params,
            OmegaConf.create(self.config.llm.generation_params),
        )
        self.llm = instantiate(llm_config.llm)(conf=llm_config)

    def initialize_sim(self, scene_name: str, dataset_path: str) -> None:
        """
        Initialize a new Simulator object with a selected scene and dataset.
        """
        # Setup a camera coincident with the agent body node.
        # For debugging visualizations place the default agent where you want the camera with local -Z oriented toward the point of focus.
        camera_resolution = [540, 720]
        sensors = {
            "rgb": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
            }
        }

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = dataset_path
        backend_cfg.scene_id = scene_name
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = self.cfg.gpu_device_id

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            # sensor_spec = habitat_sim.EquirectangularSensorSpec()
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
        if self.sim is None:
            self.sim = habitat_sim.Simulator(hab_cfg)

            object_attr_mgr = self.sim.get_object_template_manager()
            for object_path in self.cfg.additional_object_paths:
                object_attr_mgr.load_configs(osp.abspath(object_path))
        else:
            if self.sim.config.sim_cfg.scene_id != scene_name:
                self.sim.close(destroy=True)
            if self.sim.config.sim_cfg.scene_id == scene_name:
                # we need to force a reset, so reload the NONE scene
                # TODO: we should fix this to provide an appropriate reset method
                proxy_backend_cfg = habitat_sim.SimulatorConfiguration()
                proxy_backend_cfg.scene_id = "NONE"
                proxy_backend_cfg.gpu_device_id = self.cfg.gpu_device_id
                proxy_hab_cfg = habitat_sim.Configuration(
                    proxy_backend_cfg, [agent_cfg]
                )
                self.sim.reconfigure(proxy_hab_cfg)
            self.sim.reconfigure(hab_cfg)

        # setup the debug camera state to the center of the scene bounding box
        scene_bb = self.sim.get_active_scene_graph().get_root_node().cumulative_bb
        self.sim.agents[0].scene_node.translation = scene_bb.center()

    def initialize_fresh_scene(self, scene_id: str):
        """
        Set the scene id and initialize a fresh Simulator instance with the specified scene.
        """
        self.initialize_sim(scene_id, self.config.scene_dataset)
        print("initialized scene ", scene_id)

        self.receptacles = find_receptacles(self.sim)

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

    def generate_instructions(self, tasks=None):
        """
        Generate instructions for all scenes and a set of optional guiding tasks.
        """

        if tasks is None:
            tasks = []

        output_path = self.config.output_path
        all_scene_ids = self.config.scene_ids
        for scene_id in all_scene_ids:
            config_path = f"{output_path}/config.yaml"
            scene_id = str(scene_id)
            output_path_scene = osp.join(output_path, scene_id)
            # Initialize the scene
            self.initialize_fresh_scene(scene_id)
            self.scene_info = self.obtain_scene_info()
            scene_lexicon = self.mi.get_scene_lexicon(self.sim)
            if len(scene_lexicon) == 0:
                print("skipping instruction gen for scene ", scene_id)
                continue
            print("Starting inst gen for scene ", scene_id)
            # Scene info dictionary
            scene_info_path = f"{output_path_scene}/scene_info.json"
            print(f"Parsing Scene: {scene_id}")

            output_path_scene_gen = f"{output_path_scene}/output_gen/"
            if not osp.isdir(output_path_scene_gen):
                os.makedirs(output_path_scene_gen)

            for task in tasks:
                if len(task) == 0:
                    template_instruction = ""
                    task_num = 0
                else:
                    template_instruction = task["instruction"]
                    task_num = task["episode_id"]
                for iter_call in range(self.config.calls_per_scene):
                    full_file = (
                        f"{output_path_scene_gen}/gen_{iter_call}_{task_num}.json"
                    )
                    if os.path.exists(full_file):
                        print(f"Skipping. {full_file} already exists.")
                        continue

                    if not osp.isdir(output_path_scene):
                        os.makedirs(output_path_scene)

                    if not os.path.exists(scene_info_path):
                        with open(scene_info_path, "w+") as f:
                            f.write(json.dumps(self.scene_info, indent=4))

                    if not os.path.exists(config_path):
                        with open(config_path, "w+") as f:
                            OmegaConf.save(self.config, f)

                    (
                        current_instructions,
                        output_dict,
                    ) = self.generate_instructions_on_scene(template_instruction)

                    with open(
                        f"{output_path_scene}/input_prompt_{scene_id}.json", "w+"
                    ) as f:
                        f.write(json.dumps(output_dict))

                    print("writing to ", full_file)
                    with open(full_file, "w+") as f:
                        if type(current_instructions) == str:
                            f.write(current_instructions)
                        else:
                            f.write(json.dumps(current_instructions))

    def validate_scene(self):
        region_furn = self.mi.get_region_rec_contents(self.sim)
        print("All regions and their furniture")
        print(region_furn)

        print("\nAll articulated objs")
        aom = self.sim.get_articulated_object_manager()
        all_ao = list(aom.get_objects_by_handle_substring().values())
        all_ao_info = []
        for obj in all_ao:
            obj_hash = object_shortname_from_handle(obj.handle)
            obj_cat = self.mi.get_object_category(obj_hash)
            all_ao_info.append(obj_cat)
        print(all_ao_info)

        print("\n Scene lexicon")
        scene_lexicon = self.mi.get_scene_lexicon(self.sim)
        print(scene_lexicon)

    def obtain_scene_info(self):
        metadata_dict = default_metadata_dict.copy()

        self.mi = MetadataInterface(metadata_dict)
        self.mi.refresh_scene_caches(self.sim)
        self.validate_scene()

        # Get the objects
        object_names = []
        ovmm_metadata = pd.read_csv(
            os.path.join(
                metadata_dict["metadata_folder"], metadata_dict["obj_metadata"]
            )
        )
        for index in range(ovmm_metadata.shape[0]):
            cat = ovmm_metadata.at[index, "clean_category"]
            object_names.append(cat)
        object_names = list(set(object_names))

        # Get room to id mapping
        room_to_id = {}
        for k, v in self.mi.region_semname_to_id.items():
            room_to_id[k] = self.sim.semantic_scene.regions[v].id

        # Get affordances
        affordances_csv = os.path.join(
            metadata_dict["metadata_folder"], metadata_dict["object_affordances"]
        )

        turned_on_or_off, clean_in_sink, clean_general, fill = [], [], [], []
        object_affordances = [turned_on_or_off, clean_in_sink, clean_general, fill]
        with open(affordances_csv, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                object_affordances[idx] = row

        return {
            "objects": object_names,
            "furniture": self.mi.get_region_rec_contents(self.sim),  # furniture,
            "receptacle_to_handle": self.mi.recobj_semname_to_handle,
            "room_to_id": room_to_id,
            "object_affordances": object_affordances,
        }


class SemiAutomated(InstructionGeneratorHSSD):
    # Will generate the task and json all at once
    # Takes optional pre-annotated tasks as input to guide instruction generation
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.init_prompt_template()
        self.output_length = self.config.output_length

    def init_prompt_template(self):
        prompt_file_task = self.config.prompt_file_task
        prompt_file_init = self.config.prompt_file_init
        with open(prompt_file_task, "r") as f:
            self.prompt_text_task = f.read()

        if len(prompt_file_init) > 1:
            with open(prompt_file_init, "r") as f:
                self.prompt_text_init = f.read()
        else:
            self.prompt_text_init = None

    def generate_instructions_on_scene(self, task):
        k = self.config.generations_per_call
        receptacles_str = ""
        for room, furniture in self.scene_info["furniture"].items():
            if len(furniture) == 0:
                continue
            receptacles_str += f"{room}: " + ", ".join(furniture) + "\n"
        objs_str = "\n".join(self.scene_info["objects"])
        object_affordances = self.scene_info["object_affordances"]
        turned_on_or_off = "\n".join(object_affordances[0])
        clean_in_sink = "\n".join(object_affordances[1])
        clean_general = "\n".join(object_affordances[2])
        fill = "\n".join(object_affordances[3])

        # prompt for templated and free-form instruction generation
        formatted_prompt = self.prompt_text_task.format(
            house_furniture=receptacles_str,
            objects_list=objs_str,
            k=k,
            task=task,
            turned_on_or_off=turned_on_or_off,
            clean_in_sink=clean_in_sink,
            clean_general=clean_general,
            fill=fill,
        )

        llm_answer = self.llm.generate(formatted_prompt, max_length=self.output_length)
        instructions = llm_answer
        return instructions, {"formatted_prompt": formatted_prompt}


@hydra.main(
    version_base=None,
    config_path="../conf/",
    config_name="benchmark_gen.yaml",
)
def main(cfg: omegaconf.DictConfig):
    print("current dir, ", os.getcwd())
    config = OmegaConf.create(cfg)

    # inst_gen = instantiate(config.generator)
    inst_gen = SemiAutomated(config.generator)
    print("scene id: ", inst_gen.config.scene_ids)

    template_file = inst_gen.config.template_file
    with open(template_file, "r") as f:
        tasks = json.load(f)

    inst_gen.generate_instructions(tasks)


if __name__ == "__main__":
    main()
