# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict

import cv2
import gym
import habitat
import imageio
import numpy as np
import torch
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

# HABITAT
from habitat_baselines.utils.common import batch_obs, get_num_actions
from habitat_sim.utils.viz_utils import depth_to_rgb

from habitat_llm.agent.env.sensors import SENSOR_MAPPINGS
from habitat_llm.perception import PerceptionObs, PerceptionSim
from habitat_llm.sims.metadata_interface import get_metadata_dict_from_config
from habitat_llm.utils.core import separate_agent_idx

# LOCAL
from habitat_llm.world_model import DynamicWorldGraph, WorldGraph

if hasattr(torch, "inference_mode"):
    inference_mode = torch.inference_mode
else:
    inference_mode = torch.no_grad


def camera_spec_to_intrinsics(camera_spec):
    def f(length, fov):
        return length / (2.0 * np.tan(hfov / 2.0))

    hfov = np.deg2rad(float(camera_spec.hfov))
    image_height, image_width = np.array(camera_spec.resolution).tolist()
    fx = f(image_height, hfov)
    fy = f(image_width, hfov)
    cx = image_height / 2.0
    cy = image_width / 2.0
    return np.array([[fx, fy, cx, cy]])


class EnvironmentInterface:
    def __init__(
        self, conf, dataset=None, init_wg=True, init_env=True, gym_habitat_env=None
    ):
        if init_env:
            self.env = habitat.registry.get_env("GymHabitatEnv")(
                config=conf, dataset=dataset
            )
        else:
            if gym_habitat_env is None:
                raise ValueError(
                    "Expected env to be a Habitat Env variable got None instead!"
                )
            self.env = gym_habitat_env
        self.sim = self.env.env.env._env.sim
        self.sim.dynamic_target = np.zeros(3)

        obs = self.env.reset()

        if conf.device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda",
                conf.habitat_baselines.torch_gpu_id,
            )

        self.conf = conf
        self.mappings = SENSOR_MAPPINGS
        self._dry_run = self.conf.dry_run

        # Set human and robot agent uids
        self.robot_agent_uid = self.conf.robot_agent_uid
        self.human_agent_uid = self.conf.human_agent_uid

        # merge metadata config and defaults
        self.metadata_dict = get_metadata_dict_from_config(conf.habitat.dataset)

        # Create instance perceptionSim and WorldModel
        # FIXME: below is same as self.wm_update_mode, remove one in favor of other
        self.perception_mode = conf.world_model.update_mode
        if init_wg:
            self.initialize_perception_and_world_graph()

        if "main_agent" in self.conf.trajectory.agent_names:
            self._single_agent_mode = True
        else:
            self._single_agent_mode = False

        self.ppo_cfg = conf.habitat_baselines.rl.ppo

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.obs_transforms = get_active_obs_transforms(conf)

        self.orig_action_space = self.env.original_action_space
        self.observation_space = apply_obs_transforms_obs_space(
            self.observation_space, self.obs_transforms
        )

        self.__get_internal_obs_space()

        self.frames = []
        self.batch = self.__parse_observations(obs)
        # self.reset_environment()

        # Container to store state history of both agents
        self.agent_state_history = defaultdict(list)

        # Container to store actions history of both agents
        self.agent_action_history = defaultdict(list)

        # container to store results from composite skills
        self._composite_action_response = {}
        # a dictionary where key == agent_uid and the value is a tuple with
        # ( "last-action", "arg-string", "result")

        # empty variables to store the trajectory data initialized in
        # setup_logging_for_current_episode when save_trajectory is True
        self.save_trajectory: bool = self.conf.trajectory.save
        self.save_options: list = None
        self.trajectory_agent_names: list = None
        self.trajectory_save_paths: Dict[str, str] = None
        self.trajectory_save_prefix: str = None
        self._trajectory_idx: int = None
        self._setup_current_episode_logging: bool = False

    def initialize_perception_and_world_graph(self):
        """
        This method initializes perception and world graph
        """
        # Create instance of perception
        if self.perception_mode == "gt":
            self.perception = PerceptionSim(self.sim, self.metadata_dict)
        else:
            self.perception = PerceptionObs(self.sim, self.metadata_dict)
        # Set the partial observability flag
        self.partial_obs = self.conf.world_model.partial_obs

        # set update mode flag: str: gt or obs
        self.wm_update_mode: str = self.conf.world_model.update_mode

        # Create instance of the world model
        # static world-graph for full obs setting
        # dynamic world-graph for partial obs setting

        # each agent has its own world-graph
        self.world_graph: Dict[int, Any] = {}

        # create world-graphs for both agents
        if self.partial_obs:
            self.world_graph = {
                self.robot_agent_uid: DynamicWorldGraph(),
                self.human_agent_uid: DynamicWorldGraph(),
            }
        else:
            self.world_graph = {
                self.robot_agent_uid: WorldGraph(),
                self.human_agent_uid: WorldGraph(),
            }

        # set agent-asymmetry flag if True
        if self.conf.agent_asymmetry:
            for agent_key in self.world_graph:
                self.world_graph[agent_key].agent_asymmetry = True

        # set articulated agents for each dynamic world-graph
        articulated_agents = {}
        for agent_uid in self.world_graph:
            articulated_agents[agent_uid] = self.sim.agents_mgr[
                agent_uid
            ].articulated_agent
        for agent_id in self.world_graph:
            if isinstance(self.world_graph[agent_id], DynamicWorldGraph):
                self.world_graph[agent_id].set_articulated_agents(articulated_agents)

        # maintain a copy of fully-observable world-graph
        self.full_world_graph = WorldGraph()
        most_recent_graph = self.perception.initialize(False)
        self.full_world_graph.update(most_recent_graph, False, "gt", add_only=True)

        # based on the type of world-model being used, setup the data-source
        if self.conf.world_model.type == "concept_graph":
            self.world_graph[self.robot_agent_uid].world_model_type = "non_privileged"
            # initialize the human agent's world-graph from sim with partial observability
            subgraph = self.perception.initialize(partial_obs=self.partial_obs)
            self.world_graph[self.conf.human_agent_uid].update(
                subgraph, self.partial_obs, "gt", add_only=True
            )

            # initialize robot agent's world-graph from CG
            cg_json = None
            cg_json_path = self.conf.world_model.world_model_data_path

            # CG for a given scene should be read from data based on scene-id
            current_episode_metadata = self.env.env.env._env.current_episode
            current_episode_id = current_episode_metadata.episode_id
            current_scene_id = current_episode_metadata.scene_id
            glob_expr = os.path.join(cg_json_path, f"*{current_scene_id}.json")
            cg_file = glob.glob(glob_expr)

            # handle the case if there is not CG for this scene
            if not cg_file:
                raise FileNotFoundError(
                    f"Skipping Episode# {current_episode_id}, Scene# {current_scene_id} as we do not have CG for this",
                )
            if len(cg_file) > 1:
                raise RuntimeError(
                    f"Found more than 1 CG for scene: {current_scene_id}; skipping",
                )
            print(f"Found 1 CG for scene: {current_scene_id}; file: {cg_file[0]}")
            with open(cg_file[0], "r") as f:
                cg_json = json.load(f)

            if not isinstance(
                self.world_graph[self.conf.robot_agent_uid], DynamicWorldGraph
            ):
                raise ValueError(
                    "Expected robot's world-graph to be of type DynamicWorldGraph, however found: ",
                    type(self.world_graph[self.conf.robot_agent_uid]),
                )
            self.world_graph[self.conf.robot_agent_uid].create_cg_edges(
                cg_json, include_objects=self.conf.world_model.include_objects
            )
            self.world_graph[self.conf.robot_agent_uid].initialize_agent_nodes(subgraph)
            self.world_graph[
                self.robot_agent_uid
            ]._set_sim_handles_for_non_privileged_graph(self.perception)
        elif self.conf.world_model.type == "gt_graph":
            subgraph = self.perception.initialize(self.partial_obs)
            # Get ground truth subgraph from the current observations.
            # since the graph is being initialized, we only add the new nodes and edges
            for agent_key in self.world_graph:
                self.world_graph[agent_key].update(
                    subgraph, self.partial_obs, self.wm_update_mode, add_only=True
                )
        else:
            raise ValueError(
                f"World model not implemented for type: {self.conf.world_model.type}"
            )

        return

    def get_observations(self):
        """
        Obtains the environment observations. In the form of a dictionary of tensors.
        """
        return self.batch

    def parse_observations(self, obs):
        return self.__parse_observations(obs)

    def reset_environment(self, move_to_next_episode=True, episode_id=None):
        """
        Resets the environment, moving to the next episode and obtaining a new set of observations
        :param move_to_next_episode: by default, reset moves to the next episode. If set to False, will reset the environment to the same episode.
        :param episode_id: If set, reset the environment to a given episode id. Otherwise, moves to the next episode.
        """
        if not move_to_next_episode:
            # We set this variable to reset the environment but stay in the same episode
            self.env.env.env._env.current_episode = (
                self.env.env.env._env.current_episode
            )

        if episode_id is not None:
            assert type(episode_id) == str
            episode_interest = [
                epi
                for epi in self.env.env.env._env._dataset.episodes
                if epi.episode_id == episode_id
            ][0]
            self.env.env.env._env.current_episode = episode_interest

        obs = self.env.reset()
        self.batch = self.__parse_observations(obs)
        self.initialize_perception_and_world_graph()
        self.sim = self.env.env.env._env.sim
        self.reset_internals()
        self.video_name = "debug.mp4"

        # if self.frames != []:
        #     self.__make_video()

        self.frames = []

        # Container to store state history of agents
        self.agent_state_history = defaultdict(list)

        # Container to store action history of agents
        self.agent_action_history = defaultdict(list)

        # reset episode logging to create dir structure for new episode
        self.reset_logging()

    def reset_internals(self):
        self.recurrent_hidden_states = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            self.conf.habitat_baselines.rl.ddppo.num_recurrent_layers
            * 2,  # TODO why 2?
            self.ppo_cfg.hidden_size,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            *(get_num_actions(self.action_space),),
            device=self.device,
            dtype=torch.float,
        )
        self.not_done_masks = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )

    def reset_logging(self):
        # empty variables to store the trajectory data initialized in
        # setup_logging_for_current_episode when save_trajectory is True
        self.save_options = None
        self.trajectory_agent_names = None
        self.trajectory_save_paths = None
        self.trajectory_save_prefix = None
        self._trajectory_idx = None
        self._setup_current_episode_logging = False

    def reset_composite_action_response(self):
        """resets _composite_action_response to empty"""
        self._composite_action_response = {}

    def setup_logging_for_current_episode(self):
        """
        book-keeping to dump out trajectories of given agents
        see config: conf/trajectory/trajectory_logger.yaml for details
        """
        current_episode = self.env.env.env._env.current_episode
        self.save_trajectory = self.conf.trajectory.save
        self.save_options = []
        self.trajectory_agent_names = []
        if self.save_trajectory:
            self.trajectory_agent_names = self.conf.trajectory.agent_names
            assert len(self.trajectory_agent_names) > 0
            self.trajectory_save_prefix = (
                self.conf.trajectory.save_path
                + f"epidx_{current_episode.episode_id}_scene_{current_episode.scene_id}"
            )
            self.save_options = self.conf.trajectory.save_options
            self._trajectory_idx = 0
            self.trajectory_save_paths = {}

            # create a parent directory for given episode/scene combo
            # then create agent-specific directories within it for each agent
            for curr_agent, camera_source in zip(
                self.trajectory_agent_names, self.conf.trajectory.camera_prefixes
            ):
                # sensor_uuid is different depending upon if this is a single-agent
                # or multi-agent planning problem; we expect config to send in
                # consistent naming here
                if self._single_agent_mode:
                    sensor_uuid = f"{camera_source}_rgb"
                else:
                    sensor_uuid = f"{curr_agent}_{camera_source}_rgb"
                self.trajectory_save_paths[curr_agent] = os.path.join(
                    self.trajectory_save_prefix, curr_agent
                )
                if not os.path.exists(self.trajectory_save_paths[curr_agent]):
                    os.makedirs(self.trajectory_save_paths[curr_agent])

                    # save intrinsics for current agent
                    intrinsics_array = camera_spec_to_intrinsics(
                        self.sim.agents[0]._sensors[sensor_uuid].specification()
                    )
                    np.save(
                        os.path.join(
                            self.trajectory_save_paths[curr_agent], "intrinsics.npy"
                        ),
                        intrinsics_array,
                    )

                    # create other sub-directories
                    if "rgb" in self.save_options:
                        os.makedirs(
                            os.path.join(self.trajectory_save_paths[curr_agent], "rgb")
                        )
                    if "depth" in self.save_options:
                        os.makedirs(
                            os.path.join(
                                self.trajectory_save_paths[curr_agent], "depth"
                            )
                        )
                    if "panoptic" in self.save_options:
                        os.makedirs(
                            os.path.join(
                                self.trajectory_save_paths[curr_agent], "panoptic"
                            )
                        )
                    os.makedirs(
                        os.path.join(self.trajectory_save_paths[curr_agent], "pose")
                    )

    def get_final_action_vector(
        self, low_level_actions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Takes in low_level_actions and returns a joint-space final_action_vector over all agents
        """

        # Get list of action tensors
        low_level_action_list = list(low_level_actions.values())

        # Make sure that the actions are never None
        if any(action is None for action in low_level_action_list):
            raise ValueError("low level actions cannot be None!")

        # Construct final actions vector
        if len(low_level_action_list) == 0:
            raise Exception("Cannot step through environment without low level actions")
        if len(low_level_action_list) == 1:
            final_action_vector = low_level_action_list[0]
        elif len(low_level_action_list) == 2:
            final_action_vector = low_level_action_list[0] + low_level_action_list[1]

        return final_action_vector

    def update_world_graphs(self, obs: Dict[str, Any]):
        """
        Simulates perception step using sim for GT condition and observations for non-GT condition
        Additionally, saves this trajectory step if trajectory_logger is enabled
        """

        # Update fully observed world graph (ground truth)
        # This graph is used when planner is working under full observability
        # THis graph is also used by skills to check if a certain furniture is articulated or not etc.
        most_recent_graph = self.perception.get_recent_graph()
        self.full_world_graph.update(
            most_recent_graph, partial_obs=False, update_mode="gt"
        )

        # Update agents world graphs using concept graph
        if self.conf.world_model.type == "concept_graph" and isinstance(
            self.perception, PerceptionObs
        ):
            self.update_world_graphs_using_concept_graph(obs)

        # Update agents world graphs using simulator
        elif self.conf.world_model.type == "gt_graph" and not isinstance(
            self.perception, PerceptionObs
        ):
            self.update_world_graphs_using_sim(obs)

        # if applicable save the data from trajectory step
        self.save_trajectory_step(obs)

        return

    def update_world_graphs_using_sim(self, obs):
        """
        This method updates world graphs for both agents using
        simulated perception and simulated graph
        """

        # Case 1: FULL OBSERVABILITY
        # Set both agents graphs equal to full world graph
        if not self.partial_obs:
            for agent_uid in self.world_graph:
                self.world_graph[agent_uid] = self.full_world_graph

        # Case 2: PARTIAL OBSERVABILITY
        # Update both graphs using both human and robot observations
        else:
            # Get robots subgraph using both human and robot observations
            most_recent_robot_subgraph = self.perception.get_recent_subgraph(
                [str(self.robot_agent_uid), str(self.human_agent_uid)], obs
            )

            # Get human subgraph using only human observations
            observation_sources = []
            if self.conf.agent_asymmetry:
                # under asymmetry condition human's world-graph only uses human's own observations
                observation_sources = [str(self.human_agent_uid)]
            else:
                # under symmetry condition the human's world-graph uses both human's and Spot's observations
                observation_sources = [
                    str(self.robot_agent_uid),
                    str(self.human_agent_uid),
                ]
            most_recent_human_subgraph = self.perception.get_recent_subgraph(
                observation_sources,
                obs,
            )

            # Update robot graph
            self.world_graph[self.robot_agent_uid].update(
                most_recent_robot_subgraph, self.partial_obs, self.wm_update_mode
            )

            # Update human graph
            self.world_graph[self.human_agent_uid].update(
                most_recent_human_subgraph, self.partial_obs, self.wm_update_mode
            )

        return

    def update_world_graphs_using_concept_graph(self, obs):
        """
        This method updates world graphs for both agents using
        concept graph and simulator. Under concept graph regime,
        we always operate under partial observability.
        """
        # process obs to get objects detected in the frame
        if not isinstance(self.perception, PerceptionObs):
            raise ValueError(
                "Concept graph update mode requires PerceptionObs object for perception"
            )
        processed_obs = self.perception.preprocess_obs_for_non_privileged_graph_update(
            self.sim, obs, single_agent_mode=self._single_agent_mode
        )
        # get frame-description from perception
        object_detections_in_frame = (
            self.perception.get_object_detections_for_non_privileged_graph_update(
                processed_obs
            )
        )
        # update robot's WG using object detections
        full_state_object_dict = self.sim.object_state_machine.get_snapshot_dict(
            self.sim
        )
        if object_detections_in_frame is not None:
            self.world_graph[
                self.robot_agent_uid
            ].update_non_privileged_graph_with_detected_objects(
                object_detections_in_frame,
                object_state_dict=full_state_object_dict,
            )
        else:
            raise ValueError("Frame description is None")
        # update the world-graph for the human agent
        if self.conf.agent_asymmetry:
            most_recent_human_subgraph = self.perception.get_recent_subgraph(
                self.sim, [str(self.human_agent_uid)], obs
            )
        else:
            most_recent_human_subgraph = self.perception.get_recent_subgraph(
                self.sim,
                [str(self.robot_agent_uid), str(self.human_agent_uid)],
                obs,
            )
        self.world_graph[self.conf.human_agent_uid].update(
            most_recent_human_subgraph,
            self.partial_obs,
            "gt",  # human's WG is always updated in privileged mode
        )

    def get_frame_description(self, obs):
        """
        This method returns frame_description which is used to update world graph
        when using conceptgraph
        """
        raise NotImplementedError(
            "Processing frames for object-descriptions for CG updates is not implemented yet"
        )

    def save_trajectory_step(self, obs):
        # save data from this time-step; for current episode_id and scene
        # also save the episode description in folder
        if self.save_trajectory and self.trajectory_agent_names is not None:
            for curr_agent, camera_source in zip(
                self.trajectory_agent_names, self.conf.trajectory.camera_prefixes
            ):
                if "rgb" in self.save_options:
                    if self._single_agent_mode:
                        rgb = obs[f"{camera_source}_rgb"]
                    else:
                        rgb = obs[f"{curr_agent}_{camera_source}_rgb"]
                    np.save(
                        f"{self.trajectory_save_paths[curr_agent]}/rgb/{self._trajectory_idx}.npy",
                        rgb,
                    )
                    imageio.imwrite(
                        f"{self.trajectory_save_paths[curr_agent]}/rgb/{self._trajectory_idx}.jpg",
                        rgb,
                    )
                if "depth" in self.save_options:
                    if self._single_agent_mode:
                        depth = obs[f"{camera_source}_depth"]
                    else:
                        depth = obs[f"{curr_agent}_{camera_source}_depth"]
                    depth_image = depth_to_rgb(depth)
                    cv2.imwrite(
                        f"{self.trajectory_save_paths[curr_agent]}/depth/{self._trajectory_idx}.png",
                        depth_image,
                    )
                    np.save(
                        f"{self.trajectory_save_paths[curr_agent]}/depth/{self._trajectory_idx}.npy",
                        depth,
                    )
                if "panoptic" in self.save_options:
                    if self._single_agent_mode:
                        panoptic = obs[f"{camera_source}_panoptic"]
                    else:
                        panoptic = obs[f"{curr_agent}_{camera_source}_panoptic"]
                    cv2.imwrite(
                        f"{self.trajectory_save_paths[curr_agent]}/panoptic/{self._trajectory_idx}.png",
                        panoptic,
                    )
                    np.save(
                        f"{self.trajectory_save_paths[curr_agent]}/panoptic/{self._trajectory_idx}.npy",
                        panoptic,
                    )
                # NOTE: this assumes poses for head_rgb and head_depth are the exact
                # same
                if self._single_agent_mode:
                    pose = np.linalg.inv(
                        self.sim.agents[0]
                        ._sensors[f"{camera_source}_rgb"]
                        .render_camera.camera_matrix
                    )
                else:
                    pose = np.linalg.inv(
                        self.sim.agents[0]
                        ._sensors[f"{curr_agent}_{camera_source}_rgb"]
                        .render_camera.camera_matrix
                    )
                np.save(
                    f"{self.trajectory_save_paths[curr_agent]}/pose/{self._trajectory_idx}.npy",
                    pose,
                )
                # NOTE: another way of accessing camera pose
                # fixed_pose = get_camera_transform(
                #     self.sim.agents_mgr._all_agent_data[
                #         0
                #     ].articulated_agent,
                #     camera_name=f"{curr_agent}_{camera_source}_rgb",
                # )
                # inv_T = self.sim._default_agent.scene_node.transformation
                # fixed_pose = inv_T @ fixed_pose
            self._trajectory_idx += 1

    @property
    def agents(self):
        """
        Return the agents defined in this environment
        """
        return self.sim.agents_mgr._all_agent_data

    def step(self, low_level_actions):
        """
        This method performs a single step through the environment given list of
        low level action vectors for one or both of the agents.
        """

        # Setup trajectory logging mechanism if not already done
        if self.save_trajectory and not self._setup_current_episode_logging:
            self.setup_logging_for_current_episode()
            self._setup_current_episode_logging = True

        # get the joint final_action_vector
        final_action_vector = self.get_final_action_vector(low_level_actions)

        # PHYSICS!!!
        obs, reward, done, info = self.env.step(final_action_vector)

        # Update world graphs
        self.update_world_graphs(obs)

        return obs, reward, done, info

    def filter_obs_space(self, batch, agent_uid):
        """
        This method returns observations belonging to the specified agent
        """
        agent_name = f"agent_{agent_uid}"
        agent_name_bar = f"{agent_name}_"
        output_batch = {
            obs_name.replace(agent_name_bar, ""): obs_value
            for obs_name, obs_value in batch.items()
            if agent_name in obs_name
        }
        return output_batch

    def __get_internal_obs_space(self):
        inner_observation_space = {}
        for key, value in self.observation_space.items():
            agent_id, no_agent_id_key = separate_agent_idx(key)
            if no_agent_id_key in self.mappings:
                inner_observation_space[
                    agent_id + "_" + self.mappings[no_agent_id_key]
                ] = value
            else:
                inner_observation_space[key] = value
        self.internal_observation_space = gym.spaces.Dict(inner_observation_space)

    def __parse_observations(self, obs):
        new_obs = []
        if self._single_agent_mode:
            for key in sorted(obs.keys()):
                new_obs.append((key, obs[key]))
        else:
            for key in sorted(obs.keys()):
                agent_id, no_agent_id_key = separate_agent_idx(key)
                if no_agent_id_key in self.mappings:
                    new_obs.append(
                        (agent_id + "_" + self.mappings[no_agent_id_key], obs[key])
                    )
                else:
                    new_obs.append((key, obs[key]))
        new_obs = [OrderedDict(new_obs)]
        batch = batch_obs(new_obs, device=self.device)
        return apply_obs_transforms_batch(batch, self.obs_transforms)
