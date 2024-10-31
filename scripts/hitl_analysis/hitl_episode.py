# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import gzip
import json
from typing import Dict, List, Optional

import habitat
import numpy as np
from habitat.config.default_structured_configs import ThirdRGBSensorConfig
from omegaconf import OmegaConf

from habitat_llm.agent.env import register_measures, register_sensors
from habitat_llm.agent.env.dataset import CollaborationDatasetV0


def load_json_gzip(filepath):
    with gzip.open(filepath, "rt") as f:
        dataset = json.load(f)
    return dataset


def divide_list(lst, n_cpus=10):
    n = len(lst)
    k = n_cpus
    # Calculate the size of each sublist
    size = n // k
    # Calculate the number of sublists that need an extra element
    remainder = n % k

    sublists = []
    start = 0
    for i in range(k):
        # If there are remainders left, add an extra element to the sublist
        end = start + size + (1 if i < remainder else 0)
        sublists.append(lst[start:end])
        start = end

    return sublists


class HITLSession:
    """
    A session containing multiple episodes
    """

    def __init__(
        self,
        file_list: Optional[List[str]] = None,
        multi: bool = False,
        hitl_data_file: str = "",
    ) -> None:
        if file_list is None:
            file_list = []

        self.multi_user = multi
        self.file_list = file_list
        self.episodes_info = load_json_gzip(hitl_data_file)
        self.episodes = self.get_episodes()

        # set the data path
        for i in range(len(self.episodes)):
            cfg = self.episodes[i].hitl_data["session"]["config"]
            cfg["habitat"]["dataset"]["data_path"] = hitl_data_file

    def get_episodes(self):
        episodes = []
        eid_to_episode = {ep["episode_id"]: ep for ep in self.episodes_info["episodes"]}

        for file in self.file_list:
            if file == "session.json.gz":
                continue
            if ".json.gz" not in file:
                continue
            print("loading file ", file)
            content_session = load_json_gzip(file)
            eid = content_session["episode"]["episode_id"]
            dataset_content = eid_to_episode[eid]
            episode = HITLEpisode(content_session, dataset_content, self.multi_user)
            episodes.append(episode)
        return episodes


class HITLEpisode:
    def __init__(self, episode_hitl_data, episode_info, multi=False):
        self.hitl_data = episode_hitl_data
        self.episode_info = episode_info
        self.multi_user = multi

    def sample_frames_at_frequency(self, delta_t=1):
        """
        Discretizes the events in the episodes into frames, assuming that each frame lasts delta_t in sim
        returns the event index where each frame happens, the timestamp of that event and the events that happened between
        the current frame and the past frame.
        """
        episode_dict = [frame for frame in self.hitl_data["frames"] if len(frame) != 0]
        t1 = episode_dict[0]["t"]
        tf = episode_dict[-1]["t"]
        n_frames = int((tf - t1) / delta_t)
        curr_frame = 0
        last_frame_events = 0
        frames_indices = [0 for _ in range(n_frames)]
        timestamps = [0 for _ in range(n_frames)]
        timestamp = t1
        events_log = []
        for ind in range(n_frames):
            timestamps[ind] = timestamp
            while (
                curr_frame < (len(episode_dict) - 1)
                and episode_dict[curr_frame + 1]["t"] <= timestamp
            ):
                curr_frame += 1

                if curr_frame < len(episode_dict) - 1:
                    # If not the last frame, we look which frame is closer
                    ta = timestamp - episode_dict[curr_frame]["t"]
                    tb = episode_dict[curr_frame + 1]["t"] - timestamp
                    if tb < ta:
                        curr_frame += 1
            timestamp += delta_t

            frames_indices[ind] = curr_frame
            events_so_far = [[], []]
            # print(curr_frame, last_frame_events, curr_frame+1)

            for frame_ind in range(last_frame_events, curr_frame + 1):
                events_so_far[0] += episode_dict[frame_ind]["users"][0]["events"]
                if self.multi_user:
                    events_so_far[1] += episode_dict[frame_ind]["users"][1]["events"]

            events_log.append(events_so_far)

            last_frame_events = curr_frame
        return frames_indices, timestamps, events_log


def init_env(
    session: HITLSession,
    episode_ids: Optional[List[str]] = None,
    cfg_dict: Optional[Dict] = None,
):
    # Init an environment replicating the session ran for the users
    try:
        cfg_dict = session.episodes[0].hitl_data["session"]["config"]
    except Exception:
        cfg_dict = cfg_dict
    cfg = OmegaConf.create(cfg_dict)
    multi = "agent_1" in cfg.habitat.simulator.agents
    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.agents_order = sorted(cfg.habitat.simulator.agents.keys())
        cfg.habitat.dataset.content_scenes = ["*"]
        cfg.habitat.simulator.additional_object_paths = list(
            cfg.habitat.simulator.additional_object_paths.values()
        )
        cfg.habitat.gym.obs_keys = list(cfg.habitat.gym.obs_keys.values())
        cfg.habitat.simulator.agents.agent_0.start_position = list(
            cfg.habitat.simulator.agents.agent_0.start_position.values()
        )
        cfg.habitat.simulator.agents.agent_0.start_rotation = list(
            cfg.habitat.simulator.agents.agent_0.start_rotation.values()
        )
        cfg.habitat.task.desired_resting_position = list(
            cfg.habitat.task.desired_resting_position.values()
        )
        cfg.habitat.simulator.create_renderer = True
        cfg.habitat.simulator.requires_textures = True

        for sensor_cfg in cfg.habitat.simulator.agents.agent_0.sim_sensors.values():
            sensor_cfg.orientation = list(sensor_cfg.orientation.values())
            sensor_cfg.position = list(sensor_cfg.position.values())

        if (
            "agent_0_third_rgb_sensor"
            not in cfg.habitat.simulator.agents.agent_0.sim_sensors
        ):
            # Add this sensor so that we can plot
            cfg.habitat.simulator.agents.agent_0.sim_sensors[
                "agent_0_third_rgb_sensor"
            ] = ThirdRGBSensorConfig()

        if multi:
            if "auto_update_sensor_transform" in cfg.habitat.simulator.agents.agent_0:
                del cfg.habitat.simulator.agents.agent_1["auto_update_sensor_transform"]

            cfg.habitat.simulator.agents.agent_1.start_rotation = list(
                cfg.habitat.simulator.agents.agent_1.start_rotation.values()
            )
            cfg.habitat.simulator.agents.agent_1.start_position = list(
                cfg.habitat.simulator.agents.agent_1.start_position.values()
            )
            for sensor_cfg in cfg.habitat.simulator.agents.agent_1.sim_sensors.values():
                sensor_cfg.orientation = list(sensor_cfg.orientation.values())
                sensor_cfg.position = list(sensor_cfg.position.values())

        # TOOD: there are a few more confgis that are saved as dicts
    print("Register sensors and measures")
    register_sensors(cfg)
    register_measures(cfg)
    print("Starting env...")
    if episode_ids is None:
        env = habitat.Env(config=cfg)
    else:
        dataset = CollaborationDatasetV0(cfg.habitat.dataset)
        episode_subset = [
            ep for ep in dataset.episodes if ep.episode_id in set(episode_ids)
        ]
        new_dataset = CollaborationDatasetV0(
            config=cfg.habitat.dataset, episodes=episode_subset
        )
        env = habitat.Env(config=cfg, dataset=new_dataset)

    env.sim.dynamic_target = np.zeros(3)

    return env
