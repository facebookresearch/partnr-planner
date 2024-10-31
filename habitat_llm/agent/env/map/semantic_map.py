# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import pickle

import magnum as mn
import numpy as np
import skimage.morphology
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN, KMeans

try:
    from third_party.semantic_exploration.models.semantic_map import (  # isort: skip
        Semantic_Mapping,
    )
    from third_party.semantic_exploration.constants import (
        coco_categories,  # isort: skip
    )
except ImportError:
    Semantic_Mapping = None
    pu = None
    coco_categories = None


class SemanticMap:
    def __init__(self, config):
        self.config = config
        # Environment steps counter
        self.i_step = 0
        # Global map update frequency
        self.g_step = (self.i_step // config.NUM_LOCAL_STEPS) % config.NUM_GLOBAL_STEPS
        # Local map update frequency
        self.l_step = self.i_step % config.NUM_LOCAL_STEPS
        self.num_scenes = 1
        self.cur_observation = None
        self.num_sem_categories = config.NUM_SEM_CATEGORIES
        self.last_sim_location = None

        # For querying the map
        self.init_trans = None

    def map_init(self, agent, obs_info, target, init_trans):
        """Function to initialize the map"""
        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Explored Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        num_categories = self.num_sem_categories + 4

        # Use for doing transformation of the agent location
        self.init_trans = init_trans

        # Calculating full and local map sizes
        map_size = self.config.MAP_SIZE_CM // self.config.MAP_RESOLUTION
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / self.config.GLOBAL_DOWNSCALING)
        self.local_h = int(self.full_h / self.config.GLOBAL_DOWNSCALING)

        # Initializing full and local map
        self.global_map = (
            torch.zeros(self.num_scenes, num_categories, self.full_w, self.full_h)
            .float()
            .to(self.config.DEVICE)
        )
        self.local_map = (
            torch.zeros(self.num_scenes, num_categories, self.local_w, self.local_h)
            .float()
            .to(self.config.DEVICE)
        )

        # Initial full and local pose
        self.global_pose = (
            torch.zeros(self.num_scenes, 3).float().to(self.config.DEVICE)
        )
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.config.DEVICE)

        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((self.num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))

        # Initialize the map and the pose
        self.global_map.fill_(0.0)
        self.global_pose.fill_(0.0)
        self.global_pose[:, :2] = self.config.MAP_SIZE_CM / 100.0 / 2.0

        locs = self.global_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / self.config.MAP_RESOLUTION),
                int(c * 100.0 / self.config.MAP_RESOLUTION),
            ]

            self.global_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

            self.lmb[e] = self.get_local_map_boundaries(
                (loc_r, loc_c), (self.local_w, self.local_h), (self.full_w, self.full_h)
            )

            self.planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [
                self.lmb[e][2] * self.config.MAP_RESOLUTION / 100.0,
                self.lmb[e][0] * self.config.MAP_RESOLUTION / 100.0,
                0.0,
            ]

        for e in range(self.num_scenes):
            self.local_map[e] = self.global_map[
                e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]
            ]
            self.local_pose[e] = (
                self.global_pose[e]
                - torch.from_numpy(self.origins[e]).to(self.config.DEVICE).float()
            )

        # Global policy observation space
        ngc = 8 + self.num_sem_categories

        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(self.config).to(self.config.DEVICE)
        self.sem_map_module.eval()

        self.global_input = torch.zeros(
            self.num_scenes, ngc, self.local_w, self.local_h
        )
        self.global_orientation = torch.zeros(self.num_scenes, 1).long()
        self.intrinsic_rews = torch.zeros(self.num_scenes).to(self.config.DEVICE)
        self.extras = torch.zeros(self.num_scenes, 2)

        # Predict semantic map from frame 1
        poses = (
            torch.from_numpy(
                np.asarray(
                    [obs_info["sensor_pose"] for env_idx in range(self.num_scenes)]
                )
            )
            .float()
            .to(self.config.DEVICE)
        )
        # Make it to be a tensor
        # We need agent to get the semantic prediction
        obs = torch.tensor(agent._preprocess_obs(obs_info["state"]))
        obs = torch.unsqueeze(obs, 0)
        _, self.local_map, _, self.local_pose = self.sem_map_module(
            obs,
            poses,
            self.local_map,
            self.local_pose,
        )

        # Compute Global policy input
        locs = self.local_pose.cpu().numpy()
        self.global_input = torch.zeros(
            self.num_scenes, ngc, self.local_w, self.local_h
        )
        self.global_orientation = torch.zeros(self.num_scenes, 1).long()

        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / self.config.MAP_RESOLUTION),
                int(c * 100.0 / self.config.MAP_RESOLUTION),
            ]

            self.local_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
            self.global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)

            # Set a disk around the agent to explore
            try:
                radius = self.config.FRONTIER_EXPLORE_RADIUS
                explored_disk = skimage.morphology.disk(radius)
                self.local_map[
                    e,
                    1,
                    int(r - radius) : int(r + radius + 1),
                    int(c - radius) : int(c + radius + 1),
                ][explored_disk == 1] = 1
            except IndexError:
                pass

        self.global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :].detach()
        self.global_input[:, 4:8, :, :] = nn.MaxPool2d(self.config.GLOBAL_DOWNSCALING)(
            self.global_map[:, 0:4, :, :]
        )
        self.global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()

        self.goal_cat_id = coco_categories[target.lower()]
        goal_cat_id = torch.from_numpy(
            np.asarray([self.goal_cat_id for env_idx in range(self.num_scenes)])
        )

        self.extras = torch.zeros(self.num_scenes, 2)
        self.extras[:, 0] = self.global_orientation[:, 0]
        self.extras[:, 1] = goal_cat_id

        self.goal_maps = [
            np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)
        ]

        planner_inputs = [{} for e in range(self.num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = self.local_map[e, 0, :, :].cpu().numpy()
            p_input["exp_pred"] = self.local_map[e, 1, :, :].cpu().numpy()
            p_input["pose_pred"] = self.planner_pose_inputs[e]
            p_input["goal"] = self.goal_maps[e]  # global_goals[e]
            p_input["new_goal"] = 1
            p_input["found_goal"] = 0
            p_input["wait"] = False
            if self.config.VISUALIZE:
                self.local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = (
                    self.local_map[e, 4:, :, :].argmax(0).cpu().numpy()
                )

        return obs_info, planner_inputs

    def update_step_counter(self):
        # Update the step variable
        self.i_step += 1
        self.g_step = (
            self.i_step // self.config.NUM_LOCAL_STEPS
        ) % self.config.NUM_GLOBAL_STEPS
        self.l_step = self.i_step % self.config.NUM_LOCAL_STEPS

    def map_update(self, agent, obs_info):
        # Semantic Mapping Module
        poses = (
            torch.from_numpy(
                np.asarray(
                    [obs_info["sensor_pose"] for env_idx in range(self.num_scenes)]
                )
            )
            .float()
            .to(self.config.DEVICE)
        )

        # Set people as not obstacles
        self.local_map[:, 0, :, :] *= 1 - self.local_map[:, 19, :, :]

        # Make it to be a tensor
        obs = torch.tensor(agent._preprocess_obs(obs_info["state"]))
        obs = torch.unsqueeze(obs, 0)
        # Update the semantic map
        _, self.local_map, _, self.local_pose = self.sem_map_module(
            obs, poses, self.local_map, self.local_pose
        )

        # Set people as not obstacles for planning
        e = 0
        people_mask = (
            skimage.morphology.binary_dilation(
                self.local_map[e, 15, :, :].cpu().numpy(), skimage.morphology.disk(10)
            )
        ) * 1.0
        self.local_map[e, 0, :, :] *= 1 - torch.from_numpy(people_mask).to(
            self.config.DEVICE
        )

        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs + self.origins
        self.local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / self.config.MAP_RESOLUTION),
                int(c * 100.0 / self.config.MAP_RESOLUTION),
            ]
            self.local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

        # Update the global policy and the map
        if self.l_step == self.config.NUM_LOCAL_STEPS - 1:
            # For every global step, update the full and local maps
            for e in range(self.num_scenes):
                self.update_intrinsic_rew(e)

                self.global_map[
                    e,
                    :,
                    self.lmb[e, 0] : self.lmb[e, 1],
                    self.lmb[e, 2] : self.lmb[e, 3],
                ] = self.local_map[e]
                self.global_pose[e] = (
                    self.local_pose[e]
                    + torch.from_numpy(self.origins[e]).to(self.config.DEVICE).float()
                )

                locs = self.global_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [
                    int(r * 100.0 / self.config.MAP_RESOLUTION),
                    int(c * 100.0 / self.config.MAP_RESOLUTION),
                ]

                self.lmb[e] = self.get_local_map_boundaries(
                    (loc_r, loc_c),
                    (self.local_w, self.local_h),
                    (self.full_w, self.full_h),
                )

                self.planner_pose_inputs[e, 3:] = self.lmb[e]
                self.origins[e] = [
                    self.lmb[e][2] * self.config.MAP_RESOLUTION / 100.0,
                    self.lmb[e][0] * self.config.MAP_RESOLUTION / 100.0,
                    0.0,
                ]

                self.local_map[e] = self.global_map[
                    e,
                    :,
                    self.lmb[e, 0] : self.lmb[e, 1],
                    self.lmb[e, 2] : self.lmb[e, 3],
                ]
                self.local_pose[e] = (
                    self.global_pose[e]
                    - torch.from_numpy(self.origins[e]).to(self.config.DEVICE).float()
                )

            locs = self.local_pose.cpu().numpy()
            for e in range(self.num_scenes):
                self.global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)
            self.global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :]
            self.global_input[:, 4:8, :, :] = nn.MaxPool2d(
                self.config.GLOBAL_DOWNSCALING
            )(self.global_map[:, 0:4, :, :])
            self.global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()
            goal_cat_id = torch.from_numpy(
                np.asarray([self.goal_cat_id for env_idx in range(self.num_scenes)])
            )
            self.extras[:, 0] = self.global_orientation[:, 0]
            self.extras[:, 1] = goal_cat_id

        # Update long-term goal if target object is found
        self.found_goal = [0 for _ in range(self.num_scenes)]
        self.goal_maps = [
            np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)
        ]

        for e in range(self.num_scenes):
            cn = self.goal_cat_id + 4
            if self.local_map[e, cn, :, :].sum() != 0.0:
                cat_semantic_map = self.local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                self.goal_maps[e] = cat_semantic_scores
                self.found_goal[e] = 1

        # Take action and get next observation
        planner_inputs = [{} for e in range(self.num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = self.local_map[e, 0, :, :].cpu().numpy()
            p_input["exp_pred"] = self.local_map[e, 1, :, :].cpu().numpy()
            p_input["pose_pred"] = self.planner_pose_inputs[e]
            p_input["goal"] = self.goal_maps[e]  # global_goals[e]
            p_input["new_goal"] = self.l_step == self.config.NUM_LOCAL_STEPS - 1
            p_input["found_goal"] = self.found_goal[e]
            p_input["wait"] = False
            if self.config.VISUALIZE:
                self.local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = (
                    self.local_map[e, 4:, :, :].argmax(0).cpu().numpy()
                )

        return obs_info, planner_inputs

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.config.GLOBAL_DOWNSCALING > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def update_intrinsic_rew(self, e):
        prev_explored_area = self.global_map[e, 1].sum(1).sum(0)
        self.global_map[
            e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]
        ] = self.local_map[e]
        curr_explored_area = self.global_map[e, 1].sum(1).sum(0)
        self.intrinsic_rews[e] = curr_explored_area - prev_explored_area
        self.intrinsic_rews[e] *= (self.config.MAP_RESOLUTION / 100.0) ** 2  # to m^2

    def map_reset(self, target):
        # Update the goal cat id
        self.goal_cat_id = coco_categories[target.lower()]

    def query_map(self, target_name):
        """Query the map based on the name of the class."""
        try:
            # Get the target ID
            target_name_id = coco_categories[target_name.lower()] + 4
            # Get the map
            cur_map = np.array(self.global_map[0, target_name_id, :, :])
            # Find the location if there is target
            loc_x, loc_y = np.where(cur_map > 0)
            # Make it to be a 2D array
            loc = np.concatenate((loc_x[:, None], loc_y[:, None]), axis=1)
        except Exception:
            return "does not call the explore[] before"

        # Get the means
        try:
            if self.config.CLUSTER_METHOD == "kmeans":
                cluster_model = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(
                    loc
                )
                center = cluster_model.cluster_centers_
            elif self.config.CLUSTER_METHOD == "dbscan":
                cluster_model = DBSCAN(eps=1, min_samples=3).fit(loc)
                labels = cluster_model.labels_
                labels_dict = {}
                for i, l in enumerate(labels):
                    if l not in labels_dict and l != -1:
                        labels_dict[l] = []
                    if l != -1:
                        labels_dict[l].append(loc[i])
                center = []
                for centers in labels_dict:
                    center.append(np.mean(np.array(labels_dict[centers]), axis=0))
        except Exception:
            return "0.0,0.0" + "; "

        answer = ""
        for c in center:
            map_x = (c[1] - self.full_w / 2) * self.config.MAP_RESOLUTION / 100.0
            map_y = (self.full_h / 2 - c[0]) * self.config.MAP_RESOLUTION / 100.0
            target_c = mn.Vector3d([map_x, 0, map_y])
            target_c = self.init_trans.transform_point(target_c)
            answer += str(target_c[0]) + "," + str(target_c[2]) + "; "

        return answer[0:-2]

    def map_io(self, load=False):
        """Load and save the map"""
        dump_dir = self.config.PREBUILD_MAP_DIR
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        if load:
            with open(dump_dir + self.config.PREBUILD_MAP_NAME, "rb") as handle:
                self.global_map, self.init_trans = pickle.load(handle)
                self.init_trans = mn.Matrix4(self.init_trans)
        else:
            with open(dump_dir + self.config.PREBUILD_MAP_NAME, "wb") as handle:
                # Pickle cannot do mn matrix
                pickle.dump([self.global_map, np.array(self.init_trans)], handle)
