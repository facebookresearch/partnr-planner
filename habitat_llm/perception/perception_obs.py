#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
from typing import Any, Dict, List

import cv2
import numpy as np
from habitat_sim.utils.viz_utils import depth_to_rgb

from habitat_llm.perception.perception_sim import PerceptionSim


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.
    input: fx, fy, cx, cy

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


class PerceptionObs(PerceptionSim):
    """
    This class uses only the simulated panoptic sensors to detect objects and then
    ground there location based on depth images being streamed by the agents. Note that
    no other privileged information about the state of the world is used to enhance
    object location or inter-object relations. We use previously detected objects
    and furniture through CG to ground properties of newer objects detected through
    panoptic sensors.
    """

    def __init__(self, sim, metadata_dict: Dict[str, str], *args, **kwargs):
        super().__init__(sim, metadata_dict=metadata_dict, detectors=["gt_panoptic"])

        # a list of cached images for debugging
        self._iteration = 0
        self._verbose = True

    def preprocess_obs_for_non_privileged_graph_update(
        self, sim, obs: Dict[str, Any], single_agent_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """
        ONLY FOR NON-PRIVILEGED GRAPH SETTING
        Creates a list of observations for each agent in the scene. Each observation
        contains the RGB, depth, masks, camera intrinsics and camera pose for the agent.
        """
        processed_obs = []

        def f(length, hfov):
            return length / (2.0 * np.tan(hfov / 2.0))

        # pad obs with more information about the camera
        for uid in range(2):  # NOTE: hardcoded to use obs from both agents
            agent_obs = {}
            if uid == 0:
                if single_agent_mode:
                    try:
                        camera_spec = (
                            sim.agents[0]
                            ._sensors["articulated_agent_jaw_depth"]
                            .specification()
                        )
                    except ValueError:
                        raise ValueError(
                            f"Expecting `articulated_agent_jaw_depth` in obs-keys in single-agent mode, received: {sim.agent[0]._sensors.keys()=}"
                        )
                else:
                    try:
                        camera_spec = (
                            sim.agents[0]
                            ._sensors["agent_0_articulated_agent_jaw_depth"]
                            .specification()
                        )
                    except ValueError:
                        raise ValueError(
                            f"Expecting `agent_0_articulated_agent_jaw_depth` in obs-keys in multi-agent mode, received: {sim.agent[0]._sensors.keys()=}"
                        )
            elif uid == 1:
                try:
                    camera_spec = (
                        sim.agents[0]._sensors["agent_1_head_depth"].specification()
                    )
                    # breakpoint()
                except ValueError:
                    raise ValueError(
                        f"Expecting `agent_1_head_depth` in obs-keys in multi-agent mode, received: {sim.agent[1]._sensors.keys()=}"
                    )
            hfov = np.deg2rad(float(camera_spec.hfov))
            image_height, image_width = np.array(camera_spec.resolution).tolist()
            fx = f(image_height, hfov)
            fy = f(image_width, hfov)
            cx = image_height / 2.0
            cy = image_width / 2.0
            agent_obs["camera_intrinsics"] = np.array([fx, fy, cx, cy])
            # not try/excepting here, hoping the one above would catch all agent_obs
            # malformation issues
            if uid == 0:
                if single_agent_mode:
                    agent_obs["camera_pose"] = np.linalg.inv(
                        sim.agents[0]
                        ._sensors["articulated_agent_jaw_depth"]
                        .render_camera.camera_matrix
                    )
                    # agent_obs["rgb"] = obs["articulated_agent_jaw_rgb"]
                    agent_obs["depth"] = obs["articulated_agent_jaw_depth"]
                    agent_obs["masks"] = obs["articulated_agent_jaw_panoptic"]
                else:
                    agent_obs["camera_pose"] = np.linalg.inv(
                        sim.agents[0]
                        ._sensors["agent_0_articulated_agent_jaw_depth"]
                        .render_camera.camera_matrix
                    )
                    # agent_obs["rgb"] = obs["agent_0_articulated_agent_jaw_rgb"]
                    agent_obs["depth"] = obs["agent_0_articulated_agent_jaw_depth"]
                    agent_obs["masks"] = obs["agent_0_articulated_agent_jaw_panoptic"]
            elif uid == 1:
                agent_obs["camera_pose"] = np.linalg.inv(
                    sim.agents[0]
                    ._sensors["agent_1_head_depth"]
                    .render_camera.camera_matrix
                )
                # agent_obs["rgb"] = obs["agent_1_head_rgb"]
                agent_obs["depth"] = obs["agent_1_head_depth"]
                agent_obs["masks"] = obs["agent_1_head_panoptic"]
            processed_obs.append(agent_obs)
        return processed_obs

    def get_sim_handle_and_key_from_panoptic_image(
        self, obs: np.ndarray
    ) -> Dict[int, str]:
        """
        This method uses the instance segmentation output to
        create a list of handles of all objects present in given agent's FOV
        """

        idx_to_handle_map: Dict[int, str] = {}

        unique_obj_ids = np.unique(obs)
        # 100 gets added to object IDs that are recognized by ROM/AOM in sim/lab
        # subtracting 100 here to get the original object ID
        unique_obj_ids = [idx - 100 for idx in unique_obj_ids if idx != 0]
        for obj_idx in unique_obj_ids:
            if obj_idx != 0:
                sim_object = self.rom.get_object_by_id(obj_idx)
                if sim_object is not None:
                    # add 100 to object ID to get the object ID recognized by ROM/AOM
                    idx_to_handle_map[obj_idx + 100] = sim_object.handle

        return idx_to_handle_map

    def _sim_handles_to_categories(self, id_to_handle_mapping: Dict[int, str]):
        """
        This method maps the object handles to their name
        """
        pop_list = []
        id_to_object_mapping: dict[int, str] = {}
        for obj_idx in id_to_handle_mapping:
            if id_to_handle_mapping[obj_idx] in self.sim_handle_to_name:
                match = re.match(
                    r"^(.*)_\d+$",
                    self.sim_handle_to_name[id_to_handle_mapping[obj_idx]],
                )
                if match:
                    id_to_object_mapping[obj_idx] = match.group(1)
            else:
                pop_list.append(obj_idx)
        for obj_idx in pop_list:
            id_to_handle_mapping.pop(obj_idx)
        return id_to_object_mapping

    def get_object_detections_for_non_privileged_graph_update(
        self, input_obs: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        ONLY FOR NON-PRIVILEGED GRAPH SETTING
        Use the panoptic sensor to detect objects in the agent's FOV

        NOTE: We calculate location for objects seen by robot using RGB-D images and
        camera intrinsics + extrinsics. For Human we use sim information as there is a
        known bug in using the RGB-D images for humans.
        """
        object_detections = {}
        for uid, obs in enumerate(input_obs):
            # extract RGB from obs and pass it to detectors
            out_img = obs["masks"]
            # in_img = obs["rgb"]
            depth = obs["depth"]

            depth_to_rgb(depth)

            # Get handles of all objects and receptacles in agent's FOVs
            id_to_handle_mapping = self.get_sim_handle_and_key_from_panoptic_image(
                out_img
            )

            idx_to_name_mapping = {}
            for idx, handle in id_to_handle_mapping.items():
                if handle in self.sim_handle_to_name:
                    idx_to_name_mapping[idx] = self.sim_handle_to_name[handle]

            id_to_object_mapping = self._sim_handles_to_categories(id_to_handle_mapping)

            # semantic_input is a dictionary with object index as key and mask as
            # value. masks are of shape (H, W, 1) with 1s at the location of object

            semantic_input: Dict[int, np.ndarray] = {}
            locations = {}

            # TODO: add a centered-obj heuristic. Don't consider objects on the boundary
            # of the image
            for _, obj_idx in enumerate(id_to_object_mapping):
                mask = (out_img == obj_idx).astype(np.uint8)
                mask_H, mask_W, _ = mask.shape
                img_H, img_W, _ = depth.shape
                if mask_H != img_H or mask_W != img_W:
                    mask = np.expand_dims(
                        cv2.resize(mask, (img_H, img_W), interpolation=cv2.INTER_AREA),
                        axis=2,
                    )
                semantic_input[obj_idx] = mask
                if uid == 1:
                    # TODO: remove after bug-fix in KinematicHumanoid class
                    object_name = idx_to_name_mapping[obj_idx]
                    locations[obj_idx] = self.gt_graph.get_node_from_name(
                        object_name
                    ).properties["translation"]
            # create an output dict to be used by graph updater
            object_detections[uid] = {
                "object_masks": semantic_input,
                "out_img": out_img,
                "object_category_mapping": id_to_object_mapping,
                "object_handle_mapping": id_to_handle_mapping,
                "depth": obs["depth"],
                # "rgb": obs["rgb"], # uncomment to use only when rendering is ON
                "camera_intrinsics": as_intrinsics_matrix(obs["camera_intrinsics"]),
                "camera_pose": obs["camera_pose"],
                "object_locations": locations,
            }
        self._iteration += 1
        return object_detections
