#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
from copy import deepcopy
from os import path as osp

import pytest

from dataset_generation.benchmark_generation.generate_episodes import (
    default_gen_config,
    generate_episode,
    get_generator_state_semantic_debug_info,
    initialize_generator,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.sims.metadata_interface import default_metadata_dict


@pytest.mark.skipif(
    not osp.exists("data/hssd-partnr-ci/") or not osp.exists("data/objects_ovmm/"),
    reason="Requires HSSD mini dataset and ovmm_objects for testing.",
)
def test_llm_rearrange_episode_generator():
    # first define the necessary configs

    gen_config = default_gen_config
    gen_config["ep_dataset_output"] = "test_episode_dataset.json.gz"
    gen_config[
        "scene_dataset"
    ] = "data/hssd-partnr-ci/hssd-hab-partnr.scene_dataset_config.json"

    metadata_dict = default_metadata_dict
    metadata_dict["metadata_folder"] = "data/hssd-partnr-ci/metadata/"

    # initialize the generator object
    ep_gen = initialize_generator(gen_config, metadata_dict)

    # each dict is a specific test case, all expected to succeed

    test_clutter_init_template = {
        "scene_id": "102817140",
        "episode_id": "test|102817140|0",
        "initial_state": [
            {
                "name": "common sense",
                "number": "3",
                "common_sense_object_classes": True,  # this specifies region->object metadata is used for sampling
                "location": "on",
                "furniture_names": [],
            },
        ],
    }
    test_furniture_states_init_template = {
        "scene_id": "102817140",
        "episode_id": "test|102817140|0",
        "initial_furniture_state": [
            {
                "name": "all tables are clean",
                "furniture_classes": ["table"],
                "object_states": {
                    "is_clean": True,  # default is false, so this is a change
                },
            },
        ],
    }
    init_state_dicts = [
        # test 1: all regions, all receptacles, no floor
        {
            "scene_id": "102817140",
            "episode_id": "test|102817140|1",
            "initial_state": [
                {
                    "name": "default all",
                    "number": "2",
                    "object_classes": ["bowl"],
                    "location": "on",
                    "furniture_classes": [],
                },
            ],
        },
        # test 2: all regions, all receptacles, no floor, common sense objects
        test_clutter_init_template,
    ]
    ##################################
    # more clutter (common sense) tests:

    # 1: restrict to bowls
    test_init_1 = deepcopy(test_clutter_init_template)
    test_init_1["initial_state"][0]["object_classes"] = ["bowl"]
    init_state_dicts.append(test_init_1)

    # 2: restrict to kitchen
    test_init_2 = deepcopy(test_clutter_init_template)
    test_init_2["initial_state"][0]["allowed_regions"] = ["kitchen"]
    init_state_dicts.append(test_init_2)

    # 3: restrict to tables
    test_init_3 = deepcopy(test_clutter_init_template)
    del test_init_3["initial_state"][0]["furniture_names"]
    test_init_3["initial_state"][0]["furniture_classes"] = ["table"]
    init_state_dicts.append(test_init_3)

    # 4: restrict to floor
    test_init_4 = deepcopy(test_clutter_init_template)
    test_init_4["initial_state"][0]["furniture_names"] = ["floor"]
    init_state_dicts.append(test_init_4)

    # 5: generate a range
    test_init_5 = deepcopy(test_clutter_init_template)
    test_init_5["initial_state"][0]["number"] = [1, 5]
    init_state_dicts.append(test_init_5)

    # 6: generate a range with specified target
    test_init_6 = deepcopy(test_clutter_init_template)
    test_init_6["initial_state"][0]["number"] = [1, 1, 5]
    init_state_dicts.append(test_init_6)

    # 7: test integer number
    test_init_7 = deepcopy(test_clutter_init_template)
    test_init_7["initial_state"][0]["number"] = 2
    init_state_dicts.append(test_init_7)

    # 8: test adding object states
    test_init_8 = deepcopy(test_clutter_init_template)
    test_init_8["initial_state"][0]["object_states"] = {
        "is_clean": True,
        "test_int_state": 999,
    }
    init_state_dicts.append(test_init_8)

    # test furniture object state setting
    init_state_dicts.append(test_furniture_states_init_template)

    # 1: exact number of tables
    furniture_test_init_1 = deepcopy(test_furniture_states_init_template)
    furniture_test_init_1["initial_furniture_state"][0]["number"] = 2
    init_state_dicts.append(furniture_test_init_1)

    # 2: table_1
    furniture_test_init_2 = deepcopy(test_furniture_states_init_template)
    furniture_test_init_2["initial_furniture_state"][0]["furniture_names"] = ["table_1"]
    del furniture_test_init_2["initial_furniture_state"][0]["furniture_classes"]
    init_state_dicts.append(furniture_test_init_2)

    # 3: all furniture
    furniture_test_init_3 = deepcopy(test_furniture_states_init_template)
    furniture_test_init_3["initial_furniture_state"][0]["furniture_classes"] = []
    init_state_dicts.append(furniture_test_init_3)

    # 4: all kitchen cabinets
    furniture_test_init_4 = deepcopy(test_furniture_states_init_template)
    furniture_test_init_4["initial_furniture_state"][0]["furniture_classes"] = [
        "counter"
    ]
    furniture_test_init_4["initial_furniture_state"][0]["allowed_regions"] = ["kitchen"]
    init_state_dicts.append(furniture_test_init_4)

    obj_per_init = []

    for init_state_dict in init_state_dicts:
        try_count = 0
        details_log = {}
        while try_count < 10:
            ep, details_log = generate_episode(ep_gen, init_state_dict)
            print(f"details_log = {details_log}")
            if ep is None:
                # some unstable episodes are expected in any case
                assert (
                    details_log["failure_mode"] == "unstable state"
                ), "Other failures are not expected."
            else:
                # check for object states
                if (
                    "initial_state" in init_state_dict
                    and "object_states" in init_state_dict["initial_state"][0]
                ):
                    assert len(ep.object_states) > 0
                    for state_name, _state_value in init_state_dict["initial_state"][0][
                        "object_states"
                    ].items():
                        assert state_name in ep.object_states
                if "initial_furniture_state" in init_state_dict:
                    print(f"init_state_dict = {init_state_dict}")
                    print(f"object states = {ep.object_states}")
                    num_clean = 0
                    for clean_val in ep.object_states["is_clean"].values():
                        if clean_val:
                            num_clean += 1
                    assert (
                        num_clean > 0
                    ), "All furniture test configs set clean to True against the default for at least one object."

                # check that serialization is working
                dataset = CollaborationDatasetV0(episodes=[ep])
                json_str = dataset.to_json()
                decoded_dataset = CollaborationDatasetV0()
                decoded_dataset.from_json(json_str)
                # check for object states after serialization
                if (
                    "initial_state" in init_state_dict
                    and "object_states" in init_state_dict["initial_state"][0]
                ):
                    assert len(decoded_dataset.episodes[0].object_states) > 0
                    print(decoded_dataset.episodes[0].object_states)
                break
            try_count += 1
        print(get_generator_state_semantic_debug_info(ep_gen))
        # NOTE: uncomment this for state images from each test
        # debug_obs = get_generator_state_debug_images(ep_gen)
        # for obs in debug_obs:
        #    obs.show()
        assert ep is not None

        obj_per_init.append(len(ep.rigid_objs))

        # episodes should be marked as dynamic or not:
        assert "is_dynamic" in ep.info
        assert ep.info["is_dynamic"] == ep_gen.cfg.enable_check_obj_stability

    ep_gen.sim.close(destroy=True)
    gc.collect()
