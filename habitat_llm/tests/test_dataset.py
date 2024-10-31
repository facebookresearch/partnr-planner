#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import time

import pytest
from habitat import logger
from hydra import compose, initialize

from habitat_llm.agent.env import register_sensors  # noqa
from habitat_llm.agent.env.dataset import CollaborationDatasetV0, CollaborationEpisode


def check_json_serialization(dataset: CollaborationDatasetV0):
    start_time = time.time()
    json_str = dataset.to_json()

    logger.info("JSON conversion finished. {} sec".format((time.time() - start_time)))
    decoded_dataset = CollaborationDatasetV0()
    decoded_dataset.from_json(json_str)
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, CollaborationEpisode)

    # The strings won't match exactly as dictionaries don't have an order for the keys
    # Thus we need to parse the json strings and compare the serialized forms
    assert json.loads(decoded_dataset.to_json()) == json.loads(
        json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def check_binary_serialization(dataset: CollaborationDatasetV0):
    start_time = time.time()
    bin_dict = dataset.to_binary()

    logger.info("Binary conversion finished. {} sec".format((time.time() - start_time)))
    decoded_dataset = CollaborationDatasetV0()
    decoded_dataset.from_binary(bin_dict)
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, CollaborationEpisode)
    dataset_decoded_bin = decoded_dataset.to_binary()
    # check that the expected keys are present in both encoded Dicts
    expected_keys = ["all_transforms", "idx_to_name", "all_eps"]
    for key in expected_keys:
        assert key in bin_dict
        assert key in dataset_decoded_bin

    # use JSON serialization to test integrity of dataset binary serialization -> deserialization
    assert json.loads(decoded_dataset.to_json()) == json.loads(
        dataset.to_json()
    ), "JSON dataset encoding of binary decoding isn't consistent."


def test_collaboration_dataset():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="habitat_conf/dataset/collaboration_hssd.yaml",
            overrides=["habitat.dataset.scenes_dir=data/hssd-partnr-ci"],
        )
        dataset_cfg = cfg.habitat.dataset

    if not CollaborationDatasetV0.check_config_paths_exist(dataset_cfg):
        pytest.skip("Test skipped as dataset files are missing.")

    dataset = CollaborationDatasetV0(config=dataset_cfg)
    assert len(dataset.episodes) > 0, "The dataset shouldn't be empty."
    assert (
        dataset.episodes[0].instruction != ""
    ), "The task instruction shouldn't be empty."
    check_json_serialization(dataset)
    check_binary_serialization(dataset)
