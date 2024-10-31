#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import os

import habitat
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from habitat_llm.examples.verify_episodes import compute_stats, run_verifier
from habitat_llm.utils import setup_config

DATASET_OVERRIDES = [
    "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",  # We test with a specific dataset
    "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
    "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
    "habitat.environment.iterator_options.shuffle=False",
    "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/humanoids/female_0/female_0.urdf",  # We change the config to human 0 since only human 0 in the CI testing dataset
    "habitat.simulator.agents.agent_1.motion_data_path=data/humanoids/female_0/female_0_motion_data_smplx.pkl",  # We change the config to human 0 since only human 0 in the CI testing dataset
]


def get_config(config_file, overrides):
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name=config_file,
            overrides=overrides,
        )
    HydraConfig().cfg = config
    # emulate a regular hydra initialization
    with open_dict(config):
        config.hydra = {}
        config.hydra.runtime = {}
        config.hydra.runtime.output_dir = "outputs/test"

    return config


def test_verify_episodes():
    cfg = get_config(
        "examples/planner_multi_agent_demo_config.yaml", overrides=DATASET_OVERRIDES
    )
    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.agents_order = sorted(cfg.habitat.simulator.agents.keys())
    cfg = setup_config(cfg)
    run_verifier(cfg)
    compute_stats(cfg)
    file_name = "outputs/test/episode_checks/ci.json.gz_summary.json"

    assert os.path.isfile(file_name)
    with open(file_name, "r") as f:
        summary = json.load(f)
    for i in range(4):
        assert summary[f"{i}.json"]["success_init"]
