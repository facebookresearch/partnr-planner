#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from os import path as osp

import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle
import habitat.sims.habitat_simulator.sim_utilities as sutils
import pytest
from habitat_sim import Simulator
from habitat_sim.metadata import MetadataMediator
from habitat_sim.utils.settings import default_sim_settings, make_cfg

# use this for the additional object paths
from dataset_generation.benchmark_generation.generate_episodes import default_gen_config
from habitat_llm.sims.metadata_interface import MetadataInterface, default_metadata_dict


@pytest.mark.skipif(
    not osp.exists("data/hssd-partnr-ci/"),
    reason="Requires HSSD mini dataset for testing.",
)
def test_metadata_interface():
    metadata_dict = default_metadata_dict
    metadata_dict["metadata_folder"] = "data/hssd-partnr-ci/metadata/"
    mi = MetadataInterface(metadata_dict)

    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/hssd-partnr-ci/hssd-hab-partnr.scene_dataset_config.json"
    sim_settings["scene"] = "102817140"
    hab_cfg = make_cfg(sim_settings)
    with Simulator(hab_cfg) as sim:
        # load additional object paths
        additional_object_paths = default_gen_config["additional_object_paths"]
        for obj_path in additional_object_paths:
            sim.get_object_template_manager().load_configs(obj_path)

        # find the scene filter for the scene
        scene_filter_file = hab_receptacle.get_scene_rec_filter_filepath(
            sim.metadata_mediator, sim.curr_scene_name
        )
        # we know there is one in hssd-partnr-ci, so assert it is found
        assert scene_filter_file is not None
        # get the filter strings from the filter
        filter_strings = hab_receptacle.get_excluded_recs_from_filter_file(
            scene_filter_file
        )
        # assert that loading the filter strings from file succeeds and produces something
        assert len(filter_strings) > 0

        # load all Receptacle metadata from the scene (legacy behavior)
        mi.refresh_scene_caches(sim, filter_receptacles=False)
        unfiltered_recs = mi.receptacles
        # find any filter instances which match the receptacle set
        filter_matches = [
            rec for rec in unfiltered_recs if rec.unique_name in filter_strings
        ]
        # we should find at least one Receptacle which would have been filtered out
        assert len(filter_matches) > 0

        # now load the Receptacles again, but this time apply the filter
        mi.refresh_scene_caches(sim, filter_receptacles=True)
        filtered_recs = mi.receptacles
        # there should be something remaining after filtering
        assert len(filtered_recs) > 0
        filter_matches = [
            rec for rec in filtered_recs if rec.unique_name in filter_strings
        ]
        # there should be nothing remaining which matches the filters
        assert len(filter_matches) == 0

        # there should be fewer objects in the filtered set than the legacy, unfiltered set
        assert len(unfiltered_recs) > len(filtered_recs)

        # test limiting object semantic query to dynamic (ovmm_objects) set
        mm = sim.metadata_mediator
        assert len(mi.dynamic_lexicon) < len(
            mi.lexicon
        ), "Fewer clutter object categories should be available than clutter + furniture."

        for cat in mi.lexicon:
            # get all matches in the full template set
            all_objs = mi.get_template_handles_of_class(mm, cat, dynamic_source=False)
            # get only matches from the externally added (ovmm_objects) set
            dyn_objs = mi.get_template_handles_of_class(mm, cat, dynamic_source=True)
            assert len(all_objs) >= len(dyn_objs)
            for obj_handle in dyn_objs:
                # all objects in the dynamic list should have "dynamic" source
                assert (
                    mi.hash_to_source[sutils.object_shortname_from_handle(obj_handle)]
                    == "dynamic"
                )
            # the remainder of objects should have no source or "hssd" source
            non_dynamic_objs = [handle for handle in all_objs if handle not in dyn_objs]
            for obj_handle in non_dynamic_objs:
                short_name = sutils.object_shortname_from_handle(obj_handle)
                if short_name in mi.hash_to_source:
                    assert mi.hash_to_source[short_name] == "hssd"


@pytest.mark.skipif(
    not osp.exists("data/hssd-partnr-ci/"),
    reason="Requires HSSD mini dataset metadata for testing.",
)
def test_ovmm_source_objects():
    # this test ensures that all objects in the "dynamic_lexicon" metadata are present in ovmm_objects repository
    # to do so, we load the dynamic object source repos without hssd and check against the metadata

    metadata_dict = default_metadata_dict
    metadata_dict["metadata_folder"] = "data/hssd-partnr-ci/metadata/"

    # initialize the MetadataInterface to create the `dynamic_lexicon` and fill `hash_to_source`
    mi = MetadataInterface(metadata_dict)

    # initialize the MetadataMediator to load the ovmm_objects config files
    mm = MetadataMediator()
    otm = mm.object_template_manager
    additional_object_paths = default_gen_config["additional_object_paths"]
    for obj_path in additional_object_paths:
        otm.load_configs(obj_path)

    dynamic_object_hashes = [
        obj_hash
        for obj_hash in mi.hash_to_source
        if mi.hash_to_source[obj_hash] == "dynamic"
    ]
    unfound_objects = []
    # validate that each `dynamic` entry in hash_to_source can be found without hssd
    for obj_hash in dynamic_object_hashes:
        dynamic_template_matches = otm.get_file_template_handles(obj_hash)
        potential_match_hashes = [
            filepath.split("/")[-1].split(".object_config.json")[0]
            for filepath in dynamic_template_matches
        ]
        if obj_hash not in potential_match_hashes:
            unfound_objects.append(obj_hash)

    # assert on unfound objects
    assert (
        len(unfound_objects) == 0
    ), f"Found {len(unfound_objects)}/{len(dynamic_object_hashes)} hashes without templates in dynamic asset sets: {unfound_objects}"
