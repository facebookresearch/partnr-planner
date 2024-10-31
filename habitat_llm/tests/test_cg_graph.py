#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json

from habitat_llm.world_model import DynamicWorldGraph, Object

world_model_path = "data/hssd-partnr-ci/conceptgraph/test_cg_data.json"


def test_reading_cg():
    with open(world_model_path, "r") as f:
        sg_dict_list = json.load(f)

    mygraph = DynamicWorldGraph()
    mygraph.create_cg_edges(sg_dict_list, include_objects=True)

    for idx in range(len(sg_dict_list)):
        sg_dict = sg_dict_list[idx]
        node1 = sg_dict["object1"]
        node2 = sg_dict["object2"]
        node1_name = mygraph._cg_object_to_object_uid(node1)
        node2_name = mygraph._cg_object_to_object_uid(node2)
        object_relation = sg_dict["object_relation"]
        if "floor" in node1_name or "floor" in node2_name:
            continue
        if "wall" in node1_name or "wall" in node2_name:
            continue
        for curr_node, curr_node_name in zip([node1, node2], [node1_name, node2_name]):
            if curr_node["object_tag"] != "invalid" and curr_node["category_tag"] in [
                "object",
                "furniture",
            ]:
                if not mygraph.has_node(curr_node_name):
                    print(curr_node_name)
                assert mygraph.has_node(curr_node_name)
        if (
            node1["object_tag"] != "invalid"
            and node2["object_tag"] != "invalid"
            and node1["category_tag"] in ["object", "furniture"]
            and node2["category_tag"] in ["object", "furniture"]
            and object_relation not in ["none of these", "FAIL"]
        ):
            assert mygraph.has_edge(node1_name, node2_name)

    # add a test that if `include_objects` is False then no objects are in graph
    mygraph = DynamicWorldGraph()
    mygraph.create_cg_edges(sg_dict_list, include_objects=False)
    for node in mygraph.graph:
        assert not isinstance(node, Object)
        assert node.properties["type"] != "object"


test_reading_cg()
