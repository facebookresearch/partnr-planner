#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import gc
import random
from typing import Tuple

import habitat.sims.habitat_simulator.sim_utilities as sutils
import magnum as mn
import pytest

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.tests.test_planner import get_config, setup_env
from habitat_llm.utils import setup_config
from habitat_llm.world_model import (
    Furniture,
    House,
    Human,
    Object,
    Receptacle,
    Room,
    SpotRobot,
)
from habitat_llm.world_model.world_graph import WorldGraph

DATASET_OVERRIDES = [
    "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",  # We test with a specific dataset
    "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
    "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
    "habitat.environment.iterator_options.shuffle=False",
    "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/humanoids/female_0/female_0.urdf",  # We change the config to human 0 since only human 0 in the CI testing dataset
    "habitat.simulator.agents.agent_1.motion_data_path=data/humanoids/female_0/female_0_motion_data_smplx.pkl",  # We change the config to human 0 since only human 0 in the CI testing dataset
]


def util_add_nodes(graph) -> Tuple[WorldGraph, list]:
    # add a room to the graph
    room = Room("room", {"type": "test_node"})

    # add an object to the graph
    obj = Object("object", {"type": "test_node"})

    # add a furniture to the graph
    furniture = Furniture("furniture", {"type": "test_node"})

    # add a receptacle to the graph
    receptacle = Receptacle("receptacle", {"type": "test_node"})

    for node in [room, obj, furniture, receptacle]:
        graph.add_node(node)

    return graph, [room, obj, furniture, receptacle]


def test_get_subgraph():
    graph, (room, obj, fur, recep) = util_add_nodes(WorldGraph())
    house = House("house", {"type": "house"})
    graph.add_node(house)
    graph.add_edge(house, room, "has", "in")
    graph.add_edge(room, fur, "has", "in")
    graph.add_edge(fur, obj, "under", "on")

    # get subgraph for object and ensure it has the path up to house
    path = graph.get_subgraph([obj])

    assert house in path.graph
    assert room in path.graph
    assert fur in path.graph
    assert obj in path.graph
    assert recep not in path.graph
    assert path.graph[house][room] == "has"
    assert path.graph[room][fur] == "has"
    assert path.graph[fur][obj] == "under"
    assert path.graph[obj][fur] == "on"
    assert path.graph[fur][room] == "in"
    assert path.graph[room][house] == "in"

    # get subgraph for disconnected receptacle and it should only have house as empty
    # leaf node. Basically get_subgraph gets path from input to House. If input is not
    # connected you get an empty graph with just the house in it
    path_recep = graph.get_subgraph([recep])
    assert house in path_recep.graph
    assert recep not in path_recep.graph


def test_empty_world_graph():
    # create a test-graph and assert it is as expected
    graph = WorldGraph()
    assert len(graph.graph) == 0


def test_adding_nodes_to_world_graph():
    graph = WorldGraph()
    graph, nodes = util_add_nodes(graph)

    # test if each node is named and typed and present in the graph
    for node in nodes:
        assert node in graph.graph


def test_adding_edges_to_world_graph():
    graph = WorldGraph()
    graph, nodes = util_add_nodes(graph)

    for node in nodes[1:]:
        graph.add_edge(node, nodes[0], "edge1", opposite_label="edge2")

    for node in nodes[1:]:
        for neighbors, edge_label in graph.graph[node].items():
            assert neighbors == nodes[0]
            assert edge_label == "edge1"

    for neighbors, edge_label in graph.graph[nodes[0]].items():
        assert neighbors in nodes[1:]
        assert edge_label == "edge2"


def test_get_spot_robot_error_feedback():
    graph = WorldGraph()

    with pytest.raises(ValueError) as e:
        graph.get_spot_robot()
    assert "does not contain a node of type SpotRobot" in str(e.value)

    spot_test_node = SpotRobot("spot", {"type": "test_node"})
    graph.add_node(spot_test_node)
    assert graph.get_spot_robot() == spot_test_node


def test_get_human_error_feedback():
    graph = WorldGraph()

    with pytest.raises(ValueError) as e:
        graph.get_human()
    assert "does not contain a node of type Human" in str(e.value)

    human_test_node = Human("human", {"type": "test_node"})
    graph.add_node(human_test_node)
    assert graph.get_human() == human_test_node


def test_find_furniture_for_receptacle_error_feedback():
    graph = WorldGraph()
    receptacle = Receptacle("receptacle", {"type": "test_node"})

    graph.add_node(receptacle)
    with pytest.raises(ValueError) as e:
        graph.find_furniture_for_receptacle(receptacle)
    assert "No furniture" in str(e.value)


def test_get_neighbors_of_classtype():
    graph = WorldGraph()

    # make sure error is raised if node is not present
    with pytest.raises(ValueError) as e:
        graph.get_neighbors_of_type("test", "test_node")
    assert "test not present in the graph" in str(e.value)

    graph, [room, obj, fur, rec] = util_add_nodes(graph)

    # add edges between the nodes
    graph.add_edge(room, obj, "edge1", opposite_label="edge2")
    graph.add_edge(room, fur, "edge1", opposite_label="edge2")
    graph.add_edge(room, rec, "edge1", opposite_label="edge2")

    # test getting of typed neighbors
    assert graph.get_neighbors_of_type(room, Furniture) == [fur]
    assert graph.get_neighbors_of_type(room, Object) == [obj]
    assert graph.get_neighbors_of_type(room, Receptacle) == [rec]
    assert graph.get_neighbors_of_type(fur, Room) == [room]

    # test getting of neighbors of a type that is not present
    assert graph.get_neighbors_of_type(room, SpotRobot) == []


def test_count_nodes_of_type():
    graph = WorldGraph()

    room1 = Room("room1", {"type": "test_node"})
    room2 = Room("room2", {"type": "test_node"})
    object_node = Object("object", {"type": "test_node"})

    test_nodes = [room1, room2, object_node]
    for node in test_nodes:
        graph.add_node(node)

    assert graph.count_nodes_of_type(Room) == 2
    assert graph.count_nodes_of_type(Object) == 1
    assert graph.count_nodes_of_type(Furniture) == 0


def test_get_node_with_property():
    graph = WorldGraph()
    # add a room to the graph
    room = Room("room", {"type": "test_node", "category": "floorplan"})

    # add an object to the graph
    obj = Object("object", {"type": "test_node", "category": "utensils"})

    # add a furniture to the graph
    furniture = Furniture("furniture", {"type": "test_node", "category": "seating"})

    # add a receptacle to the graph
    receptacle = Receptacle("receptacle", {"type": "test_node", "category": "drawers"})

    for node in [room, obj, furniture, receptacle]:
        graph.add_node(node)

    assert graph.get_node_with_property("category", "floorplan") == room
    assert graph.get_node_with_property("category", "utensils") == obj
    assert graph.get_node_with_property("category", "seating") == furniture
    assert graph.get_node_with_property("category", "drawers") == receptacle
    assert graph.get_node_with_property("category", "non-existent") == None


def test_find_object_furniture_pairs():
    # test room-fur-recep-obj
    graph1 = WorldGraph()
    graph1, [room, obj, furniture, receptacle] = util_add_nodes(graph1)

    # add edges between the nodes
    graph1.add_edge(room, furniture, "edge1", opposite_label="edge2")
    graph1.add_edge(furniture, receptacle, "edge1", opposite_label="edge2")
    graph1.add_edge(receptacle, obj, "edge1", opposite_label="edge2")

    assert graph1.find_object_furniture_pairs() == {obj: furniture}

    # test room-fur-obj
    graph2 = WorldGraph()
    graph2, [room, obj, furniture, receptacle] = util_add_nodes(graph2)

    # add edges between nodes
    graph2.add_edge(room, furniture, "edge1", opposite_label="edge2")
    graph2.add_edge(furniture, obj, "edge1", opposite_label="edge2")

    assert graph2.find_object_furniture_pairs() == {obj: furniture}

    # test room-obj
    graph3 = WorldGraph()
    graph3, [room, obj, furniture, receptacle] = util_add_nodes(graph3)

    # add edges between nodes
    graph3.add_edge(room, obj, "edge1", opposite_label="edge2")

    assert graph3.find_object_furniture_pairs() == {}


def test_find_furniture_for_object():
    # test room-fur-recep-obj
    graph1 = WorldGraph()
    graph1, [room, obj, furniture, receptacle] = util_add_nodes(graph1)

    # add edges between the nodes
    graph1.add_edge(room, furniture, "edge1", opposite_label="edge2")
    graph1.add_edge(furniture, receptacle, "edge1", opposite_label="edge2")
    graph1.add_edge(receptacle, obj, "edge1", opposite_label="edge2")

    assert graph1.find_furniture_for_object(obj) == furniture

    # test room-fur-obj
    graph2 = WorldGraph()
    graph2, [room, obj, furniture, receptacle] = util_add_nodes(graph2)

    # add edges between nodes
    graph2.add_edge(room, furniture, "edge1", opposite_label="edge2")
    graph2.add_edge(furniture, obj, "edge1", opposite_label="edge2")

    assert graph2.find_furniture_for_object(obj) == furniture

    # test room-obj
    graph3 = WorldGraph()
    graph3, [room, obj, furniture, receptacle] = util_add_nodes(graph3)

    # add edges between nodes
    graph3.add_edge(room, obj, "edge1", opposite_label="edge2")

    assert graph3.find_furniture_for_object(obj) == None


def test_find_furniture_for_receptacle():
    # test room-fur-recep-obj
    graph1 = WorldGraph()
    graph1, [room, obj, furniture, receptacle] = util_add_nodes(graph1)

    # add edges between the nodes
    graph1.add_edge(room, furniture, "edge1", opposite_label="edge2")
    graph1.add_edge(furniture, receptacle, "edge1", opposite_label="edge2")
    graph1.add_edge(receptacle, obj, "edge1", opposite_label="edge2")

    assert graph1.find_furniture_for_receptacle(receptacle) == furniture

    # test room-fur-obj
    graph2 = WorldGraph()
    graph2.add_node(room)
    graph2.add_node(obj)
    graph2.add_node(furniture)

    # add edges between nodes
    graph2.add_edge(room, furniture, "edge1", opposite_label="edge2")
    graph2.add_edge(furniture, obj, "edge1", opposite_label="edge2")

    with pytest.raises(KeyError):
        graph2.find_furniture_for_receptacle(receptacle)

    # test room-rec
    graph3 = WorldGraph()
    graph3, [room, obj, furniture, receptacle] = util_add_nodes(graph3)

    # add edges between nodes
    graph3.add_edge(room, receptacle, "edge1", opposite_label="edge2")

    with pytest.raises(ValueError):
        graph3.find_furniture_for_receptacle(receptacle)


def test_bad_object_transform_in_unknown_room():
    object_handle = "CREATIVE_BLOCKS_35_MM_:0000"
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=[
            "evaluation=decentralized_evaluation_runner_multi_agent",
            "planner@evaluation.agents.agent_0.planner=llm_planner",
            "agent@evaluation.agents.agent_0.config=oracle_rearrange_object_states_agent",
            "llm@evaluation.agents.agent_0.planner.plan_config.llm=mock",
            "planner@evaluation.agents.agent_1.planner=llm_planner",
            "llm@evaluation.agents.agent_1.planner.plan_config.llm=mock",
        ]
        + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, 0)
    env_interface = setup_env(config)

    # assert object is attached to expected receptacle
    sim_object = sutils.get_obj_from_handle(env_interface.sim, object_handle)
    obj_in_world_graph = env_interface.full_world_graph.get_node_from_sim_handle(
        object_handle
    )
    graph_path = env_interface.full_world_graph.find_path(obj_in_world_graph)
    assert graph_path is not None
    rec_for_object = list(graph_path[obj_in_world_graph].keys())[0]
    assert rec_for_object.name == "rec_table_25_0"

    # get the object by handle and change transformation
    sim_object.translation = [100.0, 100.0, 100.0]

    # re-initialize the world-graph
    env_interface.initialize_perception_and_world_graph()

    # assert object is in unknown room
    obj_in_world_graph = env_interface.full_world_graph.get_node_from_sim_handle(
        object_handle
    )
    graph_path = env_interface.full_world_graph.find_path(obj_in_world_graph)
    assert graph_path is not None
    rec_for_object = list(graph_path[obj_in_world_graph].keys())[0]
    assert rec_for_object.name == "unknown_room"

    # Destroy envs
    env_interface.env.close()
    del env_interface
    gc.collect()


def test_object_placement_in_region():
    # load episode as usual
    object_handle = "CREATIVE_BLOCKS_35_MM_:0000"
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=[
            "evaluation=decentralized_evaluation_runner_multi_agent",
            "planner@evaluation.agents.agent_0.planner=llm_planner",
            "agent@evaluation.agents.agent_0.config=oracle_rearrange_object_states_agent",
            "llm@evaluation.agents.agent_0.planner.plan_config.llm=mock",
            "planner@evaluation.agents.agent_1.planner=llm_planner",
            "llm@evaluation.agents.agent_1.planner.plan_config.llm=mock",
        ]
        + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, 0)
    env_interface = setup_env(config)

    # assert object is attached to expected receptacle
    sim_object = sutils.get_obj_from_handle(env_interface.sim, object_handle)
    assert sim_object is not None, "Object does not exist"
    obj_in_world_graph: Object = (
        env_interface.full_world_graph.get_node_from_sim_handle(object_handle)
    )
    graph_path = env_interface.full_world_graph.find_path(obj_in_world_graph)
    assert graph_path is not None
    rec_for_object = list(graph_path[obj_in_world_graph].keys())[0]
    assert rec_for_object.name == "rec_table_25_0"

    # move the object to be on the floor of a randomly chosen region
    chosen_room = None
    while True:
        chosen_room = random.choice(env_interface.full_world_graph.get_all_rooms())
        room_point = chosen_room.properties.get("translation", None)
        if room_point is not None:
            sim_object.translation = (
                mn.Vector3(room_point)
                + env_interface.sim.get_gravity().normalized()
                * -sutils.get_obj_size_along(
                    env_interface.sim,
                    sim_object.object_id,
                    env_interface.sim.get_gravity().normalized(),
                )[0]
            )
            break

    # re-initialize the world-graph
    env_interface.initialize_perception_and_world_graph()

    # confirm receptacle returned is the floor for the region-ID of the chosen region
    obj_neighbors = env_interface.full_world_graph.get_neighbors(obj_in_world_graph)
    assert len(obj_neighbors) == 1, "An edge should be created for this object."
    assert (
        "floor_" + chosen_room.name == list(obj_neighbors.keys())[0].name
    ), "floor_<region_id> should be the matching node"

    # one more time with the object off the floor but in the region
    # technically the object is not on the floor but WG shd show it attached to floor
    # of the chosen region since every object needs a Furniture (Floor is inherited from Furniture)
    sim_object.translation += mn.Vector3(0, 1.0, 0)
    env_interface.initialize_perception_and_world_graph()
    obj_neighbors = env_interface.full_world_graph.get_neighbors(obj_in_world_graph)
    assert len(obj_neighbors) == 1, "An edge should be created for this object."
    assert (
        "floor_" + chosen_room.name == list(obj_neighbors.keys())[0].name
    ), "floor_<region_id> should be the matching node"

    # one more time with the object off the floor but within tolerance
    # WG should show it being attached to the floor
    sim_object.translation = (
        mn.Vector3(room_point)
        + env_interface.sim.get_gravity().normalized()
        * -sutils.get_obj_size_along(
            env_interface.sim,
            sim_object.object_id,
            env_interface.sim.get_gravity().normalized(),
        )[0]
    )
    sim_object.translation += mn.Vector3(0, 0.149, 0)
    env_interface.initialize_perception_and_world_graph()
    obj_neighbors = env_interface.full_world_graph.get_neighbors(obj_in_world_graph)
    assert len(obj_neighbors) == 1, "An edge should be created for this object."
    assert (
        "floor_" + chosen_room.name == list(obj_neighbors.keys())[0].name
    ), "floor_<region_id> should be the matching node"

    # Destroy envs
    env_interface.env.close()
    del env_interface
    gc.collect()
