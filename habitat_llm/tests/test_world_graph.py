#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Tuple

import pytest

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
