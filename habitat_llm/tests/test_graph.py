#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import pytest

from habitat_llm.world_model.entity import Entity
from habitat_llm.world_model.graph import Graph


def test_get_node_from_name():
    graph = Graph()
    with pytest.raises(ValueError) as e:
        graph.get_node_from_name("test")
    assert "Node with name test not present" in str(e.value)

    test_node = Entity("test", {"type": "test_node"})
    graph.add_node(test_node)
    assert graph.get_node_from_name(test_node.name) == test_node


def test_get_node_from_sim_handle():
    graph = Graph()
    with pytest.raises(ValueError) as e:
        graph.get_node_from_sim_handle("1")
    assert "Node with sim_handle 1 not present" in str(e.value)

    test_node = Entity("test", {"type": "test_node"}, sim_handle="1")
    graph.add_node(test_node)
    assert graph.get_node_from_sim_handle(test_node.sim_handle) == test_node


def test_remove_all_edges():
    graph = Graph()
    with pytest.raises(ValueError) as e:
        graph.remove_all_edges("test")
    assert "test not present in the graph" in str(e.value)
    test_nodes = []
    for index in range(5):
        test_nodes.append(Entity(f"test{index}", {"type": "test_node"}))
        graph.add_node(test_nodes[-1])
        graph.add_edge(test_nodes[-1], test_nodes[0], "test_edge")

    assert len(graph.graph[test_nodes[0]]) == 5
    assert len(graph.graph[test_nodes[1]]) == 1
    assert len(graph.graph[test_nodes[2]]) == 1
    assert len(graph.graph[test_nodes[3]]) == 1
    assert len(graph.graph[test_nodes[4]]) == 1
    graph.remove_all_edges(test_nodes[0])
    assert len(graph.graph) == len(test_nodes)
    assert len(graph.graph[test_nodes[0]]) == 0
    assert len(graph.graph[test_nodes[1]]) == 0
    assert len(graph.graph[test_nodes[2]]) == 0
    assert len(graph.graph[test_nodes[3]]) == 0
    assert len(graph.graph[test_nodes[4]]) == 0


def test_get_neighbors():
    graph = Graph()

    # make sure error is raised if node is not present
    with pytest.raises(ValueError) as e:
        graph.get_neighbors("test")
    assert "test not present in the graph" in str(e.value)

    # test getting of neighbors
    test_nodes = []
    num_nodes = 6
    for index in range(num_nodes):
        test_nodes.append(Entity(f"test{index}", {"type": "test_node"}))
        graph.add_node(test_nodes[-1])
        if index == 0 or index == (num_nodes - 1):
            continue
        graph.add_edge(test_nodes[-1], test_nodes[0], "test_edge")

    assert graph.get_neighbors(test_nodes[0]) == {
        tnode: None for tnode in test_nodes[1:-1]
    }
    assert graph.get_neighbors(test_nodes[1]) == {test_nodes[0]: "test_edge"}
    assert graph.get_neighbors(test_nodes[2]) == {test_nodes[0]: "test_edge"}
    assert graph.get_neighbors(test_nodes[3]) == {test_nodes[0]: "test_edge"}
    assert graph.get_neighbors(test_nodes[4]) == {test_nodes[0]: "test_edge"}

    # test getting of neighbors for singleton node
    assert graph.get_neighbors(test_nodes[5]) == {}


def test_remove_edge():
    graph = Graph()
    with pytest.raises(ValueError) as e:
        graph.remove_edge("test", "test")
    assert "test not present in the graph" in str(e.value)
    test_nodes = []
    for index in range(5):
        test_nodes.append(Entity(f"test{index}", {"type": "test_node"}))
        graph.add_node(test_nodes[-1])
        graph.add_edge(test_nodes[-1], test_nodes[0], "test_edge")

    assert len(graph.graph[test_nodes[0]]) == 5
    assert len(graph.graph[test_nodes[1]]) == 1
    assert len(graph.graph[test_nodes[2]]) == 1
    assert len(graph.graph[test_nodes[3]]) == 1
    assert len(graph.graph[test_nodes[4]]) == 1
    graph.remove_edge(test_nodes[0], test_nodes[1])
    assert len(graph.graph[test_nodes[0]]) == 4
    assert len(graph.graph[test_nodes[1]]) == 0
    assert len(graph.graph[test_nodes[2]]) == 1
    assert len(graph.graph[test_nodes[3]]) == 1
    assert len(graph.graph[test_nodes[4]]) == 1
    graph.remove_edge(test_nodes[0], test_nodes[2])
    assert len(graph.graph[test_nodes[0]]) == 3
    assert len(graph.graph[test_nodes[1]]) == 0
    assert len(graph.graph[test_nodes[2]]) == 0
    assert len(graph.graph[test_nodes[3]]) == 1
    assert len(graph.graph[test_nodes[4]]) == 1
    graph.remove_edge(test_nodes[0], test_nodes[3])
    assert len(graph.graph[test_nodes[0]]) == 2
    assert len(graph.graph[test_nodes[1]]) == 0
    assert len(graph.graph[test_nodes[2]]) == 0
    assert len(graph.graph[test_nodes[3]]) == 0
    assert len(graph.graph[test_nodes[4]]) == 1
    graph.remove_edge(test_nodes[0], test_nodes[4])
    assert len(graph.graph[test_nodes[0]]) == 1
    assert len(graph.graph[test_nodes[1]]) == 0
    assert len(graph.graph[test_nodes[2]]) == 0
    assert len(graph.graph[test_nodes[3]]) == 0
    assert len(graph.graph[test_nodes[4]]) == 0
