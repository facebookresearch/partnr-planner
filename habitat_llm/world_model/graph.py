#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import copy
import random

from habitat_llm.world_model import (
    Entity,
    Furniture,
    Human,
    Object,
    Receptacle,
    Room,
    SpotRobot,
)


class Graph:
    """
    This class represents a Directed Acyclic Graph.
    """

    # Parameterized Constructor
    def __init__(self, graph=None):
        # Create a graph to store different entities in the world
        # and their relations to one another
        if graph is None:
            graph = {}
        self.graph = graph

    def __deepcopy__(self, memo):
        """
        Method to deep copy this instance
        """
        # region object is non deepcopy-able so we need a little
        # trick here to copy it over ourselves
        self._region_cache = {}
        for node in self.graph:
            if "region" in node.properties:
                self._region_cache[node.name] = node.properties["region"]
                del node.properties["region"]
        new_graph = Graph(copy.deepcopy(self.graph, memo))
        for new_node in new_graph.graph:
            if new_node.name in self._region_cache:
                new_node.properties["region"] = self._region_cache[new_node.name]
        return new_graph

    def __copy__(self, memo):
        """
        Method to copy this instance
        """
        return Graph(copy.copy(self.graph, memo))

    def deepcopy_graph(self, input_graph):
        """
        Method to deepcopy just the graph object
        """
        self._region_cache = {}
        for node in input_graph:
            if "region" in node.properties:
                self._region_cache[node.name] = node.properties["region"]
                del node.properties["region"]
        new_graph = copy.deepcopy(input_graph)
        for new_node in new_graph:
            if new_node.name in self._region_cache:
                new_node.properties["region"] = self._region_cache[new_node.name]
        return new_graph

    def size(self):
        """
        This method returns the number of nodes in the graph
        """
        return len(self.graph)

    def is_empty(self):
        """
        This method tells if the graph is empty or not
        """
        return self.size() == 0

    def get_node_from_name(self, node_name: str) -> Entity:
        """
        This method returns the node with matching name
        """
        for node in self.graph:
            if node.name == node_name:
                return node

        raise ValueError(f"Node with name {node_name} not present in the graph")

    def get_node_from_sim_handle(self, node_sim_handle):
        """
        This method returns the node with matching name
        """
        for node in self.graph:
            if node.sim_handle == node_sim_handle:
                return node

        raise ValueError(
            f"Node with sim_handle {node_sim_handle} not present in the graph."
        )

    def has_node(self, input_node):
        """
        This method checks if the graph contains given node
        """

        # Reason if the input is of type string
        if isinstance(input_node, str):
            return any(node.name == input_node for node in self.graph)

        # Reason if the input is not string
        return input_node in self.graph

    def has_edge(self, node1, node2, edge_label: str = None):
        """
        This method checks if the graph contains edge between two nodes
        """

        if isinstance(node1, str):
            node1 = self.get_node_from_name(node1)
        if isinstance(node2, str):
            node2 = self.get_node_from_name(node2)
        return any(
            neighbors == node2 and edge_label is None
            for neighbors, edges in self.graph[node1].items()
        )

    def has_node_with_sim_handle(self, sim_handle):
        """
        This method checks if the graph contains node with given sim handle
        """

        # Try to match sim handle
        return any(node.sim_handle == sim_handle for node in self.graph)

    def add_node(self, node):
        """
        This method adds a node to the world graph
        """
        if "type" not in node.properties:
            raise ValueError(
                f"Node {node.name} doesn't have a type, type is needed during initialization."
            )
        if node not in self.graph:
            self.graph[node] = {}

    def add_edge(self, node1, node2, label, opposite_label=None, verbose=False):
        """
        This method adds edge between two nodes.
        opposite label represents semantically opposite relation.
        E.g. if label is "inside" its opposite lable should be "outside"
        """
        # Fetch node1 if input type is string
        if isinstance(node1, str):
            node1 = self.get_node_from_name(node1)

        # Fetch node2 if input type is string
        if isinstance(node2, str):
            node2 = self.get_node_from_name(node2)

        # FIXME: this method silently overwrites a previous edge if you only pass label
        # or only opposite_label
        if node1 in self.graph and node2 in self.graph:
            # Add directional edge from node1 to node2
            self.graph[node1][node2] = label

            # Add opposite directional edge from node2 to node1
            self.graph[node2][node1] = opposite_label

        else:
            if verbose:
                print("Trying to add edge, but one or both nodes don't exist in graph")

    def remove_node(self, node):
        """
        This method removes node and corresponding edges from the graph.
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_name(node)

        # Delete the node and edges to it
        del self.graph[node]
        for edges in self.graph.values():
            if node in edges:
                del edges[node]

    def remove_edge(self, node1, node2):
        """
        This method removes edge between two nodes
        """

        # Fetch node1 if input type is string
        if isinstance(node1, str):
            node1 = self.get_node_from_name(node1)

        # Fetch node2 if input type is string
        if isinstance(node2, str):
            node2 = self.get_node_from_name(node2)

        if node1 in self.graph and node2 in self.graph:
            if node2 in self.graph[node1]:
                del self.graph[node1][node2]
                del self.graph[node2][node1]
            else:
                print(
                    f"Edge doesn't exist between the two nodes:{node1.name}, {node2.name}"
                )
        else:
            print("Trying to remove edge but, one or both nodes don't exist in graph")

    def remove_all_edges(self, node):
        """
        This node removes all edges associated with a perticular node
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_name(node)

        # Throw if node is invalid
        if node not in self.graph:
            raise ValueError(f"{node} not present in the graph")

        # Clear the outgoing edges
        self.graph[node] = {}

        # Clear incoming edges
        for edges in self.graph.values():
            if node in edges:
                del edges[node]

    def pop_node(self, node):
        """
        This method pops node and corresponding edges from the graph.
        """
        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_name(node)

        # Pop the node
        popped_node = self.graph.pop(node)

        # Clean up the connections
        for edges in self.graph.values():
            if node in edges:
                del edges[node]

        return popped_node

    def get_all_node_names(self):
        """
        Method to retrieve list of all node names from the graph
        """
        # Find all nodes with matching type
        node_names = [node.name for node in self.graph]

        if len(node_names) > 0:
            return node_names
        else:
            return None

    def get_all_nodes_of_type(self, class_type):
        """
        Method to retrieve all nodes of a specific class
        """
        # Find all nodes with matching type
        matching_nodes = [node for node in self.graph if isinstance(node, class_type)]

        if len(matching_nodes) > 0:
            return matching_nodes
        else:
            return None

    def get_random_node_of_type(self, class_type):
        """
        Method to get a random node of a given type
        """
        # Find all nodes with matching type
        matching_nodes = self.get_all_nodes_of_type(class_type)

        if len(matching_nodes) > 0:
            return random.choice(matching_nodes)
        else:
            return None

    def remove_all_nodes_of_type(self, class_type):
        """
        Method to remove all nodes of a given type
        """
        # Find all nodes with matching type
        matching_nodes = self.get_all_nodes_of_type(class_type)

        # Remove them from the dictionary
        if matching_nodes is None:
            return
        for node in matching_nodes:
            self.remove_node(node)

        return

    def merge(self, other_graph, add_only: bool = False):
        """
        This method merges the other graph into this graph.
        It will add all missing nodes and edges, and it will
        replace the already present nodes with their new
        counterparts from the other graph along with their edges.
        """

        # Add all new nodes and replace the existing ones
        for new_node in other_graph.graph:
            try:
                old_node = self.get_node_from_name(new_node.name)
            except ValueError:
                # Add new_node
                self.add_node(new_node)
            else:
                # NOTE: be very careful to not overwrite entire properties dict
                if "translation" in new_node.properties:
                    old_node.properties["translation"] = new_node.properties[
                        "translation"
                    ]

        # Now add all new edges
        for curr_node in other_graph.graph:
            if isinstance(curr_node, (Object, SpotRobot, Human)):
                delete_list = []
                if not add_only:
                    for old_neighbor, _edge in self.graph[curr_node].items():
                        if old_neighbor not in other_graph.graph[curr_node]:
                            delete_list.append((curr_node, old_neighbor))
                    for curr_node, to_remove_neighbor in delete_list:
                        self.remove_edge(curr_node, to_remove_neighbor)
            for new_neighbor, edge in other_graph.graph[curr_node].items():
                self.add_edge(
                    curr_node.name,
                    new_neighbor.name,
                    edge,
                    other_graph.graph[new_neighbor][curr_node],
                )

        return

    def get_neighbors(self, node):
        """
        This method returns all neighbors of the current node
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_name(node)

        # Throw if node is invalid
        if node not in self.graph:
            raise ValueError(f"{node} not present in the graph")

        return self.graph[node]

    def get_neighbors_of_type(self, node, class_type):
        """
        This method returns list of all neighbors
        of a node that have given class_type.
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_name(node)

        # Throw if node is invalid
        # TODO: we probably do not need the above node fetching + this check
        if node not in self.graph:
            raise ValueError(f"{node} not present in the graph")

        return [
            neighbor
            for neighbor in self.graph[node]
            if isinstance(neighbor, class_type)
        ]

    def count_nodes_of_type(self, class_type):
        """
        This method returns count of all nodes of given type
        """

        count = 0
        for node in self.graph:
            if isinstance(node, class_type):
                count += 1

        return count

    def display_flattened(self):
        """
        Method to print the flattened world graph
        """
        for node, edges in self.graph.items():
            print(f"Edges-{node.name}, type-{node.properties['type']}:")
            for edge, label in edges.items():
                print(f"  --{label}--> {edge.name}")

    def display_hierarchy(self, file_handle=None):
        """
        Method to print the graph with hierarchical
        """

        # Call the recursive printing method
        self.dfs_traverse(
            self.get_node_from_name("house"), set(), file_handle=file_handle
        )

        return

    def to_string(self, compact=False):
        """
        Method to convert graph into a string
        """

        # Call the recursive printing method
        out = self.dfs_traverse(self.get_node_from_name("house"), set(), "", compact)

        return out

    def dfs_traverse(
        self, node, visited_nodes_set, out=None, compact=False, file_handle=None
    ):
        """
        Recursive method to print the graph with DFS.
        """
        # Early return if the node has already been printed
        if node in visited_nodes_set:
            return None

        # Add node to the visited list
        visited_nodes_set.add(node)

        # Iterate through all neighbors of the current node.
        for neighbor in sorted(self.graph[node]):
            # Skip if the neighbor has already been visited
            if neighbor in visited_nodes_set:
                continue

            # Print the node based on the class type
            if isinstance(neighbor, SpotRobot):
                text = f"\tSpotRobot: {neighbor.name}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            elif isinstance(neighbor, Human):
                text = f"\tHuman: {neighbor.name}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            elif isinstance(neighbor, Room):
                text = f"Room: {neighbor.name}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            elif isinstance(neighbor, Furniture):
                text = f"\tFurniture: {neighbor.name}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            elif isinstance(neighbor, Receptacle) and not compact:
                text = f"\t\tReceptacle: {neighbor.name}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            elif isinstance(neighbor, Object):
                text = f"\t\t\tObject: {neighbor.name}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            else:
                ValueError("Unsupported node type")

            # Call this method recursively on the neighbor
            out = self.dfs_traverse(
                neighbor, visited_nodes_set, out, compact, file_handle=file_handle
            )

        return out
