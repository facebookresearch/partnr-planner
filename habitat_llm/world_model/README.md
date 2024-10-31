# World-graphs in PARTNR

This module implements all the data-types and logic required to maintain an intermediary world-graph representation which is used by planner for decision-making. This is a three-tiered tree going from House -> Room -> Furniture, with objects added as leaf nodes which can move around in the tree as an episode evolves.

## Entity

These are basic building blocks of a world-graph representation, consisting of following node-types:

1. Human
2. SpotRobot
3. Room
4. Floor
5. Furniture
6. objects

## World-graphs

We provide two types of graphs for our baseline:

1. Privileged world-graph: Objects are found based on observations but the location, segmentation, etc. used to find objects is ground-truth. GT information is also used to reassociate object detections over time. (`world_graph.py`)
2. Non-privileged world-graph:  Objects are found using ground-truth segmentation sensor but the location, reassociation over time is done using just RGB and depth sensors without using any privileged sim-level information. (`dynamic_world_graph.py`)

`graph.py` consists of basic graph operations agnostic of entities above. `world_graph.py` implements all PARTNR specific logic for privileged world-graphs without partial observability, i.e. graph is updated based on full observability each step. `dynamic_world_graph.py` implements all non-privileged PARTNR logic for maintaining world-graph using partial-observations and non-privileged information source.
