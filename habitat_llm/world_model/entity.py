#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import copy
from abc import ABC


class Entity(ABC):  # noqa: B024
    """
    This class is a pure abstract class to represent either a
    Room, Receptacle, Surface or an Object in the robot's model of the world.
    It contains state variables to represent a unique id, short_name,
    and various metric and semantic properties.
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Unique string for representing this node
        self.name = name

        # Container to store properties
        # Some common properties are "category", "position"
        if "type" not in properties:
            properties["type"] = "entity_node"
        self.properties = properties

        # Optional member to represent sim handle
        self.sim_handle = sim_handle

    # Deep copy
    def __deepcopy__(self, memo):
        return Entity(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )

    # Hashing Operator
    def __hash__(self):
        return hash(self.name)

    # Equality operator
    def __eq__(self, other):
        #  Make sure that the other is of type Entity
        if not isinstance(other, Entity):
            return False

        return self.name == other.name

    # Method to print the object
    def __str__(self):
        out = f"{self.__class__.__name__}[name={self.name}, type={self.properties['type']}]"
        return out

    # Method to alphabetically order object
    def __lt__(self, other):
        return self.name < other.name

    # Method to get property
    def get_property(self, prop):
        if prop in self.properties:
            return self.properties[prop]
        else:
            raise ValueError(
                f"Entity with name '{self.name}' has no property called '{prop}'"
            )

    # Method to set "states" property
    def set_state(self, state_dict):
        if "states" not in self.properties:
            self.properties["states"] = {}
        self.properties["states"].update(state_dict)


class UncategorizedEntity(Entity):
    """
    This class represents an object-type entity that is not categorized as either
    an object or receptacle per rearrangement nomenclature
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle)

    # Deep copy
    def __deepcopy__(self, memo):
        return UncategorizedEntity(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )


class House(Entity):
    """
    This class represents the house in the world
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle)

    # Deep copy
    def __deepcopy__(self, memo):
        return House(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )


class Room(Entity):
    """
    This class represents a room in the world
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle)

    # Deep copy
    def __deepcopy__(self, memo):
        return Room(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )


class Receptacle(Entity):
    """
    This class represents a receptacle in the world.
    Receptacles are sort of surfaces on the furniture.
    This is only required because simulator has a notion of it
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle)

    # Deep copy
    def __deepcopy__(self, memo):
        return Receptacle(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )


class Object(Entity):
    """
    This class represents a small movable object in the world.
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle)

    # Deep copy
    def __deepcopy__(self, memo):
        return Object(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )


class SpotRobot(Entity):
    """
    This class represents spot robot
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle)

    # Deep copy
    def __deepcopy__(self, memo):
        return SpotRobot(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )


class Human(Entity):
    """
    This class represents human
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle)

    # Deep copy
    def __deepcopy__(self, memo):
        return Human(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )
