#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from habitat.sims.habitat_simulator.object_state_machine import BooleanObjectState


class ObjectIsFilled(BooleanObjectState):
    """
    ObjectIsFilled state specifies whether an object is filled or empty.
    Following the pattern from habitat-lab/habitat/sims/habitat_simulator/object_state_machine.py
    """

    def __init__(self):
        super().__init__()
        self.name = "is_filled"
        self.display_name = "Is Full"
        self.display_name_true = "Full"
        self.display_name_false = "Empty"
        self.accepted_semantic_classes = []

    def default_value(self) -> bool:
        return False


class ObjectIsClean(BooleanObjectState):
    """
    ObjectIsClean state specifies whether an object is clean or dirty.
    """

    def __init__(self):
        super().__init__()
        self.name = "is_clean"
        self.display_name = "Is Clean"
        self.display_name_true = "Clean"
        self.display_name_false = "Dirty"
        self.accepted_semantic_classes = []

    def default_value(self) -> bool:
        return False
