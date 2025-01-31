#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""This module contains the abstract class Perception"""


from abc import ABC, abstractmethod


class Perception(ABC):
    """
    This class represents abstract perception stack of the agents.
    """

    # Parameterized Constructor
    def __init__(self, detectors=None):
        # Initialize the set of detectors
        self.detectors = detectors

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """
        This method initializes the perception stack.
        """
