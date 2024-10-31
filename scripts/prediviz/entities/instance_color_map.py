#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source


from typing import Dict, Optional


class InstanceColorMap:
    _instance_colors: Dict[str, str] = {}

    @classmethod
    def set_color(cls, instance_id: str, color: str) -> None:
        """Sets the color for a given instance ID."""
        cls._instance_colors[instance_id] = color

    @classmethod
    def get_color(cls, instance_id: str) -> Optional[str]:
        """Gets the color for a given instance ID."""
        return cls._instance_colors.get(instance_id, None)

    @classmethod
    def remove_color(cls, instance_id: str) -> None:
        """Removes the color for a given instance ID."""
        if instance_id in cls._instance_colors:
            del cls._instance_colors[instance_id]

    @classmethod
    def get_all_colors(cls) -> Dict[str, str]:
        """Gets all instance colors."""
        return cls._instance_colors.copy()

    @classmethod
    def has_color(cls, instance_id: str) -> bool:
        """Checks if a color exists for a given instance ID."""
        return instance_id in cls._instance_colors

    @classmethod
    def reset_map(cls) -> None:
        cls._instance_colors = {}
