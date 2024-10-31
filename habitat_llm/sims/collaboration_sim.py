#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from __future__ import annotations

from typing import TYPE_CHECKING

import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.object_state_machine import (
    ObjectIsPoweredOn,
    ObjectStateMachine,
    set_state_of_obj,
)
from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim

from habitat_llm.sims.metadata_interface import (
    MetadataError,
    MetadataInterface,
    get_metadata_dict_from_config,
)
from habitat_llm.world_model.object_states import ObjectIsClean, ObjectIsFilled

if TYPE_CHECKING:
    import habitat
    from omegaconf import DictConfig

    from habitat_llm.agent.env.dataset import CollaborationEpisode


def initialize_object_state_machine(
    sim: "habitat.Simulator", metadata_interface: MetadataInterface
) -> ObjectStateMachine:
    """
    Initialize an ObjectStateMachine from the Simulator contents and a pre-initialized MetadataInterface.

    :param sim: The Simulator instance.
    :param metadata_interface: a pre-initialized MetadataInterface with mappings for object semantic classes and object state affordances.

    :return: An initialized ObjectStateMachine.
    """

    objs = sutils.get_all_objects(sim)
    for obj in objs:
        try:
            obj_type = metadata_interface.get_object_instance_category(obj)
            if obj_type is not None:
                set_state_of_obj(obj, "semantic_class", obj_type)
            else:
                # NOTE: explicitly set unannotated types to 'unknown'
                set_state_of_obj(obj, "semantic_class", "unknown")
        # if this object has no metadata, skip setting the semantic class
        except MetadataError:
            pass

    power_state = ObjectIsPoweredOn()
    power_state.accepted_semantic_classes = metadata_interface.affordance_info[
        "turned on or off"
    ]
    requires_faucet = metadata_interface.affordance_info[
        "cleaned under a faucet if dirty"
    ]
    clean_state = ObjectIsClean()
    clean_state.accepted_semantic_classes = (
        metadata_interface.affordance_info["cleaned with a brush if dirty"]
        + requires_faucet
    )
    clean_state.requires_faucet_semantic_classes = requires_faucet
    filled_state = ObjectIsFilled()
    filled_state.accepted_semantic_classes = metadata_interface.affordance_info[
        "filled with water"
    ]
    active_states = [power_state, clean_state, filled_state]
    object_state_machine = ObjectStateMachine(active_states=active_states)
    object_state_machine.initialize_object_state_map(sim)
    return object_state_machine


@registry.register_simulator(name="CollaborationSim-v0")
class CollaborationSim(RearrangeSim):
    def __init__(self, config: "DictConfig") -> None:
        # Construct metadata paths from defaults and config for use if no pre-initialized MetadataInterface is provided.
        self.metadata_dict = get_metadata_dict_from_config(config)
        self.metadata_interface: MetadataInterface = None

        self._object_state_machine = None
        super().__init__(config)

    @property
    def object_state_machine(self) -> ObjectStateMachine:
        if self._object_state_machine is None:
            raise ValueError("Object state machine not initialized")
        return self._object_state_machine

    def initialize_object_state_machine(self, ep_info: "CollaborationEpisode") -> None:
        """
        Initializes the internal ObjectStateMachine such that MangedObjects can have mutable states.
        Sets the initial object state values defined in the CollaborationEpisode

        :param ep_info: The CollaborationEpisode optionally containing initial values for object states.
        """

        if self.metadata_interface is None:
            raise MetadataError(
                "'reconfigure' must be called before 'initialize_object_state_machine' to prepare the MetadataInterface."
            )

        self._object_state_machine = initialize_object_state_machine(
            self, self.metadata_interface
        )

        # Load initial states from the episode info
        for state_name, handle_value_map in ep_info.object_states.items():
            for handle, value in handle_value_map.items():
                set_state_of_obj(get_obj_from_handle(self, handle), state_name, value)

    def reconfigure(
        self, config: "DictConfig", ep_info: "CollaborationEpisode"
    ) -> None:
        super().reconfigure(config, ep_info)
        # NOTE: config == simulator.
        self.metadata_dict = get_metadata_dict_from_config(config)
        if (
            self.metadata_interface is None
            or self.metadata_dict != self.metadata_interface.metadata_source_dict
        ):
            self.metadata_interface = MetadataInterface(self.metadata_dict)
        self.metadata_interface.refresh_scene_caches(self, filter_receptacles=True)
        self.initialize_object_state_machine(ep_info)
