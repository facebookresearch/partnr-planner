# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import networkx as nx
from habitat.sims.habitat_simulator.sim_utilities import (
    above,
    get_obj_from_handle,
    get_obj_from_id,
    obj_next_to,
    object_in_region,
    on_floor,
    within,
)
from habitat_sim.physics import ManagedArticulatedObject, ManagedRigidObject
from habitat_sim.scene import SemanticRegion

from habitat_llm.sims.collaboration_sim import CollaborationSim


@dataclass
class PropositionResult:
    is_satisfied: bool
    info: Dict[str, Any] = field(default_factory=dict)


class SimBasedPredicates:
    """
    Predicates are defined in Habitat-Lab for checking the relationships and states
    of entities in a scene. Here we define wrappers for Habitat-LLM.
    """

    @staticmethod
    def sim_instance_from_handle(
        sim: CollaborationSim, handle: str
    ) -> Union[ManagedRigidObject, ManagedArticulatedObject]:
        """map handle to object/receptacle as managed by the AOM or ROM."""
        sim_instance = get_obj_from_handle(sim, handle)
        if sim_instance is None:
            raise ValueError(
                f"`{handle}` not found in {{articulated | rigid}} object managers."
            )
        return sim_instance

    @staticmethod
    def sim_region_from_id(sim: CollaborationSim, region_id: str) -> SemanticRegion:
        """Map the handle of a region to the SemanticRegion in Habitat Sim"""
        for region in sim.semantic_scene.regions:
            if region.id == region_id:
                return region
        raise ValueError(f"Region `{region_id}` not found in the scene.")

    @staticmethod
    def get_state_snapshot_if_none(
        sim: CollaborationSim, snapshot_dict: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        if snapshot_dict is not None:
            return snapshot_dict
        return sim.object_state_machine.get_snapshot_dict(sim)

    @classmethod
    def set_predicate(
        cls,
        predicate_fn: Callable[..., PropositionResult],
        sim: CollaborationSim,
        entity_handles_a: List[str],
        entity_handles_b: Optional[List[Union[str, None]]] = None,
        number: int = 1,
        is_same_entity_b: bool = False,
        default_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PropositionResult:
        """
        True if any entity from `entity_handles_a` satisfies `predicate_fn` with any
        entity from `entity_handles_b`.
        Args:
            number: at least this number of entities from `entity_handles_a` must
                satisfy `predicate_fn`.
            is_same_entity_b: at least `number` entities from `entity_handles_a` must
                satisfy `predicate_fn` with the same entity from `entity_handles_b`.
        """

        def proposition(
            entity_handle_a: str, entity_handle_b: Optional[str] = None
        ) -> PropositionResult:
            if entity_handle_b is None:
                return predicate_fn(sim, [entity_handle_a], number=number, **kwargs)
            return predicate_fn(
                sim, [entity_handle_a], [entity_handle_b], number=number, **kwargs
            )

        if entity_handles_b is None:
            entity_handles_b = [None]
        if default_info is None:
            default_info = {}

        if number < 1:
            raise ValueError("At least one relation must be queried.")

        if len(entity_handles_b) == 0:
            raise ValueError("Empty `entity_handles_b` encountered.")

        if is_same_entity_b and entity_handles_b[0] is None:
            raise ValueError("`is_same_entity_b` incompatible with NoneType entity b.")

        res = PropositionResult(False, default_info)

        # case: matching entity b is required
        if is_same_entity_b:
            for entity_handle_b in entity_handles_b:
                count = 0
                for entity_handle_a in entity_handles_a:
                    res = proposition(entity_handle_a, entity_handle_b)
                    if not res.is_satisfied:
                        continue
                    count += 1
                    if count >= number:
                        return res
            return res

        # case: any or empty entity b
        count = 0
        for entity_handle_a, entity_handle_b in itertools.product(
            entity_handles_a, entity_handles_b
        ):
            res = proposition(entity_handle_a, entity_handle_b)
            if not res.is_satisfied:
                continue
            count += 1
            if count >= number:
                return res
        return res

    @classmethod
    def is_on_top(
        cls,
        sim: CollaborationSim,
        object_handles: List[str],
        receptacle_handles: List[str],
        number: int = 1,
        is_same_receptacle: bool = False,
        ao_link_map: Optional[Dict[int, int]] = None,
        *args,
        **kwargs,
    ) -> PropositionResult:
        """True if an object from `object_handles` is on top of a receptacle from
        `receptacle_handles`.
        """
        if len(object_handles) > 1 or len(receptacle_handles) > 1:
            return cls.set_predicate(
                cls.is_on_top,
                sim,
                object_handles,
                receptacle_handles,
                number=number,
                is_same_entity_b=is_same_receptacle,
                ao_link_map=ao_link_map,
                default_info={"object_handles": [], "receptacle_handles": []},
            )

        # case: single object, single receptacle
        obj = cls.sim_instance_from_handle(sim, object_handles[0])
        info = {"object_handles": "", "receptacle_handles": ""}
        for recep_id in above(sim, obj):
            above_recep = get_obj_from_id(sim, recep_id, ao_link_map)
            if above_recep is None:
                continue
            if above_recep.handle == receptacle_handles[0]:
                info["object_handles"] = object_handles[0]
                info["receptacle_handles"] = receptacle_handles[0]
                return PropositionResult(True, info)
        return PropositionResult(False, info)

    @classmethod
    def is_inside(
        cls,
        sim: CollaborationSim,
        object_handles: List[str],
        receptacle_handles: List[str],
        number: int = 1,
        is_same_receptacle: bool = False,
        ao_link_map: Optional[Dict[int, int]] = None,
        *args,
        **kwargs,
    ) -> PropositionResult:
        """True if an object from `object_handles` is inside of a receptacle from
        `receptacle_handles`.
        """
        if len(object_handles) > 1 or len(receptacle_handles) > 1:
            return cls.set_predicate(
                cls.is_inside,
                sim,
                object_handles,
                receptacle_handles,
                number=number,
                is_same_entity_b=is_same_receptacle,
                ao_link_map=ao_link_map,
                default_info={"object_handles": "", "receptacle_handles": ""},
            )

        # case: single object, single receptacle
        obj = cls.sim_instance_from_handle(sim, object_handles[0])
        info = {"object_handles": "", "receptacle_handles": ""}
        for recep_id in within(sim, obj):
            within_recep = get_obj_from_id(sim, recep_id, ao_link_map)
            if within_recep is None:
                continue
            if within_recep.handle == receptacle_handles[0]:
                info["object_handles"] = object_handles[0]
                info["receptacle_handles"] = receptacle_handles[0]
                return PropositionResult(True, info)
        return PropositionResult(False, info)

    @classmethod
    def is_in_room(
        cls,
        sim: CollaborationSim,
        object_handles: List[str],
        room_ids: List[str],
        number: int = 1,
        is_same_room: bool = False,
        ao_link_map: Optional[Dict[int, int]] = None,
        *args,
        **kwargs,
    ) -> PropositionResult:
        """True if an object from `object_handles` is in a room from `room_ids`."""
        if len(object_handles) > 1 or len(room_ids) > 1:
            return cls.set_predicate(
                cls.is_in_room,
                sim,
                object_handles,
                room_ids,
                number=number,
                is_same_entity_b=is_same_room,
                ao_link_map=ao_link_map,
                default_info={"object_handles": "", "receptacle_handles": ""},
            )

        # case: single object, single room
        obj = cls.sim_instance_from_handle(sim, object_handles[0])
        sim_region = cls.sim_region_from_id(sim, room_ids[0])
        is_in_region, _ = object_in_region(
            sim, obj, sim_region, ao_link_map=ao_link_map
        )
        info = {
            "object_handles": object_handles[0] if is_in_region else "",
            "room_ids": room_ids[0] if is_in_region else "",
        }
        return PropositionResult(is_in_region, info)

    @classmethod
    def is_on_floor(
        cls,
        sim: CollaborationSim,
        object_handles: List[str],
        number: int = 1,
        ao_link_map: Optional[Dict[int, int]] = None,
        *args,
        **kwargs,
    ) -> PropositionResult:
        """True if an object from `object_handles` is on any navmesh island."""
        if len(object_handles) > 1:
            return cls.set_predicate(
                cls.is_on_floor,
                sim,
                object_handles,
                number=number,
                ao_link_map=ao_link_map,
                default_info={"object_handles": ""},
            )

        # case: single object
        obj = cls.sim_instance_from_handle(sim, object_handles[0])
        is_on_floor = on_floor(
            sim,
            obj,
            island_index=sim._largest_indoor_island_idx,
            ao_link_map=ao_link_map,
        )
        info = {"object_handles": object_handles[0] if is_on_floor else ""}
        return PropositionResult(is_on_floor, info)

    @classmethod
    def is_next_to(
        cls,
        sim: CollaborationSim,
        entity_handles_a: List[str],
        entity_handles_b: List[str],
        number: int = 1,
        is_same_b: bool = False,
        l2_threshold: float = 0.5,
        ao_link_map: Optional[Dict[int, int]] = None,
        *args,
        **kwargs,
    ) -> PropositionResult:
        """
        True if an entity from `entity_handles_a` is next to
        any entity from `entity_handles_a`.
        """
        if len(entity_handles_a) > 1 or len(entity_handles_b) > 1:
            return cls.set_predicate(
                cls.is_next_to,
                sim,
                entity_handles_a,
                entity_handles_b,
                number=number,
                is_same_entity_b=is_same_b,
                l2_threshold=l2_threshold,
                ao_link_map=ao_link_map,
                default_info={"entity_handles_a": "", "entity_handles_b": ""},
            )

        # case: single entity - single entity
        entity_a = cls.sim_instance_from_handle(sim, entity_handles_a[0])
        entity_b = cls.sim_instance_from_handle(sim, entity_handles_b[0])
        is_next_to = obj_next_to(
            sim=sim,
            object_id_a=entity_a.object_id,
            object_id_b=entity_b.object_id,
            hor_l2_threshold=l2_threshold,
            ao_link_map=ao_link_map,
        )
        info = {
            "entity_handles_a": entity_handles_a[0] if is_next_to else "",
            "entity_handles_b": entity_handles_b[0] if is_next_to else "",
        }
        return PropositionResult(is_next_to, info)

    @classmethod
    def is_clustered(
        cls,
        *args: List[str],
        sim: CollaborationSim,
        number: List[int] = None,
        l2_threshold: float = 0.5,
        ao_link_map: Optional[Dict[int, int]] = None,
        **kwargs,
    ):
        """
        A generalization of `is_next_to` to more than two objects.
        In short, every object must be `next_to` at least one other object.

        True if each entity in *args is `next_to` at least one other arg entity.
        Supported characteristics:
          - entity ambiguity (each arg is a list of allowed entities)
          - subset counts (required `number` for each argument)

        Args:
            *args: each arg is a list of object handles that can be used to satisfy the
                    relation. Each arg must be satisfied. In the number=[1, ..., 1] case,
                    at least one handle from each argument must satisfy the relation.
            number: n_i elements of the ith arg must satisfy the cluster criteria.
                    Default: [1, ..., 1]
            l2_threshold: Distance threshold for object-object `next_to`.

        Example:
            Instruction: "set one toy, two books, and the hat next to each other."
            Proposition: is_clustered(
                ["toy_1", "toy_2"], ["book_1", "book_2", "book_3"], ["hat_1"], number=[1,2,1]
            )
            Interpretation: either toy, any two books, and the hat.
        """

        def handle_group_satisfies(grp, n, cc) -> bool:
            return len(set(grp) & set(cc)) >= n

        def format_satisfied_info(handle_groups, cc):
            result = []
            for grp in handle_groups:
                result.append([h for h in grp if h in cc])
            return {"*args": tuple(result)}

        all_handles = {x for xs in args for x in xs}

        # create an undirected graph of obj-obj next-to relations.
        g = nx.Graph()
        for h1 in all_handles:
            g.add_node(h1)
            for h2 in all_handles:
                if h1 == h2:
                    continue
                obj_a = cls.sim_instance_from_handle(sim, h1)
                obj_b = cls.sim_instance_from_handle(sim, h2)
                if obj_next_to(
                    sim=sim,
                    object_id_a=obj_a.object_id,
                    object_id_b=obj_b.object_id,
                    hor_l2_threshold=l2_threshold,
                    ao_link_map=ao_link_map,
                ):
                    g.add_edge(h1, h2)

        # is_clustered: a group of connected vertices satisfies the given argument handle groups and subset numbers.
        for cc in sorted(nx.connected_components(g), key=len, reverse=True):
            if all(
                handle_group_satisfies(grp, number[i], cc) for i, grp in enumerate(args)
            ):
                return PropositionResult(True, format_satisfied_info(args, cc))
        return PropositionResult(False)

    @classmethod
    def boolean_object_state_predicate(
        cls,
        object_state_name: str,
        sim: CollaborationSim,
        object_handles: List[str],
        number: int = 1,
        object_states_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        negate: bool = False,
        *args,
        **kwargs,
    ) -> PropositionResult:
        """
        True if an object from `object_handles` satisfies the object state predicate
        `object_state_name`. The `negate` option returns the negation of this as
        applied on a per-object level.
        """
        object_states_dict = cls.get_state_snapshot_if_none(sim, object_states_dict)

        if len(object_handles) > 1:
            return cls.set_predicate(
                getattr(cls, object_state_name),
                sim,
                object_handles,
                number=number,
                object_states_dict=object_states_dict,
                negate=negate,
                default_info={"object_handles": ""},
            )

        handle = object_handles[0]

        # ensure that the object exists
        cls.sim_instance_from_handle(sim, handle)

        if object_state_name not in object_states_dict:
            raise KeyError(
                f"State key `{object_state_name}` not in the object state snapshot dict."
            )
        if handle not in object_states_dict[object_state_name]:
            raise KeyError(
                f"Object handle `{handle}` not registered with object state `{object_state_name}` dict."
            )

        res = bool(object_states_dict[object_state_name][handle])
        if negate:
            res = not res

        return PropositionResult(res, {"object_handles": handle if res else ""})

    @classmethod
    def is_clean(cls, *args, **kwargs) -> PropositionResult:
        return cls.boolean_object_state_predicate("is_clean", *args, **kwargs)

    @classmethod
    def is_dirty(cls, *args, **kwargs) -> PropositionResult:
        return cls.boolean_object_state_predicate(  # type: ignore
            "is_clean", *args, **kwargs, negate=True
        )

    @classmethod
    def is_powered_on(cls, *args, **kwargs) -> PropositionResult:
        return cls.boolean_object_state_predicate("is_powered_on", *args, **kwargs)

    @classmethod
    def is_powered_off(cls, *args, **kwargs) -> PropositionResult:
        return cls.boolean_object_state_predicate(  # type: ignore
            "is_powered_on", *args, **kwargs, negate=True
        )

    @classmethod
    def is_filled(cls, *args, **kwargs) -> PropositionResult:
        return cls.boolean_object_state_predicate("is_filled", *args, **kwargs)

    @classmethod
    def is_empty(cls, *args, **kwargs) -> PropositionResult:
        return cls.boolean_object_state_predicate(  # type: ignore
            "is_filled", *args, **kwargs, negate=True
        )
