#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import copy
import random
from dataclasses import dataclass
from typing import List, Union

import numpy as np

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    DifferentArgConstraint,
    SameArgConstraint,
    TemporalConstraint,
    TerminalSatisfactionConstraint,
)
from habitat_llm.planner.planner import Planner
from habitat_llm.utils.sim import is_open
from habitat_llm.world_model import Floor, Furniture

# This class distributes skills across two agents using the eval function.
# It works by looking at every node in a DAG and distributing at random the
# tasks per node between the different agents.
# The main function of this class is to distribute the work load
# amongst the available agents and co-ordinate them in order to
# accomplish the final goal in a collaborative manner.

# The planner, encapsulated in PropositionResolver, works as follows:
# 1. Constructs a set of RearrangeActions according to the propositions, which contain info
# about which object should go where and next to what.
# 2. Converts each RearrangeAction into a sequence of Nav Pick Nav Place actions,
# so that they can be run in the skills.
# 3. Assigns the actions to each agent at random, making sure the temporal constraints are respected.

# The most complex step is 1, which requires to reason about propositions and proposition groups.
# This is handled in the construct_rearrange_from_propositions function.

# For this, we first create a dictionary containing the propositions that should have the same argument
# and the ones that should have different argument. So that we can keep track of propositions that were
# assigned. Then we convert propositions into rearrange actions as follows:

# For every proposition we do:
# 1. Get a list of target objects and target receptacles for that proposition.
# 2.a. If the proposition was part of a same argument constraint, check if that argument was already sampled
# if so, reduce the list of objects/receptacles to what was already sampled.
# 2.b. If the proposition is part of diff argument constraint, remove the obj/receptacle candidates
# that were assigned before.
# 2.c. Remove objects that were already assigned as part of a terminal satisfaction constraint, since there
# should not be moved anymore.
# 3. Sample objects and receptacles from the candidate list.
# 4. If the proposition is next_to, the sampled receptacle will be another object that you should be next to.
# We will select as receptacle the location where that object was placed in the past (or where it was originally)
# And take note that it should be next to this object.
# 5. if the proposition is part of samearg or diffarg constraint, fill up the data structures so that we
# can run steps 2.a and 2.b. later.
# 6. Create the rearrange action based on the sample receptacle.
# 7. Sometimes we specify next to and also on_top inside, which will lead to repeated rearrange actions
# final step is to remove duplicates.

OBJECT_STATES = {
    "is_clean": "Clean",
    "is_filled": "Fill",
    "is_empty": "Pour",
    "is_powered_on": "PowerOn",
    "is_powered_off": "PowerOff",
}


@dataclass
class RearrangeAction:
    prop_ind: int
    to: str
    obj: str
    preposition: str
    next_to: Union[str, None]


@dataclass
class ChangeStateAction:
    prop_ind: int
    obj: str
    attribute: str
    next_to = None


class UnionFind:
    @classmethod
    def find(cls, parent, i):
        if parent[i] == i:
            return i
        else:
            # Path compression heuristic
            parent[i] = cls.find(parent, parent[i])
            return parent[i]

    @classmethod
    def union(cls, parent, rank, x, y):
        rootX = cls.find(parent, x)
        rootY = cls.find(parent, y)

        if rootX != rootY:
            # Union by rank heuristic
            parent[rootX] = rootY

    @classmethod
    def connected_components(cls, node_connections):
        flattened_nodes = sorted(
            {item for sublist in node_connections for item in sublist}
        )
        node_dict = {
            node_name: node_id for node_id, node_name in enumerate(flattened_nodes)
        }
        max_node = len(node_dict)

        # Initialize parent and rank arrays
        parent = list(range(max_node))
        rank = [0] * max_node

        # Process each connection
        for connection in node_connections:
            for i in range(1, len(connection)):
                cls.union(
                    parent, rank, node_dict[connection[0]], node_dict[connection[i]]
                )

        # Find all unique roots to determine the distinct components
        roots = set(cls.find(parent, i) for i in range(max_node))

        # Group nodes by their root parent to form the connected components
        components = {root: [] for root in roots}
        for node in range(max_node):
            root = cls.find(parent, node)
            components[root].append(node)

        final_components = []
        for connected_comp in components.values():
            final_components.append(
                [flattened_nodes[nodeid] for nodeid in connected_comp]
            )
        return final_components


def get_proposition_groups_from_episode(
    tc_constraints, num_propositions
) -> List[List[int]]:
    """Extract proposition groups from the episode's TemporalConstraint.
    If no TemporalConstraint exists, assume a flat proposition group.
    """

    default_groups = [list(range(num_propositions))]

    if len(tc_constraints) == 0:
        return default_groups
    if len(tc_constraints) > 1:
        raise AssertionError(
            f"Episode has {len(tc_constraints)} TemporalConstraints. Expected 1."
        )

    # tc constraints == 1
    prop_groups = tc_constraints[0].get_topological_generations()
    if len(prop_groups) == 0:
        return default_groups
    props_in_groups = []
    for p in prop_groups:
        props_in_groups += p
    missing_props = list(set(default_groups[0]) - set(props_in_groups))
    if len(missing_props) > 0:
        prop_groups[0] += missing_props

    return prop_groups


class PropositionResolver:
    def __init__(self, sim_handle_to_name, region_id_to_name, seed=None):
        """
        Initialize the proposition resolver class, which will assign skills to the two agents
        based on eval function.
        :param sim_handle_to_name: mapping from handles to natural language names
        :param region_id_to_name: mapping from region ids to a natural language name
        """
        self.sim_handle_to_name = sim_handle_to_name
        self.region_id_to_name = region_id_to_name
        self.random = random.Random(seed)
        self.np_random = np.random.default_rng(seed)
        self.actions_per_agent = {}
        # Objects we should not sample because they are in some proposition already
        self.block_objects = []

        # Track next_to relationships
        self.next_to_move = {}

    def assign_dag(
        self, prop_groups, prop_indices, constraints_agent_assignment, num_agents=2
    ):
        """
        Given a dag with propositions, assigns propositions to each agent so that the DAG is followed.
        E.g. if prop_groups is [Rearrange(O1,R1), Rearrange(O2,R2)], [Rearrange(O3,R3)]
        which means that there should be two rearrange in any order and a third rearrange later,
        it will do a random assignment such as:
        {0: [Rearrange(O1,R1), Wait1, Rearrange(O3, R3)], 1: [Rearrange(O2, R2), Wait1]}
        where Wait is a deadlock. Because we don't use Rearrange actions anymore, this also converts
        to Nav, Pick, Nav, Place
        :prop_groups: list of propositions with constraints set by DAG.
        :prop_indices: for every proposition group what proposition indices are in that group. This is used
        for temporal constraints
        :constraints_agent_assignment: for every proposition index, which agents can do them
        :return: dictionary from agent to skills
        """
        agent_actions = {ind: [] for ind in range(num_agents)}
        for prop_grp_ind, prop_group in enumerate(prop_groups):
            for ind, action in enumerate(prop_group):
                prop_ind = prop_indices[prop_grp_ind][ind]
                assigned_agent = self.random.choice(
                    constraints_agent_assignment[prop_ind]
                )

                if type(action) == list:
                    agent_actions[assigned_agent] += action
                else:
                    agent_actions[assigned_agent].append(action)

            if num_agents > 1:
                for ind in range(num_agents):
                    agent_actions[ind].append(f"Wait {prop_grp_ind}")

        return agent_actions

    def get_list_actions_from_rearrange(
        self,
        rearrange_action,
        environment_graph,
        closed_receptacles,
        objects_navigated,
        environment_interface,
    ):
        partial_obs = environment_interface.partial_obs

        # Get here metadata needed to decide actions (e.g. whether we should move next to faucet)
        metadata_interface = environment_interface.perception.metadata_interface
        requires_faucet_class = metadata_interface.affordance_info[
            "cleaned under a faucet if dirty"
        ]
        target_obj = rearrange_action.obj
        object_type = environment_graph.get_node_from_name(target_obj).properties[
            "type"
        ]
        list_actions = []
        # The change state action does not involve nav pick nav place.
        # TODO: if the action is clean and the object requires faucet, the
        # action should include nav to faucet
        target_obj = rearrange_action.obj
        # Is object inside closed container
        fur_object = environment_graph.find_furniture_for_object(
            environment_graph.get_node_from_name(target_obj)
        ).name

        list_actions = []
        if fur_object != target_obj:
            if fur_object in closed_receptacles:
                list_actions += [
                    ("Navigate", f"{fur_object}", ""),
                    ("Open", f"{fur_object}", ""),
                ]
            elif partial_obs:
                # If we are in partial obs, we need to navigate to the furniture
                list_actions += [("Navigate", f"{fur_object}", "")]

        if type(rearrange_action) is ChangeStateAction:
            state_action = OBJECT_STATES[rearrange_action.attribute]
            furniture_faucet = [
                obj
                for obj in environment_graph.get_all_furnitures()
                if "components" in obj.properties
                and "faucet" in obj.properties["components"]
            ]
            if len(furniture_faucet) > 0:
                furniture_faucet = furniture_faucet[0]
            else:
                furniture_faucet = None
            if furniture_faucet is not None and (
                state_action == "Fill"
                or (state_action == "Clean" and object_type in requires_faucet_class)
            ):
                # Find the furniture with faucet

                list_actions += [
                    ("Navigate", f"{target_obj}", ""),
                    ("Pick", f"{target_obj}", ""),
                    ("Navigate", f"{furniture_faucet.name}", ""),
                    (state_action, f"{target_obj}", ""),
                    ("Navigate", f"{furniture_faucet.name}", ""),
                    (
                        "Place",
                        f"{target_obj}, on, {furniture_faucet.name}, none, none",
                        "",
                    ),
                ]

            else:
                list_actions += [
                    ("Navigate", f"{target_obj}", ""),
                    (state_action, f"{target_obj}", ""),
                ]
            return list_actions

        next_to_prop = rearrange_action.next_to
        target_rec = rearrange_action.to
        target_obj = rearrange_action.obj
        preposition = rearrange_action.preposition

        list_actions += [
            ("Navigate", f"{target_obj}", ""),
            ("Pick", f"{target_obj}", ""),
            ("Navigate", f"{target_rec}", ""),
        ]
        if target_rec in closed_receptacles:  #  and preposition == "within":
            list_actions += [
                ("Open", f"{target_rec}", ""),
                ("Navigate", f"{target_rec}", ""),
            ]
        # Check if this object should be placed next to something
        if next_to_prop is None or next_to_prop == target_rec:
            list_actions += [
                ("Place", f"{target_obj}, {preposition}, {target_rec}, none, none", "")
            ]
        else:
            list_actions += [
                (
                    "Place",
                    f"{target_obj}, {preposition}, {target_rec}, next_to, {next_to_prop}",
                    "",
                )
            ]

        return list_actions

    def reduce_args_with_same_constraint(self, propositions, constraints):
        """
        Looks at all the same argument constraints and builds a set of propositions that
        are connected by at least one constraint. Returns a dictionary from set index to (prop_index, relationship)
        """
        same_arg_constraint = [
            c for c in constraints if isinstance(c, SameArgConstraint)
        ]
        component_dict = {}

        if len(same_arg_constraint) > 0:
            same_arg_components = []
            for constraint in same_arg_constraint:
                arg_names = constraint.arg_names
                prop_indices = constraint.proposition_indices
                c_list = [(prop_ind, arg_names[prop_ind]) for prop_ind in prop_indices]
                same_arg_components.append(c_list)

            # Here is the list of proposition indices and arg_names that should have the same arg
            connected_components = UnionFind.connected_components(same_arg_components)
            for component_ind, components in enumerate(connected_components):
                for component in components:
                    component_dict[component] = component_ind
            return component_dict, len(connected_components)
        else:
            return {}, 0

    def get_receptacles_objects_from_prop(
        self, proposition, prop_id, environment_graph
    ):
        """
        Given a proposition, returns all the possible receptacles and objects that could be candidates
        for that proposition.
        """
        function_args = proposition.args
        task_type = proposition.function_name
        if task_type == "is_next_to":
            target_objects = function_args[self.next_to_move[prop_id][0]]
        else:
            target_objects = function_args["object_handles"]
        if task_type == "is_in_room":
            # Get the receptacles in that room
            receptacles = function_args["room_ids"]
            room_nodes = [self.region_id_to_name[objname] for objname in receptacles]
            all_rooms = [
                room
                for room in environment_graph.get_all_rooms()
                if room.name in room_nodes
            ]
            receptacles = []

            # The object should go in the floor of this room
            for room in all_rooms:
                edges_contains = [
                    edge
                    for edge, value in environment_graph.graph[room].items()
                    if value == "contains"
                ]
                if prop_id in self.room_to_floor_prop:
                    receptacles += [
                        rec.name for rec in edges_contains if type(rec) is Floor
                    ]
                else:
                    receptacles += [
                        rec.name for rec in edges_contains if type(rec) is Furniture
                    ]
            same_rec = function_args["is_same_room"]
        elif task_type == "is_next_to":
            receptacles = function_args[self.next_to_move[prop_id][1]]
            # We only get objects that are in sim_handle_to_name, the rest of the receptacles are unreachable
            receptacles = [
                self.sim_handle_to_name[objname]
                for objname in receptacles
                if objname in self.sim_handle_to_name
            ]

            same_rec = function_args["is_same_b"]
        elif task_type in OBJECT_STATES:
            receptacles = []
            same_rec = False
        elif task_type != "is_on_floor":
            receptacles = function_args["receptacle_handles"]
            # We only get objects that are in sim_handle_to_name, the rest of the receptacles are unreachable
            receptacles = [
                self.sim_handle_to_name[objname]
                for objname in receptacles
                if objname in self.sim_handle_to_name
            ]

            same_rec = function_args["is_same_receptacle"]
        else:
            if prop_id in self.floor_to_room_prop:
                raise Exception(
                    "This predicate should not have been considered here, it is part of the room predicates"
                )
            else:
                all_rooms = environment_graph.get_all_rooms()

                receptacles = []

                # The object should go in the floor of this room
                for room in all_rooms:
                    edges_contains = [
                        edge
                        for edge, value in environment_graph.graph[room].items()
                        if value == "contains"
                    ]
                    receptacles += [
                        rec.name for rec in edges_contains if type(rec) is Floor
                    ]
                same_rec = False
        # Convert objects and receptacles to nodes in graph
        target_objects = [
            self.sim_handle_to_name[objname] for objname in target_objects
        ]
        return target_objects, receptacles, same_rec

    def build_different_arg_component_dict(self, constraints):
        """
        Return a dictionary mapping a tuple (prop_ind, arg_name) to the other prop_inds and arg_names
        that are constrained by it
        """
        res_dict = {}
        diff_arg_constraint = [
            c for c in constraints if isinstance(c, DifferentArgConstraint)
        ]
        for c in diff_arg_constraint:
            tuples_constraints = list(zip(c.proposition_indices, c.arg_names))
            for tup in tuples_constraints:
                if tup not in res_dict:
                    res_dict[tup] = []
                res_dict[tup] += tuples_constraints
        for tup in res_dict:
            res_dict[tup] = list(set(res_dict[tup]))
        return res_dict

    def get_earlier_props(self, prop_ind, proposition_groups):
        """
        Check the propositions that come before a given prop_ind
        """
        earlier_props = []
        for prop_group in proposition_groups:
            if prop_ind in prop_group:
                return earlier_props
            earlier_props += prop_group
        return earlier_props

    def is_earlier(self, prop_1, prop_2, proposition_groups):
        """
        Check if proposition with index prop_1 happens earlier then prop_2
        """
        found_1 = False
        found_2 = False
        for prop_group in proposition_groups:
            if prop_1 in prop_group:
                found_1 = True
            if prop_2 in prop_group:
                found_2 = True
            if found_1 and not found_2:
                return True
            if found_2 and not found_1:
                return False
        return False

    def construct_rearrange_from_propositions(
        self,
        propositions,
        constraints,
        proposition_groups,
        prop_dependency_dict,
        environment_graph,
        temp_constraints,
    ):
        """
        Convert a set of propositions EvaluationProposition into RearrangeAction, containing information of which objects should
        go where and next to what. E.g.
        a proposition 3 airplanes on table is converted to 3 rearrange actions  RearrangeAction(Plane, Airplane)
        :proposition: List of EvaluationPropositions that we want to convert
        :constraints: A list of Evaluation Constraints
        :environment_graph: Instance of WorldGraph containing the environment information.
        :returns: list of RearrangeAction
        """
        # The code works by first assigning terminal propositions since it forces some objects that should not be changed,
        # then it assigns the non terminal ones and finally the next to propositions, since they are based on other props
        # already assigned
        proposition_groups = copy.deepcopy(proposition_groups)

        # Look at all the propositions that have a same argument constraint and cluster them so that when we sample
        # an object, we make sure that all the propositions will have that same object.
        same_arg_component_dict, num_components = self.reduce_args_with_same_constraint(
            propositions, constraints
        )
        # For each cluster of proposition, which object was assigned?
        same_arg_prop_assignments = [None for _ in range(num_components)]

        # Dict from prop id, relation_type to the receptacles assigned
        different_arg_prop_assignments = {}
        different_arg_component_dict = self.build_different_arg_component_dict(
            constraints
        )

        # For the floor constraints, they should also have some is_in_room.
        # We map floor proposition to room proposition
        self.floor_to_room_prop = {}
        self.room_to_floor_prop = {}
        in_floor_prop = [
            (prop_ind, prop)
            for prop_ind, prop in enumerate(propositions)
            if prop.function_name == "is_on_floor"
        ]
        in_room_prop = [
            (prop_ind, prop)
            for prop_ind, prop in enumerate(propositions)
            if prop.function_name == "is_in_room"
        ]
        for f_id, prop_floor in in_floor_prop:
            for r_id, prop_room in in_room_prop:
                tuple_floor = tuple(sorted(prop_floor.args["object_handles"]))
                tuple_room = tuple(sorted(prop_room.args["object_handles"]))
                if tuple_floor == tuple_room:
                    self.floor_to_room_prop[f_id] = r_id
                    self.room_to_floor_prop[r_id] = f_id

        # Get the terminal constraints
        terminal_constraints = [
            constraint
            for constraint in constraints
            if isinstance(constraint, TerminalSatisfactionConstraint)
        ]

        # Track the propositions that should be next to
        next_to_prop_indices = [
            i
            for i, prop in enumerate(propositions)
            if prop.function_name == "is_next_to"
        ]
        assert len(terminal_constraints) < 2
        if len(terminal_constraints) > 0:
            terminal_prop_indices = terminal_constraints[0].proposition_indices
        else:
            terminal_prop_indices = []
        terminal_prop_indices = [
            i for i in terminal_prop_indices if i not in next_to_prop_indices
        ]
        non_terminal_prop_indices = [
            i
            for i in range(len(propositions))
            if i not in terminal_prop_indices + next_to_prop_indices
        ]

        # We will first assign the terminal propositions, then assign the normal propositions, finally we assign
        # the next to propositions with the assignments we already made
        sorted_propositions = [
            (index, propositions[index])
            for index in terminal_prop_indices
            + non_terminal_prop_indices
            + next_to_prop_indices
        ]

        # For the next_to proposition, it is hard to know whether a next to b means we should put a to be where b
        # is or b where a is. We will do a heuristic. If only a appears before the proposition, b should go next to a
        # if only b appears, b should go to a. If both appear the first one that appears should be the receptacle
        # We map here which target to use.
        self.next_to_move = {}
        for prop_ind in next_to_prop_indices:
            next_to_proposition = propositions[prop_ind]
            prev_propositions = self.get_earlier_props(prop_ind, proposition_groups)
            prev_propositions = [
                propositions[prop_ind]
                for prop_ind in prev_propositions
                if propositions[prop_ind].function_name in ["is_on_top", "is_inside"]
            ]
            objects_prev_propositions = [
                prop.args["object_handles"] for prop in prev_propositions
            ]
            object_to_move = ["entity_handles_a", "entity_handles_b"]
            found_prev_prop = False
            for object_group in objects_prev_propositions:
                if found_prev_prop:
                    break
                object_group_set = set(object_group)
                if (
                    len(
                        set(next_to_proposition.args["entity_handles_b"]).intersection(
                            object_group_set
                        )
                    )
                    > 0
                ):
                    found_prev_prop = True
                elif (
                    len(
                        set(next_to_proposition.args["entity_handles_a"]).intersection(
                            object_group_set
                        )
                    )
                    > 0
                ):
                    found_prev_prop = True
                    object_to_move = ["entity_handles_b", "entity_handles_a"]
            self.next_to_move[prop_ind] = object_to_move

        rearrange_actions = []
        # We address first the non next to props. We will start with the terminal ones
        for prop_ind, proposition in sorted_propositions:
            if prop_ind in self.floor_to_room_prop:
                # The in_room predicate will take care of this.
                continue
            proposition_dependencies = prop_dependency_dict.get(prop_ind, [])
            task_type = proposition.function_name
            function_args = proposition.args
            num_objects = function_args["number"]

            # Given a proposition and the graph, get all the possible objects and receptacles
            # Valid for that proposition
            (
                target_objects,
                target_receptacles,
                same_rec,
            ) = self.get_receptacles_objects_from_prop(
                proposition, prop_ind, environment_graph
            )
            # We look at whether the object or handle proposition was part of same argument constraint
            # if so, if it was previously assigned, we use that assignment

            if proposition.function_name in OBJECT_STATES:
                targets = [target_objects]
                relation_names = ["object_handles"]
            else:
                targets = [target_objects, target_receptacles]
                if proposition.function_name != "is_next_to":
                    relation_names = ["object_handles", "receptacle_handles"]
                else:
                    relation_names = self.next_to_move[prop_ind]

            for it, relation_name in enumerate(relation_names):
                if (prop_ind, relation_name) in same_arg_component_dict:
                    component_index = same_arg_component_dict[(prop_ind, relation_name)]
                    assigned_object = same_arg_prop_assignments[component_index]
                    # Has this been assigned already?
                    if assigned_object != None:
                        targets[it] = [assigned_object]

            # Make sure we do the assignment so that different arg constraints are satisfied
            for it, relation_name in enumerate(relation_names):
                # Now we look at DifferentArg constraints
                if (prop_ind, relation_name) in different_arg_component_dict:
                    props_to_compare_against = different_arg_component_dict[
                        (prop_ind, relation_name)
                    ]
                    list_objects_exclude = []
                    for other_prop_ind, relation_type in props_to_compare_against:
                        if (
                            other_prop_ind,
                            relation_type,
                        ) in different_arg_prop_assignments:
                            list_objects_exclude += different_arg_prop_assignments[
                                (other_prop_ind, relation_type)
                            ]
                    if assigned_object != None:
                        targets[it] = [
                            objn
                            for objn in target_objects
                            if objn not in list_objects_exclude
                        ]

            if len(targets) == 1:
                # It is a state predicate
                target_objects = targets[0]
                target_receptacles = None
            else:
                target_objects, target_receptacles = targets
            # Do not use objects that were part of a terminal satisfaction constraint before
            # TODO: This wont cover cases of Put A in C and then in B again
            # later we should store for block objects whether the proposition index is happening
            # before the terminal constraint
            objects_with_earlier_terminal = []
            if proposition.function_name != "is_next_to":
                objects_with_earlier_terminal = [
                    obj_name
                    for obj_name, prop_ind_terminal in self.block_objects
                    if self.is_earlier(prop_ind_terminal, prop_ind, proposition_groups)
                ]
            target_objects = [
                obj_name
                for obj_name in target_objects
                if obj_name not in objects_with_earlier_terminal
            ]

            # Sample the objects
            sampled_objects = self.random.sample(target_objects, num_objects)

            # Select the receptacles we will use. If the receptacle has to be the same
            # we will sample one receptacle times the number of objects. Otherwise,
            # we sample num_object receptacles.
            if target_receptacles is not None:
                try:
                    if same_rec:
                        sampled_receptacles = [
                            self.random.choice(target_receptacles)
                        ] * num_objects
                    else:
                        sampled_receptacles = self.random.choices(
                            target_receptacles, k=num_objects
                        )
                except Exception:
                    raise ValueError("No receptacles to sample from")

            if task_type == "is_next_to":
                # Sampled receptacles are actually objects, we need to figure out where these objects should go
                next_to = sampled_receptacles
                sampled_receptacles = []
                rearrange_ids = []
                for rec in next_to:
                    # TODO: right now we are not considering tasks where you first put two things together and later put things somewhere else
                    # we assume that the object we are placing next to will not move anymore, and that such object was only moved to a single place
                    candidate_rearrangements = [
                        (rearrange, it)
                        for it, rearrange in enumerate(rearrange_actions)
                        if rearrange.obj == rec
                    ]
                    if len(candidate_rearrangements) > 1:
                        # If there is a while satisfied dependency, this will tell us which candidate rearrangement
                        # we should pay attention to
                        candidate_prop_inds = [
                            prop_dep.depends_on
                            for prop_dep in proposition_dependencies
                            if prop_dep.relation_type == "while_satisfied"
                        ]
                        candidate_prop_inds = [
                            item for sublist in candidate_prop_inds for item in sublist
                        ]
                        if len(candidate_prop_inds) > 0:
                            candidate_rearrangements = [
                                (rearrange, it)
                                for it, rearrange in enumerate(rearrange_actions)
                                if rearrange.obj == rec
                                and rearrange.prop_ind in candidate_prop_inds
                            ]
                        else:
                            candidate_rearrangements = [candidate_rearrangements[0]]
                    assert len(candidate_rearrangements) < 2
                    if len(candidate_rearrangements) == 0:
                        # This means that the object we want to be next to was not placed anywhere, we will
                        # infer where the object was originally and place the object next to it

                        og_location = environment_graph.find_furniture_for_object(
                            environment_graph.get_node_from_name(rec)
                        ).name
                        rearrange_ids.append(None)
                        sampled_receptacles.append(og_location)
                    else:
                        rearrange_ids.append(candidate_rearrangements[0][0].prop_ind)
                        sampled_receptacles.append(candidate_rearrangements[0][0].to)
                next_to_prop_and_obj = [
                    (ind, objn) for objn, ind in zip(next_to, rearrange_ids)
                ]

            ## Fill up the same argument
            if task_type == "is_next_to":
                sampled_elements = [sampled_objects, next_to]
                relation_names = self.next_to_move[prop_ind]
            elif task_type in OBJECT_STATES:
                sampled_elements = [sampled_objects]
            else:
                sampled_elements = [sampled_objects, sampled_receptacles]
            # Based on same argument constraint, assign objects and receptacles to the graph.
            for it, relation_name in enumerate(relation_names):
                if (prop_ind, relation_name) in same_arg_component_dict:
                    group_id = same_arg_component_dict[(prop_ind, relation_name)]
                    assert len(sampled_elements[it]) > 0
                    same_arg_prop_assignments[group_id] = sampled_elements[it][0]

                different_arg_prop_assignments[
                    (prop_ind, relation_name)
                ] = sampled_elements[it]

            if prop_ind in terminal_prop_indices:
                # The objects we select here should not be moved anywhere else
                self.block_objects += [
                    (obj_name, prop_ind) for obj_name in sampled_elements[0]
                ]

            if task_type == "is_next_to":
                for objn, targetn, prop_and_obj in zip(
                    sampled_elements[0], sampled_receptacles, next_to_prop_and_obj
                ):
                    rearrange_actions.append(
                        RearrangeAction(
                            prop_ind=prop_ind,
                            to=targetn,
                            obj=objn,
                            next_to=prop_and_obj,
                            preposition=None,
                        )
                    )
            elif task_type in OBJECT_STATES:
                for objn in sampled_elements[0]:
                    rearrange_actions.append(
                        ChangeStateAction(
                            prop_ind=prop_ind, obj=objn, attribute=task_type
                        )
                    )
            else:
                for objn, targetn in zip(*sampled_elements):
                    rearrange_actions.append(
                        RearrangeAction(
                            prop_ind=prop_ind,
                            to=targetn,
                            obj=objn,
                            next_to=None,
                            preposition=None,
                        )
                    )
        #######################
        # Final Postprocessing
        #######################
        # Store dictionary from object to move to proposition index. We do this to ensure that
        # next_to propositions go after the object that was placed
        rearrange_action_dict = {}
        for ra in rearrange_actions:
            rearrange_action_dict[ra.prop_ind] = ra

        current_dag_edges = copy.deepcopy(temp_constraints[0].args["dag_edges"])

        for r_action in rearrange_actions:
            if type(r_action) == ChangeStateAction:
                # If there is some ontop/inside in the same prop group, make sure this goes after
                current_prop_ind = r_action.prop_ind
                prop_group, _ = [
                    (p_group, ind)
                    for ind, p_group in enumerate(proposition_groups)
                    if current_prop_ind in p_group
                ][0]
                r_actions_same_group = [
                    r_act
                    for r_act in rearrange_actions
                    if r_act.prop_ind in prop_group and type(r_act) != ChangeStateAction
                ]
                for r_act_sg in r_actions_same_group:
                    if r_act_sg.obj != r_action.obj:
                        # There was a rearrangement for this object, make sure sure that the change state action happens after this rearrangement
                        current_dag_edges.append([r_act_sg.prop_ind, current_prop_ind])

                continue
            # If there is a next to, we should create a new proposition_group to make sure that one goes later.
            # Meaning that the next to propositions should come right after the non next to

            if r_action.next_to is not None:
                other_prop_ind, next_to = r_action.next_to
                current_prop_ind = r_action.prop_ind
                if other_prop_ind is not None:
                    obj, to = (
                        rearrange_action_dict[other_prop_ind].obj,
                        rearrange_action_dict[other_prop_ind].to,
                    )
                    # Assign the next to proposition

                    for ra in rearrange_actions:
                        if ra.next_to is not None and ra.obj == obj and ra.to == to:
                            other_prop_ind = ra.prop_ind

                prop_group, prop_group_ind = [
                    (p_group, ind)
                    for ind, p_group in enumerate(proposition_groups)
                    if current_prop_ind in p_group
                ][0]
                if other_prop_ind in prop_group and other_prop_ind != current_prop_ind:
                    # If there is a next to and a on top, make sure that the next to goes after
                    current_dag_edges.append([other_prop_ind, current_prop_ind])

                # assert self.is_earlier(other_prop_ind, current_prop_ind, proposition_groups)
                r_action.next_to = next_to
                prop_id_for_preposition = other_prop_ind
            else:
                prop_id_for_preposition = r_action.prop_ind
            # Fill out the preposition
            if prop_id_for_preposition is None:
                # We have to place the object next to an object for which there was not preposition. We will
                # check the location of the other object
                prop_type = "is_on_top"

            else:
                # If the proposition is next to, we will put it in the same place as the next to prop
                prop_type = propositions[prop_id_for_preposition].function_name
                if prop_type == "is_next_to":
                    # Check where we put objects in the previous action
                    caction = [
                        action
                        for action in rearrange_actions
                        if type(action) is not ChangeStateAction
                        and action.preposition is not None
                        and action.prop_ind == prop_id_for_preposition
                    ]
                    if len(caction) > 0:
                        r_action.preposition = caction[0].preposition
                elif prop_type in ["is_on_top", "is_inside"]:
                    dict_prep = {"is_on_top": "on", "is_inside": "within"}
                    r_action.preposition = dict_prep[prop_type]

            neighbors = environment_graph.get_neighbors(
                environment_graph.get_node_from_name(r_action.to)
            )
            rec = [
                neighbor.get_property("type")
                for neighbor, rel in neighbors.items()
                if rel == "joint"
            ]
            if r_action.preposition is not None and r_action.preposition not in rec:
                r_action.preposition = None

            if r_action.preposition is None:
                r_action.preposition = "on" if "on" in rec else "within"
        temp_constraint = [
            TemporalConstraint(
                dag_edges=current_dag_edges, n_propositions=len(propositions)
            )
        ]
        proposition_groups = get_proposition_groups_from_episode(
            temp_constraint, num_propositions=len(propositions)
        )

        # Finally for all next_to, if there is a proposition with exact same from and to, we delete that
        # to avoid repeating rearrange actions
        # TODO: this will not work for Move to A then B then A
        next_to_tuples = [
            (r_action.obj, r_action.to)
            for r_action in rearrange_actions
            if r_action.next_to is not None
        ]
        from_to_map, from_to_map_all = {}, {}
        remove_rearrange_ids = []
        for rid, r_action in enumerate(rearrange_actions):
            if type(r_action) is ChangeStateAction:
                continue
            if not r_action.next_to is None:
                if (r_action.obj, r_action.to) in from_to_map:
                    # If there was an exact proposition with next to, remove one of them
                    remove_rearrange_ids.append(rid)
                from_to_map[(r_action.obj, r_action.to)] = True
            else:
                if (r_action.obj, r_action.to) in next_to_tuples or (
                    r_action.obj,
                    r_action.to,
                ) in from_to_map_all:
                    # If there was a proposition with next to, keep the next to, remove this one
                    remove_rearrange_ids.append(rid)
            from_to_map_all[(r_action.obj, r_action.to)] = True

        keep_rearrange_ids = [
            it for it in range(len(rearrange_actions)) if it not in remove_rearrange_ids
        ]
        rearrange_actions = [rearrange_actions[ind] for ind in keep_rearrange_ids]

        return rearrange_actions, proposition_groups

    def solve_dag(
        self,
        propositions,
        constraints,
        prop_dependencies,
        environment_graph,
        env_interface,
    ):
        """
        Given a list of propositions coming from the eval functions, returns a set of rearrange skills
        per agent, with Wait skills representing deadlocks.
        E.g.  {0: [Rearrange(O1,R1), Wait1, Rearrange(O3, R3)], 1: [Rearrange(O2, R2), Wait1]}
        where Wait is a deadlock.
        :propositions: A list of EvaluationPropositions.
        :constraints: The evaluation constraints
        :prop_dependencies: The proposition dependencies
        :environment_graph: The environmentGraph
        :return: dictionary from agent to skills
        """

        # The temporal constraints in the evaluation function will be handled at the end, so we remove those.
        temp_constraints = [c for c in constraints if isinstance(c, TemporalConstraint)]

        constraints = [c for c in constraints if not isinstance(c, TemporalConstraint)]

        proposition_groups = get_proposition_groups_from_episode(
            temp_constraints, num_propositions=len(propositions)
        )

        # The proposition dependencies tell us that for 2 propositions the argument should be the same:
        for prop_dependency in prop_dependencies:
            depends_on_indices = (
                prop_dependency.proposition_indices + prop_dependency.depends_on
            )
            if (
                prop_dependency.relation_type == "while_satisfied"
                and len(prop_dependency.proposition_indices) == 1
            ):
                p_ind = prop_dependency.proposition_indices[0]
                if propositions[p_ind].function_name == "is_next_to":
                    same_arg_constraint = SameArgConstraint(
                        proposition_indices=depends_on_indices,
                        arg_names=["receptacle_handles"] * len(depends_on_indices),
                    )
                    constraints.append(same_arg_constraint)

        # Dictionary from proposition index to the proposition dependency.
        prop_dependency_dict = {}
        for prop_dependency in prop_dependencies:
            for ind in prop_dependency.proposition_indices:
                if ind not in prop_dependency_dict:
                    prop_dependency_dict[ind] = []

                prop_dependency_dict[ind].append(prop_dependency)

        # Get the number of RearrangeActions that will be assigned to the agents later
        (
            assigned_rearrange_actions,
            proposition_groups,
        ) = self.construct_rearrange_from_propositions(
            propositions,
            constraints,
            proposition_groups,
            prop_dependency_dict,
            environment_graph,
            temp_constraints,
        )
        proposition_map = {}

        THRESH_FOR_ART_STATE = 0.01

        # We convert here the Rearrange actions into Robot skills (Nav Pick Nav Place). We additionally add actions
        # such as opening containers when an object is inside a closed container
        # Get closed receptacles
        closed_receptacles = []
        for fur in environment_graph.get_all_furnitures():
            if fur.is_articulated() and not is_open(
                fur, env_interface, THRESH_FOR_ART_STATE
            ):
                closed_receptacles.append(fur.name)

        # Convert the rearrange attributes into actions, accounting for whether objects are in a container or not
        num_agents = len(env_interface.sim.agents_mgr)
        human_agent_ids = [
            ind
            for ind in range(num_agents)
            if env_interface.sim.agents_mgr[ind].cfg.articulated_agent_type
            == "KinematicHumanoid"
        ]
        all_agent_ids = list(range(num_agents))

        constraints_agent_assignment = {}

        objects_navigated = []
        for rearrange_action in assigned_rearrange_actions:
            prop_ind = rearrange_action.prop_ind
            if prop_ind not in proposition_map:
                proposition_map[prop_ind] = []
            if type(rearrange_action) is ChangeStateAction:
                constraints_agent_assignment[prop_ind] = human_agent_ids
            else:
                constraints_agent_assignment[prop_ind] = all_agent_ids

            proposition_map[prop_ind].append(
                self.get_list_actions_from_rearrange(
                    rearrange_action,
                    environment_graph,
                    closed_receptacles,
                    objects_navigated,
                    env_interface,
                )
            )
            # Check objects we navigated to so that they are in world space
            objects_navigated.append(rearrange_action.obj)

        # Construct new prop groups
        propositions_groups_entities = []
        all_prop_indices = []
        for prop_group in proposition_groups:
            flat_prop_group = []
            prop_indices = []
            for prop_ind in prop_group:
                if prop_ind in proposition_map:
                    flat_prop_group += proposition_map[prop_ind]
                    n_actions_on_this_prop = len(proposition_map[prop_ind])
                    prop_indices += [prop_ind] * n_actions_on_this_prop
            propositions_groups_entities.append(flat_prop_group)
            all_prop_indices.append(prop_indices)

        actions_agent = self.assign_dag(
            propositions_groups_entities,
            all_prop_indices,
            constraints_agent_assignment,
            num_agents,
        )
        return actions_agent


class ScriptedCentralizedPlanner(Planner):
    def __init__(self, plan_config, env_interface):
        # Call Planner class constructor
        super().__init__(plan_config, env_interface)

        self.seed = 142
        self.reset()
        self.curr_hist = ""

    def reset(self):
        """Method to reset planner to make it ready for next episode."""

        self.last_high_level_actions = {}
        self.prop_res = PropositionResolver(
            self.env_interface.perception.sim_handle_to_name,
            self.env_interface.perception.region_id_to_name,
            seed=self.seed,
        )

        # State variables
        num_agents = len(self.agents)  # TODO: put this somewhere
        self.plan_indx = [-1 for _ in range(num_agents)]
        self.is_waiting = [False for _ in range(num_agents)]

        self.next_skill_agents = {ind: True for ind in range(num_agents)}
        self.actions_per_agent = None
        self.move_next_skill = True
        self.curr_hist = ""
        self.trace = ""
        # Reset agents
        for agent in self._agents:
            agent.reset()

        return

    def get_plan_dag(self, environment_graph):
        """
        Given a list of propositions coming the current episode, assign the actions that each agent should be taking.
        actions_per_agent will be a dictionary from agent index to a list of actions
        """

        curr_episode = self.env_interface.env.env.env._env.current_episode

        prop_dependencies = curr_episode.evaluation_proposition_dependencies
        propositions = curr_episode.evaluation_propositions

        # For every proposition, store the object and target so that we can use in future constraints
        self.actions_per_agent = self.prop_res.solve_dag(
            propositions,
            curr_episode.evaluation_constraints,
            prop_dependencies,
            self.env_interface.full_world_graph,
            self.env_interface,
        )

    def actions_parser(self, next_skill_agents):
        """
        Uses the prerecorded actions_per_agent (which contain RearrangeActions and deadlocks) to select the next action
        when an agent reaches a deadlock, stays there until the other agent reaches the deadlock as well.
        :param next_skill_agents: dictionary indicating per agent whether it should it should move to the next skill
        :returns high_level_actions: dictionary from agent to high level action (e.g. rearrange)
        :returns high_level_action_str: a string formatted to show the actions per agent
        """

        # Advance the plan for any agent that has to switch to next skills

        for agent_id in next_skill_agents:
            if next_skill_agents[agent_id] and not self.is_waiting[agent_id]:
                self.plan_indx[agent_id] += 1

        is_done = [False, False]
        for agent_id in next_skill_agents:
            curr_step_skill = self.plan_indx[agent_id]
            if curr_step_skill >= len(self.actions_per_agent[agent_id]):
                self.is_waiting[agent_id] = True
                is_done[agent_id] = True
            elif "Wait" in self.actions_per_agent[agent_id][curr_step_skill]:
                self.is_waiting[agent_id] = True

        if np.all(is_done):
            return {}, ""

        is_waiting = np.all(self.is_waiting)
        if is_waiting:
            # All the agents started to wait, means we can unblock
            self.is_waiting = [False for _ in range(len(self.actions_per_agent))]
            for agent_id in self.actions_per_agent:
                self.is_waiting[agent_id] = False

            return self.actions_parser({0: True, 1: True})

        high_level_actions = {}
        for agent_id in self.actions_per_agent:
            if self.is_waiting[agent_id]:
                high_level_actions[agent_id] = ("Wait", "", "")
            else:
                curr_step_skill = self.plan_indx[agent_id]
                agent_action_name = self.actions_per_agent[agent_id][curr_step_skill]
                high_level_actions[agent_id] = agent_action_name

        high_level_actions_str = self.stringify_actions(high_level_actions)

        return high_level_actions, high_level_actions_str

    def stringify_actions(self, actions_per_agent):
        res_str = ""
        for agent in range(len(actions_per_agent)):
            action_name, action_arg, _ = actions_per_agent[agent]
            if action_name == "Wait":
                action_str = action_name
            else:
                action_str = f"{action_name}[{action_arg}]"
            res_str += f"Agent_{agent}_Action: {action_str}\n"
        return res_str

    def is_task_done(self):
        is_done = np.all(
            [
                curr_ind >= len(self.actions_per_agent[agent_ind])
                for agent_ind, curr_ind in enumerate(self.plan_indx)
            ]
        )
        return is_done

    def get_next_action(self, instruction, observations, world_graph):
        """
        Gives the next low level action to execute
        """
        # Set variable to indicate if planner should stop
        should_stop = False

        first_action = False
        replan_required = {agent.uid: False for agent in self.agents}

        if self.trace == "":
            self.trace += f"Task: {instruction}\n"

        if self.actions_per_agent is None:
            # Generate the dag in the first step
            self.get_plan_dag(world_graph)
            first_action = True

        if len(self.curr_hist) == 0:
            # Set initial prompt. Note, this could probably go somewhere else
            # or maybe be shared across the planners
            self.curr_hist = f"Task: {instruction}\n"
            obj_list = (
                self.agents[0].get_tool_from_name("FindObjectTool")._get_object_list()
            )
            fur_list = (
                self.agents[0]
                .get_tool_from_name("FindReceptacleTool")
                ._get_receptacles_list()
            )
            self.curr_hist += f"Furniture:\n{fur_list}\n"
            self.curr_hist += f"Objects:\n{obj_list}\n"

        planner_info = {"replanned": {agent_id: False for agent_id in replan_required}}
        print_str = ""

        replanned = copy.deepcopy(self.next_skill_agents)
        for agent_id in replanned:
            if self.is_waiting[agent_id]:
                replanned[agent_id] = False

        # Get the next action for both agents
        high_level_actions, high_level_actions_str = self.actions_parser(
            self.next_skill_agents
        )

        # Mark end if both actions are empty
        if len(high_level_actions) == 0:
            should_stop = True
            low_level_actions = {}
            responses = {}
            for agent_id in replanned:
                replanned[agent_id] = True
                self.last_high_level_actions[agent_id] = ("Done", None, None)

        else:
            for agent_id, hl_action in high_level_actions.items():
                self.last_high_level_actions[agent_id] = hl_action

            # Remove Wait from high level actions. We do this because Wait is not executable, instead
            # we make the agent do nothing.
            high_level_actions = {
                agent: agent_action
                for agent, agent_action in high_level_actions.items()
                if agent_action[0].lower() != "wait"
            }
            self.attempted_high_level_action = high_level_actions

            # Get the low level action vector
            low_level_actions, responses = self.process_high_level_actions(
                high_level_actions, observations
            )
            for agent_id, resp in responses.items():
                self.next_skill_agents[int(agent_id)] = len(resp) > 0

            if any(replanned.values()):
                print_str += high_level_actions_str

            if any(responses.values()):
                self.move_next_skill = True

            else:
                self.move_next_skill = False

            # Mark if replanning is required for each agent
            if len(responses) and not first_action:
                for id_agent in [0, 1]:
                    if id_agent in responses and len(responses[id_agent]) > 0:
                        replan_required[id_agent] = True
                    else:
                        replan_required[id_agent] = False

            # A non-empty response marks completion or error in finishing the high level task
            # Thus here, we pop the high level action which receives a non-empty response and
            # select the next skill.
            for agent_uid in self.agent_indices:
                # If response is empty, then change it to indicate progress
                if agent_uid not in responses:
                    hl_action = self.last_high_level_actions[agent_uid]
                    response = (
                        f"Action {hl_action[0]}[{hl_action[1]}] is still in progress."
                    )
                    responses[agent_uid] = response

        obs_str = ""
        # Update data to send to store for fine-tuning. Maybe this should go somewhere else
        for agent_uid in sorted(responses.keys()):
            # Print the response
            response = responses[agent_uid]
            if replan_required[agent_uid]:
                obs_str += f"Agent_{agent_uid}_observation:{response}\n"

        self.curr_hist = self.curr_hist + print_str + obs_str

        if should_stop:
            self.trace = self.curr_hist
        planner_info["responses"] = responses
        if len(print_str + obs_str) > 0:
            planner_info["print"] = print_str + obs_str
        planner_info["high_level_actions"] = self.last_high_level_actions
        planner_info["prompts"] = {agent.uid: self.curr_hist for agent in self.agents}
        planner_info["replan_required"] = replan_required
        planner_info["is_done"] = {agent.uid: should_stop for agent in self.agents}
        planner_info["traces"] = {agent.uid: self.trace for agent in self.agents}
        planner_info["replanned"] = replanned
        planner_info["actions_per_agent"] = self.actions_per_agent

        # update world based on actions
        self.update_world(responses)
        return low_level_actions, planner_info, should_stop
