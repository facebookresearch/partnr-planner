# Habitat-LLM Evaluation Engine

We support the evaluation of single or multi-agent rearrange household tasks using primary metrics of Success and % Completion. This README discusses the capabilities and APIs of our evaluation components. Defining the evaluation for an episode consists of providing a list of propositions (`EvaluationProposition`), a list of temporal propositon dependencies (`EvaluationPropositionDependency`), and a list of constraints to be applied over propositions (`EvaluationConstraint`).

At a high level, the evaluation of an episode is constructed using the following flow:

1. [[Link](#predicates-and-propositions)] **Define propositions** (list of `EvaluationProposition`)
2. [[Link](#proposition-dependencies)] **Define when propositions are queried** (list of `EvaluationPropositionDependency`)
3. [[Link](#constraints)] **Define what constitutes success** (list of `EvaluationConstraint`):
    - Which propositions must be satisfied? (assumption: all of them)
    - When must propositions be satisfied? (`TemporalConstraint`, `TerminalSatisfactionConstraint`)
    - How must propositions be satisfied? (`SameArgConstraint`, `DifferentArgConstraint`)

## Predicates and Propositions

A predicate is a logical entity that defines a relation amongst objects, furniture, or rooms. A proposition is the instantiation of a predicate with arguments and has a boolean truth value. When attaching a proposition to an episode, the following must be provided:

```python
EvaluationProposition(function_name: str, args: Dict[str, Any])
```

`function_name` is a string matching the desired predicate and `args` constraints arguments necessary to instantiate it.  The vocabulary of predicates we support is as follows:

### Rearrange Predicates

Predicate: **is_on_top**
Definition: In the single-object single-furniture case, returns True if the object is on top of the furniture.
Function arguments:

```python
is_on_top(
    object_handles: List[str],                   # OR: any of `object_handles`
    receptacle_handles: List[str],               # OR: any of `receptacle_handles` (furniture)
    number: Optional[int] = 1,                   # n of `object_handles` must satisfy the proposition.
    is_same_receptacle: Optional[bool] = False,  # all n of `object_handles` must be satisfied by the same `receptacle_handles`.
)
```

Predicate: **is_inside**
Definition: In the single-object single-furniture case, returns True if the object is within the furniture.
Function arguments:

```python
is_inside(
    object_handles: List[str],                   # OR: any of `object_handles`
    receptacle_handles: List[str],               # OR: any of `receptacle_handles` (furniture)
    number: Optional[int] = 1,                   # n of `object_handles` must satisfy the proposition.
    is_same_receptacle: Optional[bool] = False,  # all n of `object_handles` must be satisfied by the same `receptacle_handles`.
)
```

Predicate: **is_in_room**
Definition: In the single-object single-room case, returns True if the object is in the provided room.
Function arguments:

```python
is_in_room(
    object_handles: List[str],   # OR: any of `object_handles`
    room_ids: List[str],         # OR: any of `room_ids`
    number: int = 1,             # n of `object_handles` must satisfy the proposition.
    is_same_room: bool = False,  # all n of `object_handles` must be satisfied by the same `room_ids`.
)
```

Predicate: **is_on_floor**
Definition: In the single-object case, returns True if the object is on the floor of any navmesh island.
Function arguments:

```python
is_on_floor(
    object_handles: List[str],  # OR: any of `object_handles`
    number: int = 1             # n of `object_handles` must satisfy the proposition.
)
```

### Spatial Predicates

Predicate: **is_next_to**
Definition: In the case of one entity A and one entity B, returns True if entity A is next to entity B.
Function arguments:

```python
is_next_to(
    entity_handles_a: List[str],  # OR: any of `entity_handles_a`
    entity_handles_b: List[str],  # OR: any of `entity_handles_b`
    number: int = 1,              # n of `entity_handles_a` must satisfy the proposition.
    is_same_b: bool = False,      # all n of `entity_handles_a` must be satisfied by the same `entity_handles_b`.
    l2_threshold: float = 0.5,    # regularized horizontal L2 distance allow between the objects' centers.
)
```

Predicate: **is_clustered**
Definition: A generalization of `is_next_to` to more than two objects. In short, every object must be `next_to` at
            least one other object. Returns True if each entity in *args is `next_to` at least one other arg entity.
            Supported characteristics:
            - entity ambiguity (each arg is a list of allowed entities)
            - subset counts (required `number` for each argument)
Function arguments:

```python
is_clustered(
    *args: List[str],           # each arg is a list of object handles that can be used to satisfy the relation.
                                # Each arg must be satisfied. In the number=[1, ..., 1] case,
                                # at least one handle from each argument must satisfy the relation.
    number: List[int] = None,   # n_i elements of the ith arg must satisfy the cluster criteria. Default: [1, ..., 1]
    l2_threshold: float = 0.5,  # Distance threshold for object-object `next_to`.
)
# Example:
#   Instruction: "set one toy, two books, and the hat next to each other."
#   Proposition: is_clustered(["toy_1", "toy_2"], ["book_1", "book_2", "book_3"], ["hat_1"], number=[1,2,1])
#   Interpretation: either toy, any two books, and the hat.
```

### Object State Predicates

When an object is queried for a particular object state, the value is pulled from the object state machine in collaboration-sim. Each object state will take on the initial value specified in the `CollaborationEpisode`. If no value is given, a default value will be adopted.

Predicate: **is_clean**
Definition: queries the state of an object and returns if it is clean.
Object Default: False
Function arguments:

```python
is_clean(
    object_handles: List[str],  # OR: any of `object_handles`
    number: int = 1,            # n of `object_handles` must satisfy the proposition.
)
```

Predicate: **is_dirty**
Definition: queries the state of an object and returns if it is dirty (opposite of clean).
Object Default: True
Function arguments:

```python
is_dirty(
    object_handles: List[str],  # OR: any of `object_handles`
    number: int = 1,            # n of `object_handles` must satisfy the proposition.
)
```

Predicate: **is_filled**
Definition: queries the state of an object and returns if it is filled.
Object Default: False
Function arguments:

```python
is_filled(
    object_handles: List[str],  # OR: any of `object_handles`
    number: int = 1,            # n of `object_handles` must satisfy the proposition.
)
```

Predicate: **is_empty**
Definition: queries the state of an object and returns if it is empty (opposite of filled).
Object Default: True
Function arguments:

```python
is_empty(
    object_handles: List[str],  # OR: any of `object_handles`
    number: int = 1,            # n of `object_handles` must satisfy the proposition.
)
```

Predicate: **is_powered_on**
Definition: queries the state of an object and returns if it is powered on.
Object Default: False
Function arguments:

```python
is_powered_on(
    object_handles: List[str],  # OR: any of `object_handles`
    number: int = 1,            # n of `object_handles` must satisfy the proposition.
)
```

Predicate: **is_powered_off**
Definition: queries the state of an object and returns if it is powered off (opposite of powered on).
Object Default: True
Function arguments:

```python
is_powered_off(
    object_handles: List[str],  # OR: any of `object_handles`
    number: int = 1,            # n of `object_handles` must satisfy the proposition.
)
```

#### Example: "Bring a spoon to the table"

note: in the following example, entity names (spoon_1, table_1) should actually be entity hash handles. Presented this way for interpretability.

If we have a single spoon and a single table in the scene, this is straightforward:

```python
is_on_top(["spoon_1"], ["table_1"])
```

What if we have multiple spoons?

```python
is_on_top(["spoon_1", "spoon_2", "spoon_3"], ["table_1"])
```

What if we also have multiple valid tables?

```python
is_on_top(["spoon_1", "spoon_2", "spoon_3"], ["table_1", "table_2"])
```

What if we want to have two spoons brought to the table?

```python
is_on_top(["spoon_1", "spoon_2", "spoon_3"], ["table_1", "table_2"], number=2)
```

What if we want to have two spoons brought to the same table?

```python
is_on_top(["spoon_1", "spoon_2", "spoon_3"], ["table_1", "table_2"], number=2, is_same_receptacle=True)
```

## Proposition Dependencies

Dependencies define when a proposition will be queried for satisfaction. Some propositions should only be queried in relation to the sequence of states of other propositions. These dependencies are critical in tasks involving multi-step and temporal rearrangements. Without such dependencies, propositions may be marked satisfied earlier than they are intended to be used. This can cause over-estimation of collaboration success, and when temporal constraints are used (see section on evaluation constraints), under-estimation of collaboration success.

### Definition

```
EvaluationPropositionDependency(
    proposition_indices: List[int],
    depends_on: [int],
    relation_type: str  # while_satisfied, after_satisfied, after_unsatisfied, before_satisfied
    dependency_mode: str = "all"  # all, any
)
```

All propositions indexed by `proposition_indices` will depend on the propositions indexed by `depends_on`. The `relation_type` specifies what must be true of all propositions in `depends_on` (or at least one if `dependency_mode="any"`) in order for the propositions in `proposition_indices` to be evaluated. The relation types are as follows:

- **while_satisfied:** evaluate propositions indexed by `proposition_indices` when propositions indexed by `depends_on` are True.
- **after_satisfied:** evaluate propositions indexed by `proposition_indices` when propositions indexed by `depends_on` have each been satisfied at some point in the past.
- **after_unsatisfied:** evaluate propositions indexed by `proposition_indices` when propositions indexed by `depends_on` were at some point satisfied and no longer are.
- **before_satisfied:** evaluate propositions indexed by `proposition_indices` when propositions indexed by `depends_on` have yet to be satisfied.

### Example 1

Move the cup from the table to the sink. Move the cup back to the table.

Propositions:

```
0 is_inside(cup, sink)
1 is_on_top(cup, table)
```

Proposition Dependencies:

```
[
    EvaluationPropositionDependency(proposition_indices=[1], depends_on=[0], relation_type="after_unsatisfied"),
]
```
This means that proposition 1 will only be evaluated after proposition 0 changes from satisfied to unsatisfied. Without this dependency, proposition 1 will be marked True at the start of the episode and the agent will be given success.

## Example 2

Move the ball and the bat to the kitchen table and set them next to each other.
Then, move the ball and the bat to the closet and set them next to each other.

Propositions:

```
0 is_on_top(ball, table)
1 is_on_top(bat, table)
2 is_next_to(ball, bat)

3 is_in_room(ball, closet)
4 is_in_room(bat, closet)
5 is_next_to(ball, bat)
```

Proposition Dependencies:

```
[
    EvaluationPropositionDependency(proposition_indices=[2], depends_on=[0, 1], relation_type="while_satisfied"),
    EvaluationPropositionDependency(proposition_indices=[5], depends_on=[3, 4], relation_type="while_satisfied")
]
```

This means that the `is_next_to` propositions will only be evaluated when their corresponding placements are satisfied.

### Example 3

Move the ball and the bat to the kitchen table and set them next to each other.
Then, move the ball and the bat to the closet and set them next to each other.
Finally, move the ball and the bat back to the kitchen table and set them next to each other.

Propositions:

```
0 is_on_top(ball, table)
1 is_on_top(bat, table)
2 is_next_to(ball, bat)

3 is_in_room(ball, closet)
4 is_in_room(bat, closet)
5 is_next_to(ball, bat)

6 is_on_top(ball, table)
7 is_on_top(bat, table)
8 is_next_to(ball, bat)
```

Proposition Dependencies:

```
[
    EvaluationPropositionDependency(proposition_indices=[2], depends_on=[0, 1], relation_type="while_satisfied"),
    EvaluationPropositionDependency(proposition_indices=[5], depends_on=[3, 4], relation_type="while_satisfied"),
    EvaluationPropositionDependency(proposition_indices=[6, 7], depends_on=[3, 4], relation_type="after_unsatisfied"),
    EvaluationPropositionDependency(proposition_indices=[8], depends_on=[6,7], relation_type="while_satisfied"),
]
```

## Constraints

The predicates above are insufficient for evaluating tasks with certain characteristics (e.g. *"do X and then do Y"*, *"Put these objects on the same shelf"*). To support a diverse range of collaboration tasks, we enable constraints to act on top of propositions. If a proposition is satisfied but does not abide by a stated constraint, it is considered unsatisfied when determining task performance. The vocabulary of predicates we support is as follows:

Constraint: **TemporalConstraint**
Definition: A directed acyclic graph (DAG) over the indices of propositions that defines the temporal requirement of when propositions should be satisfied. If propositions are satisfied out of order, then the task is marked unsuccessful. Why is this necessary if we have proposition dependencies? Proposition dependencies do not allow us to invalidate cases of incorrect temporal order.
Example: *"Tidy up the room by... After, set the table by..."*
Constraint Arguments:

```python
dag_edges: List[Tuple[int]]           # defines a set of temporal relationships. (A, B) means prop A must be satisfied before prop B.
n_propositions: Optional[int] = None  # the number of propositions in the episode. Used to verify valid proposition indices.
```

Constraint: **SameArgConstraint**
Definition: Requires the argument used to satisfy a proposition to be the same within a pre-determined set of propositions.
Example: *"Place the books on the same shelf."*
Constraint Arguments:

```python
proposition_indices: List[int]        # the indices of propositions in EvaluationPropositions that this constraint apply to.
arg_names: List[str]                  # the name of the argument of which to compare the satisfying result of.
                                      # arg_names[i] corresponds to proposition_indices[i].
n_propositions: Optional[int] = None  # the number of propositions in the episode. Used to verify valid proposition indices.
```

Constraint: **DifferentArgConstraint**
Definition: Requires the argument used to satisfy a proposition to be unique within a pre-determined set of propositions.
Example: *"Place the candles on two separate end tables."*
Constraint Arguments:

```python
proposition_indices: List[int]        # the indices of propositions in EvaluationPropositions that this constraint apply to.
arg_names: List[str]                  # the name of the argument of which to compare the satisfying result of.
                                      # arg_names[i] corresponds to proposition_indices[i].
n_propositions: Optional[int] = None  # the number of propositions in the episode. Used to verify valid proposition indices.
```

Constraint: **TerminalSatisfactionConstraint**
Definition: Some propositions should remain satisfied from the time it was first satisfied to the end of the episode. This constraint is the terminal state check for those propositions.
Example: *"wash the mug and then fill it with juice."* The mug should become dirty again before the episode finishes. is_clean(mug) does not need to remain satisfied.
Constraint Arguments:

```python
proposition_indices: List[int]        # The indices of propositions that must be satisfied in the terminal task state.
n_propositions: Optional[int] = None  # the number of propositions in the episode. Used to verify valid proposition indices.
```

## Metrics

### TaskPercentComplete

For a given episode, this metric evaluates how much of the task was accomplished during the overall human+robot collaboration as measured by the ratio of the number of propositions satisfied to the total number of propositions. Propositions must be satisfied and satisfy all dependencies and constraints to be considered True. A proposition that requires `n` objects of a set will be unrolled to `n` propositions for the purposes of this metric (e.g. placing 3 out of 4 required objects nets a 75% completion).

### TaskStateSuccess

Binary success for a given episode. Equivalent to TaskPercentComplete == 1.00.

### TaskEvaluationLog

A measure that enables detailed logging and agent feedback. Contains keys:

```python
propositions: List[EvaluationProposition]            # List of propositions defined in the episode
dependencies: List[EvaluationPropositionDependency]  # List of dependencies between propositions defined in the episode
constraints: List[EvaluationConstraint]              # List of constraints defined in the episode
proposition_satisfied_at: List[int]                  # indicates at what time step the proposition was first satisfied. defaults to -1 (not satisfied)
constraint_satisfaction: np.ndarray                  # 2d array [i,j]. i is a constraint, j is a proposition, ndarray[i,j]==False if i invalidates j.
state_sequence: List[List[PropositionResult]]        # the PropositionResult of each proposition at each point in time.
```

### Existing Habitat-Lab Metrics

We also can evaluate any metric that is defined in Habitat-Lab (e.g., `num_steps`). These metrics can be included in evaluation by modifying the experiment config at `config.habitat.task.measurements`. See the task configs at `habitat_llm/conf/habitat_conf/task/` for where the active metrics are defined.
