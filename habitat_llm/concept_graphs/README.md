# Using [ConceptGraphs](https://arxiv.org/abs/2309.16650) with PARTNR

For our non-privileged world-graph baseline (see details [here](../world_model/README.md)) we provide code to create concept-graph from our scenes as well as code to use the JSONs with PARTNR. Note you can use pre-built concept-graphs provided as part of our [episode repository](https://huggingface.co/datasets/ai-habitat/partnr_episodes/tree/main/conceptgraphs) to get started quickly. If you already have the concept-graph JSONs from our data repository, please proceed to Step 4 to run the baseline over them.

There are four main steps to this:

1. [Installation](#installation)
1. [Logging data with Habitat-LLM](#logging-data-with-habitat-llm)
1. [Processing data to create a textual 3D scenegraph through CG pipeline](#creating-a-3dsg-using-conceptgraphs)
1. [Running the non-privileged baseline](#running-the-non-privileged-baseline)

## Installation

Install CG in a separate environment than your habitat-llm one. This is because
habitat-llm does not have any dependency on concept-graphs to run. Concept-graphs repo
has dependency on `HabitatDataset` dataloader which is implemented in this repository
for self-contained code placement.

To install concept-graphs follow steps on the [forked repository installation page](https://www.github.com/zephirefaith/concept-graphs).

## Logging data with Habitat-LLM

In order to generate a concept-graph for a given scene, we minimally requires the
following data from an agent exploring this scene (all time-synced):

1. RGB frames
1. Depth frames
1. Camera intrinsics
1. Camera pose (either with respect to the world or with respect to initial location,
   requires config change to switch from one to the other)

We need to configure a handful of parameters in order to start logging the above data in
habitat-llm. These parameters are read from [here](../conf/trajectory/trajectory_logger.yaml).

```yaml
save: True
agent_names: ['agent_1']  # list of all agents to log
save_path: 'data/traj0'  # root of data where to log data
save_options: ["rgb", "depth", "pose"]  # modalities to log during execution
  # rgb: accesses agent_N_articulated_agent_arm_rgb camera
  # depth: accesses agent_N_articulated_agent_arm_depth camera
  # pose: logs agent_N_articulated_agent_arm_rgb camera pose
```

They can be accessed via: `conf.trajectory` config variable in code. In order to execute
a run with logging enabled, use the following command:

```
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.planner_demo --config-name examples/planner_multi_agent_demo_config.yaml \
 planner='habitat_centralized_planner_multi_agent' \
 llm@planner.llm=llama2 \
 mode='cli' \
 partial_obs='False' \
 habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
 llm@planner.llm=openai_chat \
 trajectory.save=True \
 instruction="send agent_0 to all receptacles in the environment"
```

Make sure you have created the output directory for above run code to store data in:
`mkdir data/traj0`

Output directory is expected to have following organization if everything is set up
correctly:

```txt
|-agent0/
|-|-rgb/
|-|-|-rgb0.png
|-|-|-rgb1.png
|-|-|-...
|-|-depth/
|-|-|-depth0.npy
|-|-|-depth1.npy
|-|-|-...
|-|-pose/
|-|-|-pose0.npy
|-|-|-pose1.npy
|-|-|-...
|-agent1/
|-|-rgb/
|-|-|-rgb0.png
|-|-|-rgb1.png
|-|-|-...
|-|-depth/
|-|-|-depth0.npy
|-|-|-depth1.npy
|-|-|-...
|-|-pose/
|-|-|-pose0.npy
|-|-|-pose1.npy
|-|-|-...
```

## Creating a 3DSG using ConceptGraphs

Please follow the instructions provided in [our fork](https://github.com/zephirefaith/concept-graphs/tree/partnr)

## Running the Non-privileged Baseline

After spawning your LLM servers use the following command to reproduce baseline as reported in PARTNR paper:

```bash
python -m habitat_llm.examples.planner_demo --config-name baselines/decentralized_zero_shot_react_summary_nn.yaml \
  +habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata/ \
  habitat.dataset.data_path="/path/to/dataset" \
  evaluation.agents.agent_0.planner.plan_config.objects_response_include_states=True \
  evaluation.agents.agent_1.planner.plan_config.objects_response_include_states=True \
  world_model=concept_graph \
  device=cpu \
  agent_asymmetry=True \
  habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor.normalize_depth=False \
  habitat.simulator.agents.agent_1.sim_sensors.head_depth_sensor.normalize_depth=False \
  habitat_conf/task=rearrange_easy_multi_agent_nn \
  num_proc=4 \
  paths.results_dir=/path/to/your/output/directory \
  evaluation.output_dir=/path/to/your/output/directory
```

```
```
