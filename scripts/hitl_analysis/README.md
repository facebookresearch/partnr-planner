# Analysis Scripts for Human-In-The-Loop (HITL) Data

## Extract the data

The raw HITL data looks like `2024-10-02-object-states.tar.gz`. Save and extract this to `data/hitl_data/raw`. The content should look like:

```text
data/hitl_data/[collection name]
├── raw
│   ├── [session name]
│   │   ├── [episode ID].json.gz
│   │   ├── ...
│   │   ├── [episode ID].json.gz
│   │   ├── session.json.gz
```

## Preprocess the data

Run:

```bash
python scripts/hitl_analysis/preprocess_data.py \
    --collection-path <path to HITL collection> \
    --recompute <flag: if the data already exists, recompute it anyway>
```

This will yield the following directory structure:

```text
data/hitl_data/[collection name]
├── processed
│   ├── best
│   │   ├── [episode ID].json.gz
│   ├── failed
│   │   ├── [episode ID].json.gz
│   ├── processed_metrics.json
```

Where `processed_metrics.json` consists of the following entries:

```
eid_to_round_needed        HITL collection attempts needed to achieve success for a given episode (-1 for never successful)
eid_to_pc                  percent complete metric (float)
eid_to_success             task success metric (0 or 1)
eid_to_explanation         failure explanation (str)
eid_to_filename            filename of the HITL episode execution data
all_users                  a list of all user IDs that participated
ratio_agent_0              percentage of tasks accomplished by Agent 0
ratio_agent_1              percentage of tasks accomplished by Agent 1
ratio_extraneous_actions   ratio of actions that did not progress the task vs all actions taken
explore_steps              exploration steps
remaining_num              number of steps taken
```

Except for `all_users`, all fields are dictionaries mapping episode ID to the quantity described.

## Compute Metrics

After preprocessing the data, you can aggregate and score the metrics contained in `processed_metrics.json`:

```bash
python scripts/hitl_analysis/compute_scores.py --collection-path <path to HITL collection>
```

## Replay HITL Episode Rollouts

We provide an example for replaying HITL rollouts in `replay_and_evaluate.py`. To run:

```bash
python scripts/hitl_analysis/replay_and_evaluate.py \
    --episodes-path <path to a flat directory containing [eid].json.gz collection files> \
    --dataset-file <path to the episode dataset used in this collection> \
    --ncpus <number of cpus for parallelization> \
    --multi <flag: the collection involved multiple users> \
    --just-evaluate <flag: the episodes have already been processed>
```

This will evaluate percent complete and success metrics for the replayed episodes.

## Generate HITL Videos

To generate videos of HITL rollouts, run:

```bash
python scripts/hitl_analysis/visualize_episode.py \
    --episodes-path <path to a flat directory containing [eid].json.gz collection files> \
    --dataset-file <path to the episode dataset used in this collection> \
    --ncpus <number of cpus for parallelization> \
    --multi <flag: the collection involved multiple users> \
    --multi-plot <flag: plot multiple agents >
```

Videos will be saved to `[episodes-path]/videos/[episode ID].mp4`.
