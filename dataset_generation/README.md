# Dataset generation

#### Setup

- Symlink your habitat datafolder to this subfolder, e.g. `ln -s /path/to/habitat-data data`
- Activate your conda environment with habitat (0.2.3) installed

#### Generate dataset

Run the following command:

```
python -m dataset_generation.benchmark_generation.generate_instructions
```

This will generate a set of jsons with a natural language instruction and an initialization in the folder specified in `output_path` in `benchmark_gen.yaml`. You will then need to parse and clean this JSON. For this, run:

```
python -m dataset_generation.benchmark_generation.parse_generated_instructions
```

This should save a dataset `output_parsed/` directory with the parsed json files.

Next filter out invalid and repeated instructions from the parsed instructions:

```
python -m dataset_generation.benchmark_generation.filter_instructions
```
This should save a dataset `output_filtered/` directory with the filtered dicts saved in `filtered_dicts.json`.

The last step is to instantiate these instructions, filter invalid samples and create a json.gzip episode dataset file. For this, run from CLI:

```
python -m dataset_generation.benchmark_generation.generate_episodes \
  --init-state-dicts <path_to_generated_inits.json> \
  --gen-config <path_to_generator_config.json> \
  --metadata-dict <path_to_metadata_config.json>
```
where
- `init-state-dicts` is the JSON config containing your parsed generated instructions. Should have a single parent key `"initial_state_dicts"` mapped to a list of per-episode initial state configs dicts, created in the last step.
- `gen-config` is the **optional** JSON config containing asset paths and output directories. A default is used if not provided.
- `metadata-dict` is the **optional** JSON config containing metadata (semantic .csv) paths. A default is used if not provided. This should ideally match the metadata_dict used in instruction generation.

See main function of `generate_episodes.py` for defaults and examples of the above configs. Also see `tests/test_episode_generator.py` for an example use of the scripting API.

#### Free-form generation and guided instruction

You can switch between guided and free-form generation by switching the prompt in `benchmark_gen.yaml`.
Prompts with `object_states` include heterogenous actions like fill, power on, clean.


When using guided instruction generation, switch the set of guiding instructions (or "templates") by changing `template_file` in `benchmark_gen.yaml`.
Specifically, `prompts_benchmark/instruction_prompts` contains the following files:

- 20_val_object_states.json - set of guiding instructions used to generate heterogenous tasks in val scenes.
- 300_rearrange_train_template.json - set of guiding instructions used to generate rearrange tasks in train scenes.
- 300_rearrange_val_template.json - set of guiding instructions used to generate rearrange tasks in val scenes.
- 300_spatial_train_template.json - set of guiding instructions used to generate spatial constraint tasks in train scenes.
- 300_spatial_val_template.json - set of guiding instructions used to generate spatial constraint tasks in val scenes.
- 300_temporal_train_template.json - set of guiding instructions used to generate temporal constraint tasks in train scenes.
- 300_temporal_val_template.json - set of guiding instructions used to generate temporal constraint tasks in val scenes.
- 30_train_object_states.json - set of guiding instructions used to generate heterogenous tasks in train scenes.
- scene_and_json_oneshot.txt - prompt used for free-form generation of heterogenous tasks
- scene_and_json_oneshot_temporal.txt - prompt used for free-form generation of temporal constraint tasks
- scene_and_json_twoshot_spatial.txt - prompt used for free-form generation of spatial constraint tasks
- templated_prompt.txt - prompt used for guided generation of rearrange, spatial and temporal tasks
- templated_prompt_object_states.txt - prompt used for free-form generation of heterogenous tasks

When using guided instruction generation, switch the set of template/guiding templates by changing `template_file` in `benchmark_gen.yaml`.

Change the simuation scenes by changing the `scene_id` in `benchmark_gen.yaml`.

For reference, here is the list of scenes, and scene split used in PARTNR dataset generation.

```
    "val": [
        "102817140",
        "106366386_174226770",
        "106366410_174226806",
        "107734176_176000019",
        "107733960_175999701",
        "103997895_171031182",
        "102344529",
        "102816756",
        "106878915_174887025",
        "104348361_171513414",
        "102815835",
        "104348010_171512832",
        "106878960_174887073"
    ],
    "train": [
        "108736824_177263559",
        "108736872_177263607",
        "108294870_176710551",
        "108736737_177263406",
        "107734449_176000403",
        "108736851_177263586",
        "108736635_177263256",
        "107734479_176000442",
        "106879044_174887172",
        "106878945_174887058",
        "106366353_174226695",
        "106366173_174226431",
        "105515448_173104512",
        "104348463_171513588",
        "104348082_171512994",
        "103997424_171030444",
        "102817200",
        "102344049",
        "102344193",
        "102344403",
        "102344457",
        "102344022",
        "102344250",
        "102816216",
        "102816009",
        "103997460_171030507",
        "104862681_172226874",
        "102344280",
        "108294897_176710602",
        "108294573_176710113",
        "105515211_173104179",
        "104862639_172226823",
        "102815859",
        "104862669_172226853",
        "103997919_171031233",
        "104862621_172226772",
        "104862660_172226844"
    ]
```

## Generate Evaluation Functions

Once the task and episode initializations have been generated (above), the next step is to generate evaluation functions, pack the dataset, and verify its episodes. The workflow is as follows.

1. **Generate plain text evaluation functions:**

```bash
# See `dataset_generation/conf/evaluation_gen.yaml` for additional config overrides.
python -m dataset_generation.benchmark_generation.generate_evaluations \
    eval_gen.path_to_dataset_in=<a directory containing scenes, which each contain dataset.json.gz and scene_info.json> \
    eval_gen.output_path=<a directory>
```

The result of this step will be:

```text
|-- [output_path]/[run_name]/[scene_id]
    |-- logs
    |   |-- episode_[i].log   # detailed generation log files
    |-- metadata
    |   |-- episode_[i].json  # scene info with episode-specific mappings
    |-- plaintext_evals
    |   |-- episode_[i].py    # plaintext evaluation data
    |-- plaintext_evals_orig
        |-- episode_[i].py    # copy of the plaintext eval for before/after comparison
```

2. **[Optional] Manually correct evaluation functions:**

Directly modify: `[output_dir]/[run_name]/plaintext_evals/episode_[i].py`
Using reference: `[output_dir]/[run_name]/metadata/episode_[i].json`

Then, pack the dataset to discover any errors in your annotations:

```bash
# Add your Step 1 overrides to this command.
python -m dataset_generation.benchmark_generation.generate_evaluations \
    eval_gen.generate=false \
    eval_gen.pack=true
```

The packing results will be in `[output_path]/[run_name]/[scene_id]/packing_failures.json`. Repeat annotation and packing until satisfied.

3. **Finalize the dataset (pack, merge, verify):**

```bash
# Add your Step 1 overrides to this command.
python -m dataset_generation.benchmark_generation.generate_evaluations \
    eval_gen.generate=false \
    eval_gen.pack=true \
    eval_gen.merge=true \
    eval_gen.verify=true
```

If your dataset is already packed, you can remove the override `eval_gen.pack=true`.

Output after `eval_gen.pack`:

```text
|-- [output_path]/[run_name]/[scene_id]
    |-- [run_name].json.gz              # the packed scene-specific collaboration dataset
    |-- packing_failures.json           # a log of the episode IDs that could not be packed
```

Output after `eval_gen.merge`:

```text
|-- [output_path]/[run_name]
    |-- [run_name].json.gz              # the merged collaboration dataset
```

Output after `eval_gen.verify`:

```text
|-- [output_path]/[run_name]
    |-- [run_name]_verified.json.gz     # the merged collaboration dataset with just sim-verified episodes
```
