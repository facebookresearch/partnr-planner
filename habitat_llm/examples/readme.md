# Included examples

Documentation for each example provided with this repository.

## Scene Mapping to get RGBD Trajectories for Val and Test scenes

`scene_mapping.py` script provides a way to run a single-agent episode where Spot agent explores one episode of each scene included in our val scenes. All the configurations are managed via `conf/trajectory/trajectory_logger.yaml`. Following snippet explains this config:

```yaml
save: False #switch this to True
agent_names: ['agent_0']  # list of all agents to log (note: this is the name of agent in multi-agent setting; agent is named "main_agent" in single-agent setting used in scene-mapping)
camera_prefixes: ['articulated_agent_jaw']
save_path: 'data/trajectories/'  # this is where your trajectories will be stored as "./data/trajectories/<main_agent_or_agent_0>/<rgb/depth/panoptic/pose>"
# root of dir where to log data; is considered the
# prefix when saving trajectories for multiple scenes
save_options: ["rgb", "depth", "panoptic", "pose"]  # modalities to log during execution
```

```
```

CMD: `python -m habitat_llm.examples.scene_mapping`
Since we want to exhaustively explore each scene for coverage, and not efficiency, change the following parameters within `habitat_llm/conf/tools/motor_skills/oracle_explore.yaml`:

```
@@ -13,7 +13,7 @@ oracle_explore:
     description                : 'Search a specific room by visiting various receptacles or furnitures in that room. The input to the skill is the exact name of the room to be visited. Make sure to use FindRoomTool before to know the exact name of the room.'
     # This threshold indicates how many steps the exploration skill can be called for
     # The threshold for encompassed nav skill is overwritten by this number.
-    max_skill_steps            : 2400
+    max_skill_steps            : 7200
     force_end_on_timeout       : True

     sim_freq                   : 120 # Hz
@@ -24,7 +24,7 @@ oracle_explore:
     nav_skill_config:
       name                       : 'Navigate'
       description                : 'Used for navigating to an entity. You must provide the name of the entity you want to navigate to.'
-      max_skill_steps            : 1200
+      max_skill_steps            : 2400
       force_end_on_timeout       : True

       dist_thresh                : 0.2
```
## Skill Runner:
Provides a headless commandline interface (CLI) for running custom sequences of oracle skills within a sandbox environment for a given episode. See the primary [README.md](../../README.md) file for the repo for details.

## Validating and Fixing Episode Placements:
`fix_episode_placements.py` script provides a tool for validating that Object->Receptacle relationships defined in the `CollaborationEpisode`.

For example to validate episode 0 of the PARTNR HSSD `val_mini` dataset:
```bash
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.fix_episode_placements hydra.run.dir="." +validator_episode_indices=[0] +validator_operations=['ep_obj_rec_inits'] +validator_correction_level=0 habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz
```

- CLI option `+validator_episode_index=` can be provided instead of `+validator_episode_id=`, in which case the integer index will be used to query the episode from the dataset.

- Validator operation `'ep_obj_rec_inits'` specifies that the Object->Receptacle relationships should be tested. Currently this is the only operation provided. In the future, more validator operations may be added.

- CLI option `+validator_correction_level=<int>` specifies which corrections to apply if any. The value of `<int>` determines the correction level:
  - `0`: Validate only, no corrections.
  - `1`: In-place corrections only (no re-association to new Receptacles).
  - `2`: Allow re-associations, but to the original Furniture only.
  - `3`: Allow re-association to a new Furniture if the placement position would reasonably match it.

- You can also configure the output path with `'+validator_output_path="<desired output directory>"'` and optionally pause the program for visual debugging with `'+validator_show_and_wait=true'` when an unrecoverable invalid state is detected.

- The dataset target can be selected with `'habitat.dataset.data_path="<path to dataset .json.gz>"'` and any modified dataset will be saved to `<output_path>/new_dataset.json.gz`.
