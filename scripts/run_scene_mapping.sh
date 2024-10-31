HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.scene_mapping \
  --config-name examples/single_agent_scene_mapping.yaml \
  trajectory.save=True \
  trajectory.agent_names="['main_agent']" \
  llm@evaluation.planner.plan_config.llm=openai_chat \
  habitat.dataset.scenes_dir="data/datasets/hssd" \
  habitat.dataset.data_path="data/datasets/all_scenes/all_scenes.json.gz" \
  habitat.simulator.agents.main_agent.sim_sensors.jaw_depth_sensor.normalize_depth=False \
  evaluation.planner.plan_config.replanning_threshold=50
