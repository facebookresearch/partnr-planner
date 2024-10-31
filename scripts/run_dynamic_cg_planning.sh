HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.planner_demo \
 --config-name examples/planner_multi_agent_demo_config.yaml \
 evaluation='habitat_centralized_planner_multi_agent_nn_skills' \
 device=cpu \
 mode='cli' \
 habitat.dataset.scenes_dir="data/datasets/hssd" \
 habitat.dataset.data_path="data/datasets/collaboration/hssd/2024_02_26_eps_5.pickle" \
 instruction="send agent_0 to the bed and then to the dresser" \
 llm@evaluation.planner.plan_config.llm=openai_chat \
 trajectory.save=False \
 habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor.normalize_depth=False \
 evaluation.save_video=True \
 world_model=concept_graph
