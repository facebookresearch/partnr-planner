 #!/bin/bash
 if [ $1 -eq 0 ] ; then
    HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.planner_demo \
    --config-name examples/planner_multi_agent_demo_config.yaml \
    evaluation='decentralized_evaluation_runner_multi_agent' \
    mode='cli' \
    world_model.partial_obs='True' \
    habitat.dataset.scenes_dir="data/datasets/hssd" \
    habitat.dataset.data_path="data/datasets/collaboration/hssd/2024_03_27_eps300_filtered.pickle" \
    instruction="Go to table_15 and search for apple" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=llama3 \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=llama3 \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=few_shot_decentpo_robot \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=few_shot_decentpo_human \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode="rlm" \
    evaluation.agents.agent_0.planner.plan_config.llm.serverdir=rlm/running_servers/default/ \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode="rlm" \
    evaluation.agents.agent_1.planner.plan_config.llm.serverdir=rlm/running_servers/default/
else
    HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.planner_demo \
    --config-name examples/planner_multi_agent_demo_config.yaml \
    evaluation='decentralized_evaluation_runner_multi_agent' \
    mode='cli' \
    world_model.partial_obs='True' \
    habitat.dataset.scenes_dir="data/datasets/hssd" \
    habitat.dataset.data_path="data/datasets/collaboration/hssd/2024_03_27_eps300_filtered.pickle" \
    instruction="Search the tables for android toy. Once found move it to the shelf" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=openai_chat \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=openai_chat \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=few_shot_decentpo_robot \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=few_shot_decentpo_human
fi

#  instruct@evaluation.planner.plan_config.instruct=few_shot_centralized_partialobs_multiagent \
#  llm@evaluation.planner.plan_config.llm=openai_chat \
#  evaluation.planner.plan_config.replanning_threshold=30
