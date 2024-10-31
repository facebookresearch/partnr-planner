# Finetuning a planner
We provide scripts to finetune an LLM as a high level planner, as described in our paper. We provide the instructions to finetune your own model below.

## Preparing a dataset
The first step is to create a dataset for finetuning. For this, run any of the multi-agent baselines, as specified in [here](../../README.md).
After running the baseline, you should have a folder with accuracy metrics for the baseline as well as details of the traces generated. Next, you want to convert those traces into text files to finetune your model on.
For this, you will run the following command:

```bash
python -m habitat_llm.finetuning.build_trace_dataset \
--path {the path where you stored the the detailed traces} \
--output-dir {path where you want the text files stored}

```

## Finetuning the planner
Once you have generated the dataset, you can finetune an LLM to predict agent actions given the context. To do this you can run:

```bash
sh scripts/finetune_planner.sh
# If you want to run it as a slurm batched script, you can run:
# sh script/finetune_planner.sh --batched
```


Make sure you update your dataset path in `finetune_planner.sh` training script. You can also check the [finetuning.yaml](../conf/finetuning/finetuning.yaml) file to change hyperparameters. This will create a log on Weights & Biases to check the model. It will also store the model checkpoints and logs into a folder of the form: `multirun/yyyy-mm-dd/hh-mm-ss/0/`.

## Testing the planner
Once the model is saved you will want to test it. We cover here how we test the model in autoregressive mode and also in the planner loop.

### Autoregressive Testing
Testing the model autoregressively will be useful to understand the performance of your model without adding an environment in the loop. This will make evaluation more efficient and useful for faster iteration.

To test the model autoregressively you can run:

```bash

path_to_checkpoint="the checkpoint you want to evaluate"
eval_dataset="path to eval dataset"
python habitat_llm/finetuning/trainer.py \
    -m \
    evaluate=True \
    training_arguments.batch_size=10 \
    eval_checkpoint_dir=$path_to_checkpoint \
    dataset.val=[old_iter0_heuristic/2024_07_29_val_mini]


```


### Interactive testing

Finally, you may want to test your model in an interactive setting. For this you will first create model checkpoint by merging the LoRa weights:

```bash
base_model_path="Path to the base model"
checkpoint_path="Path to the model you trained"
python -m habitat_llm.finetuning.flatten_checkpoint --model_name $base_model_path --checkpoint_dir $checkpoint_path
```

The model will be stored in the same folder as the checkpoint path. You can run this model together with a human react agent, using:

```bash
OUT_FOLDER="your_output_folder"
DATASET="the dataset where you want to run the experiment"
PATH_HUMAN="path to human model (e.g. Llama70B)"
PATH_FT="the path you just created."

python -m habitat_llm.examples.planner_demo  \
--config-name baselines/decentralized_ft_human_react_summary   \
+habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata/ \
evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=$PATH_FT   \
evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=$PATH_HUMAN  \
habitat.dataset.data_path=$DATASET num_proc=1 paths.results_dir=$OUT_FOLDER

```
