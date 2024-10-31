# if $1 is not sent ask for model size
if [ -z "$1" ]; then
    echo -e "Please enter model, options: \n - Llama1 Models: [7B_llama1, 13B_llama1, 30B_llama1, 70B_llama1] \n - Llama2 Models: [7B_llama2, 13B_llama2, 70B_llama2] \n - Llama2 Chat Models [7B_llama2_chat, 13B_llama2_chat, 70B_llama2_chat] "
    read model_size
else
    model_size=$1
fi

#  ls /checkpoint/komeili/llama2_xlformer /checkpoint/komeili/llama2_xlformer
# llama-2-13b  llama-2-13b-chat  llama-2-70b  llama-2-70b-chat  llama-2-7b  llama-2-7b-chat
# Do not use chat model for habitat-llm task

if [ "$model_size" = "7B_llama1" ]; then
    model_path=/large_experiments/fair_llm/genesis/consolidated_ckpts/7B_1T_consolidated_fp16_mp1/
elif [ "$model_size" = "13B_llama1" ]; then
    model_path=/large_experiments/fair_llm/genesis/consolidated_ckpts/13B_1T_consolidated_fp16_mp2/
elif [ "$model_size" = "30B_llama1" ]; then
    model_path=/large_experiments/fair_llm/genesis/consolidated_ckpts/30B_1.4T_consolidated_fp16_mp4/
elif [ "$model_size" = "70B_llama1" ]; then
    model_path=/large_experiments/fair_llm/genesis/consolidated_ckpts/70B_1.4T_consolidated_fp16_mp8/
elif [ "$model_size" = "7B_llama2" ]; then
    model_path=/checkpoint/komeili/llama2_xlformer/llama-2-7b/
elif [ "$model_size" = "13B_llama2" ]; then
    model_path=/checkpoint/komeili/llama2_xlformer/llama-2-13b/
elif [ "$model_size" = "70B_llama2" ]; then
    model_path=/checkpoint/komeili/llama2_xlformer/llama-2-70b/
elif [ "$model_size" = "7B_llama2_chat" ]; then
    model_path=/checkpoint/komeili/llama2_xlformer/llama-2-7b-chat/
elif [ "$model_size" = "13B_llama2_chat" ]; then
    model_path=/checkpoint/komeili/llama2_xlformer/llama-2-13b-chat/
elif [ "$model_size" = "70B_llama2_chat" ]; then
    model_path=/checkpoint/komeili/llama2_xlformer/llama-2-70b-chat/
else
    echo "Invalid model size, options: \n - Llama1 Models: [7B_llama1, 13B_llama1, 30B_llama1, 70B_llama1] \n - Llama2 Models: [7B_llama2, 13B_llama2, 70B_llama2] \n - Llama2 Chat Models [7B_llama2_chat, 13B_llama2_chat, 70B_llama2_chat] "
    exit 1
fi

if [[ $model_size == *"70B"* ]]; then
    gpus=8
    mem=512g
elif [[ $model_size == *"30B"* ]]; then
    gpus=4
    mem=256g
elif [[ $model_size == *"13B"* ]]; then
    gpus=2
    mem=128g
elif [[ $model_size == *"7B"* ]]; then
    gpus=1
    mem=64g
else
    echo "Invalid model size, sizes: [7B, 13B, 30B, 70B]"
    exit 1
fi

# Expected miniconda3 or anaconda3 path
CONDA_PATH=~/miniconda3

# If the conda path does not exist, prompt the user to enter it
while [ ! -d "$CONDA_PATH" ]; do
    read -p "Could not find $CONDA_PATH. Please enter your conda path (miniconda3 or anaconda3) or modify the default conda path in init_llama_remote.sh file: " CONDA_PATH
done

port=9999

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

srun --partition=devlab --nodes=1 --mem=$mem --ntasks=1 --cpus-per-task=10 --gres=gpu:$gpus --time 2:00:00 --constraint=volta32gb,ib4 --pty /bin/bash - << EOF

source $CONDA_PATH/etc/profile.d/conda.sh
echo "Running $model_size on [$gpus:GPUs | $mem:memory] - [Host:$HOSTNAME Port:$port]"

conda activate /private/home/par/miniconda3/envs/shared_rlm
export XLFORMERS_PATH=/checkpoint/par/shared_rlm_env/xlformers

cd "${DIR}/../../third_party/rlm/src"
python run_xlformers.py --model-parallel $gpus --port $port --host 0.0.0.0 $model_path

echo "Done"
exit
EOF
