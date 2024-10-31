# Habitat-LLM Installation Instructions

### Requirements:
- Conda or Mamba

### Create and activate Conda environment
```bash
conda create -n habitat-llm  python=3.9.2 cmake=3.14.0 -y
conda activate habitat-llm
```

### Initialize third party submodules
```bash
git submodule sync
git submodule update --init --recursive
```

### Install dependencies and requirements
```bash
# Adjust the cuda version depending on your hardware stack
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
# Install habitat-sim version 0.3.2
conda install habitat-sim=0.3.2 withbullet headless -c conda-forge -c aihabitat -y
# NOTE: If the above fails, packages may not be available for your system. Install from source (see https://github.com/facebookresearch/habitat-sim).
pip install -e ./third_party/habitat-lab/habitat-lab
pip install -e ./third_party/habitat-lab/habitat-baselines
pip install -e ./third_party/transformers-CFG
pip install -r requirements.txt
```
If you have issues with library linking make sure that the conda libraries are in your LD_LIBRARY_PATH (e.g `export LD_LIBRARY_PATH=/path/to/anaconda/envs/myenv/lib:$LD_LIBRARY_PATH`)

### Download datasets
```bash
# You may have to re-run downloader commands in case of network errors.
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets hab_spot_arm hab3-episodes habitat_humanoids --data-path data/ --no-replace --no-prune

# Download ovmm objects
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive
```

### Setup HSSD scene dataset
```bash
# Download and link the data.
git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
cd data/versioned_data/hssd-hab
git lfs pull
cd ../../..
ln -s versioned_data/hssd-hab data/hssd-hab
```

### Download task datasets and neural network skill checkpoints

```bash
# Download the data
git clone https://huggingface.co/datasets/ai-habitat/partnr_episodes data/versioned_data/partnr_episodes
cd data/versioned_data/partnr_episodes
git lfs pull
cd ../../..

# Link task datasets
mkdir -p data/datasets
ln -s ../versioned_data/partnr_episodes data/datasets/partnr_episodes

# Link skill checkpoints
ln -s versioned_data/partnr_episodes/checkpoints data/models
```

### (Optional) Install pybullet for IK based controllers
```bash
pip install pybullet==3.0.4
```

### Install pre-commit
```bash
pip install pre-commit && pre-commit install
```

### Install the habitat-llm library
```bash
pip install -e .
```

### Setup api keys if needed:
```bash
# Add the following to your ~/.bashrc file
export OPENAI_API_KEY=...
```

### Run the tests
```bash
# make sure to use bash shell if on zsh
bash

# Download and link the data.
git clone https://huggingface.co/datasets/ai-habitat/hssd-partnr-ci data/versioned_data/hssd-partnr-ci
ln -s versioned_data/hssd-partnr-ci data/hssd-partnr-ci
cd data/hssd-partnr-ci
git lfs pull
cd ../..

# link RAG testing data
ln -s versioned_data/partnr_episodes/test_rag data/test_rag

# then, run the tests
python -m pytest habitat_llm/tests [-v]
```
