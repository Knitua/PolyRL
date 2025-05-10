
# PolyRL: Reinforcement Learning-Guided Polymer Generation for Multi-Objective Gas Membrane Discovery

[![license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Acellera/PolyRL-open/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/downloads/)



---

## Overview

**PolyRL** is an open-source framework for reinforcement learning-based molecular generation, designed to accelerate the discovery of polymeric membranes for CO₂/N₂ gas separation. It integrates multiple generative models and reinforcement learning algorithms to optimize key properties, including permeability, selectivity, and proximity to Robeson’s upper bound, within a unified and extensible pipeline.

The framework supports both pretraining and goal-directed generation, enabling flexible application to a wide range of molecular design tasks.


---

## Key Features


- __**RL Algorithms:**__ 
PolyRL offers task optimization with various reinforcement learning algorithms such as [Proximal Policy Optimization (PPO)][1], [Advantage Actor-Critic (A2C)][2], [Reinforce][3], [Reinvent][4], and [Augmented Hill-Climb (AHC)][5], [Direct Preference Optimization (DPO)][8] and Hill Climbing.

- __**Pre-trained Models:**__ PolyRL contains pre-trained models including Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), GPT-2, LLama2.

- __**Scoring Functions :**__ 



---

## Contents
1. **Installation**
   - 1.1. Conda environment and required dependencies
   - 1.2. Install PolyRL
2. **Generators**
   - 2.1. Dataset
   - 2.2. Run Pretraining Script
   - 2.3. Direct Use of Generators
3. **Scoring Models**
   - 3.1. Dataset
   - 3.2. Run Pretraining Script
   - 3.3. About Scoring Function
4. **Polymer Molecule Generation**
   
5. **SHAP Analysis**
6. **Visualization of Results**



---
## Main Text


## 1. Installation

### 1.1. Create Conda Environment

To ensure consistent and reproducible environments, we provide a Conda environment YAML file.

```bash
conda env create -f PolyRL.yml
conda activate PolyRL
```

This will install:
- Python 3.10
- TensorFlow, RDKit, scikit-learn, SHAP, and other ML packages
- TorchRL and TensorDict from fixed Git commits
- pip-based packages like `wandb`, `hydra-core`, etc.

If using GPU, install the appropriate version of PyTorch with CUDA support. For example, for CUDA 11.8:

```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

See [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) for more CUDA options.

---

### 1.2. Install PolyRL Package

Clone and install the PolyRL package:

```bash
git clone https://github.com/Knitua/PolyRL.git
cd PolyRL
pip install -e ./
```

Use `pip install ./` if you do not plan to modify the source code.

---

### 1.3. Verify Installation

Run the following command to check that installation was successful:

```bash
python -c "import torch; import torchrl; import tensordict; print('Installed successfully.')"
```
<br/>

## 2. Generators

### 2.1. Dataset

We constructed two types of SMILES datasets for pretraining, each targeting polymers with different structural characteristics:

- **datasetB**: Contains approximately 1 million linear polymer SMILES, each with two `*` atoms representing polymerization sites. This dataset is used to train generators for linear polymers.  
  - File path: `PolyRL/Pretrain_models/dataset/datasetB.csv`  
  - Construction method: XXX

- **enhanced_datasetD**: Contains around 60,000 bridged polymer SMILES, each with four `*` atoms. It is used to train generators for more complex topologies, such as dendritic or bridged polymers.  
  - File path: `PolyRL/Pretrain_models/dataset/enhanced_datasetD.csv`  
  - Construction method: XXX

Both datasets are in `.csv` format, with one SMILES per line and no header.

---

### 2.2. Run Pretraining Script

We provide pretraining scripts for different generator architectures, located under `PolyRL/Pretrain_models/`.

Use the following commands to pretrain GPT-2 and LLaMA2 architectures:

```bash
python PolyRL/Pretrain_models/gpt2/gpt2_pretrain.py 
python PolyRL/Pretrain_models/llama2/llama2_pretrain.py
```
For GRU and LSTM architectures, we provide a unified script `pretrain_single_node.py` that allows model selection via the configuration file. Simply set the model field to gru or lstm in config.yaml to start training:

```bash
python pretrain/pretrain_single_node.py
python pretrain/pretrain_distributed.py
```
The difference between the two commands lies in whether distributed training across multiple machines/GPUs is enabled.

---
### 2.3. Direct Use of Generators

All pretrained generators have been integrated into the reinforcement learning framework and are stored under `PolyRL/priors/`.

To use your newly pretrained model, modify the corresponding interface in `PolyRL/models/__init__.py`.

To switch between different types of generators within the RL framework, adjust the relevant parameters in the reinforcement learning YAML configuration file.  
Refer to: `Section 4. Polymer Molecule Generation`.

<br/>


## 3. Scoring Models


### 3.1. Dataset

The data used in this work is derived from the benchmark dataset published by [Yang et al.](https://www.science.org/doi/10.1126/sciadv.abn9545), which supports machine learning applications for polymer membrane design.

- `datasetA_imputed_all.csv` contains SMILES strings and their corresponding gas permeability values.  
- `datasets/datasetAX_fing.csv` contains molecular fingerprints computed from SMILES after grouping and averaging duplicate entries by `.groupby('Smiles').mean()`.


### 3.2. Pretraining Process

The scoring model is formulated as a regression task that maps molecular fingerprints to gas permeability values.  Trained models serve as surrogate property predictors and are used as scoring functions during reinforcement learning.

- **Input features**: Morgan fingerprints  
- **Output labels**: CO₂ and N₂ permeability  
- **Supported models**: Random Forest (RF), Support Vector Machine (SVM)  
- **Evaluation metrics**: R², MAE, MSE, RMSE

To start training, run the following command:

```bash
python Scoring_models/Train.py --model 'RF'
python Scoring_models/Train.py --model 'SVM'
```
To generate the fingerprint matrix for a given dataset (e.g., `datasetX.csv`), run the following command:
```bash
python Scoring_models/Generate_MFF.py --dataset 'datasetX'
```
### 3.3. About Scoring Function


<br/>

## 4. **Polymer Molecule Generation**


PolyRL has multiple RL algorithms available, each in a different directory within the `PolyRL/scripts` directory. 

Each algorithm is associated with a dedicated YAML configuration file (e.g., `config_denovo.yaml`) located in the same directory as the training script. You can modify this file to customize training parameters as well as to select different types of generator architectures.


To initiate training and generate polymer SMILES with different RL strategies, run one of the following commands:

```bash
python scripts/reinforce/reinforce.py --config-name config_denovo
python scripts/a2c/a2c.py --config-name config_denovo
python scripts/ppo/ppo.py --config-name config_denovo
python scripts/reinvent/reinvent.py --config-name config_denovo
python scripts/ahc/ahc.py --config-name config_denovo
python scripts/dpo/dpo.py --config-name config_denovo
python scripts/hill_climb/hill_climb.py --config-name config_denovo
```
The generated molecules and training logs are saved in the `result/` directory under the corresponding script path. For example, when running:
```bash
python scripts/reinvent/reinvent.py --config-name config_denovo
```
The outputs will be saved in:
```bash
scripts/reinvent/result/
```
Each row contains a generated polymer and its predicted properties at a given training step, including the SMILES string, overall reward score, predicted permeability, and selectivity.


For visualization and performance analysis of the generated polymer libraries, including score progression over training iterations, Robeson plots, and functional group contribution analysis, refer to `Section 6. Visualization of Results`.

<br/>

## 5. SHAP Analysis

<br/>

## 6. Visualization of Results




[1]: https://arxiv.org/abs/1707.06347
[2]: https://arxiv.org/abs/1602.01783
[3]: https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
[4]: https://arxiv.org/abs/1704.07555
[5]: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00646-z
[6]: https://arxiv.org/pdf/2206.12411.pdf
[7]: https://arxiv.org/abs/2007.03328
[8]: https://arxiv.org/abs/2305.18290




