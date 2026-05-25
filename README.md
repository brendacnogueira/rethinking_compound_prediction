# Rethinking Compound Potency Prediction

A framework for machine learning, deep learning, and graph neural network models for **compound potency prediction**, including methods for **imbalanced regression**, **Autofocused Oracle (AFO)** optimization, and **model-based optimization (MBO)** workflows.

---

# Overview

This repository contains scripts and notebooks for:

- Compound potency prediction
- Imbalanced regression analysis
- Molecular representation learning
- SERA-based optimization and evaluation
- Autofocused Oracle (AFO) workflows
- Model-based optimization using CMA-ES

The framework supports both traditional machine learning and deep learning approaches for molecular property prediction.

---

# Repository Structure

```text
.
├── dataset/
├── ccr_results/
├── regression_results/
├── results_plots/
├── ml_models.py
├── oracle_model.py
├── ml_utils.py
├── fingerprint.py
├── machine_learning_models.py
├── sera_opt_proto.py
├── xgboost_sera.py
├── descriptors.py
├── oracle.py
├── mbo.py
├── data_analysis_figures.ipynb
├── conda_env_ml.yml
├── README.md
└── LICENSE
```

---

# Scripts

## Python Scripts

### `ml_models.py`

Implements machine learning and deep learning regression models for compound potency prediction.

Supported models:
- Multiple Regression (MR)
- k-Nearest Neighbors (kNN)
- Support Vector Regression (SVR)
- XGBoost
- Deep Neural Networks (DNN)

Supported losses/metrics:
- MAE
- MSE
- SERA

---

### `oracle_model.py`

Implements machine learning, deep learning, and graph neural network models integrated with the Autofocused Oracle (AFO).

Supported models:
- MR
- kNN
- SVR
- XGBoost
- DNN
---

### `ml_utils.py`

Utility functions supporting:
- Model training
- Evaluation
- Data preprocessing
- Experiment handling

---

### `fingerprint.py`

Computes molecular fingerprints using:
- Morgan fingerprints

---

### `machine_learning_models.py`

Core implementation of ML/DL regression models.

---

### `sera_opt_proto.py`

Implements:
- SERA evaluation metric
- SERA loss functions for PyTorch

---

### `xgboost_sera.py`

Provides SERA derivative calculations for integration with XGBoost optimization.

---

### `descriptors.py`

Computes molecular descriptors.

---

### `oracle.py`

Implements deep learning models for the Autofocused Oracle framework.

---

### `mbo.py`

Implements:
- Model-Based Optimization (MBO)

---

# Jupyter Notebook

### `data_analysis_figures.ipynb`

Provides a complete workflow for:
- Regression result analysis
- Compound potency prediction evaluation
- Figure generation
- Statistical analysis

Generated figures are stored in:

```text
results_plots/
```

---

# Data and Results Folders

| Folder | Description |
|---|---|
| `dataset/` | Compound potency datasets |
| `ccr_results/` | CCR algorithm outputs |
| `regression_results/` | Model prediction outputs |
| `results_plots/` | Generated analysis figures |

---

# Installation

## Create the Conda Environment

Open an Anaconda terminal and run:

```bash
conda env create -n ENVNAME --file conda_env_ml.yml
```

Replace:
- `ENVNAME` with your preferred environment name

---

## Activate the Environment

```bash
conda activate ENVNAME
```

---

# Environment Export

To export the environment:

```bash
conda env export -n ENVNAME > ENV.yml
```

---

# Required Modifications

The following file in DeepChem must be modified:

```text
ENVNAME/lib/python3.9/site-packages/deepchem/models/torch_models/gcn.py
```

---

# CUDA Requirements

Required CUDA modules:

```text
cudnn == 8.0.4
cuda  == 11.6
```

---

# Package Requirements

```text
python=3.9.18
scipy=1.8.1
numpy=1.22.4
scikit-learn=1.1.1
tensorflow=2.9.1
keras=2.9.0
rdkit=2022.3.3
cudatoolkit=11.2.2
dgl-cuda11.1=0.8.1
deepchem=2.6.1
tqdm=4.64.0
torch=2.2.0
IRonPy=0.3.88
xgboost=2.0.3
skorch=0.15.0
```

---

# Execution Workflow

## Step 1 — Generate Model Predictions

Run:

```text
ml_models.py
oracle_model.py
ml_utils.py
```

These scripts generate prediction results stored in:

```text
regression_results/
```

---

## Step 2 — Generate Analysis Figures

Run:

```text
data_analysis_figures.ipynb
```

Generated figures are saved to:

```text
results_plots/
```



---

# License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
