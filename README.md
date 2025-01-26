#Rethinking Compound Potency Prediction

--------------------- SCRIPTS: GENERAL INFORMATION ---------------------

This folder contains scripts to build and analyse ML models. 

Should be downloaded and all scripts can be used inside the folder.

The content of the folders is summarized in the following.

Python scripts (.py):

(1) ml_models: script with machine and deep learning models (MR, kNN, SVR, XGBoost, DNN), with diferents metric (MAE,MSE,SERA),  able to predict compound potency.

(2) gcn_models: script with graph neural networks models (GCN), with diferents metric (MAE,MSE,SERA), to predict compound potency.

(3) oracle_model: script with machine, deep learning models  and graph neural networks models (MR, kNN, SVR, XGBoost, DNN, GCN), with Autofocused Oracle (AFO), able to predict compound potency.

(4) ml_utils: script that provide supporting functions for ML/DL models generation

(5) fingerprint: script to calculate molecular fingerprints (Morgan fingerprints)

(6) machine_learning_models: script to build ML/DL models for regression (MR, kNN, SVR, XGBoost, DNN)

(7) sera_opt_proto: script providing SERA functions for evaluations and loss for torch.

(8) xgboost_sera: script to calculate derivatives of the SERA loss function for XGBoost.

(9) gcn: scrip that replaces the standard file from deepchem library for GCN predictions with L1Loss, L2Loss, and SERA loss (see below). 

(10) descriptors: script to calculate molecular descriptiors.

(11) oracle: script to build deep leraning models for autofocused oracle.

(12) mbo: script to model-based optimization (MBO) and Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for AFO. 


Jupyter Notebooks (.ipynb):

(13) data_analysis_figures: Jupyter notebook with a workflow for the data analysis of compound potency 
predictions from regression models.


(14) Folders:
	
	- dataset: stores the compound potency dataset used in this analysis

	- ccr_results : stores the CCR algorithm results

	- regression_results: stores regression models predictions

	- results_plots: stores plots generated in data_analysis_figures.ipynb.


(15) Python environment:

	- conda_env_ml.yml provides the python environment used for this analysis. (Requires instalation see below)


Order of execution:

1. (1), (2), (3) to generate the model predictions (results). 

2. (13) to generate the figures. 



Python environment installation:

1. Open Anaconda command line

2. Type 'conda env create -n ENVNAME --file ENV.yml', where 'ENVNAME' is the desired environment and 'ENV' the full path to the yml file.


Python environment export:

1. Open Anaconda command line

2. Type 'conda env export ENVNAME>ENV.yml', where 'ENVNAME' is the desired environment and 'ENV' the full path to the yml file.

Changes required:

- Change file gcn.py in the deepchem library on the path: ENVNAME/lib/python3.9/site-packages/deepchem/models/torch_models/gcn.py

Modules:
- cudnn==8.0.4
- cuda==11.6

Requirements:

- python=3.9.18
- scipy=1.8.1
- numpy=1.22.4
- scikit-learn==1.1.1
- tensorflow==2.9.1
- keras==2.9.0
- rdkit==2022.3.3
- cudatoolkit=11.2.2
- dgl-cuda11.1=0.8.1
- deepchem==2.6.1
- tqdm=4.64.0
- torch=2.2.0
- IRonPy==0.3.88
- xgboost==2.0.3
- skorch==0.15.0
- dgl (?)
- pydantic (?)
- dgllife (?)






