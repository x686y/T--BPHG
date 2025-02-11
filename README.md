# Tαβ-BPHG
Tαβ-BPHG is a TCRαβ-peptide binding prediction tool based on a dual-branch heterogeneous graph neural network. By explicitly modeling the interactions between TCR dual chains and peptides, and integrating multi-level neighborhood information and similarity features, it effectively captures the topological relationships between TCR dual chains and peptides, thereby significantly enhancing the predictive performance of Tαβ-BPHG. Additionally, it can quantitatively assess the respective contributions of the TCRα and β chains in peptide binding prediction, providing deeper insights into their cooperative mechanisms.
# Environmental requirements
The pre-training module of Tαβ-BPHG is implemented based on Keras 2.6.0 (https://keras.io) with a TensorFlow backend in a Python 3.7.0 environment. The heterogeneous GNN module and MLP classifier are implemented based on PyTorch 1.9.1 (https://pytorch.org) and Python 3.7.11. The modeling and processing of graph-structured data are handled using PyTorch Geometric (PyG), a graph neural network library built on PyTorch. The computation of model evaluation metrics uses the TorchMetrics library. 
## Pre-training Module

numpy==1.19.5
keras==2.6.0
pandas==1.1.5
scikit-learn==1.0.2
matplotlib==3.4.3
## Dual-Branch Heterogeneous Graph Module
numpy ==1.17.4
pandas==1.3.5
scikit-learn==1.0.2
 matplotlib==3.4.3
# Code
The source code for feature extraction, model construction and training, as well as prediction, is stored in the 'code' folder.
HeteroTCRAB_Modelnew4.py: Defines the baseline model for ablation experiments.  
Base_dataprogressNettcrfull.py: Generates graph-structured data for the baseline model in ablation experiments across different datasets.  
Basecnn_AB.py: Pre-training script for the baseline model in ablation experiments.  
run_heteronetcrfull.py: Execution script for the heterogeneous graph model of the baseline model in ablation experiments.  
test_HeteroBaseline.py: Generates prediction result files for the baseline model in ablation experiments.  
CNNnetfullAB.py: Runs the pre-training module of Tαβ-HGHP.  
HeteroTCRAB_Modelnew4.py: Defines the heterogeneous graph model of Tαβ-HGHP.  
run_heteronetcrfull.py: Executes the heterogeneous graph module of Tαβ-HGHP.  
dataprogressNettcrfull.py: Generates graph-structured data for Tαβ-HGHP.  
test_Heteronettcrfull.py: Prediction result files for Tαβ-HGHP across different datasets.  
config.py: Configuration file for the Tαβ-HGHP model.
# Data
nettcr_bal5folds: This folder contains the 5-fold balanced dataset and prediction result files.  
nettcrfull5folds: This folder contains the 5-fold visible epitope dataset and prediction result files.  
nettcr_strict0/1/2/3/4/_5folds: These five folders contain the 5-fold unseen epitope datasets and corresponding prediction result files, divided under five different seeds.
