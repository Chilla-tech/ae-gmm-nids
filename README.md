# AE-GMM: A Hybrid, Interpretable Approach for Robust Network Intrusion Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation and artifacts for our research on **AE-GMM: A Hybrid, Interpretable Approach for Robust Network Intrusion Detection**.

## Overview

This work presents a novel two-stage hybrid approach for network intrusion detection that combines:
- **Stage 1**: Autoencoder (AE) learns normal traffic patterns; anomalies produce higher reconstruction errors
- **Stage 2**: Gaussian Mixture Model (GMM) models the distribution of reconstruction error vectors for probabilistic anomaly scoring
- **Explainability**: SHAP (SHapley Additive exPlanations) integration for feature-level attribution

## Key Results

| Metric | Stage 1 (AE Only) | Stage 2 (AE+GMM) |
|--------|-------------------|------------------|
| **F1-Score ** | - | **99.1%** |
| **MAE Threshold** | 0.135069 | - |
| **GMM Threshold** | - | 2.741976 |
| **Features** | 17 (selected from 80+) | 17 |
| **Training Samples** | 286,000 | 286,000 |

## Repository Structure

```
ae_gmm_nids/
├── README.md                          # This file                         # MIT License
├── CITATION.md                        # citations
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Dataset directory
│   ├── README.md                      # Dataset instructions
│   └── toy_dataset.csv                # Small dataset for verification (2-3K samples)
│
├── models/                            # Model definitions
│   ├── ae.py                          # Autoencoder implementation
│   ├── gmm.py                         # GMM implementation
│   └── ae_gmm_hybrid.py               # Hybrid pipeline
│
├── training/                          # Training scripts
│   ├── train_ae.py                    # AE training logic
│   └── full_train.py                  # Full pipeline training (main entry)
│
├── inference/                         # Inference scripts
│   ├── calculate_thres.py             # Threshold computation
│   ├── load_models_n_explainers.py    # Model loading utilities
│   └── predict_n_explain.py           # Prediction with SHAP explanations
│
├── utils/                             # Utility functions
│   ├── prepro.py                      # Data preprocessing
│   ├── evaluation.py                  # Evaluation metrics
│   ├── visual.py                      # Visualization functions
│   └── shap_aegmm_wrappers.py         # SHAP wrapper classes
│
├── scripts/                           # Helper scripts
│   └── create_toy_dataset.py          # Generate toy dataset from full data
│
├── pretrained/                        # Pretrained models (paper baseline)
│   ├── README.md                      # Model documentation
│   └── complete_package_20250914_065942/
│       ├── aegmm_model_package.joblib
│       ├── ae_shap_explainer.joblib
│       └── gmm_shap_explainer.joblib
│
├── aegmm_nids(full_train)/            # Output directory for training runs
│   └── README.md                      # Directory purpose
│
├── results/                           # Reference outputs for verification
│   ├── README.md                      # Output descriptions
│   ├── confusion_matrix_ae.png
│   ├── confusion_matrix_gmm.png
│   └── classification_reports.txt
│
└── demo_notebooks/                         # Demonstration notebooks
    ├── Demo.ipynb                     # Full training pipeline demo
    ├── AEGMM_Demo.ipynb               # Demo for newly trained models
    └── pretrained/
        └── test_pre_ae_gmm.ipynb      # Pretrained model demo (primary)
```

## Installation

### Requirements
- Python 3.8 or higher
- numpy>=1.24
- pandas>=1.5
- scikit-learn>=1.3
- tensorflow>=2.12
- keras>=2.12
- matplotlib>=3.7
- seaborn>=0.12
- shap>=0.44
- joblib>=1.3
- tqdm>=4.66
- scikit-learn>=1.6.1
- statsmodels>=0.14.0

### Setup

1. Clone this repository:
```bash
git clone https://github.com/Chilla-tech/ae-gmm-nids.git
cd ae_gmm_nids
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: For GPU support with TensorFlow, install the GPU version:
```bash
pip install tensorflow[and-cuda]>=2.12
```

## Quick Verification

We provide two paths for verification:

### Path A: Quick Verification 

Uses the pretrained model and toy dataset included in this repository.

1. Install dependencies (see Installation above)

2. Run the pretrained model demonstration notebook:
```bash
jupyter notebook demo_notebooks/pretrained/test_pre_ae_gmm.ipynb
```

3. Compare outputs with reference results in `results/` folder

**Expected outputs**:
- Stage 1 (AE) classification report
- Stage 2 (AE+GMM) classification report
- Confusion matrices for both stages
- SHAP waterfall plots explaining predictions
- MAE and GMM score distribution histograms

### Path B: Full Reproduction 

Reproduces the complete training pipeline from scratch.

1. Download the full CSE-CIC-IDS2018 dataset (see `data/README.md`)

2. Run the full training pipeline:
```bash
python training/full_train.py --data data/raw/CSECICIDS2018_improved.csv --top_n 23 --corr_thr 0.9 --total 286000
```

3. Trained models will be saved to `aegmm_nids(full_train)/AEGMM_hybrid_<timestamp>/`

4. Use `demo_notebooks/AEGMM_Demo.ipynb` to test the newly trained model

**Training parameters**:
- `--data`: Path to the dataset CSV
- `--top_n 23`: Select top 23 features via Random Forest importance
- `--corr_thr 0.9`: Remove features with correlation > 0.9
- `--total 286000`: Create balanced dataset with 286k samples (68.3% BENIGN, 31.7% attacks)

## Usage

### Quick Inference with Pretrained Model

```python
from inference.load_models_n_explainers import load_complete_package
from inference.predict_n_explain import predict_and_visualize_single_flow

# Load pretrained model
model_dir = "pretrained/complete_package_20250914_065942"
package = load_complete_package(model_dir)

# Make prediction with explanation
sample_flow = ...  # Your network flow features
predict_and_visualize_single_sample(package, sample_flow, actual_label)
```

### Training a New Model

See `training/full_train.py` for the complete training pipeline or follow `demo_notebooks/Demo.ipynb` for a step-by-step walkthrough.

## Dataset

This work uses the **CSE-CIC-IDS2018-Improved** dataset, an improved version of the CIC-IDS2018 dataset.

- **Download**: See `data/README.md` for instructions
- **Attack Types**: 14 types (DDoS variants, DoS, Infiltration, SSH/FTP Brute Force, Botnet, Web attacks)
- **Preprocessing**: Feature selection reduces ~90 features to 17 via Random Forest + correlation pruning

A toy dataset (`data/toy_dataset.csv`) with 5K samples is included for quick verification.

## Model Architecture

### Autoencoder
- **Encoder**: 90 → 70 → 30 → 17 → 16 (latent space)
- **Decoder**: 16 → 17 → 30 → 70 → 90 → input_dim
- **Regularization**: L1 regularization for sparsity
- **Optimizer**: Adam with MAE loss
- **Training**: Only on BENIGN samples (anomaly detection paradigm)

### Gaussian Mixture Model
- **Components**: 21 components
- **Covariance**: Full covariance matrix
- **Input**: Reconstruction error vectors from AE
- **Output**: Log-probability scores (lower = more anomalous)

### Thresholds
- **MAE Threshold**: 98th percentile of validation BENIGN reconstruction errors
- **GMM Threshold**: 1.2th percentile of validation GMM scores

## Explainability

SHAP (SHapley Additive exPlanations) integration provides:
- Feature-level attribution for both AE and GMM stages
- Waterfall plots showing feature contributions

See `demo_notebooks/pretrained/test_pre_ae_gmm.ipynb` for examples.

## Citing the Dataset

```bibtex
@inproceedings{liu2022error,
title={Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018},
author={Liu, Lisa and Engelen, Gints and Lynar, Timothy and Essam, Daryl and Joosen, Wouter},
booktitle={2022 IEEE Conference on Communications and Network Security (CNS)},
pages={254--262},
year={2022},
organization={IEEE}
}
```

