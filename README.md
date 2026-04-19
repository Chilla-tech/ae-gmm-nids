# AE-GMM Network Intrusion Detection System with Explainability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation and artifacts for our research on **AE-GMM: A Hybrid, Interpretable Approach for Robust Network Intrusion Detection**.

## Overview

A two-stage hybrid approach for network intrusion detection:
- **Stage 1**: Autoencoder (AE) learns normal traffic patterns; anomalies produce higher reconstruction errors
- **Stage 2**: Gaussian Mixture Model (GMM) models the distribution of reconstruction error vectors for probabilistic anomaly scoring
- **Explainability**: SHAP integration for feature-level attribution

## Key Results

| Metric | Stage 1 (AE Only) | Stage 2 (AE+GMM) |
|--------|-------------------|------------------|
| **F1-Score** | - | **99.1%** |
| **MAE Threshold** | 0.135069 | - |
| **GMM Threshold** | - | 2.741976 |
| **Features** | 17 (selected from 80+) | 17 |
| **Training Samples** | 286,000 | 286,000 |

## Repository Structure

```
ae_gmm_nids/
├── README.md
├── LICENSE
├── CITATION.md
├── requirements.txt
├── data/
│   ├── README.md              # Dataset download instructions
│   └── toy_dataset.csv        # 5K samples for quick verification
├── models/
│   ├── ae.py                  # Autoencoder implementation
│   ├── gmm.py                 # GMM implementation
│   └── ae_gmm_hybrid.py       # Hybrid pipeline
├── training/
│   ├── train_ae.py            # AE training logic
│   └── full_train.py          # Full pipeline training
├── inference/
│   ├── calculate_thres.py     # Threshold computation
│   ├── load_models_n_explainers.py
│   └── predict_n_explain.py   # Prediction with SHAP explanations
├── utils/
│   ├── prepro.py              # Data preprocessing
│   ├── evaluation.py          # Evaluation metrics
│   ├── visual.py              # Visualization
│   └── shap_aegmm_wrappers.py # SHAP wrapper classes
├── pretrained/                # Pretrained models (paper baseline)
│   └── complete_package_20250914_065942/
├── results/                   # Reference outputs for verification
└── demo_notebooks/
    └── pretrained/
        └── test_pre_ae_gmm.ipynb   # Primary verification notebook
```

## Installation

### Requirements
- Python 3.8+
- 8GB RAM minimum (for inference)

### Setup

1. Download this repository:

   **During review**: Download the ZIP from https://anonymous.4open.science/r/ae-gmm-nids-BD18 and extract it.


2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Verification (Path A)

Uses the pretrained model and toy dataset included in this repository. No full dataset download required.

1. Install dependencies (see above)

2. Run the pretrained model demonstration notebook:
   ```bash
   jupyter notebook demo_notebooks/pretrained/test_pre_ae_gmm.ipynb
   ```

3. Compare outputs with reference results in `results/`

**Expected outputs**:
- Stage 1 (AE) and Stage 2 (AE+GMM) classification reports
- Confusion matrices for both stages
- MAE and GMM score distribution plots
- SHAP waterfall plots explaining sample predictions

## Full Reproduction (Path B)

Reproduces the complete training pipeline from scratch.

1. Download the full CSE-CIC-IDS2018 dataset (see `data/README.md`)

2. Run the full training pipeline:
   ```bash
   python training/full_train.py --data data/raw/CSECICIDS2018_improved.csv --top_n 23 --corr_thr 0.9 --total 286000
   ```

## Dataset

Uses the **CSE-CIC-IDS2018-Improved** dataset. See `data/README.md` for download instructions.

A toy dataset (`data/toy_dataset.csv`, 5K samples) is included for quick verification.

## License

MIT License — see [LICENSE](LICENSE) for details.
