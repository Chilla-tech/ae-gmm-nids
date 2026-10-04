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
├── README.md                          # This file
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata (used by GitHub/Zenodo)
├── CITATION.md                        # BibTeX for this work, the dataset and SHAP
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
│   └── reproduce_heldout_eval.py      # Regenerate the paper's held-out split and McNemar test
│
├── pretrained/                        # Pretrained models (paper baseline)
│   ├── README.md                      # Model documentation
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

1. Clone this repository:
```bash
git clone https://github.com/Chilla-tech/ae-gmm-nids.git
cd ae-gmm-nids
```

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

3. Trained models will be saved to `aegmm_nids(full_train)/AEGMM_hybrid_<timestamp>/`

**Training parameters**:
- `--data`: Path to the dataset CSV
- `--top_n 23`: Select top 23 features via Random Forest importance
- `--corr_thr 0.9`: Remove features with correlation > 0.9
- `--total 286000`: Subsample 286k flows (about 68.3% BENIGN, 31.7% attacks; intrusion fraction 1/3.15)

### Reproducing the paper's split and held-out evaluation

The preprocessing in `utils/prepro.py` follows the pipeline used for the reported results: no deduplication, flows with labels missing from `ATTACK_MAP` (8,490 `DoS Slowloris` and 39 `Web Attack - SQL` flows) are discarded, the intrusion fraction of the 286,000-flow subsample is 1/3.15, and the Random Forest used for feature selection is fit on a stratified 70% split. With the full CSV, the script below regenerates the exact held-out test set (85,800 flows), verifies that it gives the pretrained model's 17 features, and reproduces the AE and AE-GMM results and McNemar's test:

```bash
python scripts/reproduce_heldout_eval.py --data path/to/CSECIC-IDS2018_subset.csv
```

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
predict_and_visualize_single_flow(package, sample_flow, actual_label)
```

### Training a New Model

See `training/full_train.py` for the complete training pipeline.

## Dataset

Uses the **CSE-CIC-IDS2018-Improved** dataset. See `data/README.md` for download instructions.

A toy dataset (`data/toy_dataset.csv`, 5K samples) is included for quick verification.

## Citation

If you use this code, please cite it via [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button). A Zenodo DOI will be added here after the archived release is published.

## License

MIT License — see [LICENSE](LICENSE) for details.
