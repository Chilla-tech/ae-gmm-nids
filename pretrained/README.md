# Pretrained Models

This directory contains the pretrained AE-GMM model used as the baseline in the paper.

## Model Package: complete_package_20250914_065942

**Created**: September 14, 2025 (Date: 06:59:42)  
**Status**: Paper Baseline Model  
**Purpose**: Primary pretrained model for verification

### Package Contents

```
complete_package_20250914_065942/
├── aegmm_model_package.joblib          # Complete model package (autoencoder, GMM, scaler, etc.)
├── ae_shap_explainer.joblib            # Pre-trained SHAP explainer for AE stage
├── gmm_shap_explainer.joblib           # Pre-trained SHAP explainer for GMM stage
├── model_components.joblib             # Components for SHAP recreation
├── package_summary.txt                 # Model metadata and performance summary
└── individual_components/              # Individual model files (optional)
    ├── autoencoder.joblib
    ├── gmm_model.joblib
    ├── label_encoder.joblib
    ├── scaler.joblib
    └── thresholds.joblib
```

### Model Specifications

**Autoencoder Architecture**:
- Input features: 17 (selected from 80+ via RF + correlation pruning)
- Encoder: 17 → 70 → 30 → 17 → 16 (latent space)
- Decoder: 16 → 17 → 30 → 70 → 17 (reconstruction)
- Activation: ReLU
- Regularization: L1 regularization for sparsity
- Loss: Mean Absolute Error (MAE)
- Training: Benign samples only (anomaly detection paradigm)

**Gaussian Mixture Model**:
- Components: 21
- Covariance type: Full covariance matrix
- Input: Reconstruction error vectors from autoencoder
- Output: Log-probability scores (lower = more anomalous)

**Preprocessing**:
- Scaler: StandardScaler fitted on BENIGN training samples only
- Label Encoder: 2 classes (BENIGN=1, Anormal=0)

### Performance Metrics

**Stage 2 (AE+GMM) - Paper Results**:
- **F1-Score**: 99.1%
- **Precision**: High (see paper for detailed metrics)
- **Recall**: High (see paper for detailed metrics)

**Thresholds**:
- **MAE Threshold**: 0.135069 (98th percentile of validation BENIGN reconstruction errors)
- **GMM Threshold**: 2.741976 (1.2th percentile of validation GMM scores)

### Training Details

**Dataset**: CSE-CIC-IDS2018 (Improved)
- Total samples Selected: 286,000
- Distribution: 68.3% BENIGN, 31.7% attacks (realistic real-world ratio)
- Train/Val/Test split: 70/30, then 80/20 for train/val

**Feature Selection**:
- Random Forest top-23 feature selection
- Correlation pruning (threshold > 0.9)
- Final features: 17

**Attack Types Covered**: 14 types
- DDoS attacks (HOIC, LOIC-HTTP, LOIC-UDP)
- DoS attacks (Hulk, SlowHTTPTest, GoldenEye, Slowloris)
- Brute Force attacks (Web, XSS, SSH-Patator, FTP-Patator)
- SQL Injection
- Infiltration
- Botnet

### Usage

#### Loading the Complete Package

```python
from inference.load_models_n_explainers import load_complete_package

# Load all components
package = load_complete_package("pretrained/complete_package_20250914_065942")

# Access components
autoencoder = package['autoencoder']
gmm_model = package['gmm_model']
scaler = package['scaler']
label_encoder = package['label_encoder']
ae_explainer = package['ae_explainer']
gmm_explainer = package['gmm_explainer']
feature_names = package['feature_names']
thresholds = package['thresholds']
```

#### Making Predictions

```python
from inference.predict_n_explain import predict_single_sample, predict_and_visualize_single_sample

# Predict on a single sample
result = predict_single_flow( package, sample_flow)

print(f"Stage 1 (AE): {result['stage1_prediction']}")
print(f"Stage 2 (GMM): {result['stage2_prediction']}")
print(f"Final: {result['final_class']}")

# With SHAP explanation
predict_and_visualize_single_sample(package, sample_flow, actual_label)
```

#### Quick Demo

See `demo_notebooks/pretrained/test_pre_ae_gmm.ipynb` for a complete working example using this model on the toy dataset.

### Reproducibility

To reproduce this model from scratch:

```bash
python training/full_train.py \
    --data data/raw/CSECICIDS2018_improved.csv \
    --top_n 23 \
    --corr_thr 0.9 \
    --total 286000
```

New models will be saved to `aegmm_nids(full_train)/AEGMM_hybrid_<timestamp>/`

### Notes

- This model is the **official baseline** used in the paper
- SHAP explainers are pre-trained for fast inference
- Model files use joblib serialization for compatibility
- All components are included for full reproducibility
- Feature names are stored to maintain consistency across predictions

### File Sizes

- `aegmm_model_package.joblib`: ~2-3 MB
- `ae_shap_explainer.joblib`: ~1-2 MB
- `gmm_shap_explainer.joblib`: ~1-2 MB
- Total package: ~5-6 MB

---

