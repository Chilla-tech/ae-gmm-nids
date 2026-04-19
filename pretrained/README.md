# Pretrained Models

This directory contains the pretrained AE-GMM model used as the paper baseline.

## Package: complete_package_20250914_065942

```
complete_package_20250914_065942/
├── aegmm_model_package.joblib      # Complete model (autoencoder, GMM, scaler, thresholds)
├── ae_shap_explainer.joblib        # Pre-trained SHAP explainer for AE stage
├── gmm_shap_explainer.joblib       # Pre-trained SHAP explainer for GMM stage
├── model_components.joblib         # Components for SHAP recreation
└── individual_components/
    ├── autoencoder.joblib
    ├── gmm_model.joblib
    ├── label_encoder.joblib
    ├── scaler.joblib
    └── thresholds.joblib
```

## Model Specifications

- **Autoencoder**: 17 input features → 16 latent space → 17 reconstruction; L1 regularization; MAE loss; trained on BENIGN samples only
- **GMM**: 21 components, full covariance; input is reconstruction error vectors
- **MAE Threshold**: 0.135069 (98th percentile of validation BENIGN errors)
- **GMM Threshold**: 2.741976 (1.2th percentile of validation GMM scores)

## Usage

See `demo_notebooks/pretrained/test_pre_ae_gmm.ipynb` for a complete working example.
