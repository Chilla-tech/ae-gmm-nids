# Training Output Directory

This directory is designated for storing models trained via the `training/full_train.py` script.

## Purpose

When you run the full training pipeline, trained models will be automatically saved here with timestamps:

```
aegmm_nids(full_train)/
└── AEGMM_hybrid_<timestamp>/
    ├── aegmm_model_package.joblib
    ├── ae_shap_explainer.joblib
    └── gmm_shap_explainer.joblib
```

## Usage

### Training a New Model

```bash
python training/full_train.py \
    --data data/raw/CSECICIDS2018_improved.csv \
    --top_n 23 \
    --corr_thr 0.9 \
    --total 286000
```

This will create a new directory like `AEGMM_hybrid_20251206_143022/` with all model artifacts.

### Testing Newly Trained Models

After training, use `demo_notebooks/AEGMM_Demo.ipynb` to test your newly trained model:

1. Open `demo_notebooks/AEGMM_Demo.ipynb`
2. Update the model path to point to your trained model
3. Run the notebook to evaluate performance

## Directory Structure

```
aegmm_nids(full_train)/
├── README.md                           # This file
└── AEGMM_hybrid_<timestamp>/           # Your trained models (created during training)
    ├── aegmm_model_package.joblib      # Complete model package
    ├── ae_shap_explainer.joblib        # AE SHAP explainer
    └── gmm_shap_explainer.joblib       # GMM SHAP explainer
```

## Pretrained Models

For the **paper baseline model**, see the `pretrained/` directory instead. That model is used for paper results.

## Training Parameters

The `full_train.py` script accepts the following parameters:

- `--data`: Path to dataset CSV file
- `--top_n`: Number of top features to select (default: 23)
- `--corr_thr`: Correlation threshold for feature pruning (default: 0.9)
- `--total`: Total number of samples in balanced dataset (default: 286000)

## Notes

- Training only uses BENIGN samples for the autoencoder (anomaly detection paradigm)
- GMM is trained on reconstruction errors from validation data
- Thresholds are computed automatically during training:
  - MAE threshold: 98th percentile of validation BENIGN errors
  - GMM threshold: 1.2th percentile of validation GMM scores

---

**For full training instructions**, see the main README.md or `training/full_train.py` documentation.
