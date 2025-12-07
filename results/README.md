# Reference Outputs

This directory contains reference outputs from executing `demo_notebooks/pretrained/test_pre_ae_gmm.ipynb` with the toy dataset.

## Purpose

These files serve as **verification checkpoints**:
- Compare your notebook execution outputs with these reference results
- Verify that the pipeline is working correctly
- Ensure reproducibility of the paper's results

## Contents

#### Classification Reports
- `classification_report_stage1_ae.png` - Stage 1 (AE only) performance metrics
- `classification_report_stage2_gmm.png` - Stage 2 (AE+GMM) performance metrics

#### Confusion Matrices
- `confusion_matrix_stage1_ae.png` - Visual confusion matrix for AE-based predictions
- `confusion_matrix_stage2_aegmm.png` - Visual confusion matrix for AE+GMM predictions

#### Score Distributions
- `ae_mae_distribution.png` - Histogram of Mean Absolute Error scores with threshold
- `aegmm_score_distribution.png` - Histogram of GMM probability scores with threshold

#### SHAP Explanations
- `shap_waterfall_ae_sample.png` - SHAP explanation for AE reconstruction error
- `shap_waterfall_aegmm_sample.png` - SHAP explanation for GMM anomaly score

## Usage for Verification

When you run `demo_notebooks/pretrained/test_pre_ae_gmm.ipynb`:

1. **Visual Comparison**: Compare your output plots with the saved PNG files
2. **Metrics Comparison**: Compare classification reports (accuracy, precision, recall, F1)
3. **Threshold Verification**: Ensure MAE and GMM thresholds match (MAE≈0.135, GMM≈2.74)


## Notes
- Results may vary slightly depending on:
  - Toy dataset random sampling
  - TensorFlow/NumPy random number generator
  - CPU vs GPU execution

