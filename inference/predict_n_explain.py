# Prediction and explanation functions for AE-GMM model

import shap
import pandas as pd
import numpy as np

# For batch predictions
def batch_predict(pkg, df: pd.DataFrame):
    X = df[pkg['models']['feature_names']]
    Xs = pkg['models']['scaler'].transform(X)
    recon = pkg['models']['autoencoder'].predict(Xs, verbose=0)
    mae = np.mean(np.abs(Xs - recon), axis=1)
    err = np.abs(Xs - recon)
    gmm_score = pkg['models']['gmm_recon'].score_samples(err)
    pred = (gmm_score > pkg['models']['threshold_gmm']).astype(int) # 1: Normal, 0: Anomaly to align with the label encoding which encodes 'BENIGN' as 1
    return pd.DataFrame({
        'mae': mae,
        'gmm_score': gmm_score,
        'anomaly': pred
    })

def _to_dataframe(sample, feature_names):
    """Normalize input to a single-row DataFrame."""
    if isinstance(sample, pd.DataFrame):
        return sample.iloc[[0]] if len(sample) > 1 else sample
    sample_np = np.array(sample)
    if sample_np.ndim == 1:
        sample_np = sample_np.reshape(1, -1)
    return pd.DataFrame(sample_np, columns=feature_names)

# Single sample prediction
def predict_single_flow(package, sample):
    """
    Return minimal prediction info:
    - stage1_mae
    - stage2_gmm_score
    - final_class ('ANOMALY' or 'BENIGN')

    Parameters:
    - package: loaded package containing model components   
    - sample: input sample to predict
    """
    models = package['models']
    feature_names = models['feature_names']
    sample =sample[feature_names]
    df = _to_dataframe(sample, feature_names)

    Xs = models['scaler'].transform(df)
    recon = models['autoencoder'].predict(Xs, verbose=0)

    mae = float(np.mean(np.abs(Xs - recon)))
    err_vec = np.abs(Xs - recon)
    gmm_score = float(models['gmm_recon'].score_samples(err_vec)[0])

    mae_thr = models['threshold_mae']
    gmm_thr = models['threshold_gmm']

    stage1 = 'ANOMALY' if mae > mae_thr else 'BENIGN'
    stage2 = 'ANOMALY' if gmm_score < gmm_thr else 'BENIGN'
    final_class = stage2  # use GMM as final stage

    return {
        'stage1_mae': mae,
        'stage1_prediction': stage1,
        'stage2_gmm_score': gmm_score,
        'final_class': final_class,
        'thresholds': {'mae': mae_thr, 'gmm': gmm_thr},
        'feature_names': feature_names,
        'sample_df': df
    }

def shap_values(pkg, sample):
    """
    Compute SHAP values for the given sample (both AE MAE and GMM score stages).
    Returns (ae_shap_values, gmm_shap_values).
    parameters:
    - pkg: loaded package with models and explainers
    - sample: input sample to explain
    """
    ae_explainer = pkg['ae_explainer']
    gmm_explainer = pkg['gmm_explainer']
    if ae_explainer is None or gmm_explainer is None:
       raise RuntimeError("Saved SHAP explainers not found. Please save and load ae_shap_explainer.joblib and gmm_shap_explainer.joblib.")
    
    sample = sample[pkg['models']['feature_names']]
    df = _to_dataframe(sample, pkg['models']['feature_names'])
    ae_sv = ae_explainer(df)
    gmm_sv = gmm_explainer(df)
    return ae_sv, gmm_sv

def shap_values_v2(ae_explainer, gmm_explainer, feature_names, sample):
    """
    Used only when explainers are loaded separately.
    Compute SHAP values for the given sample (both AE MAE and GMM score stages).
    Requires saved explainers. Returns (ae_shap_values, gmm_shap_values).
    Parameters:
    - ae_explainer: loaded AE SHAP explainer
    - gmm_explainer: loaded GMM SHAP explainer
    - feature_names: list of feature names
    - sample: input sample to explain
    """
    if ae_explainer is None or gmm_explainer is None:
       raise RuntimeError("Saved SHAP explainers not found. Please save and load ae_shap_explainer.joblib and gmm_shap_explainer.joblib.")
    
    sample = sample[feature_names]
    df = _to_dataframe(sample, feature_names)
    ae_sv = ae_explainer(df)
    gmm_sv = gmm_explainer(df)
    return ae_sv, gmm_sv

def plot_waterfalls(ae_shap_values, gmm_shap_values, max_display=15, show_ae=False):
    """
    Show waterfall plots for SHAP values.
    Parameters:
    - ae_shap_values: SHAP values for AE MAE stage
    - gmm_shap_values: SHAP values for GMM score stage
    - max_display: maximum number of features to display
    - show_ae: whether to show AE MAE SHAP explanation
    """
    print("="*100)
    print("=== SHAP Explanation Results ===")
    print("="*100)
    if show_ae:
        print("Autoencoder MAE SHAP Explanation:")
        shap.plots.waterfall(ae_shap_values[0], max_display=max_display, show=True)
    print("======== GMM Score SHAP Explanation: =========")
    shap.plots.waterfall(gmm_shap_values[0], max_display=max_display, show=True)

def predict_and_visualize_single_flow(pkg, sample, actual_label, show_ae=False):
    """
    predict and explain a single sample using the loaded package.
    Parameters:
    - pkg: loaded package with models and explainers
    - sample: input sample to predict and explain
    - actual_label: actual label of the sample for comparison
    - show_ae: whether to show AE MAE SHAP explanation
    """
    prediction_result=predict_single_flow(pkg, sample)
    ae_shap_values, gmm_shap_values = shap_values(pkg, sample)

    print("="*100)
    print(f"AE-GMM PREDICTION & EXPLANATION \nFinal Stage Threshold (GMM Threshold) is {prediction_result['thresholds']['gmm']:.6f}")
    print(f"Actual Label: {actual_label}")
    print("="*100)
    print(f"Stage 1 (AE MAE) Score: {prediction_result['stage1_mae']:.6f}")
    print(f"Stage 2 (GMM Score) Score: {prediction_result['stage2_gmm_score']:.6f}")
    print(f"Final Classification: {prediction_result['final_class']}")

    plot_waterfalls(ae_shap_values, gmm_shap_values, show_ae=show_ae)
