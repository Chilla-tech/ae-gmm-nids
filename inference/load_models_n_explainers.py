# load models and SHAP explainers for inference

import joblib
import pandas as pd
import numpy as np

# GLOBAL FUNCTIONS FOR SHAP because SHAP explainers in the pretrained folder need them at module level
def ae_reconstruction_function_global(X, model_components):
    """Global standalone AE function for SHAP"""
    autoencoder = model_components['autoencoder']
    scaler = model_components['scaler']
    feature_names = model_components['feature_names']

    X = X[list(feature_names)]

    if isinstance(X, pd.DataFrame):
        X_scaled = scaler.transform(X)
    else:
        temp_df = pd.DataFrame(X, columns=feature_names)
        X_scaled = scaler.transform(temp_df)

    reconstructions = autoencoder.predict(X_scaled, verbose=0)
    mae_errors = np.mean(np.abs(X_scaled - reconstructions), axis=1)
    return mae_errors


def gmm_score_function_global(X, model_components):
    """Global standalone GMM function for SHAP"""
    autoencoder = model_components['autoencoder']
    gmm = model_components['gmm']
    scaler = model_components['scaler']
    feature_names = model_components['feature_names']

    X = X[list(feature_names)]

    if isinstance(X, pd.DataFrame):
        X_scaled = scaler.transform(X)
    else:
        temp_df = pd.DataFrame(X, columns=feature_names)
        X_scaled = scaler.transform(temp_df)

    reconstructions = autoencoder.predict(X_scaled, verbose=0)
    error_vectors = np.abs(X_scaled - reconstructions)
    gmm_scores = gmm.score_samples(error_vectors)
    return gmm_scores

class PicklableShapWrapper:
    """Wrapper class that can be pickled with SHAP explainers from the pretrained folder"""

    def __init__(self, function_name, model_components):
        self.function_name = function_name
        self.model_components = model_components

    def __call__(self, X):
        if self.function_name == 'ae_reconstruction':
            return ae_reconstruction_function_global(X, self.model_components)
        elif self.function_name == 'gmm_score':
            return gmm_score_function_global(X, self.model_components)
        else:
            raise ValueError(f"Unknown function: {self.function_name}")
    


def load_complete_package(package_dir):
    """
    Load minimal components: model package and pre-trained SHAP explainers.
    Parameters:
    - package_dir: directory containing the model package and SHAP explainers
    """
    models = joblib.load(f"{package_dir}/aegmm_model_package.joblib")
    print("Model package loaded.")

    # Load pre-trained SHAP explainers
    ae_explainer = joblib.load(f"{package_dir}/ae_shap_explainer.joblib")
    print("AE SHAP explainer loaded.")
    gmm_explainer = joblib.load(f"{package_dir}/gmm_shap_explainer.joblib")
    print("GMM SHAP explainer loaded.")

    return {
        'models': models,
        'ae_explainer': ae_explainer,
        'gmm_explainer': gmm_explainer
    }