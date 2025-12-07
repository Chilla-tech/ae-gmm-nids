import shap
import joblib
import numpy as np
import pandas as pd

# Wrappers to make AE and GMM compatible with SHAP

class AEReconstructionWrapper:
    # Returns AE MAE reconstruction error for SHAP (stage-1)
    def __init__(self, autoencoder, scaler, feature_names):
        self.autoencoder = autoencoder
        self.scaler = scaler
        self.feature_names = feature_names

    def __call__(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X[self.feature_names]
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names)

        X_scaled = self.scaler.transform(X_df)
        recon = self.autoencoder.predict(X_scaled, verbose=0)
        return np.mean(np.abs(X_scaled - recon), axis=1)
    
class AEGMMWrapper:
    # Returns GMM anomaly scores for SHAP (stage-2)
    def __init__(self, autoencoder, gmm, scaler, feature_names):
        self.autoencoder = autoencoder
        self.gmm = gmm
        self.scaler = scaler
        self.feature_names = feature_names

    def __call__(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X[self.feature_names]
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names)

        X_scaled = self.scaler.transform(X_df)
        recon = self.autoencoder.predict(X_scaled, verbose=0)
        err_vec = np.abs(X_scaled - recon)
        return self.gmm.score_samples(err_vec)