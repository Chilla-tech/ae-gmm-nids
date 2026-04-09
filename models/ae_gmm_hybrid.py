# Autoencoder + GMM hybrid model pipeline for NIDS
# Combines AE for reconstruction error extraction and GMM for probability modeling on error vectors.
# Usage: import AEGMMPipeline, then call fit() with training and validation data.

import numpy as np
from sklearn.preprocessing import StandardScaler
from .ae import build_autoencoder
from training.train_ae import train_autoencoder
from .gmm import fit_gmm, score_gmm
from inference.calculate_thres import mae_val_threshold, gmm_val_threshold

class AEGMMPipeline:
    def __init__(self):
        #self.scaler = scaler # Pre-fitted scaler from preprocessing
        self.autoencoder = None
        self.gmm = None
        self.threshold_mae = None
        self.threshold_gmm = None
        self.feature_names = None

    def fit(self, X_train_norm, X_val_norm, feature_names,epochs=500, batch_size=32, lr=1e-3, l1=1e-5, clipN=1.0):
        self.feature_names = list(feature_names)
        # X_* already scaled from preprocessing
        # Train Autoencoder
        self.autoencoder = build_autoencoder(X_train_norm.shape[1], l1=l1, lr=lr, clipN=clipN)
        self.autoencoder, _ = train_autoencoder(self.autoencoder, X_train_norm, X_val_norm, epochs=epochs, batch_size=batch_size)
        # thresholds
        self.threshold_mae, val_mae = mae_val_threshold(self.autoencoder, X_val_norm, percentile=98)

        # GMM on error vectors
        self.gmm = fit_gmm(self.autoencoder, X_train_norm)
        val_scores = score_gmm(self.autoencoder, self.gmm, X_val_norm)
        self.threshold_gmm = gmm_val_threshold(val_scores, percentile=1.2)
        return self
    
    
    def predict_mae(self, X_scaled):
        recons = self.autoencoder.predict(X_scaled, verbose=0)
        mae_errors = np.mean(np.abs(X_scaled - recons), axis=1)
        return mae_errors

    def predict_gmm_scores(self, X_scaled):
        gmm_scores = score_gmm(self.autoencoder, self.gmm, X_scaled)
        preds = (gmm_scores > self.threshold_gmm).astype(int) # 1: Normal, 0: Anomaly to align with the label encoding which encodes 'BENIGN' as 1
        return preds, gmm_scores