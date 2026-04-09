## Threshold calculation for AE + GMM NIDS
# Computes thresholds for MAE and GMM scores based on validation data percentiles.

import numpy as np

def mae_val_threshold(autoencoder, X_val_scaled, percentile=98):
    recon = autoencoder.predict(X_val_scaled, verbose=0)
    mae = (abs(X_val_scaled - recon)).mean(axis=1)
    return float(np.percentile(mae, percentile)), mae

def gmm_val_threshold(gmm_scores_val, percentile=1.2):
    return float(np.percentile(gmm_scores_val, percentile))