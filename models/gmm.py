# Autoencoder reconstruction error modeling with Gaussian Mixture Model (GMM)
# Fits a GMM to the AE reconstruction error vectors and scores new samples.

import numpy as np
from sklearn.mixture import GaussianMixture

def fit_gmm(autoencoder, train_scaled, n_components=21, random_state=42,
                      max_iter=500, n_init=10, tol=1e-4, covariance_type='full'):
    recon = autoencoder.predict(train_scaled, verbose=0)
    err_train = np.abs(train_scaled - recon)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        tol=tol
    )
    print("Fitting GMM to error vectors...")
    gmm.fit(err_train)
    return gmm

def score_gmm(autoencoder, gmm, X_scaled):
    recon = autoencoder.predict(X_scaled, verbose=0)
    err = np.abs(X_scaled - recon)
    return gmm.score_samples(err)