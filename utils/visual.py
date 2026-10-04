# Utility functions for visualization of anomaly detection results

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def _finish(save_path=None):
    """Save to a vector/raster file if save_path is given (e.g. 'fig.pdf', 'fig.svg'),
    in addition to showing the plot. The format is inferred from the file extension."""
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_mae_distribution(mae_errors, labels, threshold_mae, bins=50, range_=None, save_path=None):
    """
    Plot MAE reconstruction error distribution for Normal vs Anomaly with threshold line.
    - mae_errors: 1D numpy array of MAE per sample (scaled space).
    - labels: array-like of string labels; 'BENIGN' considered normal, others anomaly.
    - threshold_mae: float, MAE decision threshold.
    - bins: int, histogram bins.
    - range_: tuple(min, max) for histogram x-range (optional).
    - save_path: optional path to also save the figure to, e.g. 'fig.pdf' for a true vector file.
    """
    labels = np.array(labels)
    normal_mask = labels == 'BENIGN'
    anomaly_mask = labels != 'BENIGN'

    plt.figure(figsize=(10, 6))
    plt.hist(mae_errors[normal_mask], bins=bins, alpha=0.6, label='Normal',
             range=range_ if range_ is not None else None)
    plt.hist(mae_errors[anomaly_mask], bins=bins, alpha=0.6, label='Anomaly',
             range=range_ if range_ is not None else None)
    plt.axvline(threshold_mae, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold_mae:.4f}')
    plt.xlabel('Mean Absolute Error (MAE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    _finish(save_path)

def plot_gmm_score_distribution(gmm_scores, labels, threshold_gmm, bins=50, range_=(-50, 100), save_path=None):
    """
    Plot GMM log-probability scores for Normal vs Anomaly with decision boundary.
    - gmm_scores: 1D numpy array of GMM score_samples outputs.
    - labels: array-like of string labels; 'BENIGN' considered normal.
    - threshold_gmm: float, GMM decision threshold (lower => more anomalous).
    - bins: int, histogram bins.
    - range_: tuple(min, max) for histogram x-range.
    - save_path: optional path to also save the figure to, e.g. 'fig.pdf' for a true vector file.
    """
    labels = np.array(labels)
    normal_mask = labels == 'BENIGN'
    anomaly_mask = labels != 'BENIGN'

    plt.figure(figsize=(10, 6))
    plt.hist(gmm_scores[normal_mask], bins=bins, alpha=0.6, label='Normal', range=range_)
    plt.hist(gmm_scores[anomaly_mask], bins=bins, alpha=0.6, label='Anomaly', range=range_)
    plt.axvline(threshold_gmm, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold_gmm:.4f}')
    plt.xlabel('GMM Log Probability Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of GMM Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    _finish(save_path)

def plot_confusion_matrices(y_true_binary, y_pred_mae_binary, y_pred_gmm_binary, class_names=('Normal','Anomaly'), save_path=None):
    """
    Plot side-by-side confusion matrices for MAE-only and AE+GMM predictions.
    - y_true_binary: 1D array of ground truth (0=Normal, 1=Anomaly).
    - y_pred_mae_binary: 1D array of MAE-based predictions (0/1).
    - y_pred_gmm_binary: 1D array of GMM-based predictions (0/1).
    - class_names: tuple of class display names aligned with 0/1.
    - save_path: optional path to also save the figure to, e.g. 'fig.pdf' for a true vector file.
    """
    cm_mae = confusion_matrix(y_true_binary, y_pred_mae_binary)
    cm_gmm = confusion_matrix(y_true_binary, y_pred_gmm_binary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_mae, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix - MAE')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_gmm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix - AE+GMM')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    _finish(save_path)
