# Evaluation utilities for AE + GMM NIDS
# McNemar's test implementation for paired classifier comparison to assess statistical significance of differences in performance 
# between Baseline AE and hybrid AE+GMM.

from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

def evaluate_mae(actual_attack_type: Sequence[str], mae_scores, threshold_mae):
    pred = np.where(mae_scores > threshold_mae, 'Anomaly', 'Normal') # Predict 'Anomaly' if MAE > threshold
    actual = np.where(np.array(actual_attack_type) != 'BENIGN', 'Anomaly', 'Normal') 
    print("\nClassification Report (AutoEncoder Model):")
    print(classification_report(actual, pred))
    rep=classification_report(actual, pred, output_dict=True)
    cm = confusion_matrix(actual, pred).tolist()
    return {'report': rep, 'confusion_matrix':cm }

def evaluate_gmm(true_labels_binary, y_pred_binary, class_names):
    rep = classification_report(true_labels_binary, y_pred_binary, target_names=class_names, output_dict=True)
    cm = confusion_matrix(true_labels_binary, y_pred_binary).tolist()
    print("\nClassification Report (GMM Model):")
    print(classification_report(true_labels_binary, y_pred_binary, target_names=class_names))
    return {'report': rep, 'confusion_matrix': cm}

# McNemar's Test for paired classifier comparison
def mcnemar_test(y_true_binary, y_pred_A_binary, y_pred_B_binary, exact=False, continuity=True):
    """
    Run McNemar's test comparing classifier A vs B on the same instances.
    """
    # correctness arrays
    A_correct = (y_pred_A_binary == y_true_binary).astype(int)
    B_correct = (y_pred_B_binary == y_true_binary).astype(int)

    # contingency cells
    both_correct = int(np.sum((A_correct == 1) & (B_correct == 1)))
    A_correct_B_wrong = int(np.sum((A_correct == 1) & (B_correct == 0)))
    A_wrong_B_correct = int(np.sum((A_correct == 0) & (B_correct == 1)))
    both_wrong = int(np.sum((A_correct == 0) & (B_correct == 0)))

    table = np.array([[both_correct, A_correct_B_wrong],
                      [A_wrong_B_correct, both_wrong]])

    # McNemar’s test (focuses on off-diagonal b vs c)
    result = mcnemar(table, exact=exact, correction=continuity)

    print("Contingency table (correctness):")
    print(table)
    print(f"Off-diagonals: b={A_correct_B_wrong}, c={A_wrong_B_correct}")
    print(f"McNemar statistic: {result.statistic:.4f}, p-value: {result.pvalue:.6f}")

    return result, table