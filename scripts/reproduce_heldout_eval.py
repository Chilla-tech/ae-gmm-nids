# Regenerates the paper's train/test split from the full CSE-CIC-IDS2018 CSV using the repo's own
# preprocessing, checks that it yields the pretrained model's 17 features, and evaluates the
# pretrained AE and AE-GMM (with McNemar's test) on the held-out test set.
#
# Usage:
#   python scripts/reproduce_heldout_eval.py --data path/to/CSECIC-IDS2018_subset.csv [--save-indices]

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from inference.load_models_n_explainers import load_complete_package
from inference.predict_n_explain import batch_predict
from utils.evaluation import mcnemar_test
from utils.prepro import load_and_clean, make_balanced_split, rf_top_features, drop_correlated

REPO = Path(__file__).resolve().parent.parent


def regenerate_split(csv_path, top_n=23, corr_thr=0.9, total=286000):
    df = load_and_clean(csv_path)
    subsample = make_balanced_split(df, total=total)
    top_feats = rf_top_features(subsample, top_n=top_n)
    df_small = subsample[top_feats + ['Attack Type']]
    df_small, _ = drop_correlated(df_small, threshold=corr_thr)
    train_set, test_set = train_test_split(df_small, test_size=0.30, random_state=42,
                                           stratify=df_small['Attack Type'])
    return df_small, train_set, test_set


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to the full CSE-CIC-IDS2018 CSV')
    p.add_argument('--package', default=str(REPO / 'pretrained' / 'complete_package_20250914_065942'))
    p.add_argument('--save-indices', action='store_true',
                   help='Write the held-out test row indices to data/heldout_test_indices.csv')
    args = p.parse_args()

    _, _, test_set = regenerate_split(args.data)

    pkg = load_complete_package(args.package)
    feats = list(pkg['models']['feature_names'])
    regen = [c for c in test_set.columns if c != 'Attack Type']
    if feats != regen:
        sys.exit(f'Feature mismatch.\n pretrained: {feats}\n regenerated: {regen}')
    print('OK: regenerated features are identical to the pretrained model features.')

    if args.save_indices:
        out = REPO / 'data' / 'heldout_test_indices.csv'
        test_set.index.to_series().to_csv(out, index=False, header=['row_index'])
        print(f'Saved {len(test_set)} held-out row indices to {out}')

    le = pkg['models']['label_encoder']
    res = batch_predict(pkg, test_set)
    y = le.transform(test_set['Attack Type'])
    mae_pred = (res['mae'] < pkg['models']['threshold_mae']).astype(int).values
    gmm_pred = res['anomaly'].values

    print(f'\nHeld-out test set: n={len(test_set)}')
    print('\nStage 1 (AE only)')
    print(classification_report(y, mae_pred, target_names=le.classes_, digits=3))
    print('Stage 2 (AE + GMM)')
    print(classification_report(y, gmm_pred, target_names=le.classes_, digits=3))
    mcnemar_test(y, mae_pred, gmm_pred, exact=False, continuity=True)


if __name__ == '__main__':
    main()
