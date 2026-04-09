# Full training script for AE + GMM hybrid model on CIC-IDS2017 dataset
# Saves trained model package with pre-fitted scaler and SHAP explainers
# Usage: python full_train.py --data <path_to_CIC-IDS2017_subset_csv>

import os
import joblib
import argparse
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import shap

from utils.prepro import load_and_clean, rf_top_features, drop_correlated, make_balanced_split, scale_for_ae, get_label_encoder
from models.ae_gmm_hybrid import AEGMMPipeline
from models.gmm import score_gmm
from inference.calculate_thres import mae_val_threshold
from utils.evaluation import evaluate_mae, evaluate_gmm
from utils.shap_aegmm_wrappers import AEReconstructionWrapper, AEGMMWrapper

def main(args):
    # 1) Load + clean
    df = load_and_clean(args.data)

    # Create subsampled balanced dataset because of limited resources
    baln_df = make_balanced_split(df, ratio_normal_to_intrusions=0.317, total=286000) 
    
    # Feature selection then correlation pruning
    top_feats = rf_top_features(baln_df, top_n=args.top_n)
    df_small = baln_df[top_feats + ['Attack Type']]
    df_small, dropped = drop_correlated(df_small, threshold=args.corr_thr)

    # Train/Val/Test split
    train_set, test = train_test_split(df_small, test_size = 0.30, random_state = 42, stratify=df_small['Attack Type'])
    train, val = train_test_split(train_set, test_size=0.2, random_state=42)

    scaler, X_train_norm, X_val_norm, X_test_all = scale_for_ae(train, val, test) # Scale data for AE

    #Training pipeline
    pipe = AEGMMPipeline().fit(X_train_norm, X_val_norm, feature_names=df_small.columns[:-1])

    
    # Stage-1 MAE
    thr_mae, mae_val = mae_val_threshold(pipe.autoencoder, X_val_norm, percentile=98)
    
    # Stage-2 GMM
    lencode=get_label_encoder()
    y_true_enc = lencode.fit_transform(test['Attack Type']) # BENIGN=1, Anomaly=0
    gmm_scores = score_gmm(pipe.autoencoder, pipe.gmm, X_test_all)
    pred_gmm = (gmm_scores > pipe.threshold_gmm).astype(int) # 1: Normal, 0: Anomaly to align with the label encoding which encodes 'BENIGN' as 1

    pred_mae=pipe.predict_mae(X_test_all)
    
    rep=evaluate_mae(actual_attack_type=test['Attack Type'], mae_scores=pred_mae, threshold_mae=thr_mae)
    gmm_rep=evaluate_gmm(true_labels_binary=y_true_enc, y_pred_binary=pred_gmm, class_names=lencode.classes_)

    # Create SHAP explainers
    print('='*100)
    print('Creating SHAP explainers...')
    print('='*100)
    background_data = pd.DataFrame(
        scaler.inverse_transform(X_train_norm[:1000]), # use unscaled normal samples for background
        columns=list(df_small.columns[:-1])
    )

    ae_wrapper = AEReconstructionWrapper(
        autoencoder=pipe.autoencoder,
        scaler=scaler,
        feature_names=list(df_small.columns[:-1])
    )
    gmm_wrapper = AEGMMWrapper(
        autoencoder=pipe.autoencoder,
        gmm=pipe.gmm,
        scaler=scaler,
        feature_names=list(df_small.columns[:-1])
    )

    ae_explainer = shap.Explainer(ae_wrapper, background_data)
    gmm_explainer = shap.Explainer(gmm_wrapper, background_data)

    # Save package
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('aegmm_nids(full_train)', f'AEGMM_hybrid_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    models_package = {
        'autoencoder': pipe.autoencoder,
        'gmm_recon': pipe.gmm,
        'scaler': scaler,
        'threshold_mae': thr_mae,
        'threshold_gmm': pipe.threshold_gmm,
        'feature_names': list(df_small.columns[:-1]),
        'train_info': {'time': ts},
        'label_encoder': lencode,
        'ae_perf_rep' : rep,
        'ae_gmm_perf_rep' : gmm_rep
    }
    joblib.dump(models_package, os.path.join(out_dir, 'aegmm_model_package.joblib'))
    #print(f'Saved: {out_dir}')
    joblib.dump(ae_explainer, os.path.join(out_dir, 'ae_shap_explainer.joblib'))
    joblib.dump(gmm_explainer, os.path.join(out_dir, 'gmm_shap_explainer.joblib'))
    print(f'Models and SHAP explainers saved at {out_dir}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to CSECIC-IDS2018 subset CSV')
    p.add_argument('--top_n', type=int, default=23)
    p.add_argument('--corr_thr', type=float, default=0.9)
    p.add_argument('--total', type=int, default=286000)
    args = p.parse_args()
    main(args)