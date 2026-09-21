# Utility functions for data preprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


ATTACK_MAP = {
    'BENIGN': 'BENIGN',
    'DDoS': 'Anormal', 'DoS Hulk': 'Anormal', 'DoS GoldenEye': 'Anormal',
    'DoS slowloris': 'Anormal', 'DoS Slowhttptest': 'Anormal',
    'Infiltration - NMAP Portscan': 'Anormal',
    'Infiltration - Communication Victim Attacker': 'Anormal',
    'SSH-BruteForce': 'Anormal', 'Botnet Ares': 'Anormal',
    'Web Attack - Brute Force': 'Anormal', 'Web Attack - XSS': 'Anormal',
    'DDoS-LOIC-HTTP': 'Anormal', 'Infiltration - Dropbox Download': 'Anormal',
    'DDoS-HOIC': 'Anormal', 'DDoS-LOIC-UDP': 'Anormal'
}

DROP_COLS = ['id','Flow ID','Timestamp','Src IP','Dst IP','Attempted Category']

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print("Cleaning data...")
    # remove "Attempted" rows if any
    if 'Label' in df.columns:
        df = df[~df['Label'].astype(str).str.contains('Attempted', na=False)]
    # drop obvious identifiers
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    # infinities -> NaN. NOTE: duplicates are intentionally NOT dropped: the published results were
    # obtained on the data without deduplication (see README, "Reproducing the paper's split").
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in ['Flow Bytes/s','Flow Packets/s']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # map to Attack Type
    df['Attack Type'] = df['Label'].map(ATTACK_MAP) if 'Label' in df.columns else df['Attack Type']
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])

    # drop 1-unique columns
    nunique = df.nunique()
    df = df[nunique[nunique > 1].index]
    # drop rows without target. Labels missing from ATTACK_MAP (e.g. 'DoS Slowloris', 'Web Attack - SQL')
    # are discarded here, exactly as in the experiments reported in the paper.
    n_unmapped = int(df['Attack Type'].isna().sum())
    if n_unmapped:
        print(f'Discarding {n_unmapped} flows whose label is not in ATTACK_MAP')
    df = df.dropna(subset=['Attack Type'])
    return df

def rf_top_features(df: pd.DataFrame, label_col='Attack Type', top_n=23, random_state=0, split_seed=42):
    print(f'Selecting top {top_n} features using Random Forest...')
    X = df.drop(columns=[label_col])
    y = df[label_col]
    # As in the paper's experiments, the forest is fit on a stratified 70% split of the subsample
    X, _, y, _ = train_test_split(X, y, test_size=0.30, random_state=split_seed, stratify=y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, max_features='sqrt',
        min_samples_split=5, random_state=random_state, n_jobs=-1
    ).fit(X, y_enc)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]
    print(f'Top {top_n} features: {X.columns[top_idx].tolist()}')
    return list(X.columns[top_idx])

def drop_correlated(df: pd.DataFrame, threshold=0.9):
    print(f'Dropping correlated features with threshold > {threshold}...')
    num = df.select_dtypes(include=np.number)
    corr = num.corr()
    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                to_drop.add(cols[i])
    print(f'Dropped {len(to_drop)} features: {to_drop}')
    print(f'Remaining features: {df.shape[1] - len(to_drop)}')
    return df.drop(columns=list(to_drop)), list(to_drop)

def make_balanced_split(df: pd.DataFrame, label_col='Attack Type', total=286000, intrusion_fraction=1/3.15, seed=42):
    """Subsample `total` flows, of which a fraction `intrusion_fraction` (default 1/3.15, ~31.7%) are intrusions."""
    print(f'Creating subsampled dataset with total={total} and intrusion fraction={intrusion_fraction:.4f}...')
    normal = df[df[label_col]=='BENIGN']
    intru = df[df[label_col]!='BENIGN']
    intru_needed = int(total * intrusion_fraction)
    normal_needed = total - intru_needed
    normal_sample = normal.sample(n=min(normal_needed, len(normal)), replace=False, random_state=seed)
    intru_sample = intru.sample(n=min(intru_needed, len(intru)), replace=False, random_state=seed)
    balanced_df = pd.concat([normal_sample, intru_sample]).sample(frac=1, random_state=seed)
    print(balanced_df['Attack Type'].value_counts())
    return balanced_df


def scale_for_ae(train_df, val_df, test_df, label_col='Attack Type'):
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(train_df[train_df[label_col]=='BENIGN'].drop(columns=[label_col]))
    X_val_norm   = scaler.transform(val_df[val_df[label_col]=='BENIGN'].drop(columns=[label_col]))
    X_test_all   = scaler.transform(test_df.drop(columns=[label_col]))
    return scaler, X_train_norm, X_val_norm, X_test_all

def get_label_encoder():
    """Return configured label encoder"""
    return LabelEncoder()