# Dataset Information

This directory contains dataset files for the AE-GMM Network Intrusion Detection System.

## Toy Dataset (Included)

**File**: `toy_dataset.csv`  
**Size**: ~2.9MB  
**Samples**: 5k  
**Purpose**: Quick verification of the pipeline without downloading the full dataset

### Contents
The toy dataset is a stratified sample from the full CSE-CIC-IDS2018 dataset:
- Stratified sampling ensures all attack types are proportionally represented
- All original columns and values preserved (no preprocessing)
- Preprocessing (feature selection, cleaning) happens in demo notebooks
- Default size: 5,000 samples

### Usage
The toy dataset is automatically used by `demo_notebooks/pretrained/test_pre_ae_gmm.ipynb` for demonstration purposes.

---

## Full Dataset (Not Included - Download Required)

### CSE-CIC-IDS2018 (Improved Version)

**Dataset**: CSE-CIC-IDS2018  
**Download URL**: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip  
**Size**: ~2 GB (compressed), ~4-5 GB (uncompressed)  
**Format**: CSV  

### Download Instructions

1. Download the dataset from the URL above
2. Extract the ZIP file
3. Place the CSV file in `data/raw/CSECICIDS2018_improved.csv`

```bash
# Create raw data directory
mkdir -p data/raw

# Download (Linux/Mac)
wget https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip -O data/raw/dataset.zip

# Or use curl
curl -L https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip -o data/raw/dataset.zip

# Extract
unzip data/raw/dataset.zip -d data/raw/

# Windows PowerShell
# Invoke-WebRequest -Uri "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip" -OutFile "data\raw\dataset.zip"
# Expand-Archive -Path "data\raw\dataset.zip" -DestinationPath "data\raw\"
```

### Dataset Description

The CSE-CIC-IDS2018 is an improved version of the original CIC-IDS2018 dataset, containing:

- **Normal Traffic**: BENIGN network flows
- **14 Attack Types**:
  - DDoS attacks-HOIC
  - DoS attacks-Hulk
  - DoS attacks-SlowHTTPTest
  - DoS attacks-GoldenEye
  - DoS attacks-Slowloris
  - DDOS attack-LOIC-UDP
  - Brute Force -Web
  - Brute Force -XSS
  - SQL Injection
  - Infiltration
  - Bot
  - SSH-Patator
  - FTP-Patator
  - DDoS attacks-LOIC-HTTP

### Features

The dataset contains 90+ network flow features, including:
- Flow duration, bytes, packets
- Packet length statistics
- Inter-arrival times
- Flag counts
- Protocol information
- And more...

### Preprocessing Pipeline

The full training script (`training/full_train.py`) performs the following preprocessing:

1. **Load and Clean**
   - Remove identifier columns: `id, Flow ID, Timestamp, Src IP, Dst IP, Attempted Category`
   - Handle infinite values and NaN
   - Map 14 attack types → "Anormal" (Anomaly label)

2. **Feature Selection** (via `--top_n 23`)
   - Random Forest feature importance ranking
   - Select top N features

3. **Correlation Pruning** (via `--corr_thr 0.9`)
   - Remove highly correlated features (threshold > 0.9)
   - Final feature count: ~17 features

4. **Balanced Sampling** (via `--total 286000`)
   - Create balanced dataset with specified total samples
   - Ratio: 68.3% BENIGN, 31.7% attacks (realistic real-world distribution)

5. **Scaling**
   - StandardScaler fitted on BENIGN training samples only
   - Applied to all samples

6. **Train/Validation/Test Split**
   - 70/30 train/test split
   - 80/20 train/validation split from training set

### Citation

```bibtex
@inproceedings{liu2022error,
title={Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018},
author={Liu, Lisa and Engelen, Gints and Lynar, Timothy and Essam, Daryl and Joosen, Wouter},
booktitle={2022 IEEE Conference on Communications and Network Security (CNS)},
pages={254--262},
year={2022},
organization={IEEE}
}
```

---

## Generating Your Own Toy Dataset

If you want to regenerate the toy dataset with different parameters:

```bash
python scripts/create_toy_dataset.py \
    --input data/raw/CSECICIDS2018_improved.csv \
    --output data/toy_dataset.csv \
    --n_samples 5000
```

**Options:**
- `--n_samples`: Number of samples (default: 5000)
- `--random_state`: Random seed for reproducibility (default: 42)

The script performs stratified sampling by Label column to ensure all attack types are represented proportionally.

---

## Directory Structure

```
data/
├── README.md                  # This file
├── toy_dataset.csv            # Included toy dataset (5K samples)
```

---

**Note**: The full dataset is NOT included in this repository due to its size. The toy dataset can be used for quick verification or download the full dataset for complete reproduction.
