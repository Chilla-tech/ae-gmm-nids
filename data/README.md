# Dataset Information

## Toy Dataset (Included)

**File**: `toy_dataset.csv` — 5,000 samples, stratified from the full dataset  
**Purpose**: Quick verification of the pipeline without downloading the full dataset  
**Usage**: Automatically used by `demo_notebooks/pretrained/test_pre_ae_gmm.ipynb`

## Full Dataset (Download Required for Path B)

**Dataset**: CSE-CIC-IDS2018 (Improved Version)  
**Download**: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip  
**Size**: ~2 GB compressed  
**Format**: CSV

After downloading, place the CSV at `data/raw/CSECICIDS2018_improved.csv`.

**Citation**:
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
