"""
Create Toy Dataset for Quick Verification

This script creates a stratified toy dataset from the full CSE-CIC-IDS2018 dataset
for quick verification of the AE-GMM NIDS pipeline. 

The script performs simple stratified sampling without preprocessing - all columns
and values are preserved as-is. Preprocessing will be done in the demo notebooks.

Usage:
    python scripts/create_toy_dataset.py --input data/raw/CSECICIDS2018_improved.csv --output data/toy_dataset.csv --n_samples 5000

Requirements:
    - Full CSE-CIC-IDS2018 dataset downloaded to data/raw/
    - pandas installed
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def create_toy_dataset(
    input_csv: str,
    output_csv: str,
    n_samples: int = 5000,
    random_state: int = 42
):
    """
    Create a toy dataset with stratified sampling by Label column.
    
    Args:
        input_csv: Path to full CSE-CIC-IDS2018 dataset
        output_csv: Path to save toy dataset
        n_samples: Total number of samples in toy dataset
        random_state: Random seed for reproducibility
    """
    print(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if Label column exists
    if 'Label' not in df.columns:
        print("\nError: 'Label' column not found in dataset.")
        print("Available columns:", list(df.columns))
        sys.exit(1)
    
    print(f"\nLabel distribution in original dataset:")
    print(df['Label'].value_counts())
    
    # Perform stratified sampling
    print(f"\nPerforming stratified sampling of {n_samples} samples...")
    
    # Calculate fraction to sample
    sample_fraction = min(n_samples / len(df), 1.0)
    
    if sample_fraction >= 1.0:
        print(f"\nWarning: Requested samples ({n_samples}) >= dataset size ({len(df)})")
        print(f"Using entire dataset.")
        toy_dataset = df.copy()
    else:
        # Stratified sampling by Label
        toy_dataset = df.groupby('Label', group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=random_state)
        ).reset_index(drop=True)
        
        # If we didn't get exactly n_samples, adjust
        if len(toy_dataset) < n_samples:
            # Sample more to reach target
            remaining = n_samples - len(toy_dataset)
            extra_samples = df.sample(n=remaining, random_state=random_state)
            toy_dataset = pd.concat([toy_dataset, extra_samples]).drop_duplicates().reset_index(drop=True)
        elif len(toy_dataset) > n_samples:
            # Randomly remove excess
            toy_dataset = toy_dataset.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    
    # Shuffle the dataset
    toy_dataset = toy_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    toy_dataset.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Toy dataset created successfully!")
    print(f"{'='*60}")
    print(f"Saved to: {output_csv}")
    print(f"Total samples: {len(toy_dataset)}")
    print(f"Number of columns: {len(toy_dataset.columns)}")
    print(f"\nLabel distribution in toy dataset:")
    print(toy_dataset['Label'].value_counts())
    print(f"\nFile size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return toy_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Create stratified toy dataset from CSE-CIC-IDS2018',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 5000 samples
  python scripts/create_toy_dataset.py --input data/raw/CSECICIDS2018_improved.csv --output data/toy_dataset.csv
  
  # Larger toy dataset
  python scripts/create_toy_dataset.py --input data/raw/CSECICIDS2018_improved.csv --output data/toy_dataset.csv --n_samples 10000
  
  # Smaller toy dataset for quick testing
  python scripts/create_toy_dataset.py --input data/raw/CSECICIDS2018_improved.csv --output data/toy_dataset.csv --n_samples 2000
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to full CSE-CIC-IDS2018 dataset CSV'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/toy_dataset.csv',
        help='Path to save toy dataset (default: data/toy_dataset.csv)'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=5000,
        help='Total number of samples in toy dataset (default: 5000)'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("\nPlease download the CSE-CIC-IDS2018 dataset first.")
        print("See data/README.md for instructions.")
        sys.exit(1)
    
    if args.n_samples < 100:
        print(f"Error: n_samples must be at least 100 (got {args.n_samples})")
        sys.exit(1)
    
    # Create toy dataset
    create_toy_dataset(
        input_csv=args.input,
        output_csv=args.output,
        n_samples=args.n_samples,
        random_state=args.random_state
    )


if __name__ == '__main__':
    main()
