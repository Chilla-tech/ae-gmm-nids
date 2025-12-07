"""
Quick Start Example - AE-GMM NIDS

This script demonstrates the minimal code needed to:
1. Load the pretrained model
2. Make predictions on sample data
3. Get SHAP explanations for the predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.load_models_n_explainers import load_complete_package, PicklableShapWrapper
from inference.predict_n_explain import batch_predict, predict_single_flow, predict_and_visualize_single_flow
from utils.prepro import load_and_clean
from utils.evaluation import evaluate_gmm
from utils.ignore_non_critical_warnings import suppress_non_critical_warnings
sys.modules['__main__'].PicklableShapWrapper = PicklableShapWrapper

suppress_non_critical_warnings()

def main():
    print("="*60)
    print("AE-GMM NIDS - Quick Start Example")
    print("="*60)
    
    # 1. Load pretrained model
    print("\n1. Loading pretrained model...")
    model_dir = "pretrained/complete_package_20250914_065942"
    
    try:
        package = load_complete_package(model_dir)
        print("✓ Model loaded successfully!")
        print(f"  - Features: {len(package['models']['feature_names'])}")
        print(f"  - Loaded MAE Threshold: {package['models']['threshold_mae']:.6f}")
        print(f"  - Loaded GMM Threshold: {package['models']['threshold_gmm']:.6f}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nMake sure you're running this script from the project root directory.")
        return
    
    # 2. Load toy dataset
    print("\n2. Loading toy dataset...")
    try:
        data = load_and_clean("data/toy_dataset.csv")
        print(f"✓ Dataset loaded: {len(data)} samples")
        
        # Separate features and labels
        X = data.drop(columns=['Attack Type'])
        y = data['Attack Type']
        
        # Get a sample (first attack sample for demonstration)
        attack_idx = (y == 'Anormal').idxmax()
        sample_features = X.loc[attack_idx:attack_idx]
        sample_label = y.loc[attack_idx]
        
        print(f"  - Selected sample: {sample_label}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nMake sure the toy dataset exists at data/toy_dataset.csv")
        print("Run: python scripts/create_toy_dataset.py to generate it")
        return
    
    # 3. Make prediction
    print("\n3. Making prediction...")
    try:
        result = predict_single_flow(package, sample_features)
        
        print("✓ Prediction complete!")
        print(f"\n  Stage 1 (AE):")
        print(f"    - MAE Score: {result['stage1_mae']:.6f}")
        print(f"    - Prediction: {result['stage1_prediction']}")
        
        print(f"\n  Stage 2 (AE+GMM):")
        print(f"    - GMM Score: {result['stage2_gmm_score']:.6f}")
        
        print(f"\n  Final Label: {result['final_class']}")
        print(f"  True Label: {sample_label}")
        
            
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        return
    
    # 4. SHAP Explanation (optional - may take a minute)
    print("\n4. Generating SHAP explanation...")
    print("  (This may take 30-60 seconds...)")
    try:
        predict_and_visualize_single_flow(
            package,
            sample_features,
            sample_label,
            show_ae=False
        )
        print("✓ SHAP waterfall plots displayed!")
        print("  (Close the plot windows to continue)")
        
    except Exception as e:
        print(f"✗ Error generating SHAP explanation: {e}")
        print("  (This is optional - predictions still work)")
    
    # 5. Batch prediction example
    print("\n5. Batch prediction on first 100 samples...")
    try:
        
        X_batch = X.head(100)
        y_batch = y.head(100)
        
        results_df = batch_predict(
            package,
            X_batch
        )
        
        # generate evaluation report
        y_true_binary = (y_batch == 'BENIGN').astype(int) # BENIGN=1, Anormal=0 to align with label encoding used in training
        y_pred_binary = results_df['anomaly'].values
        evaluate_gmm(true_labels_binary=y_true_binary, y_pred_binary=y_pred_binary, class_names=package['models']['label_encoder'].classes_)
        
    except Exception as e:
        print(f"✗ Error during batch prediction: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Quick Start Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run full demo: jupyter notebook demo_notebooks/pretrained/test_pre_ae_gmm.ipynb")
    print("  2. See examples/batch_inference.py for larger scale prediction")
    print("  3. See examples/explain_prediction.py for detailed SHAP analysis")
    print("  4. Train your own model: python training/full_train.py --help")
    print("\n")


if __name__ == '__main__':
    main()
