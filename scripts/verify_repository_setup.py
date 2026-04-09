"""
Repository Setup Checklist

Run this script to verify your GitHub repository is ready for anonymous review.
"""

import os
from pathlib import Path
import sys


def check_file_exists(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        print(f"  ✓ {description}")
        if size > 0:
            if size > 1024*1024:
                print(f"    Size: {size/(1024*1024):.2f} MB")
            else:
                print(f"    Size: {size/1024:.2f} KB")
        return True
    else:
        print(f"  ✗ {description} - MISSING")
        return False


def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    path = Path(dirpath)
    if path.exists() and path.is_dir():
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - MISSING")
        return False


def search_for_identifiers(root_dir):
    """Search for potentially identifying information"""
    issues = []
    
    # Patterns to search for
    patterns = [
        (r"C:\\Users\\PC", "Absolute Windows path"),
        (r"Fudan", "University name"),
        # Add more patterns as needed
    ]
    
    extensions = ['.py', '.ipynb', '.md', '.txt']
    
    print("\n5. Checking for identifying information...")
    
    for ext in extensions:
        for file in Path(root_dir).rglob(f'*{ext}'):
            # Skip certain directories
            if any(skip in str(file) for skip in ['.git', '__pycache__', '.ipynb_checkpoints', 'venv', 'env']):
                continue
            
            try:
                content = file.read_text(encoding='utf-8', errors='ignore')
                for pattern, desc in patterns:
                    if pattern.lower() in content.lower():
                        issues.append(f"{file}: Found {desc}")
            except:
                pass
    
    if issues:
        print("  ✗ Found potential identifying information:")
        for issue in issues[:10]:  # Show first 10
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues)-10} more")
        return False
    else:
        print("  ✓ No identifying information found")
        return True


def main():
    print("="*70)
    print("AE-GMM NIDS - Repository Setup Checklist")
    print("="*70)
    print("\nThis script verifies your repository is ready for anonymous review.\n")
    
    all_checks_passed = True
    
    # 1. Core Documentation
    print("1. Core Documentation Files")
    checks = [
        ("README.md", "Main README with installation and usage"),
        ("LICENSE", "MIT License file"),
        ("CITATION.md", "Citation information (anonymous)"),
        (".gitignore", "Git ignore file"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    for filepath, desc in checks:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # 2. Data Directory
    print("\n2. Data Directory")
    data_checks = [
        ("data/README.md", "Dataset documentation"),
        ("data/toy_dataset.csv", "Toy dataset for verification"),
    ]
    
    for filepath, desc in data_checks:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    if not check_directory_exists("data", "Data directory"):
        all_checks_passed = False
    
    # 3. Code Structure
    print("\n3. Code Structure")
    code_dirs = [
        ("models", "Model definitions"),
        ("training", "Training scripts"),
        ("inference", "Inference scripts"),
        ("utils", "Utility functions"),
        ("scripts", "Helper scripts"),
        ("examples", "Usage examples"),
    ]
    
    for dirpath, desc in code_dirs:
        if not check_directory_exists(dirpath, desc):
            all_checks_passed = False
    
    # 4. Pretrained Models
    print("\n4. Pretrained Models")
    model_checks = [
        ("pretrained/README.md", "Model documentation"),
        ("pretrained/complete_package_20250914_065942/aegmm_model_package.joblib", "Main model package"),
        ("pretrained/complete_package_20250914_065942/ae_shap_explainer.joblib", "AE SHAP explainer"),
        ("pretrained/complete_package_20250914_065942/gmm_shap_explainer.joblib", "GMM SHAP explainer"),
    ]
    
    for filepath, desc in model_checks:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # 5. Notebooks
    print("\n5. Demonstration Notebooks")
    notebook_checks = [
        ("demo_notebooks/Demo.ipynb", "Full training pipeline demo"),
        ("demo_notebooks/AEGMM_Demo.ipynb", "Trained model demo"),
        ("demo_notebooks/pretrained/test_pre_ae_gmm.ipynb", "Pretrained model demo (primary)"),
    ]
    
    for filepath, desc in notebook_checks:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # 6. Output Directories
    print("\n6. Output Directories")
    if not check_directory_exists("aegmm_nids(full_train)", "Training output directory"):
        all_checks_passed = False
    
    if check_file_exists("aegmm_nids(full_train)/README.md", "Training output README"):
        pass
    else:
        all_checks_passed = False
    
    if check_directory_exists("results", "Reference results directory"):
        check_file_exists("results/README.md", "Results README")
    else:
        print("  ⚠ Results directory will be created after running notebooks")
    
    # 7. Search for identifying information
    root_dir = Path(__file__).parent.parent
    if not search_for_identifiers(root_dir):
        all_checks_passed = False
    
    # 8. Additional Documentation
    print("\n6. Additional Documentation (Optional)")
    optional_docs = [
        ("docs/environment_setup.md", "Environment setup guide"),
    ]
    
    for filepath, desc in optional_docs:
        if Path(filepath).exists():
            print(f"  ✓ {desc}")
        else:
            print(f"  ⚠ {desc} - Optional but recommended")
    
    # Summary
    print("\n" + "="*70)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED!")
        print("="*70)
        print("\nYour repository is ready for anonymous review!")
        print("\nNext steps:")
        print("  1. Test in clean environment:")
        print("     - Create new virtual environment")
        print("     - pip install -r requirements.txt")
        print("     - python examples/quick_start.py")
        print("  2. Create private GitHub repository")
        print("  3. Push all files")
        print("  4. Verify on GitHub (clone and test)")
        print("  5. Generate anonymous link or grant reviewer access")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*70)
        print("\nPlease fix the issues above before uploading to GitHub.")
        print("\nCommon issues:")
        print("  - Missing toy dataset: Run 'python scripts/create_toy_dataset.py'")
        print("  - Missing documentation: Check if all README files are created")
        print("  - Identifying info: Review and sanitize files marked above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
