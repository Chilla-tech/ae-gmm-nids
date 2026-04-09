# Python Environment Specification

This file documents the Python environment requirements for the AE-GMM NIDS project.

## Python Version

**Supported versions**: Python 3.8, 3.9, 3.10, 3.11

**Recommended**: Python 3.10

## Installation Methods

### Method 1: pip (Recommended for Quick Setup)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: conda (Recommended for Reproducibility)

```bash
# Create conda environment
conda create -n aegmm python=3.10

# Activate environment
conda activate aegmm

# Install dependencies
pip install -r requirements.txt
```

## Core Dependencies

### Deep Learning
- **TensorFlow** >= 2.12: Neural network framework for Autoencoder
- **Keras** >= 2.12: High-level neural networks API

**Note**: For GPU support (optional):
```bash
# TensorFlow with CUDA support (requires NVIDIA GPU)
pip install tensorflow[and-cuda]>=2.12
```

### Machine Learning
- **scikit-learn** >= 1.3: GMM, preprocessing, evaluation metrics
- **numpy** >= 1.24: Numerical computations
- **pandas** >= 1.5: Data manipulation

### Explainability
- **SHAP** >= 0.44: SHapley Additive exPlanations for model interpretability

### Visualization
- **matplotlib** >= 3.7: Plotting and visualization
- **seaborn** >= 0.12: Statistical data visualization

### Utilities
- **joblib** >= 1.3: Model serialization
- **tqdm** >= 4.66: Progress bars

## GPU Support (Optional)

### NVIDIA GPU with CUDA

If you have an NVIDIA GPU and want to accelerate training:

1. **Install CUDA Toolkit** (recommended: CUDA 11.8)
   - Download from: https://developer.nvidia.com/cuda-downloads

2. **Install cuDNN** (compatible with CUDA version)
   - Download from: https://developer.nvidia.com/cudnn

3. **Install TensorFlow with GPU support**:
   ```bash
   pip install tensorflow[and-cuda]>=2.12
   ```

4. **Verify GPU detection**:
   ```python
   import tensorflow as tf
   print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
   ```

### CPU-Only Setup

TensorFlow will automatically use CPU if no GPU is detected. No additional configuration needed.

**Training time comparison**:
- With GPU: 30-60 minutes
- CPU only: 2-4 hours

## Jupyter Notebook (Optional)

For running the demonstration notebooks:

```bash
pip install jupyter notebook
# or
pip install jupyterlab
```

## Testing Installation

Verify your environment is set up correctly:

```python
# test_installation.py
import sys
print(f"Python version: {sys.version}")

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

import shap
print(f"SHAP version: {shap.__version__}")

import pandas as pd
print(f"pandas version: {pd.__version__}")

print("\n✓ All core dependencies installed successfully!")
```

Run with:
```bash
python test_installation.py
```

## Known Issues and Solutions

### Issue: TensorFlow import error on Windows

**Error**: `ImportError: DLL load failed`

**Solution**: Install Microsoft Visual C++ Redistributable
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue: SHAP installation fails

**Error**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**: 
- Windows: Install Visual Studio Build Tools
- Linux: `sudo apt-get install build-essential`
- Mac: `xcode-select --install`

### Issue: NumPy/Pandas version conflicts

**Solution**: Use the specific versions in requirements.txt:
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: Out of Memory during training

**Solutions**:
1. Reduce dataset size with `--total` parameter (e.g., `--total 100000`)
2. Close other applications
3. Use a machine with more RAM (minimum 16GB recommended)

## Freezing Dependencies (for exact reproducibility)

After successful installation, freeze your environment:

```bash
pip freeze > requirements-frozen.txt
```

This captures exact versions for perfect reproducibility.

## Docker Alternative (Advanced)

For complete environment isolation, consider using Docker:

```dockerfile
# Dockerfile (example)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
```

Build and run:
```bash
docker build -t aegmm-nids .
docker run -it -v $(pwd):/app aegmm-nids
```

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- Activate venv: `venv\Scripts\activate`
- Path separators: Use `\` or raw strings `r"path\to\file"`

### Linux/Mac
- Use terminal
- Activate venv: `source venv/bin/activate`
- Path separators: Use `/`

### Apple Silicon (M1/M2/M3)
TensorFlow on Apple Silicon uses Metal acceleration:
```bash
pip install tensorflow-macos tensorflow-metal
```

Then install other dependencies:
```bash
pip install -r requirements.txt --no-deps
pip install numpy pandas scikit-learn matplotlib seaborn shap joblib tqdm
```

## Recommended Workflow

1. **Create environment** (Method 1 or 2 above)
2. **Install dependencies** from `requirements.txt`
3. **Test installation** with the test script
4. **Run quick verification** with `demo_notebooks/pretrained/test_pre_ae_gmm.ipynb`
5. **Full training** (optional) with `training/full_train.py`

---

**For issues**, check the Troubleshooting section in the main README.md or open an issue.
