# Installation & Setup Guide

## Prerequisites Check

### Python Version
```bash
python --version
# Should be 3.8 or higher
```

### Pip Version
```bash
pip --version
# Update if needed: pip install --upgrade pip
```

## Option 1: Conda Environment (Recommended for Clean Install)

### Windows/Mac/Linux
```bash
# Create new environment
conda create -n deeplearning python=3.10

# Activate
conda activate deeplearning

# Install TensorFlow (this installs keras, numpy, etc.)
conda install tensorflow

# Install additional packages
pip install matplotlib scikit-learn jupyter
```

## Option 2: Pip Install (Quick)

### Windows
```bash
pip install tensorflow keras numpy matplotlib
```

### Mac (Intel)
```bash
pip install tensorflow keras numpy matplotlib
```

### Mac (Apple Silicon M1/M2)
```bash
# Use conda for M1/M2 compatibility
conda install tensorflow
pip install keras numpy matplotlib
```

### Linux
```bash
pip install tensorflow keras numpy matplotlib
```

## Option 3: Install with GPU Support

### GPU Requirements
- NVIDIA GPU (GeForce, Tesla, RTX series)
- CUDA 11.8+ installed
- cuDNN 8.6+

### GPU Installation
```bash
# Complete GPU setup (automatic with conda)
conda install tensorflow-gpu

# Or manual pip
pip install tensorflow[and-cuda]

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Verify Installation

### Test Script
```python
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"NumPy version: {np.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs found: {len(gpus)}")
if gpus:
    print(f"GPU: {gpus[0]}")

print("✓ All packages installed successfully!")
```

### Run Test
```bash
python verify_install.py
```

### Expected Output
```
TensorFlow version: 2.13.0
Keras version: 2.13.0
NumPy version: 1.24.3
GPUs found: 0  # or 1+ if GPU available
✓ All packages installed successfully!
```

## Troubleshooting Installation

### Error: "No module named tensorflow"

**Solution 1**: Reinstall fresh
```bash
pip uninstall tensorflow -y
pip install tensorflow --upgrade
```

**Solution 2**: Use conda
```bash
conda install tensorflow -c conda-forge
```

### Error: "ModuleNotFoundError: No module named 'keras'"

**Note**: TensorFlow 2.x includes Keras by default

```bash
# Verify TensorFlow includes Keras
python -c "from tensorflow import keras; print(keras.__version__)"

# If not, install explicitly
pip install keras
```

### Error: "CUDA not found" (GPU install)

**Solution**: Install CUDA and cuDNN
```bash
# Check NVIDIA drivers
nvidia-smi

# Install CUDA: https://developer.nvidia.com/cuda-11-8-0-download-archive
# Install cuDNN: https://developer.nvidia.com/cudnn

# Then reinstall TensorFlow
pip install tensorflow[and-cuda] --upgrade
```

### Error: Out of Memory (OOM)

**Solution**: Use CPU instead
```python
import tensorflow as tf
# Disable GPU
tf.config.set_visible_devices([], 'GPU')
```

## Recommended Setup for Lab

### Development Environment
```bash
# Create project directory
mkdir deep_learning_lab
cd deep_learning_lab

# Create Python environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install packages
pip install tensorflow keras numpy matplotlib jupyter notebook
```

### With Jupyter Notebook
```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter
jupyter notebook

# In browser, create new Python notebook
# Run tasks interactively
```

## Dependency Versions

**Minimum Working Setup**:
```
Python >= 3.8
TensorFlow >= 2.10
NumPy >= 1.21
Matplotlib >= 3.5
```

**Recommended Setup**:
```
Python 3.10
TensorFlow 2.13
NumPy 1.24
Matplotlib 3.7
Jupyter 1.0
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8+ GB |
| Disk | 2 GB | 5 GB |
| CPU | Quad-core | Octa-core |
| GPU | None | NVIDIA RTX 2060+ |
| OS | Windows/Mac/Linux | Ubuntu 20.04 |

## Quick Setup Script

### setup.sh (Linux/Mac)
```bash
#!/bin/bash
python3 -m venv deeplearning_env
source deeplearning_env/bin/activate
pip install --upgrade pip
pip install tensorflow keras numpy matplotlib jupyter
echo "✓ Setup complete! Activate with: source deeplearning_env/bin/activate"
```

### setup.bat (Windows)
```batch
python -m venv deeplearning_env
call deeplearning_env\Scripts\activate
pip install --upgrade pip
pip install tensorflow keras numpy matplotlib jupyter
echo ✓ Setup complete! Activate with: deeplearning_env\Scripts\activate
```

## Running Tasks After Installation

```bash
# Activate environment (if using venv)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate   # Windows

# Navigate to lab directory
cd deep_learning_lab

# Run tasks
python task1_mnist.py
python task2_fashion_mnist.py
python task3_cifar10.py
```

## Additional Resources

### TensorFlow Documentation
- https://www.tensorflow.org/install
- https://www.tensorflow.org/guide

### Keras API
- https://keras.io/

### GPU Setup
- https://www.tensorflow.org/install/gpu
- https://docs.nvidia.com/cuda/

### Conda Environments
- https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

---

**Need Help?** Check the README.md or official TensorFlow documentation.

