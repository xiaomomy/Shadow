# Shadow Detection via Leave-One-Out Kernel Optimization

This repository implements a shadow detection system based on the paper **"Leave-One-Out Kernel Optimization for Shadow Detection"** (Vicente et al., ICCV 2015). It leverages Kernel Least-Squares Support Vector Machines (LSSVM) with efficient Leave-One-Out (LOO) cross-validation for hyperparameter tuning and multi-kernel learning.

## 🚀 Key Features

- **Efficient LSSVM Solver**: Direct analytical solution for classifier parameters ($\alpha, b$) using linear system solvers.
- **GPU Acceleration**: High-performance matrix operations and kernel computations using PyTorch CUDA backend.
- **Closed-form LOO Error**: Compute Leave-One-Out cross-validation error in a single pass without retraining, enabling rapid model selection.
- **Automatic Dataset Management**: Integrated logic to automatically download and extract the SBU-Shadow dataset.
- **Regional Analysis**: Advanced preprocessing pipeline including SLIC superpixels, Mean-Shift region merging, and MR8-based Texton features.

## 📂 Project Architecture

```text
.
├── data/                   # Data management module
│   ├── dataset_loader.py   # SBU dataset loader with auto-download logic
│   └── sbu/                # Local storage for SBU-Shadow dataset (ignored by git)
├── models/                 # Machine learning models
│   ├── lssvm.py            # Core LSSVM implementation (GPU-accelerated)
│   ├── kernels.py          # RBF and Multi-Kernel definitions
│   ├── distances.py        # EMD and Chi-Square distance metrics
│   └── platt_scaling.py    # Probability calibration
├── preprocessing/          # Image processing pipeline
│   ├── superpixel.py       # SLIC superpixel segmentation
│   ├── region.py           # Mean-Shift region generation
│   ├── features.py         # LAB color and texture feature extraction
│   └── texton.py           # MR8 filter bank and GPU K-Means dictionary
├── output/                 # Training results and visualizations
├── train_quick.py          # Entry point for training and validation
└── requirements.txt        # Project dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xiaomomy/Shadow.git
   cd Shadow
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ and CUDA (optional but recommended) installed.
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Quick Start

To start the training pipeline (which includes automatic data preparation, feature extraction, and model optimization):

```bash
python train_quick.py
```

### What happens during execution?
1. **Data**: The script checks for the SBU dataset. If not found, it downloads it from the official Stony Brook source.
2. **Features**: It builds a Texton dictionary (GPU-accelerated) and extracts regional LAB/Texture features.
3. **Training**: It solves the LSSVM linear system on the GPU to find optimal support values.
4. **Optimization**: It uses the closed-form LOO formula to find the best `sigma_multiplier` for the kernels.
5. **Output**: Results (BER, Accuracy) and the trained model are saved in the `output/` directory.

## 🧪 Mathematical Foundation

The model solves the following linear system to find the support values $\boldsymbol{\alpha}$ and bias $b$:

$$\begin{bmatrix} 0 & \mathbf{1}^T \\ \mathbf{1} & \mathbf{K} + \gamma^{-1}\mathbf{I} \end{bmatrix} \begin{bmatrix} b \\ \boldsymbol{\alpha} \end{bmatrix} = \begin{bmatrix} 0 \\ \mathbf{y} \end{bmatrix}$$

The **Leave-One-Out residual** for sample $i$ is computed efficiently as:
$$e_i^{LOO} = \frac{\alpha_i}{(\mathbf{M}^{-1})_{i+1, i+1}}$$

## 👥 Authors

- **Shadow Detection Project Team** (HKU MSc CIML)

---
*This project is for educational purposes as part of a course project.*
