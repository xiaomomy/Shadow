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
├── train_quick.py          # Entry point for quick validation
├── train_sbu.py            # Entry point for formal SBU training (Full Paper Reproduction)
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

## 🏃 Getting Started

### 1. Quick Validation
To start the quick training pipeline (using fewer images for rapid validation):
```bash
python train_quick.py
```

### 2. Formal SBU Training (Full Reproduction)
To reproduce the results of the ICCV 2015 paper on the full SBU dataset:
```bash
python train_sbu.py
```

### 📊 Performance Comparison
Experimental results on the SBU hold-out set (200 images). Metrics are computed at the pixel level.

| Method | Pixel FPR (%) | Pixel FNR (%) | Pixel BER (%) |
| :--- | :---: | :---: | :---: |
| Unary SVM | 2.78 | 49.10 | 25.94 |
| MK-SVM | 2.59 | 53.54 | 28.07 |
| ConvNet (CNN) | 4.15 | 43.05 | 23.60 |
| **LooKOP (Ours)** | - | - | - |

### What happens during formal training?
1. **Full Dataset**: The script uses the entire SBU-Shadow dataset (approx. 4,000 training and 638 test images).
2. **Joint Optimization (Phase 3)**: It uses the `PaperBeamSearchOptimizer` to perform a 9-dimensional grid search (4 weights, 4 sigmas, 1 gamma) for 500 iterations.
3. **LOO Objective**: It minimizes the Leave-One-Out Balanced Error Rate (BER) using the closed-form residual formula.
4. **Caching**: Feature extraction and Texton dictionary building are cached for performance.
5. **Output**: The best model and detailed results are saved in `output/sbu_formal/`.

## 🧪 Mathematical Foundation

The model solves the following linear system to find the support values $\boldsymbol{\alpha}$ and bias $b$:

$$\begin{bmatrix} 0 & \mathbf{1}^T \\ \mathbf{1} & \mathbf{K} + \gamma^{-1}\mathbf{I} \end{bmatrix} \begin{bmatrix} b \\ \boldsymbol{\alpha} \end{bmatrix} = \begin{bmatrix} 0 \\ \mathbf{y} \end{bmatrix}$$

The **Leave-One-Out residual** for sample $i$ is computed efficiently as:
$$e_i^{LOO} = \frac{\alpha_i}{(\mathbf{M}^{-1})_{i+1, i+1}}$$

## 📜 Citation

If you find this work useful for your research, please cite the original paper:

```bibtex
@inproceedings{vicente2015leave,
  title={Leave-one-out kernel optimization for shadow detection},
  author={Vicente, Tom{\'a}s F Yery and Hou, Le and Samaras, Dimitris and Hoai, Minh and Nguyen, Minh-Hoai},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages={3388--3396},
  year={2015}
}
```

---
*This project is for educational purposes as part of a course project at HKU (MSc CIML).*
