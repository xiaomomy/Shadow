# Shadow Detection via Leave-One-Out Kernel Optimization

We present a faithful reproduction of the ICCV 2015 paper **“Leave-One-Out Kernel Optimization for Shadow Detection”** (Vicente et al.). Our implementation follows the paper’s pipeline—from region-based features to joint kernel learning and optional contextual refinement—and is developed as part of the **HKU MSc COMP7404** course project. Through this repository, we aim to make the original method accessible for reproduction and extension.

## 🌑 Background: Why Shadow Detection Matters

Shadows are ubiquitous in outdoor and indoor imagery. Detecting them accurately supports higher-level vision tasks such as object recognition, scene understanding and image editing. Yet shadows are challenging: they alter appearance without a simple intensity cue, interact with illumination and surface materials, and often blend gradually with lit regions. Region-based approaches that combine color, texture and context remain a principled way to address these ambiguities.

## 🔬 Methodology: LooKOP

We adopt a **region-centric** design. First, we **oversegment** each image with SLIC superpixels, then **merge** superpixels into larger regions via Mean-shift clustering in LAB space, as in the original paper. 
Each region is described by **CIELAB histograms** and **MR8 texton** statistics, enabling multi-channel kernel comparisons.

**Leave-One-Out Kernel Optimization (LooKOP)** is the core of our system. We use a **Least-Squares SVM (LSSVM)** with a **multi-kernel** structure over the L\*, a\*, b\*, and texton channels. Instead of hand-tuning kernel weights and bandwidths, we **jointly optimize** them by minimizing a **closed-form Leave-One-Out balanced error** under a beam-search schedule, so hyperparameters are chosen to generalize well on the training regions.

Finally, we optionally apply a **Markov Random Field (MRF)** stage that combines unary potentials from calibrated region scores with **pairwise affinities** derived from kernel similarity, refining predictions with **QPBO**-style energy minimization. Together, segmentation, LooKOP and MRF form the end-to-end story we reproduce.

## ⭐ Core Advantages

- ⚡ **Efficient LSSVM solver**: We solve for support values and bias via a linear system, avoiding iterative inner loops for the classifier itself.
- 🚀 **GPU acceleration**: Distance and kernel computations can leverage PyTorch CUDA for large region sets.
- 📐 **Closed-form LOO error**: We evaluate leave-one-out residuals without retraining from scratch, enabling fast kernel selection during beam search.
- 📥 **Dataset handling**: We integrate loading (and optional download) of the SBU-Shadow dataset for reproducible experiments.
- 🔄 **Full preprocessing stack**: SLIC, Mean-shift regions, paper-style LAB and texton features, Platt calibration, and MRF post-processing are wired into a single training script.

## 📂 Repository Structure

```text
.
├── baseline/               # Baseline methods (Unary/MK-SVM kernels, region CNN)
├── data/                   # Dataset loader; local SBU data under data/sbu/ (git-ignored)
├── models/                 # LSSVM, multi-kernel, LOO beam search, distances, Platt, MRF
├── preprocessing/          # SLIC, Mean-shift regions, features, MR8 textons
├── utils/                  # Shared utilities
├── output/                 # Training outputs and visualizations (git-ignored by default)
├── train_sbu.py            # Main entry: SBU training, evaluation, and optional baselines
├── config.py               # Auxiliary configuration (optional)
├── requirements.txt
└── README.md
```

## 🛠️ Reproduction Guide

### 1. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xiaomomy/Shadow.git
   cd Shadow
   ```

2. **Install dependencies** (Python 3.8+; CUDA optional but recommended for speed):
   ```bash
   pip install -r requirements.txt
   ```

### 2. Running the Full Pipeline

We consolidate the formal experiment in **`train_sbu.py`**. After placing or downloading the SBU-Shadow data under the expected paths (see `data/dataset_loader.py`), run:

```bash
python train_sbu.py
```

## 📊 Reproduction Results

### Reproduction Setup
We use 60 images (about 15,000 regions) as the training set for LookOP and other baseline appoaches, and 
evaluate their performances on the SBU hold-out set (60 images). 
All metrics are computed at the pixel level.
Hyperparameters such as the number of training images, hold-out size, SLIC superpixel count, Mean-shift bandwidth, optimization iterations, and the choice of method (`lookop`, `unary_svm`, `mk_svm`, `cnn`) are set in the **`CONFIG`** dictionary at the top of `train_sbu.py`, most of which are consistent the original implementation in the original paper. 
Additionally, we cache heavy artifacts (e.g., texton dictionary and region features) under `output/cache/` and write models and visualizations under `output/sbu_formal/` by default. 
Adjust paths in `CONFIG` if your layout differs.

### Quantitative Results

We report **pixel-level** false positive rate (FPR), false negative rate (FNR), and balanced error rate (BER) on our SBU hold-out evaluation. Metrics are computed by expanding region predictions to pixels against ground-truth masks.

| Method | Pixel FPR (%) | Pixel FNR (%) | Pixel BER (%) |
| :--- | :---: | :---: | :---: |
| Unary SVM | 4.58 | 37.00 | 20.79 |
| MK-SVM | 2.73 | 54.04 | 28.39 |
| ConvNet (CNN) | 7.84 | 36.52 | 22.18 |
| **LooKOP (Ours)** | 6.59 | **15.75** | **11.17** |

### Visual Gallery


## 🙏 Acknowledgements
This repository builds entirely on the methodology and experimental spirit of Vicente et al. We are grateful to the authors for their clear formulation of leave-one-out kernel optimization for shadow detection and for the foundations it provides for reproducible research.

If you use ideas or code derived from this reproduction, please cite the original paper:

```bibtex
@inproceedings{vicente2015leave,
  title={Leave-one-out kernel optimization for shadow detection},
  author={Vicente, Tom{\'a}s F Yery and Hou, Le and Samaras, Dimitris and Hoai, Minh and Nguyen, Minh-Hoai},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages={3388--3396},
  year={2015}
}
```
