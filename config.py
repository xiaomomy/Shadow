"""
Configuration parameters for shadow detection preprocessing.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015

Author: [Your Name]
Date: 2026
"""

import os

# =============================================================================
# Path Configuration
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# =============================================================================
# Superpixel Segmentation Parameters
# =============================================================================
# SLIC (Simple Linear Iterative Clustering) parameters
# Reference: Achanta et al., "SLIC Superpixels Compared to State-of-the-art 
#            Superpixel Methods", TPAMI 2012
SUPERPIXEL_CONFIG = {
    'n_segments': 500,          # Target number of superpixels (approx. 500-1000 in paper)
    'compactness': 10.0,        # Balance between color and spatial proximity
    'sigma': 1.0,               # Width of Gaussian smoothing kernel
    'convert2lab': True,        # Convert to LAB color space for better perceptual uniformity
    'enforce_connectivity': True,  # Enforce connectivity of segments
    'min_size_factor': 0.25,    # Minimum segment size as fraction of average
    'max_size_factor': 3.0,     # Maximum segment size as fraction of average
}

# =============================================================================
# Region Generation Parameters (Mean-shift Clustering)
# =============================================================================
# Paper Section 3: "apply Mean-shift clustering [4] and merge superpixels 
# in the same cluster into a larger region."
# Reference: [4] Comaniciu & Meer, "Mean Shift: A Robust Approach Toward 
#            Feature Space Analysis", TPAMI 2002
REGION_CONFIG = {
    # Mean-shift bandwidth: controls cluster size
    # None = auto-estimate using quantile
    # Paper does NOT specify exact bandwidth value
    'bandwidth': None,
    
    # Quantile for bandwidth estimation (smaller = more regions)
    # This is the key parameter that affects region granularity
    'quantile': 0.2,
    
    # Whether to use spatial position in clustering
    # Paper [4] describes joint spatial-color, but exact setting unclear
    'use_spatial': False,
    
    # Spatial weight (only used if use_spatial=True)
    'spatial_weight': 0.1,
}

# =============================================================================
# Feature Extraction Parameters
# =============================================================================
# Based on paper: "chromatic, intensity, and texture properties"

# Chromatic features (in LAB color space)
# Paper Section 3.1: "we compute a 21-bin histogram for each of the components (L*,a*,b*)"
CHROMATIC_CONFIG = {
    'color_space': 'LAB',       # LAB color space for perceptual uniformity
    'n_bins_histogram': 21,     # Paper explicitly specifies 21-bin histogram
    'compute_mean': True,       # Compute mean of each channel
    'compute_std': True,        # Compute standard deviation
    'compute_histogram': True,  # Compute color histogram
}

# Intensity features (grayscale)
INTENSITY_CONFIG = {
    'n_bins_histogram': 32,     # Number of bins for intensity histogram
    'compute_mean': True,       # Mean intensity
    'compute_std': True,        # Standard deviation
    'compute_percentiles': True,  # Compute intensity percentiles
    'percentile_values': [5, 25, 50, 75, 95],  # Percentile values to compute
}

# Texture features (Texton)
# Paper Section 3.1: "To represent texture, we compute a 128-bin texton histogram. 
# We run the full MR8 filter set [33] in the whole dataset and cluster the filter 
# responses into 128 textons using k-means."
# Reference: Varma & Zisserman, "A Statistical Approach to Texture Classification 
#            from Single Images", IJCV 2005
TEXTURE_CONFIG = {
    'method': 'texton',         # Paper uses texton instead of LBP
    'n_textons': 128,           # Paper explicitly specifies 128-bin texton histogram
    'filter_bank': 'MR8',       # Paper uses MR8 filter bank
    # Alternative LBP configuration (for quick testing)
    'lbp_radius': 1,
    'lbp_n_points': 8,
    'lbp_method': 'uniform',
    'n_bins_histogram': 10,
}

# Edge/Boundary features
EDGE_CONFIG = {
    'compute_gradient': True,   # Compute gradient magnitude statistics
    'compute_boundary_strength': True,  # Boundary strength with neighbors
}

# =============================================================================
# Combined Feature Vector Configuration
# =============================================================================
# Feature normalization
FEATURE_CONFIG = {
    'normalize': True,          # L2 normalize features
    'standardize': True,        # Zero mean, unit variance standardization
}

# =============================================================================
# LSSVM Configuration
# =============================================================================
# Reference: Paper Section 2.2 and 3
LSSVM_CONFIG = {
    'gamma': 1.0,  # Regularization parameter (γ in paper)
    # Note: γ in LSSVM controls the trade-off between fitting and regularization
}

# =============================================================================
# LOO Optimization Configuration
# =============================================================================
# Reference: Paper Section 3.2 "Optimization grid details"
OPTIMIZATION_CONFIG = {
    # Number of iterations
    'max_iterations': 500,       # Paper: "we perform 500 iterations"
    'stagnation_threshold': 25,  # Paper: "after 25 consecutive iterations"
    
    # Sigma grid: σ_l ∈ {s·μ_l | s ∈ {1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8}}
    'sigma_multipliers': [1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8],
    
    # Weight grid: w_l ∈ {s/40 | s ∈ {1, 2, ..., 10}}
    'weight_values': [s/40 for s in range(1, 11)],
    
    # Kernel channels
    'channels': ['L', 'a', 'b', 't'],
}

# =============================================================================
# Visualization Parameters
# =============================================================================
VIS_CONFIG = {
    'save_superpixel_overlay': True,
    'save_region_overlay': True,
    'boundary_color': (255, 255, 0),  # Yellow boundaries
    'boundary_thickness': 1,
}

