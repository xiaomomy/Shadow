"""
Machine Learning Models for Shadow Detection.

This module implements the core classification models:
    - Kernel Least-Squares SVM (LSSVM)
    - Multi-Kernel Learning with Extended Gaussian Kernels
    - Leave-One-Out Optimization
    - Platt Scaling for probability calibration

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
"""

from .kernels import (
    RBFKernel, 
    LinearKernel, 
    PolynomialKernel, 
    MultiKernel,
    ExtendedGaussianKernel,
    ShadowDetectionMultiKernel,
    create_default_kernels_for_shadow_detection
)
from .lssvm import LSSVM
from .loo_optimizer import LOOKernelOptimizer
from .distances import (
    chi_square_distance,
    chi_square_distance_matrix,
    emd_1d,
    emd_1d_matrix,
    extended_gaussian_kernel,
    compute_mean_distance,
    compute_sigma_grid,
    compute_weight_grid,
    DistanceComputer
)
from .platt_scaling import (
    PlattScaler,
    balanced_error_rate,
    compute_loo_balanced_error,
    false_positive_rate,
    false_negative_rate
)
from .mrf import (
    MRFShadowDetector,
    DisparityClassifier,
    compute_region_areas,
    compute_region_adjacency,
    compute_region_mean_rgb
)

__all__ = [
    # Kernels
    'RBFKernel', 
    'LinearKernel', 
    'PolynomialKernel',
    'MultiKernel',
    'ExtendedGaussianKernel',
    'ShadowDetectionMultiKernel',
    'create_default_kernels_for_shadow_detection',
    # LSSVM
    'LSSVM',
    # LOO Optimization
    'LOOKernelOptimizer',
    # Distance metrics
    'chi_square_distance',
    'chi_square_distance_matrix',
    'emd_1d',
    'emd_1d_matrix',
    'extended_gaussian_kernel',
    'compute_mean_distance',
    'compute_sigma_grid',
    'compute_weight_grid',
    'DistanceComputer',
    # Platt Scaling and Error Metrics
    'PlattScaler',
    'balanced_error_rate',
    'compute_loo_balanced_error',
    'false_positive_rate',
    'false_negative_rate',
    # MRF
    'MRFShadowDetector',
    'DisparityClassifier',
    'compute_region_areas',
    'compute_region_adjacency',
    'compute_region_mean_rgb'
]

