"""
Distance Metrics for Shadow Detection Kernels.

This module implements the distance metrics used in the paper's kernel functions.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 3.1:
    "To compare textures between regions we use the χ² distance between their 
    texton histograms. For color histograms, it is more appropriate to use the 
    Earth Mover's Distance (EMD) [25] because neighboring bins in the L*,a*,b* 
    histograms represent proximate values and their ground distance is uniform."

Mathematical Background:
========================

1. Chi-Square (χ²) Distance:
   D_χ²(p, q) = Σ_i (p_i - q_i)² / (p_i + q_i)
   
   Used for comparing texton histograms because bins are not ordered.

2. Earth Mover's Distance (EMD):
   The minimum cost to transform distribution p into q, where the cost
   is the amount of "earth" moved times the ground distance.
   
   For 1D histograms with uniform ground distance, EMD simplifies to
   the cumulative distribution function (CDF) difference.
   
   Used for color histograms (L*, a*, b*) because bins represent 
   ordered values with uniform spacing.

Author: [Your Name]
Date: 2026
"""

import numpy as np
from typing import Optional, Callable
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def chi_square_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Chi-Square distance between two histograms.
    
    D_χ²(p, q) = Σ_i (p_i - q_i)² / (p_i + q_i + ε)
    
    This is used for texton histogram comparison in the paper.
    
    Reference:
        Paper Section 3.1: "To compare textures between regions we use the 
        χ² distance between their texton histograms."
    
    Args:
        p: First histogram (normalized, sums to 1)
        q: Second histogram (normalized, sums to 1)
        
    Returns:
        Chi-square distance value (non-negative)
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    
    # Ensure non-negative values
    p = np.maximum(p, 0)
    q = np.maximum(q, 0)
    
    # Chi-square formula
    denominator = p + q + eps
    distance = np.sum((p - q) ** 2 / denominator)
    
    return distance


def chi_square_distance_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute pairwise Chi-Square distance matrix.
    
    Args:
        X: Feature matrix (n_samples_X, n_features), each row is a histogram
        Y: Feature matrix (n_samples_Y, n_features), or None for X vs X
        
    Returns:
        Distance matrix (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    
    # Ensure non-negative
    X = np.maximum(X, 0)
    Y = np.maximum(Y, 0)
    
    # Vectorized computation
    eps = 1e-10
    D = np.zeros((n_X, n_Y))
    
    for i in range(n_X):
        # Broadcast computation
        diff_sq = (X[i:i+1, :] - Y) ** 2  # (n_Y, n_features)
        sum_xy = X[i:i+1, :] + Y + eps     # (n_Y, n_features)
        D[i, :] = np.sum(diff_sq / sum_xy, axis=1)
    
    return D


def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Earth Mover's Distance (EMD) for 1D histograms.
    
    For 1D distributions with uniform ground distance, EMD equals the
    L1 distance between cumulative distribution functions:
    
        EMD(p, q) = Σ_i |CDF_p(i) - CDF_q(i)|
    
    This is also known as the Wasserstein-1 distance.
    
    Reference:
        Paper Section 3.1: "For color histograms, it is more appropriate to use 
        the Earth Mover's Distance (EMD) because neighboring bins in the L*,a*,b* 
        histograms represent proximate values and their ground distance is uniform."
    
    Args:
        p: First histogram (should sum to 1)
        q: Second histogram (should sum to 1)
        
    Returns:
        EMD distance value (non-negative)
    """
    # Normalize to ensure they sum to 1
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p_sum = np.sum(p)
    q_sum = np.sum(q)
    
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum
    
    # Compute CDFs
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    
    # EMD = L1 distance between CDFs
    emd = np.sum(np.abs(cdf_p - cdf_q))
    
    return emd


def emd_1d_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute pairwise EMD distance matrix for 1D histograms.
    
    Args:
        X: Feature matrix (n_samples_X, n_bins)
        Y: Feature matrix (n_samples_Y, n_bins), or None
        
    Returns:
        Distance matrix (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    
    D = np.zeros((n_X, n_Y))
    
    # Normalize histograms
    X_norm = X / (np.sum(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.sum(Y, axis=1, keepdims=True) + 1e-10)
    
    # Compute CDFs
    X_cdf = np.cumsum(X_norm, axis=1)
    Y_cdf = np.cumsum(Y_norm, axis=1)
    
    # Vectorized EMD computation
    for i in range(n_X):
        D[i, :] = np.sum(np.abs(X_cdf[i:i+1, :] - Y_cdf), axis=1)
    
    return D


def euclidean_distance_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.
    
    This is NOT used in the paper's kernel, but provided for comparison.
    
    Args:
        X: Feature matrix (n_samples_X, n_features)
        Y: Feature matrix (n_samples_Y, n_features), or None
        
    Returns:
        Distance matrix (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    
    # ||x - y||² = ||x||² + ||y||² - 2<x,y>
    X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sqnorm = np.sum(Y ** 2, axis=1, keepdims=True)
    
    sq_dist = X_sqnorm + Y_sqnorm.T - 2 * np.dot(X, Y.T)
    sq_dist = np.maximum(sq_dist, 0)  # Numerical stability
    
    return np.sqrt(sq_dist)


def compute_mean_distance(D: np.ndarray, exclude_diagonal: bool = True) -> float:
    """
    Compute the mean distance from a distance matrix.
    
    This is used in the paper for setting the σ parameter:
    "The possible discrete values of σ_l are {sμ_l | s ∈ {1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8}}"
    where μ_l is the mean distance.
    
    Reference:
        Paper Section 3.2: "using the mean of the pairwise distances as the 
        scaling factor [35]. If {x_1,...,x_n} is the set of training examples, 
        the mean distance is computed as μ_l = 1/((n-1)n) Σ_{i≠j} D_l(x_i, x_j)"
    
    Args:
        D: Distance matrix (n_samples, n_samples)
        exclude_diagonal: Whether to exclude self-distances (diagonal)
        
    Returns:
        Mean distance value
    """
    n = D.shape[0]
    
    if exclude_diagonal:
        # Exclude diagonal elements
        mask = ~np.eye(n, dtype=bool)
        mean_dist = np.mean(D[mask])
    else:
        mean_dist = np.mean(D)
    
    return mean_dist


class DistanceComputer:
    """
    Unified interface for computing distance matrices.
    
    This class provides a consistent interface for computing different
    types of distance matrices used in the paper's kernel functions.
    """
    
    def __init__(self, distance_type: str = 'emd'):
        """
        Initialize distance computer.
        
        Args:
            distance_type: Type of distance metric
                - 'emd': Earth Mover's Distance (for color histograms)
                - 'chi2': Chi-Square distance (for texton histograms)
                - 'euclidean': Euclidean distance (not used in paper)
        """
        self.distance_type = distance_type
        
        # Map to distance functions
        self._distance_functions = {
            'emd': emd_1d_matrix,
            'chi2': chi_square_distance_matrix,
            'euclidean': euclidean_distance_matrix
        }
        
        if distance_type not in self._distance_functions:
            raise ValueError(f"Unknown distance type: {distance_type}")
        
        self._compute_func = self._distance_functions[distance_type]
    
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute distance matrix.
        
        Args:
            X: Feature matrix (n_samples_X, n_features)
            Y: Feature matrix (n_samples_Y, n_features), or None
            
        Returns:
            Distance matrix
        """
        return self._compute_func(X, Y)
    
    def compute_mean(self, X: np.ndarray) -> float:
        """
        Compute mean pairwise distance.
        
        Args:
            X: Feature matrix
            
        Returns:
            Mean distance
        """
        D = self.compute(X)
        return compute_mean_distance(D, exclude_diagonal=True)


# =============================================================================
# Extended Gaussian Kernel using custom distances
# =============================================================================

def extended_gaussian_kernel(
    D: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    Compute Extended Gaussian Kernel from distance matrix.
    
    K(x, y) = exp(-1/σ · D(x, y))
    
    This is the kernel form used in the paper, NOT the standard RBF kernel.
    
    Reference:
        Paper Section 3.1 (Equation 5):
        K(x,y) = Σ_{l∈{L,a,b,t}} w_l exp(-1/σ_l D_l(x,y))
    
    Note:
        The standard RBF kernel is K = exp(-γ ||x-y||²)
        The paper's kernel is K = exp(-D/σ) where D can be any distance metric.
    
    Args:
        D: Distance matrix (precomputed)
        sigma: Scaling factor (σ in the paper)
        
    Returns:
        Kernel matrix
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    
    # K = exp(-D / σ)
    K = np.exp(-D / sigma)
    
    return K


def compute_sigma_grid(mean_distance: float) -> np.ndarray:
    """
    Compute the grid of possible σ values.
    
    Reference:
        Paper Section 3.2: "The possible discrete values of σ_l are 
        {sμ_l | s ∈ {1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8}}"
    
    Args:
        mean_distance: Mean pairwise distance (μ_l)
        
    Returns:
        Array of σ values
    """
    multipliers = np.array([1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8])
    sigma_values = multipliers * mean_distance
    
    return sigma_values


def compute_weight_grid() -> np.ndarray:
    """
    Compute the grid of possible kernel weight values.
    
    Reference:
        Paper Section 3.2: "For the weight w_l of a base kernel, we use 
        {s/40 | s ∈ {1,...,10}} as the set of possible values."
    
    Returns:
        Array of weight values
    """
    s_values = np.arange(1, 11)  # {1, 2, ..., 10}
    weight_values = s_values / 40.0
    
    return weight_values

