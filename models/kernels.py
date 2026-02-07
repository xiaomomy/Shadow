"""
Kernel Functions for LSSVM.

This module implements various kernel functions used in the shadow detection pipeline.
The paper uses multiple kernels that are combined for different feature types:
    - Chromatic features (L*, a*, b* color channels)
    - Texture features (texton histogram)

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 3.1 (Equation 5):
    K(x,y) = Σ_{l∈{L,a,b,t}} w_l exp(-1/σ_l D_l(x,y))
    
    where:
    - D_L, D_a, D_b are EMD distances for L*, a*, b* histograms
    - D_t is χ² distance for texton histograms
    - w_l are kernel weights (non-negative, sum to 1)
    - σ_l are scaling factors

Mathematical Background:
    The paper uses Extended Gaussian Kernels with custom distance metrics:
        K(x, x') = exp(-D(x, x') / σ)
    
    This is different from standard RBF kernel:
        K_RBF(x, x') = exp(-γ ||x - x'||²)
    
    For multi-kernel learning, the combined kernel is:
        K(x, x') = Σ_m w_m · K_m(x, x')
    where w_m >= 0 are the kernel weights.

Author: [Your Name]
Date: 2026
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union
from scipy.spatial.distance import cdist

from .distances import (
    emd_1d_matrix, chi_square_distance_matrix, 
    extended_gaussian_kernel, compute_mean_distance
)


class BaseKernel(ABC):
    """
    Abstract base class for kernel functions.
    
    All kernel implementations must inherit from this class and implement
    the compute() method.
    """
    
    @abstractmethod
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the kernel matrix.
        
        Args:
            X: First set of samples, shape (n_samples_X, n_features)
            Y: Second set of samples, shape (n_samples_Y, n_features)
               If None, compute K(X, X)
               
        Returns:
            Kernel matrix of shape (n_samples_X, n_samples_Y) or (n_samples_X, n_samples_X)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """Return kernel parameters as a dictionary."""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> None:
        """Set kernel parameters."""
        pass


class RBFKernel(BaseKernel):
    """
    Radial Basis Function (RBF) / Gaussian Kernel.
    
    Mathematical Definition:
        K(x, x') = exp(-γ ||x - x'||²)
        
    where γ = 1/(2σ²) is the kernel bandwidth parameter.
    
    This is the most commonly used kernel for LSSVM due to its ability
    to model complex non-linear relationships.
    
    Attributes:
        gamma: Kernel bandwidth parameter (default: 1/n_features)
    """
    
    def __init__(self, gamma: Optional[float] = None):
        """
        Initialize RBF kernel.
        
        Args:
            gamma: Kernel bandwidth. If None, will be set to 1/n_features
                   when compute() is first called.
        """
        self.gamma = gamma
        self._auto_gamma = gamma is None
    
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute RBF kernel matrix.
        
        K(x_i, x_j) = exp(-γ ||x_i - x_j||²)
        
        Args:
            X: First set of samples (n_samples_X, n_features)
            Y: Second set of samples (n_samples_Y, n_features), or None
            
        Returns:
            Kernel matrix K
        """
        if Y is None:
            Y = X
        
        # Auto-set gamma based on feature dimension
        if self._auto_gamma or self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        
        # Compute squared Euclidean distances
        # ||x - y||² = ||x||² + ||y||² - 2<x,y>
        X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)
        Y_sqnorm = np.sum(Y ** 2, axis=1, keepdims=True)
        sq_dist = X_sqnorm + Y_sqnorm.T - 2 * np.dot(X, Y.T)
        
        # Numerical stability: ensure non-negative distances
        sq_dist = np.maximum(sq_dist, 0)
        
        # Apply RBF formula
        K = np.exp(-self.gamma * sq_dist)
        
        return K
    
    def get_params(self) -> Dict:
        return {'gamma': self.gamma}
    
    def set_params(self, **params) -> None:
        if 'gamma' in params:
            self.gamma = params['gamma']
            self._auto_gamma = params['gamma'] is None


class LinearKernel(BaseKernel):
    """
    Linear Kernel.
    
    Mathematical Definition:
        K(x, x') = <x, x'> = x^T x'
        
    This is the simplest kernel, equivalent to no feature mapping.
    """
    
    def __init__(self):
        pass
    
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute linear kernel matrix.
        
        K(x_i, x_j) = x_i^T x_j
        
        Args:
            X: First set of samples (n_samples_X, n_features)
            Y: Second set of samples (n_samples_Y, n_features), or None
            
        Returns:
            Kernel matrix K
        """
        if Y is None:
            Y = X
        
        return np.dot(X, Y.T)
    
    def get_params(self) -> Dict:
        return {}
    
    def set_params(self, **params) -> None:
        pass


class PolynomialKernel(BaseKernel):
    """
    Polynomial Kernel.
    
    Mathematical Definition:
        K(x, x') = (γ <x, x'> + c)^d
        
    where:
        - γ: scale factor
        - c: coefficient (coef0)
        - d: polynomial degree
    """
    
    def __init__(
        self, 
        degree: int = 3, 
        gamma: Optional[float] = None,
        coef0: float = 1.0
    ):
        """
        Initialize polynomial kernel.
        
        Args:
            degree: Polynomial degree
            gamma: Scale factor (default: 1/n_features)
            coef0: Independent coefficient
        """
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self._auto_gamma = gamma is None
    
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute polynomial kernel matrix.
        
        K(x_i, x_j) = (γ x_i^T x_j + c)^d
        """
        if Y is None:
            Y = X
        
        if self._auto_gamma or self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        
        linear = np.dot(X, Y.T)
        K = (self.gamma * linear + self.coef0) ** self.degree
        
        return K
    
    def get_params(self) -> Dict:
        return {
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0
        }
    
    def set_params(self, **params) -> None:
        if 'degree' in params:
            self.degree = params['degree']
        if 'gamma' in params:
            self.gamma = params['gamma']
            self._auto_gamma = params['gamma'] is None
        if 'coef0' in params:
            self.coef0 = params['coef0']


class ExtendedGaussianKernel(BaseKernel):
    """
    Extended Gaussian Kernel with custom distance metric.
    
    This is the kernel form used in the paper, NOT the standard RBF kernel.
    
    Mathematical Definition:
        K(x, x') = exp(-D(x, x') / σ)
        
    where D is a distance metric (EMD or χ²) and σ is the scaling factor.
    
    Reference:
        Paper Section 3.1 (Equation 5):
        K(x,y) = Σ_{l∈{L,a,b,t}} w_l exp(-1/σ_l D_l(x,y))
        
        "The function exp(-1/σ D(x,y)) is called the extended Gaussian kernel [16,31,35], 
        and the kernel K is the linear combination of extended Gaussian kernels."
    
    Note:
        This is DIFFERENT from standard RBF kernel:
        - RBF: K = exp(-γ ||x-y||²) where γ is the bandwidth
        - Extended Gaussian: K = exp(-D/σ) where D is any distance metric
    
    Attributes:
        distance_type: Type of distance ('emd' for color, 'chi2' for texture)
        sigma: Scaling factor (σ in the paper)
    """
    
    def __init__(
        self, 
        distance_type: str = 'emd',
        sigma: Optional[float] = None,
        auto_sigma: bool = True
    ):
        """
        Initialize Extended Gaussian kernel.
        
        Args:
            distance_type: 'emd' (Earth Mover's Distance) for color histograms
                          'chi2' (χ² distance) for texton histograms
            sigma: Scaling factor. If None and auto_sigma=True, 
                   will be set based on mean distance.
            auto_sigma: Whether to auto-compute sigma from mean distance
        """
        self.distance_type = distance_type
        self.sigma = sigma
        self.auto_sigma = auto_sigma and (sigma is None)
        
        # Select distance function
        if distance_type == 'emd':
            self._distance_func = emd_1d_matrix
        elif distance_type == 'chi2':
            self._distance_func = chi_square_distance_matrix
        else:
            raise ValueError(f"Unknown distance type: {distance_type}. Use 'emd' or 'chi2'.")
        
        # Cache for distance matrix and mean distance
        self._D = None
        self._mean_distance = None
    
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Extended Gaussian kernel matrix.
        
        K(x_i, x_j) = exp(-D(x_i, x_j) / σ)
        
        Args:
            X: First set of samples (n_samples_X, n_features)
               Each row should be a histogram (for EMD or χ² distance)
            Y: Second set of samples (n_samples_Y, n_features), or None
            
        Returns:
            Kernel matrix K
        """
        # Compute distance matrix
        self._D = self._distance_func(X, Y)
        
        # Auto-compute sigma based on mean distance if not set
        if self.auto_sigma or self.sigma is None:
            if Y is None:
                # For training, compute mean distance
                self._mean_distance = compute_mean_distance(self._D, exclude_diagonal=True)
            else:
                # For test, use training mean if available
                if self._mean_distance is None:
                    self._mean_distance = np.mean(self._D)
            
            # Set sigma to mean distance (multiplier of 1)
            self.sigma = self._mean_distance if self._mean_distance > 0 else 1.0
        
        # Compute kernel: K = exp(-D / σ)
        K = extended_gaussian_kernel(self._D, self.sigma)
        
        return K
    
    def compute_from_distance(self, D: np.ndarray) -> np.ndarray:
        """
        Compute kernel from precomputed distance matrix.
        
        Args:
            D: Precomputed distance matrix
            
        Returns:
            Kernel matrix
        """
        if self.sigma is None:
            raise ValueError("sigma must be set before computing from distance")
        return extended_gaussian_kernel(D, self.sigma)
    
    def get_distance_matrix(self) -> Optional[np.ndarray]:
        """Return the last computed distance matrix."""
        return self._D
    
    def get_mean_distance(self) -> Optional[float]:
        """Return the mean distance (μ in the paper)."""
        return self._mean_distance
    
    def get_params(self) -> Dict:
        return {
            'distance_type': self.distance_type,
            'sigma': self.sigma,
            'mean_distance': self._mean_distance
        }
    
    def set_params(self, **params) -> None:
        if 'sigma' in params:
            self.sigma = params['sigma']
            self.auto_sigma = params['sigma'] is None
        if 'distance_type' in params:
            self.distance_type = params['distance_type']
            if params['distance_type'] == 'emd':
                self._distance_func = emd_1d_matrix
            elif params['distance_type'] == 'chi2':
                self._distance_func = chi_square_distance_matrix


class MultiKernel(BaseKernel):
    """
    Multi-Kernel Combination.
    
    This class implements the multi-kernel learning approach described in the paper.
    Multiple base kernels are combined with learnable weights.
    
    Reference (Paper Section 3):
        "we jointly learn a classifier and a discriminative kernel that combines
        chromatic, intensity, and texture properties"
    
    Mathematical Definition:
        K(x, x') = Σ_{m=1}^{M} θ_m · k_m(φ_m(x), φ_m(x'))
        
    where:
        - M is the number of base kernels (feature types)
        - θ_m >= 0 are the kernel weights
        - k_m are the base kernel functions
        - φ_m are the feature extractors for each type
    
    In the paper's context:
        - m=1: Chromatic features (LAB color)
        - m=2: Intensity features (grayscale)
        - m=3: Texture features (LBP)
    
    Attributes:
        kernels: List of base kernel objects
        weights: Kernel combination weights θ_m
        feature_slices: Indices to slice features for each kernel
    """
    
    def __init__(
        self,
        kernels: List[BaseKernel],
        weights: Optional[np.ndarray] = None,
        feature_slices: Optional[List[slice]] = None
    ):
        """
        Initialize multi-kernel.
        
        Args:
            kernels: List of base kernel objects, one for each feature type
            weights: Initial kernel weights (default: uniform weights)
            feature_slices: How to slice the feature vector for each kernel.
                           If None, assumes features are passed separately.
        """
        self.kernels = kernels
        self.n_kernels = len(kernels)
        
        # Initialize weights uniformly if not provided
        # Paper uses non-negative weights that sum to 1
        if weights is None:
            self.weights = np.ones(self.n_kernels) / self.n_kernels
        else:
            self.weights = np.array(weights)
            # Ensure non-negativity (as per paper)
            self.weights = np.maximum(self.weights, 0)
        
        self.feature_slices = feature_slices
        
        # Store individual kernel matrices for LOO optimization
        self._kernel_matrices = None
    
    def compute(
        self, 
        X: np.ndarray, 
        Y: Optional[np.ndarray] = None,
        X_by_type: Optional[Dict[str, np.ndarray]] = None,
        Y_by_type: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute the combined kernel matrix.
        
        K = Σ_m θ_m · K_m
        
        Args:
            X: Combined features (n_samples_X, total_features) - used if feature_slices set
            Y: Combined features for second set
            X_by_type: Dict mapping feature type to feature matrix
            Y_by_type: Dict mapping feature type to feature matrix for Y
            
        Returns:
            Combined kernel matrix
        """
        if X_by_type is not None:
            # Features provided separately by type
            return self._compute_from_typed_features(X_by_type, Y_by_type)
        elif self.feature_slices is not None:
            # Slice features from combined matrix
            return self._compute_from_sliced_features(X, Y)
        else:
            # Each kernel uses full feature vector
            return self._compute_from_full_features(X, Y)
    
    def _compute_from_typed_features(
        self,
        X_by_type: Dict[str, np.ndarray],
        Y_by_type: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Compute kernel when features are provided by type."""
        feature_types = list(X_by_type.keys())
        
        if len(feature_types) != self.n_kernels:
            raise ValueError(
                f"Number of feature types ({len(feature_types)}) must match "
                f"number of kernels ({self.n_kernels})"
            )
        
        # Compute individual kernel matrices
        self._kernel_matrices = []
        n_X = X_by_type[feature_types[0]].shape[0]
        
        if Y_by_type is not None:
            n_Y = Y_by_type[feature_types[0]].shape[0]
        else:
            n_Y = n_X
            Y_by_type = X_by_type
        
        K_combined = np.zeros((n_X, n_Y))
        
        for m, (kernel, f_type) in enumerate(zip(self.kernels, feature_types)):
            X_m = X_by_type[f_type]
            Y_m = Y_by_type[f_type] if Y_by_type is not X_by_type else None
            
            K_m = kernel.compute(X_m, Y_m)
            self._kernel_matrices.append(K_m)
            
            # Weighted sum: K = Σ θ_m · K_m
            K_combined += self.weights[m] * K_m
        
        return K_combined
    
    def _compute_from_sliced_features(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute kernel by slicing combined feature vector."""
        self._kernel_matrices = []
        
        n_X = X.shape[0]
        n_Y = X.shape[0] if Y is None else Y.shape[0]
        K_combined = np.zeros((n_X, n_Y))
        
        for m, (kernel, feat_slice) in enumerate(zip(self.kernels, self.feature_slices)):
            X_m = X[:, feat_slice]
            Y_m = Y[:, feat_slice] if Y is not None else None
            
            K_m = kernel.compute(X_m, Y_m)
            self._kernel_matrices.append(K_m)
            
            K_combined += self.weights[m] * K_m
        
        return K_combined
    
    def _compute_from_full_features(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute kernel when each base kernel uses full features."""
        self._kernel_matrices = []
        
        n_X = X.shape[0]
        n_Y = X.shape[0] if Y is None else Y.shape[0]
        K_combined = np.zeros((n_X, n_Y))
        
        for m, kernel in enumerate(self.kernels):
            K_m = kernel.compute(X, Y)
            self._kernel_matrices.append(K_m)
            K_combined += self.weights[m] * K_m
        
        return K_combined
    
    def compute_individual_kernels(
        self,
        X_by_type: Dict[str, np.ndarray],
        Y_by_type: Optional[Dict[str, np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Compute individual kernel matrices without combining.
        
        This is useful for LOO optimization where we need to recompute
        the combined kernel with different weights.
        
        Args:
            X_by_type: Features by type
            Y_by_type: Features by type for Y (optional)
            
        Returns:
            List of kernel matrices [K_1, K_2, ..., K_M]
        """
        feature_types = list(X_by_type.keys())
        kernel_matrices = []
        
        for m, (kernel, f_type) in enumerate(zip(self.kernels, feature_types)):
            X_m = X_by_type[f_type]
            if Y_by_type is not None:
                Y_m = Y_by_type[f_type]
            else:
                Y_m = None
            
            K_m = kernel.compute(X_m, Y_m)
            kernel_matrices.append(K_m)
        
        self._kernel_matrices = kernel_matrices
        return kernel_matrices
    
    def combine_kernels(
        self, 
        kernel_matrices: List[np.ndarray],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Combine precomputed kernel matrices with given weights.
        
        K = Σ_m θ_m · K_m
        
        Args:
            kernel_matrices: List of precomputed kernel matrices
            weights: Combination weights (uses self.weights if None)
            
        Returns:
            Combined kernel matrix
        """
        if weights is None:
            weights = self.weights
        
        K_combined = np.zeros_like(kernel_matrices[0])
        for m, K_m in enumerate(kernel_matrices):
            K_combined += weights[m] * K_m
        
        return K_combined
    
    def get_params(self) -> Dict:
        return {
            'weights': self.weights.copy(),
            'kernel_params': [k.get_params() for k in self.kernels]
        }
    
    def set_params(self, **params) -> None:
        if 'weights' in params:
            self.weights = np.array(params['weights'])
            self.weights = np.maximum(self.weights, 0)  # Ensure non-negativity
    
    @property
    def kernel_matrices(self) -> Optional[List[np.ndarray]]:
        """Return the individual kernel matrices from last computation."""
        return self._kernel_matrices


def create_default_kernels_for_shadow_detection(
    sigma_L: Optional[float] = None,
    sigma_a: Optional[float] = None,
    sigma_b: Optional[float] = None,
    sigma_t: Optional[float] = None
) -> 'ShadowDetectionMultiKernel':
    """
    Create the default multi-kernel setup for shadow detection.
    
    Following the paper's approach with FOUR kernel types:
    - L* channel histogram (EMD distance)
    - a* channel histogram (EMD distance)
    - b* channel histogram (EMD distance)
    - Texton histogram (χ² distance)
    
    Reference:
        Paper Section 3.1 (Equation 5):
        K(x,y) = Σ_{l∈{L,a,b,t}} w_l exp(-1/σ_l D_l(x,y))
        
        where D_L, D_a, D_b are EMD distances and D_t is χ² distance.
    
    Args:
        sigma_L: Scaling factor for L* kernel
        sigma_a: Scaling factor for a* kernel
        sigma_b: Scaling factor for b* kernel
        sigma_t: Scaling factor for texton kernel
        
    Returns:
        ShadowDetectionMultiKernel configured per paper specifications
    """
    return ShadowDetectionMultiKernel(
        sigma_L=sigma_L,
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        sigma_t=sigma_t
    )


class ShadowDetectionMultiKernel(BaseKernel):
    """
    Multi-kernel specifically designed for shadow detection per paper specifications.
    
    This class implements the exact kernel formulation from the paper:
        K(x,y) = Σ_{l∈{L,a,b,t}} w_l exp(-1/σ_l D_l(x,y))
    
    where:
        - D_L, D_a, D_b: EMD distance for L*, a*, b* color histograms
        - D_t: χ² distance for texton histograms
        - w_l: kernel weights (non-negative, sum to 1)
        - σ_l: scaling factors
    
    Reference:
        Paper Section 3.1 (Equation 5)
    
    Feature Structure Expected:
        - hist_L: 21-bin histogram for L* channel
        - hist_a: 21-bin histogram for a* channel
        - hist_b: 21-bin histogram for b* channel
        - hist_t: 128-bin texton histogram
    """
    
    # Feature names following paper convention
    FEATURE_NAMES = ['L', 'a', 'b', 't']
    
    def __init__(
        self,
        sigma_L: Optional[float] = None,
        sigma_a: Optional[float] = None,
        sigma_b: Optional[float] = None,
        sigma_t: Optional[float] = None,
        weights: Optional[np.ndarray] = None
    ):
        """
        Initialize shadow detection multi-kernel.
        
        Args:
            sigma_L, sigma_a, sigma_b: Scaling factors for color kernels
            sigma_t: Scaling factor for texton kernel
            weights: Initial weights [w_L, w_a, w_b, w_t]
        """
        self.n_kernels = 4
        
        # Initialize sigmas (will be auto-set if None)
        self.sigmas = {
            'L': sigma_L,
            'a': sigma_a,
            'b': sigma_b,
            't': sigma_t
        }
        
        # Initialize weights uniformly
        if weights is None:
            self.weights = np.ones(self.n_kernels) / self.n_kernels
        else:
            self.weights = np.array(weights)
            self.weights = np.maximum(self.weights, 0)
        
        # Create base kernels
        # L*, a*, b*: EMD distance (Extended Gaussian)
        # t (texton): χ² distance (Extended Gaussian)
        self.base_kernels = {
            'L': ExtendedGaussianKernel(distance_type='emd', sigma=sigma_L),
            'a': ExtendedGaussianKernel(distance_type='emd', sigma=sigma_a),
            'b': ExtendedGaussianKernel(distance_type='emd', sigma=sigma_b),
            't': ExtendedGaussianKernel(distance_type='chi2', sigma=sigma_t)
        }
        
        # Store individual kernel matrices and distance matrices
        self._kernel_matrices = {}
        self._distance_matrices = {}
        self._mean_distances = {}
    
    def compute(
        self, 
        X: np.ndarray = None, 
        Y: np.ndarray = None,
        features_by_channel: Dict[str, np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the combined kernel matrix.
        
        K = Σ_l w_l · K_l = Σ_l w_l · exp(-D_l / σ_l)
        
        Args:
            X: Not used (for interface compatibility)
            Y: Not used (for interface compatibility)
            features_by_channel: Dict mapping channel name to histogram features
                {'L': hist_L, 'a': hist_a, 'b': hist_b, 't': hist_t}
                Each value is (n_samples, n_bins) array
                
        Returns:
            Combined kernel matrix (n_samples, n_samples)
        """
        if features_by_channel is None:
            raise ValueError("features_by_channel must be provided")
        
        # Validate feature channels
        for channel in self.FEATURE_NAMES:
            if channel not in features_by_channel:
                raise ValueError(f"Missing feature channel: {channel}")
        
        n_samples = features_by_channel['L'].shape[0]
        K_combined = np.zeros((n_samples, n_samples))
        
        # Compute each kernel and combine
        for i, channel in enumerate(self.FEATURE_NAMES):
            X_channel = features_by_channel[channel]
            
            # Compute individual kernel
            K_channel = self.base_kernels[channel].compute(X_channel)
            
            # Store for later use
            self._kernel_matrices[channel] = K_channel
            self._distance_matrices[channel] = self.base_kernels[channel].get_distance_matrix()
            self._mean_distances[channel] = self.base_kernels[channel].get_mean_distance()
            self.sigmas[channel] = self.base_kernels[channel].sigma
            
            # Weighted sum
            K_combined += self.weights[i] * K_channel
        
        return K_combined
    
    def compute_individual_kernels(
        self,
        features_by_channel: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute individual kernel matrices without combining.
        
        Useful for kernel optimization where we need to recompute
        the combined kernel with different weights/sigmas.
        
        Args:
            features_by_channel: Features by channel
            
        Returns:
            Dict mapping channel name to kernel matrix
        """
        kernel_matrices = {}
        
        for channel in self.FEATURE_NAMES:
            X_channel = features_by_channel[channel]
            K_channel = self.base_kernels[channel].compute(X_channel)
            
            kernel_matrices[channel] = K_channel
            self._kernel_matrices[channel] = K_channel
            self._distance_matrices[channel] = self.base_kernels[channel].get_distance_matrix()
            self._mean_distances[channel] = self.base_kernels[channel].get_mean_distance()
            self.sigmas[channel] = self.base_kernels[channel].sigma
        
        return kernel_matrices
    
    def compute_cross(
        self,
        features_X: Dict[str, np.ndarray],
        features_Y: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute kernel matrix between two different sets of features.
        
        K(X, Y) = Σ_l w_l · exp(-D_l(X, Y) / σ_l)
        
        Args:
            features_X: Features for first set {'L': ..., 'a': ..., 'b': ..., 't': ...}
            features_Y: Features for second set
            
        Returns:
            Combined kernel matrix (n_X, n_Y)
        """
        # Validate feature channels
        for channel in self.FEATURE_NAMES:
            if channel not in features_X:
                raise ValueError(f"Missing feature channel in X: {channel}")
            if channel not in features_Y:
                raise ValueError(f"Missing feature channel in Y: {channel}")
        
        n_X = features_X['L'].shape[0]
        n_Y = features_Y['L'].shape[0]
        K_combined = np.zeros((n_X, n_Y))
        
        # Compute each kernel and combine
        for i, channel in enumerate(self.FEATURE_NAMES):
            X_channel = features_X[channel]
            Y_channel = features_Y[channel]
            
            # Compute cross kernel
            K_channel = self.base_kernels[channel].compute(X_channel, Y_channel)
            
            # Weighted sum
            K_combined += self.weights[i] * K_channel
        
        return K_combined
    
    def combine_kernels(
        self,
        kernel_matrices: Dict[str, np.ndarray],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Combine precomputed kernel matrices with given weights.
        
        Args:
            kernel_matrices: Dict of precomputed kernel matrices by channel
            weights: Combination weights [w_L, w_a, w_b, w_t]
            
        Returns:
            Combined kernel matrix
        """
        if weights is None:
            weights = self.weights
        
        K_combined = np.zeros_like(kernel_matrices['L'])
        
        for i, channel in enumerate(self.FEATURE_NAMES):
            K_combined += weights[i] * kernel_matrices[channel]
        
        return K_combined
    
    def recompute_with_sigmas(
        self,
        features_by_channel: Dict[str, np.ndarray],
        sigmas: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Recompute kernel matrices with new sigma values.
        
        Used during kernel optimization.
        
        Args:
            features_by_channel: Features by channel
            sigmas: New sigma values {'L': σ_L, 'a': σ_a, 'b': σ_b, 't': σ_t}
            
        Returns:
            Dict of kernel matrices
        """
        kernel_matrices = {}
        
        for channel in self.FEATURE_NAMES:
            # Update sigma
            self.base_kernels[channel].set_params(sigma=sigmas[channel])
            self.sigmas[channel] = sigmas[channel]
            
            # Recompute kernel
            X_channel = features_by_channel[channel]
            K_channel = self.base_kernels[channel].compute(X_channel)
            kernel_matrices[channel] = K_channel
        
        return kernel_matrices
    
    def get_params(self) -> Dict:
        return {
            'weights': self.weights.copy(),
            'sigmas': self.sigmas.copy(),
            'mean_distances': self._mean_distances.copy()
        }
    
    def set_params(self, **params) -> None:
        if 'weights' in params:
            self.weights = np.array(params['weights'])
            self.weights = np.maximum(self.weights, 0)
        if 'sigmas' in params:
            for channel, sigma in params['sigmas'].items():
                if channel in self.base_kernels:
                    self.base_kernels[channel].set_params(sigma=sigma)
                    self.sigmas[channel] = sigma
    
    @property
    def kernel_matrices(self) -> Dict[str, np.ndarray]:
        """Return individual kernel matrices from last computation."""
        return self._kernel_matrices
    
    @property
    def distance_matrices(self) -> Dict[str, np.ndarray]:
        """Return distance matrices from last computation."""
        return self._distance_matrices
    
    @property
    def mean_distances(self) -> Dict[str, float]:
        """Return mean distances (μ_l in the paper)."""
        return self._mean_distances

