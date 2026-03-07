"""
Leave-One-Out Kernel Optimization Framework.

This module implements the core optimization framework described in the paper
for jointly learning kernel parameters and classifier weights by minimizing
the Leave-One-Out balanced error rate.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 3:
    "The parameters of the kernel and the classifier are jointly learned to 
    minimize the leave-one-out cross validation error."
    
    Paper Section 3.2 (Optimization grid):
    "We define an 8-dimensional grid; one dimension per kernel parameter.
    The discrete values for each scaling factor σ_l form a set of multiples 
    of the mean distance: {sμ_l | s ∈ {1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8}}.
    For the weight w_l of a base kernel, we use {s/40 | s ∈ {1,...,10}}."
    
    Paper Section 3.3 (Optimization strategy - Beam Search):
    "To optimize the set of kernel parameters, we propose to use beam search 
    with random steps. We first discretize the space of kernel parameters using 
    a grid. Starting from a random parameter vector, we perform a number of 
    iterative updates, and in each update we:
    1. Randomly choose one kernel parameter and assign a new random value
    2. Train an LSSVM and compute the leave-one-out error
    3. Update the parameter set if it yields lower leave-one-out error
    
    In our experiments, we perform 500 iterations. If the leave-one-out error 
    does not decrease after 25 consecutive iterations, we randomly assign new 
    values to all parameters."

Optimization Objective:
=======================

The objective is to find optimal kernel weights w = (w_L, w_a, w_b, w_t) and 
scaling factors σ = (σ_L, σ_a, σ_b, σ_t) that minimize the LOO balanced error:

    min_{w, σ}  BER_LOO(w, σ)

where:
    - BER = (FPR + FNR) / 2 is the balanced error rate
    - LOO predictions are computed using Platt scaling with threshold 0.5

Author: [Your Name]
Date: 2026
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy import linalg
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings

from .kernels import BaseKernel, MultiKernel, RBFKernel, ExtendedGaussianKernel, ShadowDetectionMultiKernel
from .lssvm import LSSVM, fit_lssvm_with_precomputed_kernel, compute_loo_error_from_solution
from .distances import (
    emd_1d_matrix, chi_square_distance_matrix, 
    extended_gaussian_kernel, compute_mean_distance,
    compute_sigma_grid, compute_weight_grid
)
from .platt_scaling import PlattScaler, balanced_error_rate, compute_loo_balanced_error


class LOOKernelOptimizer:
    """
    Leave-One-Out Kernel Optimization for LSSVM.
    
    This class implements the joint optimization of kernel weights and
    regularization parameter by minimizing LOO cross-validation error.
    
    The optimization follows the paper's approach:
    1. Precompute individual kernel matrices K_1, ..., K_M
    2. For each candidate (θ, γ), compute combined kernel K = Σ θ_m K_m
    3. Fit LSSVM with K and γ
    4. Compute LOO error using closed-form formula
    5. Use optimization algorithm to find optimal (θ, γ)
    
    Attributes:
        base_kernels: List of base kernel functions
        optimize_gamma: Whether to optimize regularization parameter
        error_type: Type of LOO error to minimize
    """
    
    def __init__(
        self,
        base_kernels: Optional[List[BaseKernel]] = None,
        optimize_gamma: bool = True,
        gamma_bounds: Tuple[float, float] = (1e-4, 1e4),
        weight_bounds: Tuple[float, float] = (0.0, 10.0),
        error_type: str = 'classification',
        verbose: bool = True
    ):
        """
        Initialize the LOO kernel optimizer.
        
        Args:
            base_kernels: List of base kernel objects for multi-kernel learning.
                         If None, will be created based on feature types.
            optimize_gamma: Whether to optimize LSSVM regularization parameter
            gamma_bounds: Bounds for gamma optimization (log scale used internally)
            weight_bounds: Bounds for kernel weight optimization
            error_type: Type of LOO error to minimize:
                       - 'classification': 0-1 misclassification rate
                       - 'mse': Mean squared error of LOO residuals
            verbose: Whether to print optimization progress
        """
        self.base_kernels = base_kernels
        self.optimize_gamma = optimize_gamma
        self.gamma_bounds = gamma_bounds
        self.weight_bounds = weight_bounds
        self.error_type = error_type
        self.verbose = verbose
        
        # Optimized parameters
        self.optimal_weights_ = None
        self.optimal_gamma_ = None
        self.optimal_loo_error_ = None
        
        # Cached kernel matrices
        self._kernel_matrices = None
        self._n_kernels = None
        
        # Training data
        self._y_train = None
        
        # Optimization history
        self._optimization_history = []
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[LOO-Optimizer] {message}")
    
    def optimize(
        self,
        features_by_type: Dict[str, np.ndarray],
        y: np.ndarray,
        initial_gamma: float = 1.0,
        method: str = 'L-BFGS-B',
        max_iter: int = 100
    ) -> Tuple[np.ndarray, float, float]:
        """
        Optimize kernel weights and regularization parameter.
        
        This is the main optimization routine that implements the paper's approach.
        
        Args:
            features_by_type: Dictionary mapping feature type to feature matrix
                             {'chromatic': X_c, 'intensity': X_i, 'texture': X_t}
            y: Training labels (n_samples,), values in {-1, +1}
            initial_gamma: Initial value for regularization parameter
            method: Optimization method ('L-BFGS-B', 'SLSQP', 'differential_evolution')
            max_iter: Maximum number of optimization iterations
            
        Returns:
            optimal_weights: Optimized kernel weights
            optimal_gamma: Optimized regularization parameter
            optimal_loo_error: Minimum LOO error achieved
            
        Reference:
            Paper Section 3: Joint optimization of kernel and classifier parameters
        """
        self._log("Starting Leave-One-Out Kernel Optimization...")
        
        # Store training labels
        self._y_train = np.array(y, dtype=np.float64)
        if set(np.unique(self._y_train)).issubset({0, 1}):
            self._y_train = 2 * self._y_train - 1
        
        n_samples = len(self._y_train)
        feature_types = list(features_by_type.keys())
        self._n_kernels = len(feature_types)
        
        self._log(f"  Samples: {n_samples}, Feature types: {self._n_kernels}")
        
        # Create base kernels if not provided
        if self.base_kernels is None:
            self.base_kernels = [RBFKernel() for _ in range(self._n_kernels)]
        
        # Step 1: Precompute individual kernel matrices
        # This is done once and reused for all optimization iterations
        self._log("  Precomputing individual kernel matrices...")
        self._kernel_matrices = []
        for i, f_type in enumerate(feature_types):
            X_m = features_by_type[f_type]
            K_m = self.base_kernels[i].compute(X_m)
            self._kernel_matrices.append(K_m)
            self._log(f"    K_{f_type}: shape {K_m.shape}")
        
        # Step 2: Set up optimization problem
        # Variables: [θ_1, ..., θ_M, log(γ)] (using log for γ for better scaling)
        n_params = self._n_kernels + (1 if self.optimize_gamma else 0)
        
        # Initial values: uniform weights, initial gamma
        x0 = np.ones(self._n_kernels) / self._n_kernels
        if self.optimize_gamma:
            x0 = np.append(x0, np.log(initial_gamma))
        
        # Bounds
        bounds = [(self.weight_bounds[0], self.weight_bounds[1])] * self._n_kernels
        if self.optimize_gamma:
            bounds.append((np.log(self.gamma_bounds[0]), np.log(self.gamma_bounds[1])))
        
        self._log(f"  Optimization method: {method}, max_iter: {max_iter}")
        
        # Step 3: Run optimization
        self._optimization_history = []
        
        if method == 'differential_evolution':
            # Global optimization (more robust but slower)
            result = differential_evolution(
                self._objective_function,
                bounds=bounds,
                maxiter=max_iter,
                polish=True,
                disp=self.verbose,
                workers=1  # Serial execution
            )
        else:
            # Local optimization (faster)
            result = minimize(
                self._objective_function,
                x0,
                method=method,
                bounds=bounds,
                options={
                    'maxiter': max_iter,
                    'disp': self.verbose
                }
            )
        
        # Step 4: Extract optimal parameters
        optimal_params = result.x
        self.optimal_weights_ = optimal_params[:self._n_kernels]
        
        # Normalize weights to sum to 1 (optional, for interpretability)
        weight_sum = np.sum(self.optimal_weights_)
        if weight_sum > 0:
            self.optimal_weights_ = self.optimal_weights_ / weight_sum
        
        if self.optimize_gamma:
            self.optimal_gamma_ = np.exp(optimal_params[-1])
        else:
            self.optimal_gamma_ = initial_gamma
        
        self.optimal_loo_error_ = result.fun
        
        # Log results
        self._log("\nOptimization completed!")
        self._log(f"  Optimal kernel weights: {self.optimal_weights_}")
        self._log(f"  Optimal gamma: {self.optimal_gamma_:.6f}")
        self._log(f"  Minimum LOO error: {self.optimal_loo_error_:.6f}")
        
        return self.optimal_weights_, self.optimal_gamma_, self.optimal_loo_error_
    
    def _objective_function(self, params: np.ndarray) -> float:
        """
        Compute LOO error for given parameters.
        
        This is the objective function to be minimized:
            L_LOO(θ, γ) = (1/N) Σ ℓ(y_i, f_{-i}(x_i))
        
        Args:
            params: [θ_1, ..., θ_M, log(γ)] or [θ_1, ..., θ_M]
            
        Returns:
            LOO error value
        """
        # Extract parameters
        weights = params[:self._n_kernels]
        
        if self.optimize_gamma:
            gamma = np.exp(params[-1])
        else:
            gamma = 1.0
        
        # Ensure non-negative weights
        weights = np.maximum(weights, 0)
        
        # Compute combined kernel: K = Σ θ_m K_m
        K_combined = np.zeros_like(self._kernel_matrices[0])
        for m, K_m in enumerate(self._kernel_matrices):
            K_combined += weights[m] * K_m
        
        # Fit LSSVM and compute LOO error
        try:
            alpha, bias, M_inv = fit_lssvm_with_precomputed_kernel(
                K_combined, self._y_train, gamma
            )
            
            loo_error = compute_loo_error_from_solution(
                alpha, self._y_train, M_inv, self.error_type
            )
        except (linalg.LinAlgError, ValueError) as e:
            # If fitting fails, return a large error
            warnings.warn(f"LSSVM fitting failed: {e}")
            loo_error = 1.0
        
        # Record history
        self._optimization_history.append({
            'weights': weights.copy(),
            'gamma': gamma,
            'loo_error': loo_error
        })
        
        return loo_error
    
    def grid_search(
        self,
        features_by_type: Dict[str, np.ndarray],
        y: np.ndarray,
        gamma_values: List[float] = None,
        weight_grid_size: int = 5
    ) -> Tuple[np.ndarray, float, float]:
        """
        Grid search for kernel weights and gamma.
        
        This is a simpler alternative to continuous optimization that
        evaluates LOO error on a discrete grid of parameters.
        
        Args:
            features_by_type: Features by type
            y: Labels
            gamma_values: List of gamma values to try
            weight_grid_size: Number of weight values to try per kernel
            
        Returns:
            optimal_weights, optimal_gamma, optimal_loo_error
        """
        self._log("Starting Grid Search...")
        
        # Store labels
        self._y_train = np.array(y, dtype=np.float64)
        if set(np.unique(self._y_train)).issubset({0, 1}):
            self._y_train = 2 * self._y_train - 1
        
        feature_types = list(features_by_type.keys())
        self._n_kernels = len(feature_types)
        
        # Create base kernels
        if self.base_kernels is None:
            self.base_kernels = [RBFKernel() for _ in range(self._n_kernels)]
        
        # Precompute kernels
        self._log("  Precomputing kernel matrices...")
        self._kernel_matrices = []
        for i, f_type in enumerate(feature_types):
            K_m = self.base_kernels[i].compute(features_by_type[f_type])
            self._kernel_matrices.append(K_m)
        
        # Default gamma values
        if gamma_values is None:
            gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        # Weight grid (simplex-like: weights sum to approximately 1)
        weight_values = np.linspace(0.1, 1.0, weight_grid_size)
        
        best_error = float('inf')
        best_weights = None
        best_gamma = None
        
        n_configs = len(gamma_values) * (weight_grid_size ** self._n_kernels)
        self._log(f"  Evaluating approximately {n_configs} configurations...")
        
        # Simplified grid for 3 kernels
        if self._n_kernels == 3:
            weight_configs = []
            for w1 in weight_values:
                for w2 in weight_values:
                    w3 = max(0, 1.0 - w1 - w2)
                    if w3 <= 1.0:
                        weight_configs.append([w1, w2, w3])
        else:
            # For other numbers of kernels, use uniform grid
            from itertools import product
            weight_configs = list(product(weight_values, repeat=self._n_kernels))
        
        count = 0
        for gamma in gamma_values:
            for weights in weight_configs:
                weights = np.array(weights)
                
                # Combine kernels
                K_combined = np.zeros_like(self._kernel_matrices[0])
                for m, K_m in enumerate(self._kernel_matrices):
                    K_combined += weights[m] * K_m
                
                # Compute LOO error
                try:
                    alpha, bias, M_inv = fit_lssvm_with_precomputed_kernel(
                        K_combined, self._y_train, gamma
                    )
                    loo_error = compute_loo_error_from_solution(
                        alpha, self._y_train, M_inv, self.error_type
                    )
                    
                    if loo_error < best_error:
                        best_error = loo_error
                        best_weights = weights.copy()
                        best_gamma = gamma
                except:
                    continue
                
                count += 1
        
        # Normalize weights
        if best_weights is not None:
            best_weights = best_weights / np.sum(best_weights)
        
        self.optimal_weights_ = best_weights
        self.optimal_gamma_ = best_gamma
        self.optimal_loo_error_ = best_error
        
        self._log(f"\nGrid search completed ({count} configurations evaluated)")
        self._log(f"  Optimal weights: {best_weights}")
        self._log(f"  Optimal gamma: {best_gamma}")
        self._log(f"  Minimum LOO error: {best_error:.6f}")
        
        return best_weights, best_gamma, best_error
    
    def get_optimized_lssvm(
        self,
        features_by_type: Dict[str, np.ndarray],
        y: np.ndarray
    ) -> LSSVM:
        """
        Create an LSSVM model with optimized parameters.
        
        Args:
            features_by_type: Features by type
            y: Labels
            
        Returns:
            Fitted LSSVM model with optimal kernel and gamma
        """
        if self.optimal_weights_ is None:
            raise RuntimeError("Must call optimize() or grid_search() first")
        
        # Create multi-kernel with optimal weights
        multi_kernel = MultiKernel(
            kernels=self.base_kernels,
            weights=self.optimal_weights_
        )
        
        # Compute optimal kernel matrix
        feature_types = list(features_by_type.keys())
        K_optimal = multi_kernel.compute(None, None, features_by_type)
        
        # Create and fit LSSVM
        lssvm = LSSVM(kernel=multi_kernel, gamma=self.optimal_gamma_)
        
        # We need to pass a dummy X for the LSSVM interface
        # but use precomputed kernel
        X_dummy = np.column_stack([features_by_type[ft] for ft in feature_types])
        lssvm.fit(X_dummy, y, K=K_optimal)
        
        return lssvm
    
    @property
    def optimization_history(self) -> List[Dict]:
        """Return optimization history."""
        return self._optimization_history


class KernelParameterOptimizer:
    """
    Optimizer for individual kernel hyperparameters (e.g., RBF gamma).
    
    This class optimizes the bandwidth parameters of individual base kernels
    in addition to the combination weights.
    
    Reference:
        Paper: The kernel parameters are part of the joint optimization.
    """
    
    def __init__(
        self,
        n_kernels: int = 3,
        optimize_bandwidths: bool = True,
        bandwidth_bounds: Tuple[float, float] = (1e-4, 1e2)
    ):
        """
        Initialize kernel parameter optimizer.
        
        Args:
            n_kernels: Number of base kernels
            optimize_bandwidths: Whether to optimize RBF bandwidths
            bandwidth_bounds: Bounds for bandwidth optimization
        """
        self.n_kernels = n_kernels
        self.optimize_bandwidths = optimize_bandwidths
        self.bandwidth_bounds = bandwidth_bounds
        
        self.optimal_bandwidths_ = None
    
    def compute_kernel_with_bandwidth(
        self,
        X: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """
        Compute RBF kernel with specific bandwidth.
        
        Args:
            X: Feature matrix
            gamma: RBF bandwidth parameter
            
        Returns:
            Kernel matrix
        """
        # Squared distances
        sq_dist = np.sum(X ** 2, axis=1, keepdims=True) + \
                  np.sum(X ** 2, axis=1, keepdims=True).T - \
                  2 * np.dot(X, X.T)
        sq_dist = np.maximum(sq_dist, 0)
        
        return np.exp(-gamma * sq_dist)
    
    def optimize_bandwidths_loo(
        self,
        features_by_type: Dict[str, np.ndarray],
        y: np.ndarray,
        gamma_lssvm: float = 1.0
    ) -> np.ndarray:
        """
        Optimize RBF bandwidths using LOO error.
        
        Args:
            features_by_type: Features by type
            y: Labels
            gamma_lssvm: LSSVM regularization parameter
            
        Returns:
            Optimal bandwidth for each kernel
        """
        feature_types = list(features_by_type.keys())
        n_kernels = len(feature_types)
        
        # Prepare labels
        y = np.array(y, dtype=np.float64)
        if set(np.unique(y)).issubset({0, 1}):
            y = 2 * y - 1
        
        optimal_bandwidths = []
        
        for i, f_type in enumerate(feature_types):
            X = features_by_type[f_type]
            
            # Grid search for this kernel's bandwidth
            best_bandwidth = 1.0 / X.shape[1]  # Default: 1/n_features
            best_error = float('inf')
            
            for log_gamma in np.linspace(
                np.log(self.bandwidth_bounds[0]),
                np.log(self.bandwidth_bounds[1]),
                20
            ):
                bandwidth = np.exp(log_gamma)
                K = self.compute_kernel_with_bandwidth(X, bandwidth)
                
                try:
                    alpha, bias, M_inv = fit_lssvm_with_precomputed_kernel(
                        K, y, gamma_lssvm
                    )
                    loo_error = compute_loo_error_from_solution(
                        alpha, y, M_inv, 'classification'
                    )
                    
                    if loo_error < best_error:
                        best_error = loo_error
                        best_bandwidth = bandwidth
                except:
                    continue
            
            optimal_bandwidths.append(best_bandwidth)
        
        self.optimal_bandwidths_ = np.array(optimal_bandwidths)
        return self.optimal_bandwidths_


class PaperBeamSearchOptimizer:
    """
    Beam Search with Random Steps Optimizer as described in the paper.
    
    This implements the exact optimization algorithm from Section 3.2-3.3:
    
    Reference:
        Paper Section 3.2 (Parameter Grid):
        "We define an 8-dimensional grid; one dimension per kernel parameter.
        The discrete values for each scaling factor σ_l form a set of multiples 
        of the mean distance: {sμ_l | s ∈ {1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8}}.
        For the weight w_l of a base kernel, we use {s/40 | s ∈ {1,...,10}}."
        
        Paper Section 3.3 (Optimization Strategy):
        "Starting from a random parameter vector, we perform a number of 
        iterative updates, and in each update we:
        1. Randomly choose one kernel parameter and assign a new random value
        2. Train an LSSVM and compute the leave-one-out error
        3. Update the parameter set if it yields lower leave-one-out error
        
        In our experiments, we perform 500 iterations. If the leave-one-out 
        error does not decrease after 25 consecutive iterations, we randomly 
        assign new values to all parameters."
    
    Attributes:
        n_iterations: Number of optimization iterations (default: 500)
        stagnation_threshold: Reset after this many non-improving iterations (default: 25)
        use_platt_scaling: Whether to use Platt scaling for BER computation
    """
    
    # Kernel channels following paper: L*, a*, b*, texture
    CHANNELS = ['L', 'a', 'b', 't']
    
    # σ multipliers from paper: {1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8}
    SIGMA_MULTIPLIERS = np.array([1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8])
    
    # Weight values from paper: {s/40 | s ∈ {1,...,10}}
    WEIGHT_VALUES = np.arange(1, 11) / 40.0  # [0.025, 0.05, ..., 0.25]
    
    def __init__(
        self,
        n_iterations: int = 500,
        stagnation_threshold: int = 25,
        use_platt_scaling: bool = True,
        gamma_lssvm: float = 1.0,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize beam search optimizer.
        
        Args:
            n_iterations: Number of optimization iterations
            stagnation_threshold: Reset after this many non-improving iterations
            use_platt_scaling: Whether to use Platt scaling
            gamma_lssvm: LSSVM regularization parameter
            verbose: Whether to print progress
            random_state: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.stagnation_threshold = stagnation_threshold
        self.use_platt_scaling = use_platt_scaling
        self.gamma_lssvm = gamma_lssvm
        self.verbose = verbose
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Optimal parameters
        self.optimal_weights_ = None
        self.optimal_sigmas_ = None
        self.optimal_ber_ = None
        
        # Distance matrices and mean distances
        self._distance_matrices = {}
        self._mean_distances = {}
        self._sigma_grids = {}
        
        # Training data
        self._y_train = None
        
        # Optimization history
        self._history = []
    
    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[BeamSearch] {msg}")
    
    def optimize(
        self,
        features_by_channel: Dict[str, np.ndarray],
        y: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        Optimize kernel parameters using beam search with random steps.
        
        Args:
            features_by_channel: Dict mapping channel to histogram features
                {'L': hist_L, 'a': hist_a, 'b': hist_b, 't': hist_t}
            y: Training labels (n_samples,)
            
        Returns:
            optimal_weights: Dict {channel: weight}
            optimal_sigmas: Dict {channel: sigma}
            optimal_ber: Best leave-one-out balanced error rate
        """
        try:
            from tqdm import tqdm  # type: ignore
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        self._log("Starting Beam Search Optimization (Paper Algorithm)")
        self._log(f"  Iterations: {self.n_iterations}, Stagnation threshold: {self.stagnation_threshold}")
        
        # Store training labels
        self._y_train = np.asarray(y, dtype=np.float64)
        if set(np.unique(self._y_train)).issubset({0, 1}):
            self._y_train = 2 * self._y_train - 1
        
        n_samples = len(self._y_train)
        self._log(f"  Training samples: {n_samples}")
        
        # Step 1: Precompute distance matrices for each channel
        self._log("  Precomputing distance matrices...")
        for channel in self.CHANNELS:
            X = features_by_channel[channel]
            
            if channel == 't':  # Texture uses χ² distance
                D = chi_square_distance_matrix(X)
            else:  # Color channels use EMD
                D = emd_1d_matrix(X)
            
            self._distance_matrices[channel] = D
            self._mean_distances[channel] = compute_mean_distance(D)
            self._sigma_grids[channel] = self.SIGMA_MULTIPLIERS * self._mean_distances[channel]
            
            self._log(f"    {channel}: mean_dist={self._mean_distances[channel]:.4f}")
        
        # Step 2: Initialize with random parameters
        current_params = self._random_params()
        current_ber = self._compute_ber(current_params)
        
        best_params = current_params.copy()
        best_ber = current_ber
        
        self._log(f"  Initial BER: {best_ber:.4f}")
        
        # Step 3: Beam search with random steps
        stagnation_count = 0
        iterator = range(self.n_iterations)
        if use_tqdm and self.verbose:
            iterator = tqdm(iterator, desc="BeamSearch", total=self.n_iterations)
        
        for iteration in iterator:
            # Randomly choose one parameter to change
            param_type = np.random.choice(['weight', 'sigma'])
            channel = np.random.choice(self.CHANNELS)
            
            # Create new parameters with one random change
            # Deep copy to avoid modifying original
            new_params = {
                'weights': current_params['weights'].copy(),
                'sigmas': current_params['sigmas'].copy()
            }
            
            if param_type == 'weight':
                new_params['weights'][channel] = np.random.choice(self.WEIGHT_VALUES)
                # Paper Section 3.3: "If necessary, re-normalize {w_i} to have unit sum"
                new_params['weights'] = self._normalize_weights(new_params['weights'])
            else:
                new_params['sigmas'][channel] = np.random.choice(self._sigma_grids[channel])
            
            # Compute BER for new parameters
            new_ber = self._compute_ber(new_params)
            
            # Update if better
            if new_ber < current_ber:
                current_params = new_params
                current_ber = new_ber
                stagnation_count = 0
                
                if new_ber < best_ber:
                    best_params = new_params.copy()
                    best_ber = new_ber
                    
                    if self.verbose and iteration % 50 == 0:
                        self._log(f"  Iter {iteration}: New best BER = {best_ber:.4f}")
            else:
                stagnation_count += 1
            
            # Check for stagnation: reset if no improvement
            if stagnation_count >= self.stagnation_threshold:
                current_params = self._random_params()
                current_ber = self._compute_ber(current_params)
                stagnation_count = 0
                
                if self.verbose:
                    self._log(f"  Iter {iteration}: Reset due to stagnation")
            
            # Record history
            self._history.append({
                'iteration': iteration,
                'ber': current_ber,
                'best_ber': best_ber
            })
            
            if use_tqdm and self.verbose:
                iterator.set_postfix({
                    'best_ber': f"{best_ber:.4f}",
                    'stagnation': stagnation_count
                })
        
        # Store optimal results
        self.optimal_weights_ = best_params['weights']
        self.optimal_sigmas_ = best_params['sigmas']
        self.optimal_ber_ = best_ber
        
        self._log(f"\nOptimization completed!")
        self._log(f"  Best BER: {best_ber:.4f}")
        self._log(f"  Optimal weights: {self.optimal_weights_}")
        self._log(f"  Optimal sigmas: {self.optimal_sigmas_}")
        
        return self.optimal_weights_, self.optimal_sigmas_, best_ber
    
    def _random_params(self) -> Dict:
        """
        Generate random parameter set.
        
        Note: Paper Section 3 states "we constrain the kernel weights to be 
        non-negative and have unit sum, i.e., Σw_i = 1"
        """
        # Generate random weights and normalize to sum to 1
        raw_weights = {ch: np.random.choice(self.WEIGHT_VALUES) for ch in self.CHANNELS}
        weights = self._normalize_weights(raw_weights)
        
        return {
            'weights': weights,
            'sigmas': {ch: np.random.choice(self._sigma_grids[ch]) for ch in self.CHANNELS}
        }
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.
        
        Reference:
            Paper Section 3: "we constrain the kernel weights to be non-negative 
            and have unit sum, i.e., Σ_{i=1}^k w_i = 1"
            
            Paper Section 3.3: "If necessary, re-normalize {w_i} to have unit sum."
        """
        total = sum(weights.values())
        if total > 0:
            return {ch: w / total for ch, w in weights.items()}
        else:
            # Fallback to uniform weights
            return {ch: 1.0 / len(self.CHANNELS) for ch in self.CHANNELS}
    
    def _compute_ber(self, params: Dict) -> float:
        """
        Compute leave-one-out balanced error rate for given parameters.
        
        Args:
            params: Dict with 'weights' and 'sigmas'
            
        Returns:
            Balanced error rate
        """
        weights = params['weights']
        sigmas = params['sigmas']
        
        # Compute combined kernel matrix
        n = len(self._y_train)
        K_combined = np.zeros((n, n))
        
        for channel in self.CHANNELS:
            D = self._distance_matrices[channel]
            sigma = sigmas[channel]
            w = weights[channel]
            
            # Extended Gaussian kernel: K = exp(-D / σ)
            K_channel = np.exp(-D / (sigma + 1e-10))
            K_combined += w * K_channel
        
        # Fit LSSVM and get LOO predictions
        try:
            alpha, bias, M_inv = fit_lssvm_with_precomputed_kernel(
                K_combined, self._y_train, self.gamma_lssvm
            )
            
            # Compute LOO residuals and predictions
            h_diag = np.diag(M_inv)[1:]
            h_diag = np.where(np.abs(h_diag) < 1e-10, 1e-10, h_diag)
            loo_residuals = alpha / h_diag
            loo_predictions = self._y_train - loo_residuals  # f_{-i}(x_i)
            
            # Compute BER with Platt scaling
            if self.use_platt_scaling:
                ber, _ = compute_loo_balanced_error(loo_predictions, self._y_train, use_platt_scaling=True)
            else:
                y_pred = np.sign(loo_predictions)
                y_pred = np.where(y_pred == 0, 1, y_pred)
                ber = balanced_error_rate(self._y_train, y_pred)
            
            return ber
            
        except Exception as e:
            warnings.warn(f"BER computation failed: {e}")
            return 1.0  # Return worst case error
    
    def get_optimized_kernel(self) -> ShadowDetectionMultiKernel:
        """
        Create optimized multi-kernel with found parameters.
        
        Returns:
            ShadowDetectionMultiKernel with optimal weights and sigmas
        """
        if self.optimal_weights_ is None or self.optimal_sigmas_ is None:
            raise RuntimeError("Must call optimize() first")
        
        kernel = ShadowDetectionMultiKernel(
            sigma_L=self.optimal_sigmas_['L'],
            sigma_a=self.optimal_sigmas_['a'],
            sigma_b=self.optimal_sigmas_['b'],
            sigma_t=self.optimal_sigmas_['t'],
            weights=np.array([self.optimal_weights_[ch] for ch in self.CHANNELS])
        )
        
        return kernel
    
    @property
    def optimization_history(self) -> List[Dict]:
        """Return optimization history."""
        return self._history


def optimize_shadow_classifier(
    features_by_type: Dict[str, np.ndarray],
    y: np.ndarray,
    method: str = 'beam_search',
    verbose: bool = True
) -> Tuple[LSSVM, Dict]:
    """
    Convenience function to optimize and create a shadow classifier.
    
    This implements the full optimization pipeline from the paper.
    
    Args:
        features_by_type: Dictionary of features by type
            For paper-compliant usage, should have keys: 'L', 'a', 'b', 't'
        y: Labels
        method: Optimization method 
            - 'beam_search': Paper's algorithm (recommended)
            - 'grid_search': Simple grid search (faster but less accurate)
            - 'L-BFGS-B': Gradient-based (may not work well for discrete grid)
        verbose: Whether to print progress
        
    Returns:
        lssvm: Optimized LSSVM classifier
        info: Dictionary with optimization information
    """
    if method == 'beam_search':
        # Use paper's algorithm
        optimizer = PaperBeamSearchOptimizer(
            n_iterations=500,
            stagnation_threshold=25,
            use_platt_scaling=True,
            verbose=verbose
        )
        
        weights, sigmas, ber = optimizer.optimize(features_by_type, y)
        
        # Create LSSVM with optimized kernel
        kernel = optimizer.get_optimized_kernel()
        K = kernel.compute(features_by_channel=features_by_type)
        
        lssvm = LSSVM(kernel=kernel, gamma=optimizer.gamma_lssvm)
        X_dummy = np.column_stack([features_by_type[ch] for ch in PaperBeamSearchOptimizer.CHANNELS])
        lssvm.fit(X_dummy, y, K=K)
        
        info = {
            'optimal_weights': weights,
            'optimal_sigmas': sigmas,
            'loo_ber': ber,
            'optimization_history': optimizer.optimization_history
        }
        
        return lssvm, info
    
    else:
        # Fall back to previous optimizer
        optimizer = LOOKernelOptimizer(
            optimize_gamma=True,
            error_type='classification',
            verbose=verbose
        )
        
        if method == 'grid_search':
            weights, gamma, loo_error = optimizer.grid_search(
                features_by_type, y
            )
        else:
            weights, gamma, loo_error = optimizer.optimize(
                features_by_type, y,
                method=method
            )
        
        lssvm = optimizer.get_optimized_lssvm(features_by_type, y)
        
        info = {
            'optimal_weights': weights,
            'optimal_gamma': gamma,
            'loo_error': loo_error,
            'optimization_history': optimizer.optimization_history
        }
        
        return lssvm, info

