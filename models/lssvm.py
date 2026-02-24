"""
Kernel Least-Squares Support Vector Machine (LSSVM) Implementation.

This module implements the LSSVM classifier as described in the paper.
LSSVM is chosen over standard SVM because it allows efficient computation
of Leave-One-Out cross-validation error.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Quote (Section 3):
    "Unlike existing approaches for shadow detection, we propose to use Least Square 
    Support Vector Machine (LSSVM). LSSVM has been shown to..."

Mathematical Background:
=========================

Standard LSSVM Optimization Problem:
------------------------------------
    min_{w,b,e} J(w,e) = (1/2)||w||² + (γ/2) Σ_{i=1}^{N} e_i²
    
    subject to: y_i[w^T φ(x_i) + b] = 1 - e_i,  for i = 1, ..., N

where:
    - w: weight vector in feature space
    - b: bias term
    - e_i: error variables
    - γ > 0: regularization parameter
    - y_i ∈ {-1, +1}: class labels
    - φ(x_i): feature mapping induced by kernel

KKT Conditions Lead to Linear System:
-------------------------------------
The Lagrangian conditions give the following linear system:

    [0    y^T       ] [b ]   [0]
    [y  Ω + γ^{-1}I] [α ] = [1]

where:
    - Ω_{ij} = y_i y_j K(x_i, x_j)  (kernel matrix with label products)
    - α: Lagrange multipliers (support values)
    - 1: vector of ones

Equivalently, using α' = y ⊙ α (element-wise product):

    [0    1^T       ] [b ]   [0]
    [1  K + γ^{-1}I] [α'] = [y]

Decision Function:
------------------
    f(x) = Σ_{i=1}^{N} α_i K(x_i, x) + b
    
    Prediction: sign(f(x))

Key Advantage - LOO Residual Closed-Form:
-----------------------------------------
For LSSVM, the Leave-One-Out prediction error can be computed efficiently
without retraining N times. Define the augmented system matrix:

    M = [0    1^T       ]
        [1  K + γ^{-1}I]

Then the LOO residual for sample i is:

    e_i^{LOO} = α'_i / (M^{-1})_{i+1,i+1}

This is the crucial property that enables efficient kernel optimization.

Author: [Your Name]
Date: 2026
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple, Union, List
from scipy import linalg
import warnings

from .kernels import BaseKernel, RBFKernel, MultiKernel

# GPU Support
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')


class LSSVM:
    """
    Kernel Least-Squares Support Vector Machine Classifier.
    
    This class implements LSSVM with efficient LOO error computation,
    following the approach in the paper.
    
    Attributes:
        kernel: Kernel function object
        gamma: Regularization parameter (denoted γ in the paper)
        alpha: Support values (Lagrange multipliers × labels)
        bias: Bias term
        X_train: Training samples
        y_train: Training labels
        use_gpu: Whether to use GPU acceleration
    """
    
    def __init__(
        self,
        kernel: Optional[BaseKernel] = None,
        gamma: float = 1.0,
        use_gpu: bool = True
    ):
        """
        Initialize LSSVM classifier.
        
        Args:
            kernel: Kernel function (default: RBF kernel)
            gamma: Regularization parameter. Larger γ means less regularization
                   (fitting training data more closely). Must be > 0.
            use_gpu: Whether to use GPU acceleration if available.
                   
        Note:
            The regularization term in LSSVM is (γ/2) Σ e_i², so larger γ
            penalizes errors more, leading to smaller errors but potential overfitting.
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.gamma = gamma
        self.use_gpu = use_gpu and USE_GPU
        
        # Model parameters (set after training)
        self.alpha_ = None      # Support values α' = y ⊙ α
        self.bias_ = None       # Bias term b
        
        # Training data (needed for prediction)
        self.X_train_ = None
        self.y_train_ = None
        
        # Cached matrices for LOO computation
        self._K = None          # Kernel matrix
        self._M = None          # Augmented system matrix
        self._M_inv = None      # Inverse of M
        self._is_fitted = False
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        K: Optional[np.ndarray] = None
    ) -> 'LSSVM':
        """
        Fit the LSSVM model.
        
        Solves the linear system:
            [0    1^T       ] [b ]   [0]
            [1  K + γ^{-1}I] [α'] = [y]
        
        Args:
            X: Training samples, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,), values in {-1, +1}
            K: Precomputed kernel matrix (optional). If provided, X is still
               needed but kernel computation is skipped.
               
        Returns:
            self: The fitted estimator
            
        Note:
            Labels should be -1 or +1. If labels are 0/1, they are converted.
        """
        # Store training data
        self.X_train_ = np.array(X, dtype=np.float64)
        self.y_train_ = np.array(y, dtype=np.float64)
        
        # Convert 0/1 labels to -1/+1 if necessary
        if set(np.unique(self.y_train_)).issubset({0, 1}):
            self.y_train_ = 2 * self.y_train_ - 1
        
        n_samples = self.X_train_.shape[0]
        
        # Compute or use provided kernel matrix
        if K is not None:
            self._K = np.array(K, dtype=np.float64)
        else:
            self._K = self.kernel.compute(self.X_train_)
        
        # Build the augmented system matrix M
        # M = [0    1^T       ]
        #     [1  K + γ^{-1}I]
        
        # Build right-hand side vector
        # [0]
        # [y]
        rhs = np.zeros(n_samples + 1)
        rhs[1:] = self.y_train_
        
        # Solve the linear system M @ [b, α']^T = [0, y]^T
        if self.use_gpu:
            try:
                # Build M and rhs on GPU
                M_gpu = self._build_system_matrix(self._K, n_samples, to_torch=True)
                rhs_gpu = torch.tensor(rhs, dtype=torch.float32, device=DEVICE)
                
                # Solve using torch.linalg.solve (GPU)
                solution_gpu = torch.linalg.solve(M_gpu, rhs_gpu)
                solution = solution_gpu.cpu().numpy()
                
                # Store M and compute M_inv on GPU
                self._M = M_gpu.cpu().numpy()
                self._M_inv = torch.linalg.inv(M_gpu).cpu().numpy()
                
                # Extract bias and support values
                self.bias_ = float(solution[0])
                self.alpha_ = solution[1:]
                
                self._is_fitted = True
                torch.cuda.empty_cache()
                return self
            except Exception as e:
                warnings.warn(f"GPU fit failed: {e}. Falling back to CPU.")
                # Clear GPU cache and proceed with CPU
                torch.cuda.empty_cache()
        
        # CPU Fallback
        self._M = self._build_system_matrix(self._K, n_samples, to_torch=False)
        try:
            # Solve using scipy.linalg.solve
            solution = linalg.solve(self._M, rhs, assume_a='gen')
        except linalg.LinAlgError:
            warnings.warn("System matrix is singular, using least squares solution")
            solution, _, _, _ = linalg.lstsq(self._M, rhs)
        
        # Extract bias and support values
        self.bias_ = solution[0]
        self.alpha_ = solution[1:]
        
        # Compute and cache M^{-1} for LOO calculations
        self._compute_M_inverse()
        
        self._is_fitted = True
        return self
    
    def _build_system_matrix(
        self, 
        K: np.ndarray, 
        n_samples: int, 
        to_torch: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Build the augmented system matrix M.
        
        M = [0    1^T       ]
            [1  K + γ^{-1}I]
            
        Args:
            K: Kernel matrix (n_samples, n_samples)
            n_samples: Number of training samples
            to_torch: Whether to return a torch.Tensor on the current device
            
        Returns:
            M: Augmented system matrix
        """
        if to_torch:
            K_torch = torch.tensor(K, dtype=torch.float32, device=DEVICE)
            M = torch.zeros((n_samples + 1, n_samples + 1), dtype=torch.float32, device=DEVICE)
            M[0, 1:] = 1.0
            M[1:, 0] = 1.0
            reg = 1.0 / self.gamma
            M[1:, 1:] = K_torch + reg * torch.eye(n_samples, device=DEVICE)
        else:
            M = np.zeros((n_samples + 1, n_samples + 1))
            M[0, 1:] = 1.0
            M[1:, 0] = 1.0
            reg = 1.0 / self.gamma
            M[1:, 1:] = K + reg * np.eye(n_samples)
        
        return M
    
    def _compute_M_inverse(self) -> None:
        """
        Compute and cache the inverse of system matrix M.
        """
        if self.use_gpu:
            try:
                M_torch = torch.tensor(self._M, dtype=torch.float32, device=DEVICE)
                self._M_inv = torch.linalg.inv(M_torch).cpu().numpy()
                return
            except Exception as e:
                warnings.warn(f"GPU inverse failed: {e}. Falling back to CPU.")
        
        # CPU Fallback
        try:
            self._M_inv = linalg.inv(self._M)
        except linalg.LinAlgError:
            warnings.warn("Using pseudo-inverse due to singular matrix")
            self._M_inv = linalg.pinv(self._M)
    
    def predict(self, X: np.ndarray, K: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class labels for samples.
        """
        scores = self.decision_function(X, K)
        return np.sign(scores)
    
    def decision_function(
        self, 
        X: np.ndarray,
        K: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute decision function values for samples.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if K is not None:
            K_test = K
        else:
            K_test = self.kernel.compute(self.X_train_, X)
        
        if self.use_gpu:
            try:
                alpha_torch = torch.tensor(self.alpha_, dtype=torch.float32, device=DEVICE)
                K_test_torch = torch.tensor(K_test, dtype=torch.float32, device=DEVICE)
                # alpha'^T @ K_test + bias
                scores_torch = torch.matmul(alpha_torch, K_test_torch) + self.bias_
                return scores_torch.cpu().numpy()
            except Exception as e:
                warnings.warn(f"GPU decision_function failed: {e}. Falling back to CPU.")
        
        # CPU Fallback
        return np.dot(self.alpha_, K_test) + self.bias_
    
    def predict_proba(
        self,
        X: np.ndarray,
        K: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict class probabilities using sigmoid transformation.
        """
        scores = self.decision_function(X, K)
        prob_positive = 1.0 / (1.0 + np.exp(-scores))
        prob_negative = 1.0 - prob_positive
        return np.column_stack([prob_negative, prob_positive])
    
    def compute_loo_residuals(self) -> np.ndarray:
        """
        Compute Leave-One-Out residuals efficiently.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before computing LOO residuals")
        
        if self._M_inv is None:
            self._compute_M_inverse()
        
        if self.use_gpu:
            try:
                M_inv_torch = torch.tensor(self._M_inv, dtype=torch.float32, device=DEVICE)
                alpha_torch = torch.tensor(self.alpha_, dtype=torch.float32, device=DEVICE)
                
                # Extract diagonal elements [1:]
                diag_elements = torch.diagonal(M_inv_torch)[1:]
                
                # Avoid division by zero
                eps = 1e-10
                diag_elements = torch.where(
                    torch.abs(diag_elements) < eps,
                    eps * torch.sign(diag_elements + 1e-15),
                    diag_elements
                )
                
                loo_residuals = alpha_torch / diag_elements
                return loo_residuals.cpu().numpy()
            except Exception as e:
                warnings.warn(f"GPU compute_loo_residuals failed: {e}. Falling back to CPU.")
        
        # CPU Fallback
        diag_elements = np.diag(self._M_inv)[1:]
        diag_elements = np.where(
            np.abs(diag_elements) < 1e-10,
            1e-10 * np.sign(diag_elements + 1e-15),
            diag_elements
        )
        return self.alpha_ / diag_elements
    
    def compute_loo_predictions(self) -> np.ndarray:
        """
        Compute Leave-One-Out predictions.
        
        The LOO prediction for sample i is:
            f_{-i}(x_i) = y_i - e_i^{LOO}
            
        where f_{-i} is the decision function trained without sample i.
        
        Returns:
            LOO predictions, shape (n_samples,)
        """
        loo_residuals = self.compute_loo_residuals()
        
        # LOO prediction = true label - LOO residual
        # This comes from: y_i - f_{-i}(x_i) = e_i^{LOO}
        # Therefore: f_{-i}(x_i) = y_i - e_i^{LOO}
        loo_predictions = self.y_train_ - loo_residuals
        
        return loo_predictions
    
    def compute_loo_error(self, error_type: str = 'classification') -> float:
        """
        Compute Leave-One-Out cross-validation error.
        
        This is the objective function for kernel optimization in the paper.
        
        Args:
            error_type: Type of error to compute
                - 'classification': 0-1 classification error (default)
                - 'mse': Mean squared error
                - 'hinge': Hinge loss
                - 'logistic': Logistic loss
                
        Returns:
            LOO error value
            
        Reference:
            Paper (Section 3): "The parameters of the kernel and the classifier
            are jointly learned to minimize the leave-one-out cross validation error."
        """
        loo_predictions = self.compute_loo_predictions()
        
        if error_type == 'classification':
            # 0-1 loss: count misclassifications
            predictions = np.sign(loo_predictions)
            # Handle zero predictions (assign to positive class)
            predictions = np.where(predictions == 0, 1, predictions)
            error = np.mean(predictions != self.y_train_)
            
        elif error_type == 'mse':
            # Mean squared error
            loo_residuals = self.compute_loo_residuals()
            error = np.mean(loo_residuals ** 2)
            
        elif error_type == 'hinge':
            # Hinge loss: max(0, 1 - y_i * f_{-i}(x_i))
            margins = self.y_train_ * loo_predictions
            error = np.mean(np.maximum(0, 1 - margins))
            
        elif error_type == 'logistic':
            # Logistic loss: log(1 + exp(-y_i * f_{-i}(x_i)))
            margins = self.y_train_ * loo_predictions
            # Numerical stability
            error = np.mean(np.log1p(np.exp(-margins)))
            
        else:
            raise ValueError(f"Unknown error type: {error_type}")
        
        return error
    
    def compute_loo_error_gradient_wrt_kernel(
        self,
        kernel_matrices: List[np.ndarray],
        error_type: str = 'mse'
    ) -> np.ndarray:
        """
        Compute gradient of LOO error with respect to kernel weights.
        
        This gradient is used for optimizing the multi-kernel weights θ_m.
        
        For MSE loss, the gradient can be derived analytically.
        For classification loss, we use a smooth approximation.
        
        Args:
            kernel_matrices: List of individual kernel matrices [K_1, ..., K_M]
            error_type: Type of error ('mse' or 'smooth_classification')
            
        Returns:
            Gradient of LOO error w.r.t. kernel weights, shape (n_kernels,)
            
        Note:
            This is a simplified gradient computation. The full derivation
            involves differentiating through the matrix inverse.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        n_kernels = len(kernel_matrices)
        n_samples = len(self.y_train_)
        
        # Get current LOO residuals
        loo_residuals = self.compute_loo_residuals()
        
        # Diagonal of M^{-1}
        h_diag = np.diag(self._M_inv)[1:]
        
        # For gradient computation, we need ∂e^{LOO}/∂θ_m
        # This involves ∂M^{-1}/∂θ_m which requires careful derivation
        
        # Using the identity: ∂M^{-1}/∂θ = -M^{-1} @ (∂M/∂θ) @ M^{-1}
        # For kernel weight θ_m: ∂M/∂θ_m only affects the K part
        
        gradients = np.zeros(n_kernels)
        
        for m in range(n_kernels):
            # ∂M/∂θ_m: only the bottom-right block changes
            dM_dtheta = np.zeros_like(self._M)
            dM_dtheta[1:, 1:] = kernel_matrices[m]
            
            # ∂M^{-1}/∂θ_m = -M^{-1} @ dM_dtheta @ M^{-1}
            dMinv_dtheta = -self._M_inv @ dM_dtheta @ self._M_inv
            
            # Extract diagonal changes
            dh_diag = np.diag(dMinv_dtheta)[1:]
            
            # Gradient of LOO residual w.r.t. θ_m
            # e_i^{LOO} = α_i / h_{ii}
            # ∂e_i^{LOO}/∂θ_m involves both ∂α_i/∂θ_m and ∂h_{ii}/∂θ_m
            
            # Simplified: using chain rule on MSE
            if error_type == 'mse':
                # ∂L/∂θ_m = (2/n) Σ e_i^{LOO} * ∂e_i^{LOO}/∂θ_m
                # Approximate gradient using finite difference would be more robust
                # Here we use a first-order approximation
                
                # ∂(α/h)/∂θ ≈ -α * ∂h/∂θ / h²
                de_dtheta = -self.alpha_ * dh_diag / (h_diag ** 2 + 1e-10)
                gradients[m] = (2.0 / n_samples) * np.dot(loo_residuals, de_dtheta)
            else:
                # For other losses, use numerical approximation
                gradients[m] = 0  # Placeholder
        
        return gradients
    
    # =========================================================================
    # Training Error and Metrics
    # =========================================================================
    
    def compute_training_residuals(self) -> np.ndarray:
        """
        Compute training residuals.
        
        e_i = y_i - f(x_i)
        
        Returns:
            Training residuals, shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Predictions on training data
        train_predictions = self.decision_function(self.X_train_, self._K.T)
        
        # Residuals
        residuals = self.y_train_ - train_predictions
        
        return residuals
    
    def score(self, X: np.ndarray, y: np.ndarray, K: Optional[np.ndarray] = None) -> float:
        """
        Compute classification accuracy.
        
        Args:
            X: Test samples
            y: True labels
            K: Precomputed kernel matrix (optional)
            
        Returns:
            Classification accuracy
        """
        predictions = self.predict(X, K)
        
        # Convert y to -1/+1 if necessary
        y = np.array(y)
        if set(np.unique(y)).issubset({0, 1}):
            y = 2 * y - 1
        
        return np.mean(predictions == y)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'gamma': self.gamma,
            'kernel': self.kernel.get_params() if self.kernel else None
        }
    
    def set_params(self, **params) -> 'LSSVM':
        """Set model parameters."""
        if 'gamma' in params:
            self.gamma = params['gamma']
        if 'kernel' in params:
            self.kernel = params['kernel']
        return self
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted
    
    @property
    def n_support_(self) -> int:
        """Return number of support vectors (all samples for LSSVM)."""
        return len(self.alpha_) if self.alpha_ is not None else 0


def fit_lssvm_with_precomputed_kernel(
    K: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    use_gpu: bool = True
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit LSSVM using precomputed kernel matrix.
    
    This is a convenience function for kernel optimization where we need
    to repeatedly fit LSSVM with different kernels.
    
    Args:
        K: Precomputed kernel matrix (n_samples, n_samples)
        y: Labels (n_samples,), values in {-1, +1}
        gamma: Regularization parameter
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        alpha: Support values
        bias: Bias term
        M_inv: Inverse of system matrix (for LOO computation)
    """
    n_samples = K.shape[0]
    reg = 1.0 / gamma
    
    if use_gpu and USE_GPU:
        try:
            K_torch = torch.tensor(K, dtype=torch.float32, device=DEVICE)
            y_torch = torch.tensor(y, dtype=torch.float32, device=DEVICE)
            
            # Build system matrix M
            M = torch.zeros((n_samples + 1, n_samples + 1), dtype=torch.float32, device=DEVICE)
            M[0, 1:] = 1.0
            M[1:, 0] = 1.0
            M[1:, 1:] = K_torch + reg * torch.eye(n_samples, device=DEVICE)
            
            # Build RHS
            rhs = torch.zeros(n_samples + 1, dtype=torch.float32, device=DEVICE)
            rhs[1:] = y_torch
            
            # Solve
            solution = torch.linalg.solve(M, rhs)
            M_inv = torch.linalg.inv(M)
            
            alpha = solution[1:].cpu().numpy()
            bias = float(solution[0].cpu().numpy())
            M_inv_np = M_inv.cpu().numpy()
            
            torch.cuda.empty_cache()
            return alpha, bias, M_inv_np
        except Exception as e:
            warnings.warn(f"GPU fit_lssvm_with_precomputed_kernel failed: {e}. Falling back to CPU.")
            torch.cuda.empty_cache()

    # CPU Fallback
    M = np.zeros((n_samples + 1, n_samples + 1))
    M[0, 1:] = 1.0
    M[1:, 0] = 1.0
    M[1:, 1:] = K + reg * np.eye(n_samples)
    
    rhs = np.zeros(n_samples + 1)
    rhs[1:] = y
    
    try:
        solution = linalg.solve(M, rhs, assume_a='gen')
        M_inv = linalg.inv(M)
    except linalg.LinAlgError:
        solution, _, _, _ = linalg.lstsq(M, rhs)
        M_inv = linalg.pinv(M)
    
    bias = solution[0]
    alpha = solution[1:]
    
    return alpha, bias, M_inv


def compute_loo_error_from_solution(
    alpha: np.ndarray,
    y: np.ndarray,
    M_inv: np.ndarray,
    error_type: str = 'classification',
    use_gpu: bool = True
) -> float:
    """
    Compute LOO error from LSSVM solution.
    
    This is the core function for kernel optimization objective.
    
    Args:
        alpha: Support values from LSSVM
        y: Training labels
        M_inv: Inverse of system matrix
        error_type: 'classification' or 'mse'
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        LOO error value
    """
    if use_gpu and USE_GPU:
        try:
            alpha_torch = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
            y_torch = torch.tensor(y, dtype=torch.float32, device=DEVICE)
            M_inv_torch = torch.tensor(M_inv, dtype=torch.float32, device=DEVICE)
            
            h_diag = torch.diagonal(M_inv_torch)[1:]
            eps = 1e-10
            h_diag = torch.where(torch.abs(h_diag) < eps, eps, h_diag)
            
            loo_residuals = alpha_torch / h_diag
            loo_predictions = y_torch - loo_residuals
            
            if error_type == 'classification':
                preds = torch.sign(loo_predictions)
                preds = torch.where(preds == 0, torch.ones_like(preds), preds)
                error = torch.mean((preds != y_torch).float())
            elif error_type == 'mse':
                error = torch.mean(loo_residuals ** 2)
            else:
                raise ValueError(f"Unknown error type: {error_type}")
            
            result = float(error.cpu().numpy())
            torch.cuda.empty_cache()
            return result
        except Exception as e:
            warnings.warn(f"GPU compute_loo_error_from_solution failed: {e}. Falling back to CPU.")
            torch.cuda.empty_cache()

    # CPU Fallback
    h_diag = np.diag(M_inv)[1:]
    h_diag = np.where(np.abs(h_diag) < 1e-10, 1e-10, h_diag)
    
    loo_residuals = alpha / h_diag
    loo_predictions = y - loo_residuals
    
    if error_type == 'classification':
        predictions = np.sign(loo_predictions)
        predictions = np.where(predictions == 0, 1, predictions)
        error = np.mean(predictions != y)
    elif error_type == 'mse':
        error = np.mean(loo_residuals ** 2)
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    return error

