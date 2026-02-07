"""
Platt Scaling for LSSVM Probability Calibration.

This module implements Platt scaling to convert LSSVM decision values
to calibrated probabilities, as described in the paper.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 3.3:
    "To resolve this issue, we first use Platt scaling [23] to map the decision 
    values of LSSVM to probabilities, and then use the probability threshold of 0.5."
    
    "More specifically, suppose f_i is the leave-one-out score for the i-th training 
    example. We map f_i to a probability using a sigmoid function:
    P_{a,b}(f_i) = 1/(1 + exp(af_i + b))
    where a, b are the parameters of the function."
    
    "The parameters a, b are set by solving, using Newton's method with 
    backtracking line search [21], the regularized maximum likelihood problem:
    max_{a,b} Σ_i (t_i log(P_{a,b}(f_i)) + (1-t_i) log(1 - P_{a,b}(f_i)))"
    
    "Let N_+ and N_- be the number of positive and negative training examples, 
    and y_i the label of the i-th training example. Assuming a uniform uninformative 
    prior over the probabilities of the correct labels, the MAP estimates for the 
    target probabilities are:
    t_i = (N_+ + 1)/(N_+ + 2) if y_i = 1
    t_i = 1/(N_- + 2) otherwise"

Mathematical Background:
========================

The goal is to find parameters (a, b) that maximize the log-likelihood:

    L(a,b) = Σ_i [t_i log(P_{a,b}(f_i)) + (1-t_i) log(1 - P_{a,b}(f_i))]

where:
    - f_i: decision value (LOO prediction score)
    - P_{a,b}(f) = 1/(1 + exp(af + b)): sigmoid function
    - t_i: target probability (smoothed label)

This is solved using Newton-Raphson optimization with backtracking line search.

Author: [Your Name]
Date: 2026
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class PlattScaler:
    """
    Platt Scaling for probability calibration.
    
    Converts decision function values to calibrated probabilities using
    the sigmoid function P(y=1|f) = 1/(1 + exp(a*f + b)).
    
    Reference:
        Platt, "Probabilistic Outputs for Support Vector Machines and 
        Comparisons to Regularized Likelihood Methods", 1999
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-7,
        min_step: float = 1e-10
    ):
        """
        Initialize Platt scaler.
        
        Args:
            max_iter: Maximum Newton iterations
            tol: Convergence tolerance
            min_step: Minimum step size for line search
        """
        self.max_iter = max_iter
        self.tol = tol
        self.min_step = min_step
        
        # Fitted parameters
        self.a_ = None  # Slope parameter
        self.b_ = None  # Intercept parameter
        
        self._is_fitted = False
    
    def fit(self, f: np.ndarray, y: np.ndarray) -> 'PlattScaler':
        """
        Fit Platt scaling parameters using Newton's method.
        
        Args:
            f: Decision function values (n_samples,)
               These are typically LOO predictions from LSSVM
            y: True labels (n_samples,), values in {-1, +1} or {0, 1}
            
        Returns:
            self: Fitted scaler
            
        Reference:
            Paper Section 3.3: "The parameters a, b are set by solving, using 
            Newton's method with backtracking line search, the regularized 
            maximum likelihood problem"
        """
        f = np.asarray(f, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        
        # Convert labels to {0, 1} if necessary
        if set(np.unique(y)).issubset({-1, 1}):
            y = (y + 1) / 2  # Convert {-1, +1} to {0, 1}
        
        n = len(f)
        
        # Count positive and negative samples
        N_pos = np.sum(y == 1)
        N_neg = n - N_pos
        
        # Handle degenerate cases
        if N_pos == 0 or N_neg == 0:
            # All same class - use default sigmoid
            self.a_ = -1.0
            self.b_ = 0.0
            self._is_fitted = True
            return self
        
        # Compute target probabilities (smoothed labels)
        # Paper: "t_i = (N_+ + 1)/(N_+ + 2) if y_i = 1, t_i = 1/(N_- + 2) otherwise"
        t = np.zeros(n)
        t[y == 1] = (N_pos + 1) / (N_pos + 2)
        t[y == 0] = 1 / (N_neg + 2)
        
        # Initialize parameters using heuristic from Platt's paper
        # a should be negative for standard SVM convention (positive f -> positive class)
        a = 0.0
        b = np.log((N_neg + 1) / (N_pos + 1))
        
        # Constrain parameter bounds for stability
        a_max = 10.0
        b_max = 10.0
        
        # Newton's method with backtracking line search
        for iteration in range(self.max_iter):
            # Compute probabilities
            fval = a * f + b
            # Numerical stability: clip to avoid overflow in exp
            fval = np.clip(fval, -35, 35)
            p = 1.0 / (1.0 + np.exp(fval))
            
            # Clip probabilities away from 0 and 1 for numerical stability
            p = np.clip(p, 1e-10, 1 - 1e-10)
            
            # Compute gradient
            # dL/da = Σ f_i (t_i - p_i)
            # dL/db = Σ (t_i - p_i)
            diff = t - p
            grad_a = np.dot(f, diff)
            grad_b = np.sum(diff)
            
            # Check convergence
            grad_norm = np.sqrt(grad_a**2 + grad_b**2)
            if grad_norm < self.tol:
                break
            
            # Compute Hessian
            # d²L/da² = -Σ f_i² p_i (1-p_i)
            # d²L/db² = -Σ p_i (1-p_i)
            # d²L/dadb = -Σ f_i p_i (1-p_i)
            d = p * (1 - p)
            d = np.maximum(d, 1e-10)  # Avoid zeros
            
            H_aa = -np.dot(f**2, d)
            H_bb = -np.sum(d)
            H_ab = -np.dot(f, d)
            
            # Add regularization for stability
            reg = 1e-6
            H_aa = min(H_aa - reg, -reg)
            H_bb = min(H_bb - reg, -reg)
            
            # Hessian matrix (should be negative definite for maximization)
            det = H_aa * H_bb - H_ab * H_ab
            
            if det > -1e-10:
                # Hessian not negative definite, use gradient ascent
                step_size = 0.1
                a += step_size * grad_a / (np.abs(grad_a) + 1)
                b += step_size * grad_b / (np.abs(grad_b) + 1)
            else:
                # Newton direction: solve H @ delta = -gradient
                # Using direct formula for 2x2 inverse
                delta_a = (-H_bb * grad_a + H_ab * grad_b) / det
                delta_b = (H_ab * grad_a - H_aa * grad_b) / det
                
                # Limit step size
                step_limit = 1.0
                delta_norm = np.sqrt(delta_a**2 + delta_b**2)
                if delta_norm > step_limit:
                    delta_a *= step_limit / delta_norm
                    delta_b *= step_limit / delta_norm
                
                # Line search
                step = 1.0
                old_obj = self._objective(a, b, f, t)
                
                for _ in range(20):
                    a_new = a + step * delta_a
                    b_new = b + step * delta_b
                    
                    # Constrain parameters
                    a_new = np.clip(a_new, -a_max, a_max)
                    b_new = np.clip(b_new, -b_max, b_max)
                    
                    new_obj = self._objective(a_new, b_new, f, t)
                    
                    if new_obj > old_obj - 1e-10:
                        a = a_new
                        b = b_new
                        break
                    
                    step *= 0.5
        
        self.a_ = a
        self.b_ = b
        self._is_fitted = True
        
        return self
    
    def _objective(
        self, 
        a: float, 
        b: float, 
        f: np.ndarray, 
        t: np.ndarray
    ) -> float:
        """
        Compute the log-likelihood objective.
        
        L(a,b) = Σ [t_i log(p_i) + (1-t_i) log(1-p_i)]
        
        Args:
            a, b: Current parameters
            f: Decision values
            t: Target probabilities
            
        Returns:
            Log-likelihood value
        """
        fval = a * f + b
        fval = np.clip(fval, -35, 35)
        p = 1.0 / (1.0 + np.exp(fval))
        
        # Numerical stability
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        # Log-likelihood
        obj = np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))
        
        return obj
    
    def predict_proba(self, f: np.ndarray) -> np.ndarray:
        """
        Convert decision values to probabilities.
        
        P(y=1|f) = 1/(1 + exp(a*f + b))
        
        Args:
            f: Decision function values
            
        Returns:
            Probabilities P(y=1|f)
        """
        if not self._is_fitted:
            raise RuntimeError("PlattScaler must be fitted first")
        
        f = np.asarray(f, dtype=np.float64).ravel()
        fval = self.a_ * f + self.b_
        fval = np.clip(fval, -35, 35)
        
        proba = 1.0 / (1.0 + np.exp(fval))
        
        return proba
    
    def predict(self, f: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels using probability threshold.
        
        Reference:
            Paper: "and then use the probability threshold of 0.5"
        
        Args:
            f: Decision function values
            threshold: Probability threshold (default 0.5)
            
        Returns:
            Predicted labels in {-1, +1}
        """
        proba = self.predict_proba(f)
        labels = np.where(proba >= threshold, 1, -1)
        return labels
    
    def get_params(self) -> dict:
        """Return fitted parameters."""
        return {'a': self.a_, 'b': self.b_}


def balanced_error_rate(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    Compute Balanced Error Rate (BER).
    
    BER = (FPR + FNR) / 2 = (FP/(FP+TN) + FN/(FN+TP)) / 2
    
    This is the main evaluation metric used in the paper.
    
    Reference:
        Paper Section 3.3:
        "The balanced error rate is the average of false positive rate and 
        false negative rate. For brevity, we refer to the balanced error rate 
        simply as error or error rate."
        
        Paper Section 5.2:
        "On the UIUC dataset, our method reduces the false negative rate of [10] 
        by 35.3% while maintaining a similar false positive rate."
    
    Args:
        y_true: True labels in {-1, +1} or {0, 1}
                Convention: +1 or 1 = shadow (positive class)
        y_pred: Predicted labels in {-1, +1} or {0, 1}
        
    Returns:
        Balanced error rate (lower is better, range [0, 1])
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Convert to {0, 1} for easier computation
    if set(np.unique(y_true)).issubset({-1, 1}):
        y_true = (y_true + 1) / 2
    if set(np.unique(y_pred)).issubset({-1, 1}):
        y_pred = (y_pred + 1) / 2
    
    # Positive class (shadow): y = 1
    # Negative class (non-shadow): y = 0
    
    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # False Positive Rate: FPR = FP / (FP + TN)
    # False Negative Rate: FNR = FN / (FN + TP)
    
    # Avoid division by zero
    if (FP + TN) == 0:
        FPR = 0.0
    else:
        FPR = FP / (FP + TN)
    
    if (FN + TP) == 0:
        FNR = 0.0
    else:
        FNR = FN / (FN + TP)
    
    # Balanced Error Rate
    BER = (FPR + FNR) / 2.0
    
    return BER


def compute_loo_balanced_error(
    loo_scores: np.ndarray,
    y_true: np.ndarray,
    use_platt_scaling: bool = True
) -> Tuple[float, Optional[PlattScaler]]:
    """
    Compute LOO balanced error rate with optional Platt scaling.
    
    This is the optimization criterion used in the paper.
    
    Reference:
        Paper Section 3.3:
        "Our optimization criterion is the leave-one-out balanced error rate. 
        This requires having a threshold for separating between positive and 
        negative predictions. While the LSSVM classifier also has the default 
        threshold of 0, this setting is optimized for the total error rate 
        instead of the balanced error rate."
    
    Args:
        loo_scores: LOO decision scores from LSSVM (n_samples,)
        y_true: True labels (n_samples,)
        use_platt_scaling: Whether to use Platt scaling
        
    Returns:
        ber: Leave-one-out balanced error rate
        scaler: Fitted PlattScaler (if use_platt_scaling=True)
    """
    y_true = np.asarray(y_true).ravel()
    loo_scores = np.asarray(loo_scores).ravel()
    
    # Convert labels to {-1, +1}
    if set(np.unique(y_true)).issubset({0, 1}):
        y_true = 2 * y_true - 1
    
    if use_platt_scaling:
        # Fit Platt scaler on LOO scores
        scaler = PlattScaler()
        scaler.fit(loo_scores, y_true)
        
        # Predict using calibrated probabilities
        y_pred = scaler.predict(loo_scores, threshold=0.5)
        
        # Compute BER
        ber = balanced_error_rate(y_true, y_pred)
        
        return ber, scaler
    else:
        # Use sign of scores directly
        y_pred = np.sign(loo_scores)
        y_pred = np.where(y_pred == 0, 1, y_pred)
        
        ber = balanced_error_rate(y_true, y_pred)
        
        return ber, None


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute False Positive Rate.
    
    FPR = FP / (FP + TN) = FP / N_negative
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        False positive rate
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if set(np.unique(y_true)).issubset({-1, 1}):
        y_true = (y_true + 1) / 2
    if set(np.unique(y_pred)).issubset({-1, 1}):
        y_pred = (y_pred + 1) / 2
    
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    if (FP + TN) == 0:
        return 0.0
    
    return FP / (FP + TN)


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute False Negative Rate.
    
    FNR = FN / (FN + TP) = FN / N_positive
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        False negative rate
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if set(np.unique(y_true)).issubset({-1, 1}):
        y_true = (y_true + 1) / 2
    if set(np.unique(y_pred)).issubset({-1, 1}):
        y_pred = (y_pred + 1) / 2
    
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    if (FN + TP) == 0:
        return 0.0
    
    return FN / (FN + TP)

