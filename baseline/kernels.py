"""
Lightweight baseline kernel implementations.

Supports:
1) Unary SVM: concatenate all channel features and use a single RBF kernel.
2) MK-SVM: fixed-weight multi-kernel with per-channel RBFs.

These classes expose compute/compute_cross similar to project kernels so
they can be used with the existing LSSVM pipeline.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


def estimate_median_sigma(X: np.ndarray, max_samples: int = 2000) -> float:
    """
    Estimate an RBF sigma via median heuristic on a subsample.
    """
    if X.shape[0] <= 1:
        return 1.0
    n = min(max_samples, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    Xs = X[idx]
    # Pairwise squared distances
    dists = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
    # Take upper triangle without diagonal
    triu = dists[np.triu_indices_from(dists, k=1)]
    med = np.median(triu) if triu.size > 0 else 1.0
    # Avoid zero sigma
    return float(np.sqrt(med / 2.0 + 1e-10))


def _rbf_kernel(X: np.ndarray, Y: Optional[np.ndarray], sigma: float) -> np.ndarray:
    """
    Compute RBF kernel matrix between X and Y (or self-kernel if Y is None).
    """
    if Y is None:
        Y = X
    X_norm = np.sum(X ** 2, axis=1)[:, None]
    Y_norm = np.sum(Y ** 2, axis=1)[None, :]
    dists = X_norm + Y_norm - 2 * X @ Y.T
    dists = np.maximum(dists, 0.0)
    denom = 2.0 * (sigma ** 2) + 1e-10
    return np.exp(-dists / denom)


class ConcatRBFKernel:
    """
    Single RBF kernel on concatenated channel features (Unary SVM baseline).
    """

    def __init__(self, train_features: Dict[str, np.ndarray], sigma: Optional[float] = None):
        self.X_train = self._concat(train_features)
        self.sigma = sigma or estimate_median_sigma(self.X_train)

    @staticmethod
    def _concat(feats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([feats[k] for k in ['L', 'a', 'b', 't']], axis=1)

    def compute(self, features_by_channel: Dict[str, np.ndarray], **_: Dict) -> np.ndarray:
        X = self._concat(features_by_channel)
        return _rbf_kernel(X, None, self.sigma)

    def compute_cross(
        self,
        train_features: Dict[str, np.ndarray],
        test_features: Dict[str, np.ndarray],
        **_: Dict
    ) -> np.ndarray:
        X_train = self._concat(train_features)
        X_test = self._concat(test_features)
        # Return transposed matrix (n_train, n_test) to match project LSSVM convention
        return _rbf_kernel(X_test, X_train, self.sigma).T


class MultiChannelFixedKernel:
    """
    Fixed-weight multi-kernel with per-channel RBFs (MK-SVM baseline).
    """

    def __init__(
        self,
        train_features: Dict[str, np.ndarray],
        weights: Dict[str, float],
        sigmas: Optional[Dict[str, float]] = None,
    ):
        self.train_features = train_features
        self.weights = weights
        if sigmas is None:
            sigmas = {}
            for k, v in train_features.items():
                sigmas[k] = estimate_median_sigma(v)
        self.sigmas = sigmas

    def compute(self, features_by_channel: Dict[str, np.ndarray], **_: Dict) -> np.ndarray:
        K = None
        for ch, X_ch in features_by_channel.items():
            sigma = self.sigmas[ch]
            K_ch = _rbf_kernel(X_ch, None, sigma)
            if K is None:
                K = self.weights[ch] * K_ch
            else:
                K += self.weights[ch] * K_ch
        return K

    def compute_cross(
        self,
        train_features: Dict[str, np.ndarray],
        test_features: Dict[str, np.ndarray],
        **_: Dict
    ) -> np.ndarray:
        K = None
        for ch, X_test in test_features.items():
            X_train = train_features[ch]
            sigma = self.sigmas[ch]
            K_ch = _rbf_kernel(X_test, X_train, sigma)
            if K is None:
                K = self.weights[ch] * K_ch
            else:
                K += self.weights[ch] * K_ch
        # Return transposed matrix (n_train, n_test) to match project LSSVM convention
        return K.T if K is not None else np.array([])


def build_baseline_kernel(
    method: str,
    train_features: Dict[str, np.ndarray],
    config: Dict,
    fixed_weights: Optional[Dict[str, float]] = None,
    fixed_sigmas: Optional[Dict[str, float]] = None,
) -> Tuple[object, Dict]:
    """
    Factory to build baseline kernels and return their params for saving.

    Returns:
        kernel: object with compute / compute_cross
        params: dict of params for persistence (weights, sigmas, sigma_single)
    """
    if method == 'unary_svm':
        sigma = config.get('unary_sigma', None)
        kernel = ConcatRBFKernel(train_features, sigma=sigma)
        params = {'sigma_single': kernel.sigma}
        return kernel, params

    if method == 'mk_svm':
        weights = fixed_weights or {k: 0.25 for k in ['L', 'a', 'b', 't']}
        sigmas = fixed_sigmas or None
        kernel = MultiChannelFixedKernel(train_features, weights=weights, sigmas=sigmas)
        params = {'weights': weights, 'sigmas': kernel.sigmas}
        return kernel, params

    raise ValueError(f"Unsupported baseline method: {method}")
