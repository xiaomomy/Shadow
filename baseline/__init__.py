"""
Baseline utilities for alternative classifiers (UnarySVM, MK-SVM).

These helpers provide lightweight kernel wrappers that mimic the interface
of the project kernels, exposing ``compute`` and ``compute_cross`` so they
can be plugged into the existing LSSVM pipeline without modifying callers.
All baselines share the same feature extraction pipeline and dataset
processing as the main LooKOP implementation.
"""

from .kernels import (
    ConcatRBFKernel,
    MultiChannelFixedKernel,
    build_baseline_kernel,
    estimate_median_sigma,
)

__all__ = [
    "ConcatRBFKernel",
    "MultiChannelFixedKernel",
    "build_baseline_kernel",
    "estimate_median_sigma",
]
