"""
Preprocessing module for shadow detection.

This module implements Stage 1-2 of the shadow detection pipeline:
    Stage 1: Image segmentation (superpixels → regions)
    Stage 2: Feature extraction (chromatic, intensity, texture)

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 3:
    "we first segment it into regions using a two-step process [34]: 
    1) apply SLIC superpixel segmentation; 
    2) apply Mean-shift clustering [4] and merge superpixels in the same cluster"
"""

from .superpixel import SuperpixelSegmenter
from .region import (
    RegionGenerator,  # Alias for MeanShiftRegionGenerator (paper-compliant)
    MeanShiftRegionGenerator,
    HierarchicalRegionGenerator  # Alternative method for comparison
)
from .features import FeatureExtractor, PaperCompliantFeatureExtractor
from .texton import (
    MR8FilterBank,
    TextonDictionary,
    TextonFeatureExtractor,
    create_default_texton_extractor
)

__all__ = [
    'SuperpixelSegmenter', 
    # Region generation - Paper uses Mean-shift [4]
    'RegionGenerator',  # Default = MeanShiftRegionGenerator
    'MeanShiftRegionGenerator',
    'HierarchicalRegionGenerator',  # Alternative
    # Feature extraction
    'FeatureExtractor',
    'PaperCompliantFeatureExtractor',
    # Texton (MR8 + k-means) - Paper Section 3.1
    'MR8FilterBank',
    'TextonDictionary',
    'TextonFeatureExtractor',
    'create_default_texton_extractor'
]

