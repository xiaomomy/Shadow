"""
Superpixel Segmentation Module.

This module implements superpixel segmentation as the first step of the 
shadow detection pipeline. The paper mentions dividing images into multiple 
regions where "each region is a group of superpixels".

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    Achanta et al., "SLIC Superpixels", TPAMI 2012

Author: [Your Name]
Date: 2026
"""

import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.color import rgb2lab
from typing import Tuple, Dict, Optional
import warnings


class SuperpixelSegmenter:
    """
    Superpixel segmentation using SLIC algorithm.
    
    The paper states: "Given an image, we first divide it into multiple regions,
    where each region is a group of superpixels."
    
    Attributes:
        n_segments (int): Target number of superpixels
        compactness (float): Balance between color and spatial proximity
        sigma (float): Gaussian smoothing kernel width
        convert2lab (bool): Whether to convert to LAB color space
    """
    
    def __init__(
        self,
        n_segments: int = 500,
        compactness: float = 10.0,
        sigma: float = 1.0,
        convert2lab: bool = True,
        enforce_connectivity: bool = True,
        min_size_factor: float = 0.25,
        max_size_factor: float = 3.0
    ):
        """
        Initialize the superpixel segmenter.
        
        Args:
            n_segments: Approximate number of superpixels to generate
            compactness: Balances color proximity and space proximity.
                         Higher values give more weight to space proximity.
            sigma: Width of Gaussian smoothing kernel for pre-processing
            convert2lab: Convert image to LAB color space before segmentation
            enforce_connectivity: Enforce connectivity of segments
            min_size_factor: Minimum segment size as fraction of average
            max_size_factor: Maximum segment size as fraction of average
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.convert2lab = convert2lab
        self.enforce_connectivity = enforce_connectivity
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        
        # Store results
        self._labels = None
        self._n_superpixels = None
        self._properties = None
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform superpixel segmentation on the input image.
        
        Args:
            image: Input RGB image with shape (H, W, 3), values in [0, 255] or [0, 1]
            
        Returns:
            labels: Integer label array of shape (H, W) where each unique value
                   represents a distinct superpixel
        """
        # Validate input
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        # Apply SLIC algorithm
        # Note: skimage's SLIC internally converts to LAB if convert2lab=True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._labels = slic(
                image,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
                convert2lab=self.convert2lab,
                enforce_connectivity=self.enforce_connectivity,
                min_size_factor=self.min_size_factor,
                max_size_factor=self.max_size_factor,
                start_label=0  # Labels start from 0
            )
        
        # Ensure contiguous labels starting from 0
        unique_labels = np.unique(self._labels)
        self._n_superpixels = len(unique_labels)
        
        # Relabel to ensure consecutive integers
        if not np.array_equal(unique_labels, np.arange(self._n_superpixels)):
            label_mapping = {old: new for new, old in enumerate(unique_labels)}
            self._labels = np.vectorize(label_mapping.get)(self._labels)
        
        return self._labels
    
    def get_superpixel_properties(self, image: np.ndarray) -> Dict[int, Dict]:
        """
        Compute properties for each superpixel.
        
        Args:
            image: Original RGB image (H, W, 3)
            
        Returns:
            Dictionary mapping superpixel ID to its properties:
                - 'centroid': (row, col) center position
                - 'area': number of pixels
                - 'bbox': bounding box (min_row, min_col, max_row, max_col)
                - 'mean_color': mean RGB color
                - 'pixels': list of (row, col) pixel coordinates
        """
        if self._labels is None:
            raise RuntimeError("Must call segment() before get_superpixel_properties()")
        
        # Normalize image
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        properties = {}
        
        for sp_id in range(self._n_superpixels):
            mask = self._labels == sp_id
            coords = np.argwhere(mask)
            
            if len(coords) == 0:
                continue
            
            # Compute properties
            centroid = coords.mean(axis=0)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            
            # Extract pixel colors
            pixel_colors = image[mask]
            mean_color = pixel_colors.mean(axis=0)
            
            properties[sp_id] = {
                'centroid': tuple(centroid),
                'area': len(coords),
                'bbox': (min_coords[0], min_coords[1], max_coords[0], max_coords[1]),
                'mean_color': mean_color,
                'pixel_coords': coords,
            }
        
        self._properties = properties
        return properties
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Compute adjacency matrix for superpixels.
        
        Two superpixels are adjacent if they share at least one boundary pixel.
        
        Returns:
            adjacency: Binary matrix of shape (n_superpixels, n_superpixels)
                      adjacency[i,j] = 1 if superpixels i and j are adjacent
        """
        if self._labels is None:
            raise RuntimeError("Must call segment() before get_adjacency_matrix()")
        
        n = self._n_superpixels
        adjacency = np.zeros((n, n), dtype=np.uint8)
        
        # Check horizontal adjacency
        horizontal_diff = self._labels[:, :-1] != self._labels[:, 1:]
        for i, j in zip(*np.where(horizontal_diff)):
            sp1, sp2 = self._labels[i, j], self._labels[i, j + 1]
            adjacency[sp1, sp2] = 1
            adjacency[sp2, sp1] = 1
        
        # Check vertical adjacency
        vertical_diff = self._labels[:-1, :] != self._labels[1:, :]
        for i, j in zip(*np.where(vertical_diff)):
            sp1, sp2 = self._labels[i, j], self._labels[i + 1, j]
            adjacency[sp1, sp2] = 1
            adjacency[sp2, sp1] = 1
        
        return adjacency
    
    def visualize(self, image: np.ndarray, color: Tuple[int, int, int] = (255, 255, 0)) -> np.ndarray:
        """
        Visualize superpixel boundaries overlaid on the original image.
        
        Args:
            image: Original RGB image
            color: Boundary color in RGB
            
        Returns:
            Visualization image with superpixel boundaries
        """
        if self._labels is None:
            raise RuntimeError("Must call segment() before visualize()")
        
        # Normalize image for visualization
        if image.max() > 1.0:
            vis_image = image.astype(np.float64) / 255.0
        else:
            vis_image = image.copy()
        
        # Normalize color
        color_normalized = tuple(c / 255.0 for c in color)
        
        # Mark boundaries
        vis_image = mark_boundaries(
            vis_image, 
            self._labels, 
            color=color_normalized,
            mode='outer'
        )
        
        return (vis_image * 255).astype(np.uint8)
    
    @property
    def labels(self) -> Optional[np.ndarray]:
        """Return the superpixel label map."""
        return self._labels
    
    @property
    def n_superpixels(self) -> Optional[int]:
        """Return the number of superpixels."""
        return self._n_superpixels


def compute_superpixel_features_for_clustering(
    image: np.ndarray,
    labels: np.ndarray,
    color_weight: float = 1.0,
    position_weight: float = 0.5
) -> np.ndarray:
    """
    Compute feature vectors for each superpixel for subsequent clustering.
    
    This function computes features used to merge superpixels into regions.
    Features include: mean LAB color and normalized centroid position.
    
    Args:
        image: RGB image (H, W, 3)
        labels: Superpixel label map (H, W)
        color_weight: Weight for color features
        position_weight: Weight for position features
        
    Returns:
        features: Array of shape (n_superpixels, feature_dim)
    """
    # Convert to LAB for perceptually uniform color representation
    if image.max() > 1.0:
        image = image.astype(np.float64) / 255.0
    
    lab_image = rgb2lab(image)
    
    n_superpixels = labels.max() + 1
    H, W = labels.shape
    
    # Feature vector: [L, a, b, normalized_row, normalized_col]
    features = np.zeros((n_superpixels, 5))
    
    for sp_id in range(n_superpixels):
        mask = labels == sp_id
        
        if not mask.any():
            continue
        
        # Mean LAB color
        lab_values = lab_image[mask]
        mean_lab = lab_values.mean(axis=0)
        
        # Normalized centroid position
        coords = np.argwhere(mask)
        centroid = coords.mean(axis=0)
        normalized_centroid = centroid / np.array([H, W])
        
        # Apply weights
        features[sp_id, :3] = mean_lab * color_weight
        features[sp_id, 3:] = normalized_centroid * position_weight * 100  # Scale position
    
    return features

