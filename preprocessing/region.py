"""
Region Generation Module.

This module implements Mean-shift clustering for merging superpixels into regions,
as described in the paper.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 3:
    "Given an image, we first segment it into regions using a two-step process [34]: 
    1) apply SLIC superpixel segmentation to oversegment the image and obtain a 
    set of superpixels; 
    2) apply Mean-shift clustering [4] and merge superpixels in the same cluster 
    into a larger region."
    
    [4] Comaniciu & Meer, "Mean Shift: A Robust Approach Toward Feature Space Analysis", TPAMI 2002
    [34] Yago et al., "Detecting Ground Shadows in Outdoor Consumer Photographs", ECCV 2010

Author: [Your Name]
Date: 2026
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2lab
from sklearn.cluster import MeanShift, estimate_bandwidth
from typing import Tuple, Dict, List, Optional
import warnings


class MeanShiftRegionGenerator:
    """
    Generate regions by merging superpixels using Mean-shift clustering.
    
    This is the method specified in the paper:
        "apply Mean-shift clustering [4] and merge superpixels in the same 
        cluster into a larger region."
    
    Reference:
        [4] Comaniciu & Meer, "Mean Shift: A Robust Approach Toward Feature 
            Space Analysis", TPAMI 2002
    
    Mean-shift clustering works by:
    1. Computing mean LAB color for each superpixel
    2. Running mean-shift in the LAB color space
    3. Merging superpixels that converge to the same mode
    
    Note on bandwidth parameter:
        The paper does not specify the bandwidth. We use sklearn's 
        estimate_bandwidth() with quantile parameter, which adapts to data.
        
        IMPORTANT: If the paper's reference [34] specifies bandwidth, 
        please update this implementation accordingly.
    
    Attributes:
        bandwidth: Mean-shift bandwidth (None = auto-estimate)
        quantile: Quantile for bandwidth estimation (used if bandwidth is None)
        min_bin_freq: Minimum bin frequency for mean-shift
    """
    
    def __init__(
        self,
        bandwidth: Optional[float] = None,
        quantile: float = 0.3,
        min_bin_freq: int = 1,
        use_spatial: bool = False,
        spatial_weight: float = 0.1
    ):
        """
        Initialize Mean-shift region generator.
        
        Args:
            bandwidth: Mean-shift bandwidth. If None, auto-estimated.
                      Paper does not specify - this is a key parameter to tune.
            quantile: Quantile for bandwidth estimation (default 0.3).
                     Smaller quantile = smaller bandwidth = more regions.
            min_bin_freq: Minimum number of samples in a bin for mean-shift.
            use_spatial: Whether to include spatial position in clustering.
                        Paper [4] mentions joint spatial-color clustering,
                        but paper [34] may only use color. Set False by default.
            spatial_weight: Weight for spatial coordinates if use_spatial=True.
        
        Note:
            The exact bandwidth and feature space used in the paper is unclear.
            This implementation uses LAB color space, which is common for 
            perceptually uniform color clustering.
        """
        self.bandwidth = bandwidth
        self.quantile = quantile
        self.min_bin_freq = min_bin_freq
        self.use_spatial = use_spatial
        self.spatial_weight = spatial_weight
        
        # Store results
        self._region_labels = None
        self._n_actual_regions = None
        self._superpixel_to_region = None
        self._region_properties = None
        self._estimated_bandwidth = None
    
    def generate_regions(
        self,
        image: np.ndarray,
        superpixel_labels: np.ndarray,
        superpixel_adjacency: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Merge superpixels into regions using Mean-shift clustering.
        
        Reference:
            Paper Section 3: "apply Mean-shift clustering [4] and merge 
            superpixels in the same cluster into a larger region."
        
        Args:
            image: Original RGB image (H, W, 3)
            superpixel_labels: Superpixel label map from SuperpixelSegmenter
            superpixel_adjacency: Not used for mean-shift (kept for API compatibility)
            
        Returns:
            region_labels: Integer label array (H, W) where each value represents a region
        """
        # Normalize image
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        n_superpixels = superpixel_labels.max() + 1
        
        # Compute superpixel features for clustering (LAB color)
        features = self._compute_superpixel_features(image, superpixel_labels)
        
        # Estimate bandwidth if not provided
        if self.bandwidth is None:
            self._estimated_bandwidth = estimate_bandwidth(
                features, 
                quantile=self.quantile,
                n_samples=min(500, len(features))
            )
            # Ensure minimum bandwidth to avoid too many clusters
            if self._estimated_bandwidth < 0.1:
                self._estimated_bandwidth = 0.1
            bandwidth = self._estimated_bandwidth
        else:
            bandwidth = self.bandwidth
            self._estimated_bandwidth = bandwidth
        
        # Run Mean-shift clustering
        # Paper [4]: "Mean Shift: A Robust Approach Toward Feature Space Analysis"
        ms = MeanShift(
            bandwidth=bandwidth,
            min_bin_freq=self.min_bin_freq,
            bin_seeding=True,  # Faster for larger datasets
            cluster_all=True   # Ensure all points are assigned
        )
        
        cluster_labels = ms.fit_predict(features)
        
        # Create mapping from superpixel to region
        self._superpixel_to_region = {
            sp_id: cluster_labels[sp_id] for sp_id in range(n_superpixels)
        }
        
        # Generate region label map
        self._region_labels = np.zeros_like(superpixel_labels)
        for sp_id, region_id in self._superpixel_to_region.items():
            self._region_labels[superpixel_labels == sp_id] = region_id
        
        # Ensure contiguous labels starting from 0
        unique_regions = np.unique(self._region_labels)
        self._n_actual_regions = len(unique_regions)
        
        if not np.array_equal(unique_regions, np.arange(self._n_actual_regions)):
            label_mapping = {old: new for new, old in enumerate(unique_regions)}
            self._region_labels = np.vectorize(label_mapping.get)(self._region_labels)
            self._superpixel_to_region = {
                sp_id: label_mapping[region_id] 
                for sp_id, region_id in self._superpixel_to_region.items()
            }
        
        return self._region_labels
    
    def _compute_superpixel_features(
        self,
        image: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute feature vectors for each superpixel for Mean-shift clustering.
        
        Reference:
            Paper uses Mean-shift clustering [4] to merge superpixels.
            The typical feature space for Mean-shift in image segmentation is
            LAB color (for perceptual uniformity).
            
            Comaniciu & Meer [4] describe joint spatial-color clustering,
            but the exact feature space for superpixel merging is not specified.
        
        Features:
            - Mean LAB color (primary - used for color similarity)
            - Optionally: Normalized spatial centroid (if use_spatial=True)
        
        Args:
            image: RGB image (H, W, 3), normalized to [0, 1]
            labels: Superpixel label map (H, W)
            
        Returns:
            features: Array of shape (n_superpixels, feature_dim)
                     feature_dim = 3 (LAB only) or 5 (LAB + position)
        """
        # Convert to LAB color space - perceptually uniform
        lab_image = rgb2lab(image)
        
        n_superpixels = labels.max() + 1
        H, W = labels.shape
        
        # Feature dimension depends on whether spatial features are used
        feature_dim = 5 if self.use_spatial else 3
        features = np.zeros((n_superpixels, feature_dim))
        
        for sp_id in range(n_superpixels):
            mask = labels == sp_id
            
            if not mask.any():
                continue
            
            # Mean LAB color
            lab_values = lab_image[mask]
            mean_lab = lab_values.mean(axis=0)
            
            # Normalize LAB values to [0, 1] range for clustering
            # L: [0, 100] -> [0, 1]
            # a, b: [-128, 127] -> approximately [-1, 1]
            features[sp_id, 0] = mean_lab[0] / 100.0
            features[sp_id, 1] = (mean_lab[1] + 128) / 255.0
            features[sp_id, 2] = (mean_lab[2] + 128) / 255.0
            
            # Optionally add spatial position
            if self.use_spatial:
                coords = np.argwhere(mask)
                centroid = coords.mean(axis=0)
                normalized_centroid = centroid / np.array([H, W])
                features[sp_id, 3:] = normalized_centroid * self.spatial_weight
        
        return features
    
    @property
    def estimated_bandwidth(self) -> Optional[float]:
        """Return the estimated/used bandwidth."""
        return self._estimated_bandwidth
    
    def get_region_properties(self, image: np.ndarray) -> Dict[int, Dict]:
        """
        Compute properties for each region.
        
        Args:
            image: Original RGB image (H, W, 3)
            
        Returns:
            Dictionary mapping region ID to properties:
                - 'centroid': (row, col) center position
                - 'area': number of pixels
                - 'bbox': bounding box
                - 'mean_color_rgb': mean RGB color
                - 'mean_color_lab': mean LAB color
                - 'superpixels': list of superpixel IDs in this region
        """
        if self._region_labels is None:
            raise RuntimeError("Must call generate_regions() first")
        
        # Normalize image
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        lab_image = rgb2lab(image)
        
        properties = {}
        
        for region_id in range(self._n_actual_regions):
            mask = self._region_labels == region_id
            coords = np.argwhere(mask)
            
            if len(coords) == 0:
                continue
            
            # Basic properties
            centroid = coords.mean(axis=0)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            
            # Color properties
            rgb_values = image[mask]
            lab_values = lab_image[mask]
            
            # Find which superpixels belong to this region
            superpixels_in_region = [
                sp_id for sp_id, r_id in self._superpixel_to_region.items()
                if r_id == region_id
            ]
            
            properties[region_id] = {
                'centroid': tuple(centroid),
                'area': len(coords),
                'bbox': (min_coords[0], min_coords[1], max_coords[0], max_coords[1]),
                'mean_color_rgb': rgb_values.mean(axis=0),
                'mean_color_lab': lab_values.mean(axis=0),
                'superpixels': superpixels_in_region,
                'pixel_coords': coords,
            }
        
        self._region_properties = properties
        return properties
    
    def get_region_adjacency(self) -> np.ndarray:
        """
        Compute adjacency matrix for regions.
        
        Returns:
            adjacency: Binary matrix (n_regions, n_regions)
        """
        if self._region_labels is None:
            raise RuntimeError("Must call generate_regions() first")
        
        n = self._n_actual_regions
        adjacency = np.zeros((n, n), dtype=np.uint8)
        
        # Check horizontal adjacency
        h_diff = self._region_labels[:, :-1] != self._region_labels[:, 1:]
        for i, j in zip(*np.where(h_diff)):
            r1, r2 = self._region_labels[i, j], self._region_labels[i, j + 1]
            adjacency[r1, r2] = 1
            adjacency[r2, r1] = 1
        
        # Check vertical adjacency
        v_diff = self._region_labels[:-1, :] != self._region_labels[1:, :]
        for i, j in zip(*np.where(v_diff)):
            r1, r2 = self._region_labels[i, j], self._region_labels[i + 1, j]
            adjacency[r1, r2] = 1
            adjacency[r2, r1] = 1
        
        return adjacency
    
    def get_boundary_features(self, image: np.ndarray) -> Dict[Tuple[int, int], Dict]:
        """
        Compute features for boundaries between adjacent regions.
        
        These features are used for pairwise potentials in the MRF.
        
        Args:
            image: Original RGB image
            
        Returns:
            Dictionary mapping (region_i, region_j) pairs to boundary features:
                - 'boundary_length': number of boundary pixels
                - 'color_difference': LAB color difference across boundary
                - 'boundary_strength': average gradient magnitude at boundary
        """
        if self._region_labels is None:
            raise RuntimeError("Must call generate_regions() first")
        
        # Normalize image
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        lab_image = rgb2lab(image)
        
        # Compute gradient magnitude
        gray = np.mean(image, axis=2)
        gy, gx = np.gradient(gray)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        boundary_features = {}
        
        # Find all boundary pixels
        # Horizontal boundaries
        h_diff = self._region_labels[:, :-1] != self._region_labels[:, 1:]
        for i, j in zip(*np.where(h_diff)):
            r1, r2 = int(self._region_labels[i, j]), int(self._region_labels[i, j + 1])
            key = (min(r1, r2), max(r1, r2))
            
            if key not in boundary_features:
                boundary_features[key] = {
                    'boundary_length': 0,
                    'gradient_sum': 0,
                    'lab_diff_sum': np.zeros(3),
                }
            
            boundary_features[key]['boundary_length'] += 1
            boundary_features[key]['gradient_sum'] += gradient_mag[i, j]
            
            # LAB color difference
            lab_diff = np.abs(lab_image[i, j] - lab_image[i, j + 1])
            boundary_features[key]['lab_diff_sum'] += lab_diff
        
        # Vertical boundaries
        v_diff = self._region_labels[:-1, :] != self._region_labels[1:, :]
        for i, j in zip(*np.where(v_diff)):
            r1, r2 = int(self._region_labels[i, j]), int(self._region_labels[i + 1, j])
            key = (min(r1, r2), max(r1, r2))
            
            if key not in boundary_features:
                boundary_features[key] = {
                    'boundary_length': 0,
                    'gradient_sum': 0,
                    'lab_diff_sum': np.zeros(3),
                }
            
            boundary_features[key]['boundary_length'] += 1
            boundary_features[key]['gradient_sum'] += gradient_mag[i, j]
            
            lab_diff = np.abs(lab_image[i, j] - lab_image[i + 1, j])
            boundary_features[key]['lab_diff_sum'] += lab_diff
        
        # Compute averages
        for key in boundary_features:
            n = boundary_features[key]['boundary_length']
            boundary_features[key]['boundary_strength'] = (
                boundary_features[key]['gradient_sum'] / n if n > 0 else 0
            )
            boundary_features[key]['color_difference'] = (
                boundary_features[key]['lab_diff_sum'] / n if n > 0 else np.zeros(3)
            )
            # Clean up temporary fields
            del boundary_features[key]['gradient_sum']
            del boundary_features[key]['lab_diff_sum']
        
        return boundary_features
    
    def visualize(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Visualize region boundaries overlaid on the original image.
        
        Args:
            image: Original RGB image
            color: Boundary color in RGB
            
        Returns:
            Visualization image with region boundaries
        """
        if self._region_labels is None:
            raise RuntimeError("Must call generate_regions() first")
        
        # Normalize image
        if image.max() > 1.0:
            vis_image = image.astype(np.float64) / 255.0
        else:
            vis_image = image.copy()
        
        color_normalized = tuple(c / 255.0 for c in color)
        
        vis_image = mark_boundaries(
            vis_image,
            self._region_labels,
            color=color_normalized,
            mode='outer'
        )
        
        return (vis_image * 255).astype(np.uint8)
    
    @property
    def region_labels(self) -> Optional[np.ndarray]:
        """Return the region label map."""
        return self._region_labels
    
    @property
    def labels_(self) -> Optional[np.ndarray]:
        """Return the region label map (sklearn-style alias)."""
        return self._region_labels
    
    @property
    def n_regions(self) -> Optional[int]:
        """Return the actual number of regions."""
        return self._n_actual_regions
    
    @property
    def superpixel_to_region(self) -> Optional[Dict[int, int]]:
        """Return mapping from superpixel ID to region ID."""
        return self._superpixel_to_region
    
    def fit(
        self,
        image: np.ndarray,
        superpixel_labels: np.ndarray
    ) -> 'MeanShiftRegionGenerator':
        """
        Fit the region generator (sklearn-style interface).
        
        This is an alias for generate_regions() for API compatibility.
        
        Args:
            image: Original RGB image
            superpixel_labels: Superpixel label map
            
        Returns:
            self
        """
        self.generate_regions(image, superpixel_labels)
        return self


class HierarchicalRegionGenerator:
    """
    Generate regions by merging superpixels using hierarchical clustering.
    
    This is an ALTERNATIVE method to Mean-shift, kept for comparison.
    
    NOTE: The paper specifies Mean-shift clustering [4] for region generation.
    Use MeanShiftRegionGenerator for paper-compliant implementation.
    
    Attributes:
        n_regions (int): Target number of regions
        linkage_method (str): Linkage criterion for hierarchical clustering
    """
    
    def __init__(
        self,
        n_regions: int = 80,
        linkage_method: str = 'ward',
        affinity: str = 'euclidean',
        color_weight: float = 1.0,
        position_weight: float = 0.5
    ):
        """
        Initialize hierarchical region generator.
        
        Args:
            n_regions: Target number of regions after merging
            linkage_method: Linkage criterion ('ward', 'complete', 'average')
            affinity: Distance metric for clustering
            color_weight: Weight for color features
            position_weight: Weight for spatial position
        """
        self._target_n_regions = n_regions
        self.linkage_method = linkage_method
        self.affinity = affinity
        self.color_weight = color_weight
        self.position_weight = position_weight
        
        self._region_labels = None
        self._n_actual_regions = None
        self._superpixel_to_region = None
    
    def generate_regions(
        self,
        image: np.ndarray,
        superpixel_labels: np.ndarray,
        superpixel_adjacency: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Merge superpixels using hierarchical clustering."""
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        n_superpixels = superpixel_labels.max() + 1
        
        if n_superpixels <= self._target_n_regions:
            self._region_labels = superpixel_labels.copy()
            self._n_actual_regions = n_superpixels
            self._superpixel_to_region = {i: i for i in range(n_superpixels)}
            return self._region_labels
        
        features = self._compute_superpixel_features(image, superpixel_labels)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.linkage_method == 'ward':
                Z = linkage(features, method='ward')
            else:
                distance_matrix = pdist(features, metric=self.affinity)
                Z = linkage(distance_matrix, method=self.linkage_method)
        
        cluster_labels = fcluster(Z, t=self._target_n_regions, criterion='maxclust') - 1
        
        self._superpixel_to_region = {
            sp_id: cluster_labels[sp_id] for sp_id in range(n_superpixels)
        }
        
        self._region_labels = np.zeros_like(superpixel_labels)
        for sp_id, region_id in self._superpixel_to_region.items():
            self._region_labels[superpixel_labels == sp_id] = region_id
        
        unique_regions = np.unique(self._region_labels)
        self._n_actual_regions = len(unique_regions)
        
        if not np.array_equal(unique_regions, np.arange(self._n_actual_regions)):
            label_mapping = {old: new for new, old in enumerate(unique_regions)}
            self._region_labels = np.vectorize(label_mapping.get)(self._region_labels)
            self._superpixel_to_region = {
                sp_id: label_mapping[region_id] 
                for sp_id, region_id in self._superpixel_to_region.items()
            }
        
        return self._region_labels
    
    def _compute_superpixel_features(self, image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute LAB color + position features for each superpixel."""
        lab_image = rgb2lab(image)
        n_superpixels = labels.max() + 1
        H, W = labels.shape
        
        features = np.zeros((n_superpixels, 5))
        
        for sp_id in range(n_superpixels):
            mask = labels == sp_id
            if not mask.any():
                continue
            
            lab_values = lab_image[mask]
            mean_lab = lab_values.mean(axis=0)
            mean_lab[0] /= 100.0
            mean_lab[1:] /= 128.0
            
            coords = np.argwhere(mask)
            centroid = coords.mean(axis=0)
            normalized_centroid = centroid / np.array([H, W])
            
            features[sp_id, :3] = mean_lab * self.color_weight
            features[sp_id, 3:] = normalized_centroid * self.position_weight
        
        return features
    
    def fit(self, image: np.ndarray, superpixel_labels: np.ndarray) -> 'HierarchicalRegionGenerator':
        """Fit the region generator (sklearn-style interface)."""
        self.generate_regions(image, superpixel_labels)
        return self
    
    @property
    def labels_(self) -> Optional[np.ndarray]:
        return self._region_labels
    
    @property
    def n_regions(self) -> Optional[int]:
        return self._n_actual_regions


# Paper-compliant alias: Use Mean-shift as the default
# Reference: Paper Section 3 - "apply Mean-shift clustering [4]"
RegionGenerator = MeanShiftRegionGenerator

