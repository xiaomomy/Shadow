"""
Feature Extraction Module.

This module implements feature extraction for shadow detection regions.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 3.1:
    "For each region, we compute a 21-bin histogram for each of the components 
    (L*,a*,b*) of the perceptually uniform color space CIELAB."
    
    "To represent texture, we compute a 128-bin texton histogram. We run the 
    full MR8 filter set [33] in the whole dataset and cluster the filter 
    responses into 128 textons using k-means."
    
    Note: The paper uses 4 separate kernels with different distance metrics:
    - L*, a*, b* histograms: EMD (Earth Mover's Distance)
    - Texton histogram: χ² distance

Author: [Your Name]
Date: 2026
"""

import numpy as np
from skimage.color import rgb2lab, rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from scipy.ndimage import generic_filter
from typing import Dict, List, Tuple, Optional
import warnings


class FeatureExtractor:
    """
    Extract features for each region in the image.
    
    The paper uses three main types of features:
    1. Chromatic features: Color information in LAB space
    2. Intensity features: Grayscale intensity statistics
    3. Texture features: Local Binary Pattern (LBP) descriptors
    
    These features are combined into a discriminative kernel for classification.
    """
    
    def __init__(
        self,
        # Chromatic feature parameters
        chromatic_n_bins: int = 16,
        chromatic_compute_stats: bool = True,
        # Intensity feature parameters
        intensity_n_bins: int = 32,
        intensity_percentiles: List[int] = None,
        # Texture feature parameters
        lbp_radius: int = 1,
        lbp_n_points: int = 8,
        lbp_method: str = 'uniform',
        lbp_n_bins: int = 10,
        # Normalization
        normalize_features: bool = True
    ):
        """
        Initialize the feature extractor.
        
        Args:
            chromatic_n_bins: Number of bins for color histograms
            chromatic_compute_stats: Whether to compute color statistics (mean, std)
            intensity_n_bins: Number of bins for intensity histogram
            intensity_percentiles: Percentile values to compute [5, 25, 50, 75, 95]
            lbp_radius: Radius of LBP circle
            lbp_n_points: Number of sampling points for LBP
            lbp_method: LBP method ('uniform', 'default', 'ror', 'nri_uniform')
            lbp_n_bins: Number of bins for LBP histogram
            normalize_features: Whether to L2-normalize feature vectors
        """
        # Chromatic parameters
        self.chromatic_n_bins = chromatic_n_bins
        self.chromatic_compute_stats = chromatic_compute_stats
        
        # Intensity parameters
        self.intensity_n_bins = intensity_n_bins
        self.intensity_percentiles = intensity_percentiles or [5, 25, 50, 75, 95]
        
        # Texture parameters
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.lbp_method = lbp_method
        self.lbp_n_bins = lbp_n_bins
        
        # Normalization
        self.normalize_features = normalize_features
        
        # Store computed features
        self._features = None
        self._feature_names = None
        self._feature_dim = None
        
        # Precompute feature dimensions
        self._compute_feature_dimensions()
    
    def _compute_feature_dimensions(self):
        """Precompute the dimension of each feature type."""
        dims = {}
        
        # Chromatic: histogram for L, a, b channels + optional statistics
        dims['chromatic_histogram'] = self.chromatic_n_bins * 3
        dims['chromatic_stats'] = 6 if self.chromatic_compute_stats else 0  # mean+std for L,a,b
        
        # Intensity: histogram + statistics + percentiles
        dims['intensity_histogram'] = self.intensity_n_bins
        dims['intensity_stats'] = 2  # mean, std
        dims['intensity_percentiles'] = len(self.intensity_percentiles)
        
        # Texture (LBP): histogram
        # For uniform LBP, number of bins is n_points + 2
        if self.lbp_method == 'uniform':
            dims['texture_lbp'] = self.lbp_n_points + 2
        else:
            dims['texture_lbp'] = self.lbp_n_bins
        
        self._feature_dims = dims
        self._feature_dim = sum(dims.values())
    
    def extract_features(
        self,
        image: np.ndarray,
        region_labels: np.ndarray
    ) -> np.ndarray:
        """
        Extract features for all regions in the image.
        
        Args:
            image: RGB image (H, W, 3)
            region_labels: Region label map (H, W)
            
        Returns:
            features: Feature matrix (n_regions, feature_dim)
        """
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        n_regions = region_labels.max() + 1
        
        # Precompute image representations
        lab_image = rgb2lab(image)
        gray_image = rgb2gray(image)
        lbp_image = self._compute_lbp(gray_image)
        
        # Initialize feature matrix
        features = np.zeros((n_regions, self._feature_dim))
        
        # Extract features for each region
        for region_id in range(n_regions):
            mask = region_labels == region_id
            
            if not mask.any():
                continue
            
            # Extract and concatenate all features
            feature_vector = self._extract_region_features(
                image, lab_image, gray_image, lbp_image, mask
            )
            features[region_id] = feature_vector
        
        # Normalize if requested
        if self.normalize_features:
            features = self._normalize(features)
        
        self._features = features
        return features
    
    def _extract_region_features(
        self,
        rgb_image: np.ndarray,
        lab_image: np.ndarray,
        gray_image: np.ndarray,
        lbp_image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract all features for a single region.
        
        Args:
            rgb_image: RGB image [0, 1]
            lab_image: LAB image
            gray_image: Grayscale image [0, 1]
            lbp_image: LBP response image
            mask: Boolean mask for the region
            
        Returns:
            Feature vector for the region
        """
        features = []
        
        # 1. Chromatic features (LAB color space)
        chromatic_feats = self._extract_chromatic_features(lab_image, mask)
        features.extend(chromatic_feats)
        
        # 2. Intensity features
        intensity_feats = self._extract_intensity_features(gray_image, mask)
        features.extend(intensity_feats)
        
        # 3. Texture features (LBP)
        texture_feats = self._extract_texture_features(lbp_image, mask)
        features.extend(texture_feats)
        
        return np.array(features)
    
    def _extract_chromatic_features(
        self,
        lab_image: np.ndarray,
        mask: np.ndarray
    ) -> List[float]:
        """
        Extract chromatic features from LAB color space.
        
        LAB is chosen for perceptual uniformity, making it well-suited
        for distinguishing shadow vs non-shadow regions.
        
        Args:
            lab_image: Image in LAB color space
            mask: Boolean mask for the region
            
        Returns:
            List of chromatic features
        """
        features = []
        
        # Extract LAB values for the region
        lab_values = lab_image[mask]  # Shape: (n_pixels, 3)
        
        # Normalize LAB channels to [0, 1] range for histogram
        # L: [0, 100], a: [-128, 127], b: [-128, 127]
        L = lab_values[:, 0] / 100.0
        a = (lab_values[:, 1] + 128) / 255.0
        b = (lab_values[:, 2] + 128) / 255.0
        
        # Compute histograms for each channel
        for channel, name in [(L, 'L'), (a, 'a'), (b, 'b')]:
            hist, _ = np.histogram(
                channel,
                bins=self.chromatic_n_bins,
                range=(0, 1),
                density=True
            )
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-10)
            features.extend(hist.tolist())
        
        # Compute statistics if requested
        if self.chromatic_compute_stats:
            for channel in [lab_values[:, 0], lab_values[:, 1], lab_values[:, 2]]:
                features.append(np.mean(channel))
                features.append(np.std(channel))
        
        return features
    
    def _extract_intensity_features(
        self,
        gray_image: np.ndarray,
        mask: np.ndarray
    ) -> List[float]:
        """
        Extract intensity features from grayscale image.
        
        Shadow regions typically have lower intensity values compared
        to their surroundings.
        
        Args:
            gray_image: Grayscale image [0, 1]
            mask: Boolean mask for the region
            
        Returns:
            List of intensity features
        """
        features = []
        
        # Extract intensity values
        intensity_values = gray_image[mask]
        
        # Histogram
        hist, _ = np.histogram(
            intensity_values,
            bins=self.intensity_n_bins,
            range=(0, 1),
            density=True
        )
        hist = hist / (hist.sum() + 1e-10)
        features.extend(hist.tolist())
        
        # Statistics
        features.append(np.mean(intensity_values))
        features.append(np.std(intensity_values))
        
        # Percentiles
        percentiles = np.percentile(intensity_values, self.intensity_percentiles)
        features.extend(percentiles.tolist())
        
        return features
    
    def _compute_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Compute Local Binary Pattern for the entire image.
        
        LBP is a powerful texture descriptor that captures local 
        texture patterns. It's useful for distinguishing shadow
        boundaries from other edges.
        
        Args:
            gray_image: Grayscale image [0, 1]
            
        Returns:
            LBP response image
        """
        # Convert to uint8 for LBP computation
        gray_uint8 = (gray_image * 255).astype(np.uint8)
        
        # Compute LBP
        lbp = local_binary_pattern(
            gray_uint8,
            P=self.lbp_n_points,
            R=self.lbp_radius,
            method=self.lbp_method
        )
        
        return lbp
    
    def _extract_texture_features(
        self,
        lbp_image: np.ndarray,
        mask: np.ndarray
    ) -> List[float]:
        """
        Extract texture features using Local Binary Pattern histogram.
        
        Args:
            lbp_image: LBP response image
            mask: Boolean mask for the region
            
        Returns:
            List of texture features (LBP histogram)
        """
        # Extract LBP values for the region
        lbp_values = lbp_image[mask]
        
        # Determine number of bins based on LBP method
        if self.lbp_method == 'uniform':
            n_bins = self.lbp_n_points + 2
            bin_range = (0, n_bins)
        else:
            n_bins = self.lbp_n_bins
            bin_range = (0, 2 ** self.lbp_n_points)
        
        # Compute histogram
        hist, _ = np.histogram(
            lbp_values,
            bins=n_bins,
            range=bin_range,
            density=True
        )
        
        # Normalize
        hist = hist / (hist.sum() + 1e-10)
        
        return hist.tolist()
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        L2 normalize feature vectors.
        
        Args:
            features: Feature matrix (n_regions, feature_dim)
            
        Returns:
            Normalized features
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        return features / norms
    
    def get_feature_info(self) -> Dict:
        """
        Get information about the extracted features.
        
        Returns:
            Dictionary with feature dimensions and names
        """
        return {
            'total_dim': self._feature_dim,
            'dimensions': self._feature_dims,
            'chromatic_bins': self.chromatic_n_bins,
            'intensity_bins': self.intensity_n_bins,
            'lbp_params': {
                'radius': self.lbp_radius,
                'n_points': self.lbp_n_points,
                'method': self.lbp_method
            }
        }
    
    @property
    def features(self) -> Optional[np.ndarray]:
        """Return the extracted features."""
        return self._features
    
    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        return self._feature_dim


class MultiKernelFeatureExtractor(FeatureExtractor):
    """
    Extended feature extractor that organizes features for multi-kernel learning.
    
    The paper uses multiple kernels, each operating on different feature types.
    This class separates features by type for the kernel optimization stage.
    """
    
    def extract_features_by_type(
        self,
        image: np.ndarray,
        region_labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract features organized by type for multi-kernel learning.
        
        Args:
            image: RGB image (H, W, 3)
            region_labels: Region label map (H, W)
            
        Returns:
            Dictionary mapping feature type to feature matrix:
                - 'chromatic': Chromatic features (n_regions, chromatic_dim)
                - 'intensity': Intensity features (n_regions, intensity_dim)
                - 'texture': Texture features (n_regions, texture_dim)
        """
        # Normalize image
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        n_regions = region_labels.max() + 1
        
        # Precompute image representations
        lab_image = rgb2lab(image)
        gray_image = rgb2gray(image)
        lbp_image = self._compute_lbp(gray_image)
        
        # Compute dimensions
        chromatic_dim = (self._feature_dims['chromatic_histogram'] + 
                        self._feature_dims['chromatic_stats'])
        intensity_dim = (self._feature_dims['intensity_histogram'] + 
                        self._feature_dims['intensity_stats'] + 
                        self._feature_dims['intensity_percentiles'])
        texture_dim = self._feature_dims['texture_lbp']
        
        # Initialize feature matrices
        chromatic_features = np.zeros((n_regions, chromatic_dim))
        intensity_features = np.zeros((n_regions, intensity_dim))
        texture_features = np.zeros((n_regions, texture_dim))
        
        # Extract features for each region
        for region_id in range(n_regions):
            mask = region_labels == region_id
            
            if not mask.any():
                continue
            
            # Chromatic
            chromatic = self._extract_chromatic_features(lab_image, mask)
            chromatic_features[region_id] = chromatic
            
            # Intensity
            intensity = self._extract_intensity_features(gray_image, mask)
            intensity_features[region_id] = intensity
            
            # Texture
            texture = self._extract_texture_features(lbp_image, mask)
            texture_features[region_id] = texture
        
        # Normalize each feature type separately
        if self.normalize_features:
            chromatic_features = self._normalize(chromatic_features)
            intensity_features = self._normalize(intensity_features)
            texture_features = self._normalize(texture_features)
        
        return {
            'chromatic': chromatic_features,
            'intensity': intensity_features,
            'texture': texture_features
        }


class PaperCompliantFeatureExtractor:
    """
    Feature extractor that follows the exact paper specification.
    
    This class extracts features exactly as described in Section 3.1:
    - 21-bin histogram for L* channel (EMD distance)
    - 21-bin histogram for a* channel (EMD distance)
    - 21-bin histogram for b* channel (EMD distance)
    - 128-bin texton histogram (χ² distance) using MR8 filter bank
    
    Reference:
        Paper Section 3.1:
        "For each region, we compute a 21-bin histogram for each of the 
        components (L*,a*,b*) of the perceptually uniform color space CIELAB."
        
        "To represent texture, we compute a 128-bin texton histogram. We run 
        the full MR8 filter set [33] in the whole dataset and cluster the 
        filter responses into 128 textons using k-means."
    """
    
    # Paper specifies 21 bins for color histograms
    N_COLOR_BINS = 21
    
    # Paper specifies 128 bins for texton histogram
    N_TEXTON_BINS = 128
    
    def __init__(
        self, 
        use_texton: bool = True,
        texton_extractor: Optional['TextonFeatureExtractor'] = None
    ):
        """
        Initialize paper-compliant feature extractor.
        
        Args:
            use_texton: If True, use MR8 texton features (paper method)
                       If False, use LBP as simplified texture (for testing)
            texton_extractor: Pre-built texton extractor (optional)
                             If None and use_texton=True, will create new one
                             (but dictionary needs to be built separately)
        """
        self.use_texton = use_texton
        
        if use_texton:
            if texton_extractor is not None:
                self.texton_extractor = texton_extractor
            else:
                # Import here to avoid circular dependency
                from .texton import TextonFeatureExtractor
                self.texton_extractor = TextonFeatureExtractor(n_textons=self.N_TEXTON_BINS)
        else:
            self.texton_extractor = None
            # LBP parameters for simplified texture
            self.lbp_radius = 2
            self.lbp_n_points = 16
    
    def build_texton_dictionary(
        self,
        images: List[np.ndarray],
        verbose: bool = True
    ) -> None:
        """
        Build texton dictionary from training images.
        
        This must be called before extracting features if use_texton=True.
        
        Reference:
            Paper: "We run the full MR8 filter set in the whole dataset and 
            cluster the filter responses into 128 textons using k-means."
        
        Args:
            images: List of training images
            verbose: Whether to print progress
        """
        if not self.use_texton:
            raise RuntimeError("Cannot build texton dictionary when use_texton=False")
        
        self.texton_extractor.build_dictionary(images, verbose=verbose)
    
    def load_texton_dictionary(self, filepath: str) -> None:
        """Load prebuilt texton dictionary."""
        if not self.use_texton:
            raise RuntimeError("Cannot load texton dictionary when use_texton=False")
        self.texton_extractor.load_dictionary(filepath)
    
    def save_texton_dictionary(self, filepath: str) -> None:
        """Save texton dictionary to file."""
        if not self.use_texton:
            raise RuntimeError("Cannot save texton dictionary when use_texton=False")
        self.texton_extractor.save_dictionary(filepath)
    
    def extract_features_by_channel(
        self,
        image: np.ndarray,
        region_labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract features organized by channel for paper's multi-kernel.
        
        This returns features in the exact format expected by the paper's
        kernel formulation (Equation 5):
        
        K(x,y) = Σ_{l∈{L,a,b,t}} w_l exp(-1/σ_l D_l(x,y))
        
        where:
        - D_L, D_a, D_b: EMD distance for color histograms
        - D_t: χ² distance for texton histogram
        
        Args:
            image: RGB image (H, W, 3)
            region_labels: Region label map (H, W)
            
        Returns:
            Dictionary with keys 'L', 'a', 'b', 't':
                - 'L': L* histogram (n_regions, 21)
                - 'a': a* histogram (n_regions, 21)
                - 'b': b* histogram (n_regions, 21)
                - 't': Texton histogram (n_regions, 128) or LBP histogram
        """
        # Normalize image
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        n_regions = int(region_labels.max()) + 1
        
        # Convert to LAB
        lab_image = rgb2lab(image)
        
        # Initialize color feature matrices
        hist_L = np.zeros((n_regions, self.N_COLOR_BINS))
        hist_a = np.zeros((n_regions, self.N_COLOR_BINS))
        hist_b = np.zeros((n_regions, self.N_COLOR_BINS))
        
        # Extract color features for each region
        for region_id in range(n_regions):
            mask = region_labels == region_id
            
            if not mask.any():
                continue
            
            # Extract LAB values
            lab_values = lab_image[mask]  # (n_pixels, 3)
            
            # L* channel: range [0, 100]
            L_values = lab_values[:, 0]
            hist, _ = np.histogram(L_values, bins=self.N_COLOR_BINS, range=(0, 100))
            hist_L[region_id] = hist / (hist.sum() + 1e-10)
            
            # a* channel: range approximately [-128, 127]
            a_values = lab_values[:, 1]
            hist, _ = np.histogram(a_values, bins=self.N_COLOR_BINS, range=(-128, 127))
            hist_a[region_id] = hist / (hist.sum() + 1e-10)
            
            # b* channel: range approximately [-128, 127]
            b_values = lab_values[:, 2]
            hist, _ = np.histogram(b_values, bins=self.N_COLOR_BINS, range=(-128, 127))
            hist_b[region_id] = hist / (hist.sum() + 1e-10)
        
        # Extract texture features
        if self.use_texton:
            # Use MR8 texton features (paper method)
            if not self.texton_extractor.texton_dict.is_built:
                raise RuntimeError(
                    "Texton dictionary not built. Call build_texton_dictionary() first, "
                    "or set use_texton=False to use LBP as a simplified alternative."
                )
            hist_t = self.texton_extractor.extract_features(image, region_labels)
        else:
            # Use LBP as simplified texture (for testing)
            gray_image = rgb2gray(image)
            n_texture_bins = self.lbp_n_points + 2
            lbp_image = local_binary_pattern(
                (gray_image * 255).astype(np.uint8),
                P=self.lbp_n_points,
                R=self.lbp_radius,
                method='uniform'
            )
            
            hist_t = np.zeros((n_regions, n_texture_bins))
            for region_id in range(n_regions):
                mask = region_labels == region_id
                if not mask.any():
                    continue
                lbp_values = lbp_image[mask]
                hist, _ = np.histogram(lbp_values, bins=n_texture_bins, range=(0, n_texture_bins))
                hist_t[region_id] = hist / (hist.sum() + 1e-10)
        
        return {
            'L': hist_L,
            'a': hist_a,
            'b': hist_b,
            't': hist_t
        }
    
    def get_feature_info(self) -> Dict:
        """Return feature dimensions."""
        if self.use_texton:
            n_texture_bins = self.N_TEXTON_BINS
        else:
            n_texture_bins = self.lbp_n_points + 2
        
        return {
            'L_dim': self.N_COLOR_BINS,
            'a_dim': self.N_COLOR_BINS,
            'b_dim': self.N_COLOR_BINS,
            't_dim': n_texture_bins,
            'total_dim': 3 * self.N_COLOR_BINS + n_texture_bins,
            'use_texton': self.use_texton
        }


def extract_all_features(
    image: np.ndarray,
    region_labels: np.ndarray,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convenience function to extract all features from an image.
    
    Args:
        image: RGB image (H, W, 3)
        region_labels: Region label map (H, W)
        config: Optional configuration dictionary
        
    Returns:
        combined_features: Combined feature matrix (n_regions, total_dim)
        features_by_type: Dictionary of features organized by type
    """
    config = config or {}
    
    extractor = MultiKernelFeatureExtractor(
        chromatic_n_bins=config.get('chromatic_n_bins', 16),
        intensity_n_bins=config.get('intensity_n_bins', 32),
        lbp_radius=config.get('lbp_radius', 1),
        lbp_n_points=config.get('lbp_n_points', 8),
        normalize_features=config.get('normalize', True)
    )
    
    # Extract combined features
    combined_features = extractor.extract_features(image, region_labels)
    
    # Extract features by type
    features_by_type = extractor.extract_features_by_type(image, region_labels)
    
    return combined_features, features_by_type

