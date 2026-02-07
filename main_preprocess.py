"""
Main preprocessing script for shadow detection.

This script implements Stage 1-2 of the shadow detection pipeline:
    Stage 1: Superpixel segmentation and region generation
    Stage 2: Feature extraction (chromatic, intensity, texture)

Usage:
    python main_preprocess.py --image <image_path> [--output_dir <output_dir>]

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015

Author: [Your Name]
Date: 2026
"""

import os
import sys
import argparse
import numpy as np
import time
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUPERPIXEL_CONFIG, 
    REGION_CONFIG, 
    CHROMATIC_CONFIG,
    INTENSITY_CONFIG,
    TEXTURE_CONFIG,
    OUTPUT_DIR
)
from preprocessing import SuperpixelSegmenter, RegionGenerator, FeatureExtractor
from preprocessing.features import MultiKernelFeatureExtractor
from utils.io_utils import (
    load_image, 
    save_image, 
    save_results, 
    create_output_dirs,
    save_label_map
)


class ShadowPreprocessor:
    """
    Complete preprocessing pipeline for shadow detection.
    
    This class orchestrates:
    1. Superpixel segmentation (SLIC)
    2. Region generation (hierarchical clustering)
    3. Feature extraction (chromatic, intensity, texture)
    """
    
    def __init__(
        self,
        superpixel_config: Dict = None,
        region_config: Dict = None,
        feature_config: Dict = None,
        verbose: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            superpixel_config: Configuration for superpixel segmentation
            region_config: Configuration for region generation
            feature_config: Configuration for feature extraction
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
        # Initialize components with configurations
        sp_config = superpixel_config or SUPERPIXEL_CONFIG
        rg_config = region_config or REGION_CONFIG
        
        self.superpixel_segmenter = SuperpixelSegmenter(
            n_segments=sp_config['n_segments'],
            compactness=sp_config['compactness'],
            sigma=sp_config['sigma'],
            convert2lab=sp_config['convert2lab'],
            enforce_connectivity=sp_config['enforce_connectivity'],
            min_size_factor=sp_config['min_size_factor'],
            max_size_factor=sp_config['max_size_factor']
        )
        
        # Region generator using Mean-shift clustering (paper-compliant)
        # Reference: Paper Section 3 - "apply Mean-shift clustering [4]"
        self.region_generator = RegionGenerator(
            bandwidth=rg_config.get('bandwidth', None),  # Auto-estimate if None
            quantile=rg_config.get('quantile', 0.3),     # Controls granularity
            use_spatial=rg_config.get('use_spatial', False)
        )
        
        self.feature_extractor = MultiKernelFeatureExtractor(
            chromatic_n_bins=CHROMATIC_CONFIG.get('n_bins_histogram', 16),
            intensity_n_bins=INTENSITY_CONFIG.get('n_bins_histogram', 32),
            intensity_percentiles=INTENSITY_CONFIG.get('percentile_values', [5, 25, 50, 75, 95]),
            lbp_radius=TEXTURE_CONFIG.get('lbp_radius', 1),
            lbp_n_points=TEXTURE_CONFIG.get('lbp_n_points', 8),
            lbp_method=TEXTURE_CONFIG.get('lbp_method', 'uniform'),
            normalize_features=True
        )
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Preprocessor] {message}")
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Dictionary containing all preprocessing outputs:
                - 'superpixel_labels': Superpixel label map (H, W)
                - 'region_labels': Region label map (H, W)
                - 'n_superpixels': Number of superpixels
                - 'n_regions': Number of regions
                - 'features': Combined feature matrix (n_regions, feature_dim)
                - 'features_by_type': Features organized by type
                - 'region_adjacency': Region adjacency matrix
                - 'boundary_features': Boundary features between regions
                - 'superpixel_adjacency': Superpixel adjacency matrix
        """
        start_time = time.time()
        results = {}
        
        # ===== Stage 1.1: Superpixel Segmentation =====
        self._log("Stage 1.1: Performing superpixel segmentation...")
        sp_start = time.time()
        
        superpixel_labels = self.superpixel_segmenter.segment(image)
        superpixel_adjacency = self.superpixel_segmenter.get_adjacency_matrix()
        superpixel_properties = self.superpixel_segmenter.get_superpixel_properties(image)
        
        results['superpixel_labels'] = superpixel_labels
        results['n_superpixels'] = self.superpixel_segmenter.n_superpixels
        results['superpixel_adjacency'] = superpixel_adjacency
        results['superpixel_properties'] = superpixel_properties
        
        self._log(f"  Generated {results['n_superpixels']} superpixels in {time.time() - sp_start:.2f}s")
        
        # ===== Stage 1.2: Region Generation =====
        self._log("Stage 1.2: Merging superpixels into regions...")
        rg_start = time.time()
        
        region_labels = self.region_generator.generate_regions(
            image, 
            superpixel_labels,
            superpixel_adjacency
        )
        region_adjacency = self.region_generator.get_region_adjacency()
        region_properties = self.region_generator.get_region_properties(image)
        boundary_features = self.region_generator.get_boundary_features(image)
        
        results['region_labels'] = region_labels
        results['n_regions'] = self.region_generator._n_actual_regions
        results['region_adjacency'] = region_adjacency
        results['region_properties'] = region_properties
        results['boundary_features'] = boundary_features
        results['superpixel_to_region'] = self.region_generator.superpixel_to_region
        
        self._log(f"  Generated {results['n_regions']} regions in {time.time() - rg_start:.2f}s")
        
        # ===== Stage 2: Feature Extraction =====
        self._log("Stage 2: Extracting features...")
        fe_start = time.time()
        
        # Extract combined features
        features = self.feature_extractor.extract_features(image, region_labels)
        
        # Extract features by type (for multi-kernel learning)
        features_by_type = self.feature_extractor.extract_features_by_type(image, region_labels)
        
        results['features'] = features
        results['features_by_type'] = features_by_type
        results['feature_info'] = self.feature_extractor.get_feature_info()
        
        self._log(f"  Extracted features with dimension {features.shape[1]} in {time.time() - fe_start:.2f}s")
        self._log(f"    - Chromatic: {features_by_type['chromatic'].shape[1]} dims")
        self._log(f"    - Intensity: {features_by_type['intensity'].shape[1]} dims")
        self._log(f"    - Texture: {features_by_type['texture'].shape[1]} dims")
        
        total_time = time.time() - start_time
        self._log(f"Preprocessing completed in {total_time:.2f}s")
        
        results['preprocessing_time'] = total_time
        
        return results
    
    def visualize(
        self,
        image: np.ndarray,
        results: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate visualization of preprocessing results.
        
        Args:
            image: Original RGB image
            results: Preprocessing results dictionary
            
        Returns:
            Tuple of (superpixel_vis, region_vis) visualization images
        """
        # Superpixel visualization
        superpixel_vis = self.superpixel_segmenter.visualize(
            image, 
            color=(255, 255, 0)  # Yellow
        )
        
        # Region visualization
        region_vis = self.region_generator.visualize(
            image,
            color=(0, 255, 0)  # Green
        )
        
        return superpixel_vis, region_vis


def main():
    """Main function for preprocessing."""
    parser = argparse.ArgumentParser(
        description='Shadow Detection Preprocessing Pipeline'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--n_superpixels', '-sp',
        type=int,
        default=SUPERPIXEL_CONFIG['n_segments'],
        help='Number of superpixels'
    )
    parser.add_argument(
        '--n_regions', '-r',
        type=int,
        default=REGION_CONFIG['n_regions'],
        help='Number of regions'
    )
    parser.add_argument(
        '--save_vis', '-v',
        action='store_true',
        help='Save visualization images'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output messages'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    output_dirs = create_output_dirs(args.output_dir)
    
    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    print(f"Image shape: {image.shape}")
    
    # Get image name for output files
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # Configure preprocessor
    sp_config = SUPERPIXEL_CONFIG.copy()
    sp_config['n_segments'] = args.n_superpixels
    
    rg_config = REGION_CONFIG.copy()
    rg_config['n_regions'] = args.n_regions
    
    # Initialize and run preprocessor
    preprocessor = ShadowPreprocessor(
        superpixel_config=sp_config,
        region_config=rg_config,
        verbose=not args.quiet
    )
    
    results = preprocessor.process(image)
    
    # Save results
    results_path = os.path.join(output_dirs['features'], f'{image_name}_features.pkl')
    save_results(results, results_path)
    print(f"Results saved to: {results_path}")
    
    # Save label maps
    sp_labels_path = os.path.join(output_dirs['segmentation'], f'{image_name}_superpixels.npy')
    save_label_map(results['superpixel_labels'], sp_labels_path)
    
    region_labels_path = os.path.join(output_dirs['segmentation'], f'{image_name}_regions.npy')
    save_label_map(results['region_labels'], region_labels_path)
    
    # Save visualizations if requested
    if args.save_vis:
        superpixel_vis, region_vis = preprocessor.visualize(image, results)
        
        sp_vis_path = os.path.join(output_dirs['visualizations'], f'{image_name}_superpixels.png')
        save_image(superpixel_vis, sp_vis_path)
        
        region_vis_path = os.path.join(output_dirs['visualizations'], f'{image_name}_regions.png')
        save_image(region_vis, region_vis_path)
        
        print(f"Visualizations saved to: {output_dirs['visualizations']}")
    
    # Print summary
    print("\n" + "="*50)
    print("Preprocessing Summary")
    print("="*50)
    print(f"Image: {image_name}")
    print(f"Superpixels: {results['n_superpixels']}")
    print(f"Regions: {results['n_regions']}")
    print(f"Feature dimension: {results['features'].shape[1]}")
    print(f"  - Chromatic: {results['features_by_type']['chromatic'].shape[1]}")
    print(f"  - Intensity: {results['features_by_type']['intensity'].shape[1]}")
    print(f"  - Texture: {results['features_by_type']['texture'].shape[1]}")
    print(f"Processing time: {results['preprocessing_time']:.2f}s")
    print("="*50)


if __name__ == '__main__':
    main()

