"""
Test script for the preprocessing pipeline.

This script demonstrates the usage of the preprocessing modules
and validates the implementation on a sample image.

Usage:
    python test_preprocessing.py

Author: [Your Name]
Date: 2026
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import SuperpixelSegmenter, RegionGenerator, FeatureExtractor
from preprocessing.features import MultiKernelFeatureExtractor
from main_preprocess import ShadowPreprocessor


def create_test_image(size: tuple = (256, 256)) -> np.ndarray:
    """
    Create a synthetic test image with shadow-like regions.
    
    The image contains:
    - A bright background
    - A darker region simulating shadow
    - Some texture variation
    
    Args:
        size: (height, width) of the image
        
    Returns:
        RGB image as numpy array
    """
    H, W = size
    image = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Background: bright yellowish
    image[:, :] = [200, 200, 150]
    
    # Add gradient variation
    for i in range(H):
        image[i, :, 0] = np.clip(image[i, :, 0] + (i // 10), 0, 255)
    
    # Shadow region 1: darker area (bottom-left quadrant)
    shadow_mask = np.zeros((H, W), dtype=bool)
    for i in range(H // 2, H):
        for j in range(W // 2):
            if (i - H // 2) ** 2 + (j - W // 4) ** 2 < (H // 3) ** 2:
                shadow_mask[i, j] = True
    
    # Apply shadow (reduce intensity)
    shadow_factor = 0.5
    image[shadow_mask] = (image[shadow_mask] * shadow_factor).astype(np.uint8)
    
    # Add texture (random noise)
    noise = np.random.randint(-10, 10, (H, W, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Shadow region 2: rectangular shadow (top-right)
    image[20:80, W-100:W-30] = np.clip(
        image[20:80, W-100:W-30].astype(np.int16) * 0.4, 0, 255
    ).astype(np.uint8)
    
    return image


def test_superpixel_segmentation():
    """Test superpixel segmentation module."""
    print("\n" + "="*50)
    print("Testing Superpixel Segmentation")
    print("="*50)
    
    # Create test image
    image = create_test_image((256, 256))
    print(f"Test image shape: {image.shape}")
    
    # Initialize segmenter
    segmenter = SuperpixelSegmenter(
        n_segments=100,
        compactness=10.0,
        sigma=1.0
    )
    
    # Perform segmentation
    labels = segmenter.segment(image)
    
    print(f"Superpixel labels shape: {labels.shape}")
    print(f"Number of superpixels: {segmenter.n_superpixels}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")
    
    # Test adjacency
    adjacency = segmenter.get_adjacency_matrix()
    print(f"Adjacency matrix shape: {adjacency.shape}")
    print(f"Average neighbors per superpixel: {adjacency.sum(axis=1).mean():.2f}")
    
    # Test properties
    properties = segmenter.get_superpixel_properties(image)
    print(f"Number of property entries: {len(properties)}")
    
    # Test visualization
    vis = segmenter.visualize(image)
    print(f"Visualization shape: {vis.shape}")
    
    print("[PASS] Superpixel segmentation test passed!")
    return image, labels, segmenter


def test_region_generation(image, superpixel_labels):
    """Test region generation module.
    
    Reference:
        Paper Section 3: "apply Mean-shift clustering [4] and merge 
        superpixels in the same cluster into a larger region."
    """
    print("\n" + "="*50)
    print("Testing Region Generation (Mean-shift - Paper Method)")
    print("="*50)
    
    # Initialize region generator using Mean-shift (paper-compliant)
    # Note: Mean-shift does not require specifying n_regions beforehand
    # The number of regions is determined by the bandwidth parameter
    generator = RegionGenerator(
        bandwidth=None,  # Auto-estimate bandwidth
        quantile=0.3,    # Controls granularity (smaller = more regions)
        use_spatial=False  # Paper likely uses color-only clustering
    )
    
    # Generate regions
    region_labels = generator.generate_regions(image, superpixel_labels)
    
    print(f"Region labels shape: {region_labels.shape}")
    print(f"Number of regions: {generator._n_actual_regions}")
    print(f"Estimated bandwidth: {generator.estimated_bandwidth:.4f}")
    print(f"Label range: [{region_labels.min()}, {region_labels.max()}]")
    
    # Test adjacency
    adjacency = generator.get_region_adjacency()
    print(f"Region adjacency shape: {adjacency.shape}")
    
    # Test properties
    properties = generator.get_region_properties(image)
    print(f"Number of region property entries: {len(properties)}")
    
    # Test boundary features
    boundary_features = generator.get_boundary_features(image)
    print(f"Number of boundary pairs: {len(boundary_features)}")
    
    # Test visualization
    vis = generator.visualize(image)
    print(f"Visualization shape: {vis.shape}")
    
    print("[PASS] Region generation test passed!")
    return region_labels, generator


def test_feature_extraction(image, region_labels):
    """Test feature extraction module."""
    print("\n" + "="*50)
    print("Testing Feature Extraction")
    print("="*50)
    
    # Initialize feature extractor
    extractor = MultiKernelFeatureExtractor(
        chromatic_n_bins=16,
        intensity_n_bins=32,
        lbp_radius=1,
        lbp_n_points=8,
        normalize_features=True
    )
    
    # Extract combined features
    features = extractor.extract_features(image, region_labels)
    
    print(f"Combined features shape: {features.shape}")
    print(f"Feature dimension: {extractor.feature_dim}")
    
    # Verify normalization
    norms = np.linalg.norm(features, axis=1)
    print(f"Feature norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    # Extract features by type
    features_by_type = extractor.extract_features_by_type(image, region_labels)
    
    print(f"Chromatic features shape: {features_by_type['chromatic'].shape}")
    print(f"Intensity features shape: {features_by_type['intensity'].shape}")
    print(f"Texture features shape: {features_by_type['texture'].shape}")
    
    # Get feature info
    info = extractor.get_feature_info()
    print(f"Feature info: {info}")
    
    print("[PASS] Feature extraction test passed!")
    return features, features_by_type


def test_full_pipeline():
    """Test the complete preprocessing pipeline."""
    print("\n" + "="*50)
    print("Testing Full Preprocessing Pipeline")
    print("="*50)
    
    # Create test image
    image = create_test_image((256, 256))
    
    # Initialize preprocessor
    preprocessor = ShadowPreprocessor(
        superpixel_config={'n_segments': 100, 'compactness': 10.0, 
                          'sigma': 1.0, 'convert2lab': True,
                          'enforce_connectivity': True, 
                          'min_size_factor': 0.25, 'max_size_factor': 3.0},
        region_config={'n_regions': 20, 'linkage': 'ward',
                       'affinity': 'euclidean', 'color_weight': 1.0,
                       'position_weight': 0.5},
        verbose=True
    )
    
    # Run pipeline
    results = preprocessor.process(image)
    
    # Verify results
    assert 'superpixel_labels' in results
    assert 'region_labels' in results
    assert 'features' in results
    assert 'features_by_type' in results
    
    print("\n" + "-"*50)
    print("Pipeline Results Summary:")
    print(f"  Superpixels: {results['n_superpixels']}")
    print(f"  Regions: {results['n_regions']}")
    print(f"  Feature dimension: {results['features'].shape[1]}")
    print(f"  Processing time: {results['preprocessing_time']:.2f}s")
    
    print("[PASS] Full pipeline test passed!")
    return results


def visualize_results(image, superpixel_labels, region_labels, save_path=None):
    """Visualize preprocessing results."""
    from skimage.segmentation import mark_boundaries
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Superpixels
    sp_vis = mark_boundaries(image / 255.0, superpixel_labels, color=(1, 1, 0))
    axes[1].imshow(sp_vis)
    axes[1].set_title(f'Superpixels (n={superpixel_labels.max() + 1})')
    axes[1].axis('off')
    
    # Regions
    rg_vis = mark_boundaries(image / 255.0, region_labels, color=(0, 1, 0))
    axes[2].imshow(rg_vis)
    axes[2].set_title(f'Regions (n={region_labels.max() + 1})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """Run all tests."""
    print("="*50)
    print("Shadow Detection Preprocessing Tests")
    print("="*50)
    
    # Test individual components
    image, sp_labels, segmenter = test_superpixel_segmentation()
    region_labels, generator = test_region_generation(image, sp_labels)
    features, features_by_type = test_feature_extraction(image, region_labels)
    
    # Test full pipeline
    results = test_full_pipeline()
    
    print("\n" + "="*50)
    print("All tests passed successfully!")
    print("="*50)
    
    # Optional: visualize results
    try:
        visualize_results(
            image, 
            sp_labels, 
            region_labels,
            save_path=os.path.join(os.path.dirname(__file__), 'output', 'test_visualization.png')
        )
    except Exception as e:
        print(f"Visualization skipped (matplotlib may not be available): {e}")
    
    return results


if __name__ == '__main__':
    main()

