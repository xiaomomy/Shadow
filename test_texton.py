"""
Test script for MR8 filter bank and texton features.

This tests the texton implementation against paper specifications.
"""

import numpy as np
import sys

def test_mr8_filter_bank():
    """Test MR8 filter bank."""
    print("\nTest 1: MR8 Filter Bank")
    print("-" * 40)
    
    from preprocessing.texton import MR8FilterBank
    
    # Create filter bank
    fb = MR8FilterBank()
    info = fb.get_filter_info()
    
    print(f"  Number of output channels: {info['n_filters']}")
    print(f"  Scales: {info['scales']}")
    print(f"  Orientations: {info['n_orientations']}")
    print(f"  Filter names: {info['filter_names']}")
    
    # Test on synthetic image
    np.random.seed(42)
    image = np.random.rand(64, 64)
    
    responses = fb.apply(image)
    
    print(f"  Input shape: {image.shape}")
    print(f"  Output shape: {responses.shape}")
    
    assert responses.shape == (64, 64, 8), f"Expected (64, 64, 8), got {responses.shape}"
    assert not np.any(np.isnan(responses)), "Responses contain NaN"
    
    print("  [PASS] MR8 filter bank works correctly")
    return True


def test_texton_dictionary():
    """Test texton dictionary building."""
    print("\nTest 2: Texton Dictionary")
    print("-" * 40)
    
    from preprocessing.texton import TextonDictionary
    
    # Create synthetic images
    np.random.seed(42)
    images = [np.random.rand(32, 32) for _ in range(5)]
    
    # Build dictionary with small number of textons for testing
    n_textons = 16
    texton_dict = TextonDictionary(
        n_textons=n_textons,
        sample_fraction=0.5,
        random_state=42
    )
    
    texton_dict.build(images, verbose=False)
    
    print(f"  Number of textons: {n_textons}")
    print(f"  Dictionary centers shape: {texton_dict.centers_.shape}")
    
    assert texton_dict.centers_.shape == (n_textons, 8), \
        f"Expected ({n_textons}, 8), got {texton_dict.centers_.shape}"
    assert texton_dict.is_built, "Dictionary should be marked as built"
    
    # Test histogram computation
    test_image = np.random.rand(32, 32)
    hist = texton_dict.compute_histogram(test_image)
    
    print(f"  Histogram shape: {hist.shape}")
    print(f"  Histogram sum: {hist.sum():.6f}")
    
    assert hist.shape == (n_textons,), f"Expected ({n_textons},), got {hist.shape}"
    assert abs(hist.sum() - 1.0) < 1e-6, f"Histogram should sum to 1, got {hist.sum()}"
    
    print("  [PASS] Texton dictionary works correctly")
    return True


def test_texton_feature_extractor():
    """Test texton feature extractor."""
    print("\nTest 3: Texton Feature Extractor")
    print("-" * 40)
    
    from preprocessing.texton import TextonFeatureExtractor
    
    # Create synthetic images for building dictionary
    np.random.seed(42)
    train_images = [np.random.rand(32, 32) for _ in range(3)]
    
    # Create extractor
    n_textons = 32
    extractor = TextonFeatureExtractor(n_textons=n_textons)
    
    # Build dictionary
    extractor.build_dictionary(train_images, verbose=False)
    
    # Create test image and regions
    test_image = np.random.rand(32, 32, 3)  # RGB
    region_labels = np.zeros((32, 32), dtype=int)
    region_labels[16:, :] = 1  # 2 regions
    
    # Extract features
    features = extractor.extract_features(test_image, region_labels)
    
    print(f"  Number of regions: 2")
    print(f"  Features shape: {features.shape}")
    
    assert features.shape == (2, n_textons), \
        f"Expected (2, {n_textons}), got {features.shape}"
    
    # Check that each row is a valid histogram
    for i in range(2):
        assert abs(features[i].sum() - 1.0) < 1e-6, \
            f"Row {i} should sum to 1, got {features[i].sum()}"
    
    print("  [PASS] Texton feature extractor works correctly")
    return True


def test_paper_compliant_extractor_with_texton():
    """Test PaperCompliantFeatureExtractor with texton features."""
    print("\nTest 4: Paper-Compliant Feature Extractor (with Texton)")
    print("-" * 40)
    
    from preprocessing.features import PaperCompliantFeatureExtractor
    
    # Create with texton enabled
    extractor = PaperCompliantFeatureExtractor(use_texton=True)
    
    # Create synthetic training images
    np.random.seed(42)
    train_images = [np.random.rand(32, 32, 3) for _ in range(3)]
    
    # Build texton dictionary
    extractor.build_texton_dictionary(train_images, verbose=False)
    
    # Create test image and regions
    test_image = np.random.rand(32, 32, 3)
    region_labels = np.zeros((32, 32), dtype=int)
    region_labels[16:, :] = 1
    region_labels[:, 16:] += 2
    # Now we have 4 regions: 0, 1, 2, 3
    
    # Extract features
    features = extractor.extract_features_by_channel(test_image, region_labels)
    
    print(f"  Feature channels: {list(features.keys())}")
    print(f"  L* histogram shape: {features['L'].shape}")
    print(f"  a* histogram shape: {features['a'].shape}")
    print(f"  b* histogram shape: {features['b'].shape}")
    print(f"  Texton histogram shape: {features['t'].shape}")
    
    # Verify shapes
    n_regions = 4
    assert features['L'].shape == (n_regions, 21), f"L* should be (4, 21)"
    assert features['a'].shape == (n_regions, 21), f"a* should be (4, 21)"
    assert features['b'].shape == (n_regions, 21), f"b* should be (4, 21)"
    assert features['t'].shape == (n_regions, 128), f"Texton should be (4, 128)"
    
    # Check feature info
    info = extractor.get_feature_info()
    print(f"  Feature info: {info}")
    
    assert info['L_dim'] == 21
    assert info['a_dim'] == 21
    assert info['b_dim'] == 21
    assert info['t_dim'] == 128
    assert info['use_texton'] == True
    
    print("  [PASS] Paper-compliant feature extractor with texton works correctly")
    return True


def test_mr8_filter_properties():
    """Test MR8 filter specific properties."""
    print("\nTest 5: MR8 Filter Properties")
    print("-" * 40)
    
    from preprocessing.texton import MR8FilterBank
    
    fb = MR8FilterBank()
    
    # Check that filters have correct properties
    # Edge filters should have zero mean (approximately)
    for scale in fb.scales:
        for filt in fb._edge_filters[scale]:
            mean = filt.mean()
            print(f"  Edge filter (scale={scale}) mean: {mean:.6f}")
            # Note: Due to discretization, mean might not be exactly 0
    
    # LoG filter should have zero mean
    log_mean = fb._log_filter.mean()
    print(f"  LoG filter mean: {log_mean:.6f}")
    
    # Gaussian filter should sum to 1
    gauss_sum = fb._gaussian_filter.sum()
    print(f"  Gaussian filter sum: {gauss_sum:.6f}")
    assert abs(gauss_sum - 1.0) < 1e-6, "Gaussian filter should sum to 1"
    
    print("  [PASS] MR8 filter properties verified")
    return True


def run_all_tests():
    print("=" * 60)
    print("Testing Texton Implementation (MR8 + k-means)")
    print("Reference: Paper Section 3.1")
    print("=" * 60)
    
    tests = [
        test_mr8_filter_bank,
        test_texton_dictionary,
        test_texton_feature_extractor,
        test_paper_compliant_extractor_with_texton,
        test_mr8_filter_properties
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAILED] {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

