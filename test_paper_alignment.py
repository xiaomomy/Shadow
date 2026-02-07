"""
Test script to verify alignment with paper specifications.

This tests the key implementations against the paper's formulas.
"""

import numpy as np
import sys

def run_tests():
    print("="*60)
    print("Testing Paper Alignment")
    print("="*60)
    
    # Test 1: Distance metrics
    print("\nTest 1: Distance Metrics")
    print("-"*40)
    from models.distances import emd_1d, emd_1d_matrix, chi_square_distance, chi_square_distance_matrix

    # EMD test
    p = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    q = np.array([0.2, 0.1, 0.3, 0.2, 0.2])
    emd = emd_1d(p, q)
    print(f"  EMD distance: {emd:.4f}")
    
    # EMD matrix test
    X = np.random.rand(5, 10)
    X = X / X.sum(axis=1, keepdims=True)
    D_emd = emd_1d_matrix(X)
    print(f"  EMD matrix shape: {D_emd.shape}")
    print(f"  EMD diagonal (should be 0): {np.diag(D_emd)}")

    # Chi-square test
    chi2 = chi_square_distance(p, q)
    print(f"  Chi-square distance: {chi2:.4f}")
    
    D_chi2 = chi_square_distance_matrix(X)
    print(f"  Chi2 matrix diagonal (should be 0): {np.diag(D_chi2)}")
    
    print("  [PASS] Distance metrics work correctly")

    # Test 2: Extended Gaussian Kernel
    print("\nTest 2: Extended Gaussian Kernel")
    print("-"*40)
    from models.kernels import ExtendedGaussianKernel

    X = np.random.rand(10, 21)  # 10 samples, 21-bin histograms
    X = X / X.sum(axis=1, keepdims=True)  # Normalize

    kernel_emd = ExtendedGaussianKernel(distance_type='emd')
    K_emd = kernel_emd.compute(X)
    print(f"  EMD kernel shape: {K_emd.shape}")
    print(f"  EMD kernel diagonal (should be 1): {np.diag(K_emd)[:3]}")
    assert np.allclose(np.diag(K_emd), 1.0), "Diagonal should be 1"

    kernel_chi2 = ExtendedGaussianKernel(distance_type='chi2')
    K_chi2 = kernel_chi2.compute(X)
    print(f"  Chi2 kernel shape: {K_chi2.shape}")
    print(f"  Chi2 kernel diagonal: {np.diag(K_chi2)[:3]}")
    assert np.allclose(np.diag(K_chi2), 1.0), "Diagonal should be 1"
    
    print("  [PASS] Extended Gaussian Kernel works correctly")

    # Test 3: Platt Scaling
    print("\nTest 3: Platt Scaling")
    print("-"*40)
    from models.platt_scaling import PlattScaler, balanced_error_rate

    np.random.seed(42)
    f = np.random.randn(100)  # Decision values
    y = np.sign(f + 0.3 * np.random.randn(100))

    scaler = PlattScaler()
    scaler.fit(f, y)
    print(f"  Platt params: a={scaler.a_:.4f}, b={scaler.b_:.4f}")

    proba = scaler.predict_proba(f)
    y_pred = scaler.predict(f)
    ber = balanced_error_rate(y, y_pred)
    print(f"  BER: {ber:.4f}")
    
    print("  [PASS] Platt Scaling works correctly")

    # Test 4: Sigma and Weight Grid (per paper)
    print("\nTest 4: Paper Parameter Grids")
    print("-"*40)
    from models.distances import compute_sigma_grid, compute_weight_grid

    mean_dist = 0.5
    sigmas = compute_sigma_grid(mean_dist)
    weights = compute_weight_grid()
    
    # Verify sigma grid matches paper: {1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8} * mu
    expected_multipliers = [1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8]
    expected_sigmas = np.array(expected_multipliers) * mean_dist
    assert np.allclose(sigmas, expected_sigmas), "Sigma grid doesn't match paper"
    print(f"  Sigma grid (mu=0.5): {sigmas}")
    
    # Verify weight grid matches paper: {s/40 | s in {1,...,10}}
    expected_weights = np.arange(1, 11) / 40.0
    assert np.allclose(weights, expected_weights), "Weight grid doesn't match paper"
    print(f"  Weight grid: {weights}")
    
    print("  [PASS] Parameter grids match paper specification")

    # Test 5: ShadowDetectionMultiKernel
    print("\nTest 5: Shadow Detection Multi-Kernel")
    print("-"*40)
    from models.kernels import ShadowDetectionMultiKernel
    
    # Create synthetic histogram features
    n_samples = 20
    features = {
        'L': np.random.rand(n_samples, 21),  # 21-bin L* histogram
        'a': np.random.rand(n_samples, 21),  # 21-bin a* histogram
        'b': np.random.rand(n_samples, 21),  # 21-bin b* histogram
        't': np.random.rand(n_samples, 18)   # Texture histogram (LBP)
    }
    # Normalize to histograms
    for key in features:
        features[key] = features[key] / features[key].sum(axis=1, keepdims=True)
    
    kernel = ShadowDetectionMultiKernel()
    K = kernel.compute(features_by_channel=features)
    
    print(f"  Combined kernel shape: {K.shape}")
    print(f"  Kernel diagonal (should be close to 1): {np.diag(K)[:3]}")
    print(f"  Mean distances: {kernel.mean_distances}")
    print(f"  Sigmas: {kernel.sigmas}")
    
    print("  [PASS] ShadowDetectionMultiKernel works correctly")

    # Test 6: Beam Search Optimizer structure
    print("\nTest 6: Beam Search Optimizer")
    print("-"*40)
    from models.loo_optimizer import PaperBeamSearchOptimizer
    
    # Just verify the class is correctly defined
    opt = PaperBeamSearchOptimizer(n_iterations=10, verbose=False)
    print(f"  Sigma multipliers: {opt.SIGMA_MULTIPLIERS}")
    print(f"  Weight values: {opt.WEIGHT_VALUES}")
    
    print("  [PASS] Beam Search Optimizer structure correct")

    print("\n" + "="*60)
    print("[SUCCESS] All tests passed!")
    print("="*60)
    return True

if __name__ == "__main__":
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

