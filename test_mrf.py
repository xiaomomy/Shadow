"""
Test script for MRF module.

Tests the implementation of Section 4 of the paper:
- Unary potentials (Section 4.1)
- Affinity pairwise potentials (Section 4.2)
- Disparity pairwise potentials (Section 4.3)
- QPBO/ICM optimization

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
"""

import numpy as np
import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mrf import (
    MRFShadowDetector,
    DisparityClassifier,
    compute_region_areas,
    compute_region_adjacency,
    compute_region_mean_rgb
)


class TestRegionUtilities(unittest.TestCase):
    """Test utility functions for region computation."""
    
    def test_compute_region_areas(self):
        """Test area computation for regions."""
        # Create a simple region map
        region_labels = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 2, 3],
            [2, 2, 3, 3]
        ])
        
        areas = compute_region_areas(region_labels)
        
        # Region 0: 4 pixels
        # Region 1: 4 pixels
        # Region 2: 6 pixels
        # Region 3: 3 pixels - wait let me recalculate
        # Region 3: (2,3), (3,2), (3,3) = 3 pixels
        
        self.assertEqual(len(areas), 4)
        self.assertEqual(areas[0], 4)
        self.assertEqual(areas[1], 4)
        self.assertEqual(areas[2], 5)  # Actually: (2,0), (2,1), (2,2), (3,0), (3,1) = 5 pixels
        self.assertEqual(areas[3], 3)  # (2,3), (3,2), (3,3) = 3 pixels
        
    def test_compute_region_adjacency(self):
        """Test adjacency matrix computation."""
        region_labels = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 2, 3],
            [2, 2, 3, 3]
        ])
        
        adjacency = compute_region_adjacency(region_labels)
        
        # Check adjacencies
        # Region 0 adjacent to: 1 (horizontal at (0,2)), 2 (vertical at (2,0))
        # Region 1 adjacent to: 0, 2, 3
        # Region 2 adjacent to: 0, 1, 3
        # Region 3 adjacent to: 1, 2
        
        self.assertEqual(adjacency.shape, (4, 4))
        self.assertEqual(adjacency[0, 1], 1)
        self.assertEqual(adjacency[1, 0], 1)
        self.assertEqual(adjacency[0, 2], 1)
        self.assertEqual(adjacency[1, 3], 1)
        self.assertEqual(adjacency[2, 3], 1)
        # Non-adjacent
        self.assertEqual(adjacency[0, 3], 0)
        
    def test_compute_region_mean_rgb(self):
        """Test mean RGB computation."""
        # Create simple image
        image = np.zeros((2, 2, 3))
        image[0, 0] = [1.0, 0.0, 0.0]  # Red
        image[0, 1] = [1.0, 0.0, 0.0]  # Red
        image[1, 0] = [0.0, 1.0, 0.0]  # Green
        image[1, 1] = [0.0, 0.0, 1.0]  # Blue
        
        region_labels = np.array([
            [0, 0],
            [1, 2]
        ])
        
        mean_rgb = compute_region_mean_rgb(image, region_labels)
        
        self.assertEqual(mean_rgb.shape, (3, 3))
        np.testing.assert_array_almost_equal(mean_rgb[0], [1.0, 0.0, 0.0])  # Region 0: all red
        np.testing.assert_array_almost_equal(mean_rgb[1], [0.0, 1.0, 0.0])  # Region 1: green
        np.testing.assert_array_almost_equal(mean_rgb[2], [0.0, 0.0, 1.0])  # Region 2: blue


class TestUnaryPotentials(unittest.TestCase):
    """
    Test unary potentials (Section 4.1).
    
    Reference:
        Paper: "φ(x_i) = -ω_i · P(x_i|R_i)"
    """
    
    def setUp(self):
        self.mrf = MRFShadowDetector()
        
    def test_unary_potential_computation(self):
        """Test that unary potentials are computed correctly."""
        # 3 regions with different probabilities and areas
        region_probs = np.array([0.9, 0.1, 0.5])  # P(shadow|R_i)
        region_areas = np.array([100, 200, 150])  # pixels
        
        self.mrf.set_unary_data(region_probs, region_areas)
        unary_shadow, unary_nonshadow = self.mrf.compute_unary_potentials()
        
        # φ(x_i=shadow) = -ω_i · P(shadow|R_i)
        # φ(x_i=nonshadow) = -ω_i · P(nonshadow|R_i)
        expected_shadow = -region_areas * region_probs
        expected_nonshadow = -region_areas * (1 - region_probs)
        
        np.testing.assert_array_almost_equal(unary_shadow, expected_shadow)
        np.testing.assert_array_almost_equal(unary_nonshadow, expected_nonshadow)
        
    def test_unary_potential_encourages_agreement(self):
        """
        Test that unary potential encourages correct labeling.
        
        Reference:
            Paper: "The unary potential encourages agreement between the label 
            of a region and the prediction based on the appearance of the region."
        """
        # High shadow probability should lead to lower energy for shadow label
        region_probs = np.array([0.9])  # Very likely shadow
        region_areas = np.array([100])
        
        self.mrf.set_unary_data(region_probs, region_areas)
        unary_shadow, unary_nonshadow = self.mrf.compute_unary_potentials()
        
        # Lower energy = better, so shadow should have lower energy
        self.assertLess(unary_shadow[0], unary_nonshadow[0])


class TestAffinityPotentials(unittest.TestCase):
    """
    Test affinity pairwise potentials (Section 4.2).
    
    Reference:
        Paper: "ψ_a(x_i, x_j) = { ω_ij · K(R_i, R_j)  if x_i ≠ x_j and K(R_i, R_j) > 0.5
                               { 0                    otherwise"
    """
    
    def setUp(self):
        self.mrf = MRFShadowDetector(affinity_threshold=0.5)
        
    def test_affinity_potential_computation(self):
        """Test affinity potential calculation."""
        # 3 regions
        region_probs = np.array([0.9, 0.1, 0.5])
        region_areas = np.array([100, 200, 150])
        
        # Adjacency: 0-1 neighbors, 1-2 neighbors
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        # Kernel similarities
        kernel_sim = np.array([
            [1.0, 0.8, 0.3],  # K(0,1)=0.8 > 0.5, K(0,2)=0.3 < 0.5
            [0.8, 1.0, 0.4],  # K(1,2)=0.4 < 0.5
            [0.3, 0.4, 1.0]
        ])
        
        self.mrf.set_unary_data(region_probs, region_areas)
        self.mrf.set_adjacency(adjacency)
        self.mrf.set_affinity_data(kernel_sim)
        
        affinity = self.mrf.compute_affinity_potentials()
        
        # Only (0,1) should have non-zero affinity (K > 0.5 and neighbors)
        omega_01 = np.sqrt(100 * 200)  # geometric mean
        expected_01 = omega_01 * 0.8
        
        self.assertAlmostEqual(affinity[0, 1], expected_01)
        self.assertAlmostEqual(affinity[1, 0], expected_01)
        
        # (1,2) are neighbors but K < 0.5
        self.assertEqual(affinity[1, 2], 0)
        
        # (0,2) not neighbors
        self.assertEqual(affinity[0, 2], 0)
        
    def test_affinity_threshold_effect(self):
        """
        Test that K(R_i, R_j) > 0.5 threshold is applied.
        
        Reference:
            Paper: "For R_i, R_j where K(R_i, R_j) > 0.5..."
        """
        region_probs = np.array([0.5, 0.5])
        region_areas = np.array([100, 100])
        adjacency = np.array([[0, 1], [1, 0]])
        
        # Test with K = 0.51 (just above threshold)
        kernel_sim = np.array([[1.0, 0.51], [0.51, 1.0]])
        
        self.mrf.set_unary_data(region_probs, region_areas)
        self.mrf.set_adjacency(adjacency)
        self.mrf.set_affinity_data(kernel_sim)
        
        affinity = self.mrf.compute_affinity_potentials()
        self.assertGreater(affinity[0, 1], 0)
        
        # Test with K = 0.49 (below threshold)
        kernel_sim = np.array([[1.0, 0.49], [0.49, 1.0]])
        self.mrf.set_affinity_data(kernel_sim)
        
        affinity = self.mrf.compute_affinity_potentials()
        self.assertEqual(affinity[0, 1], 0)


class TestDisparityPotentials(unittest.TestCase):
    """
    Test disparity pairwise potentials (Section 4.3).
    
    Reference:
        Paper: "ψ_d(x_i, x_j) = { 0                          if x_i ≠ x_j
                               { ω_ij · P^d(1|R_i, R_j)     otherwise"
    """
    
    def setUp(self):
        self.mrf = MRFShadowDetector(use_disparity=True)
        
    def test_disparity_potential_computation(self):
        """Test disparity potential calculation."""
        region_probs = np.array([0.9, 0.1, 0.5])
        region_areas = np.array([100, 200, 150])
        
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        # Disparity classifier predictions
        disparity_probs = np.array([
            [0.0, 0.7, 0.0],  # P^d(1|0,1) = 0.7
            [0.7, 0.0, 0.3],  # P^d(1|1,2) = 0.3
            [0.0, 0.3, 0.0]
        ])
        
        self.mrf.set_unary_data(region_probs, region_areas)
        self.mrf.set_adjacency(adjacency)
        self.mrf.set_disparity_data(disparity_probs)
        
        disparity = self.mrf.compute_disparity_potentials()
        
        # Check (0,1) disparity
        omega_01 = np.sqrt(100 * 200)
        expected_01 = omega_01 * 0.7
        self.assertAlmostEqual(disparity[0, 1], expected_01)
        
        # Check (1,2) disparity
        omega_12 = np.sqrt(200 * 150)
        expected_12 = omega_12 * 0.3
        self.assertAlmostEqual(disparity[1, 2], expected_12)


class TestMRFOptimization(unittest.TestCase):
    """Test MRF optimization methods."""
    
    def test_icm_simple_case(self):
        """Test ICM optimization on a simple case."""
        mrf = MRFShadowDetector(use_disparity=False)
        
        # Two regions, one clearly shadow, one clearly non-shadow
        region_probs = np.array([0.95, 0.05])
        region_areas = np.array([100, 100])
        adjacency = np.array([[0, 1], [1, 0]])
        kernel_sim = np.array([[1.0, 0.2], [0.2, 1.0]])  # Not similar
        
        mrf.set_unary_data(region_probs, region_areas)
        mrf.set_adjacency(adjacency)
        mrf.set_affinity_data(kernel_sim)
        
        labels = mrf.optimize(method='icm')
        
        # Region 0 should be shadow (+1), Region 1 should be non-shadow (-1)
        self.assertEqual(labels[0], 1)
        self.assertEqual(labels[1], -1)
        
    def test_icm_affinity_effect(self):
        """
        Test that affinity potential encourages same labels for similar regions.
        
        Reference:
            Paper: "The affinity potentials encourage similar adjacent regions 
            to have the same label"
        """
        mrf = MRFShadowDetector(affinity_threshold=0.5, use_disparity=False)
        
        # Two regions with borderline predictions but very similar
        region_probs = np.array([0.55, 0.45])  # Close to 0.5
        region_areas = np.array([100, 100])
        adjacency = np.array([[0, 1], [1, 0]])
        
        # Very high kernel similarity - should pull labels together
        kernel_sim = np.array([[1.0, 0.95], [0.95, 1.0]])
        
        mrf.set_unary_data(region_probs, region_areas)
        mrf.set_adjacency(adjacency)
        mrf.set_affinity_data(kernel_sim)
        
        labels = mrf.optimize(method='icm')
        
        # Due to high similarity, both should get same label
        self.assertEqual(labels[0], labels[1])
        
    def test_energy_computation(self):
        """Test energy computation is correct."""
        mrf = MRFShadowDetector(use_disparity=True)
        
        region_probs = np.array([0.8, 0.2])
        region_areas = np.array([100, 100])
        adjacency = np.array([[0, 1], [1, 0]])
        kernel_sim = np.array([[1.0, 0.6], [0.6, 1.0]])
        disparity_probs = np.array([[0.0, 0.5], [0.5, 0.0]])
        
        mrf.set_unary_data(region_probs, region_areas)
        mrf.set_adjacency(adjacency)
        mrf.set_affinity_data(kernel_sim)
        mrf.set_disparity_data(disparity_probs)
        
        unary_shadow, unary_nonshadow = mrf.compute_unary_potentials()
        affinity = mrf.compute_affinity_potentials()
        disparity = mrf.compute_disparity_potentials()
        
        # Test labels: [+1, -1] (different labels)
        labels_diff = np.array([1, -1])
        energy_diff = mrf.compute_energy(
            labels_diff, unary_shadow, unary_nonshadow, affinity, disparity
        )
        
        # Expected:
        # Unary: φ(shadow) for region 0 + φ(nonshadow) for region 1
        # Pairwise: affinity (different labels)
        expected_unary = unary_shadow[0] + unary_nonshadow[1]
        expected_pairwise = affinity[0, 1]  # x_0 ≠ x_1, so add affinity
        expected_diff = expected_unary + expected_pairwise
        
        self.assertAlmostEqual(energy_diff, expected_diff)
        
        # Test labels: [+1, +1] (same labels)
        labels_same = np.array([1, 1])
        energy_same = mrf.compute_energy(
            labels_same, unary_shadow, unary_nonshadow, affinity, disparity
        )
        
        expected_unary_same = unary_shadow[0] + unary_shadow[1]
        expected_pairwise_same = disparity[0, 1]  # x_0 = x_1, so add disparity
        expected_same = expected_unary_same + expected_pairwise_same
        
        self.assertAlmostEqual(energy_same, expected_same)


class TestDisparityClassifier(unittest.TestCase):
    """
    Test disparity classifier for region pairs.
    
    Reference:
        Paper Section 4.3:
        "For disparity potentials, we classify shadow/non-shadow transitions 
        between two regions of the same material."
    """
    
    def test_feature_extraction(self):
        """
        Test that pairwise features are extracted correctly.
        
        Reference:
            Paper: "We use an RBF kernel with the following features:
            - The χ² distance between the texton histograms
            - The EMD between corresponding L*, a* and b* histograms
            - The average RGB ratios"
        """
        from models.distances import chi_square_distance, emd_1d
        
        classifier = DisparityClassifier()
        
        # Create mock features
        features_i = {
            'L': np.random.rand(21),
            'a': np.random.rand(21),
            'b': np.random.rand(21),
            't': np.random.rand(128)
        }
        # Normalize to sum to 1
        for k in features_i:
            features_i[k] = features_i[k] / features_i[k].sum()
        
        features_j = {
            'L': np.random.rand(21),
            'a': np.random.rand(21),
            'b': np.random.rand(21),
            't': np.random.rand(128)
        }
        for k in features_j:
            features_j[k] = features_j[k] / features_j[k].sum()
        
        rgb_i = np.array([0.5, 0.4, 0.3])
        rgb_j = np.array([0.3, 0.4, 0.5])
        
        pair_features = classifier.extract_pairwise_features(
            features_i, features_j, rgb_i, rgb_j
        )
        
        # Should have 7 features:
        # 1 (χ²) + 3 (EMD L,a,b) + 3 (RGB ratios) = 7
        self.assertEqual(len(pair_features), 7)
        
        # Verify χ² distance
        expected_chi2 = chi_square_distance(features_i['t'], features_j['t'])
        self.assertAlmostEqual(pair_features[0], expected_chi2)
        
        # Verify EMD
        expected_emd_L = emd_1d(features_i['L'], features_j['L'])
        self.assertAlmostEqual(pair_features[1], expected_emd_L)
        
        # Verify RGB ratios
        rho_R = 0.5 / 0.3
        rho_G = 0.4 / 0.4
        rho_B = 0.3 / 0.5
        expected_avg = (rho_R + rho_G + rho_B) / 3
        self.assertAlmostEqual(pair_features[4], expected_avg, places=5)


class TestPaperAlignment(unittest.TestCase):
    """
    Test alignment with paper specifications.
    """
    
    def test_affinity_threshold_value(self):
        """
        Verify default affinity threshold is 0.5 as per paper.
        
        Reference:
            Paper Section 4.2: "K(R_i, R_j) > 0.5"
        """
        mrf = MRFShadowDetector()
        self.assertEqual(mrf.affinity_threshold, 0.5)
        
    def test_geometric_mean_weighting(self):
        """
        Verify geometric mean is used for area weighting.
        
        Reference:
            Paper: "ω_ij = √(ω_i · ω_j)"
        """
        mrf = MRFShadowDetector()
        
        region_probs = np.array([0.5, 0.5])
        region_areas = np.array([100, 400])  # Different areas
        adjacency = np.array([[0, 1], [1, 0]])
        kernel_sim = np.array([[1.0, 0.8], [0.8, 1.0]])
        
        mrf.set_unary_data(region_probs, region_areas)
        mrf.set_adjacency(adjacency)
        mrf.set_affinity_data(kernel_sim)
        
        affinity = mrf.compute_affinity_potentials()
        
        # ω_ij = √(100 × 400) = √40000 = 200
        expected_omega = np.sqrt(100 * 400)
        expected_affinity = expected_omega * 0.8
        
        self.assertAlmostEqual(affinity[0, 1], expected_affinity)


def run_tests():
    """Run all MRF tests."""
    print("=" * 60)
    print("MRF Module Tests")
    print("Testing Section 4: Incorporating Context in Shadow Detection")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRegionUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestUnaryPotentials))
    suite.addTests(loader.loadTestsFromTestCase(TestAffinityPotentials))
    suite.addTests(loader.loadTestsFromTestCase(TestDisparityPotentials))
    suite.addTests(loader.loadTestsFromTestCase(TestMRFOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestDisparityClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestPaperAlignment))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("[PASS] All MRF tests passed!")
    else:
        print("[FAIL] Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

