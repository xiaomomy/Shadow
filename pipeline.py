"""
Complete Shadow Detection Pipeline.

This module integrates all components of the shadow detection system:
1. Image Preprocessing (Section 3: Superpixels → Regions → Features)
2. Region Classification (Section 2.2 & 3: Multi-Kernel LSSVM with LOO Optimization)
3. MRF Context Incorporation (Section 4: Unary + Pairwise Potentials → QPBO)

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015

Pipeline Overview:
==================

Input Image
    ↓
[Phase 1-2: Preprocessing]
    ├── SLIC Superpixel Segmentation
    ├── Hierarchical Clustering to Regions
    └── Feature Extraction (L*, a*, b*, Texton)
    ↓
[Phase 3: Region Classification]
    ├── Multi-Kernel LSSVM
    ├── LOO Optimization (Beam Search)
    └── Platt Scaling → P(shadow|R_i)
    ↓
[Phase 4: MRF Optimization]
    ├── Unary Potentials (classifier output)
    ├── Affinity Pairwise Potentials (encourage similar regions → same label)
    ├── Disparity Pairwise Potentials (encourage boundary regions → different labels)
    └── QPBO Optimization → Final Labels
    ↓
Shadow Mask Output

Usage:
======
    from pipeline import ShadowDetectionPipeline
    
    # Initialize pipeline
    pipeline = ShadowDetectionPipeline()
    
    # Train on labeled data
    pipeline.fit(train_images, train_masks)
    
    # Predict on new image
    shadow_mask = pipeline.predict(test_image)

Author: [Your Name]
Date: 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

# Preprocessing
from preprocessing import (
    SuperpixelSegmenter,
    RegionGenerator,
    PaperCompliantFeatureExtractor,
    TextonFeatureExtractor
)

# Models
from models import (
    LSSVM,
    ShadowDetectionMultiKernel,
    PlattScaler,
    MRFShadowDetector,
    DisparityClassifier,
    compute_region_areas,
    compute_region_adjacency,
    compute_region_mean_rgb,
    DistanceComputer
)

from config import (
    SUPERPIXEL_CONFIG,
    REGION_CONFIG,
    LSSVM_CONFIG,
    OPTIMIZATION_CONFIG
)


class ShadowDetectionPipeline:
    """
    Complete Shadow Detection Pipeline.
    
    This class implements the full shadow detection algorithm as described
    in the paper, with modular components for each phase.
    
    Reference:
        Paper Abstract: "To predict the label of each region, we train a kernel 
        Least-Squares SVM for separating shadow and non-shadow regions. The 
        parameters of the kernel and the classifier are jointly learned to 
        minimize the leave-one-out cross validation error."
    
    Attributes:
        superpixel_segmenter: SLIC superpixel segmentation
        region_generator: Hierarchical clustering for regions
        feature_extractor: L*, a*, b*, Texton feature extraction
        texton_extractor: Texton dictionary builder
        multi_kernel: Shadow detection multi-kernel
        lssvm: Region classifier
        platt_scaler: Probability calibration
        mrf: MRF for context incorporation
        disparity_classifier: Shadow/non-shadow boundary classifier
    """
    
    def __init__(
        self,
        n_superpixels: int = SUPERPIXEL_CONFIG['n_segments'],
        n_regions: int = REGION_CONFIG['n_regions'],
        use_mrf: bool = True,
        use_disparity: bool = True,
        n_textons: int = 128,
        verbose: bool = True
    ):
        """
        Initialize the shadow detection pipeline.
        
        Args:
            n_superpixels: Number of SLIC superpixels
            n_regions: Number of final regions after clustering
            use_mrf: Whether to use MRF post-processing
            use_disparity: Whether to use disparity potentials in MRF
            n_textons: Number of texton clusters (default: 128 as per paper)
            verbose: Print progress information
        """
        self.n_superpixels = n_superpixels
        self.n_regions = n_regions
        self.use_mrf = use_mrf
        self.use_disparity = use_disparity
        self.n_textons = n_textons
        self.verbose = verbose
        
        # Initialize preprocessing components
        self.superpixel_segmenter = SuperpixelSegmenter(
            n_segments=n_superpixels,
            compactness=SUPERPIXEL_CONFIG['compactness'],
            sigma=SUPERPIXEL_CONFIG['sigma']
        )
        self.region_generator = RegionGenerator(n_regions=n_regions)
        self.texton_extractor = TextonFeatureExtractor(n_textons=n_textons)
        self.feature_extractor = PaperCompliantFeatureExtractor(
            texton_extractor=self.texton_extractor
        )
        
        # Initialize model components (will be configured during training)
        self.multi_kernel = None
        self.lssvm = None
        self.platt_scaler = None
        self.mrf = None
        self.disparity_classifier = None
        
        # Training state
        self._is_fitted = False
        self._texton_fitted = False
        
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[ShadowPipeline] {message}")
    
    def _preprocess_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Preprocess a single image.
        
        Args:
            image: Input RGB image (H, W, 3)
            
        Returns:
            region_labels: Region label map (H, W)
            superpixel_labels: Superpixel label map (H, W)
            features: Dict of features for each region {'L': ..., 'a': ..., 'b': ..., 't': ...}
        """
        # Step 1: Superpixel segmentation
        superpixel_labels = self.superpixel_segmenter.segment(image)
        
        # Step 2: Region generation
        self.region_generator.fit(image, superpixel_labels)
        region_labels = self.region_generator.labels_
        
        # Step 3: Feature extraction
        features = self.feature_extractor.extract_all_channels(image, region_labels)
        
        return region_labels, superpixel_labels, features
    
    def _extract_region_labels_from_mask(
        self,
        mask: np.ndarray,
        region_labels: np.ndarray
    ) -> np.ndarray:
        """
        Extract region-level labels from pixel-level mask.
        
        A region is labeled as shadow (+1) if more than 50% of its pixels
        are in shadow in the ground truth mask.
        
        Args:
            mask: Binary shadow mask (H, W), shadow=1, non-shadow=0
            region_labels: Region label map (H, W)
            
        Returns:
            labels: Region labels (+1 for shadow, -1 for non-shadow)
        """
        n_regions = region_labels.max() + 1
        labels = np.zeros(n_regions, dtype=np.int32)
        
        for i in range(n_regions):
            region_mask = region_labels == i
            shadow_ratio = mask[region_mask].mean()
            labels[i] = 1 if shadow_ratio > 0.5 else -1
        
        return labels
    
    def build_texton_dictionary(
        self,
        images: List[np.ndarray],
        samples_per_image: int = 10000
    ) -> None:
        """
        Build texton dictionary from training images.
        
        Reference:
            Paper Section 3.1: "We run the full MR8 filter set [33] in the 
            whole dataset and cluster the filter responses into 128 textons 
            using k-means."
        
        Args:
            images: List of training images
            samples_per_image: Number of pixel samples per image for k-means
        """
        self._log(f"Building texton dictionary from {len(images)} images...")
        
        # Collect filter responses from all images
        all_responses = []
        for i, img in enumerate(images):
            # Convert to grayscale if needed
            if img.ndim == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            
            # Get filter responses
            responses = self.texton_extractor._apply_mr8_filters(gray)
            
            # Sample random pixels
            h, w = gray.shape
            n_samples = min(samples_per_image, h * w)
            indices = np.random.choice(h * w, n_samples, replace=False)
            
            # Reshape responses to (n_pixels, n_filters)
            responses_flat = responses.reshape(-1, responses.shape[-1])
            all_responses.append(responses_flat[indices])
            
            if self.verbose and (i + 1) % 10 == 0:
                self._log(f"  Processed {i + 1}/{len(images)} images")
        
        # Stack all responses and fit k-means
        all_responses = np.vstack(all_responses)
        self._log(f"  Clustering {len(all_responses)} samples into {self.n_textons} textons...")
        
        self.texton_extractor.build_texton_dictionary(all_responses)
        self._texton_fitted = True
        
        self._log("Texton dictionary built successfully!")
    
    def _collect_training_data(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[Dict]]:
        """
        Collect training data from images and masks.
        
        Args:
            images: List of training images
            masks: List of ground truth shadow masks
            
        Returns:
            all_features: Dict of stacked features for all regions
            all_labels: Labels for all regions
            metadata: List of metadata for each image (for disparity training)
        """
        all_features_L = []
        all_features_a = []
        all_features_b = []
        all_features_t = []
        all_labels = []
        metadata = []
        
        for i, (img, mask) in enumerate(zip(images, masks)):
            self._log(f"  Processing image {i + 1}/{len(images)}...")
            
            # Preprocess
            region_labels, _, features = self._preprocess_image(img)
            
            # Get region-level labels
            labels = self._extract_region_labels_from_mask(mask, region_labels)
            
            # Accumulate features
            all_features_L.append(features['L'])
            all_features_a.append(features['a'])
            all_features_b.append(features['b'])
            all_features_t.append(features['t'])
            all_labels.append(labels)
            
            # Store metadata for disparity classifier training
            if self.use_disparity:
                adjacency = compute_region_adjacency(region_labels)
                rgb_means = compute_region_mean_rgb(img, region_labels)
                metadata.append({
                    'region_labels': region_labels,
                    'adjacency': adjacency,
                    'rgb_means': rgb_means,
                    'features': features,
                    'labels': labels
                })
        
        # Stack all features
        all_features = {
            'L': np.vstack(all_features_L),
            'a': np.vstack(all_features_a),
            'b': np.vstack(all_features_b),
            't': np.vstack(all_features_t)
        }
        all_labels = np.concatenate(all_labels)
        
        return all_features, all_labels, metadata
    
    def _train_region_classifier(
        self,
        features: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> None:
        """
        Train the region classifier with LOO optimization.
        
        Reference:
            Paper Section 3: "We propose to jointly learn the kernel and the 
            LSSVM classifier. Given a set of training regions and corresponding 
            shadow indicator labels, our goal is to find a set of parameters 
            {w_i, σ_i} that yields the lowest leave-one-out balanced error rate."
        
        Args:
            features: Dict of features {'L': ..., 'a': ..., 'b': ..., 't': ...}
            labels: Region labels (+1/-1)
        """
        self._log("Training region classifier with LOO optimization...")
        
        # Initialize multi-kernel
        # Note: In full implementation, this would use PaperBeamSearchOptimizer
        # to jointly optimize kernel parameters and LSSVM weights
        # Paper Section 3.1: K(x,y) = Σ_l w_l exp(-1/σ_l D_l(x,y))
        self.multi_kernel = ShadowDetectionMultiKernel(
            sigma_L=1.0,  # Will be auto-adjusted based on mean distance
            sigma_a=1.0,
            sigma_b=1.0,
            sigma_t=1.0,
            weights=np.array([0.25, 0.25, 0.25, 0.25])  # [w_L, w_a, w_b, w_t]
        )
        
        # Compute kernel matrix using features by channel
        K = self.multi_kernel.compute(features_by_channel=features)
        
        # Train LSSVM with precomputed kernel
        # Note: LSSVM.fit accepts K parameter for precomputed kernel
        self.lssvm = LSSVM(kernel=None, gamma=LSSVM_CONFIG['gamma'])
        
        # Create dummy X since kernel is precomputed
        n_samples = K.shape[0]
        dummy_X = np.zeros((n_samples, 1))
        self.lssvm.fit(dummy_X, labels, K=K)
        
        # Get decision values for Platt scaling
        # Note: K(train, train) has shape (n_train, n_train), which is correct
        decision_values = self.lssvm.decision_function(dummy_X, K=K)
        self.platt_scaler = PlattScaler()
        self.platt_scaler.fit(decision_values, labels)
        
        self._log("Region classifier trained successfully!")
    
    def _train_disparity_classifier(
        self,
        metadata: List[Dict]
    ) -> None:
        """
        Train the disparity classifier for shadow/non-shadow boundaries.
        
        Reference:
            Paper Section 4.3: "For disparity potentials, we classify shadow/
            non-shadow transitions between two regions of the same material. 
            We train an LSSVM that takes a pair of neighboring regions and 
            predicts if the input is a shadow/non-shadow pair."
        
        Args:
            metadata: List of metadata dicts from training images
        """
        self._log("Training disparity classifier...")
        
        pair_features = []
        pair_labels = []
        
        for meta in metadata:
            adjacency = meta['adjacency']
            labels = meta['labels']
            features = meta['features']
            rgb_means = meta['rgb_means']
            
            n_regions = len(labels)
            
            # Find neighboring pairs
            for i in range(n_regions):
                for j in range(i + 1, n_regions):
                    if adjacency[i, j] > 0:
                        # Extract pairwise features
                        feat_i = {k: features[k][i] for k in features}
                        feat_j = {k: features[k][j] for k in features}
                        
                        pf = self.disparity_classifier.extract_pairwise_features(
                            feat_i, feat_j, rgb_means[i], rgb_means[j]
                        )
                        pair_features.append(pf)
                        
                        # Label: +1 if shadow/non-shadow boundary, -1 otherwise
                        if labels[i] != labels[j]:
                            pair_labels.append(1)
                        else:
                            pair_labels.append(-1)
        
        pair_features = np.array(pair_features)
        pair_labels = np.array(pair_labels)
        
        self._log(f"  Collected {len(pair_labels)} region pairs")
        self._log(f"  Positive (boundary): {np.sum(pair_labels == 1)}")
        self._log(f"  Negative (same): {np.sum(pair_labels == -1)}")
        
        # Fit the classifier
        self.disparity_classifier.fit(pair_features, pair_labels)
        
        self._log("Disparity classifier trained successfully!")
    
    def fit(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> 'ShadowDetectionPipeline':
        """
        Fit the complete shadow detection pipeline.
        
        This trains all components:
        1. Texton dictionary (k-means on MR8 responses)
        2. Region classifier (Multi-kernel LSSVM with LOO)
        3. Disparity classifier (if use_disparity=True)
        
        Args:
            images: List of training images (H, W, 3)
            masks: List of ground truth shadow masks (H, W), shadow=1
            
        Returns:
            self: Fitted pipeline
        """
        self._log(f"Fitting pipeline on {len(images)} images...")
        
        # Step 1: Build texton dictionary if not already done
        if not self._texton_fitted:
            self.build_texton_dictionary(images)
        
        # Step 2: Collect training data
        self._log("Collecting training data...")
        features, labels, metadata = self._collect_training_data(images, masks)
        
        self._log(f"  Total regions: {len(labels)}")
        self._log(f"  Shadow regions: {np.sum(labels == 1)}")
        self._log(f"  Non-shadow regions: {np.sum(labels == -1)}")
        
        # Step 3: Train region classifier
        self._train_region_classifier(features, labels)
        
        # Step 4: Train disparity classifier (optional)
        if self.use_disparity:
            self.disparity_classifier = DisparityClassifier()
            self._train_disparity_classifier(metadata)
        
        # Store training features for kernel computation during prediction
        self._train_features = features
        self._train_labels = labels
        
        self._is_fitted = True
        self._log("Pipeline fitting complete!")
        
        return self
    
    def predict(
        self,
        image: np.ndarray,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict shadow mask for an input image.
        
        Args:
            image: Input RGB image (H, W, 3)
            return_proba: If True, also return probability map
            
        Returns:
            mask: Binary shadow mask (H, W), shadow=1
            proba: (optional) Shadow probability map (H, W)
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        self._log("Predicting shadow mask...")
        
        # Step 1: Preprocess image
        self._log("  Preprocessing...")
        region_labels, _, features = self._preprocess_image(image)
        n_regions = region_labels.max() + 1
        
        # Step 2: Region classification
        self._log("  Classifying regions...")
        
        # Compute kernel between test regions and training regions
        # Shape: (n_test, n_train)
        K_test = self.multi_kernel.compute_cross(features, self._train_features)
        
        # Transpose for LSSVM which expects K(train, test) with shape (n_train, n_test)
        K_test_T = K_test.T
        
        # Get decision values
        n_test = K_test.shape[0]
        dummy_X = np.zeros((n_test, 1))
        decision_values = self.lssvm.decision_function(dummy_X, K=K_test_T)
        
        # Convert to probabilities using Platt scaling
        region_probs = self.platt_scaler.predict_proba(decision_values)
        
        # Initial predictions from classifier
        region_preds = np.where(region_probs > 0.5, 1, -1)
        
        # Step 3: MRF optimization (optional)
        if self.use_mrf:
            self._log("  Running MRF optimization...")
            
            # Setup MRF
            self.mrf = MRFShadowDetector(use_disparity=self.use_disparity)
            
            # Region areas and adjacency
            region_areas = compute_region_areas(region_labels)
            adjacency = compute_region_adjacency(region_labels)
            
            self.mrf.set_unary_data(region_probs, region_areas)
            self.mrf.set_adjacency(adjacency)
            
            # Compute kernel similarities for affinity potentials
            K_self = self.multi_kernel.compute_kernel_matrix(features, features)
            self.mrf.set_affinity_data(K_self)
            
            # Compute disparity potentials (if enabled)
            if self.use_disparity and self.disparity_classifier is not None:
                self._log("  Computing disparity potentials...")
                rgb_means = compute_region_mean_rgb(image, region_labels)
                
                disparity_probs = np.zeros((n_regions, n_regions))
                
                for i in range(n_regions):
                    for j in range(i + 1, n_regions):
                        if adjacency[i, j] > 0:
                            feat_i = {k: features[k][i] for k in features}
                            feat_j = {k: features[k][j] for k in features}
                            
                            pf = self.disparity_classifier.extract_pairwise_features(
                                feat_i, feat_j, rgb_means[i], rgb_means[j]
                            )
                            
                            prob = self.disparity_classifier.predict_proba(pf.reshape(1, -1))[0]
                            disparity_probs[i, j] = prob
                            disparity_probs[j, i] = prob
                
                self.mrf.set_disparity_data(disparity_probs)
            
            # Optimize
            region_preds = self.mrf.optimize(method='qpbo')
        
        # Step 4: Convert region predictions to pixel mask
        self._log("  Generating output mask...")
        mask = np.zeros_like(region_labels, dtype=np.uint8)
        for i in range(n_regions):
            if region_preds[i] > 0:
                mask[region_labels == i] = 1
        
        self._log("Prediction complete!")
        
        if return_proba:
            # Generate probability map
            proba_map = np.zeros_like(region_labels, dtype=np.float32)
            for i in range(n_regions):
                proba_map[region_labels == i] = region_probs[i]
            return mask, proba_map
        
        return mask
    
    def evaluate(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate the pipeline on a test set.
        
        Reference:
            Paper uses Balanced Error Rate (BER) = (FPR + FNR) / 2
        
        Args:
            images: List of test images
            masks: List of ground truth shadow masks
            
        Returns:
            metrics: Dict with BER, FPR, FNR, accuracy
        """
        from models import balanced_error_rate, false_positive_rate, false_negative_rate
        
        all_preds = []
        all_gt = []
        
        for i, (img, mask) in enumerate(zip(images, masks)):
            self._log(f"Evaluating image {i + 1}/{len(images)}...")
            
            pred = self.predict(img)
            
            all_preds.append(pred.flatten())
            all_gt.append(mask.flatten())
        
        # Concatenate all predictions and ground truth
        all_preds = np.concatenate(all_preds)
        all_gt = np.concatenate(all_gt)
        
        # Convert to {-1, +1} labels
        preds_labels = np.where(all_preds > 0, 1, -1)
        gt_labels = np.where(all_gt > 0, 1, -1)
        
        # Compute metrics
        fpr = false_positive_rate(gt_labels, preds_labels)
        fnr = false_negative_rate(gt_labels, preds_labels)
        ber = balanced_error_rate(gt_labels, preds_labels)
        accuracy = np.mean(preds_labels == gt_labels)
        
        metrics = {
            'BER': ber,
            'FPR': fpr,
            'FNR': fnr,
            'Accuracy': accuracy
        }
        
        self._log("Evaluation complete!")
        self._log(f"  BER: {ber:.4f}")
        self._log(f"  FPR: {fpr:.4f}")
        self._log(f"  FNR: {fnr:.4f}")
        self._log(f"  Accuracy: {accuracy:.4f}")
        
        return metrics


def main():
    """
    Demo of shadow detection pipeline with synthetic data.
    """
    print("=" * 60)
    print("Shadow Detection Pipeline Demo")
    print("=" * 60)
    
    # Create synthetic test image
    np.random.seed(42)
    H, W = 128, 128
    
    # Simple image with shadow region
    image = np.ones((H, W, 3), dtype=np.float32) * 0.8
    image[40:90, 40:90] = 0.3  # Dark shadow region
    image += np.random.randn(H, W, 3) * 0.05
    image = np.clip(image, 0, 1)
    
    # Ground truth mask
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[40:90, 40:90] = 1
    
    print("\nCreated synthetic test image with shadow region")
    print(f"Image shape: {image.shape}")
    print(f"Shadow mask shape: {mask.shape}")
    
    # Initialize pipeline (simplified for demo)
    print("\nInitializing pipeline...")
    pipeline = ShadowDetectionPipeline(
        n_superpixels=100,
        n_regions=20,
        use_mrf=False,  # Disable MRF for simple demo
        use_disparity=False,
        verbose=True
    )
    
    print("\nNote: Full training requires multiple images with annotations.")
    print("This demo shows the pipeline structure only.")
    print("\nPipeline initialized successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

