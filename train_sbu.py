"""
Formal Training Script for Shadow Detection on SBU Dataset.
Implements the joint kernel and classifier optimization (Phase 3)
using Leave-One-Out (LOO) balanced error rate and Beam Search.

This script follows the technical route of the ICCV 2015 paper:
"Leave-One-Out Kernel Optimization for Shadow Detection" (Vicente et al.)

Key features:
1. Full SBU Dataset support (Train/Test).
2. Paper-compliant Joint Optimization (4 weights, 4 sigmas).
3. LOO-based Beam Search (500 iterations).
4. GPU acceleration for distance and kernel calculations.
5. Extensive logging for real-time monitoring.

Author: Shadow Detection Project
Date: 2026-02-12
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add project root to sys.path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# Project modules
from data.dataset_loader import SBUDataset
from preprocessing.superpixel import SuperpixelSegmenter
from preprocessing.region import MeanShiftRegionGenerator
from preprocessing.features import PaperCompliantFeatureExtractor
from preprocessing.texton import TextonFeatureExtractor
from models.loo_optimizer import PaperBeamSearchOptimizer
from models.platt_scaling import PlattScaler, balanced_error_rate
from models.lssvm import LSSVM

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Dataset
    'dataset_name': 'sbu',
    'n_train_images': 4000,       # Use almost full SBU train set (approx 4014)
    'n_test_images': 638,         # Full SBU test set
    'random_seed': 42,
    
    # Superpixel & Region (Mean-shift)
    'n_segments': 150,            # SLIC segments per image
    'compactness': 20,
    'region_bandwidth': 0.3,      # Bandwidth for Mean-shift region clustering
    
    # Texton
    'n_textons': 128,             # Paper standard
    'texton_train_images': 200,   # More images for better dictionary
    
    # Optimization (Phase 3)
    'n_iterations': 500,          # Paper standard for Beam Search
    'stagnation_threshold': 25,   # Reset after 25 iterations of no improvement
    'gamma_lssvm': 1.0,           # LSSVM regularization parameter
    
    # Paths
    'output_dir': PROJECT_DIR / 'output' / 'sbu_formal',
    'cache_dir': PROJECT_DIR / 'output' / 'cache',
}

# GPU Setup
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

# =============================================================================
# Utility Functions
# =============================================================================

def log(message: str):
    """Timestamped logging"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_region_label(mask: np.ndarray, region_mask: np.ndarray) -> int:
    """Determine region label based on majority pixel label"""
    if mask is None: return 0
    shadow_ratio = np.mean(mask[region_mask] > 0)
    return 1 if shadow_ratio > 0.5 else 0

# =============================================================================
# Feature Extraction Pipeline
# =============================================================================

def process_dataset(
    dataset: SBUDataset,
    indices: List[int],
    slic: SuperpixelSegmenter,
    region_gen: MeanShiftRegionGenerator,
    feat_extractor: PaperCompliantFeatureExtractor,
    desc: str = "Processing"
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Extract region features and labels for a subset of the dataset"""
    all_features = {'L': [], 'a': [], 'b': [], 't': []}
    all_labels = []
    
    log(f"Starting feature extraction for {len(indices)} images ({desc})...")
    
    for i, idx in enumerate(indices):
        if (i + 1) % 50 == 0 or i == 0:
            log(f"  Progress: {i+1}/{len(indices)} images processed...")
            
        img, mask = dataset[idx]
        
        try:
            # 1. Superpixel Segmentation
            sp_labels = slic.segment(img)
            
            # 2. Region Generation (Mean-shift)
            region_labels = region_gen.generate_regions(img, sp_labels)
            n_regions = int(region_labels.max()) + 1
            
            # 3. Extract Features (GPU accelerated if enabled)
            feats = feat_extractor.extract_features_by_channel(img, region_labels, use_gpu=USE_GPU)
            
            # 4. Filter and label regions
            for r_id in range(n_regions):
                r_mask = (region_labels == r_id)
                if np.sum(r_mask) < 20: continue # Skip tiny regions
                
                for k in all_features.keys():
                    all_features[k].append(feats[k][r_id])
                all_labels.append(get_region_label(mask, r_mask))
                
        except Exception as e:
            log(f"  [Warning] Failed to process image {idx}: {e}")
            continue
            
    # Concatenate features into matrices
    for k in all_features:
        all_features[k] = np.array(all_features[k])
    all_labels = np.array(all_labels)
    
    log(f"  Extraction complete. Total regions: {len(all_labels)}")
    log(f"  Shadow regions: {np.sum(all_labels==1)} ({np.mean(all_labels)*100:.1f}%)")
    
    return all_features, all_labels

# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    log("=== SBU Formal Training Pipeline (ICCV 2015 Reproduction) ===")
    log(f"Device: {DEVICE}")
    
    # Initialize paths
    CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['cache_dir'].mkdir(parents=True, exist_ok=True)
    
    # 1. Load Dataset
    log("Step 1: Loading SBU dataset...")
    train_ds = SBUDataset(split='train')
    test_ds = SBUDataset(split='test')
    
    # Sample selection
    np.random.seed(CONFIG['random_seed'])
    train_indices = np.random.choice(len(train_ds), min(CONFIG['n_train_images'], len(train_ds)), replace=False).tolist()
    test_indices = list(range(min(CONFIG['n_test_images'], len(test_ds))))
    
    # 2. Build Texton Dictionary
    log("Step 2: Building Texton dictionary...")
    texton_extractor = TextonFeatureExtractor(n_textons=CONFIG['n_textons'])
    
    texton_cache = CONFIG['cache_dir'] / f"texton_dict_{CONFIG['n_textons']}.pkl"
    if texton_cache.exists():
        log(f"  Loading cached texton dictionary from {texton_cache}")
        texton_extractor.load_dictionary(str(texton_cache))
    else:
        log(f"  Training texton dictionary on {CONFIG['texton_train_images']} images...")
        tx_imgs = [train_ds[i][0] for i in train_indices[:CONFIG['texton_train_images']]]
        texton_extractor.build_dictionary(tx_imgs, verbose=True)
        texton_extractor.save_dictionary(str(texton_cache))
        log(f"  Texton dictionary saved to {texton_cache}")

    # 3. Feature Extraction
    log("Step 3: Extracting features for all regions...")
    slic = SuperpixelSegmenter(n_segments=CONFIG['n_segments'], compactness=CONFIG['compactness'])
    region_gen = MeanShiftRegionGenerator(bandwidth=CONFIG['region_bandwidth'])
    feat_extractor = PaperCompliantFeatureExtractor(texton_extractor=texton_extractor)
    
    # Check cache for features
    feat_cache = CONFIG['cache_dir'] / f"sbu_features_n{len(train_indices)}.pkl"
    if feat_cache.exists():
        log(f"  Loading cached features from {feat_cache}")
        with open(feat_cache, 'rb') as f:
            data = pickle.load(f)
            train_features, train_labels = data['train']
            test_features, test_labels = data['test']
    else:
        train_features, train_labels = process_dataset(train_ds, train_indices, slic, region_gen, feat_extractor, "Train")
        test_features, test_labels = process_dataset(test_ds, test_indices, slic, region_gen, feat_extractor, "Test")
        log(f"  Caching features to {feat_cache}...")
        with open(feat_cache, 'wb') as f:
            pickle.dump({'train': (train_features, train_labels), 'test': (test_features, test_labels)}, f)

    # 4. Joint Kernel Optimization (Beam Search)
    log("Step 4: Joint Kernel Optimization (Phase 3)...")
    log("  Initializing Paper-compliant Beam Search Optimizer...")
    optimizer = PaperBeamSearchOptimizer(
        n_iterations=CONFIG['n_iterations'],
        stagnation_threshold=CONFIG['stagnation_threshold'],
        gamma_lssvm=CONFIG['gamma_lssvm'],
        verbose=True
    )
    
    start_opt = time.time()
    optimal_weights, optimal_sigmas, min_loo_ber = optimizer.optimize(train_features, train_labels)
    opt_duration = time.time() - start_opt
    
    log(f"  Optimization finished in {opt_duration/60:.1f} minutes.")
    log(f"  Best LOO BER: {min_loo_ber*100:.2f}%")
    log(f"  Optimal Weights: {optimal_weights}")
    log(f"  Optimal Sigmas: {optimal_sigmas}")

    # 5. Final Model Training & Evaluation
    log("Step 5: Training final LSSVM with optimal parameters...")
    # Create final kernel and fit
    final_kernel = optimizer.get_optimized_kernel()
    
    # Precompute kernel for training (efficient)
    log("  Computing final kernel matrix for training samples...")
    K_train = final_kernel.compute(features_by_channel=train_features)
    
    # Fit LSSVM
    lssvm = LSSVM(kernel=final_kernel, gamma=CONFIG['gamma_lssvm'], use_gpu=USE_GPU)
    X_dummy = np.zeros((len(train_labels), 1)) # Dummy X as we use precomputed K
    lssvm.fit(X_dummy, train_labels, K=K_train)
    
    # Platt Scaling for probabilities
    log("  Fitting Platt scaling for probability calibration...")
    dec_values = lssvm.decision_function(X_dummy, K=K_train)
    platt = PlattScaler()
    platt.fit(dec_values, train_labels)
    
    # Evaluation
    log("Step 6: Final Evaluation on Test Set...")
    # Compute test kernel K(train, test)
    log("  Computing cross-kernel matrix for test set...")
    K_test = final_kernel.compute_cross(train_features, test_features)
    
    # Predictions
    test_dec_values = lssvm.decision_function(None, K=K_test)
    test_preds = (test_dec_values > 0).astype(int)
    test_ber = balanced_error_rate(test_labels, test_preds)
    
    log(f"=== Results Summary ===")
    log(f"  Test BER: {test_ber*100:.2f}%")
    log(f"  Test Accuracy: {np.mean(test_preds == test_labels)*100:.2f}%")
    
    # Save Model
    save_path = CONFIG['output_dir'] / 'sbu_final_model.pkl'
    log(f"Step 7: Saving model to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'lssvm': lssvm,
            'platt': platt,
            'texton_extractor': texton_extractor,
            'optimal_weights': optimal_weights,
            'optimal_sigmas': optimal_sigmas,
            'train_features': train_features, # Needed for future cross-kernel prediction
            'config': CONFIG,
            'test_ber': test_ber
        }, f)
    
    log("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
