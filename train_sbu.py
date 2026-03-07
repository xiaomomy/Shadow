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
import cv2
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
from models.platt_scaling import PlattScaler
from models.lssvm import LSSVM
from models.mrf import (
    MRFShadowDetector,
    compute_region_adjacency,
    compute_region_areas
)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Dataset
    'dataset_name': 'sbu',
    'n_train_images': 400,        # Target ~12K regions (paper-scale)
    'holdout_images': 200,        # Held-out images from train set for evaluation
    'random_seed': 42,
    
    # Superpixel & Region (Mean-shift)
    'n_segments': 150,            # SLIC segments per image
    'compactness': 20,
    'region_bandwidth': 0.005,      # Bandwidth for Mean-shift region clustering
    
    # Texton
    'n_textons': 128,             # Paper standard
    'texton_train_images': 200,   # More images for better dictionary
    
    # Optimization (Phase 3)
    'n_iterations': 200,          # Paper standard for Beam Search
    'stagnation_threshold': 25,   # Reset after 25 iterations of no improvement
    'gamma_lssvm': 1.0,           # LSSVM regularization parameter
    
    # Paths
    'output_dir': PROJECT_DIR / 'output' / 'sbu_formal',
    'cache_dir': PROJECT_DIR / 'output' / 'cache',

    # Visualization
    'save_region_visualizations': True,
    'region_vis_candidate_images': 50,  # Randomly sample from first N images
    'region_vis_n_samples': 10,         # Number of images to save
    'region_vis_seed': 42,

    # Test prediction visualization
    'save_test_mask_predictions': True,
    'test_pred_candidate_images': 100,  # Randomly sample from first N test images
    'test_pred_n_samples': 8,           # Number of prediction examples to save
    'test_pred_seed': 7,
    'use_mrf': True,                    # Enable MRF post-processing for predictions

    # Optimization subset to avoid O(N^2) explosion
    'max_region_samples_opt': 15000,    # Upper bound of regions for Beam Search
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


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to uint8 format."""
    if image.dtype == np.uint8:
        return image.copy()
    image = np.asarray(image)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def _colorize_regions(region_labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Create a deterministic pseudo-color map for region labels."""
    n_regions = int(region_labels.max()) + 1
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(n_regions, 3), dtype=np.uint8)
    return palette[region_labels]


def _overlay_shadow_mask(image_rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Overlay ground-truth shadow mask on an RGB image."""
    overlay = image_rgb.copy()
    if mask is None:
        return overlay
    mask_bool = mask > 0
    # Yellow overlay for shadow pixels.
    yellow = np.zeros_like(overlay)
    yellow[..., 0] = 255
    yellow[..., 1] = 255
    alpha = 0.45
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool] + alpha * yellow[mask_bool]
    ).astype(np.uint8)
    return overlay


def _build_segmentation_preview(
    original_rgb: np.ndarray,
    superpixel_overlay: np.ndarray,
    region_overlay: np.ndarray,
    region_colorized: np.ndarray
) -> np.ndarray:
    """Build a 2x2 preview image for segmentation visualization."""
    top = np.concatenate([original_rgb, superpixel_overlay], axis=1)
    bottom = np.concatenate([region_overlay, region_colorized], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _add_title_bar(image_rgb: np.ndarray, title: str) -> np.ndarray:
    """Add a title bar above an RGB panel image."""
    title_bar_height = 34
    h, w = image_rgb.shape[:2]
    title_bar = np.full((title_bar_height, w, 3), 255, dtype=np.uint8)
    cv2.putText(
        title_bar,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA
    )
    return np.concatenate([title_bar, image_rgb], axis=0)


def _build_pred_overlay_panel(
    original_rgb: np.ndarray,
    pred_mask: np.ndarray
) -> np.ndarray:
    """Build a panel with predicted shadow overlay on a darkened image."""
    panel = (original_rgb.astype(np.float32) * 0.4).astype(np.uint8)
    yellow = np.array([240, 240, 30], dtype=np.uint8)
    pred_bool = pred_mask > 0
    alpha = 0.7
    panel[pred_bool] = (
        (1.0 - alpha) * panel[pred_bool] + alpha * yellow
    ).astype(np.uint8)
    return panel


def _build_gt_mask_panel(gt_mask: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    """Build a panel showing the ground-truth mask in white on black."""
    h, w = shape_hw
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    if gt_mask is None:
        return panel
    gt_bool = gt_mask > 0
    panel[gt_bool] = 245
    return panel


def _build_comparison_panel(
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    shape_hw: Tuple[int, int]
) -> np.ndarray:
    """
    Build a panel that compares prediction and annotation.

    Color coding:
    - True Positive: white
    - False Positive: orange
    - False Negative: green
    """
    h, w = shape_hw
    panel = np.full((h, w, 3), 28, dtype=np.uint8)
    if gt_mask is None:
        panel[pred_mask > 0] = np.array([240, 240, 30], dtype=np.uint8)
        return panel

    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    tp = pred_bool & gt_bool
    fp = pred_bool & (~gt_bool)
    fn = (~pred_bool) & gt_bool

    panel[tp] = np.array([245, 245, 245], dtype=np.uint8)
    panel[fp] = np.array([255, 155, 45], dtype=np.uint8)
    panel[fn] = np.array([150, 235, 80], dtype=np.uint8)
    return panel


def _predict_mask_for_image(
    image: np.ndarray,
    slic: SuperpixelSegmenter,
    region_gen: MeanShiftRegionGenerator,
    feat_extractor: PaperCompliantFeatureExtractor,
    final_kernel,
    lssvm: LSSVM,
    train_features: Dict[str, np.ndarray],
    platt: Optional[PlattScaler] = None,
    apply_mrf: bool = False
) -> np.ndarray:
    """Predict a binary shadow mask for a single image using region inference."""
    sp_labels = slic.segment(image)
    region_labels = region_gen.generate_regions(image, sp_labels)
    features = feat_extractor.extract_features_by_channel(
        image, region_labels, use_gpu=USE_GPU
    )

    k_cross = final_kernel.compute_cross(train_features, features)
    decision_values = lssvm.decision_function(None, K=k_cross)
    if platt is not None:
        region_probs = platt.predict_proba(decision_values)
    else:
        region_probs = 1.0 / (1.0 + np.exp(-decision_values))
    region_preds = (region_probs > 0.5).astype(np.int32)

    if apply_mrf:
        try:
            mrf = MRFShadowDetector(use_disparity=False)
            region_areas = compute_region_areas(region_labels)
            adjacency = compute_region_adjacency(region_labels)
            # Unary from Platt probabilities, per paper φ = -ω_i P(x_i|R_i)
            mrf.set_unary_data(region_probs, region_areas)
            mrf.set_adjacency(adjacency)
            # Affinity potentials from self kernel
            K_self = final_kernel.compute(features_by_channel=features)
            mrf.set_affinity_data(K_self)
            region_preds = mrf.optimize(method='qpbo')
        except Exception as exc:  # noqa: BLE001
            log(f"  [MRF] Fallback to raw predictions due to: {exc}")
            region_preds = (region_probs > 0.5).astype(np.int32)

    pred_mask = np.zeros(region_labels.shape, dtype=np.uint8)
    for region_id in range(len(region_preds)):
        if region_preds[region_id] > 0:
            pred_mask[region_labels == region_id] = 1
    return pred_mask


def save_random_test_prediction_visualizations(
    dataset: SBUDataset,
    test_indices: List[int],
    slic: SuperpixelSegmenter,
    region_gen: MeanShiftRegionGenerator,
    feat_extractor: PaperCompliantFeatureExtractor,
    final_kernel,
    lssvm: LSSVM,
    train_features: Dict[str, np.ndarray],
    platt: PlattScaler,
    output_dir: Path,
    candidate_images: int = 100,
    n_samples: int = 8,
    seed: int = 7,
    apply_mrf: bool = False
) -> None:
    """
    Save random test-set mask prediction visualizations.

    A 2x2 panel is saved for each sampled image:
    (a) input image, (b) predicted shadow, (c) annotated shadow, (d) comparison.
    """
    if len(test_indices) == 0:
        log("  [Prediction Preview] No test indices available. Skip.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    first_n = min(candidate_images, len(test_indices))
    if first_n == 0:
        log("  [Prediction Preview] No candidate test images. Skip.")
        return

    sample_count = min(n_samples, first_n)
    rng = np.random.default_rng(seed)
    selected_positions = np.sort(
        rng.choice(first_n, size=sample_count, replace=False)
    )

    log(
        f"Step 7: Saving {sample_count} test prediction previews "
        f"from first {first_n} test images..."
    )

    for rank, pos in enumerate(selected_positions, start=1):
        idx = test_indices[int(pos)]
        try:
            image, gt_mask = dataset[idx]
            original_rgb = _to_uint8_rgb(image)
            pred_mask = _predict_mask_for_image(
                image=image,
                slic=slic,
                region_gen=region_gen,
                feat_extractor=feat_extractor,
                final_kernel=final_kernel,
                lssvm=lssvm,
                train_features=train_features,
                platt=platt,
                apply_mrf=apply_mrf
            )

            pred_panel = _build_pred_overlay_panel(original_rgb, pred_mask)
            gt_panel = _build_gt_mask_panel(gt_mask, original_rgb.shape[:2])
            cmp_panel = _build_comparison_panel(pred_mask, gt_mask, original_rgb.shape[:2])

            p1 = _add_title_bar(original_rgb, "(a) Input image")
            p2 = _add_title_bar(pred_panel, "(b) Predicted shadow")
            p3 = _add_title_bar(gt_panel, "(c) Annotated shadow")
            p4 = _add_title_bar(cmp_panel, "(d) Prediction vs. annotation")

            top = np.concatenate([p1, p2], axis=1)
            bottom = np.concatenate([p3, p4], axis=1)
            preview = np.concatenate([top, bottom], axis=0)

            save_path = output_dir / f"{rank:02d}_test_idx_{idx:05d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            log(f"  [Prediction Preview] Failed for test idx={idx}: {exc}")

    log(f"  [Prediction Preview] Saved to {output_dir}")


def save_random_region_visualizations(
    dataset: SBUDataset,
    indices: List[int],
    slic: SuperpixelSegmenter,
    region_gen: MeanShiftRegionGenerator,
    output_dir: Path,
    candidate_images: int = 50,
    n_samples: int = 10,
    seed: int = 42
) -> None:
    """
    Save random region-segmentation visualizations for qualitative inspection.

    The images are sampled from the first `candidate_images` entries in `indices`.
    """
    if len(indices) == 0:
        log("  [Visualization] No indices available. Skip saving previews.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    first_n = min(candidate_images, len(indices))
    if first_n == 0:
        log("  [Visualization] No candidate images. Skip saving previews.")
        return

    sample_count = min(n_samples, first_n)
    rng = np.random.default_rng(seed)
    selected_positions = np.sort(
        rng.choice(first_n, size=sample_count, replace=False)
    )

    log(
        f"  [Visualization] Saving {sample_count} previews "
        f"from first {first_n} images to {output_dir}"
    )

    for rank, pos in enumerate(selected_positions, start=1):
        idx = indices[int(pos)]
        try:
            image, mask = dataset[idx]
            sp_labels = slic.segment(image)
            region_labels = region_gen.generate_regions(image, sp_labels)

            original_rgb = _to_uint8_rgb(image)
            superpixel_overlay = slic.visualize(original_rgb, color=(255, 0, 0))
            region_overlay = region_gen.visualize(original_rgb, color=(255, 0, 0))

            region_colorized = _colorize_regions(region_labels, seed=seed + idx)
            region_colorized = _overlay_shadow_mask(region_colorized, mask)

            preview = _build_segmentation_preview(
                original_rgb=original_rgb,
                superpixel_overlay=superpixel_overlay,
                region_overlay=region_overlay,
                region_colorized=region_colorized
            )

            save_path = output_dir / f"{rank:02d}_dataset_idx_{idx:05d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            log(f"  [Visualization] Failed for image idx={idx}: {exc}")

    log("  [Visualization] Preview image export finished.")


def compute_pixel_metrics_from_regions(
    region_preds: np.ndarray,
    region_pixel_stats: np.ndarray
) -> Dict[str, float]:
    """
    Compute pixel-level metrics from region predictions.

    Each region prediction is expanded to all pixels in that region by using:
    - region pixel count
    - number of shadow pixels in the ground-truth mask
    """
    region_preds = np.asarray(region_preds, dtype=np.int32).ravel()
    region_pixel_stats = np.asarray(region_pixel_stats, dtype=np.int64)

    if region_pixel_stats.ndim != 2 or region_pixel_stats.shape[1] != 2:
        raise ValueError(
            "region_pixel_stats must have shape (n_regions, 2) "
            "with columns [n_pixels, n_shadow_pixels]."
        )
    if len(region_preds) != len(region_pixel_stats):
        raise ValueError(
            "region_preds and region_pixel_stats must have the same length."
        )

    n_pixels = region_pixel_stats[:, 0].astype(np.float64)
    n_shadow_pixels = region_pixel_stats[:, 1].astype(np.float64)
    n_non_shadow_pixels = n_pixels - n_shadow_pixels
    pred_shadow = (region_preds == 1)

    tp = np.sum(n_shadow_pixels[pred_shadow])
    fp = np.sum(n_non_shadow_pixels[pred_shadow])
    fn = np.sum(n_shadow_pixels[~pred_shadow])
    tn = np.sum(n_non_shadow_pixels[~pred_shadow])

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    fpr_den = fp + tn
    fnr_den = fn + tp
    fpr = fp / fpr_den if fpr_den > 0 else 0.0
    fnr = fn / fnr_den if fnr_den > 0 else 0.0
    ber = 0.5 * (fpr + fnr)

    return {
        'accuracy': float(accuracy),
        'ber': float(ber),
        'fpr': float(fpr),
        'fnr': float(fnr),
    }

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
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Extract region features and labels for a subset of the dataset"""
    all_features = {'L': [], 'a': [], 'b': [], 't': []}
    all_labels = []
    all_region_pixel_stats = []
    
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
                if mask is not None:
                    n_region_pixels = int(np.sum(r_mask))
                    n_shadow_pixels = int(np.sum(mask[r_mask] > 0))
                else:
                    n_region_pixels = int(np.sum(r_mask))
                    n_shadow_pixels = 0
                all_region_pixel_stats.append(
                    [n_region_pixels, n_shadow_pixels]
                )
                
        except Exception as e:
            log(f"  [Warning] Failed to process image {idx}: {e}")
            continue
            
    # Concatenate features into matrices
    for k in all_features:
        all_features[k] = np.array(all_features[k])
    all_labels = np.array(all_labels)
    all_region_pixel_stats = np.array(all_region_pixel_stats, dtype=np.int64)
    
    log(f"  Extraction complete. Total regions: {len(all_labels)}")
    log(f"  Shadow regions: {np.sum(all_labels==1)} ({np.mean(all_labels)*100:.1f}%)")
    
    return all_features, all_labels, all_region_pixel_stats

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
    
    # Sample selection
    rng = np.random.default_rng(CONFIG['random_seed'])
    total_train = min(CONFIG['n_train_images'], len(train_ds))
    all_indices = rng.choice(len(train_ds), total_train, replace=False)
    holdout_count = min(CONFIG['holdout_images'], max(1, total_train // 2))
    holdout_indices = sorted(all_indices[:holdout_count].tolist())
    train_indices = sorted(all_indices[holdout_count:].tolist())
    
    log(f"  Training images: {len(train_indices)}")
    log(f"  Hold-out images: {len(holdout_indices)}")
    
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

    if CONFIG.get('save_region_visualizations', False):
        vis_dir = CONFIG['output_dir'] / 'region_visualizations'
        save_random_region_visualizations(
            dataset=train_ds,
            indices=train_indices,
            slic=slic,
            region_gen=region_gen,
            output_dir=vis_dir,
            candidate_images=CONFIG.get('region_vis_candidate_images', 50),
            n_samples=CONFIG.get('region_vis_n_samples', 10),
            seed=CONFIG.get('region_vis_seed', CONFIG['random_seed'])
        )
    
    # Check cache for features
    feat_cache = CONFIG['cache_dir'] / f"sbu_features_train{len(train_indices)}_holdout{len(holdout_indices)}.pkl"
    if feat_cache.exists():
        log(f"  Loading cached features from {feat_cache}")
        with open(feat_cache, 'rb') as f:
            data = pickle.load(f)
            train_cached = data['train']
            holdout_cached = data['holdout']

            cache_has_pixel_stats = (
                isinstance(train_cached, tuple) and
                isinstance(holdout_cached, tuple) and
                len(train_cached) == 3 and
                len(holdout_cached) == 3
            )

            if cache_has_pixel_stats:
                train_features, train_labels, train_region_stats = train_cached
                holdout_features, holdout_labels, holdout_region_stats = holdout_cached
            else:
                log("  Cache format is outdated (missing pixel stats). Re-extracting...")
                train_features, train_labels, train_region_stats = process_dataset(
                    train_ds, train_indices, slic, region_gen, feat_extractor, "Train"
                )
                holdout_features, holdout_labels, holdout_region_stats = process_dataset(
                    train_ds, holdout_indices, slic, region_gen, feat_extractor, "Holdout"
                )
                with open(feat_cache, 'wb') as wf:
                    pickle.dump({
                        'train': (train_features, train_labels, train_region_stats),
                        'holdout': (holdout_features, holdout_labels, holdout_region_stats)
                    }, wf)
    else:
        train_features, train_labels, train_region_stats = process_dataset(
            train_ds, train_indices, slic, region_gen, feat_extractor, "Train"
        )
        holdout_features, holdout_labels, holdout_region_stats = process_dataset(
            train_ds, holdout_indices, slic, region_gen, feat_extractor, "Holdout"
        )
        log(f"  Caching features to {feat_cache}...")
        with open(feat_cache, 'wb') as f:
            pickle.dump({
                'train': (train_features, train_labels, train_region_stats),
                'holdout': (holdout_features, holdout_labels, holdout_region_stats)
            }, f)

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
    # Subsample regions for optimization to avoid O(N^2) memory blow-up
    max_opt_samples = CONFIG.get('max_region_samples_opt', 12000)
    if len(train_labels) > max_opt_samples:
        rng = np.random.default_rng(CONFIG['random_seed'])
        sample_indices = rng.choice(len(train_labels), size=max_opt_samples, replace=False)
        log(f"  Subsampling {max_opt_samples} regions (out of {len(train_labels)}) for optimization.")
        train_features_opt = {k: v[sample_indices] for k, v in train_features.items()}
        train_labels_opt = train_labels[sample_indices]
    else:
        train_features_opt = train_features
        train_labels_opt = train_labels
        sample_indices = None

    optimal_weights, optimal_sigmas, min_loo_ber = optimizer.optimize(train_features_opt, train_labels_opt)
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
    K_train = final_kernel.compute(
        features_by_channel=train_features,
        use_tqdm=True,
        desc="Final train kernel"
    )
    
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
    log("Step 6: Final Evaluation on Hold-out Set (per-image, MRF optional)...")
    try:
        from tqdm import tqdm  # type: ignore
        eval_iterator = tqdm(holdout_indices, desc="Hold-out eval", total=len(holdout_indices))
    except ImportError:
        eval_iterator = holdout_indices

    tp = fp = fn = tn = 0.0
    evaluated = 0
    for idx in eval_iterator:
        image, gt_mask = train_ds[idx]
        if gt_mask is None:
            continue
        pred_mask = _predict_mask_for_image(
            image=image,
            slic=slic,
            region_gen=region_gen,
            feat_extractor=feat_extractor,
            final_kernel=final_kernel,
            lssvm=lssvm,
            train_features=train_features,
            platt=platt,
            apply_mrf=CONFIG.get('use_mrf', False)
        )
        pred_bool = pred_mask.astype(bool)
        gt_bool = gt_mask.astype(bool)
        tp += float(np.logical_and(pred_bool, gt_bool).sum())
        fp += float(np.logical_and(pred_bool, ~gt_bool).sum())
        fn += float(np.logical_and(~pred_bool, gt_bool).sum())
        tn += float(np.logical_and(~pred_bool, ~gt_bool).sum())
        evaluated += 1

    if evaluated == 0:
        log("  [Warning] No hold-out images with masks available for evaluation.")
        pixel_metrics = {'ber': 0.0, 'accuracy': 0.0, 'fpr': 0.0, 'fnr': 0.0}
    else:
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        ber = (fpr + fnr) / 2.0
        pixel_metrics = {
            'ber': ber,
            'accuracy': accuracy,
            'fpr': fpr,
            'fnr': fnr
        }

    log("=== Results Summary on Hold-out (Pixel-level) ===")
    log(f"  Hold-out BER: {pixel_metrics['ber']*100:.2f}%")
    log(f"  Hold-out Accuracy: {pixel_metrics['accuracy']*100:.2f}%")
    log(f"  Hold-out FPR: {pixel_metrics['fpr']*100:.2f}%")
    log(f"  Hold-out FNR: {pixel_metrics['fnr']*100:.2f}%")
    
    if CONFIG.get('save_test_mask_predictions', False):
        pred_vis_dir = CONFIG['output_dir'] / 'holdout_predictions'
        save_random_test_prediction_visualizations(
            dataset=train_ds,
            test_indices=holdout_indices,
            slic=slic,
            region_gen=region_gen,
            feat_extractor=feat_extractor,
            final_kernel=final_kernel,
            lssvm=lssvm,
            train_features=train_features,
            platt=platt,
            output_dir=pred_vis_dir,
            candidate_images=CONFIG.get('test_pred_candidate_images', 100),
            n_samples=CONFIG.get('test_pred_n_samples', 8),
            seed=CONFIG.get('test_pred_seed', CONFIG['random_seed']),
            apply_mrf=CONFIG.get('use_mrf', False)
        )

    # Save Model
    save_path = CONFIG['output_dir'] / 'sbu_final_model.pkl'
    log(f"Step 8: Saving model to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'lssvm': lssvm,
            'platt': platt,
            'texton_extractor': texton_extractor,
            'optimal_weights': optimal_weights,
            'optimal_sigmas': optimal_sigmas,
            'train_features': train_features, # Needed for future cross-kernel prediction
            'config': CONFIG,
            'test_pixel_metrics': pixel_metrics
        }, f)
    
    log("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
