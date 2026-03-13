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
from baseline import build_baseline_kernel
from baseline.cnn_baseline import run_cnn_region_baseline

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
    'method': 'lookop',              # ['lookop', 'unary_svm', 'mk_svm', 'cnn']
    'load_existing_model': True,     # Skip optimization if model exists
    'n_iterations': 500,          # Paper standard for Beam Search
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
    'test_pred_best_n': 20,             # Number of best prediction examples to save
    'use_mrf': True,                    # Enable MRF post-processing for predictions
    'max_region_samples_opt': 15000,    # Upper bound of regions for Beam Search
    
    # CNN baseline
    'cnn_resize': 64,
    'cnn_epochs': 3,
    'cnn_batch_size': 4,
    'cnn_lr': 1e-3,
    'cnn_num_workers': 0,
    'cnn_context_ratio': 0.1,      # Context padding around region bbox
    'cnn_min_region_pixels': 20,   # Skip tiny regions
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

def log_dataset_stats(labels: np.ndarray, n_images: int, desc: str):
    """Log dataset region statistics"""
    n_regions = len(labels)
    log(f"  [{desc}] Total images: {n_images}")
    log(f"  [{desc}] Total regions: {n_regions}")
    if n_images > 0:
        log(f"  [{desc}] Avg regions per image: {n_regions/n_images:.1f}")
    if n_regions > 0:
        log(f"  [{desc}] Shadow regions: {np.sum(labels==1)} ({np.mean(labels)*100:.1f}%)")

def extract_region_patches(
    dataset: SBUDataset,
    indices: List[int],
    slic: SuperpixelSegmenter,
    region_gen: MeanShiftRegionGenerator,
    context_ratio: float = 0.1,
    min_pixels: int = 20,
    desc: str = "Dataset"
) -> Tuple[List[np.ndarray], List[int], List[List[int]]]:
    """
    Extract region patches (RGB) and labels for CNN baseline.
    The region generation strictly matches the pipeline (SLIC + MeanShift).
    """
    patches: List[np.ndarray] = []
    labels: List[int] = []
    region_stats: List[List[int]] = []
    for i, idx in enumerate(indices):
        if (i + 1) % 50 == 0 or i == 0:
            log(f"  [{desc}] Progress: {i+1}/{len(indices)} images processed...")
        image, mask = dataset[idx]
        sp_labels = slic.segment(image)
        region_labels = region_gen.generate_regions(image, sp_labels)
        n_regions = int(region_labels.max()) + 1
        h, w = region_labels.shape
        for r_id in range(n_regions):
            r_mask = (region_labels == r_id)
            pixel_count = int(np.sum(r_mask))
            if pixel_count < min_pixels:
                continue
            ys, xs = np.nonzero(r_mask)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            pad_y = int((y1 - y0 + 1) * context_ratio)
            pad_x = int((x1 - x0 + 1) * context_ratio)
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)
            x0 = max(0, x0 - pad_x)
            x1 = min(w - 1, x1 + pad_x)
            crop = image[y0:y1 + 1, x0:x1 + 1]
            label = get_region_label(mask, r_mask)
            patches.append(crop.astype(np.uint8))
            labels.append(label)
            if mask is not None:
                n_shadow = int(np.sum(mask[r_mask] > 0))
            else:
                n_shadow = 0
            region_stats.append([pixel_count, n_shadow])
    return patches, labels, region_stats

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
) -> Tuple[np.ndarray, int]:
    """Predict a binary shadow mask for a single image using region inference."""
    sp_labels = slic.segment(image)
    region_labels = region_gen.generate_regions(image, sp_labels)
    n_img_regions = int(region_labels.max()) + 1
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
    return pred_mask, n_img_regions


def save_prediction_visualizations(
    dataset: SBUDataset,
    indices: List[int],
    slic: SuperpixelSegmenter,
    region_gen: MeanShiftRegionGenerator,
    feat_extractor: PaperCompliantFeatureExtractor,
    final_kernel,
    lssvm: LSSVM,
    train_features: Dict[str, np.ndarray],
    platt: PlattScaler,
    output_dir: Path,
    apply_mrf: bool = False,
    metrics_map: Optional[Dict[int, float]] = None
) -> None:
    """
    Save prediction visualizations for a specific list of indices.

    A 2x2 panel is saved for each image:
    (a) input image, (b) predicted shadow, (c) annotated shadow, (d) comparison.
    If metrics_map is provided, the BER value is added to the panel title.
    """
    if len(indices) == 0:
        log("  [Prediction Preview] No indices provided. Skip.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Step 7: Saving {len(indices)} prediction previews to {output_dir}...")

    for rank, idx in enumerate(indices, start=1):
        try:
            image, gt_mask = dataset[idx]
            original_rgb = _to_uint8_rgb(image)
            pred_mask, _ = _predict_mask_for_image(
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

            title_suffix = ""
            if metrics_map and idx in metrics_map:
                title_suffix = f" (BER: {metrics_map[idx]*100:.2f}%)"

            p1 = _add_title_bar(original_rgb, f"(a) Input image {idx}")
            p2 = _add_title_bar(pred_panel, f"(b) Predicted shadow{title_suffix}")
            p3 = _add_title_bar(gt_panel, "(c) Annotated shadow")
            p4 = _add_title_bar(cmp_panel, "(d) Prediction vs. annotation")

            top = np.concatenate([p1, p2], axis=1)
            bottom = np.concatenate([p3, p4], axis=1)
            preview = np.concatenate([top, bottom], axis=0)

            save_path = output_dir / f"rank_{rank:02d}_idx_{idx:05d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            log(f"  [Prediction Preview] Failed for idx={idx}: {exc}")

    log(f"  [Prediction Preview] Export finished for {len(indices)} images.")


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
    
    log(f"  Extraction complete for {desc}.")
    log_dataset_stats(all_labels, len(indices), desc)
    
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
    method = CONFIG.get('method', 'lookop')
    log(f"  Selected method: {method}")
    
    # Sample selection
    rng = np.random.default_rng(CONFIG['random_seed'])
    total_train = min(CONFIG['n_train_images'], len(train_ds))
    all_indices = rng.choice(len(train_ds), total_train, replace=False)
    holdout_count = min(CONFIG['holdout_images'], max(1, total_train // 2))
    holdout_indices = sorted(all_indices[:holdout_count].tolist())
    train_indices = sorted(all_indices[holdout_count:].tolist())
    
    log(f"  Training images: {len(train_indices)}")
    log(f"  Hold-out images: {len(holdout_indices)}")
    
    # Fast path for CNN region baseline (needs region patches)
    if method == 'cnn':
        log("Method is CNN baseline; extracting region patches.")
        slic = SuperpixelSegmenter(n_segments=CONFIG['n_segments'], compactness=CONFIG['compactness'])
        region_gen = MeanShiftRegionGenerator(bandwidth=CONFIG['region_bandwidth'])
        train_patches, train_patch_labels, _ = extract_region_patches(
            dataset=train_ds,
            indices=train_indices,
            slic=slic,
            region_gen=region_gen,
            context_ratio=CONFIG.get('cnn_context_ratio', 0.1),
            min_pixels=CONFIG.get('cnn_min_region_pixels', 20),
            desc="Train"
        )
        holdout_patches, holdout_patch_labels, holdout_region_stats = extract_region_patches(
            dataset=train_ds,
            indices=holdout_indices,
            slic=slic,
            region_gen=region_gen,
            context_ratio=CONFIG.get('cnn_context_ratio', 0.1),
            min_pixels=CONFIG.get('cnn_min_region_pixels', 20),
            desc="Holdout"
        )
        log_dataset_stats(np.array(train_patch_labels), len(train_indices), "CNN Train")
        log_dataset_stats(np.array(holdout_patch_labels), len(holdout_indices), "CNN Holdout")
        run_cnn_region_baseline(
            train_patches=train_patches,
            train_labels=train_patch_labels,
            holdout_patches=holdout_patches,
            holdout_labels=holdout_patch_labels,
            holdout_region_stats=holdout_region_stats,
            config=CONFIG,
            device=DEVICE,
            log_fn=log
        )
        log("Pipeline finished successfully (CNN region baseline).")
        return
    
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
                log_dataset_stats(train_labels, len(train_indices), "Train (Cached)")
                log_dataset_stats(holdout_labels, len(holdout_indices), "Holdout (Cached)")
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

    # 4. Model preparation (LooKOP or baselines)
    log("Step 4: Model preparation...")
    model_path = CONFIG['output_dir'] / f'sbu_model_{method}.pkl'
    baseline_params = None
    if method == 'lookop':
        log("  Initializing Paper-compliant Beam Search Optimizer...")
        optimizer = PaperBeamSearchOptimizer(
            n_iterations=CONFIG['n_iterations'],
            stagnation_threshold=CONFIG['stagnation_threshold'],
            gamma_lssvm=CONFIG['gamma_lssvm'],
            verbose=True
        )
        skip_optimization = False
        if CONFIG.get('load_existing_model', True) and model_path.exists():
            log(f"  Existing model found at {model_path}. Loading parameters...")
            try:
                with open(model_path, 'rb') as f:
                    saved_model = pickle.load(f)
                    if saved_model.get('method', 'lookop') == 'lookop':
                        optimizer.optimal_weights_ = saved_model['optimal_weights']
                        optimizer.optimal_sigmas_ = saved_model['optimal_sigmas']
                        optimal_weights = optimizer.optimal_weights_
                        optimal_sigmas = optimizer.optimal_sigmas_
                        min_loo_ber = saved_model.get('test_pixel_metrics', {}).get('ber', 0.0)
                        skip_optimization = True
                        log("  Parameters loaded successfully. Skipping Beam Search optimization.")
                    else:
                        log("  Saved model method mismatch. Proceeding with optimization.")
            except Exception as e:
                log(f"  Failed to load existing model: {e}. Proceeding with optimization.")

        if not skip_optimization:
            start_opt = time.time()
            # Subsample regions for optimization to avoid O(N^2) memory blow-up
            max_opt_samples = CONFIG.get('max_region_samples_opt', 15000)
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
        log(f"  Final/Best LOO BER: {min_loo_ber*100:.2f}%")
        log(f"  Optimal Weights: {optimal_weights}")
        log(f"  Optimal Sigmas: {optimal_sigmas}")
        final_kernel = optimizer.get_optimized_kernel()
    else:
        log(f"  Building baseline kernel for method={method}...")
        if CONFIG.get('load_existing_model', True) and model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    saved_model = pickle.load(f)
                    if saved_model.get('method', '') == method:
                        baseline_params = saved_model.get('baseline_params', None)
                        log("  Loaded baseline parameters from existing model.")
                    else:
                        log("  Saved model method mismatch. Rebuilding baseline parameters.")
            except Exception as e:
                log(f"  Failed to load baseline parameters: {e}. Rebuilding.")
        if baseline_params is not None:
            weights = baseline_params.get('weights')
            sigmas = baseline_params.get('sigmas')
            sigma_single = baseline_params.get('sigma_single')
            if method == 'unary_svm' and sigma_single is not None:
                CONFIG['unary_sigma'] = sigma_single
            final_kernel, baseline_params = build_baseline_kernel(
                method=method,
                train_features=train_features,
                config=CONFIG,
                fixed_weights=weights,
                fixed_sigmas=sigmas
            )
        else:
            final_kernel, baseline_params = build_baseline_kernel(
                method=method,
                train_features=train_features,
                config=CONFIG
            )
        optimal_weights = baseline_params.get('weights', {})
        optimal_sigmas = baseline_params.get('sigmas', baseline_params.get('sigma_single', {}))
        min_loo_ber = 0.0
        log("  Baseline kernel prepared.")

    # 5. Final Model Training & Evaluation
    log("Step 5: Training final LSSVM with selected parameters...")
    
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
    total_regions_processed = 0
    image_metrics = []  # To store (idx, ber) for sorting
    for idx in eval_iterator:
        image, gt_mask = train_ds[idx]
        if gt_mask is None:
            continue
            
        pred_mask, n_img_regions = _predict_mask_for_image(
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
        total_regions_processed += n_img_regions
        pred_bool = pred_mask.astype(bool)
        gt_bool = gt_mask.astype(bool)

        # Per-image stats for finding best predictions
        img_tp = float(np.logical_and(pred_bool, gt_bool).sum())
        img_fp = float(np.logical_and(pred_bool, ~gt_bool).sum())
        img_fn = float(np.logical_and(~pred_bool, gt_bool).sum())
        img_tn = float(np.logical_and(~pred_bool, ~gt_bool).sum())

        img_fpr = img_fp / (img_fp + img_tn) if (img_fp + img_tn) > 0 else 0.0
        img_fnr = img_fn / (img_fn + img_tp) if (img_fn + img_tp) > 0 else 0.0
        img_ber = (img_fpr + img_fnr) / 2.0
        image_metrics.append((idx, img_ber))

        tp += img_tp
        fp += img_fp
        fn += img_fn
        tn += img_tn
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

    log(f"=== Results Summary on Hold-out (Pixel-level) ===")
    log(f"  Images evaluated: {evaluated}")
    log(f"  Total regions processed: {total_regions_processed}")
    if evaluated > 0:
        log(f"  Avg regions per image: {total_regions_processed/evaluated:.1f}")
    log(f"  Hold-out BER: {pixel_metrics['ber']*100:.2f}%")
    log(f"  Hold-out Accuracy: {pixel_metrics['accuracy']*100:.2f}%")
    log(f"  Hold-out FPR: {pixel_metrics['fpr']*100:.2f}%")
    log(f"  Hold-out FNR: {pixel_metrics['fnr']*100:.2f}%")
    
    if CONFIG.get('save_test_mask_predictions', False):
        pred_vis_dir = CONFIG['output_dir'] / 'holdout_best_predictions'
        # Sort by BER ascending (best first)
        image_metrics.sort(key=lambda x: x[1])
        best_n = CONFIG.get('test_pred_best_n', 20)
        best_indices = [x[0] for x in image_metrics[:best_n]]
        metrics_map = {x[0]: x[1] for x in image_metrics}

        save_prediction_visualizations(
            dataset=train_ds,
            indices=best_indices,
            slic=slic,
            region_gen=region_gen,
            feat_extractor=feat_extractor,
            final_kernel=final_kernel,
            lssvm=lssvm,
            train_features=train_features,
            platt=platt,
            output_dir=pred_vis_dir,
            apply_mrf=CONFIG.get('use_mrf', False),
            metrics_map=metrics_map
        )

    # Save Model
    save_path = CONFIG['output_dir'] / f'sbu_model_{method}.pkl'
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
            'test_pixel_metrics': pixel_metrics,
            'method': method,
            'baseline_params': baseline_params
        }, f)
    
    log("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
