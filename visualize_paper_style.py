"""
Paper-style visualization script for shadow detection preprocessing.

This script creates a figure similar to the one in the paper (Vicente et al., ICCV 2015):
1. Original image
2. Superpixel segmentation (SLIC)
3. Region segmentation (Mean-shift clustering [4])
4. Ground truth shadow mask

Reference:
    Paper Section 3: "apply Mean-shift clustering [4] and merge superpixels
    in the same cluster into a larger region."
    [4] Comaniciu & Meer, "Mean Shift: A Robust Approach Toward Feature
        Space Analysis", TPAMI 2002

Usage:
    # Visualize dataset sample
    python visualize_paper_style.py --dataset sbu --index 0

    # Visualize specific image
    python visualize_paper_style.py --image <path> --mask <path>

    # Adjust Mean-shift bandwidth (larger = fewer regions)
    python visualize_paper_style.py --dataset sbu --index 0 --bandwidth 0.5

    # Auto-estimate bandwidth with different quantile
    python visualize_paper_style.py --dataset sbu --index 0 --quantile 0.1

Author: Shadow Detection Project
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import SuperpixelSegmenter, RegionGenerator
from data.dataset_loader import SBUDataset
from config import SUPERPIXEL_CONFIG, REGION_CONFIG


def extract_region_labels_from_mask(mask: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    """
    Extract region-level shadow labels from pixel-level mask.

    A region is labeled as shadow (+1) if more than 50% of its pixels
    are in shadow in the ground truth mask.

    Reference:
        pipeline.py:_extract_region_labels_from_mask

    Args:
        mask: Binary shadow mask (H, W), shadow=1, non-shadow=0
        region_labels: Region label map (H, W)

    Returns:
        region_shadow_labels: Array of shape (n_regions,) with values:
            +1 = shadow region, -1 = non-shadow region
    """
    n_regions = region_labels.max() + 1
    labels = np.zeros(n_regions, dtype=np.int32)

    for i in range(n_regions):
        region_mask = region_labels == i
        if region_mask.any():
            shadow_ratio = mask[region_mask].mean()
            labels[i] = 1 if shadow_ratio > 0.5 else -1

    return labels


def load_image_and_mask(image_path: str, mask_path: str = None):
    """
    Load image and optional mask.

    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file (optional)

    Returns:
        image: RGB image array
        mask: Binary mask array (None if mask_path not provided)
    """
    # Load image (BGR -> RGB)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load mask if provided
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

    return image, mask


def create_paper_style_visualization(
    image: np.ndarray,
    mask: np.ndarray = None,
    superpixel_config: dict = None,
    region_config: dict = None,
    save_path: str = None,
    show: bool = True,
    dpi: int = 150
):
    """
    Create a paper-style visualization showing preprocessing steps.

    Layout (similar to paper figure):
    - Row 1: Original Image | Superpixel Segmentation
    - Row 2: Region Segmentation | Ground Truth Mask (if available)

    Args:
        image: RGB image array
        mask: Binary shadow mask (optional)
        superpixel_config: Configuration for superpixel segmentation
        region_config: Configuration for region generation
        save_path: Path to save the figure
        show: Whether to display the figure
        dpi: DPI for saved figure

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Use default configs if not provided
    if superpixel_config is None:
        superpixel_config = SUPERPIXEL_CONFIG.copy()
    if region_config is None:
        region_config = REGION_CONFIG.copy()

    print("=" * 60)
    print("Generating Paper-Style Visualization")
    print("=" * 60)

    # Step 1: Superpixel Segmentation
    print("\n[1/3] Running superpixel segmentation...")
    segmenter = SuperpixelSegmenter(**superpixel_config)
    sp_labels = segmenter.segment(image)
    n_superpixels = segmenter.n_superpixels
    print(f"      Generated {n_superpixels} superpixels")

    # Step 2: Region Generation (Mean-shift clustering - Paper Section 3)
    print("\n[2/3] Running region generation (Mean-shift clustering [4])...")
    print("      Reference: Comaniciu & Meer, TPAMI 2002")
    region_generator = RegionGenerator(**region_config)
    region_labels = region_generator.generate_regions(image, sp_labels)
    n_regions = region_generator.n_regions
    bandwidth = region_generator.estimated_bandwidth
    print(f"      Generated {n_regions} regions")
    if region_config['bandwidth'] is None:
        print(f"      Auto-estimated bandwidth: {bandwidth:.4f} (quantile={region_config['quantile']})")
    else:
        print(f"      Using specified bandwidth: {bandwidth:.4f}")

    # Step 3: Create visualization
    print("\n[3/3] Creating visualization...")

    # Create figure with subplots
    if mask is not None:
        # 5 subplots: original, superpixels, regions, pixel mask, region labels
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)
        axes = [
            fig.add_subplot(gs[0, 0]),  # (a) Original
            fig.add_subplot(gs[0, 1]),  # (b) Superpixels
            fig.add_subplot(gs[0, 2]),  # (c) Regions
            fig.add_subplot(gs[1, 0]),  # (d) Pixel-level Mask
            fig.add_subplot(gs[1, 1]),  # (e) Region-level Labels
        ]
        # Hide the unused subplot
        ax_unused = fig.add_subplot(gs[1, 2])
        ax_unused.axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten()

    # Normalize image for display
    if image.max() > 1.0:
        img_display = image.astype(np.float64) / 255.0
    else:
        img_display = image.copy()

    # Plot 1: Original Image
    ax = axes[0]
    ax.imshow(img_display)
    ax.set_title('(a) Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot 2: Superpixel Segmentation
    ax = axes[1]
    from skimage.segmentation import mark_boundaries
    sp_vis = mark_boundaries(img_display, sp_labels, color=(1, 1, 0), mode='outer')
    ax.imshow(sp_vis)
    ax.set_title(f'(b) Superpixel Segmentation (n={n_superpixels})',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot 3: Region Segmentation
    ax = axes[2]
    region_vis = mark_boundaries(img_display, region_labels, color=(0, 1, 0), mode='outer')
    ax.imshow(region_vis)
    ax.set_title(f'(c) Region Segmentation (n={n_regions})',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot 4: Pixel-level Ground Truth Mask (if available)
    if mask is not None:
        ax = axes[3]

        # Create overlay visualization with pixel-level mask
        mask_overlay = img_display.copy()

        # Red color for shadow regions (semi-transparent)
        shadow_color = np.array([1.0, 0.2, 0.2])  # Red
        alpha = 0.4

        for c in range(3):
            mask_overlay[:, :, c] = np.where(
                mask == 1,
                (1 - alpha) * img_display[:, :, c] + alpha * shadow_color[c],
                img_display[:, :, c]
            )

        ax.imshow(mask_overlay)
        ax.set_title('(d) Pixel-Level Shadow Mask', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.4, edgecolor='red', label='Shadow')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Plot 5: Region-level Ground Truth (if available)
    if mask is not None:
        ax = axes[4]

        # Extract region-level shadow labels
        region_shadow_labels = extract_region_labels_from_mask(mask, region_labels)

        # Create region-level shadow visualization
        region_shadow_map = np.zeros_like(img_display)

        # Color for shadow regions (red tint) and non-shadow (green tint)
        shadow_color = np.array([0.85, 0.25, 0.25])    # Red for shadow
        non_shadow_color = np.array([0.25, 0.7, 0.25])  # Green for non-shadow
        alpha = 0.4

        for region_id in range(len(region_shadow_labels)):
            region_mask = region_labels == region_id
            is_shadow = region_shadow_labels[region_id] == 1

            # Color based on region label
            color = shadow_color if is_shadow else non_shadow_color

            for c in range(3):
                region_shadow_map[:, :, c] = np.where(
                    region_mask,
                    (1 - alpha) * img_display[:, :, c] + alpha * color[c],
                    region_shadow_map[:, :, c]
                )

        # Mark region boundaries
        from skimage.segmentation import find_boundaries
        boundaries = find_boundaries(region_labels, mode='outer')
        for c in range(3):
            region_shadow_map[:, :, c] = np.where(
                boundaries,
                0,  # Black boundaries
                region_shadow_map[:, :, c]
            )

        ax.imshow(region_shadow_map)
        ax.set_title('(e) Region-Level Shadow Labels', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, edgecolor='red', label='Shadow Region'),
            Patch(facecolor='green', alpha=0.5, edgecolor='green', label='Non-Shadow Region')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        # Add statistics
        n_shadow_regions = np.sum(region_shadow_labels == 1)
        n_non_shadow_regions = np.sum(region_shadow_labels == -1)
        ax.text(0.02, 0.02, f'Shadow: {n_shadow_regions}\nNon-shadow: {n_non_shadow_regions}',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                    exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"\nFigure saved to: {save_path}")

    print("\n" + "=" * 60)
    print("Visualization Summary:")
    print(f"  Image size: {image.shape}")
    print(f"  Superpixels: {n_superpixels}")
    print(f"  Regions: {n_regions}")
    if mask is not None:
        shadow_pixels = np.sum(mask)
        total_pixels = mask.size
        print(f"  Shadow ratio: {shadow_pixels/total_pixels*100:.1f}%")
    print("=" * 60)

    if show:
        plt.show()

    return fig, axes


def visualize_from_dataset(dataset_name: str = 'sbu', index: int = 0,
                           split: str = 'train', save_dir: str = 'output',
                           superpixel_config: dict = None,
                           region_config: dict = None,
                           show: bool = True):
    """
    Visualize a sample from a dataset.

    Args:
        dataset_name: Name of the dataset ('sbu')
        index: Index of the sample to visualize
        split: 'train' or 'test'
        save_dir: Directory to save the output
        superpixel_config: Configuration for superpixel segmentation
        region_config: Configuration for region generation
        show: Whether to display the figure
    """
    print(f"Loading {dataset_name} dataset ({split} split)...")

    if dataset_name.lower() == 'sbu':
        dataset = SBUDataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if index >= len(dataset):
        raise ValueError(f"Index {index} out of range (dataset has {len(dataset)} samples)")

    # Load image and mask
    image, mask = dataset[index]

    # Get image name for saving
    img_name = os.path.splitext(os.path.basename(dataset.image_paths[index]))[0]
    save_path = os.path.join(save_dir, f'paper_style_{img_name}.png')

    print(f"Visualizing sample {index}: {img_name}")

    # Create visualization with provided configs
    create_paper_style_visualization(
        image=image,
        mask=mask,
        superpixel_config=superpixel_config,
        region_config=region_config,
        save_path=save_path,
        show=show
    )

    return image, mask


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create paper-style visualization of shadow detection preprocessing'
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to input image')
    input_group.add_argument('--dataset', type=str, choices=['sbu'],
                             help='Use dataset sample')

    # Additional arguments
    parser.add_argument('--mask', type=str, help='Path to ground truth mask')
    parser.add_argument('--index', type=int, default=0,
                        help='Sample index (for dataset mode)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='Dataset split (for dataset mode)')
    parser.add_argument('--output', type=str, default='output/paper_style_vis.png',
                        help='Output path for the visualization')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figure')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the figure')

    # Superpixel parameters
    parser.add_argument('--n-superpixels', type=int, default=500,
                        help='Number of superpixels')
    parser.add_argument('--compactness', type=float, default=10.0,
                        help='SLIC compactness parameter')

    # Region parameters (Mean-shift clustering)
    parser.add_argument('--bandwidth', type=float, default=None,
                        help='Mean-shift bandwidth (None=auto-estimate). '
                             'Larger bandwidth = fewer regions. '
                             'Paper uses auto-estimation.')
    parser.add_argument('--quantile', type=float, default=0.2,
                        help='Quantile for bandwidth estimation when bandwidth=None. '
                             'Smaller quantile = smaller bandwidth = more regions. '
                             'Default 0.2 follows config.py.')

    args = parser.parse_args()

    # Build configurations
    superpixel_config = {
        'n_segments': args.n_superpixels,
        'compactness': args.compactness,
        'sigma': 1.0,
        'convert2lab': True,
        'enforce_connectivity': True,
    }

    # Mean-shift clustering configuration (Paper Section 3)
    region_config = {
        'bandwidth': args.bandwidth,  # None = auto-estimate using quantile
        'quantile': args.quantile,
        'use_spatial': False,
    }

    try:
        if args.dataset:
            # Dataset mode
            image, mask = visualize_from_dataset(
                dataset_name=args.dataset,
                index=args.index,
                split=args.split,
                save_dir=os.path.dirname(args.output) or 'output',
                superpixel_config=superpixel_config,
                region_config=region_config,
                show=not args.no_show
            )
        else:
            # Single image mode
            image, mask = load_image_and_mask(args.image, args.mask)
            create_paper_style_visualization(
                image=image,
                mask=mask,
                superpixel_config=superpixel_config,
                region_config=region_config,
                save_path=args.output,
                show=not args.no_show,
                dpi=args.dpi
            )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
