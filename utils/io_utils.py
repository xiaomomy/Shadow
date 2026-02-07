"""
Input/Output utility functions.

Author: [Your Name]
Date: 2026
"""

import os
import numpy as np
from PIL import Image
import pickle
from typing import Dict, Any, Optional, Tuple


def create_output_dirs(base_dir: str) -> Dict[str, str]:
    """
    Create output directory structure.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary mapping purpose to directory path
    """
    dirs = {
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'features': os.path.join(base_dir, 'features'),
        'segmentation': os.path.join(base_dir, 'segmentation'),
        'models': os.path.join(base_dir, 'models'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def load_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        target_size: Optional (width, height) to resize image
        
    Returns:
        RGB image as numpy array (H, W, 3), values in [0, 255]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load with PIL
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if requested
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    return np.array(img)


def save_image(image: np.ndarray, save_path: str) -> None:
    """
    Save an image to disk.
    
    Args:
        image: Image array (H, W, 3) or (H, W)
        save_path: Output path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Save with PIL
    Image.fromarray(image).save(save_path)


def save_results(
    results: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save preprocessing results to disk.
    
    Args:
        results: Dictionary containing preprocessing outputs
        save_path: Output path (.pkl)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


def load_results(load_path: str) -> Dict[str, Any]:
    """
    Load preprocessing results from disk.
    
    Args:
        load_path: Path to saved results (.pkl)
        
    Returns:
        Dictionary containing preprocessing outputs
    """
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def save_label_map(
    labels: np.ndarray,
    save_path: str
) -> None:
    """
    Save a label map (superpixels or regions) to disk.
    
    Args:
        labels: Integer label array (H, W)
        save_path: Output path (.npy)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, labels)


def load_label_map(load_path: str) -> np.ndarray:
    """
    Load a label map from disk.
    
    Args:
        load_path: Path to saved labels (.npy)
        
    Returns:
        Integer label array (H, W)
    """
    return np.load(load_path)

