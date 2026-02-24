"""
Shadow detection dataset loader
Supports SBU dataset

Author: Shadow Detection Project
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

# Data directory
DATA_DIR = Path(__file__).parent
SBU_DOWNLOAD_URL = "http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip"


class SBUDataset:
    """
    SBU-Shadow Dataset

    Directory structure:
    SBU-shadow/
        SBUTrain4KRecoveredSmall/
            ShadowImages/   (training images)
            ShadowMasks/    (training masks)
        SBU-Test/
            ShadowImages/   (test images)
            ShadowMasks/    (test masks)
    """

    def __init__(self, root_dir: str = None, split: str = 'train', download: bool = True):
        """
        Initialize SBU dataset.
        
        Args:
            root_dir (str, optional): Root directory of the dataset.
            split (str): 'train' or 'test'.
            download (bool): Whether to download the dataset if not found.
        """
        if root_dir is None:
            root_dir = DATA_DIR / 'sbu' / 'SBU-shadow'
        self.root_dir = Path(root_dir)
        self.split = split

        if download:
            self._check_and_download()

        self.image_paths = []
        self.mask_paths = []
        self._load_paths()

    def _check_and_download(self):
        """
        Check if the dataset exists locally, download and extract if not.
        """
        # Check for training image directory as an indicator
        check_dir = self.root_dir / 'SBUTrain4KRecoveredSmall' / 'ShadowImages'
        if not check_dir.exists():
            print(f"Dataset not found at {self.root_dir}. Attempting to download...")
            
            # Create parent directory (e.g., data/sbu/)
            os.makedirs(self.root_dir.parent, exist_ok=True)
            zip_path = self.root_dir.parent / 'SBU-shadow.zip'

            try:
                print(f"Downloading dataset from {SBU_DOWNLOAD_URL}...")
                urllib.request.urlretrieve(SBU_DOWNLOAD_URL, zip_path)
                
                print(f"Extracting dataset to {self.root_dir.parent}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir.parent)
                
                # Cleanup the zip file
                os.remove(zip_path)
                print("Dataset download and extraction successful.")
            except Exception as e:
                if zip_path.exists():
                    os.remove(zip_path)
                print(f"Failed to download/extract dataset: {e}")
                raise RuntimeError(f"Could not prepare SBU dataset: {e}")

    def _load_paths(self):
        """Load SBU dataset image and mask paths."""
        if self.split == 'train':
            # Training set paths
            image_dir = self.root_dir / 'SBUTrain4KRecoveredSmall' / 'ShadowImages'
            mask_dir = self.root_dir / 'SBUTrain4KRecoveredSmall' / 'ShadowMasks'
        else:
            # Test set paths
            image_dir = self.root_dir / 'SBU-Test' / 'ShadowImages'
            mask_dir = self.root_dir / 'SBU-Test' / 'ShadowMasks'

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Get all images with common extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_paths.extend(sorted(image_dir.glob(f'*{ext}')))

        # Match corresponding masks by filename stem
        for img_path in self.image_paths:
            # Try matching mask with .png or .jpg extensions
            mask_path = mask_dir / (img_path.stem + '.png')
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + '.jpg')
            self.mask_paths.append(mask_path)

        print(f"[SBU {self.split}] Successfully loaded {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Returns (image, mask)"""
        # Load image (BGR -> RGB)
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = None
        if self.mask_paths[idx].exists():
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        
        return img, mask


def get_dataset(name: str, split: str = 'train'):
    """Get dataset by name"""
    if name.lower() == 'sbu':
        return SBUDataset(split=split)
    raise ValueError(f"Unknown dataset: {name}")
