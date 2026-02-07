"""
Verify if per-channel independent sigma fix is effective
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset_loader import SBUDataset
from preprocessing.superpixel import SuperpixelSegmenter
from preprocessing.region import RegionGenerator
from preprocessing.features import PaperCompliantFeatureExtractor
from preprocessing.texton import TextonFeatureExtractor
from models.platt_scaling import balanced_error_rate
import cv2

# Import the fixed class
from train_quick import MultiKernelLSSVM

print("="*60)
print("  Verify per-channel independent sigma fix")
print("="*60)

# Load data
dataset = SBUDataset('data/sbu/SBU-shadow', split='train')

n_images = 20
segmenter = SuperpixelSegmenter(n_segments=200, compactness=20)
region_generator = RegionGenerator(bandwidth=None, quantile=0.3, use_spatial=False)

# Texton
print("\n[1] Build texton dictionary...")
texton_extractor = TextonFeatureExtractor(n_textons=128)
gray_images = [cv2.cvtColor(dataset[i][0], cv2.COLOR_RGB2GRAY) for i in range(3) if dataset[i][0] is not None]
texton_extractor.build_dictionary(gray_images, verbose=False)

feature_extractor = PaperCompliantFeatureExtractor(texton_extractor=texton_extractor)

# Extract features
print("\n[2] Extract features...")
all_features = {'L': [], 'a': [], 'b': [], 'texture': []}
all_labels = []

for idx in range(n_images):
    img, mask = dataset[idx]
    if img is None or mask is None:
        continue
    
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    sp_labels = segmenter.segment(img)
    region_labels = region_generator.generate_regions(img, sp_labels)
    features_dict = feature_extractor.extract_features_by_channel(img, region_labels)
    
    n_regions = len(np.unique(region_labels))
    
    for r in range(n_regions):
        region_mask = (region_labels == r)
        shadow_pixels = np.sum(mask[region_mask] > 0)
        total_pixels = np.sum(region_mask)
        label = 1 if shadow_pixels > total_pixels * 0.5 else 0
        
        all_features['L'].append(features_dict['L'][r])
        all_features['a'].append(features_dict['a'][r])
        all_features['b'].append(features_dict['b'][r])
        all_features['texture'].append(features_dict['t'][r])  # Note: feature key conversion
        all_labels.append(label)

X = {k: np.array(v) for k, v in all_features.items()}
y = np.array(all_labels)

print(f"  Number of samples: {len(y)}, Shadow: {np.sum(y==1)} ({100*np.mean(y):.1f}%)")

# Test different sigma_multiplier
print("\n[3] Test different sigma_multiplier...")

for mult in [0.25, 0.5, 1.0, 2.0, 4.0]:
    print(f"\n--- sigma_multiplier={mult} ---")
    
    classifier = MultiKernelLSSVM(gamma=1.0, sigma_multiplier=mult)
    classifier.fit(X, y)
    
    # Prediction
    preds = classifier.predict(X)
    ber = balanced_error_rate(y, preds)
    acc = np.mean(preds == y)
    
    shadow_mask = y == 1
    non_shadow_mask = y == 0
    fpr = np.mean(preds[non_shadow_mask] == 1) if np.sum(non_shadow_mask) > 0 else 0
    fnr = np.mean(preds[shadow_mask] == 0) if np.sum(shadow_mask) > 0 else 0
    
    print(f"  BER={100*ber:.1f}%, Acc={100*acc:.1f}%, FPR={100*fpr:.1f}%, FNR={100*fnr:.1f}%")

print("\n" + "="*60)
print("If BER < 50%, it means the fix was successful!")
print("="*60)
