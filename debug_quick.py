"""
Quick validation of sigma fix
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
from models.lssvm import LSSVM
from models.platt_scaling import PlattScaler
from models.distances import emd_1d_matrix, chi_square_distance_matrix
import cv2

print("="*60)
print("  Verify sigma fix")
print("="*60)

# Load data
dataset = SBUDataset('data/sbu/SBU-shadow', split='train')

n_images = 15  # Use more images
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
all_features = {'L': [], 'a': [], 'b': [], 't': []}
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
        all_features['t'].append(features_dict['t'][r])
        all_labels.append(label)

X = {k: np.array(v) for k, v in all_features.items()}
y = np.array(all_labels)

print(f"  Number of samples: {len(y)}, Shadow: {np.sum(y==1)} ({100*np.mean(y):.1f}%)")

# Compute distance matrix
print("\n[3] Compute distance matrix...")
D_L = emd_1d_matrix(X['L'], X['L'])
D_a = emd_1d_matrix(X['a'], X['a'])
D_b = emd_1d_matrix(X['b'], X['b'])
D_t = chi_square_distance_matrix(X['t'], X['t'])

print(f"  D_L: mean={D_L.mean():.2f}, max={D_L.max():.2f}")
print(f"  D_a: mean={D_a.mean():.2f}, max={D_a.max():.2f}")
print(f"  D_b: mean={D_b.mean():.2f}, max={D_b.max():.2f}")
print(f"  D_t: mean={D_t.mean():.2f}, max={D_t.max():.2f}")

# Test different sigma
print("\n[4] Test different sigma...")
test_sigmas = [1.0, 2.0, 5.0, 10.0, 20.0]

for sigma in test_sigmas:
    # Build kernel matrix
    K_L = np.exp(-D_L / sigma)
    K_a = np.exp(-D_a / sigma)
    K_b = np.exp(-D_b / sigma)
    K_t = np.exp(-D_t / sigma)  # use same sigma for texton for simplified testing
    K = 0.25 * (K_L + K_a + K_b + K_t)
    
    # Train LSSVM
    y_train = 2 * y - 1
    lssvm = LSSVM(gamma=1.0)
    lssvm.fit(np.zeros((len(y), 1)), y_train, K=K)
    
    # Decision values
    dv = lssvm.decision_function(np.zeros((len(y), 1)), K=K)
    
    # Direct prediction
    pred = (dv > 0).astype(int)
    
    # Compute BER
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    tn = np.sum((pred == 0) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    ber = (fpr + fnr) / 2
    acc = np.mean(pred == y)
    
    print(f"  sigma={sigma:5.1f}: K_mean={K.mean():.3f}, K_std={K.std():.3f}, "
          f"DV=[{dv.min():.2f},{dv.max():.2f}], "
          f"BER={100*ber:.1f}%, Acc={100*acc:.1f}%")

print("\n" + "="*60)
print("If BER < 50%, it means the fix was successful!")
print("="*60)
