"""
Deep debug - trace the root cause of BER=50%
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
from models.platt_scaling import PlattScaler, balanced_error_rate
from models.distances import emd_1d_matrix, chi_square_distance_matrix
import cv2

print("="*60)
print("  Deep Debug")
print("="*60)

# Load data
dataset = SBUDataset('data/sbu/SBU-shadow', split='train')

n_images = 20
segmenter = SuperpixelSegmenter(n_segments=200, compactness=20)
region_generator = RegionGenerator(bandwidth=None, quantile=0.3, use_spatial=False)

# Texton
print("\n[1] Build texton...")
texton_extractor = TextonFeatureExtractor(n_textons=128)
gray_images = [cv2.cvtColor(dataset[i][0], cv2.COLOR_RGB2GRAY) for i in range(3)]
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
n = len(y)
D_L = emd_1d_matrix(X['L'], X['L'])
D_a = emd_1d_matrix(X['a'], X['a'])
D_b = emd_1d_matrix(X['b'], X['b'])
D_t = chi_square_distance_matrix(X['t'], X['t'])

# Mean excluding diagonal
mask = ~np.eye(n, dtype=bool)
mean_L = np.mean(D_L[mask])
mean_a = np.mean(D_a[mask])
mean_b = np.mean(D_b[mask])
mean_t = np.mean(D_t[mask])

print(f"  D_L: mean={mean_L:.4f}")
print(f"  D_a: mean={mean_a:.4f}")
print(f"  D_b: mean={mean_b:.4f}")
print(f"  D_t: mean={mean_t:.4f}")

# Build kernel matrix using per-channel independent sigma
print("\n[4] Build kernel matrix (per-channel independent sigma)...")
mult = 1.0  # sigma = mult * mean_distance

K_L = np.exp(-D_L / (mult * mean_L + 1e-8))
K_a = np.exp(-D_a / (mult * mean_a + 1e-8))
K_b = np.exp(-D_b / (mult * mean_b + 1e-8))
K_t = np.exp(-D_t / (mult * mean_t + 1e-8))

print(f"  K_L: mean={K_L.mean():.4f}, std={K_L.std():.4f}")
print(f"  K_a: mean={K_a.mean():.4f}, std={K_a.std():.4f}")
print(f"  K_b: mean={K_b.mean():.4f}, std={K_b.std():.4f}")
print(f"  K_t: mean={K_t.mean():.4f}, std={K_t.std():.4f}")

K = 0.25 * (K_L + K_a + K_b + K_t)
print(f"  K (combined): mean={K.mean():.4f}, std={K.std():.4f}, range=[{K.min():.4f},{K.max():.4f}]")

# Train LSSVM
print("\n[5] Train LSSVM...")
y_lssvm = 2 * y - 1  # convert to -1/+1
lssvm = LSSVM(gamma=1.0)
lssvm.fit(np.zeros((n, 1)), y_lssvm, K=K)

print(f"  alpha: mean={lssvm.alpha_.mean():.4f}, std={lssvm.alpha_.std():.4f}")
print(f"  bias: {lssvm.bias_:.4f}")

# Decision values
print("\n[6] Check decision values...")
dv = lssvm.decision_function(np.zeros((n, 1)), K=K)
print(f"  Decision values: mean={dv.mean():.4f}, std={dv.std():.4f}, range=[{dv.min():.4f},{dv.max():.4f}]")
print(f"  >0: {np.sum(dv > 0)}, <0: {np.sum(dv < 0)}")

# Direct prediction (without Platt)
pred_direct = (dv > 0).astype(int)
ber_direct = balanced_error_rate(y, pred_direct)
print(f"\n[7] Direct prediction (decision > 0):")
print(f"  BER={100*ber_direct:.1f}%")

# Platt scaling
print("\n[8] Platt scaling...")
platt = PlattScaler()
platt.fit(dv, y)
print(f"  Platt a={platt.a_:.4f}, b={platt.b_:.4f}")

probs = platt.predict_proba(dv)
print(f"  Probs: mean={probs.mean():.4f}, range=[{probs.min():.4f},{probs.max():.4f}]")

pred_platt = (probs > 0.5).astype(int)
ber_platt = balanced_error_rate(y, pred_platt)
print(f"  Platt BER={100*ber_platt:.1f}%")

# Problem location
print("\n" + "="*60)
print("  Diagnostic conclusion")
print("="*60)

if ber_direct < 0.5 and ber_platt >= 0.5:
    print("\n[Issue] Platt scaling causing prediction inversion!")
    print("  Direct prediction is effective, but Platt probability inverted the results.")
    print("  Suggestion: Check Platt scaling implementation or use decision > 0 directly.")
elif ber_direct >= 0.5:
    print("\n[Issue] LSSVM itself cannot distinguish samples!")
    print("  Decision value distribution has no discriminative power.")
else:
    print("\nFix successful! BER < 50%")
