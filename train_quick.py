"""
Quick training script for shadow detection
Based on "Leave-One-Out Kernel Optimization for Shadow Detection" (Vicente et al., ICCV 2015)

Quick validation version: using fewer samples for preliminary training and testing
Estimated training time: 10-20 minutes (GPU) / 20-30 minutes (CPU)

Training flow:
1. Data loading and splitting
2. Texton dictionary training
3. Region feature extraction
4. LSSVM classifier training (GPU acceleration)
5. Validation set evaluation

Run command: python train_quick.py

Author: Shadow Detection Project
"""

import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# GPU Support (PyTorch)
import torch
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')
if USE_GPU:
    print(f"[GPU] Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[CPU] CUDA not available, using CPU")

# Add project root to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# Import project modules
from data.dataset_loader import get_dataset
from preprocessing.superpixel import SuperpixelSegmenter
from preprocessing.region import MeanShiftRegionGenerator
from preprocessing.features import PaperCompliantFeatureExtractor
from preprocessing.texton import TextonFeatureExtractor
from models.lssvm import LSSVM
from models.platt_scaling import PlattScaler, balanced_error_rate
from models.distances import emd_1d_matrix, chi_square_distance_matrix


# =============================================================================
# Configuration Parameters - Quick Validation Version
# =============================================================================

QUICK_CONFIG = {
    # Data configuration
    'n_train_images': 500,       # Number of training images
    'n_val_images': 100,         # Number of validation images
    'random_seed': 42,
    
    # Superpixel configuration
    'n_segments': 150,          # Number of superpixels per image
    'compactness': 20,
    
    # Region configuration  
    'region_bandwidth': 0.3,
    
    # Texton configuration
    'n_textons': 128,            # number of textons (paper: 128, quick version: 64)
    'texton_train_images': 30,
    
    # LSSVM configuration
    'gamma': 1.0,
    # sigma_multiplier: sigma = multiplier * mean_distance
    # Paper recommendation: {1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8}
    'sigma_multipliers': [0.25, 0.5, 1.0, 2.0, 4.0],
    
    # Output configuration
    'output_dir': PROJECT_DIR / 'output' / 'quick_train',
    'save_model': True,
}


# =============================================================================
# Utility Functions
# =============================================================================

def print_section(title):
    """Print section title"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def get_region_label(mask: np.ndarray, region_mask: np.ndarray) -> int:
    """Get region label (based on majority vote)"""
    if mask is None:
        return 0
    region_pixels = mask[region_mask]
    if len(region_pixels) == 0:
        return 0
    shadow_ratio = np.mean(region_pixels > 0)
    return 1 if shadow_ratio > 0.5 else 0


class Timer:
    """Timer class for benchmarking"""
    def __init__(self):
        self.start_time = None
        self.records = {}
        self.current_name = None
    
    def start(self, name):
        self.start_time = time.time()
        self.current_name = name
        print(f"\n[Timer] Starting: {name}")
    
    def stop(self):
        elapsed = time.time() - self.start_time
        self.records[self.current_name] = elapsed
        print(f"[Timer] {self.current_name}: {elapsed:.1f}s")
        return elapsed
    
    def summary(self):
        print("\n[Timing Summary]")
        total = sum(self.records.values())
        for name, t in self.records.items():
            print(f"  {name}: {t:.1f}s ({t/total*100:.1f}%)")
        print(f"  Total: {total:.1f}s ({total/60:.1f}min)")


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_image_features(
    image: np.ndarray,
    mask: np.ndarray,
    slic: SuperpixelSegmenter,
    region_gen: MeanShiftRegionGenerator,
    feat_extractor: PaperCompliantFeatureExtractor
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """
    Extract all region features from a single image
    
    Returns:
        features_dict: {'L': (n, 21), 'a': (n, 21), 'b': (n, 21), 'texture': (n, n_textons)}
        labels: Region label list
    """
    # 1. Superpixel segmentation
    superpixel_labels = slic.segment(image)
    
    # 2. Region generation
    region_labels = region_gen.generate_regions(image, superpixel_labels)
    n_regions = int(region_labels.max()) + 1
    
    # 3. Extract all region features
    all_features = feat_extractor.extract_features_by_channel(image, region_labels, use_gpu=USE_GPU)
    
    # 4. Filter small regions and get labels
    valid_indices = []
    labels = []
    
    for region_id in range(n_regions):
        region_mask = (region_labels == region_id)
        if np.sum(region_mask) < 10:
            continue
        valid_indices.append(region_id)
        labels.append(get_region_label(mask, region_mask))
    
    # 5. Keep only features of valid regions
    features_dict = {
        'L': all_features['L'][valid_indices],
        'a': all_features['a'][valid_indices],
        'b': all_features['b'][valid_indices],
        'texture': all_features['t'][valid_indices]
    }
    
    return features_dict, labels


def extract_dataset_features(
    dataset,
    indices: List[int],
    slic: SuperpixelSegmenter,
    region_gen: MeanShiftRegionGenerator,
    feat_extractor: PaperCompliantFeatureExtractor,
    desc: str = "Extracting"
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Extract all features of the dataset"""
    all_features = {'L': [], 'a': [], 'b': [], 'texture': []}
    all_labels = []
    
    n_images = len(indices)
    print(f"\n[{desc}] Processing {n_images} images...")
    
    for i, idx in enumerate(indices):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Image {i+1}/{n_images}...")
        
        image, mask = dataset[idx]
        
        try:
            features, labels = extract_image_features(
                image, mask, slic, region_gen, feat_extractor
            )
            
            for key in all_features:
                all_features[key].append(features[key])
            all_labels.extend(labels)
            
        except Exception as e:
            print(f"  [Warning] Image {idx} failed: {e}")
            continue
    
    # Concatenate all features
    for key in all_features:
        all_features[key] = np.vstack(all_features[key]) if all_features[key] else np.array([])
    
    all_labels = np.array(all_labels)
    
    print(f"  Total regions: {len(all_labels)}")
    if len(all_labels) > 0:
        print(f"  Shadow: {np.sum(all_labels == 1)} ({np.mean(all_labels)*100:.1f}%)")
        print(f"  Non-shadow: {np.sum(all_labels == 0)} ({np.mean(all_labels == 0)*100:.1f}%)")
    
    return all_features, all_labels


# =============================================================================
# Classifier
# =============================================================================

class MultiKernelLSSVM:
    """Multi-kernel LSSVM classifier (supports GPU acceleration)"""
    
    def __init__(self, gamma: float = 1.0, sigma_multiplier: float = 1.0):
        """Initialization
        
        Args:
            gamma: LSSVM regularization parameter
            sigma_multiplier: sigma multiplier relative to mean distance
                             sigma_l = multiplier * mean(D_l)
        """
        self.gamma = gamma
        self.sigma_multiplier = sigma_multiplier
        self.lssvm = None
        self.platt = None
        self.train_features = None
        self._train_K = None
        self._channel_sigmas = {}  # sigma for each channel
    
    def _emd_1d_gpu(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """GPU accelerated EMD distance calculation (1D histogram)"""
        # Cumulative Distribution Function
        cdf1 = torch.cumsum(X1, dim=1)
        cdf2 = torch.cumsum(X2, dim=1)
        # EMD = sum of absolute differences of CDFs
        # Broadcast calculation of distances between all pairs
        D = torch.sum(torch.abs(cdf1.unsqueeze(1) - cdf2.unsqueeze(0)), dim=2)
        return D
    
    def _chi_square_gpu(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """GPU accelerated Chi-square distance calculation"""
        # chi2(x,y) = sum((x-y)^2 / (x+y+eps))
        X1_exp = X1.unsqueeze(1)  # (n1, 1, d)
        X2_exp = X2.unsqueeze(0)  # (1, n2, d)
        diff = X1_exp - X2_exp
        sum_val = X1_exp + X2_exp + 1e-10
        D = torch.sum(diff ** 2 / sum_val, dim=2)
        return D
        
    def _compute_kernel(self, X1: Dict[str, np.ndarray], X2: Dict[str, np.ndarray] = None, 
                        compute_sigmas: bool = False):
        """Compute combined kernel matrix (GPU acceleration)
        
        Args:
            X1: First set of features
            X2: Second set of features (defaults to X1 if None)
            compute_sigmas: Whether to compute and store per-channel sigma (True during training only)
        """
        if X2 is None:
            X2 = X1
            is_training = True  # Considered training if X1 == X2
        else:
            is_training = False
        
        n1 = X1['L'].shape[0]
        n2 = X2['L'].shape[0]
        
        weights = {'L': 0.25, 'a': 0.25, 'b': 0.25, 'texture': 0.25}
        
        if USE_GPU:
            K = torch.zeros((n1, n2), device=DEVICE)
            
            for channel, weight in weights.items():
                feat1 = torch.tensor(X1[channel], dtype=torch.float32, device=DEVICE)
                feat2 = torch.tensor(X2[channel], dtype=torch.float32, device=DEVICE)
                
                # Compute distance matrix
                if channel in ['L', 'a', 'b']:
                    D = self._emd_1d_gpu(feat1, feat2)
                else:
                    D = self._chi_square_gpu(feat1, feat2)
                
                # Compute or use stored sigma
                if compute_sigmas or channel not in self._channel_sigmas:
                    # Compute sigma based on mean distance (excluding diagonal)
                    D_np = D.cpu().numpy()
                    if is_training:
                        mask = ~np.eye(n1, dtype=bool)
                        mean_dist = np.mean(D_np[mask]) if n1 > 1 else np.mean(D_np)
                    else:
                        mean_dist = np.mean(D_np)
                    sigma = max(mean_dist * self.sigma_multiplier, 1e-6)
                    if compute_sigmas:
                        self._channel_sigmas[channel] = sigma
                else:
                    sigma = self._channel_sigmas[channel]
                
                # Extended Gaussian Kernel
                K += weight * torch.exp(-D / (sigma + 1e-8))
            
            return K.cpu().numpy()
        else:
            # CPU fallback
            K = np.zeros((n1, n2))
            
            for channel, weight in weights.items():
                feat1 = X1[channel]
                feat2 = X2[channel]
                
                if channel in ['L', 'a', 'b']:
                    D = emd_1d_matrix(feat1, feat2)
                else:
                    D = chi_square_distance_matrix(feat1, feat2)
                
                # Compute or use stored sigma
                if compute_sigmas or channel not in self._channel_sigmas:
                    if is_training:
                        mask = ~np.eye(n1, dtype=bool)
                        mean_dist = np.mean(D[mask]) if n1 > 1 else np.mean(D)
                    else:
                        mean_dist = np.mean(D)
                    sigma = max(mean_dist * self.sigma_multiplier, 1e-6)
                    if compute_sigmas:
                        self._channel_sigmas[channel] = sigma
                else:
                    sigma = self._channel_sigmas[channel]
                
                K += weight * np.exp(-D / (sigma + 1e-8))
            
            return K
    
    def fit(self, features: Dict[str, np.ndarray], y: np.ndarray):
        """Train classifier"""
        print("\n[Training LSSVM]")
        n_samples = features['L'].shape[0]
        
        # Compute kernel matrix (GPU accelerated), also computing per-channel sigma
        print(f"  Computing kernel matrix ({n_samples}x{n_samples})...")
        if USE_GPU:
            print(f"  [GPU] Memory before: {torch.cuda.memory_allocated()/1e6:.1f} MB")
        
        self._train_K = self._compute_kernel(features, compute_sigmas=True)
        
        # Print per-channel sigma
        print(f"  Channel sigmas (multiplier={self.sigma_multiplier}):")
        for ch, sigma in self._channel_sigmas.items():
            print(f"    {ch}: {sigma:.4f}")
        
        if USE_GPU:
            print(f"  [GPU] Memory after kernel: {torch.cuda.memory_allocated()/1e6:.1f} MB")
            torch.cuda.empty_cache()
        
        # Train LSSVM (using precomputed kernel)
        print("  Fitting LSSVM...")
        dummy_X = np.zeros((n_samples, 1))
        self.lssvm = LSSVM(gamma=self.gamma, use_gpu=USE_GPU)
        self.lssvm.fit(dummy_X, y, K=self._train_K)
        
        # Platt scaling
        print("  Fitting Platt scaling...")
        decision_values = self.lssvm.decision_function(dummy_X, K=self._train_K)
        self.platt = PlattScaler()
        self.platt.fit(decision_values, y)
        
        # Save training features
        self.train_features = features
        self._n_train = n_samples
        
        # Compute training BER (using decision value sign)
        preds = (decision_values > 0).astype(int)
        ber = balanced_error_rate(y, preds)
        print(f"  Training BER: {ber*100:.2f}%")
        
        return ber
    
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probability (GPU acceleration)"""
        # Compute test kernel matrix K(train, test)
        K_test = self._compute_kernel(self.train_features, features)
        
        n_test = features['L'].shape[0]
        dummy_X = np.zeros((n_test, 1))
        decision_values = self.lssvm.decision_function(dummy_X, K=K_test)
        
        if USE_GPU:
            torch.cuda.empty_cache()
        
        return self.platt.predict_proba(decision_values)
    
    def predict(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict labels (using decision value sign, not dependent on Platt scaling)"""
        # Compute test kernel matrix K(train, test)
        K_test = self._compute_kernel(self.train_features, features)
        
        n_test = features['L'].shape[0]
        dummy_X = np.zeros((n_test, 1))
        decision_values = self.lssvm.decision_function(dummy_X, K=K_test)
        
        if USE_GPU:
            torch.cuda.empty_cache()
        
        # Directly use decision value sign: >0 for shadow(1), <0 for non-shadow(0)
        return (decision_values > 0).astype(int)


def evaluate(classifier, features: Dict[str, np.ndarray], labels: np.ndarray, name: str):
    """Evaluate classifier"""
    print(f"\n[{name}]")
    
    preds = classifier.predict(features)
    probs = classifier.predict_proba(features)
    
    ber = balanced_error_rate(labels, preds)
    acc = np.mean(preds == labels)
    
    shadow_mask = labels == 1
    non_shadow_mask = labels == 0
    fpr = np.mean(preds[non_shadow_mask] == 1) if np.sum(non_shadow_mask) > 0 else 0
    fnr = np.mean(preds[shadow_mask] == 0) if np.sum(shadow_mask) > 0 else 0
    
    print(f"  BER: {ber*100:.2f}%")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  FPR: {fpr*100:.2f}%, FNR: {fnr*100:.2f}%")
    
    return {'ber': ber, 'accuracy': acc, 'fpr': fpr, 'fnr': fnr}


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main training function"""
    print_section("Shadow Detection Quick Training")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    timer = Timer()
    config = QUICK_CONFIG
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. Data loading
    # =========================================================================
    print_section("1. Loading Dataset")
    timer.start("Data Loading")
    
    dataset = get_dataset('sbu', split='train')
    n_total = len(dataset)
    
    np.random.seed(config['random_seed'])
    indices = np.random.permutation(n_total)
    
    n_train = config['n_train_images']
    n_val = config['n_val_images']
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    
    print(f"Total: {n_total}, Train: {n_train}, Val: {n_val}")
    timer.stop()
    
    # =========================================================================
    # 2. Texton dictionary training
    # =========================================================================
    print_section("2. Building Texton Dictionary")
    timer.start("Texton Dictionary")
    
    n_texton_train = min(config['texton_train_images'], n_train)
    print(f"Using {n_texton_train} images for texton dictionary...")
    
    texton_images = []
    for i, idx in enumerate(train_indices[:n_texton_train]):
        if (i + 1) % 10 == 0:
            print(f"  Loading image {i+1}/{n_texton_train}...")
        img, _ = dataset[idx]
        texton_images.append(img)
    
    print("Building texton dictionary...")
    texton_extractor = TextonFeatureExtractor(n_textons=config['n_textons'])
    texton_extractor.build_dictionary(texton_images, verbose=True)
    
    # Save texton dictionary
    texton_path = output_dir / 'texton_dict.pkl'
    texton_extractor.save_dictionary(str(texton_path))
    print(f"Saved to: {texton_path}")
    
    timer.stop()
    
    # =========================================================================
    # 3. Feature extraction
    # =========================================================================
    print_section("3. Extracting Features")
    timer.start("Feature Extraction")
    
    # Create components
    slic = SuperpixelSegmenter(
        n_segments=config['n_segments'],
        compactness=config['compactness']
    )
    region_gen = MeanShiftRegionGenerator(bandwidth=config['region_bandwidth'])
    feat_extractor = PaperCompliantFeatureExtractor(texton_extractor=texton_extractor)
    
    # Extract training set features
    train_features, train_labels = extract_dataset_features(
        dataset, train_indices, slic, region_gen, feat_extractor, "Training"
    )
    
    # Extract validation set features
    val_features, val_labels = extract_dataset_features(
        dataset, val_indices, slic, region_gen, feat_extractor, "Validation"
    )
    
    timer.stop()
    
    if len(train_labels) == 0:
        print("[ERROR] No training features!")
        return None
    
    # =========================================================================
    # 4. Train classifier
    # =========================================================================
    print_section("4. Training Classifier")
    timer.start("Training")
    
    best_ber = float('inf')
    best_model = None
    best_multiplier = None
    
    print("\nSearching for best sigma_multiplier...")
    for multiplier in config['sigma_multipliers']:
        print(f"\n--- sigma_multiplier={multiplier} ---")
        
        classifier = MultiKernelLSSVM(gamma=config['gamma'], sigma_multiplier=multiplier)
        classifier.fit(train_features, train_labels)
        
        if len(val_labels) > 0:
            val_result = evaluate(classifier, val_features, val_labels, f"Val (mult={multiplier})")
            if val_result['ber'] < best_ber:
                best_ber = val_result['ber']
                best_model = classifier
                best_multiplier = multiplier
                print(f"  [NEW BEST]")
    
    print(f"\nBest: sigma_multiplier={best_multiplier}, BER={best_ber*100:.2f}%")
    timer.stop()
    
    # =========================================================================
    # 5. Final evaluation
    # =========================================================================
    print_section("5. Final Evaluation")
    timer.start("Evaluation")
    
    train_result = evaluate(best_model, train_features, train_labels, "Final Train")
    val_result = evaluate(best_model, val_features, val_labels, "Final Val") if len(val_labels) > 0 else None
    
    timer.stop()
    
    # =========================================================================
    # 6. Save results
    # =========================================================================
    print_section("6. Saving Results")
    
    if config['save_model']:
        model_path = output_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': best_model,
                'texton_extractor': texton_extractor,
                'config': config,
                'best_sigma_multiplier': best_multiplier,
                'channel_sigmas': best_model._channel_sigmas if best_model else None,
                'train_result': train_result,
                'val_result': val_result
            }, f)
        print(f"Model saved: {model_path}")
    
    # Save results summary
    results_path = output_dir / 'results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("Shadow Detection Quick Training Results\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Train images: {n_train}\n")
        f.write(f"Val images: {n_val}\n")
        f.write(f"Train regions: {len(train_labels)}\n")
        f.write(f"Val regions: {len(val_labels)}\n")
        f.write(f"Best sigma_multiplier: {best_multiplier}\n")
        if best_model and best_model._channel_sigmas:
            f.write(f"Channel sigmas: {best_model._channel_sigmas}\n\n")
        f.write(f"Train BER: {train_result['ber']*100:.2f}%\n")
        f.write(f"Train Acc: {train_result['accuracy']*100:.2f}%\n")
        if val_result:
            f.write(f"Val BER: {val_result['ber']*100:.2f}%\n")
            f.write(f"Val Acc: {val_result['accuracy']*100:.2f}%\n")
    print(f"Results saved: {results_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Training Complete!")
    print(f"\nTrain BER: {train_result['ber']*100:.2f}%")
    if val_result:
        print(f"Val BER: {val_result['ber']*100:.2f}%")
    timer.summary()
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {'train': train_result, 'val': val_result, 'best_sigma_multiplier': best_multiplier}


if __name__ == "__main__":
    main()
