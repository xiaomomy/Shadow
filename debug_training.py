"""
Quick debug script - locate training failure causes
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_labels():
    """Check label data"""
    from data.dataset_loader import SBUDataset
    from preprocessing.superpixel import SuperpixelSegmenter
    from preprocessing.region import RegionGenerator
    import cv2
    
    print("=" * 60)
    print("  Debug labels - check masks")
    print("=" * 60)
    
    # Load data
    dataset = SBUDataset('data/sbu/SBU-shadow', split='train')
    
    n_debug = 5
    segmenter = SuperpixelSegmenter(n_segments=200, compactness=20)
    region_generator = RegionGenerator(bandwidth=None, quantile=0.3, use_spatial=False)
    
    total_shadow = 0
    total_non_shadow = 0
    
    for i in range(min(n_debug, len(dataset))):
        img, mask = dataset[i]
        
        if img is None or mask is None:
            print(f"[{i}] Image or mask is None!")
            continue
            
        print(f"\n[Image {i}]")
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"  Mask unique values: {np.unique(mask)}")
        print(f"  Mask value range: [{mask.min()}, {mask.max()}]")
        print(f"  Mask shadow pixels (>0): {np.sum(mask > 0)} / {mask.size} ({100*np.mean(mask > 0):.2f}%)")
        
        # Check if image and mask sizes match
        if mask.shape[:2] != img.shape[:2]:
            print(f"  [WARNING] Image and mask size mismatch!")
            print(f"    Image: {img.shape[:2]}, Mask: {mask.shape[:2]}")
            # Adjust mask size
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            print(f"    Resized mask to: {mask.shape}")
        
        # Segmentation
        sp_labels = segmenter.segment(img)
        region_labels = region_generator.generate_regions(img, sp_labels)
        n_regions = len(np.unique(region_labels))
        
        print(f"  Regions: {n_regions}")
        
        # Compute label for each region
        shadow_regions = 0
        for r in range(n_regions):
            region_mask = (region_labels == r)
            total_pixels = np.sum(region_mask)
            
            # Compute the ratio of shadow pixels within this region
            shadow_pixels = np.sum(mask[region_mask] > 0)
            ratio = shadow_pixels / total_pixels if total_pixels > 0 else 0
            
            label = 1 if ratio > 0.5 else 0
            
            if label == 1:
                shadow_regions += 1
                total_shadow += 1
            else:
                total_non_shadow += 1
        
        print(f"  Shadow regions: {shadow_regions} / {n_regions} ({100*shadow_regions/n_regions:.1f}%)")
    
    print("\n" + "=" * 60)
    print(f"Total: Shadow regions {total_shadow}, Non-shadow regions {total_non_shadow}")
    print(f"Shadow ratio: {100*total_shadow/(total_shadow+total_non_shadow):.2f}%")
    print("=" * 60)


def debug_kernel_and_lssvm():
    """Check kernel matrix and LSSVM"""
    from data.dataset_loader import SBUDataset
    from preprocessing.superpixel import SuperpixelSegmenter
    from preprocessing.region import RegionGenerator
    from preprocessing.features import PaperCompliantFeatureExtractor
    from preprocessing.texton import TextonFeatureExtractor
    from models.lssvm import LSSVM
    from models.kernels import ShadowDetectionMultiKernel
    from models.platt_scaling import PlattScaler
    from models.distances import emd_1d_matrix, chi_square_distance_matrix
    import cv2
    
    print("\n" + "=" * 60)
    print("  Debug kernel matrix and LSSVM")
    print("=" * 60)
    
    # Load data
    dataset = SBUDataset('data/sbu/SBU-shadow', split='train')
    
    n_debug = 10  # Use 10 images
    segmenter = SuperpixelSegmenter(n_segments=200, compactness=20)
    region_generator = RegionGenerator(bandwidth=None, quantile=0.3, use_spatial=False)
    
    # Build texton
    print("\n[1] Build texton dictionary...")
    texton_extractor = TextonFeatureExtractor(n_textons=128)
    gray_images = []
    for i in range(min(3, len(dataset))):
        img, _ = dataset[i]
        if img is None:
            continue
        gray_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    texton_extractor.build_dictionary(gray_images, verbose=False)
    
    feature_extractor = PaperCompliantFeatureExtractor(texton_extractor=texton_extractor)
    
    # Extract features and labels
    print("\n[2] Extract features...")
    all_features = {'L': [], 'a': [], 'b': [], 't': []}
    all_labels = []
    
    for idx in range(min(n_debug, len(dataset))):
        img, mask = dataset[idx]
        if img is None or mask is None:
            continue
        
        # Ensure mask and image sizes match
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
        
        print(f"  Image {idx}: {n_regions} regions")
    
    X = {k: np.array(v) for k, v in all_features.items()}
    y = np.array(all_labels)
    
    print(f"\n[Statistics]")
    print(f"  Total regions: {len(y)}")
    print(f"  Shadow regions: {np.sum(y == 1)} ({100*np.mean(y):.1f}%)")
    print(f"  Non-shadow regions: {np.sum(y == 0)} ({100*np.mean(y==0):.1f}%)")
    
    # Check if there are enough positive and negative samples
    if np.sum(y == 1) == 0:
        print("\n[CRITICAL ERROR] No shadow samples!")
        print("  Possible causes: mask data loading error or label computation issues")
        return
    
    if np.sum(y == 0) == 0:
        print("\n[CRITICAL ERROR] No non-shadow samples!")
        return
    
    # Check features
    print("\n[3] Feature distribution...")
    for channel in ['L', 'a', 'b', 't']:
        feat = X[channel]
        print(f"  {channel}: shape={feat.shape}, "
              f"min={feat.min():.4f}, max={feat.max():.4f}, "
              f"mean={feat.mean():.4f}, std={feat.std():.4f}")
    
    # Check distance matrix
    print("\n[4] Distance matrix...")
    D_L = emd_1d_matrix(X['L'], X['L'])
    D_a = emd_1d_matrix(X['a'], X['a'])
    D_b = emd_1d_matrix(X['b'], X['b'])
    D_t = chi_square_distance_matrix(X['t'], X['t'])
    
    for name, D in [('L', D_L), ('a', D_a), ('b', D_b), ('t', D_t)]:
        print(f"  D_{name}: min={D.min():.4f}, max={D.max():.4f}, "
              f"mean={D.mean():.4f}, std={D.std():.4f}")
    
    # Test impact of different sigma
    print("\n[5] Test impact of different sigma on kernel matrix (using D_L as example)...")
    test_sigmas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for sigma in test_sigmas:
        K = np.exp(-D_L / sigma)
        print(f"  sigma={sigma:6.3f}: K=[{K.min():.4f}, {K.max():.4f}], "
              f"mean={K.mean():.4f}, std={K.std():.4f}")
    
    # Find best sigma (making kernel matrix have enough variance)
    print("\n[6] Finding appropriate sigma...")
    best_sigma = None
    best_std = 0
    for sigma in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        K = np.exp(-D_L / sigma)
        if K.std() > best_std and K.mean() > 0.1 and K.mean() < 0.9:
            best_std = K.std()
            best_sigma = sigma
    print(f"  Recommended sigma (for L): {best_sigma} (std={best_std:.4f})")
    
    # Train LSSVM
    print("\n[7] Train LSSVM...")
    
    # Use appropriate sigma found
    sigma_val = best_sigma if best_sigma else 0.5
    
    # Build combined kernel matrix
    K_L = np.exp(-D_L / sigma_val)
    K_a = np.exp(-D_a / sigma_val)
    K_b = np.exp(-D_b / sigma_val)
    
    # use different sigma for texton due to different distance range
    sigma_t = D_t.mean() if D_t.mean() > 0 else 1.0
    K_t = np.exp(-D_t / sigma_t)
    
    # Combination
    K = 0.25 * K_L + 0.25 * K_a + 0.25 * K_b + 0.25 * K_t
    
    print(f"  Combined kernel matrix K: [{K.min():.4f}, {K.max():.4f}], "
          f"mean={K.mean():.4f}, std={K.std():.4f}")
    
    # Convert labels
    y_train = 2 * y - 1  # 0->-1, 1->+1
    
    lssvm = LSSVM(gamma=1.0)
    lssvm.fit(K, y_train)
    
    print(f"  alpha: [{lssvm.alpha_.min():.4f}, {lssvm.alpha_.max():.4f}], "
          f"mean={lssvm.alpha_.mean():.4f}")
    print(f"  bias: {lssvm.bias_:.4f}")
    
    # Decision values
    decision_values = lssvm.decision_function(K)
    print(f"\n[8] Decision value distribution...")
    print(f"  Decision values: [{decision_values.min():.4f}, {decision_values.max():.4f}], "
          f"mean={decision_values.mean():.4f}, std={decision_values.std():.4f}")
    print(f"  >0: {np.sum(decision_values > 0)}, <0: {np.sum(decision_values < 0)}")
    
    # Direct prediction
    pred_direct = (decision_values > 0).astype(int)
    acc_direct = np.mean(pred_direct == y)
    
    # Compute BER
    tp = np.sum((pred_direct == 1) & (y == 1))
    fp = np.sum((pred_direct == 1) & (y == 0))
    tn = np.sum((pred_direct == 0) & (y == 0))
    fn = np.sum((pred_direct == 0) & (y == 1))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    ber = (fpr + fnr) / 2
    
    print(f"\n[9] Evaluation results...")
    print(f"  Direct prediction accuracy: {100*acc_direct:.2f}%")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  FPR={100*fpr:.2f}%, FNR={100*fnr:.2f}%")
    print(f"  BER={100*ber:.2f}%")
    
    # Platt scaling
    print(f"\n[10] Platt scaling...")
    platt = PlattScaler()
    platt.fit(decision_values, y)
    print(f"  Platt parameters: a={platt.a_:.4f}, b={platt.b_:.4f}")
    
    probs = platt.predict_proba(decision_values)
    print(f"  Probability: [{probs.min():.4f}, {probs.max():.4f}], mean={probs.mean():.4f}")
    
    pred_platt = (probs > 0.5).astype(int)
    acc_platt = np.mean(pred_platt == y)
    print(f"  Platt prediction accuracy: {100*acc_platt:.2f}%")
    print(f"  Predicted as shadow: {np.sum(pred_platt)}, Predicted as non-shadow: {len(pred_platt) - np.sum(pred_platt)}")


if __name__ == '__main__':
    debug_labels()
    debug_kernel_and_lssvm()
