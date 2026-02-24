"""
Texton Feature Extraction Module.

This module implements the texton-based texture features as described in the paper.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 3.1:
    "To represent texture, we compute a 128-bin texton histogram. We run the 
    full MR8 filter set [33] in the whole dataset and cluster the filter 
    responses into 128 textons using k-means."
    
    [33] Varma & Zisserman, "A Statistical Approach to Texture Classification 
    from Single Images", IJCV 2005

MR8 Filter Bank (Maximum Response 8):
=====================================
The MR8 filter bank consists of:
1. Edge filters at 3 scales × 6 orientations → take max response over orientations (3 values)
2. Bar filters at 3 scales × 6 orientations → take max response over orientations (3 values)
3. Gaussian filter (1 value)
4. Laplacian of Gaussian filter (1 value)

Total: 8 filter responses per pixel (hence "MR8")

The key insight is that by taking the maximum response over orientations,
the filter bank becomes rotationally invariant.

Texton Dictionary:
==================
1. Apply MR8 filters to all training images
2. Collect filter responses from all pixels
3. Cluster responses using k-means into 128 clusters (textons)
4. The cluster centers form the "texton dictionary"

Texton Histogram:
=================
For each region:
1. Get MR8 filter responses for all pixels in the region
2. Assign each pixel to the nearest texton (cluster center)
3. Build a 128-bin histogram of texton assignments
4. Normalize the histogram

Author: [Your Name]
Date: 2026
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve
from scipy.cluster.vq import kmeans2, vq
from skimage.color import rgb2gray
from typing import Optional, Tuple, Dict, List
import warnings

# GPU Support
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')


class MR8FilterBank:
    """
    Maximum Response 8 (MR8) Filter Bank.
    
    This implements the MR8 filter bank from Varma & Zisserman (2005),
    which is used in the paper for texton computation.
    
    The filter bank consists of:
    - Edge filters (first derivative of Gaussian) at 3 scales, 6 orientations
    - Bar filters (second derivative of Gaussian) at 3 scales, 6 orientations
    - Gaussian filter
    - Laplacian of Gaussian filter
    
    The "Maximum Response" means we take the maximum response over all
    orientations for each scale, resulting in 8 values per pixel:
    - 3 edge responses (one per scale)
    - 3 bar responses (one per scale)
    - 1 Gaussian response
    - 1 LoG response
    
    Reference:
        Varma & Zisserman, "A Statistical Approach to Texture Classification 
        from Single Images", IJCV 2005
    """
    
    # Default scales (σ values) for the filters
    # Paper [33] uses σ = 1, 2, 4 for the oriented filters
    DEFAULT_SCALES = [1, 2, 4]
    
    # Number of orientations for edge and bar filters
    N_ORIENTATIONS = 6
    
    # Elongation factor for oriented filters
    ELONGATION = 3
    
    def __init__(
        self,
        scales: Optional[List[float]] = None,
        n_orientations: int = 6
    ):
        """
        Initialize MR8 filter bank.
        
        Args:
            scales: List of σ values for the filters (default: [1, 2, 4])
            n_orientations: Number of orientations for edge/bar filters (default: 6)
        """
        self.scales = scales if scales is not None else self.DEFAULT_SCALES
        self.n_orientations = n_orientations
        
        # Precompute all filters
        self._filters = {}
        self._build_filters()
        
        # Prepare GPU filters if available
        self._filters_torch = {}
        if USE_GPU:
            self._prepare_gpu_filters()
    
    def _prepare_gpu_filters(self) -> None:
        """Prepare filters as torch tensors for GPU acceleration."""
        # Convert oriented filters to (N_orientations, 1, H, W) tensors
        self._filters_torch['edge'] = {}
        self._filters_torch['bar'] = {}
        
        for scale in self.scales:
            # Edge filters
            edge_list = self._edge_filters[scale]
            max_h = max(f.shape[0] for f in edge_list)
            max_w = max(f.shape[1] for f in edge_list)
            
            edge_tensor = torch.zeros((len(edge_list), 1, max_h, max_w), device=DEVICE, dtype=torch.float32)
            for i, f in enumerate(edge_list):
                h, w = f.shape
                y_off, x_off = (max_h - h) // 2, (max_w - w) // 2
                edge_tensor[i, 0, y_off:y_off+h, x_off:x_off+w] = torch.tensor(f, device=DEVICE, dtype=torch.float32)
            self._filters_torch['edge'][scale] = edge_tensor
            
            # Bar filters
            bar_list = self._bar_filters[scale]
            bar_tensor = torch.zeros((len(bar_list), 1, max_h, max_w), device=DEVICE, dtype=torch.float32)
            for i, f in enumerate(bar_list):
                h, w = f.shape
                y_off, x_off = (max_h - h) // 2, (max_w - w) // 2
                bar_tensor[i, 0, y_off:y_off+h, x_off:x_off+w] = torch.tensor(f, device=DEVICE, dtype=torch.float32)
            self._filters_torch['bar'][scale] = bar_tensor
            
        # Gaussian filter
        g_f = self._gaussian_filter
        self._filters_torch['gaussian'] = torch.tensor(g_f, device=DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # LoG filter
        log_f = self._log_filter
        self._filters_torch['log'] = torch.tensor(log_f, device=DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def _build_filters(self) -> None:
        """Build all filters in the filter bank."""
        
        # Orientations: evenly spaced from 0 to π
        orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        
        # Build edge and bar filters at each scale and orientation
        self._edge_filters = {}  # {scale: [filters for each orientation]}
        self._bar_filters = {}
        
        for scale in self.scales:
            self._edge_filters[scale] = []
            self._bar_filters[scale] = []
            
            for theta in orientations:
                # Edge filter (first derivative of Gaussian)
                edge = self._make_edge_filter(scale, theta)
                self._edge_filters[scale].append(edge)
                
                # Bar filter (second derivative of Gaussian)
                bar = self._make_bar_filter(scale, theta)
                self._bar_filters[scale].append(bar)
        
        # Gaussian filter (use largest scale)
        self._gaussian_filter = self._make_gaussian_filter(max(self.scales))
        
        # Laplacian of Gaussian filter (use largest scale)
        self._log_filter = self._make_log_filter(max(self.scales))
    
    def _make_gaussian_filter(self, sigma: float) -> np.ndarray:
        """
        Create a 2D Gaussian filter.
        
        G(x,y) = (1/2πσ²) exp(-(x² + y²)/(2σ²))
        """
        # Filter size: 6σ to capture most of the Gaussian
        size = int(np.ceil(sigma * 6)) | 1  # Ensure odd size
        half = size // 2
        
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        X, Y = np.meshgrid(x, y)
        
        G = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        G = G / G.sum()  # Normalize
        
        return G
    
    def _make_log_filter(self, sigma: float) -> np.ndarray:
        """
        Create a Laplacian of Gaussian (LoG) filter.
        
        LoG(x,y) = -1/(πσ⁴) [1 - (x²+y²)/(2σ²)] exp(-(x²+y²)/(2σ²))
        
        Also known as "Mexican hat" wavelet.
        """
        size = int(np.ceil(sigma * 6)) | 1
        half = size // 2
        
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        X, Y = np.meshgrid(x, y)
        
        r2 = X**2 + Y**2
        sigma2 = sigma**2
        
        # LoG formula
        LoG = -(1 / (np.pi * sigma2**2)) * (1 - r2 / (2 * sigma2)) * np.exp(-r2 / (2 * sigma2))
        
        # Zero-mean normalization
        LoG = LoG - LoG.mean()
        
        return LoG
    
    def _make_edge_filter(self, sigma: float, theta: float) -> np.ndarray:
        """
        Create an oriented edge filter (first derivative of Gaussian).
        
        This is a Gaussian derivative along the direction perpendicular to theta.
        The filter is elongated by a factor of 3 along the theta direction.
        
        Args:
            sigma: Scale parameter
            theta: Orientation angle in radians
        """
        # Elongated filter: σ_x = σ, σ_y = 3σ (elongation factor)
        sigma_x = sigma
        sigma_y = sigma * self.ELONGATION
        
        # Filter size
        size = int(np.ceil(max(sigma_x, sigma_y) * 6)) | 1
        half = size // 2
        
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        X_rot = X * cos_t + Y * sin_t
        Y_rot = -X * sin_t + Y * cos_t
        
        # Gaussian envelope
        G = np.exp(-X_rot**2 / (2 * sigma_x**2) - Y_rot**2 / (2 * sigma_y**2))
        
        # First derivative along X (perpendicular to elongation)
        dG = -X_rot / (sigma_x**2) * G
        
        # Normalize
        dG = dG / (np.abs(dG).sum() + 1e-10)
        
        return dG
    
    def _make_bar_filter(self, sigma: float, theta: float) -> np.ndarray:
        """
        Create an oriented bar filter (second derivative of Gaussian).
        
        This detects bar-like structures at orientation theta.
        
        Args:
            sigma: Scale parameter
            theta: Orientation angle in radians
        """
        sigma_x = sigma
        sigma_y = sigma * self.ELONGATION
        
        size = int(np.ceil(max(sigma_x, sigma_y) * 6)) | 1
        half = size // 2
        
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        X_rot = X * cos_t + Y * sin_t
        Y_rot = -X * sin_t + Y * cos_t
        
        # Gaussian envelope
        G = np.exp(-X_rot**2 / (2 * sigma_x**2) - Y_rot**2 / (2 * sigma_y**2))
        
        # Second derivative along X
        d2G = (X_rot**2 / sigma_x**4 - 1 / sigma_x**2) * G
        
        # Zero-mean normalization
        d2G = d2G - d2G.mean()
        
        return d2G
    
    def apply(self, image: np.ndarray, use_gpu: bool = True) -> np.ndarray:
        """
        Apply MR8 filter bank to an image.
        
        Args:
            image: Grayscale image (H, W) with values in [0, 1]
            use_gpu: Whether to use GPU if available
            
        Returns:
            Filter responses: (H, W, 8) array
                - channels 0-2: max edge responses at 3 scales
                - channels 3-5: max bar responses at 3 scales
                - channel 6: Gaussian response
                - channel 7: LoG response
        """
        if image.ndim == 3:
            image = rgb2gray(image)
        
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        
        # Use GPU implementation if requested and available
        if use_gpu and USE_GPU:
            return self._apply_gpu(image)
        
        H, W = image.shape
        responses = np.zeros((H, W, 8), dtype=np.float64)
        
        # Apply edge filters and take max over orientations
        for i, scale in enumerate(self.scales):
            edge_responses = []
            for edge_filter in self._edge_filters[scale]:
                resp = convolve(image, edge_filter, mode='reflect')
                edge_responses.append(np.abs(resp))  # Take absolute value
            
            # Maximum response over orientations
            responses[:, :, i] = np.max(edge_responses, axis=0)
        
        # Apply bar filters and take max over orientations
        for i, scale in enumerate(self.scales):
            bar_responses = []
            for bar_filter in self._bar_filters[scale]:
                resp = convolve(image, bar_filter, mode='reflect')
                bar_responses.append(np.abs(resp))
            
            # Maximum response over orientations
            responses[:, :, 3 + i] = np.max(bar_responses, axis=0)
        
        # Gaussian response
        responses[:, :, 6] = convolve(image, self._gaussian_filter, mode='reflect')
        
        # LoG response
        responses[:, :, 7] = np.abs(convolve(image, self._log_filter, mode='reflect'))
        
        return responses
    
    def _apply_gpu(self, image: np.ndarray) -> np.ndarray:
        """Apply filters using GPU acceleration."""
        H, W = image.shape
        img_tensor = torch.tensor(image, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        
        responses = torch.zeros((H, W, 8), dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            # Edge and Bar filters
            for i, scale in enumerate(self.scales):
                # Edge
                edge_filters = self._filters_torch['edge'][scale]
                pad_h, pad_w = edge_filters.shape[2] // 2, edge_filters.shape[3] // 2
                edge_resp = F.conv2d(img_tensor, edge_filters, padding=(pad_h, pad_w))
                responses[:, :, i] = torch.max(torch.abs(edge_resp), dim=1)[0].squeeze()
                
                # Bar
                bar_filters = self._filters_torch['bar'][scale]
                pad_h, pad_w = bar_filters.shape[2] // 2, bar_filters.shape[3] // 2
                bar_resp = F.conv2d(img_tensor, bar_filters, padding=(pad_h, pad_w))
                responses[:, :, 3 + i] = torch.max(torch.abs(bar_resp), dim=1)[0].squeeze()
            
            # Gaussian
            g_filter = self._filters_torch['gaussian']
            pad_h, pad_w = g_filter.shape[2] // 2, g_filter.shape[3] // 2
            g_resp = F.conv2d(img_tensor, g_filter, padding=(pad_h, pad_w))
            responses[:, :, 6] = g_resp.squeeze()
            
            # LoG
            log_filter = self._filters_torch['log']
            pad_h, pad_w = log_filter.shape[2] // 2, log_filter.shape[3] // 2
            log_resp = F.conv2d(img_tensor, log_filter, padding=(pad_h, pad_w))
            responses[:, :, 7] = torch.abs(log_resp).squeeze()
            
        return responses.cpu().numpy().astype(np.float64)

    def get_filter_info(self) -> Dict:
        """Return information about the filter bank."""
        return {
            'n_filters': 8,
            'scales': self.scales,
            'n_orientations': self.n_orientations,
            'filter_names': [
                f'edge_scale{s}' for s in self.scales
            ] + [
                f'bar_scale{s}' for s in self.scales
            ] + ['gaussian', 'log']
        }


class TextonDictionary:
    """
    Texton Dictionary for texture representation.
    
    This class builds a texton dictionary by clustering MR8 filter responses
    using k-means, as described in the paper.
    
    Reference:
        Paper Section 3.1: "We run the full MR8 filter set [33] in the whole 
        dataset and cluster the filter responses into 128 textons using k-means."
    
    Usage:
        1. Create dictionary: dict = TextonDictionary(n_textons=128)
        2. Build from training images: dict.build(images)
        3. Compute histogram for a region: hist = dict.compute_histogram(image, mask)
    """
    
    def __init__(
        self,
        n_textons: int = 128,
        filter_bank: Optional[MR8FilterBank] = None,
        sample_fraction: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize texton dictionary.
        
        Args:
            n_textons: Number of texton clusters (default: 128 per paper)
            filter_bank: MR8 filter bank (default: create new one)
            sample_fraction: Fraction of pixels to sample for k-means (for efficiency)
            random_state: Random seed for k-means
        """
        self.n_textons = n_textons
        self.filter_bank = filter_bank if filter_bank is not None else MR8FilterBank()
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        
        # Texton centers (set after building)
        self.centers_ = None
        self._is_built = False
    
    def build(self, images: List[np.ndarray], verbose: bool = True) -> 'TextonDictionary':
        """
        Build texton dictionary from a set of images.
        
        This applies MR8 filters to all images, samples filter responses,
        and clusters them into textons using k-means.
        
        Args:
            images: List of images (grayscale or RGB)
            verbose: Whether to print progress
            
        Returns:
            self: The fitted dictionary
        """
        if verbose:
            print(f"[TextonDict] Building texton dictionary with {self.n_textons} textons...")
        
        # Collect filter responses from all images
        all_responses = []
        
        for i, image in enumerate(images):
            if verbose and i % 10 == 0:
                print(f"  Processing image {i+1}/{len(images)}...")
            
            # Apply MR8 filter bank
            responses = self.filter_bank.apply(image)  # (H, W, 8)
            
            # Reshape to (N_pixels, 8)
            responses_flat = responses.reshape(-1, 8)
            
            # Sample a fraction of pixels (for efficiency)
            n_samples = int(len(responses_flat) * self.sample_fraction)
            if n_samples > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state + i)
                indices = np.random.choice(len(responses_flat), n_samples, replace=False)
                all_responses.append(responses_flat[indices])
        
        # Concatenate all responses
        all_responses = np.vstack(all_responses).astype(np.float32)
        
        self.build_from_responses(all_responses, verbose=verbose)
        return self
    
    def build_from_responses(
        self, 
        responses: np.ndarray,
        verbose: bool = True
    ) -> 'TextonDictionary':
        """
        Build dictionary from precomputed filter responses.
        
        Args:
            responses: Filter responses (N_samples, 8)
            verbose: Whether to print progress
            
        Returns:
            self: The fitted dictionary
        """
        if verbose:
            print(f"[TextonDict] Building from {len(responses)} precomputed responses...")
        
        if USE_GPU:
            try:
                self.centers_ = self._kmeans_gpu(responses, self.n_textons, verbose=verbose)
                self._is_built = True
                return self
            except Exception as e:
                warnings.warn(f"GPU K-Means failed: {e}. Falling back to CPU.")
                torch.cuda.empty_cache()

        # CPU Fallback
        # Normalize responses
        norms = np.linalg.norm(responses, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        responses_norm = responses / norms
        
        # Run k-means
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.centers_, _ = kmeans2(
                responses_norm.astype(np.float32),
                self.n_textons,
                iter=20,
                minit='points'
            )
        
        self._is_built = True
        
        if verbose:
            print(f"  Dictionary built successfully!")
        
        return self

    def _kmeans_gpu(self, data: np.ndarray, k: int, max_iter: int = 20, verbose: bool = True) -> np.ndarray:
        """K-Means implementation using PyTorch for GPU acceleration."""
        X = torch.tensor(data, dtype=torch.float32, device=DEVICE)
        
        # Normalize data (L2)
        norms = torch.norm(X, dim=1, keepdim=True)
        X = X / torch.where(norms > 0, norms, torch.ones_like(norms))
        
        # Random initialization
        n_samples = X.shape[0]
        indices = torch.randperm(n_samples, device=DEVICE)[:k]
        centers = X[indices].clone()
        
        if verbose:
            print(f"  [GPU] Running K-Means (k={k}, samples={n_samples})...")
            
        for i in range(max_iter):
            # Compute distances to centers: |x-c|^2 = |x|^2 + |c|^2 - 2xc
            # Since |x|=1 and |c| is close to 1, we can minimize -2xc or maximize xc
            # For robustness, we use the standard formula
            dist = torch.cdist(X, centers) # (N, k)
            labels = torch.argmin(dist, dim=1)
            
            # Update centers
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(k, 1, device=DEVICE)
            
            # Vectorized center update
            ones = torch.ones(n_samples, 1, device=DEVICE)
            new_centers.index_add_(0, labels, X)
            counts.index_add_(0, labels, ones)
            
            # Avoid division by zero
            new_centers = new_centers / torch.where(counts > 0, counts, torch.ones_like(counts))
            
            # Check convergence (optional, here we just run max_iter)
            center_shift = torch.norm(new_centers - centers)
            centers = new_centers
            
            if center_shift < 1e-5:
                break
                
        torch.cuda.empty_cache()
        return centers.cpu().numpy()
    
    def assign_textons(self, responses: np.ndarray, use_gpu: bool = True) -> np.ndarray:
        """
        Assign filter responses to nearest texton.
        
        Args:
            responses: Filter responses (N, 8) or (H, W, 8)
            use_gpu: Whether to use GPU if available
            
        Returns:
            Texton assignments (N,) or (H, W)
        """
        if not self._is_built:
            raise RuntimeError("Dictionary must be built first. Call build() or load().")
        
        original_shape = responses.shape[:-1]
        responses_flat = responses.reshape(-1, 8)
        
        # Use GPU implementation if requested and available
        if use_gpu and USE_GPU:
            return self._assign_textons_gpu(responses_flat, original_shape)
        
        # Normalize
        norms = np.linalg.norm(responses_flat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        responses_norm = responses_flat / norms
        
        # Find nearest texton for each response
        assignments, _ = vq(responses_norm.astype(np.float32), self.centers_)
        
        return assignments.reshape(original_shape)
    
    def _assign_textons_gpu(self, responses_flat: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """Assign textons using GPU acceleration."""
        resp_tensor = torch.tensor(responses_flat, dtype=torch.float32, device=DEVICE)
        centers_tensor = torch.tensor(self.centers_, dtype=torch.float32, device=DEVICE)
        
        # Normalize responses (to match CPU build process)
        r_norms = torch.norm(resp_tensor, dim=1, keepdim=True)
        resp_norm = resp_tensor / torch.where(r_norms > 0, r_norms, torch.ones_like(r_norms))
        
        # Precompute centers squared norm for Euclidean distance
        # |x - y|^2 = |x|^2 + |y|^2 - 2<x, y>
        centers_sq_norm = torch.sum(centers_tensor**2, dim=1) # (128,)
        
        batch_size = 100000
        assignments = torch.zeros(len(responses_flat), dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            for i in range(0, len(responses_flat), batch_size):
                end = min(i + batch_size, len(responses_flat))
                batch_resp = resp_norm[i:end]
                
                # |x|^2
                resp_sq_norm = torch.sum(batch_resp**2, dim=1, keepdim=True) # (batch, 1)
                
                # -2<x, y>
                dot_products = torch.matmul(batch_resp, centers_tensor.t()) # (batch, 128)
                
                # |x - y|^2
                dists_sq = resp_sq_norm + centers_sq_norm - 2 * dot_products
                
                assignments[i:end] = torch.argmin(dists_sq, dim=1)
                
        return assignments.cpu().numpy().reshape(original_shape)

    def compute_histogram(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute texton histogram for an image region.
        
        Args:
            image: Grayscale or RGB image
            mask: Boolean mask for the region (if None, use entire image)
            
        Returns:
            Normalized texton histogram (n_textons,)
        """
        if not self._is_built:
            raise RuntimeError("Dictionary must be built first.")
        
        # Apply filter bank
        responses = self.filter_bank.apply(image)
        
        # Assign textons
        assignments = self.assign_textons(responses)
        
        # Select region
        if mask is not None:
            region_textons = assignments[mask]
        else:
            region_textons = assignments.ravel()
        
        # Compute histogram
        hist, _ = np.histogram(
            region_textons,
            bins=self.n_textons,
            range=(0, self.n_textons)
        )
        
        # Normalize
        hist = hist.astype(np.float64)
        hist = hist / (hist.sum() + 1e-10)
        
        return hist
    
    def compute_histogram_from_responses(
        self,
        responses: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute histogram from precomputed filter responses.
        
        Args:
            responses: Filter responses (H, W, 8)
            mask: Boolean mask for the region
            
        Returns:
            Normalized texton histogram
        """
        if not self._is_built:
            raise RuntimeError("Dictionary must be built first.")
        
        # Assign textons
        assignments = self.assign_textons(responses)
        
        # Select region
        if mask is not None:
            region_textons = assignments[mask]
        else:
            region_textons = assignments.ravel()
        
        # Compute histogram
        hist, _ = np.histogram(
            region_textons,
            bins=self.n_textons,
            range=(0, self.n_textons)
        )
        
        # Normalize
        hist = hist.astype(np.float64)
        hist = hist / (hist.sum() + 1e-10)
        
        return hist
    
    def save(self, filepath: str) -> None:
        """Save texton dictionary to file."""
        if not self._is_built:
            raise RuntimeError("Dictionary must be built first.")
        
        np.savez(
            filepath,
            centers=self.centers_,
            n_textons=self.n_textons
        )
    
    def load(self, filepath: str) -> 'TextonDictionary':
        """Load texton dictionary from file."""
        data = np.load(filepath)
        self.centers_ = data['centers']
        self.n_textons = int(data['n_textons'])
        self._is_built = True
        return self
    
    @property
    def is_built(self) -> bool:
        """Check if dictionary is built."""
        return self._is_built


class TextonFeatureExtractor:
    """
    Texton feature extractor for shadow detection.
    
    This class provides a high-level interface for extracting texton features
    from image regions, following the paper's approach.
    
    Reference:
        Paper Section 3.1: "To represent texture, we compute a 128-bin texton 
        histogram."
    """
    
    def __init__(
        self,
        n_textons: int = 128,
        texton_dict: Optional[TextonDictionary] = None
    ):
        """
        Initialize texton feature extractor.
        
        Args:
            n_textons: Number of textons (default: 128 per paper)
            texton_dict: Prebuilt texton dictionary (optional)
        """
        self.n_textons = n_textons
        
        if texton_dict is not None:
            self.texton_dict = texton_dict
        else:
            self.texton_dict = TextonDictionary(n_textons=n_textons)
        
        self.filter_bank = self.texton_dict.filter_bank
        
        # Cache for filter responses
        self._cached_responses = None
        self._cached_image_id = None
    
    def build_dictionary(
        self,
        images: List[np.ndarray],
        verbose: bool = True
    ) -> None:
        """
        Build texton dictionary from training images.
        
        Args:
            images: List of training images
            verbose: Whether to print progress
        """
        self.texton_dict.build(images, verbose=verbose)
    
    def load_dictionary(self, filepath: str) -> None:
        """Load texton dictionary from file."""
        self.texton_dict.load(filepath)
    
    def save_dictionary(self, filepath: str) -> None:
        """Save texton dictionary to file."""
        self.texton_dict.save(filepath)
    
    def extract_features(
        self,
        image: np.ndarray,
        region_labels: np.ndarray,
        cache_responses: bool = True,
        use_gpu: bool = True
    ) -> np.ndarray:
        """
        Extract texton histogram features for all regions.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            region_labels: Region label map (H, W)
            cache_responses: Whether to cache filter responses
            use_gpu: Whether to use GPU for calculation
            
        Returns:
            Feature matrix (n_regions, n_textons)
        """
        if not self.texton_dict.is_built:
            raise RuntimeError("Texton dictionary must be built first. "
                             "Call build_dictionary() or load_dictionary().")
        
        n_regions = int(region_labels.max()) + 1
        features = np.zeros((n_regions, self.n_textons))
        
        # Compute filter responses for the whole image
        image_id = id(image)
        if cache_responses and self._cached_image_id == image_id:
            responses = self._cached_responses
        else:
            responses = self.filter_bank.apply(image, use_gpu=use_gpu)
            if cache_responses:
                self._cached_responses = responses
                self._cached_image_id = image_id
        
        # Assign textons for the whole image (more efficient on GPU)
        assignments = self.texton_dict.assign_textons(responses, use_gpu=use_gpu)
        
        # Compute histogram for each region
        for region_id in range(n_regions):
            mask = region_labels == region_id
            
            if not mask.any():
                continue
            
            # Select region textons from pre-assigned map
            region_textons = assignments[mask]
            
            # Compute histogram
            hist, _ = np.histogram(
                region_textons,
                bins=self.n_textons,
                range=(0, self.n_textons)
            )
            
            # Normalize
            hist = hist.astype(np.float64)
            features[region_id] = hist / (hist.sum() + 1e-10)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Return feature dimension."""
        return self.n_textons


def create_default_texton_extractor(
    n_textons: int = 128
) -> TextonFeatureExtractor:
    """
    Create a texton feature extractor with default settings.
    
    Note: The dictionary needs to be built or loaded before use.
    
    Args:
        n_textons: Number of textons (default: 128 per paper)
        
    Returns:
        TextonFeatureExtractor instance
    """
    return TextonFeatureExtractor(n_textons=n_textons)

