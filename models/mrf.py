"""
Markov Random Field (MRF) for Shadow Detection.

This module implements the MRF-based context incorporation described in Section 4
of the paper. The MRF combines the region classifier predictions with pairwise
contextual cues to improve shadow detection accuracy.

Reference:
    Vicente et al., "Leave-One-Out Kernel Optimization for Shadow Detection", ICCV 2015
    
    Paper Section 4:
    "We enhance shadow detection by embedding the shadow region classifier in an 
    MRF framework. We construct a graph where each node corresponds to a region 
    and each edge corresponds to a pair of neighboring regions."
    
    Energy Function (Equation 6):
    E = Σ_i φ(x_i) + Σ_{i,j∈Ω^a} ψ_a(x_i, x_j) + Σ_{i,j∈Ω^d} ψ_d(x_i, x_j)
    
    where:
    - x_i ∈ {-1, +1} is the label for region R_i (+1 = shadow, -1 = non-shadow)
    - φ(x_i) is the unary potential based on region classifier
    - ψ_a is the affinity pairwise potential (submodular)
    - ψ_d is the disparity pairwise potential (supermodular)
    - Ω^a, Ω^d are sets of neighboring region pairs

Mathematical Details:
=====================

4.1 Unary Potentials:
    φ(x_i) = -ω_i · P(x_i|R_i)
    
    where:
    - ω_i is the area (in pixels) of region R_i
    - P(x_i|R_i) is the Platt-scaled shadow probability

4.2 Affinity Pairwise Potentials:
    ψ_a(x_i, x_j) = { ω_ij · K(R_i, R_j)  if x_i ≠ x_j and K(R_i, R_j) > 0.5
                   { 0                    otherwise
    
    where:
    - ω_ij = √(ω_i · ω_j) is the geometric mean of region areas
    - K(R_i, R_j) is the kernel similarity between regions
    
    "The affinity potentials encourage similar adjacent regions to have the same label"

4.3 Disparity Pairwise Potentials:
    ψ_d(x_i, x_j) = { 0                          if x_i ≠ x_j
                   { ω_ij · P^d(1|R_i, R_j)     otherwise
    
    where P^d is the probability from a trained disparity classifier.
    
    "The disparity potentials prefer different labels for shadow/non-shadow 
    region pairs (using the output of a classifier for region pairs)."
    
    Features for disparity classifier:
    - χ² distance between texton histograms
    - EMD between L*, a*, b* histograms
    - RGB ratios: ((ρ_R + ρ_G + ρ_B)/3, ρ_R/ρ_B, ρ_G/ρ_B)

Optimization:
=============
    "The energy function (6) requires optimizing the node labels of a sparse graph. 
    This energy function has submodular pairwise interactions ψ_a(x_i, x_j) and 
    supermodular interactions ψ_d(x_i, x_j). We optimize it using QPBO [18, 24]."

Author: [Your Name]
Date: 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scipy.sparse import csr_matrix
import warnings

# For QPBO optimization
try:
    import maxflow
    HAS_MAXFLOW = True
except ImportError:
    HAS_MAXFLOW = False
    warnings.warn("maxflow library not found. QPBO optimization will not be available. "
                  "Install with: pip install PyMaxflow")


class MRFShadowDetector:
    """
    MRF-based Shadow Detection with Contextual Cues.
    
    This class implements the full MRF framework from Section 4 of the paper,
    combining region classifier predictions with pairwise contextual potentials.
    
    Reference:
        Paper Section 4: "We enhance shadow detection by embedding the shadow 
        region classifier in an MRF framework."
    
    Attributes:
        region_probs: Shadow probabilities for each region (from Platt scaling)
        region_areas: Area in pixels for each region
        adjacency: Region adjacency matrix
        kernel_similarities: Pairwise kernel similarities K(R_i, R_j)
        disparity_probs: Disparity classifier probabilities P^d
    """
    
    def __init__(
        self,
        affinity_threshold: float = 0.5,
        use_disparity: bool = True
    ):
        """
        Initialize MRF shadow detector.
        
        Args:
            affinity_threshold: Threshold for affinity potential (default: 0.5)
                               Paper: "K(R_i, R_j) > 0.5"
            use_disparity: Whether to use disparity potentials
        """
        self.affinity_threshold = affinity_threshold
        self.use_disparity = use_disparity
        
        # Data to be set
        self.n_regions = None
        self.region_probs = None      # P(x_i=1|R_i) from Platt scaling
        self.region_areas = None      # ω_i
        self.adjacency = None         # Neighboring region pairs
        self.kernel_similarities = None  # K(R_i, R_j)
        self.disparity_probs = None   # P^d(1|R_i, R_j)
        
        # Results
        self.labels_ = None
        self.energy_ = None
    
    def set_unary_data(
        self,
        region_probs: np.ndarray,
        region_areas: np.ndarray
    ) -> None:
        """
        Set data for unary potentials.
        
        Reference:
            Paper Section 4.1: "φ(x_i) = -ω_i · P(x_i|R_i), where ω_i is the 
            area in pixels of the region R_i, and P(x_i|R_i) is the Platt's 
            scaling probability."
        
        Args:
            region_probs: Shadow probabilities P(x=1|R) for each region, shape (n_regions,)
            region_areas: Area in pixels for each region, shape (n_regions,)
        """
        self.n_regions = len(region_probs)
        self.region_probs = np.asarray(region_probs, dtype=np.float64)
        self.region_areas = np.asarray(region_areas, dtype=np.float64)
    
    def set_adjacency(
        self,
        adjacency: np.ndarray
    ) -> None:
        """
        Set region adjacency matrix.
        
        Args:
            adjacency: Binary adjacency matrix (n_regions, n_regions)
                      adjacency[i,j] = 1 if regions i and j are neighbors
        """
        self.adjacency = np.asarray(adjacency, dtype=np.float64)
    
    def set_affinity_data(
        self,
        kernel_similarities: np.ndarray
    ) -> None:
        """
        Set data for affinity pairwise potentials.
        
        Reference:
            Paper Section 4.2: "The similarity metric between two regions R_i, R_j 
            is based on the kernel used for the single region classifier."
        
        Args:
            kernel_similarities: Kernel matrix K(R_i, R_j), shape (n_regions, n_regions)
        """
        self.kernel_similarities = np.asarray(kernel_similarities, dtype=np.float64)
    
    def set_disparity_data(
        self,
        disparity_probs: np.ndarray
    ) -> None:
        """
        Set data for disparity pairwise potentials.
        
        Reference:
            Paper Section 4.3: "For disparity potentials, we classify shadow/non-shadow 
            transitions between two regions of the same material."
        
        Args:
            disparity_probs: P^d(1|R_i, R_j) for each region pair, shape (n_regions, n_regions)
                            This should be the output of the disparity classifier.
        """
        self.disparity_probs = np.asarray(disparity_probs, dtype=np.float64)
    
    def compute_unary_potentials(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute unary potentials for all regions.
        
        Reference:
            Paper Section 4.1 (Equation for φ):
            "φ(x_i) = -ω_i · P(x_i|R_i)"
            
            "The unary potential encourages agreement between the label of a 
            region and the prediction based on the appearance of the region."
        
        Returns:
            unary_shadow: Potential for label=shadow (+1) for each region
            unary_nonshadow: Potential for label=non-shadow (-1) for each region
            
        Note:
            In MRF convention, lower energy = better.
            φ(x_i=+1) = -ω_i · P(shadow|R_i)
            φ(x_i=-1) = -ω_i · P(non-shadow|R_i) = -ω_i · (1 - P(shadow|R_i))
        """
        if self.region_probs is None or self.region_areas is None:
            raise RuntimeError("Unary data not set. Call set_unary_data() first.")
        
        # P(shadow) and P(non-shadow)
        p_shadow = self.region_probs
        p_nonshadow = 1.0 - self.region_probs
        
        # Unary potentials: φ(x_i) = -ω_i · P(x_i|R_i)
        # Lower is better, so high probability -> low (negative) potential
        unary_shadow = -self.region_areas * p_shadow
        unary_nonshadow = -self.region_areas * p_nonshadow
        
        return unary_shadow, unary_nonshadow
    
    def compute_affinity_potentials(self) -> np.ndarray:
        """
        Compute affinity pairwise potentials.
        
        Reference:
            Paper Section 4.2:
            "ψ_a(x_i, x_j) = { ω_ij · K(R_i, R_j)  if x_i ≠ x_j and K(R_i, R_j) > 0.5
                            { 0                    otherwise"
            
            "The penalty for having different shadow labels is the similarity 
            between the two regions weighted by the geometric mean of the areas 
            of the regions, i.e., ω_ij = √(ω_i · ω_j)"
        
        Returns:
            affinity: Matrix of affinity potentials, shape (n_regions, n_regions)
                     Only non-zero for neighboring pairs where K > threshold
                     
        Note:
            This potential is added when x_i ≠ x_j (different labels).
            It encourages similar regions to have the same label.
        """
        if self.kernel_similarities is None or self.adjacency is None:
            raise RuntimeError("Affinity data not set.")
        
        n = self.n_regions
        affinity = np.zeros((n, n))
        
        # Compute geometric mean of areas: ω_ij = √(ω_i · ω_j)
        omega = np.sqrt(np.outer(self.region_areas, self.region_areas))
        
        # Get neighboring pairs
        neighbors = np.where(self.adjacency > 0)
        
        for i, j in zip(neighbors[0], neighbors[1]):
            if i >= j:  # Only process each pair once
                continue
            
            K_ij = self.kernel_similarities[i, j]
            
            # Paper: "K(R_i, R_j) > 0.5"
            if K_ij > self.affinity_threshold:
                # ψ_a = ω_ij · K(R_i, R_j) when x_i ≠ x_j
                affinity[i, j] = omega[i, j] * K_ij
                affinity[j, i] = affinity[i, j]  # Symmetric
        
        return affinity
    
    def compute_disparity_potentials(self) -> np.ndarray:
        """
        Compute disparity pairwise potentials.
        
        Reference:
            Paper Section 4.3:
            "ψ_d(x_i, x_j) = { 0                          if x_i ≠ x_j
                            { ω_ij · P^d(1|R_i, R_j)     otherwise"
            
            "We penalize same shadow labeling for the pairs of regions that are 
            classified as positive by the learned classifier. The penalty is the 
            prediction confidence weighted by the geometric mean of the regions' areas."
        
        Returns:
            disparity: Matrix of disparity potentials, shape (n_regions, n_regions)
                      Only non-zero for neighboring pairs
                      
        Note:
            This potential is added when x_i = x_j (same labels).
            It encourages different labels for shadow/non-shadow boundary regions.
        """
        if self.disparity_probs is None or self.adjacency is None:
            raise RuntimeError("Disparity data not set.")
        
        n = self.n_regions
        disparity = np.zeros((n, n))
        
        # Compute geometric mean of areas
        omega = np.sqrt(np.outer(self.region_areas, self.region_areas))
        
        # Get neighboring pairs
        neighbors = np.where(self.adjacency > 0)
        
        for i, j in zip(neighbors[0], neighbors[1]):
            if i >= j:
                continue
            
            # ψ_d = ω_ij · P^d(1|R_i, R_j) when x_i = x_j
            P_d = self.disparity_probs[i, j]
            disparity[i, j] = omega[i, j] * P_d
            disparity[j, i] = disparity[i, j]
        
        return disparity
    
    def compute_energy(
        self,
        labels: np.ndarray,
        unary_shadow: np.ndarray,
        unary_nonshadow: np.ndarray,
        affinity: np.ndarray,
        disparity: np.ndarray
    ) -> float:
        """
        Compute total MRF energy for given labels.
        
        E = Σ_i φ(x_i) + Σ_{i,j∈Ω^a} ψ_a(x_i, x_j) + Σ_{i,j∈Ω^d} ψ_d(x_i, x_j)
        
        Args:
            labels: Binary labels, +1 for shadow, -1 for non-shadow
            unary_shadow, unary_nonshadow: Unary potentials
            affinity: Affinity pairwise potentials (applied when x_i ≠ x_j)
            disparity: Disparity pairwise potentials (applied when x_i = x_j)
            
        Returns:
            Total energy
        """
        n = len(labels)
        energy = 0.0
        
        # Unary term: Σ_i φ(x_i)
        for i in range(n):
            if labels[i] > 0:  # Shadow
                energy += unary_shadow[i]
            else:  # Non-shadow
                energy += unary_nonshadow[i]
        
        # Pairwise terms (only count each pair once)
        for i in range(n):
            for j in range(i + 1, n):
                if self.adjacency[i, j] > 0:
                    if labels[i] != labels[j]:
                        # Different labels: add affinity potential
                        energy += affinity[i, j]
                    else:
                        # Same labels: add disparity potential
                        energy += disparity[i, j]
        
        return energy
    
    def optimize_qpbo(
        self,
        unary_shadow: np.ndarray,
        unary_nonshadow: np.ndarray,
        affinity: np.ndarray,
        disparity: np.ndarray
    ) -> np.ndarray:
        """
        Optimize MRF energy using QPBO.
        
        Reference:
            Paper Section 4: "We optimize it using QPBO [18, 24]."
            
            QPBO can handle both submodular (affinity) and supermodular (disparity)
            pairwise potentials, but may leave some nodes unlabeled.
        
        Args:
            unary_shadow, unary_nonshadow: Unary potentials
            affinity: Affinity pairwise potentials (submodular, x_i ≠ x_j)
            disparity: Disparity pairwise potentials (supermodular, x_i = x_j)
            
        Returns:
            labels: Optimized labels (+1 for shadow, -1 for non-shadow)
        """
        if not HAS_MAXFLOW:
            raise RuntimeError("maxflow library not available. Install with: pip install PyMaxflow")
        
        n = self.n_regions
        
        # Create graph for QPBO
        # QPBO uses doubled graph representation
        g = maxflow.GraphFloat()
        nodes = g.add_nodes(n)
        
        # Add unary potentials (terminal edges)
        # In graph cut: source=shadow(1), sink=non-shadow(0)
        for i in range(n):
            # Capacity source->i: cost of labeling i as non-shadow
            # Capacity i->sink: cost of labeling i as shadow
            g.add_tedge(nodes[i], unary_nonshadow[i], unary_shadow[i])
        
        # Add pairwise potentials
        # For graph cut, we need to convert to submodular form
        # E(x_i, x_j) = E00 + (E01-E00)·x_i + (E10-E00)·x_j + (E00+E11-E01-E10)·x_i·x_j
        
        neighbors = np.where(self.adjacency > 0)
        for i, j in zip(neighbors[0], neighbors[1]):
            if i >= j:
                continue
            
            # Pairwise potentials:
            # E(0,0) = disparity[i,j]  (both non-shadow)
            # E(0,1) = affinity[i,j]   (i=non-shadow, j=shadow)
            # E(1,0) = affinity[i,j]   (i=shadow, j=non-shadow)
            # E(1,1) = disparity[i,j]  (both shadow)
            
            E00 = disparity[i, j]
            E01 = affinity[i, j]
            E10 = affinity[i, j]
            E11 = disparity[i, j]
            
            # Add edge (using standard graph cut formulation)
            # The interaction term is: (E00 + E11 - E01 - E10) * x_i * x_j
            # If this is non-negative (submodular), standard graph cut works
            # If negative (supermodular), QPBO can handle it
            
            interaction = E00 + E11 - E01 - E10
            
            if interaction >= 0:
                # Submodular: use standard graph cut edges
                g.add_edge(nodes[i], nodes[j], E01 - E00, E10 - E00)
            else:
                # Supermodular: QPBO handles this via auxiliary variables
                # For simplicity, we use an approximation
                g.add_edge(nodes[i], nodes[j], max(0, E01 - E00), max(0, E10 - E00))
        
        # Run max-flow/min-cut
        g.maxflow()
        
        # Get labels
        labels = np.zeros(n, dtype=np.int32)
        for i in range(n):
            # get_segment returns 0 for source (shadow) and 1 for sink (non-shadow)
            if g.get_segment(nodes[i]) == 0:
                labels[i] = 1   # Shadow
            else:
                labels[i] = -1  # Non-shadow
        
        return labels
    
    def optimize_icm(
        self,
        unary_shadow: np.ndarray,
        unary_nonshadow: np.ndarray,
        affinity: np.ndarray,
        disparity: np.ndarray,
        max_iter: int = 100,
        init_labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Optimize MRF energy using Iterated Conditional Modes (ICM).
        
        This is a simpler alternative to QPBO that doesn't require external libraries.
        
        Args:
            unary_shadow, unary_nonshadow: Unary potentials
            affinity: Affinity pairwise potentials
            disparity: Disparity pairwise potentials
            max_iter: Maximum number of iterations
            init_labels: Initial labels (default: use unary predictions)
            
        Returns:
            labels: Optimized labels
        """
        n = self.n_regions
        
        # Initialize labels based on unary potentials
        if init_labels is not None:
            labels = init_labels.copy()
        else:
            labels = np.where(unary_shadow < unary_nonshadow, 1, -1)
        
        # ICM iterations
        for iteration in range(max_iter):
            changed = False
            
            for i in range(n):
                # Compute energy for both labels
                energy_shadow = unary_shadow[i]
                energy_nonshadow = unary_nonshadow[i]
                
                # Add pairwise terms
                for j in range(n):
                    if self.adjacency[i, j] > 0:
                        # If i is shadow (+1) and j has different label
                        if labels[j] < 0:  # j is non-shadow
                            energy_shadow += affinity[i, j]      # Different labels
                            energy_nonshadow += disparity[i, j]  # Same labels
                        else:  # j is shadow
                            energy_shadow += disparity[i, j]     # Same labels
                            energy_nonshadow += affinity[i, j]   # Different labels
                
                # Choose label with lower energy
                new_label = 1 if energy_shadow < energy_nonshadow else -1
                if new_label != labels[i]:
                    labels[i] = new_label
                    changed = True
            
            if not changed:
                break
        
        return labels
    
    def optimize(
        self,
        method: str = 'qpbo'
    ) -> np.ndarray:
        """
        Optimize MRF to get final shadow labels.
        
        Args:
            method: Optimization method ('qpbo' or 'icm')
            
        Returns:
            labels: Optimized labels (+1 for shadow, -1 for non-shadow)
        """
        # Compute potentials
        unary_shadow, unary_nonshadow = self.compute_unary_potentials()
        affinity = self.compute_affinity_potentials()
        
        if self.use_disparity and self.disparity_probs is not None:
            disparity = self.compute_disparity_potentials()
        else:
            disparity = np.zeros_like(affinity)
        
        # Optimize
        if method == 'qpbo':
            if HAS_MAXFLOW:
                labels = self.optimize_qpbo(unary_shadow, unary_nonshadow, affinity, disparity)
            else:
                warnings.warn("QPBO not available, falling back to ICM")
                labels = self.optimize_icm(unary_shadow, unary_nonshadow, affinity, disparity)
        elif method == 'icm':
            labels = self.optimize_icm(unary_shadow, unary_nonshadow, affinity, disparity)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute final energy
        self.energy_ = self.compute_energy(
            labels, unary_shadow, unary_nonshadow, affinity, disparity
        )
        self.labels_ = labels
        
        return labels


class DisparityClassifier:
    """
    Disparity Classifier for shadow/non-shadow boundary detection.
    
    This classifier determines whether a pair of neighboring regions
    represents a shadow/non-shadow boundary.
    
    Reference:
        Paper Section 4.3:
        "For disparity potentials, we classify shadow/non-shadow transitions 
        between two regions of the same material. We train an LSSVM that takes 
        a pair of neighboring regions and predicts if the input is a shadow/
        non-shadow pair."
        
        Features (with RBF kernel):
        - The χ² distance between the texton histograms
        - The EMD between corresponding L*, a* and b* histograms of the two regions
        - The average RGB ratios: ρ_R = R_i/R_j, ρ_G = G_i/G_j, ρ_B = B_i/B_j
          and the feature vector is: ((ρ_R + ρ_G + ρ_B)/3, ρ_R/ρ_B, ρ_G/ρ_B)
          
    Note on RBF kernel γ parameter:
        The paper does NOT specify the γ (gamma) parameter for the RBF kernel.
        This is common practice - γ is typically chosen via cross-validation
        or using heuristics like:
        - γ = 1 / (n_features * X.var())  [scikit-learn style]
        - γ = 1 / median(pairwise_squared_distances)  [median heuristic]
        
        Impact on results: MODERATE-LOW
        - Disparity potentials only affect neighboring region pairs with same labels
        - Unary potentials (from main classifier) dominate the MRF energy
        - Wrong γ may cause some boundary detection errors but overall effect is limited
    """
    
    def __init__(self, gamma: Optional[float] = None, gamma_strategy: str = 'auto'):
        """
        Initialize disparity classifier.
        
        Args:
            gamma: RBF kernel parameter. If None, will be auto-determined.
            gamma_strategy: Strategy for auto gamma selection:
                - 'auto': Use 1/(n_features * X.var()) [default]
                - 'median': Use 1/median(squared_distances)
                - 'fixed': Use provided gamma value (default 1.0 if gamma is None)
                
        Note:
            Paper Section 4.3 does not specify γ. We use 'auto' strategy by default,
            which is a common heuristic that adapts to the feature scale.
        """
        self.gamma = gamma
        self.gamma_strategy = gamma_strategy
        self.lssvm = None
        self.platt_scaler = None
        self._is_fitted = False
        self._auto_gamma = None  # Store auto-computed gamma for reference
    
    def extract_pairwise_features(
        self,
        features_i: Dict[str, np.ndarray],
        features_j: Dict[str, np.ndarray],
        rgb_i: np.ndarray,
        rgb_j: np.ndarray
    ) -> np.ndarray:
        """
        Extract features for a region pair.
        
        Reference:
            Paper Section 4.3:
            "We use an RBF kernel with the following features:
            - The χ² distance between the texton histograms
            - The EMD between corresponding L*, a* and b* histograms of the two regions
            - The average RGB ratios"
        
        Args:
            features_i: Features for region i {'L': hist_L, 'a': hist_a, 'b': hist_b, 't': hist_t}
            features_j: Features for region j
            rgb_i: Mean RGB values for region i [R, G, B]
            rgb_j: Mean RGB values for region j [R, G, B]
            
        Returns:
            Feature vector for the pair
        """
        from .distances import chi_square_distance, emd_1d
        
        features = []
        
        # 1. χ² distance between texton histograms
        chi2_t = chi_square_distance(features_i['t'], features_j['t'])
        features.append(chi2_t)
        
        # 2. EMD between L*, a*, b* histograms
        emd_L = emd_1d(features_i['L'], features_j['L'])
        emd_a = emd_1d(features_i['a'], features_j['a'])
        emd_b = emd_1d(features_i['b'], features_j['b'])
        features.extend([emd_L, emd_a, emd_b])
        
        # 3. RGB ratios
        # ρ_R = R_i/R_j, ρ_G = G_i/G_j, ρ_B = B_i/B_j
        eps = 1e-10  # Avoid division by zero
        rho_R = rgb_i[0] / (rgb_j[0] + eps)
        rho_G = rgb_i[1] / (rgb_j[1] + eps)
        rho_B = rgb_i[2] / (rgb_j[2] + eps)
        
        # Feature vector: ((ρ_R + ρ_G + ρ_B)/3, ρ_R/ρ_B, ρ_G/ρ_B)
        avg_ratio = (rho_R + rho_G + rho_B) / 3
        ratio_RB = rho_R / (rho_B + eps)
        ratio_GB = rho_G / (rho_B + eps)
        features.extend([avg_ratio, ratio_RB, ratio_GB])
        
        return np.array(features)
    
    def _compute_auto_gamma(self, X: np.ndarray) -> float:
        """
        Compute gamma automatically based on the data.
        
        Reference:
            Since paper Section 4.3 does not specify γ, we use common heuristics:
            - 'auto': γ = 1 / (n_features * X.var())
            - 'median': γ = 1 / median(squared_pairwise_distances)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Computed gamma value
        """
        if self.gamma_strategy == 'auto':
            # scikit-learn style: scale-aware gamma
            # This adapts to the variance of the features
            n_features = X.shape[1]
            variance = X.var()
            if variance < 1e-10:
                variance = 1.0
            gamma = 1.0 / (n_features * variance)
            
        elif self.gamma_strategy == 'median':
            # Median heuristic: more robust to outliers
            from scipy.spatial.distance import pdist
            squared_distances = pdist(X, metric='sqeuclidean')
            median_dist = np.median(squared_distances)
            if median_dist < 1e-10:
                median_dist = 1.0
            gamma = 1.0 / median_dist
            
        else:  # 'fixed'
            gamma = self.gamma if self.gamma is not None else 1.0
        
        return gamma
    
    def fit(
        self,
        pair_features: np.ndarray,
        pair_labels: np.ndarray
    ) -> 'DisparityClassifier':
        """
        Fit the disparity classifier.
        
        Reference:
            Paper Section 4.3: "We train an LSSVM that takes a pair of 
            neighboring regions and predicts if the input is a shadow/
            non-shadow pair. We use an RBF kernel..."
            
            Note: γ is not specified in the paper, so we use auto-selection.
        
        Args:
            pair_features: Feature matrix for region pairs (n_pairs, n_features)
                          Shape: (n_pairs, 7) - 7 features as per paper
            pair_labels: Labels for pairs (+1 = shadow/non-shadow boundary, -1 = same type)
            
        Returns:
            self: Fitted classifier
        """
        from .lssvm import LSSVM
        from .kernels import RBFKernel
        from .platt_scaling import PlattScaler
        
        # Determine gamma
        if self.gamma is not None:
            gamma = self.gamma
        else:
            gamma = self._compute_auto_gamma(pair_features)
            self._auto_gamma = gamma
        
        # Train LSSVM with RBF kernel
        # Paper: "We use an RBF kernel with the following features"
        kernel = RBFKernel(gamma=gamma)
        self.lssvm = LSSVM(kernel=kernel, gamma=1.0)
        self.lssvm.fit(pair_features, pair_labels)
        
        # Fit Platt scaler for probability output
        # Paper implicitly requires P^d(1|R_i, R_j) which needs probability calibration
        decision_values = self.lssvm.decision_function(pair_features)
        self.platt_scaler = PlattScaler()
        self.platt_scaler.fit(decision_values, pair_labels)
        
        self._is_fitted = True
        return self
    
    def get_gamma(self) -> float:
        """Return the actual gamma value used (useful for 'auto' strategy)."""
        if self._auto_gamma is not None:
            return self._auto_gamma
        return self.gamma if self.gamma is not None else 1.0
    
    def predict_proba(self, pair_features: np.ndarray) -> np.ndarray:
        """
        Predict probability of shadow/non-shadow boundary.
        
        Args:
            pair_features: Feature matrix for region pairs
            
        Returns:
            Probabilities P^d(1|R_i, R_j)
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        decision_values = self.lssvm.decision_function(pair_features)
        proba = self.platt_scaler.predict_proba(decision_values)
        
        return proba


def compute_region_areas(region_labels: np.ndarray) -> np.ndarray:
    """
    Compute area (in pixels) for each region.
    
    Reference:
        Paper Section 4.1: "ω_i is the area in pixels of the region R_i"
    
    Args:
        region_labels: Region label map (H, W)
        
    Returns:
        areas: Area for each region, shape (n_regions,)
    """
    n_regions = int(region_labels.max()) + 1
    areas = np.zeros(n_regions)
    
    for i in range(n_regions):
        areas[i] = np.sum(region_labels == i)
    
    return areas


def compute_region_adjacency(region_labels: np.ndarray) -> np.ndarray:
    """
    Compute region adjacency matrix.
    
    Two regions are adjacent if they share at least one boundary pixel.
    
    Args:
        region_labels: Region label map (H, W)
        
    Returns:
        adjacency: Binary adjacency matrix (n_regions, n_regions)
    """
    n_regions = int(region_labels.max()) + 1
    adjacency = np.zeros((n_regions, n_regions), dtype=np.float64)
    
    H, W = region_labels.shape
    
    # Check horizontal neighbors
    for i in range(H):
        for j in range(W - 1):
            r1 = region_labels[i, j]
            r2 = region_labels[i, j + 1]
            if r1 != r2:
                adjacency[r1, r2] = 1
                adjacency[r2, r1] = 1
    
    # Check vertical neighbors
    for i in range(H - 1):
        for j in range(W):
            r1 = region_labels[i, j]
            r2 = region_labels[i + 1, j]
            if r1 != r2:
                adjacency[r1, r2] = 1
                adjacency[r2, r1] = 1
    
    return adjacency


def compute_region_mean_rgb(
    image: np.ndarray,
    region_labels: np.ndarray
) -> np.ndarray:
    """
    Compute mean RGB values for each region.
    
    Args:
        image: RGB image (H, W, 3) with values in [0, 1] or [0, 255]
        region_labels: Region label map (H, W)
        
    Returns:
        mean_rgb: Mean RGB for each region, shape (n_regions, 3)
    """
    if image.max() > 1.0:
        image = image.astype(np.float64) / 255.0
    
    n_regions = int(region_labels.max()) + 1
    mean_rgb = np.zeros((n_regions, 3))
    
    for i in range(n_regions):
        mask = region_labels == i
        if mask.any():
            mean_rgb[i] = image[mask].mean(axis=0)
    
    return mean_rgb

