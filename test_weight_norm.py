"""Test weight normalization constraint."""
from models.loo_optimizer import PaperBeamSearchOptimizer
import numpy as np

# Test weight normalization
opt = PaperBeamSearchOptimizer(verbose=False)

# Mock the sigma grids for testing
opt._sigma_grids = {
    'L': np.array([0.1, 0.2, 0.3]),
    'a': np.array([0.1, 0.2, 0.3]),
    'b': np.array([0.1, 0.2, 0.3]),
    't': np.array([0.1, 0.2, 0.3])
}

print("Testing weight normalization constraint...")

# Test random params
for i in range(5):
    params = opt._random_params()
    weight_sum = sum(params['weights'].values())
    print(f"  Trial {i+1}: weights = {params['weights']}")
    print(f"           sum = {weight_sum:.6f}")
    assert abs(weight_sum - 1.0) < 1e-10, f"Weights should sum to 1, got {weight_sum}"

print("\n[PASS] Weight normalization constraint verified!")
print("       (Paper: 'we constrain the kernel weights to be non-negative and have unit sum')")

