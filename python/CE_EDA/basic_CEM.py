import numpy as np

def objective_function(x):
    """Example cost function: minimize (x-3)^2 + 4"""
    return (x - 1)**2 * (x - 3)**2 + 4

# Cross-Entropy Method parameters
num_samples = 200        # samples per iteration
elite_frac = 0.2         # fraction of samples to keep as elites
num_iterations = 50
mean = 0.0               # initial mean of sampling distribution
std = 5.0                # initial standard deviation

for iteration in range(num_iterations):
    # 1. Sample from Gaussian distribution
    samples = np.random.normal(mean, std, size=num_samples)
    
    # 2. Evaluate the cost (lower is better)
    costs = objective_function(samples)
    
    # 3. Select elites (lowest cost)
    num_elites = int(elite_frac * num_samples)
    elite_indices = np.argsort(costs)[:num_elites]
    elite_samples = samples[elite_indices]
    
    # 4. Update mean and std based on elites
    mean = np.mean(elite_samples)
    std = np.std(elite_samples)
    
    # 5. Logging
    print(f"Iter {iteration+1}: mean={mean:.4f}, std={std:.4f}, best_cost={np.min(costs):.4f}")

print("\nEstimated optimum:", mean)

