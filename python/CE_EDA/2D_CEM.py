import numpy as np

def cost_function(x):
    # Soft constraint: low cost if all values are similar
    return np.var(x, axis=1)  # variance along each row

# Parameters
num_vars = 4
num_samples = 500
elite_frac = 0.2
num_iterations = 100

# Initial mean and covariance
mean = np.linspace(-5, 5, num_vars)   # start spread out
cov = np.eye(num_vars) * 5.0          # allow correlation

for iteration in range(num_iterations):
    # 1. Sample from multivariate Gaussian
    samples = np.random.multivariate_normal(mean, cov, size=num_samples)
    
    # 2. Evaluate cost
    costs = cost_function(samples)
    
    # 3. Select elites
    num_elites = int(elite_frac * num_samples)
    elite_indices = np.argsort(costs)[:num_elites]
    elites = samples[elite_indices]
    
    # 4. Update mean & covariance
    mean = np.mean(elites, axis=0)
    cov = np.cov(elites.T)
    
    # 5. Logging
    print(f"Iter {iteration+1}: mean[0:4]={mean[:4]}, cost_min={np.min(costs):.4f}")

print("\nEstimated optimum (4 vars shown):", mean[:4])

