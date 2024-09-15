import numpy as np
import pandas as pd
from scipy.optimize import minimize

def mean_semivariance_optimizer(returns, target_return):
    n_assets = returns.shape[1]
    
    def objective(weights):
        portfolio_return = np.dot(weights, returns.mean())
        neg_semivariance = -np.dot(weights, np.minimum(returns - target_return, 0) ** 2)
        return neg_semivariance
    
    def constraint(weights):
        return np.sum(weights) - 1.0
    
    initial_weights = np.ones(n_assets) / n_assets
    bounds = [(0, 1) for _ in range(n_assets)]
    constraints = ({'type': 'eq', 'fun': constraint})
    
    opt_result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return opt_result.x

# Step 1: Define your inputs
returns = pd.read_csv('returns_data.csv', index_col=0)  # Replace 'returns_data.csv' with your data file
target_return = 0.05  # Replace with your target return

# Step 2: Perform mean-semivariance optimization
weights = mean_semivariance_optimizer(returns, target_return)

# Step 3: Print results
print("Asset Weights:")
for i, asset in enumerate(returns.columns):
    print(f"{asset}: {weights[i]}")
