import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Step 1: Define your inputs
# - Market data
returns = pd.read_csv('returns_data.csv', index_col=0)  # Replace 'returns_data.csv' with your data file
cov_matrix = returns.cov()
risk_free_rate = 0.03  # Replace with your risk-free rate

# - Investor views
# Replace the following arrays with your own views
views = np.array([0.02, -0.03, 0.05])
view_assets = np.array([0, 2, 3])
view_confidences = np.array([0.6, 0.8, 0.7])

# Step 2: Perform the Black-Litterman model

# - Equilibrium returns
market_weights = np.ones(len(returns.columns)) / len(returns.columns)
equilibrium_returns = np.dot(cov_matrix, market_weights)

# - Investor views adjustment
tau = 0.025  # Constant scaling factor
omega = np.diag(view_confidences ** 2)  # Uncertainty matrix

# Calculate P, Q matrices
P = np.zeros((len(view_assets), len(returns.columns)))
for i, asset in enumerate(view_assets):
    P[i, asset] = 1.0

Q = np.array(views).reshape(-1, 1)

# Calculate combined returns and covariance matrix
combined_returns = np.vstack((equilibrium_returns.reshape(-1, 1), np.dot(np.dot(P, cov_matrix), P.T)))
combined_cov_matrix = tau * np.dot(np.dot(P, cov_matrix), P.T) + omega

# Step 3: Perform optimization

def objective(x):
    return -np.dot(x, combined_returns)

def constraint(x):
    return np.dot(x, np.ones(len(returns.columns))) - 1

x0 = np.ones(len(returns.columns)) / len(returns.columns)
bounds = [(0, 1) for _ in range(len(returns.columns))]
constraints = ({'type': 'eq', 'fun': constraint})

opt_results = minimize(objective, x0, bounds=bounds, constraints=constraints)
weights = opt_results.x

# Step 4: Print results

print("Asset Weights:")
for i, asset in enumerate(returns.columns):
    print(f"{asset}: {weights[i]}")
