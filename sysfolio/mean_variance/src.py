import numpy as np
import pandas as pd
from scipy.optimize import minimize

def mean_variance_optimizer(returns, risk_free_rate=0.0):
    n_assets = returns.shape[1]

    def objective(weights):
        # Calculate portfolio expected return
        portfolio_return = np.dot(weights, returns.mean())

        # Calculate portfolio risk (standard deviation)
        portfolio_std = np.sqrt(np.dot(weights, np.dot(returns.cov(), weights)))

        # Calculate portfolio Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std

        return -sharpe_ratio  # Maximize the negative Sharpe ratio

    def constraint(weights):
        return np.sum(weights) - 1.0  # Constraint: sum of weights equals 1

    # Initial guess for weights
    initial_weights = np.ones(n_assets) / n_assets

    # Define bounds for weights (between 0 and 1)
    bounds = [(0, 1) for _ in range(n_assets)]

    # Define constraint
    constraints = ({'type': 'eq', 'fun': constraint})

    # Perform optimization
    opt_result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return opt_result.x

# Step 1: Define your inputs
returns = pd.read_csv('returns_data.csv', index_col=0)  # Replace 'returns_data.csv' with your data file
risk_free_rate = 0.03  # Replace with your risk-free rate

# Step 2: Perform mean-variance optimization
weights = mean_variance_optimizer(returns, risk_free_rate)

# Step 3: Print results
print("Asset Weights:")
for i, asset in enumerate(returns.columns):
    print(f"{asset}: {weights[i]}")
