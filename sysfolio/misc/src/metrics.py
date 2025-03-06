import matplotlib.pyplot as plt
import numpy as np

def calculate_portfolio_metrics(portfolio_returns, risk_free_rate=0.02):
    """
    Calculate key portfolio performance metrics
    
    Args:
        portfolio_returns: DataFrame of portfolio returns
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        dict: Performance metrics including:
        - Sharpe Ratio
        - Sortino Ratio  
        - Maximum Drawdown
        - Information Ratio
    """
    # Annualized metrics
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol
    
    # Maximum Drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.expanding(min_periods=1).max()
    drawdowns = cumulative / rolling_max - 1.0
    max_drawdown = drawdowns.min()
    
    return {
        'sharpe_ratio': sharpe,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'max_drawdown': max_drawdown
    }

def plot_performance_comparison(boltzmann_returns, benchmark_returns):
    """Plot cumulative returns comparison"""
    plt.figure(figsize=(12,6))
    ((1 + boltzmann_returns).cumprod()).plot(label='Boltzmann Portfolio')
    ((1 + benchmark_returns).cumprod()).plot(label='Benchmark')
    plt.title('Cumulative Return Comparison')
    plt.legend()
    plt.show()