# Configuration file for Index Replication Project

# Data parameters
DATA_PARAMS = {
    "index": "^GSPC",  # S&P 500
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "data_source": "yfinance",  # Using yfinance for fetching data
}

# Portfolio parameters
PORTFOLIO_PARAMS = {
    "K": 50,  # Number of assets to select (subset size)
    "num_strata": 10,  # Number of sectors for stratified sampling
    "num_clusters": 10,  # Number of clusters for clustering method
}

# Optimization parameters
OPTIMIZATION_PARAMS = {
    "lower_bound": 0.0,  # Minimum weight
    "upper_bound": 0.1,  # Maximum weight (10% per asset)
    "tolerance": 1e-6,
}

# Backtesting parameters
BACKTEST_PARAMS = {
    "rebalance_freq": "quarterly",  # quarterly, monthly, annual
    "split_ratio": 0.7,  # 70% train, 30% test
    "slippage": 0.001,  # 0.1% trading slippage
}
