"""
run_backtest_sub1pct.py
=======================
Dedicated run script for the sub-1% TE experiment using:
    - LayeredOptimizationV2 (Ledoit-Wolf shrinkage)
    - 504-day training window (2 years)
    - K in {100, 150, 200}  (smaller K cannot realistically reach <1% OOS)

Run this AFTER the standard backtest has completed so you have a baseline.
Results land in results_rolling_v2/ to avoid overwriting the main results.

Expected outcome
────────────────
    K=100 + LW + 504 days  →  IS TE ~0.5-0.7%,  OOS TE ~0.9-1.3%
    K=150 + LW + 504 days  →  IS TE ~0.4-0.6%,  OOS TE ~0.7-1.0%
    K=200 + LW + 504 days  →  IS TE ~0.3-0.5%,  OOS TE ~0.6-0.9%

Comparison columns to add to your thesis Table (results_rolling vs results_rolling_v2):
    model | K | train_days | TE_annual_insample | TE_annual_oos | lw_delta
"""

import pandas as pd

from backtest.backtest_engine_rolling import BacktestEngineRolling
from models.layered_model_v2 import LayeredOptimizationV2

returns       = pd.read_parquet("data/returns.parquet")
index_returns = pd.read_parquet("data/index_returns.parquet")

sectors      = pd.read_csv("data/sp500_tickers.csv").set_index("Symbol")["GICS Sector"]
market_caps  = pd.read_csv("data/market_caps.csv", index_col=0)["market_cap"]

models = {
    "Layered_LW":          lambda K, sectors=None: LayeredOptimizationV2(
                               K, sectors, market_caps,
                               use_lw_shrinkage=True,
                               time_limit=300,
                               mip_gap=0.003,
                           ),
    "Layered_noLW":        lambda K, sectors=None: LayeredOptimizationV2(
                               K, sectors, market_caps,
                               use_lw_shrinkage=False,  # control: same model, no shrinkage
                               time_limit=240,
                               mip_gap=0.003,
                           ),
}

K_list = [100, 150, 200]

engine = BacktestEngineRolling(returns, index_returns, sectors, market_caps)
results = engine.run(
    models,
    K_list,
    output_dir="results_rolling_v2",
    train_length=504,   # 2-year window — THE main lever beyond K
)

print(results[["model", "K", "quarter", "TE_annual_insample", "TE_annual_oos"]].to_string())