"""
run_backtest_sub1pct.py
=======================
Ablation study: LayeredOptimizationV2 with vs without Ledoit-Wolf shrinkage.

Purpose in thesis
─────────────────
This script isolates the contribution of LW shrinkage to TE reduction.
It is NOT a full model comparison — that is run_backtest_rolling.py.
Run this AFTER the main rolling backtest so you have a baseline to compare.

Results land in results_rolling_v2/ to avoid overwriting the main results.

What to report in thesis
─────────────────────────
From full_period_metrics.csv, compare:
    Layered_LW   vs Layered_noLW  at K=125 and K=150
The delta in TE_annual_oos_full is the measured contribution of LW shrinkage.

Expected outcome (with clean data + 504-day window)
────────────────────────────────────────────────────
    K=100  LW  →  OOS TE ~0.9–1.3%
    K=150  LW  →  OOS TE ~0.7–1.0%
    K=200  LW  →  OOS TE ~0.5–0.8%
"""

import pandas as pd

from backtest.backtest_engine_rolling import BacktestEngineRolling
from models.layered_model_v2 import LayeredOptimizationV2

# ── Data ─────────────────────────────────────────────────────────────────── #
returns       = pd.read_parquet("data/returns.parquet")
index_returns = pd.read_parquet("data/index_returns.parquet")
sectors       = pd.read_csv("data/sp500_tickers.csv").set_index("Symbol")["GICS Sector"]
market_caps   = pd.read_csv("data/market_caps.csv", index_col=0)["market_cap"]

# ── Models — LW on vs off (ablation) ─────────────────────────────────────── #
models = {
    "Layered_LW": lambda K, sectors=None: LayeredOptimizationV2(
        K, sectors, market_caps,
        use_lw_shrinkage=True,
        time_limit=300,
        mip_gap=0.003,
    ),
    "Layered_noLW": lambda K, sectors=None: LayeredOptimizationV2(
        K, sectors, market_caps,
        use_lw_shrinkage=False,
        time_limit=240,
        mip_gap=0.003,
    ),
}

# ── K range ───────────────────────────────────────────────────────────────── #
# K=75 excluded: sub-1% OOS TE is structurally unreachable at that size.
# Focus on K=125 and K=150 for the thesis — consistent with main backtest.
K_list = [100, 125, 150, 175, 200]

# ── Run ───────────────────────────────────────────────────────────────────── #
engine  = BacktestEngineRolling(returns, index_returns, sectors, market_caps)
results = engine.run(
    models,
    K_list,
    output_dir="results_rolling_v2",
    train_length=504,
)

# ── Summary ───────────────────────────────────────────────────────────────── #
print("\n── Per-quarter results ──")
print(results[["model", "K", "quarter", "TE_annual_insample", "TE_annual_oos"]].to_string())

print("\n── Full-period OOS TE (primary thesis metric) ──")
fp = pd.read_csv("results_rolling_v2/summaries/full_period_metrics.csv")
print(fp[["model", "K", "TE_annual_oos_full", "IR_full_period", "n_oos_days"]].to_string())

print("\n── LW shrinkage impact (delta TE) ──")
pivot = fp.pivot(index="K", columns="model", values="TE_annual_oos_full")
if "Layered_LW" in pivot.columns and "Layered_noLW" in pivot.columns:
    pivot["delta_TE (noLW - LW)"] = pivot["Layered_noLW"] - pivot["Layered_LW"]
    print(pivot.to_string())