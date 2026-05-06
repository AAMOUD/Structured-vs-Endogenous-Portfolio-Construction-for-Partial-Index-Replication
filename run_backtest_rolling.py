import pandas as pd

from backtest.backtest_engine_rolling import BacktestEngineRolling

from models.stratified import StratifiedModel
from models.market_cap import MarketCapModel
from models.index_contribution import ContributionModel
from models.lasso import LassoModel
from models.miqp_gurobi import MIQPModel
from models.layered_model import LayeredOptimization

# ── Data ─────────────────────────────────────────────────────────────────── #
returns       = pd.read_parquet("data/returns.parquet")
index_returns = pd.read_parquet("data/index_returns.parquet")
sectors       = pd.read_csv("data/sp500_tickers.csv").set_index("Symbol")["GICS Sector"]
market_caps   = pd.read_csv("data/market_caps.csv", index_col=0)["market_cap"]

# ── 6 core models ─────────────────────────────────────────────────────────── #
# 3 heuristics  (no covariance, fast, baseline)
# 3 optimisation (LASSO, MIQP, Layered — the thesis contribution)
# Sector variants are dropped from the main comparison; run separately if needed.
models = {
    "Stratified":   lambda K, sectors=None: StratifiedModel(K, sectors, market_caps),
    "MarketCap":    lambda K, sectors=None: MarketCapModel(K, market_caps),
    "Contribution": ContributionModel,
    "LASSO":        LassoModel,
    "MIQP_Gurobi":  MIQPModel,
    "Layered":      lambda K, sectors=None: LayeredOptimization(K, sectors, market_caps),
}

# ── K range ───────────────────────────────────────────────────────────────── #
# K=50 dropped: too few assets, TE is structurally high and uninteresting.
# Thesis focus: K=125 and K=150.
K_list = [75, 100, 125, 150, 175, 200]

# ── Training window ───────────────────────────────────────────────────────── #
# 504 days = 2 calendar years of trading days.
# With ~480 assets in universe, T/N ≈ 1.05 — covariance is estimable.
# Below 504 (e.g. 252) the covariance is singular and MIQP/LASSO overfit badly.
TRAIN_LENGTH_DAYS = 504

engine = BacktestEngineRolling(returns, index_returns, sectors, market_caps)
results = engine.run(
    models,
    K_list,
    output_dir="results_rolling",
    train_length=TRAIN_LENGTH_DAYS,
)

# ── Quick summary print ───────────────────────────────────────────────────── #
print("\n── Per-quarter results ──")
print(results[["model", "K", "quarter", "TE_annual_insample", "TE_annual_oos"]].to_string())

print("\n── Full-period OOS TE (primary thesis metric) ──")
fp = pd.read_csv("results_rolling/summaries/full_period_metrics.csv")
print(fp[["model", "K", "TE_annual_oos_full", "IR_full_period", "n_oos_days"]].to_string())