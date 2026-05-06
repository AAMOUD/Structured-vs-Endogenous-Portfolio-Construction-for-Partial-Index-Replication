import pandas as pd

from backtest.backtest_engine import BacktestEngine

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
models = {
    "Stratified":   lambda K, sectors=None: StratifiedModel(K, sectors, market_caps),
    "MarketCap":    lambda K, sectors=None: MarketCapModel(K, market_caps),
    "Contribution": ContributionModel,
    "LASSO":        LassoModel,
    "MIQP_Gurobi":  MIQPModel,
    "Layered":      lambda K, sectors=None: LayeredOptimization(K, sectors, market_caps),
}

# ── K range ───────────────────────────────────────────────────────────────── #
K_list = [75, 100, 125, 150, 175, 200]

# ── Single train/eval split ───────────────────────────────────────────────── #
# 3-year training (756 days), 1-quarter eval (63 days).
# Use this as a quick sanity check. The rolling backtest is the primary result.
TRAIN_LENGTH_DAYS = 252 * 3
EVAL_LENGTH_DAYS  = 63

engine  = BacktestEngine(returns, index_returns, sectors, market_caps)
results = engine.run(
    models,
    K_list,
    train_length=TRAIN_LENGTH_DAYS,
    eval_length=EVAL_LENGTH_DAYS,
)

print(results[["model", "K", "TE_annual", "TE_annual_oos", "asset_active_share"]].to_string())