import pandas as pd

from backtest.backtest_engine_rolling import BacktestEngineRolling

from models.stratified import StratifiedModel, StratifiedSectorModel
from models.market_cap import MarketCapModel, MarketCapSectorModel
from models.index_contribution import ContributionModel, ContributionSectorModel
from models.lasso import LassoModel, LassoSectorModel
from models.miqp_gurobi import MIQPModel
from models.layered_model import LayeredOptimization

# ── Data ────────────────────────────────────────────────────────────────── #
returns       = pd.read_parquet("data/returns.parquet")
index_returns = pd.read_parquet("data/index_returns.parquet")
sectors       = pd.read_csv("data/sp500_tickers.csv").set_index("Symbol")["GICS Sector"]
market_caps   = pd.read_csv("data/market_caps.csv", index_col=0)["market_cap"]

# ── Models ───────────────────────────────────────────────────────────────── #
models = {
    "Stratified":          lambda K, sectors=None: StratifiedModel(K, sectors, market_caps),
    "Stratified_sector":   lambda K, sectors=None: StratifiedSectorModel(K, sectors, market_caps),
    "MarketCap":           lambda K, sectors=None: MarketCapModel(K, market_caps),
    "MarketCap_sector":    lambda K, sectors=None: MarketCapSectorModel(K, market_caps, sectors),
    "Contribution":        ContributionModel,
    "Contribution_sector": lambda K, sectors=None: ContributionSectorModel(K, sectors, market_caps),
    "LASSO":               LassoModel,
    "LASSO_sector":        lambda K, sectors=None: LassoSectorModel(K, sectors, market_caps),
    "MIQP_Gurobi":         MIQPModel,
    "Layered":             lambda K, sectors=None: LayeredOptimization(K, sectors, market_caps),
}

K_list = [50, 75, 100, 125, 150, 175, 200]

# ── Training window ──────────────────────────────────────────────────────── #
# 504 days (2 years) brings T/N from 0.55 to 1.10.
# This is the primary lever that fixes overfitting for all optimisation models.
# Heuristics (Stratified, MarketCap) are unaffected — they don't use the
# covariance matrix — so the 504-day window only helps the models that need it.
#
# Note: the first evaluable quarter shifts from Q1-2023 to Q1-2024 because
# 504 training days require data starting ~2 years before the eval period.
# If your data starts in 2010 this is fine; the engine will skip quarters
# that lack sufficient history automatically.
TRAIN_LENGTH_DAYS = 126

engine = BacktestEngineRolling(returns, index_returns, sectors, market_caps)
results = engine.run(
    models,
    K_list,
    output_dir="results_rolling",
    train_length=TRAIN_LENGTH_DAYS,
)

print(results[["model", "K", "quarter", "TE_annual_insample", "TE_annual_oos"]].to_string())