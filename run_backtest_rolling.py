import pandas as pd

from backtest.backtest_engine_rolling import BacktestEngineRolling

from models.stratified import StratifiedModel, StratifiedSectorModel
from models.market_cap import MarketCapModel, MarketCapSectorModel
from models.lasso import LassoModel, LassoSectorModel
from models.miqp_gurobi import MIQPModel
from models.layered_model import LayeredOptimization


# ── Data loading ────────────────────────────────────────────────────────── #
returns = pd.read_parquet("data/returns.parquet")
index_returns = pd.read_parquet("data/index_returns.parquet")

sectors = pd.read_csv("data/sp500_tickers.csv")
sectors = sectors.set_index("Symbol")["GICS Sector"]

market_caps = pd.read_csv("data/market_caps.csv", index_col=0)
market_caps = market_caps["market_cap"]


# ── Models  (identical set to run_backtest.py) ──────────────────────────── #
models = {
    "Stratified":        lambda K, sectors=None: StratifiedModel(K, sectors, market_caps),
    "Stratified_sector": lambda K, sectors=None: StratifiedSectorModel(K, sectors, market_caps),
    "MarketCap":         lambda K, sectors=None: MarketCapModel(K, market_caps),
    "MarketCap_sector":  lambda K, sectors=None: MarketCapSectorModel(K, market_caps, sectors),
    "LASSO":             LassoModel,
    "LASSO_sector":      lambda K, sectors=None: LassoSectorModel(K, sectors, market_caps),
    "MIQP_Gurobi":       MIQPModel,
    "Layered":           lambda K, sectors=None: LayeredOptimization(K, sectors, market_caps),
}

# ── Portfolio sizes ─────────────────────────────────────────────────────── #
K_list = [50, 75, 100, 125, 150, 175, 200]

# ── Run rolling backtest ────────────────────────────────────────────────── #
#   Training window : 252 trading days (≈ 1 year) before each quarter.
#   First quarter   : Q1 2023  (trained on 2022).
#   Last quarter    : last complete quarter in the available return data.
#   Output          : results_rolling/
engine = BacktestEngineRolling(returns, index_returns, sectors, market_caps)
results = engine.run(models, K_list, output_dir="results_rolling")

print(results)
