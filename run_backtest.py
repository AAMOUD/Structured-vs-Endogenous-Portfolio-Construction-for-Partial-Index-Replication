import pandas as pd

from backtest.backtest_engine import BacktestEngine

from models.stratified import StratifiedModel, StratifiedSectorModel
from models.market_cap import MarketCapModel, MarketCapSectorModel
from models.index_contribution import ContributionModel, ContributionSectorModel
from models.lasso import LassoModel, LassoSectorModel
from models.miqp_gurobi import MIQPModel
from models.layered_model import LayeredOptimization


returns = pd.read_parquet("data/returns.parquet")

index_returns = pd.read_parquet("data/index_returns.parquet")

sectors = pd.read_csv("data\\sp500_tickers.csv")
sectors = sectors.set_index("Symbol")["GICS Sector"]

market_caps = pd.read_csv("data/market_caps.csv", index_col=0)
market_caps = market_caps["market_cap"]


models = {

    "Stratified":        lambda K, sectors=None: StratifiedModel(K, sectors, market_caps),
    "Stratified_sector": lambda K, sectors=None: StratifiedSectorModel(K, sectors, market_caps),
    "MarketCap":         lambda K, sectors=None: MarketCapModel(K, market_caps),
    "MarketCap_sector":  lambda K, sectors=None: MarketCapSectorModel(K, market_caps, sectors),
    "Contribution":      ContributionModel,
    "Contribution_sector": lambda K, sectors=None: ContributionSectorModel(K, sectors, market_caps),
    "LASSO":             LassoModel,
    "LASSO_sector":      lambda K, sectors=None: LassoSectorModel(K, sectors, market_caps),
    "MIQP_Gurobi":       MIQPModel,
    "Layered":           lambda K, sectors=None: LayeredOptimization(K, sectors, market_caps)
}


K_list = [50,75,100,125,150,175,200]

TRAIN_LENGTH_DAYS = 252 * 3
EVAL_LENGTH_DAYS = 63


engine = BacktestEngine(returns, index_returns, sectors, market_caps)

results = engine.run(
    models,
    K_list,
    train_length=TRAIN_LENGTH_DAYS,
    eval_length=EVAL_LENGTH_DAYS,
)