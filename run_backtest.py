import pandas as pd

from backtest.backtest_engine import BacktestEngine

from models.stratified import StratifiedModel
from models.correlation import CorrelationModel
from models.clustering import ClusteringModel
from models.lasso import LassoModel
from models.miqp_gurobi import MIQPModel
from models.layered_model import LayeredOptimization


returns = pd.read_parquet("data/returns.parquet")

index_returns = pd.read_parquet("data/index_returns.parquet")

sectors = pd.read_csv("data\\sp500_tickers.csv")
sectors = sectors.set_index("Symbol")["GICS Sector"]

market_caps = pd.read_csv("data/market_caps.csv", index_col=0)
market_caps = market_caps["market_cap"]


models = {

    "Stratified": StratifiedModel,
    "Correlation": CorrelationModel,
    "Clustering": ClusteringModel,
    "MIQP_Gurobi": MIQPModel,
    "LASSO": LassoModel,
    "Layered": LayeredOptimization
}


K_list = [50,75,100,125,150,175,200]


engine = BacktestEngine(returns, index_returns, sectors, market_caps)

results = engine.run(models, K_list)