"""Backtesting framework for portfolio strategies."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from config import BACKTEST_PARAMS


class BacktestEngine:
    """Engine for backtesting portfolio strategies."""

    def __init__(
        self,
        index_prices: pd.Series,
        constituents_prices: pd.DataFrame,
        index_returns: pd.Series,
        constituents_returns: pd.DataFrame,
        slippage: float = BACKTEST_PARAMS["slippage"],
    ):
        """
        Initialize backtesting engine.

        Args:
            index_prices: Series of index prices
            constituents_prices: DataFrame of constituent prices
            index_returns: Series of index returns
            constituents_returns: DataFrame of constituent returns
            slippage: Transaction cost (as percentage)
        """
        self.index_prices = index_prices
        self.constituents_prices = constituents_prices
        self.index_returns = index_returns
        self.constituents_returns = constituents_returns
        self.slippage = slippage

        self.portfolio_values = None
        self.tracking_error = None
        self.daily_weights = None

    def backtest(
        self,
        selected_assets: List[str],
        weights: pd.Series,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict:
        """
        Run backtest for a portfolio strategy.

        Args:
            selected_assets: List of selected ticker symbols
            weights: Series of portfolio weights
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Dictionary with backtest results
        """
        # Prepare data
        selected_returns = self.constituents_returns[selected_assets]
        index_returns = self.index_returns

        # Filter by dates
        if start_date:
            selected_returns = selected_returns[selected_returns.index >= start_date]
            index_returns = index_returns[index_returns.index >= start_date]
        if end_date:
            selected_returns = selected_returns[selected_returns.index <= end_date]
            index_returns = index_returns[index_returns.index <= end_date]

        # Align dates
        common_dates = selected_returns.index.intersection(index_returns.index)
        selected_returns = selected_returns.loc[common_dates]
        index_returns = index_returns.loc[common_dates]

        # Compute portfolio returns
        portfolio_returns = (selected_returns @ weights).values
        
        # Apply slippage (one-time at the beginning)
        portfolio_returns_with_slippage = portfolio_returns.copy()
        portfolio_returns_with_slippage[0] -= self.slippage

        # Compute cumulative values (starting at 1)
        portfolio_values = (1 + portfolio_returns_with_slippage).cumprod()
        index_values = (1 + index_returns.values).cumprod()

        self.portfolio_values = pd.Series(portfolio_values, index=common_dates)
        
        # Compute tracking error
        tracking_error_daily = portfolio_returns - index_returns.values
        tracking_error_annual = np.std(tracking_error_daily) * np.sqrt(252)

        # Compute metrics
        portfolio_return = (portfolio_values[-1] - 1) * 100
        index_return = (index_values[-1] - 1) * 100
        mean_tracking_error = np.mean(tracking_error_daily) * 252 * 100
        
        # Compute Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (portfolio_returns.mean() * 252) / (np.std(portfolio_returns) * np.sqrt(252))

        results = {
            "portfolio_value": portfolio_values[-1],
            "index_value": index_values[-1],
            "portfolio_return": portfolio_return,
            "index_return": index_return,
            "excess_return": portfolio_return - index_return,
            "tracking_error": tracking_error_annual,
            "mean_tracking_error": mean_tracking_error,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self._compute_max_drawdown(portfolio_values),
            "num_assets": len(selected_assets),
            "start_date": common_dates[0],
            "end_date": common_dates[-1],
            "num_periods": len(common_dates),
            "portfolio_values_series": self.portfolio_values,
            "index_values_series": pd.Series(index_values, index=common_dates),
        }

        return results

    def backtest_index(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict:
        """
        Backtest the index itself (buy-and-hold benchmark).

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Dictionary with backtest results
        """
        index_returns = self.index_returns.copy()

        # Filter by dates
        if start_date:
            index_returns = index_returns[index_returns.index >= start_date]
        if end_date:
            index_returns = index_returns[index_returns.index <= end_date]

        # Compute cumulative values (starting at 1)
        index_values = (1 + index_returns.values).cumprod()
        
        # Compute metrics
        index_return = (index_values[-1] - 1) * 100
        sharpe_ratio = (index_returns.mean() * 252) / (np.std(index_returns) * np.sqrt(252))

        results = {
            "portfolio_value": index_values[-1],
            "index_value": index_values[-1],
            "portfolio_return": index_return,
            "index_return": index_return,
            "excess_return": 0.0,
            "tracking_error": 0.0,
            "mean_tracking_error": 0.0,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self._compute_max_drawdown(index_values),
            "num_assets": len(self.constituents_returns.columns),  # All constituents
            "start_date": index_returns.index[0],
            "end_date": index_returns.index[-1],
            "num_periods": len(index_returns),
            "portfolio_values_series": pd.Series(index_values, index=index_returns.index),
            "index_values_series": pd.Series(index_values, index=index_returns.index),
        }

        return results

        results = {
            "portfolio_value": portfolio_values[-1],
            "index_value": index_values[-1],
            "portfolio_return": portfolio_return,
            "index_return": index_return,
            "excess_return": portfolio_return - index_return,
            "tracking_error": tracking_error_annual,
            "mean_tracking_error": mean_tracking_error,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self._compute_max_drawdown(portfolio_values),
            "num_assets": len(selected_assets),
            "start_date": common_dates[0],
            "end_date": common_dates[-1],
            "num_periods": len(common_dates),
        }

        return results

    def _compute_max_drawdown(self, values: np.ndarray) -> float:
        """
        Compute maximum drawdown.

        Args:
            values: Cumulative portfolio values

        Returns:
            Maximum drawdown as percentage
        """
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return np.min(drawdown) * 100


class StrategyComparator:
    """Compares multiple portfolio strategies."""

    def __init__(self):
        """Initialize strategy comparator."""
        self.results = {}

    def add_result(self, strategy_name: str, backtest_result: Dict) -> None:
        """
        Add backtest result for a strategy.

        Args:
            strategy_name: Name of the strategy
            backtest_result: Dictionary with backtest results
        """
        self.results[strategy_name] = backtest_result

    def get_comparison_df(self) -> pd.DataFrame:
        """
        Get comparison of all strategies as DataFrame.

        Returns:
            DataFrame with key metrics for each strategy
        """
        comparison_data = {}
        
        for strategy_name, result in self.results.items():
            comparison_data[strategy_name] = {
                "Portfolio Return (%)": result["portfolio_return"],
                "Index Return (%)": result["index_return"],
                "Excess Return (%)": result["excess_return"],
                "Tracking Error (%)": result["tracking_error"],
                "Mean Tracking Error (%)": result["mean_tracking_error"],
                "Sharpe Ratio": result["sharpe_ratio"],
                "Max Drawdown (%)": result["max_drawdown"],
                "Num Assets": result["num_assets"],
            }

        return pd.DataFrame(comparison_data).T

    def get_summary(self) -> str:
        """Get text summary of strategy comparison."""
        df = self.get_comparison_df()
        return df.to_string()

    def print_summary(self) -> None:
        """Print summary of strategy comparison."""
        print(self.get_summary())
