"""Weight optimization for portfolio construction."""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict
from config import OPTIMIZATION_PARAMS


class PortfolioOptimizer:
    """Optimizes portfolio weights to minimize tracking error."""

    def __init__(
        self,
        lower_bound: float = OPTIMIZATION_PARAMS["lower_bound"],
        upper_bound: float = OPTIMIZATION_PARAMS["upper_bound"],
        tolerance: float = OPTIMIZATION_PARAMS["tolerance"],
    ):
        """
        Initialize optimizer.

        Args:
            lower_bound: Minimum weight for each asset
            upper_bound: Maximum weight for each asset
            tolerance: Optimization tolerance
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.tolerance = tolerance
        self.optimal_weights = None
        self.optimization_result = None

    def optimize(
        self,
        selected_returns: pd.DataFrame,
        index_returns: pd.Series,
    ) -> pd.Series:
        """
        Optimize portfolio weights using quadratic programming.

        Minimizes: ||I - R_S @ w||^2
        Subject to: sum(w) = 1, l_i <= w_i <= u_i

        Args:
            selected_returns: DataFrame of returns for selected assets
            index_returns: Series of index returns

        Returns:
            Series of optimal weights indexed by asset ticker
        """
        num_assets = selected_returns.shape[1]
        assets = selected_returns.columns.tolist()

        # Align dates
        common_dates = selected_returns.index.intersection(index_returns.index)
        X = selected_returns.loc[common_dates].values
        y = index_returns.loc[common_dates]
        
        # Ensure y is a numpy array
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=float).flatten()

        # Objective function: minimize ||y - X @ w||^2
        def objective(w):
            residuals = y - X @ w
            return np.sum(residuals ** 2)

        # Gradient of objective function
        def gradient(w):
            residuals = y - X @ w
            return -2 * (X.T @ residuals)

        # Constraint: sum(w) = 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: lower_bound <= w_i <= upper_bound
        bounds = [(self.lower_bound, self.upper_bound) for _ in range(num_assets)]

        # Initial guess: equal weights
        w0 = np.ones(num_assets) / num_assets

        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": self.tolerance, "maxiter": 1000},
        )

        self.optimization_result = result
        self.optimal_weights = pd.Series(result.x, index=assets)

        return self.optimal_weights

    def get_weights(self) -> pd.Series:
        """Get the optimal weights from last optimization."""
        if self.optimal_weights is None:
            raise ValueError("No optimization has been performed yet.")
        return self.optimal_weights

    def get_summary(self) -> Dict:
        """Get optimization summary."""
        if self.optimization_result is None:
            raise ValueError("No optimization has been performed yet.")

        return {
            "success": self.optimization_result.success,
            "message": self.optimization_result.message,
            "iterations": self.optimization_result.nit,
            "function_value": self.optimization_result.fun,
            "max_weight": self.optimal_weights.max(),
            "min_weight": self.optimal_weights.min(),
            "num_non_zero": (self.optimal_weights > self.tolerance).sum(),
        }
