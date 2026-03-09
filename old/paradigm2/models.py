"""Portfolio selection models for Paradigm 2 - O Strategy (Endogenous Construction).

Implements simultaneous asset selection and weight optimization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from config import PORTFOLIO_PARAMS, OPTIMIZATION_PARAMS
import warnings
warnings.filterwarnings('ignore')


class OStrategyModel:
    """Base class for O Strategy models (endogenous construction)."""

    def __init__(self, K: int = PORTFOLIO_PARAMS["K"]):
        """
        Initialize O Strategy model.

        Args:
            K: Target number of assets to select (may be approximate for convex methods)
        """
        self.K = K
        self.selected_assets = None
        self.optimal_weights = None
        self.solve_time = None

    def optimize_and_select(
        self, 
        returns: pd.DataFrame, 
        index_returns: pd.Series,
        **kwargs
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Simultaneously select assets and optimize weights.
        
        Args:
            returns: DataFrame of asset returns (T x N)
            index_returns: Series of index returns (T,)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (selected_assets, optimal_weights)
        """
        raise NotImplementedError("Must be implemented by subclass")
    
    def get_active_positions(self, weights: Dict[str, float], threshold: float = 1e-6) -> int:
        """Count number of non-zero positions."""
        return sum(1 for w in weights.values() if abs(w) > threshold)


# ========================================
# Model 1: MIQP Cardinality Model (Exact)
# ========================================

class MIQPCardinalityModel(OStrategyModel):
    """
    Mixed-Integer Quadratic Programming with cardinality constraint.
    
    Formulation:
        min (1/T) * ||R*w - I||²
        s.t. sum(w) = 1
             sum(x) = K
             l*x_i <= w_i <= u*x_i
             x_i ∈ {0,1}
    
    Uses: SCIP (free, open-source) or CPLEX (free academic license)
    """
    
    def __init__(
        self, 
        K: int = PORTFOLIO_PARAMS["K"],
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        solver: str = 'SCIP',  # 'SCIP' or 'CPLEX'
        time_limit: Optional[int] = 300
    ):
        super().__init__(K)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.solver = solver
        self.time_limit = time_limit
        
    def optimize_and_select(
        self, 
        returns: pd.DataFrame, 
        index_returns: pd.Series,
        **kwargs
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Solve MIQP to simultaneously select K assets and optimize weights.
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy required. Install: pip install cvxpy")
        
        import time
        start_time = time.time()
        
        # Align data
        common_idx = returns.index.intersection(index_returns.index)
        R = returns.loc[common_idx].values  # T x N
        I = index_returns.loc[common_idx].values  # T
        T, N = R.shape
        tickers = returns.columns.tolist()
        
        # Decision variables
        w = cp.Variable(N)  # Weights
        x = cp.Variable(N, boolean=True)  # Selection indicators
        
        # Objective: minimize tracking error
        tracking_error = (1/T) * cp.sum_squares(R @ w - I)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            cp.sum(x) == self.K,  # Exactly K assets
        ]
        
        # Link weights and selection: l*x_i <= w_i <= u*x_i
        for i in range(N):
            constraints.append(w[i] >= self.lower_bound * x[i])
            constraints.append(w[i] <= self.upper_bound * x[i])
        
        # Solve
        problem = cp.Problem(cp.Minimize(tracking_error), constraints)
        
        # Set solver parameters
        solver_opts = {}
        if self.time_limit:
            solver_opts['timeLimit'] = self.time_limit
            
        if self.solver == 'SCIP':
            try:
                problem.solve(solver=cp.SCIP, verbose=False, **solver_opts)
            except:
                # Fallback to ECOS_BB
                problem.solve(solver=cp.ECOS_BB, verbose=False)
        elif self.solver == 'CPLEX':
            problem.solve(solver=cp.CPLEX, verbose=False, **solver_opts)
        else:
            # Default: try ECOS_BB (built-in mixed-integer solver)
            problem.solve(solver=cp.ECOS_BB, verbose=False)
        
        self.solve_time = time.time() - start_time
        
        if w.value is None:
            raise ValueError("Solver failed to find solution")
        
        # Extract solution
        weights_array = w.value
        selection_array = x.value
        
        # Build weights dict (only selected assets)
        weights_dict = {}
        selected = []
        for i, ticker in enumerate(tickers):
            if abs(selection_array[i]) > 0.5:  # Binary threshold
                weights_dict[ticker] = float(weights_array[i])
                selected.append(ticker)
        
        self.selected_assets = selected
        self.optimal_weights = weights_dict
        
        return selected, weights_dict


# ========================================
# Model 2: LASSO (L1 Regularization)
# ========================================

class LassoModel(OStrategyModel):
    """
    LASSO for sparse portfolio construction.
    
    Formulation:
        min (1/T) * ||R*w - I||² + λ * ||w||₁
        s.t. sum(w) = 1
             w >= 0
    
    Fully convex, no integer variables.
    """
    
    def __init__(
        self, 
        K: int = PORTFOLIO_PARAMS["K"],
        lambda_param: Optional[float] = None,
        lambda_grid: Optional[List[float]] = None,
        upper_bound: float = OPTIMIZATION_PARAMS["upper_bound"]
    ):
        super().__init__(K)
        self.lambda_param = lambda_param
        self.lambda_grid = lambda_grid or [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
        self.upper_bound = upper_bound
        
    def optimize_and_select(
        self, 
        returns: pd.DataFrame, 
        index_returns: pd.Series,
        **kwargs
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Solve LASSO to get sparse portfolio.
        If lambda not specified, search for lambda that gives ~K assets.
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy required. Install: pip install cvxpy")
        
        import time
        start_time = time.time()
        
        # Align data
        common_idx = returns.index.intersection(index_returns.index)
        R = returns.loc[common_idx].values
        I = index_returns.loc[common_idx].values
        T, N = R.shape
        tickers = returns.columns.tolist()
        
        if self.lambda_param is None:
            # Search for lambda that gives approximately K assets
            best_lambda = None
            best_diff = float('inf')
            best_w = None
            
            for lam in self.lambda_grid:
                w_test = self._solve_lasso(R, I, T, lam, self.upper_bound)
                if w_test is not None:
                    n_active = sum(1 for x in w_test if abs(x) > 1e-6)
                    diff = abs(n_active - self.K)
                    if diff < best_diff:
                        best_diff = diff
                        best_lambda = lam
                        best_w = w_test
            
            self.lambda_param = best_lambda
            weights_array = best_w
        else:
            weights_array = self._solve_lasso(R, I, T, self.lambda_param, self.upper_bound)
        
        self.solve_time = time.time() - start_time
        
        if weights_array is None:
            raise ValueError("LASSO solver failed")
        
        # Extract non-zero weights
        weights_dict = {}
        selected = []
        for i, ticker in enumerate(tickers):
            if abs(weights_array[i]) > 1e-6:
                weights_dict[ticker] = float(weights_array[i])
                selected.append(ticker)
        
        self.selected_assets = selected
        self.optimal_weights = weights_dict
        
        return selected, weights_dict
    
    def _solve_lasso(self, R, I, T, lam, upper_bound):
        """Solve LASSO for given lambda."""
        import cvxpy as cp
        
        N = R.shape[1]
        w = cp.Variable(N)
        
        # Objective
        tracking_error = (1/T) * cp.sum_squares(R @ w - I)
        l1_penalty = lam * cp.norm(w, 1)
        objective = tracking_error + l1_penalty
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= upper_bound  # Max 10% per asset
        ]
        
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            return w.value if w.value is not None else None
        except:
            return None


# ========================================
# Model 3: Elastic Net
# ========================================

class ElasticNetModel(OStrategyModel):
    """
    Elastic Net for sparse portfolio construction.
    
    Formulation:
        min ||R*w - I||² + λ₁*||w||₁ + λ₂*||w||²₂
        s.t. sum(w) = 1
             w >= 0
    
    More stable than pure LASSO.
    """
    
    def __init__(
        self, 
        K: int = PORTFOLIO_PARAMS["K"],
        lambda1: Optional[float] = None,
        lambda2: float = 0.01,
        lambda_grid: Optional[List[float]] = None,
        upper_bound: float = OPTIMIZATION_PARAMS["upper_bound"]
    ):
        super().__init__(K)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_grid = lambda_grid or [0.0001, 0.0005, 0.001, 0.005, 0.01]
        self.upper_bound = upper_bound
        
    def optimize_and_select(
        self, 
        returns: pd.DataFrame, 
        index_returns: pd.Series,
        **kwargs
    ) -> Tuple[List[str], Dict[str, float]]:
        """Solve Elastic Net to get sparse portfolio."""
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy required. Install: pip install cvxpy")
        
        import time
        start_time = time.time()
        
        # Align data
        common_idx = returns.index.intersection(index_returns.index)
        R = returns.loc[common_idx].values
        I = index_returns.loc[common_idx].values
        T, N = R.shape
        tickers = returns.columns.tolist()
        
        if self.lambda1 is None:
            # Search for lambda1 that gives approximately K assets
            best_lambda = None
            best_diff = float('inf')
            best_w = None
            
            for lam1 in self.lambda_grid:
                w_test = self._solve_elastic_net(R, I, T, lam1, self.lambda2, self.upper_bound)
                if w_test is not None:
                    n_active = sum(1 for x in w_test if abs(x) > 1e-6)
                    diff = abs(n_active - self.K)
                    if diff < best_diff:
                        best_diff = diff
                        best_lambda = lam1
                        best_w = w_test
            
            self.lambda1 = best_lambda
            weights_array = best_w
        else:
            weights_array = self._solve_elastic_net(R, I, T, self.lambda1, self.lambda2, self.upper_bound)
        
        self.solve_time = time.time() - start_time
        
        if weights_array is None:
            raise ValueError("Elastic Net solver failed")
        
        # Extract non-zero weights
        weights_dict = {}
        selected = []
        for i, ticker in enumerate(tickers):
            if abs(weights_array[i]) > 1e-6:
                weights_dict[ticker] = float(weights_array[i])
                selected.append(ticker)
        
        self.selected_assets = selected
        self.optimal_weights = weights_dict
        
        return selected, weights_dict
    
    def _solve_elastic_net(self, R, I, T, lam1, lam2, upper_bound):
        """Solve Elastic Net for given lambdas."""
        import cvxpy as cp
        
        N = R.shape[1]
        w = cp.Variable(N)
        
        # Objective
        tracking_error = (1/T) * cp.sum_squares(R @ w - I)
        l1_penalty = lam1 * cp.norm(w, 1)
        l2_penalty = lam2 * cp.sum_squares(w)
        objective = tracking_error + l1_penalty + l2_penalty
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= upper_bound  # Max 10% per asset
        ]
        
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            return w.value if w.value is not None else None
        except:
            return None


# ========================================
# Model 4: Reweighted L1 (Iterative)
# ========================================

class ReweightedL1Model(OStrategyModel):
    """
    Reweighted L1 for better cardinality approximation.
    
    Iterative formulation:
        min ||R*w - I||² + λ * Σ |w_i| / (|w_i^prev| + ε)
    
    Approximates L0 norm while staying convex at each iteration.
    """
    
    def __init__(
        self, 
        K: int = PORTFOLIO_PARAMS["K"],
        lambda_param: float = 0.00001,
        epsilon: float = 1e-3,
        max_iter: int = 10,
        upper_bound: float = OPTIMIZATION_PARAMS["upper_bound"]
    ):
        super().__init__(K)
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.upper_bound = upper_bound
        
    def optimize_and_select(
        self, 
        returns: pd.DataFrame, 
        index_returns: pd.Series,
        **kwargs
    ) -> Tuple[List[str], Dict[str, float]]:
        """Solve reweighted L1 iteratively."""
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy required. Install: pip install cvxpy")
        
        import time
        start_time = time.time()
        
        # Align data
        common_idx = returns.index.intersection(index_returns.index)
        R = returns.loc[common_idx].values
        I = index_returns.loc[common_idx].values
        T, N = R.shape
        tickers = returns.columns.tolist()
        
        # Initialize with uniform weights
        weights_prev = np.ones(N) / N
        
        # Iterative reweighting
        for iteration in range(self.max_iter):
            # Compute adaptive weights
            adaptive_weights = 1.0 / (np.abs(weights_prev) + self.epsilon)
            
            # Solve weighted L1 problem
            w = cp.Variable(N)
            
            tracking_error = (1/T) * cp.sum_squares(R @ w - I)
            weighted_l1 = self.lambda_param * cp.sum(cp.multiply(adaptive_weights, cp.abs(w)))
            objective = tracking_error + weighted_l1
            
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= self.upper_bound  # Max 10% per asset
            ]
            
            problem = cp.Problem(cp.Minimize(objective), constraints)
            
            try:
                problem.solve(solver=cp.OSQP, verbose=False)
            except:
                break
            
            if w.value is None:
                break
            
            # Check convergence
            weights_new = w.value
            change = np.linalg.norm(weights_new - weights_prev)
            weights_prev = weights_new
            
            if change < 1e-5:
                break
        
        self.solve_time = time.time() - start_time
        
        if weights_prev is None:
            raise ValueError("Reweighted L1 solver failed")
        
        # Extract non-zero weights
        weights_dict = {}
        selected = []
        for i, ticker in enumerate(tickers):
            if abs(weights_prev[i]) > 1e-6:
                weights_dict[ticker] = float(weights_prev[i])
                selected.append(ticker)
        
        self.selected_assets = selected
        self.optimal_weights = weights_dict
        
        return selected, weights_dict
