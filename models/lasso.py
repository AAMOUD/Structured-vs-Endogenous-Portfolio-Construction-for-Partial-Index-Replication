"""LASSO-based model implementation."""
import numpy as np
import cvxpy as cp
from .base_model import BaseModel

class LassoModel(BaseModel):

    def __init__(self, K, sectors=None, alpha=0.001):
        super().__init__(K)
        self.alpha = alpha

    def fit(self, R, index_returns):

        R_np = R.to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()

        N = R.shape[1]

        w = cp.Variable(N)

        objective = cp.Minimize(
            cp.sum_squares(index_np - R_np @ w) + self.alpha * cp.norm1(w)
        )

        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]

        cp.Problem(objective, constraints).solve()

        weights = np.array(w.value).flatten()

        self.set_selected_from_weights(R.columns, weights)
        self.weights = self.refit_long_only_weights(R, index_returns, self.selected_assets)