"""LASSO-based model implementation."""
import numpy as np
import cvxpy as cp
from .base_model import BaseModel


class LassoModel(BaseModel):

    def __init__(self, K, sectors=None):
        super().__init__(K)

    def _select_assets(self, R, index_returns):
        R_np = R.to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()

        N = R.shape[1]
        alphas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

        best_weights = None
        best_error = float("inf")

        for alpha in alphas:

            w = cp.Variable(N)
            objective = cp.Minimize(
                cp.sum_squares(index_np - R_np @ w) + alpha * cp.norm1(w)
            )
            constraints = [cp.sum(w) == 1, w >= 0]
            cp.Problem(objective, constraints).solve()

            if w.value is None:
                continue

            weights = np.array(w.value).flatten()
            weights[weights < 0] = 0
            if weights.sum() == 0:
                continue
            weights /= weights.sum()

            error = ((index_np - R_np @ weights) ** 2).mean()
            if error < best_error:
                best_error = error
                best_weights = weights

        if best_weights is None:
            best_weights = np.ones(N) / N

        self.set_selected_from_weights(R.columns, best_weights)
        return self.selected_assets

    def fit(self, R, index_returns):
        self._select_assets(R, index_returns)
        self.weights = self.refit_long_only_weights(R, index_returns, self.selected_assets)


class LassoSectorModel(LassoModel):
    """LASSO selection + benchmark-relative sector weight constraints."""

    def __init__(self, K, sectors, market_caps):
        super().__init__(K)
        self.sectors = sectors
        self.market_caps = market_caps

    def fit(self, R, index_returns):
        self._select_assets(R, index_returns)
        sc = self._build_sector_constraints(
            self.selected_assets, R.columns, self.sectors, self.market_caps, self.K
        )
        self.weights = self.refit_long_only_weights(
            R, index_returns, self.selected_assets, sector_constraints=sc
        )