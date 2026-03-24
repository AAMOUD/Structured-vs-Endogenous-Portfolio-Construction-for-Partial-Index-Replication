"""Index-contribution models based on single-asset explanatory power."""
import numpy as np

from .base_model import BaseModel


class ContributionModel(BaseModel):

    def __init__(self, K, sectors=None):
        super().__init__(K)

    def _score_assets(self, R, index_returns):
        returns_np = R.to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()

        index_std = float(np.std(index_np))
        if index_std <= 0:
            return np.zeros(R.shape[1], dtype=float)

        asset_std = np.std(returns_np, axis=0)
        cov = np.mean((returns_np - returns_np.mean(axis=0)) * (index_np[:, None] - index_np.mean()), axis=0)
        denom = asset_std * index_std
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, 0.0)

        # Absolute correlation is variance-normalized explanatory power.
        scores = np.abs(corr)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _select_assets(self, R, index_returns):
        scores = self._score_assets(R, index_returns)
        ranked_idx = np.argsort(scores)[::-1]

        selected = [R.columns[i] for i in ranked_idx[: self.K]]
        if len(selected) < self.K:
            remaining = [asset for asset in R.columns if asset not in selected]
            selected.extend(remaining[: self.K - len(selected)])

        return selected[: self.K]

    def fit(self, R, index_returns):
        self.selected_assets = self._select_assets(R, index_returns)
        self.weights = self.refit_long_only_weights(R, index_returns, self.selected_assets)


class ContributionSectorModel(ContributionModel):
    """Contribution selection + benchmark-relative sector weight constraints."""

    def __init__(self, K, sectors, market_caps):
        super().__init__(K)
        self.sectors = sectors
        self.market_caps = market_caps

    def fit(self, R, index_returns):
        self.selected_assets = self._select_assets(R, index_returns)
        sc = self._build_sector_constraints(
            self.selected_assets, R.columns, self.sectors, self.market_caps, self.K
        )
        self.weights = self.refit_long_only_weights(
            R, index_returns, self.selected_assets, sector_constraints=sc
        )