"""Stratified sampling model: sector-proportional, market-cap ranked."""
import numpy as np
from .base_model import BaseModel


class StratifiedModel(BaseModel):

    def __init__(self, K, sectors, market_caps=None):
        super().__init__(K)
        self.sectors = sectors
        self.market_caps = market_caps

    def _get_cap(self, t):
        if self.market_caps is None:
            return 0.0
        if t in self.market_caps.index:
            v = self.market_caps[t]
            return float(v) if v == v else 0.0  # NaN-safe
        alt = t.replace("-", ".") if "-" in t else t.replace(".", "-")
        if alt in self.market_caps.index:
            v = self.market_caps[alt]
            return float(v) if v == v else 0.0
        return 0.0

    def _select_assets(self, R):
        unique_sectors = self.sectors.unique()
        N = R.shape[1]
        selected = []

        for sector in unique_sectors:
            sector_assets = self.sectors[self.sectors == sector].index
            sector_assets = [a for a in sector_assets if a in R.columns]
            Nj = len(sector_assets)
            Kj = int(self.K * Nj / N)
            if Kj == 0:
                continue
            caps = {a: self._get_cap(a) for a in sector_assets}
            chosen = sorted(caps, key=caps.get, reverse=True)[:Kj]
            selected.extend(chosen)

        selected = list(dict.fromkeys(selected))
        all_by_cap = sorted(R.columns, key=self._get_cap, reverse=True)

        if len(selected) < self.K:
            for asset in all_by_cap:
                if asset not in selected:
                    selected.append(asset)
                if len(selected) == self.K:
                    break
        elif len(selected) > self.K:
            selected = sorted(selected, key=self._get_cap, reverse=True)[: self.K]

        return selected

    def fit(self, R, index_returns):
        self.selected_assets = self._select_assets(R)
        self.weights = self.refit_long_only_weights(R, index_returns, self.selected_assets)


class StratifiedSectorModel(StratifiedModel):
    """Stratified selection + benchmark-relative sector weight constraints."""

    def fit(self, R, index_returns):
        self.selected_assets = self._select_assets(R)
        sc = self._build_sector_constraints(
            self.selected_assets, R.columns, self.sectors, self.market_caps, self.K
        )
        self.weights = self.refit_long_only_weights(
            R, index_returns, self.selected_assets, sector_constraints=sc
        )