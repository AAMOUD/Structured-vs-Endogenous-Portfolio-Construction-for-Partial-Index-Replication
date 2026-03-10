"""Market-cap model: selects the K largest stocks by market capitalisation."""
from .base_model import BaseModel


class MarketCapModel(BaseModel):

    def __init__(self, K, market_caps):
        super().__init__(K)
        self.market_caps = market_caps   # pd.Series indexed by ticker

    def _select_assets(self, R):
        caps = self.market_caps.reindex(R.columns).fillna(0)
        selected = caps.sort_values(ascending=False).head(self.K).index.tolist()
        if len(selected) < self.K:
            remaining = [t for t in R.columns if t not in selected]
            selected += remaining[: self.K - len(selected)]
        return selected[: self.K]

    def fit(self, R, index_returns):
        self.selected_assets = self._select_assets(R)
        self.weights = self.refit_long_only_weights(R, index_returns, self.selected_assets)


class MarketCapSectorModel(MarketCapModel):
    """Market-cap selection + benchmark-relative sector weight constraints."""

    def __init__(self, K, market_caps, sectors):
        super().__init__(K, market_caps)
        self.sectors = sectors

    def fit(self, R, index_returns):
        self.selected_assets = self._select_assets(R)
        sc = self._build_sector_constraints(
            self.selected_assets, R.columns, self.sectors, self.market_caps, self.K
        )
        self.weights = self.refit_long_only_weights(
            R, index_returns, self.selected_assets, sector_constraints=sc
        )