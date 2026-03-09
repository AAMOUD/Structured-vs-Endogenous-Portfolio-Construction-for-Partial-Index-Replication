"""Stratified sampling model implementation."""
from .base_model import BaseModel

class StratifiedModel(BaseModel):

    def __init__(self, K, sectors):
        super().__init__(K)
        self.sectors = sectors

    def fit(self, R, index_returns):

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

            corr = R[sector_assets].corrwith(index_returns)
            chosen = corr.sort_values(ascending=False).head(Kj).index.tolist()

            selected.extend(chosen)

        selected = list(dict.fromkeys(selected))

        corr_all = R.corrwith(index_returns).sort_values(ascending=False)

        if len(selected) < self.K:
            for asset in corr_all.index:
                if asset not in selected:
                    selected.append(asset)
                if len(selected) == self.K:
                    break
        elif len(selected) > self.K:
            selected = sorted(selected, key=lambda asset: corr_all[asset], reverse=True)[:self.K]

        self.selected_assets = selected
        self.weights = self.refit_long_only_weights(R, index_returns, selected)