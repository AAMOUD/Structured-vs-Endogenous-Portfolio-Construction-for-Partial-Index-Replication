"""Clustering-based model implementation."""
from sklearn.cluster import KMeans
from .base_model import BaseModel


class ClusteringModel(BaseModel):

    def __init__(self, K, market_caps=None):
        super().__init__(K)
        self.market_caps = market_caps

    def fit(self, R, index_returns):

        corr_matrix = R.corr().values

        kmeans = KMeans(n_clusters=self.K)
        clusters = kmeans.fit_predict(corr_matrix)

        selected = []

        # -- helper: market cap for a ticker (handles BF-B / BF.B variants, NaN) --
        def get_cap(t):
            if not hasattr(self, "market_caps") or self.market_caps is None:
                return 0.0
            if t in self.market_caps.index:
                v = self.market_caps[t]
                return float(v) if v == v else 0.0  # NaN-safe
            alt = t.replace("-", ".") if "-" in t else t.replace(".", "-")
            if alt in self.market_caps.index:
                v = self.market_caps[alt]
                return float(v) if v == v else 0.0
            return 0.0

        for k in range(self.K):

            cluster_assets = list(R.columns[clusters == k])

            if len(cluster_assets) == 0:
                continue

            # Pick representative = largest market cap in cluster
            chosen = max(cluster_assets, key=get_cap)
            selected.append(chosen)

        selected = list(dict.fromkeys(selected))

        # Fill / trim using global market-cap ranking
        all_by_cap = sorted(R.columns, key=get_cap, reverse=True)

        if len(selected) < self.K:
            for asset in all_by_cap:
                if asset not in selected:
                    selected.append(asset)
                if len(selected) == self.K:
                    break
        elif len(selected) > self.K:
            selected = sorted(selected, key=get_cap, reverse=True)[: self.K]

        self.selected_assets = selected
        self.weights = self.refit_long_only_weights(R, index_returns, selected)