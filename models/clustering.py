"""Clustering-based model implementation."""
from sklearn.cluster import KMeans
from .base_model import BaseModel

class ClusteringModel(BaseModel):

    def fit(self, R, index_returns):

        corr_matrix = R.corr().values

        kmeans = KMeans(n_clusters=self.K)
        clusters = kmeans.fit_predict(corr_matrix)

        selected = []

        for k in range(self.K):

            cluster_assets = R.columns[clusters == k]

            if len(cluster_assets) == 0:
                continue

            corr = R[cluster_assets].corrwith(index_returns).dropna()
            if corr.empty:
                continue
            selected.append(corr.idxmax())

        selected = list(dict.fromkeys(selected))

        corr_all = R.corrwith(index_returns).dropna().sort_values(ascending=False)

        if corr_all.empty:
            corr_all = R.var().sort_values(ascending=False)

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