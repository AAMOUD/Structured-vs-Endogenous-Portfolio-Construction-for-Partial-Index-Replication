import numpy as np
import cvxpy as cp

class BaseModel:

    def __init__(self, K):
        self.K = K
        self.weights = None
        self.selected_assets = None

    def fit(self, R, index_returns):
        raise NotImplementedError

    def predict_weights(self):
        return self.weights

    def selected(self):
        return self.selected_assets

    def set_selected_from_weights(self, assets, raw_weights):
        assets = np.array(assets)
        weights = np.array(raw_weights).flatten()

        if len(weights) != len(assets):
            raise ValueError("weights and assets must have the same length")

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights[weights < 0] = 0

        selected_idx = np.where(weights > 1e-6)[0]

        if len(selected_idx) > self.K:
            selected_idx = selected_idx[np.argsort(weights[selected_idx])[-self.K:]]
        elif len(selected_idx) < self.K:
            remaining = np.setdiff1d(np.arange(len(weights)), selected_idx, assume_unique=False)
            if len(remaining) > 0:
                need = min(self.K - len(selected_idx), len(remaining))
                fill_idx = remaining[np.argsort(weights[remaining])[-need:]]
                selected_idx = np.concatenate([selected_idx, fill_idx])

        if len(selected_idx) > 0:
            selected_idx = selected_idx[np.argsort(weights[selected_idx])[::-1]]

        selected_assets = assets[selected_idx]
        selected_weights = weights[selected_idx]

        if selected_weights.sum() > 0:
            selected_weights = selected_weights / selected_weights.sum()

        self.selected_assets = list(selected_assets)
        self.weights = selected_weights

    def refit_long_only_weights(self, R, index_returns, selected_assets):
        R_sel = R[selected_assets].to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()

        n = R_sel.shape[1]

        w = cp.Variable(n)

        objective = cp.Minimize(cp.sum_squares(index_np - R_sel @ w))

        constraints = [
            cp.sum(w) == 1,
            w >= 1e-4
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = np.array(w.value).flatten()
        weights[weights < 0] = 0
        weights = weights / weights.sum()

        return weights