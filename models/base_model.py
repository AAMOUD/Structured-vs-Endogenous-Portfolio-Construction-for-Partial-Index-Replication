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

    def refit_long_only_weights(self, R, index_returns, selected_assets,
                                 sector_constraints=None):
        """
        sector_constraints: list of (list_of_indices, bench_weight, bound)
            each entry adds:  b-bound <= sum(w[idx]) <= b+bound
        """
        R_sel = R[selected_assets].to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()

        n = R_sel.shape[1]

        w = cp.Variable(n)

        objective = cp.Minimize(cp.sum_squares(index_np - R_sel @ w))

        constraints = [
            cp.sum(w) == 1,
            w >= 1e-4
        ]

        if sector_constraints:
            for idx_list, bench_w, bound in sector_constraints:
                sw = cp.sum(w[idx_list])
                constraints += [
                    sw >= bench_w - bound,
                    sw <= bench_w + bound
                ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if w.value is None:
            # Infeasible (tight sector bounds) — retry without sector constraints
            prob2 = cp.Problem(
                cp.Minimize(cp.sum_squares(index_np - R_sel @ w)),
                [cp.sum(w) == 1, w >= 1e-4]
            )
            prob2.solve()

        weights = np.array(w.value).flatten()
        weights[weights < 0] = 0
        weights = weights / weights.sum()

        return weights

    @staticmethod
    def _build_sector_constraints(selected_assets, R_columns, sectors, market_caps, K):
        """Compute benchmark sector weights from market caps over R_columns,
        then return sector constraint tuples for refit_long_only_weights.
        Bound = ±0.5% for K<=100, ±1% for K>100.
        """
        def get_cap(t):
            if t in market_caps.index:
                v = market_caps[t]
                return float(v) if v == v else 0.0
            alt = t.replace("-", ".") if "-" in t else t.replace(".", "-")
            if alt in market_caps.index:
                v = market_caps[alt]
                return float(v) if v == v else 0.0
            return 0.0

        all_caps = {a: get_cap(a) for a in R_columns}
        total_cap = sum(all_caps.values())
        if not total_cap or total_cap != total_cap:   # 0 or NaN
            total_cap = 1.0

        bench_sector_w = {}
        for a, cap in all_caps.items():
            s = sectors.get(a)
            if s is None:
                continue
            bench_sector_w[s] = bench_sector_w.get(s, 0.0) + cap / total_cap

        bound = 0.005 if K <= 100 else 0.01
        result = []
        for s, b in bench_sector_w.items():
            idx = [i for i, a in enumerate(selected_assets) if sectors.get(a) == s]
            if idx:
                result.append((idx, b, bound))
        return result