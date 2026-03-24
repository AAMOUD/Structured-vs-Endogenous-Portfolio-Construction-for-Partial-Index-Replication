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
            remaining = np.setdiff1d(np.arange(len(weights)), selected_idx)
            need = min(self.K - len(selected_idx), len(remaining))
            if need > 0:
                fill_idx = remaining[np.argsort(weights[remaining])[-need:]]
                selected_idx = np.concatenate([selected_idx, fill_idx])

        if len(selected_idx) > 0:
            selected_idx = selected_idx[np.argsort(weights[selected_idx])[::-1]]

        selected_assets = assets[selected_idx]
        selected_weights = weights[selected_idx]

        if selected_weights.sum() > 0:
            selected_weights /= selected_weights.sum()

        self.selected_assets = list(selected_assets)
        self.weights = selected_weights

    def refit_long_only_weights(self, R, index_returns, selected_assets,
                                sector_constraints=None,
                                sector_penalty: float = 200.0):
        R_sel = R[selected_assets].to_numpy(dtype=float)
        idx_np = index_returns.to_numpy(dtype=float).flatten()
        n_assets = R_sel.shape[1]

        w = cp.Variable(n_assets)
        base_constraints = [cp.sum(w) == 1, w >= 1e-4]

        penalty_terms = []
        sector_cons = []

        if sector_constraints:
            for idx_list, bench_w, bound in sector_constraints:
                sw = cp.sum(w[idx_list])
                slack = cp.Variable(nonneg=True)
                sector_cons += [
                    sw >= bench_w - bound - slack,
                    sw <= bench_w + bound + slack,
                ]
                penalty_terms.append(slack)

        objective = cp.sum_squares(idx_np - R_sel @ w)
        if penalty_terms:
            objective += sector_penalty * cp.sum(cp.hstack(penalty_terms))

        prob = cp.Problem(cp.Minimize(objective), base_constraints + sector_cons)
        prob.solve(solver=cp.CLARABEL)

        if w.value is None:
            w2 = cp.Variable(n_assets)
            prob2 = cp.Problem(
                cp.Minimize(cp.sum_squares(idx_np - R_sel @ w2)),
                [cp.sum(w2) == 1, w2 >= 1e-4],
            )
            prob2.solve(solver=cp.CLARABEL)
            weights = np.array(w2.value).flatten()
        else:
            weights = np.array(w.value).flatten()

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights[weights < 0] = 0
        if weights.sum() > 0:
            weights /= weights.sum()

        return weights

    @staticmethod
    def _build_sector_constraints(selected_assets, R_columns, sectors,
                                   market_caps, K):
        def get_cap(t):
            if t in market_caps.index:
                v = market_caps[t]
                return float(v) if v == v else 0.0
            alt = t.replace("-", ".") if "-" in t else t.replace(".", "-")
            if alt in market_caps.index:
                v = market_caps[alt]
                return float(v) if v == v else 0.0
            return 0.0

        # Full-universe benchmark sector weights.
        all_caps = {a: get_cap(a) for a in R_columns}
        total_cap = sum(all_caps.values()) or 1.0

        bench_sector_w = {}
        for a, cap in all_caps.items():
            sec = sectors.get(a)
            if sec is None:
                alt = a.replace("-", ".") if "-" in a else a.replace(".", "-")
                sec = sectors.get(alt)
            if sec is None:
                continue
            bench_sector_w[sec] = bench_sector_w.get(sec, 0.0) + cap / total_cap

        result = []
        for sec, b in bench_sector_w.items():
            idx = []
            for i, a in enumerate(selected_assets):
                s = sectors.get(a)
                if s is None:
                    alt = a.replace("-", ".") if "-" in a else a.replace(".", "-")
                    s = sectors.get(alt)
                if s == sec:
                    idx.append(i)

            if not idx:
                continue

            delta = float(np.clip(0.10 * b, 0.005, 0.030))
            result.append((idx, b, delta))

        return result
