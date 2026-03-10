import cvxpy as cp
import numpy as np
import time


class LayeredOptimization:

    def __init__(self, K, sectors, market_caps=None):

        self.K = K
        self.sectors = sectors
        self.market_caps = market_caps

    def fit(self, R, index_returns):

        start_time = time.time()

        R = R.dropna(axis=1)

        assets = R.columns

        N = len(assets)

        R_np = R.to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()

        # variables
        w = cp.Variable(N)
        x = cp.Variable(N, boolean=True)

        constraints = []

        # Layer 1 : cardinality  (max 7% per asset)
        constraints += [
            cp.sum(w) == 1,
            cp.sum(x) == self.K,
            w >= 0,
            w <= 0.07 * x
        ]

        # Layer 2 : benchmark-relative sector constraints (±1%)
        def resolve_ticker(t):
            if t not in self.sectors.index:
                alt = t.replace("-", ".")
                if alt in self.sectors.index:
                    return alt
            return t

        # Compute benchmark sector weights from market caps
        if self.market_caps is not None:
            caps = self.market_caps.reindex(assets).fillna(0)
            sector_caps = {}
            for asset, cap in caps.items():
                sector = self.sectors.get(resolve_ticker(asset))
                if sector is None:
                    continue
                sector_caps[sector] = sector_caps.get(sector, 0) + cap
            total_cap = sum(sector_caps.values())
            bench_sector_w = {
                s: v / total_cap for s, v in sector_caps.items()
            } if total_cap > 0 else {}
        else:
            bench_sector_w = {}

        unique_sectors = set(
            self.sectors.get(resolve_ticker(a)) for a in assets
        ) - {None}

        for s in unique_sectors:

            sector_idx = [
                i for i, a in enumerate(assets)
                if self.sectors.get(resolve_ticker(a)) == s
            ]

            if len(sector_idx) == 0:
                continue

            sector_weight = cp.sum(w[sector_idx])

            if s in bench_sector_w:
                b = bench_sector_w[s]
                constraints += [
                    sector_weight >= b - 0.01,
                    sector_weight <= b + 0.01
                ]
            else:
                # Fallback: loose cap if benchmark weight unknown
                constraints += [sector_weight <= 0.30]

        # objective
        objective = cp.Minimize(cp.sum_squares(index_np - R_np @ w))

        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.GUROBI, TimeLimit=180)   # 3 min max

        end_time = time.time()

        self.execution_time = end_time - start_time

        if x.value is None:
            raise RuntimeError("Layered model failed to solve")

        selected_idx = np.array([i for i in range(N) if x.value[i] > 0.5])

        if len(selected_idx) != self.K:
            raw_weights = np.array(w.value).flatten()
            selected_idx = np.argsort(raw_weights)[-self.K:]

        self.selected_assets = list(assets[selected_idx])

        R_sel = R[self.selected_assets].to_numpy(dtype=float)
        n = R_sel.shape[1]
        w_refit = cp.Variable(n)
        prob_refit = cp.Problem(
            cp.Minimize(cp.sum_squares(index_np - R_sel @ w_refit)),
            [cp.sum(w_refit) == 1, w_refit >= 1e-4]
        )
        prob_refit.solve()
        ols_weights = np.array(w_refit.value).flatten()
        ols_weights[ols_weights < 0] = 0
        ols_weights = ols_weights / ols_weights.sum()

        self.weights = ols_weights

        return self