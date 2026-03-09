import cvxpy as cp
import numpy as np
import time


class LayeredOptimization:

    def __init__(self, K, sectors):

        self.K = K
        self.sectors = sectors

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

        cap = 2.0 / self.K

        constraints = []

        # Layer 1 : cardinality
        constraints += [
            cp.sum(w) == 1,
            cp.sum(x) == self.K,
            w >= 0,
            w <= cap * x          # forces w=0 when x=0, caps at 2/K when x=1
        ]

        # Layer 2 : sector constraints
        # Resolve ticker variants (BF-B -> BF.B) for sector lookup
        def resolve_ticker(t):
            if t not in self.sectors.index:
                alt = t.replace("-", ".")
                if alt in self.sectors.index:
                    return alt
            return t

        unique_sectors = self.sectors.loc[[resolve_ticker(a) for a in assets]].unique()

        for s in unique_sectors:

            sector_assets = [i for i, a in enumerate(assets)
                             if self.sectors.get(resolve_ticker(a)) == s]

            if len(sector_assets) == 0:
                continue

            sector_weight = cp.sum(w[sector_assets])

            constraints += [
                sector_weight <= 0.30
            ]

        # objective
        objective = cp.Minimize(cp.sum_squares(index_np - R_np @ w))

        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.GUROBI, TimeLimit=120, MIPGap=0.01, Threads=4,
                   reoptimize=True)

        end_time = time.time()

        self.execution_time = end_time - start_time

        # Handle timeout / infeasible: fall back to top-K by correlation
        if x.value is None:
            corr = np.corrcoef(R_np.T, index_np.flatten())[:-1, -1]
            selected_idx = np.argsort(corr)[-self.K:]
        else:
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