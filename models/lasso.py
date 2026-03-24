"""LASSO-based model implementation."""
import numpy as np
import cvxpy as cp
from .base_model import BaseModel


class LassoModel(BaseModel):

    _solve_cache = {}
    _solve_cache_order = []
    _max_cache_entries = 4000

    def __init__(self, K, sectors=None):
        super().__init__(K)

    @classmethod
    def _solve_nonnegative_lasso(cls, R_np, index_np, alpha, dataset_key=None):
        alpha = float(alpha)

        cache_key = None
        if dataset_key is not None:
            cache_key = (dataset_key, round(alpha, 12))
            cached = cls._solve_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

        n_assets = R_np.shape[1]
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.sum_squares(index_np - R_np @ w) + alpha * cp.norm1(w))
        constraints = [w >= 0]
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP)
        except Exception:
            problem.solve(solver=cp.SCS)

        if w.value is None:
            return None

        weights = np.array(w.value).flatten()
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights[weights < 0] = 0

        if cache_key is not None:
            cls._solve_cache[cache_key] = weights.copy()
            cls._solve_cache_order.append(cache_key)
            if len(cls._solve_cache_order) > cls._max_cache_entries:
                oldest = cls._solve_cache_order.pop(0)
                cls._solve_cache.pop(oldest, None)

        return weights

    def _pick_alpha_targeting_k(self, R_np, index_np, columns, tol=1e-8):
        first_block = R_np[:5, :5]
        last_block = R_np[-5:, -5:]
        dataset_key = (
            R_np.shape,
            tuple(columns),
            int(np.round(np.nansum(first_block) * 1e8)),
            int(np.round(np.nansum(last_block) * 1e8)),
            round(float(np.nanmean(index_np)), 12),
            round(float(np.nanstd(index_np)), 12),
        )

        def evaluate(alpha):
            raw_w = self._solve_nonnegative_lasso(
                R_np,
                index_np,
                float(alpha),
                dataset_key=dataset_key,
            )
            if raw_w is None or raw_w.sum() <= 0:
                return None
            nnz = int(np.sum(raw_w > tol))
            w_norm = raw_w / raw_w.sum()
            error = float(np.mean((index_np - R_np @ w_norm) ** 2))
            return {
                "alpha": float(alpha),
                "weights": raw_w,
                "nnz": nnz,
                "error": error,
            }

        candidates = []

        low_alpha = 1e-8
        high_alpha = 1.0
        low_eval = evaluate(low_alpha)
        if low_eval is not None:
            candidates.append(low_eval)

        high_eval = evaluate(high_alpha)
        if high_eval is not None:
            candidates.append(high_eval)

        while high_eval is not None and high_eval["nnz"] > self.K and high_alpha < 1e6:
            high_alpha *= 10.0
            high_eval = evaluate(high_alpha)
            if high_eval is not None:
                candidates.append(high_eval)

        if low_eval is not None and high_eval is not None:
            lo = low_alpha
            hi = high_alpha
            for _ in range(25):
                mid = np.sqrt(lo * hi)
                mid_eval = evaluate(mid)
                if mid_eval is None:
                    break

                candidates.append(mid_eval)
                if mid_eval["nnz"] > self.K:
                    lo = mid
                else:
                    hi = mid

        # Fallback scan to remain robust if support size is not strictly monotone.
        for alpha in np.logspace(-6, 2, 20):
            result = evaluate(alpha)
            if result is not None:
                candidates.append(result)

        if not candidates:
            return None

        best = min(candidates, key=lambda r: (abs(r["nnz"] - self.K), r["error"]))
        best["rank_key"] = (abs(best["nnz"] - self.K), best["error"])
        return best

    def _select_assets(self, R, index_returns):
        R_np = R.to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()

        best = self._pick_alpha_targeting_k(R_np, index_np, R.columns)
        if best is None:
            best_weights = np.ones(R.shape[1], dtype=float) / float(R.shape[1])
        else:
            best_weights = best["weights"]

        self.set_selected_from_weights(R.columns, best_weights)
        return self.selected_assets

    def fit(self, R, index_returns):
        self._select_assets(R, index_returns)
        self.weights = self.refit_long_only_weights(R, index_returns, self.selected_assets)


class LassoSectorModel(LassoModel):
    """LASSO selection + benchmark-relative sector weight constraints."""

    def __init__(self, K, sectors, market_caps):
        super().__init__(K)
        self.sectors = sectors
        self.market_caps = market_caps

    def fit(self, R, index_returns):
        self._select_assets(R, index_returns)
        sc = self._build_sector_constraints(
            self.selected_assets, R.columns, self.sectors, self.market_caps, self.K
        )
        self.weights = self.refit_long_only_weights(
            R, index_returns, self.selected_assets, sector_constraints=sc
        )