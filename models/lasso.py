"""LASSO — OMP selection. LW shrinkage applied in the base_model refit."""

import numpy as np
import cvxpy as cp
from .base_model import BaseModel
from .lw_shrinkage import ledoit_wolf

def _ols_project(R_sel, index_np):
    try:
        coeffs, _, _, _ = np.linalg.lstsq(R_sel, index_np, rcond=None)
        return R_sel @ coeffs
    except np.linalg.LinAlgError:
        return np.zeros_like(index_np)


class LassoModel(BaseModel):
    """
    OMP asset selection.

    Shrinkage note for LASSO
    ─────────────────────────
    For MIQP/Layered, LW shrinkage applies to the joint selection+weighting
    objective (the 459×459 Gram matrix).

    For LASSO, selection is done by OMP (a greedy heuristic, not a QP), so
    the main place shrinkage helps is the final weight refit.  The refit is
    done on K selected assets (K << 459), so T/K >> 1 — the covariance is
    already well-conditioned there.  The net gain from shrinking the refit
    is small, so we keep the refit as-is.

    The key shrinkage benefit for LASSO comes from using the OMP scoring
    criterion on the shrunk covariance-whitened returns, so that correlated
    asset clusters don't repeatedly consume OMP steps.  We implement this
    via the precision-weighted scoring below.
    """

    def __init__(self, K, sectors=None,
                 use_lasso_prefilter: bool = True,
                 prefilter_multiplier: int = 2,
                 use_lw_shrinkage: bool = True):
        super().__init__(K)
        self.use_lasso_prefilter  = use_lasso_prefilter
        self.prefilter_multiplier = prefilter_multiplier
        self.use_lw_shrinkage     = use_lw_shrinkage

    @staticmethod
    def _normalise(R_np, index_np):
        col_std  = np.where(R_np.std(0) < 1e-10, 1.0, R_np.std(0))
        R_norm   = (R_np - R_np.mean(0)) / col_std
        idx_std  = max(index_np.std(), 1e-10)
        idx_norm = (index_np - index_np.mean()) / idx_std
        return R_norm, idx_norm, col_std

    @staticmethod
    def _lasso_candidate_set(R_np, index_np, n_candidates):
        R_norm, idx_norm, _ = LassoModel._normalise(R_np, index_np)
        N = R_norm.shape[1]
        w = cp.Variable(N)
        prob = cp.Problem(
            cp.Minimize(
                cp.sum_squares(idx_norm - R_norm @ w) + 0.001 * cp.norm1(w)
            ),
            [w >= 0],
        )
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            try:
                prob.solve(solver=cp.SCS)
            except Exception:
                pass

        if w.value is not None:
            weights = np.nan_to_num(np.array(w.value).flatten(), nan=0.0)
            weights[weights < 0] = 0.0
        else:
            weights = np.zeros(N)

        top_idx = set(np.argsort(weights)[::-1][:n_candidates])

        if len(top_idx) < n_candidates:
            col_std = np.maximum(R_np.std(0), 1e-10)
            idx_std = max(index_np.std(), 1e-10)
            cov  = np.mean(
                (R_np - R_np.mean(0)) * (index_np - index_np.mean())[:, None],
                axis=0,
            )
            corr = np.abs(cov / (col_std * idx_std))
            for i in np.argsort(corr)[::-1]:
                top_idx.add(int(i))
                if len(top_idx) >= n_candidates:
                    break

        return sorted(top_idx)

    def _omp_select(self, R_np, index_np, candidate_indices):
        """
        OMP with precision-weighted scoring when use_lw_shrinkage=True.

        Standard OMP scores: |R_i' r| / ||R_i||
        Precision-weighted:  |R_i' Omega r| / sqrt(R_i' Omega R_i)
        where Omega = Sigma_LW^{-1}.

        This penalises assets that are highly collinear with already-selected
        ones, producing a more diversified and OOS-stable selection.
        Falls back to standard OMP if the precision matrix computation fails.
        """
        remaining = list(candidate_indices)
        selected  = []
        residual  = index_np.copy()

        # Precompute precision-weighted return matrix (or raw)
        if self.use_lw_shrinkage:
            try:
                Sigma = ledoit_wolf(R_np)
                eigv, eigvec = np.linalg.eigh(Sigma)
                eigv = np.maximum(eigv, 1e-10)
                # Omega^{1/2} = Sigma^{-1/2}
                Omega_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigv)) @ eigvec.T
                R_score = R_np @ Omega_sqrt   # T x N, precision-whitened
            except Exception:
                R_score = R_np
        else:
            R_score = R_np

        col_norms = np.linalg.norm(R_score[:, remaining], axis=0)
        col_norms = np.where(col_norms < 1e-10, 1.0, col_norms)

        for _ in range(self.K):
            if not remaining:
                break

            scores   = np.abs(R_score[:, remaining].T @ residual) / col_norms
            best_rel = int(np.argmax(scores))
            best_idx = remaining[best_rel]

            selected.append(best_idx)
            remaining.pop(best_rel)
            col_norms = np.delete(col_norms, best_rel)

            # Residual update always uses raw returns (unbiased)
            fitted   = _ols_project(R_np[:, selected], index_np)
            residual = index_np - fitted

        # Pad if OMP fell short
        if len(selected) < self.K and remaining:
            col_std = np.maximum(R_np[:, remaining].std(0), 1e-10)
            idx_std = max(index_np.std(), 1e-10)
            cov  = np.mean(
                (R_np[:, remaining] - R_np[:, remaining].mean(0))
                * (index_np - index_np.mean())[:, None], axis=0,
            )
            corr = np.abs(cov / (col_std * idx_std))
            for i in np.argsort(corr)[::-1]:
                selected.append(remaining[i])
                if len(selected) >= self.K:
                    break

        return selected[:self.K]

    def _select_assets(self, R, index_returns):
        R_np     = R.to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()
        N        = R_np.shape[1]

        if self.use_lasso_prefilter:
            n_cand        = min(self.prefilter_multiplier * self.K, N)
            candidate_idx = self._lasso_candidate_set(R_np, index_np, n_cand)
        else:
            candidate_idx = list(range(N))

        selected_idx = self._omp_select(R_np, index_np, candidate_idx)
        self.selected_assets = [R.columns[i] for i in selected_idx]
        return self.selected_assets

    def fit(self, R, index_returns):
        self._select_assets(R, index_returns)
        self.weights = self.refit_long_only_weights(R, index_returns,
                                                    self.selected_assets)


class LassoSectorModel(LassoModel):

    def __init__(self, K, sectors, market_caps, **kwargs):
        super().__init__(K, **kwargs)
        self.sectors     = sectors
        self.market_caps = market_caps

    def fit(self, R, index_returns):
        self._select_assets(R, index_returns)
        sc = self._build_sector_constraints(
            self.selected_assets, R.columns, self.sectors, self.market_caps, self.K
        )
        self.weights = self.refit_long_only_weights(
            R, index_returns, self.selected_assets, sector_constraints=sc
        )


class LassoModelNorm(BaseModel):
    """Normalised LASSO path — thesis comparison baseline only."""

    def __init__(self, K, sectors=None):
        super().__init__(K)

    def _solve(self, R_norm, idx_norm, alpha):
        N = R_norm.shape[1]
        w = cp.Variable(N)
        prob = cp.Problem(
            cp.Minimize(
                cp.sum_squares(idx_norm - R_norm @ w) + alpha * cp.norm1(w)
            ),
            [w >= 0],
        )
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            try:
                prob.solve(solver=cp.SCS)
            except Exception:
                return None
        if w.value is None:
            return None
        weights = np.nan_to_num(np.array(w.value).flatten(), nan=0.0)
        weights[weights < 0] = 0.0
        return weights

    def _select_assets(self, R, index_returns):
        R_np     = R.to_numpy(dtype=float)
        index_np = index_returns.to_numpy(dtype=float).flatten()
        col_std  = np.maximum(R_np.std(0), 1e-10)
        R_norm   = (R_np - R_np.mean(0)) / col_std
        idx_std  = max(index_np.std(), 1e-10)
        idx_norm = (index_np - index_np.mean()) / idx_std
        tol      = 1e-6 * col_std.max()
        lo, hi   = 1e-6, 1.0
        for _ in range(10):
            w_hi = self._solve(R_norm, idx_norm, hi)
            if w_hi is None or int(np.sum(w_hi > tol)) <= self.K:
                break
            hi *= 10.0
        for _ in range(30):
            mid   = np.sqrt(lo * hi)
            w_mid = self._solve(R_norm, idx_norm, mid)
            if w_mid is None:
                break
            if int(np.sum(w_mid > tol)) > self.K:
                lo = mid
            else:
                hi = mid
        w_best = self._solve(R_norm, idx_norm, hi)
        if w_best is None:
            w_best = np.ones(R_np.shape[1]) / R_np.shape[1]
        self.set_selected_from_weights(R.columns, w_best / col_std)
        return self.selected_assets

    def fit(self, R, index_returns):
        self._select_assets(R, index_returns)
        self.weights = self.refit_long_only_weights(R, index_returns,
                                                    self.selected_assets)