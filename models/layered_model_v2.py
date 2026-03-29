"""
LayeredOptimizationV2 — sub-1% OOS TE target.

The bug in the previous version
──────────────────────────────────
The previous v2 computed:
    R_tilde = R @ Sigma_sqrt
    Q = R_tilde' R_tilde = Sigma_sqrt @ R'R @ Sigma_sqrt
    c = -2 * R_tilde' @ I  = -2 * Sigma_sqrt @ R'I   ← WRONG

Applying Sigma_sqrt to c distorts the cross-covariance between returns and
the index.  The MIP then minimises a surrogate objective that is NOT tracking
error, so the selected assets do not minimise actual TE.  The refit partially
compensates but starts from a bad selection.

The correct fix
────────────────
Replace Q = R'R with its shrunk counterpart directly:

    Q_shrunk = T * Sigma_LW

where Sigma_LW is the OAS estimate of the asset return covariance.
This is identical to R'R in scale (both are T × Cov) but is full-rank.
c = -2 * R' @ I is left completely unchanged.

Result: the MIP selects assets that minimise the actual TE, and the refit
is consistent with the MIP selection.
"""

import os
import time

import cvxpy as cp
import gurobipy as gp
import numpy as np


def _ledoit_wolf_cov(R_np):
    """
    Return T * Sigma_LW  (same scale as R'R, units = return²).
    Uses sklearn OAS if available, falls back to manual LW.
    """
    T, N = R_np.shape
    try:
        from sklearn.covariance import OAS
        Sigma = OAS().fit(R_np).covariance_
        return T * Sigma
    except ImportError:
        pass

    S     = np.cov(R_np, rowvar=False, bias=False)
    mu    = np.trace(S) / N
    s2    = np.sum(S ** 2)
    b2    = 0.0
    m     = R_np.mean(0)
    for t in range(T):
        x  = R_np[t] - m
        b2 += np.sum((np.outer(x, x) - S) ** 2)
    b2 /= T ** 2
    b2    = min(b2, s2)
    delta = b2 / s2 if s2 > 0 else 0.0
    return T * ((1 - delta) * S + delta * mu * np.eye(N))


class LayeredOptimizationV2:

    def __init__(
        self,
        K,
        sectors,
        market_caps=None,
        max_weight: float = None,
        time_limit: int = 240,
        mip_gap: float = 0.003,
        sector_penalty: float = 100.0,
        use_lw_shrinkage: bool = True,
        turnover_budget: float = None,
        min_weight: float = 5e-4,
    ):
        self.K                = K
        self.sectors          = sectors
        self.market_caps      = market_caps
        self._max_weight_arg  = max_weight
        self.time_limit       = time_limit
        self.mip_gap          = mip_gap
        self.sector_penalty   = sector_penalty
        self.use_lw_shrinkage = use_lw_shrinkage
        self.turnover_budget  = turnover_budget
        self.min_weight       = min_weight

        self.solve_time        = None
        self.mip_gap_achieved  = None
        self.obj_bound         = None
        self.is_optimal        = None
        self.sector_violations = {}
        self.selected_assets   = None
        self.weights           = None

    @property
    def max_weight(self):
        if self._max_weight_arg is not None:
            return self._max_weight_arg
        return max(0.05, 2.0 / self.K)

    def _resolve(self, t):
        if t in self.sectors.index:
            return t
        for alt in (t.replace("-", "."), t.replace(".", "-")):
            if alt in self.sectors.index:
                return alt
        return t

    def _cap_for_ticker(self, ticker):
        if self.market_caps is None:
            return 0.0
        for cand in (ticker,
                     ticker.replace("-", ".") if "-" in ticker
                     else ticker.replace(".", "-")):
            if cand in self.market_caps.index:
                try:
                    v = float(self.market_caps.loc[cand])
                    if np.isfinite(v):
                        return v
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _bench_sector_weights(self, assets):
        if self.market_caps is None:
            return {}
        caps  = {a: self._cap_for_ticker(a) for a in assets}
        total = sum(caps.values())
        if total <= 0:
            return {}
        bench = {}
        for a, cap in caps.items():
            sec = self.sectors.get(self._resolve(a))
            if sec:
                bench[sec] = bench.get(sec, 0.0) + cap / total
        return bench

    @staticmethod
    def _adaptive_bound(b):
        if b < 0.05:
            return 0.015
        if b < 0.15:
            return 0.010
        return 0.020

    def _warm_start(self, w_var, x_var, assets):
        caps  = np.array([self._cap_for_ticker(a) for a in assets])
        top_k = set(np.argsort(caps)[::-1][: self.K])
        for i in range(len(assets)):
            w_var[i].Start = (1.0 / self.K) if i in top_k else 0.0
            x_var[i].Start = 1.0 if i in top_k else 0.0

    def fit(self, R, index_returns, w_prev=None):
        t0 = time.time()

        R      = R.dropna(axis=1)
        assets = list(R.columns)
        N      = len(assets)

        R_np   = R.to_numpy(dtype=float)
        idx_np = index_returns.to_numpy(dtype=float).flatten()

        # ── Quadratic form (THE FIX) ──────────────────────────────────────── #
        # Q = T * Sigma_LW  replaces R'R with a full-rank shrunk estimate.
        # c = -2 * R' @ I  is LEFT UNCHANGED — correct cross-covariance.
        if self.use_lw_shrinkage:
            Q = _ledoit_wolf_cov(R_np)      # T * Sigma_LW  (full-rank N×N)
        else:
            Q = R_np.T @ R_np

        c = -2.0 * (R_np.T @ idx_np)       # always raw — never shrunk

        # ── Sector structure ──────────────────────────────────────────────── #
        bench_sw   = self._bench_sector_weights(assets)
        unique_sec = sorted(
            {self.sectors.get(self._resolve(a)) for a in assets} - {None}
        )
        sec_idx = {
            s: [i for i, a in enumerate(assets)
                if self.sectors.get(self._resolve(a)) == s]
            for s in unique_sec
        }

        # ── Gurobi ────────────────────────────────────────────────────────── #
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        model.setParam("TimeLimit",   self.time_limit)
        model.setParam("MIPGap",      self.mip_gap)
        model.setParam("Threads",     max(1, os.cpu_count() - 1))
        model.setParam("MIPFocus",    1)
        model.setParam("Heuristics",  0.3)

        w    = model.addMVar(N, lb=0.0, ub=self.max_weight, name="w")
        x    = model.addMVar(N, vtype=gp.GRB.BINARY,        name="x")
        n_s  = len(unique_sec)
        s_lo = model.addMVar(n_s, lb=0.0, name="slack_lo")
        s_hi = model.addMVar(n_s, lb=0.0, name="slack_hi")

        model.addConstr(w.sum() == 1.0,           name="budget")
        model.addConstr(x.sum() == float(self.K), name="cardinality")
        model.addConstr(w <= self.max_weight * x,  name="link_ub")
        model.addConstr(w >= self.min_weight * x,  name="link_lb")

        for j, sec in enumerate(unique_sec):
            idx_s = sec_idx[sec]
            if not idx_s:
                continue
            sw = gp.quicksum(w[i] for i in idx_s)
            if sec in bench_sw:
                b     = bench_sw[sec]
                delta = self._adaptive_bound(b)
                model.addConstr(sw >= b - delta - s_lo[j], name=f"sec_lo_{j}")
                model.addConstr(sw <= b + delta + s_hi[j], name=f"sec_hi_{j}")
            else:
                model.addConstr(sw <= 0.30, name=f"sec_cap_{j}")

        if self.turnover_budget is not None and w_prev is not None:
            w_prev_vec = np.array([w_prev.get(a, 0.0) for a in assets])
            d = model.addMVar(N, lb=0.0, name="turn")
            model.addConstr(d >=  w - w_prev_vec, name="turn_pos")
            model.addConstr(d >= -w + w_prev_vec, name="turn_neg")
            model.addConstr(d.sum() <= self.turnover_budget, name="turn_bud")

        slack_pen = self.sector_penalty * (s_lo.sum() + s_hi.sum())
        model.setObjective(w @ Q @ w + c @ w + slack_pen, gp.GRB.MINIMIZE)

        self._warm_start(w, x, assets)
        model.optimize()

        self.solve_time = time.time() - t0

        if model.Status not in (2, 9) or model.SolCount == 0:
            model.dispose(); env.dispose()
            raise RuntimeError(f"LayeredV2 failed: status {model.Status}")

        self.is_optimal       = (model.Status == 2)
        self.mip_gap_achieved = model.MIPGap
        self.obj_bound        = model.ObjBound

        self.sector_violations = {
            sec: round(float(s_lo[j].X + s_hi[j].X), 4)
            for j, sec in enumerate(unique_sec)
            if float(s_lo[j].X + s_hi[j].X) > 1e-4
        }

        x_val = x.X
        w_val = w.X
        selected_idx = np.where(x_val > 0.5)[0]
        if len(selected_idx) != self.K:
            selected_idx = np.argsort(x_val)[::-1][: self.K]
        if len(selected_idx) != self.K:
            selected_idx = np.argsort(w_val)[::-1][: self.K]

        self.selected_assets = [assets[i] for i in selected_idx]
        model.dispose(); env.dispose()

        sc = self._build_refit_constraints(bench_sw)
        self.weights = self._refit(R, index_returns, sc)
        return self

    def _build_refit_constraints(self, bench_sw):
        result = []
        for sec, b in bench_sw.items():
            idx = [i for i, a in enumerate(self.selected_assets)
                   if self.sectors.get(self._resolve(a)) == sec]
            if idx:
                result.append((idx, b, self._adaptive_bound(b)))
        return result

    def _refit(self, R, index_returns, sector_constraints):
        R_sel  = R[self.selected_assets].to_numpy(dtype=float)
        idx_np = index_returns.to_numpy(dtype=float).flatten()
        n      = R_sel.shape[1]

        w    = cp.Variable(n)
        cons = [cp.sum(w) == 1, w >= self.min_weight]

        slacks = []
        for idx_list, b, delta in sector_constraints:
            sw    = cp.sum(w[idx_list])
            slack = cp.Variable(nonneg=True)
            cons += [sw >= b - delta - slack, sw <= b + delta + slack]
            slacks.append(slack)

        obj = cp.sum_squares(idx_np - R_sel @ w)
        if slacks:
            obj += self.sector_penalty * cp.sum(cp.hstack(slacks))

        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL)

        if w.value is None:
            w2 = cp.Variable(n)
            cp.Problem(
                cp.Minimize(cp.sum_squares(idx_np - R_sel @ w2)),
                [cp.sum(w2) == 1, w2 >= self.min_weight],
            ).solve(solver=cp.CLARABEL)
            result = np.array(w2.value).flatten() if w2.value is not None \
                     else np.ones(n) / n
        else:
            result = np.array(w.value).flatten()

        result[result < 0] = 0
        s = result.sum()
        return result / s if s > 0 else np.ones(n) / n