"""Layered MIP + sector constraints with correct Ledoit-Wolf shrinkage."""

import os
import time

import cvxpy as cp
import gurobipy as gp
import numpy as np

from .lw_shrinkage import shrunk_quadratic_form


class LayeredOptimization:

    def __init__(self, K, sectors, market_caps=None,
                 max_weight: float = 0.07,
                 time_limit: int = 180,
                 mip_gap: float = 0.005,
                 sector_penalty: float = 100.0,
                 use_lw_shrinkage: bool = True):
        self.K                = K
        self.sectors          = sectors
        self.market_caps      = market_caps
        self.max_weight       = max_weight
        self.time_limit       = time_limit
        self.mip_gap          = mip_gap
        self.sector_penalty   = sector_penalty
        self.use_lw_shrinkage = use_lw_shrinkage

        self.solve_time        = None
        self.mip_gap_achieved  = None
        self.obj_bound         = None
        self.is_optimal        = None
        self.sector_violations = {}
        self.selected_assets   = None
        self.weights           = None

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

    def _compute_bench_sector_weights(self, assets):
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

    def _warm_start_from_caps(self, w_var, x_var, assets):
        if self.market_caps is None:
            return
        caps  = np.array([self._cap_for_ticker(a) for a in assets])
        top_k = set(np.argsort(caps)[::-1][: self.K])
        for i in range(len(assets)):
            w_var[i].Start = (1.0 / self.K) if i in top_k else 0.0
            x_var[i].Start = 1.0 if i in top_k else 0.0

    def fit(self, R, index_returns):
        t0 = time.time()

        R      = R.dropna(axis=1)
        assets = list(R.columns)
        N      = len(assets)

        R_np   = R.to_numpy(dtype=float)
        idx_np = index_returns.to_numpy(dtype=float).flatten()

        # Correct LW shrinkage:
        #   Q = T * Sigma_LW  (replaces noisy R'R)
        #   c = -2 * R'I      (empirical, unchanged)
        Q, c = shrunk_quadratic_form(R_np, idx_np, self.use_lw_shrinkage)
        bench_sw   = self._compute_bench_sector_weights(assets)
        unique_sec = sorted(
            {self.sectors.get(self._resolve(a)) for a in assets} - {None}
        )
        sec_idx = {
            s: [i for i, a in enumerate(assets)
                if self.sectors.get(self._resolve(a)) == s]
            for s in unique_sec
        }

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        model.setParam("TimeLimit",  self.time_limit)
        model.setParam("MIPGap",     self.mip_gap)
        model.setParam("Threads",    max(1, os.cpu_count() - 1))
        model.setParam("MIPFocus",   1)

        w    = model.addMVar(N,     lb=0.0, ub=self.max_weight, name="w")
        x    = model.addMVar(N,     vtype=gp.GRB.BINARY,        name="x")
        n_s  = len(unique_sec)
        s_lo = model.addMVar(n_s,   lb=0.0, name="slack_lo")
        s_hi = model.addMVar(n_s,   lb=0.0, name="slack_hi")

        model.addConstr(w.sum() == 1.0,            name="budget")
        model.addConstr(x.sum() == float(self.K),  name="cardinality")
        model.addConstr(w <= self.max_weight * x,   name="link_ub")
        model.addConstr(w >= 1e-4 * x,              name="link_lb")

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

        slack_pen = self.sector_penalty * (s_lo.sum() + s_hi.sum())
        model.setObjective(w @ Q @ w + c @ w + slack_pen, gp.GRB.MINIMIZE)

        self._warm_start_from_caps(w, x, assets)
        model.optimize()

        self.solve_time = time.time() - t0

        if model.Status not in (2, 9) or model.SolCount == 0:
            model.dispose(); env.dispose()
            raise RuntimeError(f"Layered failed: status {model.Status}")

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

        sc = self._build_sector_constraints_for_refit(bench_sw)
        self.weights = self._refit(R, index_returns, sc)
        return self

    def _build_sector_constraints_for_refit(self, bench_sw):
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
        cons = [cp.sum(w) == 1, w >= 1e-4]

        penalty_terms = []
        for idx_list, b, delta in sector_constraints:
            sw    = cp.sum(w[idx_list])
            slack = cp.Variable(nonneg=True)
            cons += [sw >= b - delta - slack, sw <= b + delta + slack]
            penalty_terms.append(slack)

        obj = cp.sum_squares(idx_np - R_sel @ w)
        if penalty_terms:
            obj += self.sector_penalty * cp.sum(cp.hstack(penalty_terms))

        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL)

        if w.value is None:
            w2 = cp.Variable(n)
            cp.Problem(
                cp.Minimize(cp.sum_squares(idx_np - R_sel @ w2)),
                [cp.sum(w2) == 1, w2 >= 1e-4],
            ).solve(solver=cp.CLARABEL)
            result = np.array(w2.value).flatten() if w2.value is not None \
                     else np.ones(n) / n
        else:
            result = np.array(w.value).flatten()

        result[result < 0] = 0
        s = result.sum()
        return result / s if s > 0 else np.ones(n) / n