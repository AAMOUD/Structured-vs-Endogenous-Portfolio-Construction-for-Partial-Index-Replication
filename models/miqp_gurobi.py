"""MIQP model implementation using Gurobi (improved)."""

import os
import time as _time

import gurobipy as gp
import numpy as np
import pandas as pd

from .base_model import BaseModel


class MIQPModel(BaseModel):

    def __init__(self, K, sectors=None, max_weight: float = 0.07,
                 time_limit: int = 120, mip_gap: float = 0.005):
        super().__init__(K)
        self.max_weight = max_weight
        self.time_limit = time_limit
        self.mip_gap = mip_gap

        # Solution-quality metadata (populated after fit)
        self.solve_time = None
        self.mip_gap_achieved = None
        self.obj_bound = None
        self.is_optimal = None

    def _warm_start(self, w_var, x_var, market_caps, R_columns, n_assets):
        """Warm start from top-K market cap names (equal-weighted)."""
        if market_caps is None:
            return

        def _cap_for_ticker(ticker):
            candidates = [
                ticker,
                ticker.replace("-", ".") if "-" in ticker else ticker.replace(".", "-"),
            ]
            for cand in candidates:
                if cand in market_caps.index:
                    val = market_caps.loc[cand]
                    if pd.notna(val):
                        try:
                            return float(val)
                        except (TypeError, ValueError):
                            continue
            return 0.0

        caps = np.array([
            _cap_for_ticker(t)
            for t in R_columns
        ])

        top_k = np.argsort(caps)[::-1][: self.K]
        w_init = np.zeros(n_assets)
        w_init[top_k] = 1.0 / self.K
        x_init = np.zeros(n_assets)
        x_init[top_k] = 1.0

        for i in range(n_assets):
            w_var[i].Start = w_init[i]
            x_var[i].Start = x_init[i]

    def fit(self, R, index_returns, w_prev=None, market_caps=None,
            turnover_penalty: float = 0.0):
        if self.K * self.max_weight < 1.0:
            raise ValueError(
                "Infeasible setup: K * max_weight must be >= 1. "
                f"Received K={self.K}, max_weight={self.max_weight}."
            )

        n_assets = R.shape[1]
        R_np = R.values.astype(float)
        idx_np = index_returns.values.flatten().astype(float)

        # ||I - Rw||^2 = w'Qw + c'w + const
        Q = R_np.T @ R_np
        c = -2.0 * (R_np.T @ idx_np)

        if turnover_penalty > 0 and w_prev is not None:
            w_prev_vec = np.array([w_prev.get(t, 0.0) for t in R.columns])
        else:
            w_prev_vec = np.zeros(n_assets)
            turnover_penalty = 0.0

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)

        model.setParam("TimeLimit", self.time_limit)
        model.setParam("MIPGap", self.mip_gap)
        model.setParam("Threads", max(1, os.cpu_count() - 1))

        w = model.addMVar(n_assets, lb=0.0, ub=self.max_weight, name="w")
        x = model.addMVar(n_assets, vtype=gp.GRB.BINARY, name="x")

        model.addConstr(w.sum() == 1.0, name="budget")
        model.addConstr(x.sum() == float(self.K), name="cardinality")
        model.addConstr(w <= self.max_weight * x, name="linking_ub")
        model.addConstr(w >= 1e-4 * x, name="linking_lb")

        if turnover_penalty > 0:
            d = model.addMVar(n_assets, lb=0.0, name="abs_diff")
            model.addConstr(d >= w - w_prev_vec, name="d_pos")
            model.addConstr(d >= w_prev_vec - w, name="d_neg")
            model.setObjective(
                w @ Q @ w + c @ w + turnover_penalty * d.sum(),
                gp.GRB.MINIMIZE,
            )
        else:
            model.setObjective(w @ Q @ w + c @ w, gp.GRB.MINIMIZE)

        self._warm_start(w, x, market_caps, R.columns, n_assets)

        t0 = _time.time()
        model.optimize()
        self.solve_time = _time.time() - t0

        if model.Status not in (2, 9) or model.SolCount == 0:
            status = int(model.Status)
            sol_count = int(model.SolCount)
            model.dispose()
            env.dispose()
            raise RuntimeError(
                f"MIQP failed: Gurobi status {status} "
                f"(SolCount={sol_count})"
            )

        self.is_optimal = (model.Status == 2)
        self.mip_gap_achieved = model.MIPGap
        self.obj_bound = model.ObjBound

        x_val = x.X
        w_val = w.X

        selected_idx = np.where(x_val > 0.5)[0]
        if len(selected_idx) != self.K:
            selected_idx = np.argsort(x_val)[::-1][: self.K]
        if len(selected_idx) != self.K:
            selected_idx = np.argsort(w_val)[::-1][: self.K]

        self.selected_assets = list(R.columns[selected_idx])
        self.weights = self.refit_long_only_weights(R, index_returns, self.selected_assets)

        model.dispose()
        env.dispose()
        return self
