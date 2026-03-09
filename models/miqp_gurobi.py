"""MIQP model implementation using Gurobi."""
import gurobipy as gp
import numpy as np
from .base_model import BaseModel

class MIQPModel(BaseModel):

    def fit(self, R, index_returns):

        N = R.shape[1]

        R_np = R.values.astype(float)
        index_np = index_returns.values.flatten().astype(float)

        # Precompute matrix form: ||I - Rw||² = w'Qw + c'w + const
        # Q = R'R  (N×N),  c = -2 R'I  (N,)
        Q = R_np.T @ R_np
        c = -2.0 * (R_np.T @ index_np)

        model = gp.Model()
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", 300)   # max 2 min per window/K combo
        model.setParam("MIPGap", 0.01)     # stop at 1% optimality gap

        # Use MVar for efficient matrix-based objective
        w = model.addMVar(N, lb=0.0, ub=2.0 / self.K, name="w")
        x = model.addMVar(N, vtype=gp.GRB.BINARY, name="x")

        model.addConstr(w.sum() == 1.0)
        model.addConstr(x.sum() == float(self.K))
        model.addConstr(w <= (2.0 / self.K) * x)

        # Single matrix objective — no Python loop
        model.setObjective(w @ Q @ w + c @ w, gp.GRB.MINIMIZE)

        model.optimize()

        # Status 2 = optimal, 9 = time-limit with feasible incumbent
        if model.Status not in (2, 9) or model.SolCount == 0:
            # Fallback: pick top-K by correlation to index
            corr = np.corrcoef(R_np.T, index_np)[:-1, -1]
            selected_idx = np.argsort(corr)[-self.K:]
        else:
            selected_idx = np.where(x.X > 0.5)[0]
            if len(selected_idx) != self.K:
                selected_idx = np.argsort(w.X)[-self.K:]

        self.selected_assets = list(R.columns[selected_idx])
        self.weights = self.refit_long_only_weights(R, index_returns, self.selected_assets)