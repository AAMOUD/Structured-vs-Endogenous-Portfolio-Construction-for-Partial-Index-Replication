import numpy as np
import pandas as pd
import time
import os


class BacktestEngine:

    def __init__(self, returns, index_returns, sectors, market_caps):

        self.returns = returns
        self.index_returns = index_returns
        self.sectors = sectors
        self.market_caps = market_caps

        os.makedirs("results/portfolios", exist_ok=True)
        os.makedirs("results/summaries", exist_ok=True)

    def tracking_error(self, portfolio, benchmark):

        return np.std(portfolio - benchmark)

    def annualized_te(self, te):

        return te * np.sqrt(252)

    def turnover(self, w_old, w_new):

        return np.sum(np.abs(w_new - w_old))

    def compute_turnover(self, prev_assets, prev_w, assets, w):

        all_assets = np.union1d(prev_assets, assets)

        prev_map = dict(zip(prev_assets, prev_w))
        new_map = dict(zip(assets, w))

        prev_vec = np.array([prev_map.get(asset, 0.0) for asset in all_assets])
        new_vec = np.array([new_map.get(asset, 0.0) for asset in all_assets])

        return np.sum(np.abs(new_vec - prev_vec))

    def diversification(self, weights):

        denom = np.sum(weights ** 2)
        if denom <= 0:
            return 0.0
        return 1 / denom

    def _resolve_series_values(self, series, assets, default_value):

        values = []
        index = series.index

        for asset in assets:
            if asset in index:
                values.append(series.loc[asset])
                continue

            alternatives = {
                str(asset).replace("-", "."),
                str(asset).replace(".", "-")
            }

            found = False
            for alt in alternatives:
                if alt in index:
                    values.append(series.loc[alt])
                    found = True
                    break

            if not found:
                values.append(default_value)

        return pd.Series(values, index=assets)

    def benchmark_weights(self, assets):

        caps = self._resolve_series_values(self.market_caps, assets, np.nan)
        if caps.isna().all():
            return np.ones(len(assets)) / len(assets)

        caps = caps.fillna(caps.median())

        total = caps.sum()
        if total <= 0:
            return np.ones(len(assets)) / len(assets)

        weights = caps / total
        return weights.values

    def run(self, models, K_list):

        horizons = {
            "1y":   252,
            "5y":   252 * 5,
            "10y":  252 * 10,
            "full": len(self.returns)
        }

        results = []

        for horizon_name, horizon_length in horizons.items():

            print(f"\n=== Horizon: {horizon_name} ({horizon_length} days) ===")

            R_train = self.returns.iloc[-horizon_length:].dropna(axis=1)
            I_train = self.index_returns.iloc[-horizon_length:]

            for K in K_list:

                if len(R_train.columns) < K:
                    print(f"  Skipping K={K}: only {len(R_train.columns)} assets available")
                    continue

                for model_name, model_class in models.items():

                    print(f"  [{horizon_name}] K={K:3d}  {model_name}...", end=" ", flush=True)

                    try:
                        try:
                            model = model_class(K, self.sectors)
                        except TypeError:
                            model = model_class(K)

                        start_time = time.time()
                        model.fit(R_train, I_train)
                        exec_time = time.time() - start_time

                        assets  = np.array(model.selected_assets)
                        weights = np.array(model.weights).flatten()

                        assert len(assets) == K, (
                            f"{model_name} returned {len(assets)} assets instead of {K}"
                        )

                        weights[weights < 0] = 0
                        if weights.sum() > 0:
                            weights = weights / weights.sum()

                        # In-sample tracking error
                        R_sel = R_train[assets].to_numpy(dtype=float)
                        port_ret = R_sel @ weights
                        idx_ret  = I_train.to_numpy(dtype=float).flatten()

                        te      = np.std(port_ret - idx_ret)
                        te_ann  = te * np.sqrt(252)

                        # Portfolio CSV
                        bw = self.benchmark_weights(assets)
                        sector_vals = self._resolve_series_values(
                            self.sectors, assets.tolist(), "Unknown"
                        )
                        portfolio_df = pd.DataFrame({
                            "asset":            assets,
                            "sector":           sector_vals.values,
                            "weight":           weights,
                            "benchmark_weight": bw
                        }).sort_values("weight", ascending=False)

                        portfolio_file = (
                            f"results/portfolios/{horizon_name}_{model_name}_K{K}.csv"
                        )
                        portfolio_df.to_csv(portfolio_file, index=False)

                        results.append({
                            "horizon":        horizon_name,
                            "model":          model_name,
                            "K":              K,
                            "TE":             te,
                            "TE_annual":      te_ann,
                            "diversification": self.diversification(weights),
                            "execution_time": exec_time
                        })

                        print(f"done ({exec_time:.1f}s  TE={te_ann:.4f})")

                    except Exception as e:
                        print(f"ERROR: {e}")

        results_df = pd.DataFrame(results)
        results_df.to_csv("results/summaries/experiment_results.csv", index=False)

        return results_df