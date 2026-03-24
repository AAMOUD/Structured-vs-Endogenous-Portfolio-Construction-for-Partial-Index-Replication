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

    def benchmark_weights_series(self, assets):
        assets = list(assets)
        caps = self._resolve_series_values(self.market_caps, assets, np.nan)
        if caps.isna().all():
            return pd.Series(np.ones(len(assets)) / len(assets), index=assets, dtype=float)

        caps = caps.fillna(caps.median())
        total = caps.sum()
        if total <= 0:
            return pd.Series(np.ones(len(assets)) / len(assets), index=assets, dtype=float)

        return (caps / total).astype(float)

    @staticmethod
    def build_portfolio_series(universe_assets, selected_assets, selected_weights):
        portfolio = pd.Series(0.0, index=list(universe_assets), dtype=float)
        for a, w in zip(selected_assets, selected_weights):
            if a in portfolio.index:
                portfolio.loc[a] = float(w)
        total = portfolio.sum()
        if total > 0:
            portfolio = portfolio / total
        return portfolio

    def run(self, models, K_list, train_length=252 * 3, eval_length=63):

        required_rows = train_length + eval_length
        if len(self.returns) < required_rows or len(self.index_returns) < required_rows:
            raise ValueError(
                f"Not enough data for train_length={train_length} and eval_length={eval_length}"
            )

        R_train_raw = self.returns.iloc[-required_rows:-eval_length]
        I_train = self.index_returns.iloc[-required_rows:-eval_length]
        R_eval_raw = self.returns.iloc[-eval_length:]
        I_eval = self.index_returns.iloc[-eval_length:]

        common_train = R_train_raw.index.intersection(I_train.index)
        common_eval = R_eval_raw.index.intersection(I_eval.index)

        R_train = R_train_raw.loc[common_train].dropna(axis=1)
        I_train = I_train.loc[common_train]
        R_eval = R_eval_raw.loc[common_eval]
        I_eval = I_eval.loc[common_eval]

        print(
            f"Training window: {R_train.index[0].date()} -> {R_train.index[-1].date()} "
            f"({len(R_train)} days)"
        )
        print(
            f"Eval window: {R_eval.index[0].date()} -> {R_eval.index[-1].date()} "
            f"({len(R_eval)} days OOS)"
        )

        results = []

        for K in K_list:

            if len(R_train.columns) < K:
                print(f"Skipping K={K}: only {len(R_train.columns)} assets available")
                continue

            for model_name, model_class in models.items():

                print(f"  K={K:3d}  {model_name}...", end=" ", flush=True)

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

                    # In-sample tracking error (annualised)
                    R_sel    = R_train[assets].to_numpy(dtype=float)
                    port_ret = R_sel @ weights
                    idx_ret  = I_train.to_numpy(dtype=float).flatten()
                    te_annual = np.std(port_ret - idx_ret) * np.sqrt(252)

                    # Out-of-sample tracking error and returns
                    R_eval_sel = R_eval.reindex(columns=assets, fill_value=0.0)
                    port_oos = R_eval_sel.to_numpy(dtype=float) @ weights
                    idx_oos = I_eval.to_numpy(dtype=float).flatten()
                    te_oos_annual = float(np.std(port_oos - idx_oos) * np.sqrt(252))
                    port_oos_return = float(np.prod(1 + port_oos) - 1)
                    bench_oos_return = float(np.prod(1 + idx_oos) - 1)
                    tracking_diff_oos = port_oos_return - bench_oos_return

                    universe_assets = list(R_train.columns)
                    bench_full = self.benchmark_weights_series(universe_assets)
                    port_full = self.build_portfolio_series(universe_assets, assets, weights)

                    # Standard active share: 0.5 * ||w - b||_1 over full universe
                    asset_active_share = float(0.5 * np.sum(np.abs(port_full.values - bench_full.values)))

                    bw = bench_full.reindex(assets, fill_value=0.0).values

                    # Normalised diversification: N_eff / K  (0=concentrated, 1=equal weight)
                    n_eff = self.diversification(weights)        # 1/sum(w²)
                    div_norm = n_eff / K

                    # Benchmark diversification
                    bench_full_values = bench_full.to_numpy(dtype=float)
                    bench_n_eff = 1.0 / np.sum(bench_full_values ** 2) if np.sum(bench_full_values ** 2) > 0 else 0.0
                    bench_div_norm = bench_n_eff / len(universe_assets)

                    # Portfolio CSV
                    sector_vals = self._resolve_series_values(
                        self.sectors, assets.tolist(), "Unknown"
                    )
                    portfolio_df = pd.DataFrame({
                        "asset":            assets,
                        "sector":           sector_vals.values,
                        "weight":           weights,
                        "benchmark_weight": bw
                    }).sort_values("weight", ascending=False)

                    # Sector active share over full benchmark universe
                    sector_universe = self._resolve_series_values(
                        self.sectors, universe_assets, "Unknown"
                    )
                    sector_port = (
                        pd.DataFrame({"sector": sector_universe.values, "weight": port_full.values})
                        .groupby("sector")["weight"].sum()
                    )
                    sector_bench = (
                        pd.DataFrame({"sector": sector_universe.values, "weight": bench_full.values})
                        .groupby("sector")["weight"].sum()
                    )
                    all_sectors  = sector_port.index.union(sector_bench.index)
                    sector_active_share = float(
                        0.5 * np.sum(np.abs(
                            sector_port.reindex(all_sectors, fill_value=0).values -
                            sector_bench.reindex(all_sectors, fill_value=0).values
                        ))
                    )

                    portfolio_file = f"results/portfolios/{model_name}_K{K}.csv"
                    portfolio_df.to_csv(portfolio_file, index=False)

                    results.append({
                        "model":                 model_name,
                        "K":                     K,
                        "train_length_days":     len(R_train),
                        "eval_length_days":      len(R_eval),
                        "TE_annual":             te_annual,
                        "TE_annual_oos":         te_oos_annual,
                        "portfolio_return_oos":  port_oos_return,
                        "benchmark_return_oos":  bench_oos_return,
                        "tracking_difference_oos": tracking_diff_oos,
                        "asset_active_share":    asset_active_share,
                        "sector_active_share":   sector_active_share,
                        "diversification":       div_norm,
                        "bench_diversification": bench_div_norm,
                        "execution_time":        exec_time
                    })

                    print(
                        f"done ({exec_time:.1f}s  TE_IS={te_annual:.4f}  "
                        f"TE_OOS={te_oos_annual:.4f}  AS={asset_active_share:.3f})"
                    )

                except Exception as e:
                    print(f"ERROR: {e}")

        results_df = pd.DataFrame(results)
        results_df.to_csv("results/summaries/experiment_results.csv", index=False)
        results_df.to_excel("results/summaries/experiment_results.xlsx", index=False)

        return results_df