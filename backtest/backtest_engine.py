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
        os.makedirs("results/metrics", exist_ok=True)
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

    def run(self, models, K_list, train=500, test=21):

        summary_results = []

        T = len(self.returns)

        model_metrics_map = {model_name: [] for model_name in models}
        previous_weights_map = {(model_name, K): None for model_name in models for K in K_list}
        previous_assets_map = {(model_name, K): None for model_name in models for K in K_list}

        for start in range(0, T - train - test, test):

            train_slice = slice(start, start + train)
            test_slice = slice(start + train, start + train + test)

            R_train_full = self.returns.iloc[train_slice]
            R_test_full = self.returns.iloc[test_slice]

            I_train = self.index_returns.iloc[train_slice]
            I_test = self.index_returns.iloc[test_slice]

            R_train = R_train_full.dropna(axis=1)
            R_test = R_test_full[R_train.columns]

            for K in K_list:

                if len(R_train.columns) < K:
                    continue

                for model_name, model_class in models.items():

                    try:
                        model = model_class(K, self.sectors)
                    except TypeError:
                        model = model_class(K)

                    start_time = time.time()
                    model.fit(R_train, I_train)
                    exec_time = time.time() - start_time

                    assets = np.array(model.selected_assets)
                    weights = np.array(model.weights).flatten()

                    assert len(assets) == K, f"{model_name} returned {len(assets)} assets instead of {K}"

                    if len(assets) == 0 or len(weights) == 0:
                        continue

                    weights[weights < 0] = 0
                    if weights.sum() > 0:
                        weights = weights / weights.sum()

                    R_test_sel = R_test[assets]
                    portfolio_returns = R_test_sel.values @ weights

                    te_out = self.tracking_error(portfolio_returns, I_test.values)
                    te_ann = self.annualized_te(te_out)

                    previous_weights = previous_weights_map[(model_name, K)]
                    previous_assets = previous_assets_map[(model_name, K)]
                    if previous_weights is None or previous_assets is None:
                        turnover = 0
                    else:
                        turnover = self.compute_turnover(previous_assets, previous_weights, assets, weights)

                    diversification = self.diversification(weights)

                    export_assets = assets
                    export_weights = weights

                    portfolio_df = pd.DataFrame({
                        "asset": export_assets,
                        "weight": export_weights,
                        "benchmark_weight": self.benchmark_weights(export_assets) if len(export_assets) > 0 else []
                    })

                    if len(portfolio_df) > 0:
                        sector_values = self._resolve_series_values(self.sectors, portfolio_df["asset"].tolist(), "Unknown")
                        portfolio_df["sector"] = sector_values.values
                        portfolio_df = portfolio_df[["asset", "sector", "weight", "benchmark_weight"]]
                        portfolio_df = portfolio_df.sort_values("weight", ascending=False)

                    portfolio_file = f"results/portfolios/{model_name}_K{K}_t{start}.csv"
                    portfolio_df.to_csv(portfolio_file, index=False)

                    model_metrics_map[model_name].append({
                        "model": model_name,
                        "K": K,
                        "window": start,
                        "TE_out": te_out,
                        "TE_annual": te_ann,
                        "execution_time": exec_time,
                        "turnover": turnover,
                        "diversification": diversification
                    })

                    previous_weights_map[(model_name, K)] = weights
                    previous_assets_map[(model_name, K)] = assets

        for model_name, model_metrics in model_metrics_map.items():

            df_metrics = pd.DataFrame(model_metrics)
            df_metrics.to_csv(f"results/metrics/{model_name}_metrics.csv", index=False)

            if not df_metrics.empty:
                summary = df_metrics.groupby("K").mean().reset_index()
                summary["model"] = model_name
                summary_results.append(summary)

        if summary_results:
            final_summary = pd.concat(summary_results)
        else:
            final_summary = pd.DataFrame()

        final_summary.to_csv("results/summaries/model_comparison.csv", index=False)

        return final_summary