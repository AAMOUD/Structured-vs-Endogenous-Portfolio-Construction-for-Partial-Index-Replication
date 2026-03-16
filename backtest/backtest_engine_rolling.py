import numpy as np
import pandas as pd
import time
import os


class BacktestEngineRolling:
    """
    Rolling-window backtest engine.

    Training window : 252 trading days (≈ 1 year) ending the day before each
                      quarter start.  The first window covers calendar year 2022,
                      so the first evaluated period is Q1 2023.
    Evaluation      : out-of-sample performance over each calendar quarter.
    Rebalancing     : quarterly – one portfolio per (model, K, quarter).

    Universe filter : at most 10 assets are dropped per training window.  Any
                      asset whose return coverage falls below 90 % in the window
                      is removed, but if that would remove more than 10 assets
                      only the 10 worst-covered ones are discarded.
    """

    def __init__(self, returns, index_returns, sectors, market_caps):
        self.returns = returns
        self.index_returns = index_returns
        self.sectors = sectors
        self.market_caps = market_caps

    # ------------------------------------------------------------------ #
    #  Universe helpers                                                    #
    # ------------------------------------------------------------------ #

    def filter_universe(self, R_window, max_drop=10):
        """
        Return (valid_columns, n_dropped).

        Drops assets below 90 % return coverage, but never more than
        `max_drop` assets.  When the number of under-covered assets exceeds
        `max_drop`, only the `max_drop` worst ones are removed.
        """
        coverage = R_window.notna().mean().sort_values()   # ascending
        threshold = 0.90
        below = coverage[coverage < threshold]

        if len(below) == 0:
            return R_window.columns.tolist(), 0

        if len(below) <= max_drop:
            drop_set = set(below.index.tolist())
        else:
            # Only remove the max_drop worst covered
            drop_set = set(coverage.index[:max_drop].tolist())

        valid = [c for c in R_window.columns if c not in drop_set]
        return valid, len(drop_set)

    # ------------------------------------------------------------------ #
    #  Portfolio / metric helpers  (identical to BacktestEngine)          #
    # ------------------------------------------------------------------ #

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
                str(asset).replace(".", "-"),
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
        return (caps / total).values

    def compute_turnover(self, prev_assets, prev_w, assets, w):
        all_assets = np.union1d(prev_assets, assets)
        prev_map = dict(zip(prev_assets, prev_w))
        new_map = dict(zip(assets, w))
        prev_vec = np.array([prev_map.get(a, 0.0) for a in all_assets])
        new_vec = np.array([new_map.get(a, 0.0) for a in all_assets])
        return float(np.sum(np.abs(new_vec - prev_vec)))

    # ------------------------------------------------------------------ #
    #  Main run loop                                                       #
    # ------------------------------------------------------------------ #

    def run(self, models, K_list, output_dir="results_rolling"):
        os.makedirs(f"{output_dir}/portfolios", exist_ok=True)
        os.makedirs(f"{output_dir}/summaries",  exist_ok=True)

        # ── Build quarterly evaluation periods ────────────────────────── #
        # First quarter: Q1 2023 (trained on 2022).
        # Last quarter : the last complete quarter within the available data.
        first_eval = pd.Timestamp("2023-01-01")
        last_date  = self.returns.index[-1]

        # Generate quarter-start timestamps; add one extra so the last
        # quarter has a defined end boundary.
        all_qs = pd.date_range(
            start=first_eval,
            end=last_date + pd.DateOffset(months=3),
            freq="QS",
        )

        results        = []
        prev_portfolio = {}   # (model_name, K) -> {'assets': list, 'weights': ndarray}

        for i in range(len(all_qs) - 1):
            q_start = all_qs[i]
            q_end   = min(all_qs[i + 1] - pd.Timedelta(days=1), last_date)

            month = q_start.month
            quarter_label = f"{q_start.year}-Q{(month - 1) // 3 + 1}"

            # ── Training slice: 252 trading days ending just before q_start ── #
            idx_before = self.returns.index.searchsorted(q_start, side="left") - 1
            if idx_before < 251:
                print(f"[{quarter_label}] Not enough training data, skipping.")
                continue

            train_slice = self.returns.iloc[idx_before - 251 : idx_before + 1]   # 252 rows
            I_train     = self.index_returns.iloc[idx_before - 251 : idx_before + 1]

            # Align index (safety)
            common_idx  = train_slice.index.intersection(I_train.index)
            train_slice = train_slice.loc[common_idx]
            I_train     = I_train.loc[common_idx]

            train_start = train_slice.index[0]
            train_end   = train_slice.index[-1]

            # ── Evaluation slice ──────────────────────────────────────── #
            eval_mask = (self.returns.index >= q_start) & (self.returns.index <= q_end)
            R_eval    = self.returns.loc[eval_mask]
            I_eval    = self.index_returns.loc[eval_mask]

            if len(R_eval) == 0:
                print(f"[{quarter_label}] No evaluation data, skipping.")
                continue

            # ── Universe filter ───────────────────────────────────────── #
            valid_assets, n_dropped = self.filter_universe(train_slice, max_drop=10)
            R_train = train_slice[valid_assets]
            # Forward-fill then zero-fill residual NaNs so models can run
            R_train = R_train.ffill().fillna(0.0)

            print(
                f"\n[{quarter_label}]  "
                f"train: {train_start.date()} → {train_end.date()}  "
                f"eval: {q_start.date()} → {q_end.date()}  "
                f"universe: {len(valid_assets)} assets  ({n_dropped} dropped)"
            )

            os.makedirs(f"{output_dir}/portfolios/{quarter_label}", exist_ok=True)

            for K in K_list:
                if len(valid_assets) < K:
                    print(f"  K={K}: only {len(valid_assets)} assets available, skipping")
                    continue

                for model_name, model_class in models.items():
                    print(f"  K={K:3d}  {model_name}...", end=" ", flush=True)

                    try:
                        try:
                            model = model_class(K, self.sectors)
                        except TypeError:
                            model = model_class(K)

                        t0       = time.time()
                        model.fit(R_train, I_train)
                        exec_time = time.time() - t0

                        assets  = np.array(model.selected_assets)
                        weights = np.array(model.weights).flatten()

                        assert len(assets) == K, (
                            f"{model_name} returned {len(assets)} assets instead of {K}"
                        )

                        weights[weights < 0] = 0
                        if weights.sum() > 0:
                            weights /= weights.sum()

                        # ── In-sample metrics ──────────────────────────── #
                        R_sel_is  = R_train[assets].to_numpy(dtype=float)
                        port_is   = R_sel_is @ weights
                        idx_is    = I_train.to_numpy(dtype=float).flatten()
                        te_is     = float(np.std(port_is - idx_is) * np.sqrt(252))

                        bw = self.benchmark_weights(assets)

                        # Asset-level active share vs cap-weighted benchmark
                        asset_active_share = float(np.sum(np.abs(weights - bw)))

                        # Normalised effective-N diversification
                        n_eff        = self.diversification(weights)
                        div_norm     = n_eff / K
                        bench_n_eff  = (1.0 / np.sum(bw ** 2)) if np.sum(bw ** 2) > 0 else 0.0
                        bench_div_norm = bench_n_eff / K

                        # Sector active share
                        sector_vals  = self._resolve_series_values(
                            self.sectors, assets.tolist(), "Unknown"
                        )
                        portfolio_df = pd.DataFrame({
                            "asset":            assets,
                            "sector":           sector_vals.values,
                            "weight":           weights,
                            "benchmark_weight": bw,
                        }).sort_values("weight", ascending=False)

                        sp   = portfolio_df.groupby("sector")["weight"].sum()
                        sb   = portfolio_df.groupby("sector")["benchmark_weight"].sum()
                        allS = sp.index.union(sb.index)
                        sector_active_share = float(
                            np.sum(np.abs(
                                sp.reindex(allS, fill_value=0).values -
                                sb.reindex(allS, fill_value=0).values
                            ))
                        )

                        portfolio_df.to_csv(
                            f"{output_dir}/portfolios/{quarter_label}/{model_name}_K{K}.csv",
                            index=False,
                        )

                        # ── Out-of-sample metrics ──────────────────────── #
                        # Assets absent from the eval window receive 0 return
                        R_eval_sel  = R_eval.reindex(columns=assets, fill_value=0.0)
                        port_oos    = R_eval_sel.to_numpy(dtype=float) @ weights
                        idx_oos     = I_eval.to_numpy(dtype=float).flatten()

                        te_oos            = float(np.std(port_oos - idx_oos) * np.sqrt(252))
                        port_cum_return   = float(np.prod(1 + port_oos) - 1)
                        bench_cum_return  = float(np.prod(1 + idx_oos) - 1)
                        tracking_diff     = port_cum_return - bench_cum_return

                        # ── Turnover vs previous quarter ───────────────── #
                        key = (model_name, K)
                        if key in prev_portfolio:
                            turnover = self.compute_turnover(
                                prev_portfolio[key]["assets"],
                                prev_portfolio[key]["weights"],
                                assets.tolist(),
                                weights,
                            )
                        else:
                            turnover = np.nan
                        prev_portfolio[key] = {
                            "assets":  assets.tolist(),
                            "weights": weights.copy(),
                        }

                        results.append({
                            "model":                  model_name,
                            "K":                      K,
                            "quarter":                quarter_label,
                            "train_start":            train_start.date(),
                            "train_end":              train_end.date(),
                            "eval_start":             q_start.date(),
                            "eval_end":               q_end.date(),
                            "universe_size":          len(valid_assets),
                            "assets_dropped":         n_dropped,
                            "TE_annual_insample":     te_is,
                            "TE_annual_oos":          te_oos,
                            "portfolio_return_oos":   port_cum_return,
                            "benchmark_return_oos":   bench_cum_return,
                            "tracking_difference":    tracking_diff,
                            "asset_active_share":     asset_active_share,
                            "sector_active_share":    sector_active_share,
                            "diversification":        div_norm,
                            "bench_diversification":  bench_div_norm,
                            "turnover":               turnover,
                            "execution_time":         exec_time,
                        })

                        print(
                            f"done ({exec_time:.1f}s  "
                            f"TE_IS={te_is:.4f}  "
                            f"TE_OOS={te_oos:.4f}  "
                            f"ret_OOS={port_cum_return:+.3f})"
                        )

                    except Exception as e:
                        print(f"ERROR: {e}")

        results_df = pd.DataFrame(results)
        out_csv   = f"{output_dir}/summaries/experiment_results_rolling.csv"
        out_xlsx  = f"{output_dir}/summaries/experiment_results_rolling.xlsx"
        results_df.to_csv(out_csv,   index=False)
        results_df.to_excel(out_xlsx, index=False)

        print(f"\nResults saved → {out_csv}")
        return results_df
