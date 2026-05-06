import inspect
import os
import time
import warnings

import numpy as np
import pandas as pd


class BacktestEngineRolling:

    def __init__(self, returns, index_returns, sectors, market_caps):
        self.returns       = returns
        self.index_returns = index_returns
        self.sectors       = sectors
        self.market_caps   = market_caps

    # ── Universe helpers ─────────────────────────────────────────────────── #

    def filter_universe(self, R_window, threshold=0.90):
        """
        Keep assets whose non-NaN coverage over the training window exceeds
        `threshold`.  No arbitrary cap on how many can be dropped — we let
        the data decide.  After forward-filling, any column that still has a
        NaN (i.e. it was NaN from the very first row) is also dropped.
        """
        coverage    = R_window.notna().mean()
        valid_cols  = coverage[coverage >= threshold].index.tolist()
        n_dropped   = len(R_window.columns) - len(valid_cols)
        return valid_cols, n_dropped

    def _resolve_series_values(self, series, assets, default_value):
        values = []
        idx    = series.index
        for asset in assets:
            if asset in idx:
                values.append(series.loc[asset])
                continue
            for alt in (str(asset).replace("-", "."), str(asset).replace(".", "-")):
                if alt in idx:
                    values.append(series.loc[alt])
                    break
            else:
                values.append(default_value)
        return pd.Series(values, index=assets)

    def _full_universe_bench_weights(self, universe_assets):
        caps  = self._resolve_series_values(self.market_caps, universe_assets, np.nan)
        caps  = caps.fillna(0.0)
        total = caps.sum()
        if total <= 0:
            return pd.Series(
                np.ones(len(universe_assets)) / len(universe_assets),
                index=universe_assets,
            )
        return caps / total

    # ── Metrics ──────────────────────────────────────────────────────────── #

    @staticmethod
    def diversification(weights):
        denom = np.sum(weights ** 2)
        return 1.0 / denom if denom > 0 else 0.0

    @staticmethod
    def compute_turnover(prev_assets, prev_w, assets, w):
        all_assets = np.union1d(prev_assets, assets)
        prev_map   = dict(zip(prev_assets, prev_w))
        new_map    = dict(zip(assets, w))
        prev_vec   = np.array([prev_map.get(a, 0.0) for a in all_assets])
        new_vec    = np.array([new_map.get(a, 0.0) for a in all_assets])
        return float(np.sum(np.abs(new_vec - prev_vec)))

    @staticmethod
    def information_ratio(port_ret, idx_ret, min_obs=20):
        """
        Annualised Information Ratio.
        min_obs=20 so quarterly windows (~63 days) are not all NaN.
        Use the full-period concatenation for the definitive thesis metric.
        """
        active = port_ret - idx_ret
        if len(active) < min_obs:
            return np.nan
        vol = np.std(active)
        if vol == 0:
            return np.nan
        return float(np.mean(active) / vol * np.sqrt(252))

    @staticmethod
    def sharpe_ratio(port_ret, risk_free=0.0):
        excess = port_ret - risk_free / 252
        vol    = np.std(excess)
        if vol == 0:
            return np.nan
        return float(np.mean(excess) / vol * np.sqrt(252))

    @staticmethod
    def max_drawdown_te(port_ret, idx_ret):
        active  = port_ret - idx_ret
        cum     = np.cumsum(active)
        roll_max = np.maximum.accumulate(cum)
        dd      = cum - roll_max
        return float(dd.min()) if len(dd) > 0 else np.nan

    @staticmethod
    def composition_stability(prev_assets, curr_assets):
        if prev_assets is None or len(prev_assets) == 0:
            return np.nan
        overlap = len(set(curr_assets) & set(prev_assets))
        return overlap / len(curr_assets)

    def compute_active_share(self, selected_assets, weights, bench_weights_series):
        w_full = bench_weights_series.copy() * 0.0
        for a, wt in zip(selected_assets, weights):
            if a in w_full.index:
                w_full[a] = wt
            else:
                alt = a.replace("-", ".") if "-" in a else a.replace(".", "-")
                if alt in w_full.index:
                    w_full[alt] = wt
        b_full = bench_weights_series.reindex(w_full.index, fill_value=0.0)
        return float(0.5 * np.sum(np.abs(w_full.values - b_full.values)))

    def compute_sector_active_share(self, selected_assets, weights,
                                     bench_weights_series, universe_assets):
        sec_series = self._resolve_series_values(self.sectors, universe_assets, "Unknown")
        bench_sec  = bench_weights_series.groupby(sec_series.values).sum()

        port_w_full = pd.Series(0.0, index=universe_assets)
        for a, wt in zip(selected_assets, weights):
            if a in port_w_full.index:
                port_w_full[a] = wt
            else:
                alt = a.replace("-", ".") if "-" in a else a.replace(".", "-")
                if alt in port_w_full.index:
                    port_w_full[alt] = wt

        port_sec = port_w_full.groupby(sec_series.values).sum()
        all_sec  = bench_sec.index.union(port_sec.index)
        return float(0.5 * np.sum(np.abs(
            port_sec.reindex(all_sec, fill_value=0.0).values
            - bench_sec.reindex(all_sec, fill_value=0.0).values
        )))

    # ── Main loop ────────────────────────────────────────────────────────── #

    def run(self, models, K_list, output_dir="results_rolling", train_length=504,
            universe_threshold=0.90, miqp_turnover_penalty=0.0):

        os.makedirs(f"{output_dir}/portfolios", exist_ok=True)
        os.makedirs(f"{output_dir}/summaries",  exist_ok=True)
        os.makedirs(f"{output_dir}/paths",       exist_ok=True)

        first_eval = pd.Timestamp("2023-01-01")
        last_date  = self.returns.index[-1]

        all_qs = pd.date_range(
            start=first_eval,
            end=last_date + pd.DateOffset(months=3),
            freq="QS",
        )

        results          = []
        prev_portfolio   = {}
        oos_daily_by_key = {}

        for i in range(len(all_qs) - 1):
            q_start       = all_qs[i]
            q_end         = min(all_qs[i + 1] - pd.Timedelta(days=1), last_date)
            quarter_label = f"{q_start.year}-Q{(q_start.month - 1) // 3 + 1}"

            idx_before = self.returns.index.searchsorted(q_start, side="left") - 1
            if idx_before < (train_length - 1):
                print(f"[{quarter_label}] Not enough training data, skipping.")
                continue

            train_slice = self.returns.iloc[idx_before - (train_length - 1): idx_before + 1]
            I_train     = self.index_returns.iloc[idx_before - (train_length - 1): idx_before + 1]

            common_idx  = train_slice.index.intersection(I_train.index)
            train_slice = train_slice.loc[common_idx]
            I_train     = I_train.loc[common_idx]

            train_start = train_slice.index[0]
            train_end   = train_slice.index[-1]

            eval_mask = (self.returns.index >= q_start) & (self.returns.index <= q_end)
            R_eval    = self.returns.loc[eval_mask]
            I_eval    = self.index_returns.loc[eval_mask]

            if len(R_eval) == 0:
                print(f"[{quarter_label}] No evaluation data, skipping.")
                continue

            # ── Universe filtering ────────────────────────────────────────── #
            # Step 1: drop low-coverage columns (< 90% non-NaN over train window)
            valid_cols, n_dropped = self.filter_universe(train_slice, threshold=universe_threshold)
            # Step 2: forward-fill then drop any remaining NaN columns
            #         (fillna(0) was wrong — zero return ≠ missing data)
            R_train     = train_slice[valid_cols].ffill().dropna(axis=1)
            valid_assets = R_train.columns.tolist()   # update after dropna

            bench_w_series = self._full_universe_bench_weights(valid_assets)

            total_dropped = len(self.returns.columns) - len(valid_assets)
            print(
                f"\n[{quarter_label}] "
                f"train: {train_start.date()} -> {train_end.date()} "
                f"eval:  {q_start.date()} -> {q_end.date()} "
                f"universe: {len(valid_assets)} assets "
                f"({total_dropped} dropped, threshold={universe_threshold:.2f})"
            )

            os.makedirs(f"{output_dir}/portfolios/{quarter_label}", exist_ok=True)
            os.makedirs(f"{output_dir}/paths/{quarter_label}",       exist_ok=True)

            quarter_rows = []

            for K in K_list:
                if len(valid_assets) < K:
                    print(f"  K={K}: only {len(valid_assets)} assets available, skipping")
                    continue

                for model_name, model_class in models.items():
                    print(f"  K={K:3d}  {model_name}...", end=" ", flush=True)

                    try:
                        # ── Instantiate ──────────────────────────────────── #
                        try:
                            model = model_class(K, self.sectors)
                        except TypeError:
                            model = model_class(K)

                        # ── Fit ──────────────────────────────────────────── #
                        fit_sig    = inspect.signature(model.fit)
                        fit_kwargs = {}
                        key        = (model_name, K)
                        prev       = prev_portfolio.get(key)

                        if "market_caps"      in fit_sig.parameters:
                            fit_kwargs["market_caps"]      = self.market_caps
                        if "w_prev"           in fit_sig.parameters:
                            fit_kwargs["w_prev"]           = (
                                dict(zip(prev["assets"], prev["weights"])) if prev else None
                            )
                        if "turnover_penalty" in fit_sig.parameters:
                            fit_kwargs["turnover_penalty"] = miqp_turnover_penalty

                        t0 = time.time()
                        model.fit(R_train, I_train, **fit_kwargs)
                        exec_time = time.time() - t0

                        assets  = np.array(model.selected_assets)
                        weights = np.array(model.weights).flatten()

                        if len(assets) != K:
                            raise AssertionError(
                                f"{model_name} returned {len(assets)} assets, expected {K}"
                            )

                        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                        weights[weights < 0] = 0.0
                        if weights.sum() > 0:
                            weights /= weights.sum()

                        solver_gap     = getattr(model, "mip_gap_achieved", np.nan)
                        solver_optimal = getattr(model, "is_optimal",       np.nan)

                        # ── In-sample TE ─────────────────────────────────── #
                        R_sel_is = R_train[assets].to_numpy(dtype=float)
                        port_is  = R_sel_is @ weights
                        idx_is   = I_train.to_numpy(dtype=float).flatten()
                        te_is    = float(np.std(port_is - idx_is) * np.sqrt(252))

                        # ── Active share / diversification ───────────────── #
                        asset_active_share  = self.compute_active_share(
                            assets.tolist(), weights, bench_w_series
                        )
                        sector_active_share = self.compute_sector_active_share(
                            assets.tolist(), weights, bench_w_series, valid_assets
                        )
                        n_eff    = self.diversification(weights)
                        div_norm = n_eff / K

                        # ── OOS ──────────────────────────────────────────── #
                        R_eval_sel = R_eval.reindex(columns=assets, fill_value=0.0)
                        # Warn if any selected asset is completely missing in eval window
                        zero_cols = (R_eval_sel.abs().sum(axis=0) == 0)
                        if zero_cols.any():
                            warnings.warn(
                                f"[{quarter_label}] {model_name} K={K}: "
                                f"{zero_cols.sum()} assets have zero returns in eval window.",
                                RuntimeWarning,
                            )

                        port_oos      = R_eval_sel.to_numpy(dtype=float) @ weights
                        idx_oos       = I_eval.to_numpy(dtype=float).flatten()
                        te_oos        = float(np.std(port_oos - idx_oos) * np.sqrt(252))
                        port_cum_ret  = float(np.prod(1 + port_oos) - 1)
                        bench_cum_ret = float(np.prod(1 + idx_oos) - 1)
                        tracking_diff = port_cum_ret - bench_cum_ret

                        ir_oos     = self.information_ratio(port_oos, idx_oos)
                        sharpe_oos = self.sharpe_ratio(port_oos)
                        mdd_te     = self.max_drawdown_te(port_oos, idx_oos)

                        # ── Turnover / stability ─────────────────────────── #
                        if prev is not None:
                            turnover             = self.compute_turnover(
                                prev["assets"], prev["weights"], assets.tolist(), weights
                            )
                            stability            = self.composition_stability(
                                prev["assets"], assets.tolist()
                            )
                            avg_weight_stability = 1.0 - min(1.0, turnover / 2.0)
                            assets_added         = len(set(assets.tolist()) - set(prev["assets"]))
                            assets_removed       = len(set(prev["assets"]) - set(assets.tolist()))
                        else:
                            turnover = stability = avg_weight_stability = np.nan
                            assets_added = assets_removed = np.nan

                        prev_portfolio[key] = {
                            "assets":  assets.tolist(),
                            "weights": weights.copy(),
                        }

                        # ── Save portfolio CSV ───────────────────────────── #
                        bw_sel    = bench_w_series.reindex(assets, fill_value=0.0)
                        sec_vals  = self._resolve_series_values(
                            self.sectors, assets.tolist(), "Unknown"
                        )
                        portfolio_df = pd.DataFrame({
                            "asset":            assets,
                            "sector":           sec_vals.values,
                            "weight":           weights,
                            "benchmark_weight": bw_sel.values,
                            "active_weight":    weights - bw_sel.values,
                        }).sort_values("weight", ascending=False)
                        portfolio_df.to_csv(
                            f"{output_dir}/portfolios/{quarter_label}/{model_name}_K{K}.csv",
                            index=False,
                        )

                        # ── Save daily OOS path ──────────────────────────── #
                        active_daily = port_oos - idx_oos
                        series_key   = (model_name, K)
                        if series_key not in oos_daily_by_key:
                            oos_daily_by_key[series_key] = {"portfolio": [], "benchmark": []}
                        oos_daily_by_key[series_key]["portfolio"].append(port_oos)
                        oos_daily_by_key[series_key]["benchmark"].append(idx_oos)

                        pd.DataFrame({
                            "date":                  I_eval.index,
                            "portfolio_return":       port_oos,
                            "benchmark_return":       idx_oos,
                            "active_return":          active_daily,
                            "portfolio_cum_return":   np.cumprod(1 + port_oos) - 1,
                            "benchmark_cum_return":   np.cumprod(1 + idx_oos) - 1,
                            "active_cum_return":      np.cumsum(active_daily),
                        }).to_csv(
                            f"{output_dir}/paths/{quarter_label}/{model_name}_K{K}.csv",
                            index=False,
                        )

                        # ── Collect row ──────────────────────────────────── #
                        row = {
                            "model":                model_name,
                            "K":                    K,
                            "quarter":              quarter_label,
                            "train_start":          train_start.date(),
                            "train_end":            train_end.date(),
                            "eval_start":           q_start.date(),
                            "eval_end":             q_end.date(),
                            "universe_size":        len(valid_assets),
                            "assets_dropped":       total_dropped,
                            "train_length_days":    len(R_train),
                            "TE_annual_insample":   te_is,
                            "TE_annual_oos":        te_oos,
                            "information_ratio_oos":ir_oos,
                            "sharpe_ratio_oos":     sharpe_oos,
                            "max_drawdown_te_oos":  mdd_te,
                            "portfolio_return_oos": port_cum_ret,
                            "benchmark_return_oos": bench_cum_ret,
                            "tracking_difference":  tracking_diff,
                            "asset_active_share":   asset_active_share,
                            "sector_active_share":  sector_active_share,
                            "n_eff_absolute":       n_eff,
                            "diversification_norm": div_norm,
                            "turnover":             turnover,
                            "composition_stability":stability,
                            "avg_weight_stability": avg_weight_stability,
                            "assets_added":         assets_added,
                            "assets_removed":       assets_removed,
                            "execution_time":       exec_time,
                            "solver_gap":           solver_gap,
                            "solver_optimal":       solver_optimal,
                        }
                        results.append(row)
                        quarter_rows.append(row)

                        print(
                            f"done ({exec_time:.1f}s "
                            f"TE_IS={te_is:.4f}  TE_OOS={te_oos:.4f}  "
                            f"IR={ir_oos:.2f}  stab={stability if not np.isnan(stability) else 'n/a'})"
                            if not (isinstance(stability, float) and np.isnan(stability))
                            else
                            f"done ({exec_time:.1f}s "
                            f"TE_IS={te_is:.4f}  TE_OOS={te_oos:.4f}  IR={ir_oos:.2f})"
                        )

                    except Exception as e:
                        print(f"ERROR: {e}")

            # Save per-quarter CSV
            if quarter_rows:
                pd.DataFrame(quarter_rows).to_csv(
                    f"{output_dir}/summaries/{quarter_label}_results.csv",
                    index=False,
                )

        # ── Full-period metrics (PRIMARY THESIS METRIC) ───────────────────── #
        results_df      = pd.DataFrame(results)
        full_period_rows = []

        for (model_name, k), vals in oos_daily_by_key.items():
            if not vals["portfolio"]:
                continue
            port_all  = np.concatenate(vals["portfolio"])
            bench_all = np.concatenate(vals["benchmark"])
            active_all = port_all - bench_all

            full_period_rows.append({
                "model":                         model_name,
                "K":                             k,
                "n_oos_days":                    int(len(port_all)),
                "TE_annual_oos_full":            float(np.std(active_all) * np.sqrt(252)),
                "IR_full_period":                self.information_ratio(port_all, bench_all),
                "sharpe_full_period":            self.sharpe_ratio(port_all),
                "portfolio_return_full_period":  float(np.prod(1 + port_all) - 1),
                "benchmark_return_full_period":  float(np.prod(1 + bench_all) - 1),
                "tracking_difference_full_period": float(
                    np.prod(1 + port_all) - np.prod(1 + bench_all)
                ),
            })

        full_period_df = pd.DataFrame(full_period_rows).sort_values(["model", "K"])

        # ── Save everything ───────────────────────────────────────────────── #
        out_csv      = f"{output_dir}/summaries/experiment_results_rolling.csv"
        out_xlsx     = f"{output_dir}/summaries/experiment_results_rolling.xlsx"
        out_full_csv = f"{output_dir}/summaries/full_period_metrics.csv"

        results_df.to_csv(out_csv,      index=False)
        results_df.to_excel(out_xlsx,   index=False)
        full_period_df.to_csv(out_full_csv, index=False)

        print(f"\nResults saved        → {out_csv}")
        print(f"Full-period metrics  → {out_full_csv}  ← use this for your thesis table")

        return results_df