import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("results/summaries/experiment_results.csv")
rolling_summary_path = "results_rolling/summaries/experiment_results_rolling.csv"


def save(name):
    plt.savefig(
        f"results/plots/{name}.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.show()
    plt.close()


# ── 1. Primary static chart: OOS Tracking Error vs K ───────────────────────
if "TE_annual_oos" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="K", y="TE_annual_oos", hue="model", markers=True)
    plt.title("Annualised OOS Tracking Error vs Portfolio Size")
    plt.xlabel("K (number of assets)")
    plt.ylabel("Annualised OOS TE")
    save("TE_OOS_vs_K")
elif "TE_annual" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="K", y="TE_annual", hue="model", markers=True)
    plt.title("Annualised Tracking Error vs Portfolio Size")
    plt.xlabel("K (number of assets)")
    plt.ylabel("Annualised TE")
    save("TE_vs_K")

# ── 2. Asset Active Share vs K ──────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="K", y="asset_active_share", hue="model", markers=True)
plt.title("Asset Active Share (L1 distance from benchmark) vs Portfolio Size")
plt.xlabel("K (number of assets)")
plt.ylabel("Asset Active Share")
save("asset_active_share_vs_K")

# ── 3. Sector Active Share vs K ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="K", y="sector_active_share", hue="model", markers=True)
plt.title("Sector Active Share vs Portfolio Size")
plt.xlabel("K (number of assets)")
plt.ylabel("Sector Active Share")
save("sector_active_share_vs_K")

# ── 4. Diversification vs K ─────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="K", y="diversification", hue="model", markers=True)
plt.title("Normalised Diversification (N_eff / K) vs Portfolio Size")
plt.xlabel("K (number of assets)")
plt.ylabel("N_eff / K  (1 = equal weight)")
save("diversification_vs_K")

# ── 4. Execution Time vs K ──────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="K", y="execution_time", hue="model", markers=True)
plt.title("Execution Time vs Portfolio Size")
plt.xlabel("K (number of assets)")
plt.ylabel("Time (seconds)")
save("time_vs_K")

print("All plots saved to results/plots/")


# ── 6. Appendix: Static in-sample TE vs K (if available) ───────────────────
if "TE_annual" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="K", y="TE_annual", hue="model", markers=True)
    plt.title("Appendix: Annualised In-Sample Tracking Error vs Portfolio Size")
    plt.xlabel("K (number of assets)")
    plt.ylabel("Annualised In-Sample TE")
    save("TE_IS_vs_K_appendix")


# ── 7. Rolling cumulative model-vs-benchmark (if available) ─────────────────
if os.path.exists(rolling_summary_path):
    rolling_df = pd.read_csv(rolling_summary_path)
    if len(rolling_df) > 0:
        rolling_df = rolling_df.sort_values("quarter")

        grouped = []
        for (model_name, k), sub in rolling_df.groupby(["model", "K"]):
            sub = sub.copy().sort_values("quarter")
            sub["portfolio_cum"] = (1 + sub["portfolio_return_oos"]).cumprod() - 1
            sub["benchmark_cum"] = (1 + sub["benchmark_return_oos"]).cumprod() - 1
            sub["active_cum"] = sub["portfolio_cum"] - sub["benchmark_cum"]
            sub["model_k"] = f"{model_name}_K{k}"
            grouped.append(sub)

        roll_cum = pd.concat(grouped, axis=0)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=roll_cum, x="quarter", y="active_cum", hue="model_k")
        plt.title("Rolling Cumulative Active Return by Model and K")
        plt.xlabel("Quarter")
        plt.ylabel("Cumulative Active Return")
        plt.xticks(rotation=45)
        save("rolling_cumulative_active_return")

        if "composition_stability" in roll_cum.columns:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=roll_cum, x="quarter", y="composition_stability", hue="model_k")
            plt.title("Rolling Composition Stability")
            plt.xlabel("Quarter")
            plt.ylabel("Composition Stability vs Previous Quarter")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            save("rolling_composition_stability")

        print("Rolling plots saved to results/plots/")


# ── 8. Stitched daily rolling path chart from path files ────────────────────
paths_root = "results_rolling/paths"
if os.path.exists(paths_root):
    all_paths = []
    for quarter_dir in sorted(os.listdir(paths_root)):
        q_path = os.path.join(paths_root, quarter_dir)
        if not os.path.isdir(q_path):
            continue
        for fname in sorted(os.listdir(q_path)):
            if not fname.endswith(".csv") or "_K" not in fname:
                continue

            model_name, k_part = fname.rsplit("_K", 1)
            try:
                k_val = int(k_part.replace(".csv", ""))
            except ValueError:
                continue

            fpath = os.path.join(q_path, fname)
            p = pd.read_csv(fpath)
            if len(p) == 0:
                continue

            p["date"] = pd.to_datetime(p["date"])
            p["model"] = model_name
            p["K"] = k_val
            p["model_k"] = f"{model_name}_K{k_val}"
            all_paths.append(p[["date", "portfolio_return", "benchmark_return", "model_k"]])

    if all_paths:
        combined = pd.concat(all_paths, axis=0).sort_values(["model_k", "date"])
        combined = combined.drop_duplicates(subset=["model_k", "date"], keep="last")

        stitched = []
        for model_k, sub in combined.groupby("model_k"):
            sub = sub.sort_values("date").copy()
            sub["portfolio_cum"] = (1 + sub["portfolio_return"]).cumprod() - 1
            sub["benchmark_cum"] = (1 + sub["benchmark_return"]).cumprod() - 1
            stitched.append(sub)

        stitched_df = pd.concat(stitched, axis=0)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=stitched_df, x="date", y="portfolio_cum", hue="model_k")
        plt.title("Stitched Daily Portfolio Cumulative Return by Model and K")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        save("stitched_daily_portfolio_cum")

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=stitched_df, x="date", y="benchmark_cum", hue="model_k")
        plt.title("Stitched Daily Benchmark Cumulative Return by Model and K")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        save("stitched_daily_benchmark_cum")

