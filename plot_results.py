import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("results/summaries/experiment_results.csv")

# Consistent horizon ordering
horizon_order = ["1y", "5y", "10y", "full"]
df["horizon"] = pd.Categorical(df["horizon"], categories=horizon_order, ordered=True)


def save(name):
    plt.tight_layout()
    plt.savefig(f"results/plots/{name}.png", dpi=150)
    plt.show()
    plt.close()


# ── 1. Tracking Error vs K ──────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="K", y="TE_annual", hue="model", style="horizon",
             markers=True, dashes=True)
plt.title("Annualised Tracking Error vs Portfolio Size")
plt.xlabel("K (number of assets)")
plt.ylabel("Annualised TE")
save("TE_vs_K")

# ── 2. Execution Time vs K ──────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="K", y="execution_time", hue="model", style="horizon",
             markers=True, dashes=True)
plt.title("Execution Time vs Portfolio Size")
plt.xlabel("K (number of assets)")
plt.ylabel("Time (seconds)")
save("time_vs_K")

# ── 3. Diversification vs K ─────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="K", y="diversification", hue="model", style="horizon",
             markers=True, dashes=True)
plt.title("Effective Number of Assets (1/HHI) vs Portfolio Size")
plt.xlabel("K (number of assets)")
plt.ylabel("Effective N  (1 / sum w^2)")
save("diversification_vs_K")

# ── 4. TE heatmap: model x horizon (averaged over K) ───────────────────────
pivot = df.groupby(["model", "horizon"])["TE_annual"].mean().unstack("horizon")
pivot = pivot[horizon_order]
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd")
plt.title("Mean Annualised TE  (model x horizon)")
save("TE_heatmap")

# ── 5. TE by horizon, faceted per model ─────────────────────────────────────
g = sns.FacetGrid(df, col="model", col_wrap=3, height=3.5, sharey=True)
g.map_dataframe(sns.lineplot, x="K", y="TE_annual", hue="horizon",
                hue_order=horizon_order, markers=True)
g.add_legend()
g.set_axis_labels("K", "Annualised TE")
g.figure.suptitle("Tracking Error by Model and Horizon", y=1.02)
g.figure.savefig("results/plots/TE_facet.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print("All plots saved to results/plots/")
