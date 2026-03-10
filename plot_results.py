import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("results/summaries/experiment_results.csv")


def save(name):
    plt.tight_layout()
    plt.savefig(f"results/plots/{name}.png", dpi=150)
    plt.show()
    plt.close()


# ── 1. Tracking Error vs K ──────────────────────────────────────────────────
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

