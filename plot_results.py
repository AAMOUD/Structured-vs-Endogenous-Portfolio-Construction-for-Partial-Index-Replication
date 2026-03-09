import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)


df = pd.read_csv("results/summaries/model_comparison.csv")


sns.lineplot(data=df, x="K", y="TE_annual", hue="model")
plt.title("Tracking Error vs K")
plt.savefig("results/plots/TE_vs_K.png")
plt.show()


sns.lineplot(data=df, x="K", y="execution_time", hue="model")
plt.title("Execution Time vs K")
plt.savefig("results/plots/time_vs_K.png")
plt.show()


sns.lineplot(data=df, x="K", y="turnover", hue="model")
plt.title("Turnover vs K")
plt.savefig("results/plots/turnover_vs_K.png")
plt.show()


sns.lineplot(data=df, x="K", y="diversification", hue="model")
plt.title("Diversification vs K")
plt.savefig("results/plots/diversification_vs_K.png")
plt.show()