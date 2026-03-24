import os
import numpy as np
import pandas as pd


def _parse_entry_dates(constituents):
    entries = constituents[["Symbol", "Date added"]].copy()
    entries["Symbol"] = entries["Symbol"].str.replace(".", "-", regex=False)
    entries["entry_date"] = pd.to_datetime(entries["Date added"], errors="coerce")
    return entries[["Symbol", "entry_date"]]


def main():
    returns = pd.read_parquet("data/returns.parquet")
    constituents = pd.read_csv("data/sp500_tickers.csv")

    entry_df = _parse_entry_dates(constituents)
    entry_map = entry_df.set_index("Symbol")["entry_date"]

    all_dates = returns.index
    start_date = pd.Timestamp(all_dates.min())
    end_date = pd.Timestamp(all_dates.max())

    symbols_in_returns = set(returns.columns)
    entry_in_returns = entry_map.reindex(sorted(symbols_in_returns))

    valid_entry_dates = entry_in_returns.dropna()
    post_start_entrants = valid_entry_dates[valid_entry_dates > start_date]

    quarterly_dates = pd.date_range(start=start_date, end=end_date, freq="QS")
    rows = []
    for dt in quarterly_dates:
        active = int((valid_entry_dates <= dt).sum())
        unavailable = int((valid_entry_dates > dt).sum())
        total = int(valid_entry_dates.shape[0])
        unavailable_fraction = float(unavailable / total) if total > 0 else np.nan
        rows.append(
            {
                "date": dt.date(),
                "active_by_entry_date": active,
                "not_yet_added": unavailable,
                "total_with_known_entry": total,
                "not_yet_added_fraction": unavailable_fraction,
            }
        )

    by_date_df = pd.DataFrame(rows)

    summary = {
        "analysis_start": start_date.date(),
        "analysis_end": end_date.date(),
        "n_assets_in_returns": int(len(symbols_in_returns)),
        "n_assets_with_known_entry_date": int(valid_entry_dates.shape[0]),
        "n_post_start_entrants": int(post_start_entrants.shape[0]),
        "post_start_entrant_fraction": float(
            post_start_entrants.shape[0] / valid_entry_dates.shape[0]
        )
        if valid_entry_dates.shape[0] > 0
        else np.nan,
        "mean_not_yet_added_fraction_quarterly": float(
            by_date_df["not_yet_added_fraction"].mean()
        )
        if len(by_date_df) > 0
        else np.nan,
        "max_not_yet_added_fraction_quarterly": float(
            by_date_df["not_yet_added_fraction"].max()
        )
        if len(by_date_df) > 0
        else np.nan,
    }

    os.makedirs("results/summaries", exist_ok=True)
    by_date_df.to_csv("results/summaries/survivorship_bias_by_quarter.csv", index=False)
    pd.DataFrame([summary]).to_csv("results/summaries/survivorship_bias_summary.csv", index=False)

    print("Saved results/summaries/survivorship_bias_by_quarter.csv")
    print("Saved results/summaries/survivorship_bias_summary.csv")


if __name__ == "__main__":
    main()
