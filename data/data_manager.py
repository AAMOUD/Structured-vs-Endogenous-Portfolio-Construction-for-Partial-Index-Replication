import pandas as pd
import yfinance as yf
import time


class DataManager:

    def __init__(self, constituents_path):

        df = pd.read_csv(constituents_path)

        # correction tickers Yahoo
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

        self.constituents = df

        self.tickers = df["Symbol"].tolist()

        self.sectors = df.set_index("Symbol")["GICS Sector"]

    def download_prices(self, start="2010-01-01"):

        batch_size = 50

        all_prices = []

        valid_tickers = []

        for i in range(0, len(self.tickers), batch_size):

            batch = self.tickers[i:i + batch_size]

            print(f"Downloading batch {i}")

            try:

                data = yf.download(
                    batch,
                    start=start,
                    auto_adjust=True,
                    progress=False
                )["Close"]

                all_prices.append(data)

                valid_tickers.extend(data.columns)

            except Exception as e:

                print("Batch failed:", batch)

            time.sleep(1)

        prices = pd.concat(all_prices, axis=1)

        prices = prices.loc[:, ~prices.columns.duplicated()]

        self.prices = prices

        print("Downloaded assets:", prices.shape[1])

        return prices

    def compute_returns(self):

        returns = self.prices.pct_change()

        # on ne supprime pas les dates
        returns = returns.iloc[1:]

        self.returns = returns

        return returns

    def filter_assets(self, min_ratio=0.6):

        ratio = self.returns.notna().mean()

        valid_assets = ratio > min_ratio

        self.returns = self.returns.loc[:, valid_assets]

        print("Assets after filtering:", self.returns.shape[1])

        return self.returns

    def download_index(self, start="2010-01-01"):

        index = yf.download(
            "^GSPC",
            start=start,
            auto_adjust=True,
            progress=False
        )

        self.index_returns = index["Close"].pct_change().dropna()

        return self.index_returns

    def align_data(self):

        common_dates = self.returns.index.intersection(self.index_returns.index)

        self.returns = self.returns.loc[common_dates]

        self.index_returns = self.index_returns.loc[common_dates]

        return self.returns, self.index_returns

    def save_data(self):

        self.prices.to_parquet("data/prices.parquet")

        self.returns.to_parquet("data/returns.parquet")

        self.index_returns.to_parquet("data/index_returns.parquet")

        print("Data saved to parquet")