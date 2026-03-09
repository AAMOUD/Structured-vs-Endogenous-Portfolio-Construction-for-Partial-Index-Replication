from data_manager import DataManager

dm = DataManager("data\\sp500_tickers.csv")

prices = dm.download_prices(start="2010-01-01")

returns = dm.compute_returns()

returns = dm.filter_assets()

index_returns = dm.download_index(start="2010-01-01")

returns, index_returns = dm.align_data()

dm.save_data()

print("Final dataset shape:", returns.shape)
print("Returns shape:", dm.returns.shape)
print("Missing ratio:", dm.returns.isna().mean().mean())