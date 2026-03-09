import pandas as pd
import yfinance as yf


def main() -> None:
    tickers = pd.read_csv("data/sp500_tickers.csv")["Symbol"].tolist()

    caps = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            caps[ticker] = info.get("marketCap", None)
        except Exception:
            caps[ticker] = None

    df = pd.DataFrame.from_dict(caps, orient="index", columns=["market_cap"])
    df.to_csv("data/market_caps.csv")


if __name__ == "__main__":
    main()
