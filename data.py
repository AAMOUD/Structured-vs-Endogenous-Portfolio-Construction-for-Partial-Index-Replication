"""Data handling module for fetching and preprocessing market data."""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Dict, List, Optional
from config import DATA_PARAMS, PORTFOLIO_PARAMS
from pathlib import Path
import os

# Fallback list of major S&P 500 constituents (top by market cap + representative sectors)
FALLBACK_SP500_TICKERS = [
    'MSFT', 'AAPL', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'BRK.B', 'JNJ',
    'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'DIS', 'MCD', 'BA', 'NFLX',
    'INTC', 'AMD', 'CRM', 'NKE', 'COST', 'KO', 'ABT', 'LLY', 'XOM', 'CVX',
    'MRK', 'IBM', 'GS', 'AMEX', 'AXP', 'PYPL', 'SQ', 'SPOT', 'UBER', 'LYFT',
    'PEP', 'MO', 'PM', 'TJX', 'LOW', 'AZO', 'BBY', 'ROST', 'FIVE', 'CHWY',
    'ADBE', 'AVGO', 'QCOM', 'CSCO', 'MCHP', 'LRCX', 'ASML', 'TSM', 'ACN', 'INTU'
]


class DataHandler:
    """Handles data fetching, preprocessing, and preparation for portfolio construction."""

    def __init__(self, index_ticker: str = "^GSPC", start_date: str = None, end_date: str = None, data_dir: str = "./data"):
        """
        Initialize DataHandler.

        Args:
            index_ticker: Index ticker symbol (default: S&P 500)
            start_date: Start date for data (YYYY-MM-DD format)
            end_date: End date for data (YYYY-MM-DD format)
            data_dir: Directory to save/load cached data
        """
        self.index_ticker = index_ticker
        self.start_date = start_date or DATA_PARAMS["start_date"]
        self.end_date = end_date or DATA_PARAMS["end_date"]
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.index_data = None
        self.constituents_data = None
        self.constituents_list = None
        self.correlation_matrix = None
        self.returns = None
        self.index_returns = None
        self.sector_data = None
        self.market_caps = None

    def fetch_index_data(self) -> pd.DataFrame:
        """
        Fetch index price data.

        Returns:
            DataFrame with index adjusted close prices
        """
        print(f"Fetching {self.index_ticker} data...")
        data = yf.download(
            self.index_ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        # Handle both single ticker and multi-ticker returns
        if isinstance(data, pd.DataFrame):
            self.index_data = data["Adj Close"] if "Adj Close" in data.columns else data.iloc[:, 0]
        else:
            self.index_data = data
        return self.index_data

    def fetch_sp500_constituents(self) -> List[str]:
        """
        Fetch S&P 500 constituents list from CSV file.
        Falls back to Wikipedia scraping if CSV not found.

        Returns:
            List of S&P 500 ticker symbols
        """
        print("Fetching S&P 500 constituents...")
        csv_path = self.data_dir / "sp500_tickers.csv"
        
        try:
            # First try to load from CSV file
            if csv_path.exists():
                sp500_df = pd.read_csv(csv_path)
                if 'Symbol' in sp500_df.columns:
                    tickers = sp500_df["Symbol"].tolist()
                    self.constituents_list = tickers
                    self.sector_data = sp500_df  # Store for sector mapping
                    print(f"✓ Successfully loaded {len(tickers)} tickers from {csv_path}")
                    return tickers
                else:
                    raise ValueError("'Symbol' column not found in CSV")
            else:
                print(f"CSV file not found at {csv_path}, trying Wikipedia...")
                raise FileNotFoundError("CSV file not found")
        except Exception as e:
            print(f"✗ Error loading from CSV: {e}")
            # Fall back to Wikipedia
            try:
                tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)
                sp500_df = tables[0]
                if 'Symbol' in sp500_df.columns:
                    tickers = sp500_df["Symbol"].tolist()
                    self.constituents_list = tickers
                    self.sector_data = sp500_df
                    print(f"✓ Successfully fetched {len(tickers)} tickers from Wikipedia")
                    return tickers
                else:
                    raise ValueError("'Symbol' column not found")
            except Exception as e2:
                print(f"✗ Error fetching constituents from Wikipedia: {e2}")
                print(f"✓ Using fallback list of {len(FALLBACK_SP500_TICKERS)} major constituents")
                self.constituents_list = FALLBACK_SP500_TICKERS
                self.sector_data = None
                return FALLBACK_SP500_TICKERS

    def fetch_constituents_data(self, tickers: List[str] = None, max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch price data for index constituents.

        Args:
            tickers: List of ticker symbols. If None, fetches S&P 500 constituents
            max_retries: Maximum retries for downloading data

        Returns:
            DataFrame with adjusted close prices for all constituents
        """
        if tickers is None:
            tickers = self.fetch_sp500_constituents()

        print(f"Fetching data for {len(tickers)} constituents...")
        self.constituents_list = tickers

        # Download in batches to avoid connection issues
        batch_size = 50
        all_data = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            print(f"Downloading batch {i // batch_size + 1}/{(len(tickers) - 1) // batch_size + 1}")

            try:
                data = yf.download(
                    batch,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )
                
                # Extract adjusted close prices
                if "Adj Close" in data.columns:
                    data = data["Adj Close"]
                elif "Close" in data.columns:
                    data = data["Close"]
                else:
                    data = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
                
                # Handle single ticker case
                if len(batch) == 1 and not isinstance(data, pd.DataFrame):
                    data = data.to_frame(name=batch[0])
                
                all_data.append(data)
            except Exception as e:
                print(f"Error downloading batch: {e}")

        if all_data:
            self.constituents_data = pd.concat(all_data, axis=1)
            # Drop columns with too many NaN values
            self.constituents_data = self.constituents_data.dropna(axis=1, thresh=len(self.constituents_data) * 0.8)
            self.constituents_data = self.constituents_data.dropna()
            print(f"Successfully fetched data for {len(self.constituents_data.columns)} assets")
        
        return self.constituents_data

    def fetch_market_caps(self, tickers: List[str] = None) -> pd.DataFrame:
        """
        Fetch current market capitalizations for tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Series with market caps indexed by ticker
        """
        if tickers is None:
            tickers = self.constituents_data.columns.tolist() if self.constituents_data is not None else []
        
        print(f"Fetching market caps for {len(tickers)} tickers...")
        market_caps = {}
        
        # Download in batches
        batch_size = 50
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            print(f"  Batch {i // batch_size + 1}/{(len(tickers) - 1) // batch_size + 1}...")
            
            for ticker in batch:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    # Try different keys for market cap
                    market_cap = info.get('marketCap') or info.get('market_cap') or info.get('sharesOutstanding', 0) * info.get('currentPrice', 0)
                    if market_cap and market_cap > 0:
                        market_caps[ticker] = market_cap
                except Exception as e:
                    # If market cap fetch fails, use equal weighting for this ticker
                    pass
        
        self.market_caps = pd.Series(market_caps)
        print(f"✓ Successfully fetched market caps for {len(self.market_caps)} tickers")
        return self.market_caps
    
    def compute_returns(self, use_market_cap_weights: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute log returns for constituents and construct market-cap weighted index returns.

        Args:
            use_market_cap_weights: If True, compute index returns using market cap weights.
                                   If False, use the ^GSPC index data.

        Returns:
            Tuple of (constituents returns DataFrame, index returns Series)
        """
        if self.constituents_data is None:
            raise ValueError("Constituents data not loaded. Call fetch_constituents_data first.")
        if self.index_data is None:
            raise ValueError("Index data not loaded. Call fetch_index_data first.")

        # Compute constituent returns
        self.returns = np.log(self.constituents_data / self.constituents_data.shift(1)).dropna()
        
        if use_market_cap_weights:
            print("\nComputing market-cap weighted index returns...")
            # Fetch market caps if not already available
            if self.market_caps is None:
                self.fetch_market_caps(self.constituents_data.columns.tolist())
            
            # Filter to tickers we have both prices and market caps for
            common_tickers = self.returns.columns.intersection(self.market_caps.index)
            
            if len(common_tickers) > 0:
                # Normalize market caps to weights
                weights = self.market_caps[common_tickers] / self.market_caps[common_tickers].sum()
                
                # Compute weighted returns
                weighted_returns = self.returns[common_tickers].multiply(weights, axis=1).sum(axis=1)
                self.index_returns = weighted_returns
                print(f"✓ Computed market-cap weighted returns using {len(common_tickers)} assets")
                print(f"   Largest weights: {weights.nlargest(5).to_dict()}")
            else:
                print("⚠️  No common tickers with market caps, falling back to ^GSPC index")
                self.index_returns = np.log(self.index_data / self.index_data.shift(1)).dropna()
        else:
            # Use ^GSPC index returns
            self.index_returns = np.log(self.index_data / self.index_data.shift(1)).dropna()

        # Align indices
        common_dates = self.returns.index.intersection(self.index_returns.index)
        self.returns = self.returns.loc[common_dates]
        self.index_returns = self.index_returns.loc[common_dates]

        return self.returns, self.index_returns

    def compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix between constituents and index.

        Returns:
            DataFrame with correlations of each asset with the index
        """
        if self.returns is None:
            raise ValueError("Returns not computed. Call compute_returns first.")

        correlations = pd.Series(
            [self.returns[ticker].corr(self.index_returns) for ticker in self.returns.columns],
            index=self.returns.columns
        )
        self.correlation_matrix = correlations.sort_values(ascending=False)
        return self.correlation_matrix

    def get_sector_mapping(self) -> Dict[str, str]:
        """
        Get sector mapping for S&P 500 constituents.

        Returns:
            Dictionary mapping ticker to sector
        """
        # First try to use cached sector data from CSV
        if hasattr(self, 'sector_data') and self.sector_data is not None:
            sector_mapping = dict(zip(self.sector_data["Symbol"], self.sector_data["GICS Sector"]))
            return sector_mapping
        
        # Try to load from CSV
        csv_path = self.data_dir / "sp500_tickers.csv"
        if csv_path.exists():
            try:
                sp500_df = pd.read_csv(csv_path)
                sector_mapping = dict(zip(sp500_df["Symbol"], sp500_df["GICS Sector"]))
                return sector_mapping
            except Exception as e:
                print(f"Error loading sector mapping from CSV: {e}")
        
        # Fall back to Wikipedia
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            sp500_df = tables[0]
            sector_mapping = dict(zip(sp500_df["Symbol"], sp500_df["GICS Sector"]))
            return sector_mapping
        except Exception as e:
            print(f"Error fetching sector mapping: {e}")
            return {}

    def prepare_data(self, tickers: List[str] = None) -> Dict:
        """
        Complete data preparation pipeline.

        Args:
            tickers: List of ticker symbols (if None, fetches S&P 500)

        Returns:
            Dictionary containing all prepared data
        """
        self.fetch_index_data()
        self.fetch_constituents_data(tickers)
        self.compute_returns()
        self.compute_correlation_matrix()

        return {
            "constituents_data": self.constituents_data,
            "index_data": self.index_data,
            "returns": self.returns,
            "index_returns": self.index_returns,
            "correlation_matrix": self.correlation_matrix,
            "constituents_list": self.constituents_list,
        }

    def save_data(self, prefix: str = "sp500") -> None:
        """
        Save processed data to CSV files.

        Args:
            prefix: Prefix for saved files
        """
        if self.constituents_data is not None:
            self.constituents_data.to_csv(self.data_dir / f"{prefix}_prices.csv")
            print(f"✓ Saved constituent prices to {self.data_dir / f'{prefix}_prices.csv'}")
        
        if self.index_data is not None:
            self.index_data.to_csv(self.data_dir / f"{prefix}_index.csv")
            print(f"✓ Saved index prices to {self.data_dir / f'{prefix}_index.csv'}")
        
        if self.returns is not None:
            self.returns.to_csv(self.data_dir / f"{prefix}_returns.csv")
            print(f"✓ Saved constituent returns to {self.data_dir / f'{prefix}_returns.csv'}")
        
        if self.index_returns is not None:
            self.index_returns.to_csv(self.data_dir / f"{prefix}_index_returns.csv")
            print(f"✓ Saved index returns to {self.data_dir / f'{prefix}_index_returns.csv'}")
        
        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv(self.data_dir / f"{prefix}_correlations.csv")
            print(f"✓ Saved correlations to {self.data_dir / f'{prefix}_correlations.csv'}")
        
        if self.constituents_list is not None:
            with open(self.data_dir / f"{prefix}_constituents.txt", "w") as f:
                f.write('\n'.join(self.constituents_list))
            print(f"✓ Saved constituents list to {self.data_dir / f'{prefix}_constituents.txt'}")

    def load_data(self, prefix: str = "sp500") -> bool:
        """
        Load processed data from CSV files.

        Args:
            prefix: Prefix for saved files

        Returns:
            True if all files loaded successfully, False otherwise
        """
        try:
            prices_file = self.data_dir / f"{prefix}_prices.csv"
            index_file = self.data_dir / f"{prefix}_index.csv"
            returns_file = self.data_dir / f"{prefix}_returns.csv"
            index_returns_file = self.data_dir / f"{prefix}_index_returns.csv"
            corr_file = self.data_dir / f"{prefix}_correlations.csv"
            constituents_file = self.data_dir / f"{prefix}_constituents.txt"

            if not all([prices_file.exists(), index_file.exists(), returns_file.exists(), 
                       index_returns_file.exists(), corr_file.exists(), constituents_file.exists()]):
                return False

            self.constituents_data = pd.read_csv(prices_file, index_col=0, parse_dates=True)
            self.index_data = pd.read_csv(index_file, index_col=0, parse_dates=True).squeeze()
            self.returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            self.index_returns = pd.read_csv(index_returns_file, index_col=0, parse_dates=True, squeeze=True)
            
            # Ensure numeric types
            self.constituents_data = self.constituents_data.apply(pd.to_numeric, errors='coerce')
            self.index_data = pd.to_numeric(self.index_data, errors='coerce')
            self.returns = self.returns.apply(pd.to_numeric, errors='coerce')
            self.index_returns = pd.to_numeric(self.index_returns, errors='coerce')
            
            corr_data = pd.read_csv(corr_file, index_col=0)
            self.correlation_matrix = corr_data.iloc[:, 0]
            
            with open(constituents_file, "r") as f:
                self.constituents_list = f.read().strip().split('\n')

            print(f"✓ All data loaded successfully from {self.data_dir}")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
