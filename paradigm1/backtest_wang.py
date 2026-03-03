"""
Backtesting framework following Wang et al. (2018) evaluation methodology.

Implements proper in-sample/out-of-sample evaluation with metrics:
- TE_in, TE_out (tracking error)
- Std_TE_in, Std_TE_out (robustness)
- Mean_Corr_in, Mean_Corr_out (correlation quality)
- Consistency (|TE_in - TE_out|)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class WangBacktestEngine:
    """
    Backtesting engine following Wang et al. (2018) methodology.
    
    Key features:
    - Proper train/test split (in-sample vs out-of-sample)
    - Selection and optimization on in-sample only
    - TE computed as: TE(w) = (1/T) * ||I - R*w||_2^2, then sqrt(TE)
    - Std_TE: standard deviation of squared tracking errors
    - Mean_Corr: mean correlation of selected assets with index
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        index_returns: pd.Series,
        correlation_matrix: pd.Series,
        train_ratio: float = 0.7,
        slippage: float = 0.001,
    ):
        """
        Initialize Wang backtest engine.

        Args:
            returns: DataFrame of constituent returns (all available assets)
            index_returns: Series of index returns
            correlation_matrix: Series of correlations with index
            train_ratio: Ratio of data to use for in-sample (default: 0.7 = 70%)
            slippage: Transaction cost percentage
        """
        self.returns = returns
        self.index_returns = index_returns
        self.correlation_matrix = correlation_matrix
        self.train_ratio = train_ratio
        self.slippage = slippage

        # Split data
        self._split_data()

    def _split_data(self):
        """Split data into in-sample and out-of-sample periods."""
        n_total = len(self.returns)
        n_train = int(n_total * self.train_ratio)

        self.train_returns = self.returns.iloc[:n_train]
        self.test_returns = self.returns.iloc[n_train:]

        self.train_index_returns = self.index_returns.iloc[:n_train]
        self.test_index_returns = self.index_returns.iloc[n_train:]
        
        # Compute period-specific correlation matrices
        # IMPORTANT: Correlation must be computed separately for each period
        self.train_correlation_matrix = self._compute_correlation_matrix(
            self.train_returns, self.train_index_returns
        )
        self.test_correlation_matrix = self._compute_correlation_matrix(
            self.test_returns, self.test_index_returns
        )

        print(f"Data split:")
        print(f"  In-sample:     {self.train_returns.index[0].date()} to {self.train_returns.index[-1].date()} ({len(self.train_returns)} days)")
        print(f"  Out-of-sample: {self.test_returns.index[0].date()} to {self.test_returns.index[-1].date()} ({len(self.test_returns)} days)")

    def _compute_correlation_matrix(self, returns: pd.DataFrame, index_returns: pd.Series) -> pd.Series:
        """
        Compute correlation matrix between constituents and index for a specific period.
        
        Args:
            returns: Constituent returns for the period
            index_returns: Index returns for the period
            
        Returns:
            Series of correlations indexed by ticker
        """
        correlations = pd.Series(
            [returns[ticker].corr(index_returns) for ticker in returns.columns],
            index=returns.columns
        )
        return correlations.sort_values(ascending=False)

    def backtest_strategy(
        self,
        selection_model,
        optimizer,
        strategy_name: str,
        sector_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Backtest a strategy with proper train/test methodology.

        Args:
            selection_model: Model for asset selection (must have .select() method)
            optimizer: Portfolio optimizer (must have .optimize() method)
            strategy_name: Name of the strategy
            sector_mapping: Optional sector mapping for stratified models

        Returns:
            Dictionary with comprehensive metrics following Wang et al.
        """
        print(f"\n{'='*80}")
        print(f"BACKTESTING: {strategy_name}")
        print(f"{'='*80}")

        # ===== IN-SAMPLE PHASE =====
        print("\n[IN-SAMPLE PHASE]")
        
        # 1. Select assets on in-sample data USING IN-SAMPLE CORRELATIONS
        print("1. Selecting assets...")
        if sector_mapping is not None:
            selected_assets = selection_model.select(
                returns=self.train_returns,
                correlation_matrix=self.train_correlation_matrix,  # Use training period correlations
                sector_mapping=sector_mapping
            )
        else:
            selected_assets = selection_model.select(
                returns=self.train_returns,
                correlation_matrix=self.train_correlation_matrix  # Use training period correlations
            )
        
        print(f"   ✓ Selected {len(selected_assets)} assets")
        
        # 2. Optimize weights on in-sample data
        print("2. Optimizing weights...")
        selected_train_returns = self.train_returns[selected_assets]
        weights = optimizer.optimize(selected_train_returns, self.train_index_returns)
        print(f"   ✓ Weights optimized (objective: {optimizer.optimization_result.fun:.6f})")
        
        # 3. Compute in-sample metrics
        print("3. Computing in-sample metrics...")
        in_sample_metrics = self._compute_metrics(
            selected_assets=selected_assets,
            weights=weights,
            returns=self.train_returns,
            index_returns=self.train_index_returns,
            period_name="in-sample"
        )

        # ===== OUT-OF-SAMPLE PHASE =====
        print("\n[OUT-OF-SAMPLE PHASE]")
        print("4. Computing out-of-sample metrics...")
        out_sample_metrics = self._compute_metrics(
            selected_assets=selected_assets,
            weights=weights,
            returns=self.test_returns,
            index_returns=self.test_index_returns,
            period_name="out-of-sample"
        )

        # ===== CONSISTENCY =====
        consistency = abs(in_sample_metrics['TE'] - out_sample_metrics['TE'])
        
        # ===== FULL PERIOD =====
        print("\n[FULL PERIOD]")
        print("5. Computing full period metrics...")
        full_metrics = self._compute_metrics(
            selected_assets=selected_assets,
            weights=weights,
            returns=self.returns,
            index_returns=self.index_returns,
            period_name="full"
        )

        # Compile results
        results = {
            'strategy_name': strategy_name,
            'selected_assets': selected_assets,
            'weights': weights,
            'num_assets': len(selected_assets),
            
            # In-sample metrics (Wang et al.)
            'TE_in': in_sample_metrics['TE'],
            'Std_TE_in': in_sample_metrics['Std_TE'],
            'Mean_Corr_in': in_sample_metrics['Mean_Corr'],
            'Return_in': in_sample_metrics['Return'],
            'Sharpe_in': in_sample_metrics['Sharpe'],
            
            # Out-of-sample metrics (Wang et al.)
            'TE_out': out_sample_metrics['TE'],
            'Std_TE_out': out_sample_metrics['Std_TE'],
            'Mean_Corr_out': out_sample_metrics['Mean_Corr'],
            'Return_out': out_sample_metrics['Return'],
            'Sharpe_out': out_sample_metrics['Sharpe'],
            
            # Consistency (Wang et al.)
            'Consistency': consistency,
            
            # Full period
            'TE_full': full_metrics['TE'],
            'Return_full': full_metrics['Return'],
            'Sharpe_full': full_metrics['Sharpe'],
            
            # Period info
            'in_sample_start': self.train_returns.index[0],
            'in_sample_end': self.train_returns.index[-1],
            'out_sample_start': self.test_returns.index[0],
            'out_sample_end': self.test_returns.index[-1],
            'n_train': len(self.train_returns),
            'n_test': len(self.test_returns),
        }

        # Print summary
        self._print_summary(results)

        return results

    def _compute_metrics(
        self,
        selected_assets: List[str],
        weights: pd.Series,
        returns: pd.DataFrame,
        index_returns: pd.Series,
        period_name: str
    ) -> Dict:
        """
        Compute all metrics for a given period following Wang et al.

        TE formula from Wang et al.:
        TE(w) = (1/T) * ||I - R*w||_2^2
        Then: tracking_error = sqrt(TE)
        
        Std_TE: Standard deviation of (I_t - R_t*w)^2
        """
        # Align returns
        selected_returns = returns[selected_assets]
        common_dates = selected_returns.index.intersection(index_returns.index)
        X = selected_returns.loc[common_dates].values
        y = index_returns.loc[common_dates].values
        w = weights.values
        T = len(common_dates)

        # Portfolio returns
        portfolio_returns = X @ w

        # Tracking differences (I_t - R_t*w)
        tracking_diff = y - portfolio_returns

        # TE following Wang et al.: TE(w) = (1/T) * ||I - R*w||_2^2
        TE_raw = (1/T) * np.sum(tracking_diff ** 2)
        TE = np.sqrt(TE_raw) * 100  # Convert to percentage

        # Std_TE: Standard deviation of squared tracking errors (robustness measure)
        squared_errors = tracking_diff ** 2
        Std_TE = np.std(squared_errors) * 100

        # Mean correlation of selected assets with index FOR THIS PERIOD
        # CRITICAL: Must compute correlation on the specific period (train or test), not full sample
        period_correlations = []
        for asset in selected_assets:
            if asset in returns.columns:
                asset_returns = returns.loc[common_dates, asset]
                corr = asset_returns.corr(index_returns.loc[common_dates])
                period_correlations.append(corr)
        
        Mean_Corr = np.mean(period_correlations) if period_correlations else 0.0

        # Portfolio return
        cumulative_return = (1 + portfolio_returns).prod() - 1
        Return = cumulative_return * 100

        # Sharpe ratio (annualized, assuming 0% risk-free rate)
        if np.std(portfolio_returns) > 0:
            Sharpe = (portfolio_returns.mean() * 252) / (np.std(portfolio_returns) * np.sqrt(252))
        else:
            Sharpe = 0.0

        metrics = {
            'TE': TE,
            'Std_TE': Std_TE,
            'Mean_Corr': Mean_Corr,
            'Return': Return,
            'Sharpe': Sharpe,
            'portfolio_returns': portfolio_returns,
            'tracking_diff': tracking_diff,
        }

        print(f"   ✓ {period_name.upper()}: TE={TE:.4f}%, Std_TE={Std_TE:.4f}, Mean_Corr={Mean_Corr:.4f}, Return={Return:.2f}%")

        return metrics

    def _print_summary(self, results: Dict):
        """Print comprehensive summary following Wang et al. format."""
        print(f"\n{'='*80}")
        print(f"WANG ET AL. EVALUATION SUMMARY: {results['strategy_name']}")
        print(f"{'='*80}")
        print(f"\n📊 ASSET SELECTION")
        print(f"   Number of assets: {results['num_assets']}")
        print(f"   Mean correlation: {results['Mean_Corr_in']:.4f} (in-sample), {results['Mean_Corr_out']:.4f} (out-of-sample)")
        
        print(f"\n📈 IN-SAMPLE PERFORMANCE")
        print(f"   TE_in:      {results['TE_in']:.4f}%")
        print(f"   Std_TE_in:  {results['Std_TE_in']:.4f}")
        print(f"   Return_in:  {results['Return_in']:.2f}%")
        print(f"   Sharpe_in:  {results['Sharpe_in']:.4f}")
        
        print(f"\n📉 OUT-OF-SAMPLE PERFORMANCE")
        print(f"   TE_out:     {results['TE_out']:.4f}%")
        print(f"   Std_TE_out: {results['Std_TE_out']:.4f}")
        print(f"   Return_out: {results['Return_out']:.2f}%")
        print(f"   Sharpe_out: {results['Sharpe_out']:.4f}")
        
        print(f"\n🎯 CONSISTENCY")
        print(f"   |TE_in - TE_out|: {results['Consistency']:.4f}%")
        
        print(f"\n📅 PERIOD INFORMATION")
        print(f"   In-sample:     {results['in_sample_start'].date()} to {results['in_sample_end'].date()} ({results['n_train']} days)")
        print(f"   Out-of-sample: {results['out_sample_start'].date()} to {results['out_sample_end'].date()} ({results['n_test']} days)")
        print(f"{'='*80}\n")


class WangComparator:
    """Compare multiple strategies using Wang et al. metrics."""

    def __init__(self):
        self.results = []

    def add_result(self, result: Dict):
        """Add a strategy result."""
        self.results.append(result)

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table following Wang et al. format.
        
        Returns:
            DataFrame with rows=strategies, columns=metrics
        """
        data = []
        for r in self.results:
            data.append({
                'Strategy': r['strategy_name'],
                'K': r['num_assets'],
                'Mean_Corr_in': r['Mean_Corr_in'],
                'TE_in (%)': r['TE_in'],
                'Std_TE_in': r['Std_TE_in'],
                'Mean_Corr_out': r['Mean_Corr_out'],
                'TE_out (%)': r['TE_out'],
                'Std_TE_out': r['Std_TE_out'],
                'Consistency': r['Consistency'],
                'Return_in (%)': r['Return_in'],
                'Return_out (%)': r['Return_out'],
                'Sharpe_in': r['Sharpe_in'],
                'Sharpe_out': r['Sharpe_out'],
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('Strategy')
        return df

    def print_comparison(self):
        """Print formatted comparison table."""
        df = self.get_comparison_table()
        print("\n" + "="*120)
        print("WANG ET AL. (2018) COMPARISON TABLE - PARADIGM 1: SELECTION + OPTIMIZATION")
        print("="*120)
        print(df.to_string())
        print("="*120)
        print("\n📝 INTERPRETATION GUIDE (from Wang et al.):")
        print("   - Lower TE = better tracking")
        print("   - Lower Std_TE = more robust (less variance in tracking error)")
        print("   - Higher Mean_Corr = better asset quality")
        print("   - Lower Consistency = less overfitting (TE_in ≈ TE_out)")
        print("   - S-Strategy expected: moderate TE, good consistency, high correlation")
        print("="*120 + "\n")

    def save_to_csv(self, filepath: str = "./results/wang_comparison.csv"):
        """Save comparison table to CSV."""
        from pathlib import Path
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        df = self.get_comparison_table()
        df.to_csv(filepath)
        print(f"✓ Results saved to {filepath}")
