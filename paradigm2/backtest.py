"""Backtesting framework for Paradigm 2 - O Strategy using Wang et al. methodology."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time


class OStrategyWangBacktest:
    """
    Wang et al. (2018) backtesting for O Strategy models (Paradigm 2).
    
    Key differences from Paradigm 1:
    - Models simultaneously select and optimize (no separate selection step)
    - Tracks computational time (crucial for SO-strategy comparison)
    - Prevents lookahead bias: train on in-sample only
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
        Initialize Wang backtest for O Strategy.

        Args:
            returns: DataFrame of constituent returns (T x N)
            index_returns: Series of index returns (T,)
            correlation_matrix: Series of correlations with index
            train_ratio: Train/test split ratio (default 0.7)
            slippage: Transaction cost (default 0.001)
        """
        self.returns = returns
        self.index_returns = index_returns
        self.correlation_matrix = correlation_matrix
        self.train_ratio = train_ratio
        self.slippage = slippage
        
        # Split data
        self.train_returns, self.test_returns = self._split_data()
        self.train_index, self.test_index = self._split_index()
        self.train_corr, self.test_corr = self._split_correlations()
        
    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split returns into train/test."""
        split_idx = int(len(self.returns) * self.train_ratio)
        train = self.returns.iloc[:split_idx]
        test = self.returns.iloc[split_idx:]
        return train, test
    
    def _split_index(self) -> Tuple[pd.Series, pd.Series]:
        """Split index returns into train/test."""
        split_idx = int(len(self.index_returns) * self.train_ratio)
        train = self.index_returns.iloc[:split_idx]
        test = self.index_returns.iloc[split_idx:]
        return train, test
    
    def _split_correlations(self) -> Tuple[pd.Series, pd.Series]:
        """Compute correlations for train and test periods separately."""
        # Train period correlations
        train_corr = pd.Series(index=self.returns.columns, dtype=float)
        for ticker in self.returns.columns:
            common_idx = self.train_returns.index.intersection(self.train_index.index)
            if len(common_idx) > 10:
                corr = self.train_returns.loc[common_idx, ticker].corr(
                    self.train_index.loc[common_idx]
                )
                train_corr[ticker] = corr if not np.isnan(corr) else 0.0
            else:
                train_corr[ticker] = 0.0
        
        # Test period correlations
        test_corr = pd.Series(index=self.returns.columns, dtype=float)
        for ticker in self.returns.columns:
            common_idx = self.test_returns.index.intersection(self.test_index.index)
            if len(common_idx) > 10:
                corr = self.test_returns.loc[common_idx, ticker].corr(
                    self.test_index.loc[common_idx]
                )
                test_corr[ticker] = corr if not np.isnan(corr) else 0.0
            else:
                test_corr[ticker] = 0.0
        
        return train_corr, test_corr
    
    def backtest_strategy(
        self,
        model,
        strategy_name: str,
    ) -> Dict:
        """
        Backtest an O Strategy model.
        
        Args:
            model: OStrategyModel instance
            strategy_name: Name for results
            
        Returns:
            Dictionary with metrics
        """
        print(f"\n{'='*60}")
        print(f"Backtesting: {strategy_name}")
        print(f"{'='*60}")
        
        # === STEP 1: Optimize and select on IN-SAMPLE data ===
        print(f"\n1. Optimization (in-sample only)...")
        start_time = time.time()
        
        try:
            selected_assets, optimal_weights = model.optimize_and_select(
                self.train_returns,
                self.train_index
            )
        except Exception as e:
            print(f"   ✗ Optimization failed: {e}")
            return self._create_failed_result(strategy_name, str(e))
        
        optimization_time = time.time() - start_time
        
        print(f"   ✓ Selected {len(selected_assets)} assets")
        print(f"   ✓ Optimization time: {optimization_time:.2f}s")
        print(f"   ✓ Active positions: {len([w for w in optimal_weights.values() if abs(w) > 1e-6])}")
        
        # === STEP 2: Compute metrics ===
        metrics = self._compute_metrics(
            selected_assets,
            optimal_weights,
            strategy_name,
            optimization_time
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        selected_assets: List[str],
        weights: Dict[str, float],
        strategy_name: str,
        optimization_time: float
    ) -> Dict:
        """Compute Wang et al. metrics for O Strategy."""
        
        # Ensure weights are normalized
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Convert to aligned arrays
        tickers_ordered = list(weights.keys())
        weights_array = np.array([weights[t] for t in tickers_ordered])
        
        # === IN-SAMPLE METRICS ===
        train_subset = self.train_returns[tickers_ordered]
        train_idx_aligned = self.train_index.loc[train_subset.index]
        
        # Portfolio returns
        portfolio_returns_train = (train_subset.values @ weights_array)
        
        # Tracking errors (squared)
        te_squared_train = (portfolio_returns_train - train_idx_aligned.values) ** 2
        
        # TE_in: sqrt of mean squared TE
        TE_in = np.sqrt(np.mean(te_squared_train)) * 100  # Convert to %
        
        # Std_TE_in: std of squared TE
        Std_TE_in = np.std(te_squared_train) * 100
        
        # Mean_Corr_in: mean correlation of selected assets (in-sample)
        selected_corr_in = [self.train_corr[t] for t in tickers_ordered if t in self.train_corr.index]
        Mean_Corr_in = np.mean(selected_corr_in) if selected_corr_in else 0.0
        
        # === OUT-OF-SAMPLE METRICS ===
        test_subset = self.test_returns[tickers_ordered]
        test_idx_aligned = self.test_index.loc[test_subset.index]
        
        # Portfolio returns
        portfolio_returns_test = (test_subset.values @ weights_array)
        
        # Tracking errors (squared)
        te_squared_test = (portfolio_returns_test - test_idx_aligned.values) ** 2
        
        # TE_out
        TE_out = np.sqrt(np.mean(te_squared_test)) * 100
        
        # Std_TE_out
        Std_TE_out = np.std(te_squared_test) * 100
        
        # Mean_Corr_out
        selected_corr_out = [self.test_corr[t] for t in tickers_ordered if t in self.test_corr.index]
        Mean_Corr_out = np.mean(selected_corr_out) if selected_corr_out else 0.0
        
        # === CONSISTENCY ===
        Consistency = abs(TE_in - TE_out)
        
        # === RETURNS ===
        Returns_in = (1 + pd.Series(portfolio_returns_train)).prod() - 1
        Returns_out = (1 + pd.Series(portfolio_returns_test)).prod() - 1
        
        # === COMPUTATIONAL TIME ===
        Comp_Time = optimization_time
        
        # Print summary
        print(f"\n2. Results Summary:")
        print(f"   In-Sample:")
        print(f"      TE_in:        {TE_in:.4f}%")
        print(f"      Std_TE_in:    {Std_TE_in:.4f}%")
        print(f"      Mean_Corr_in: {Mean_Corr_in:.4f}")
        print(f"      Returns_in:   {Returns_in*100:.2f}%")
        print(f"\n   Out-of-Sample:")
        print(f"      TE_out:       {TE_out:.4f}%")
        print(f"      Std_TE_out:   {Std_TE_out:.4f}%")
        print(f"      Mean_Corr_out:{Mean_Corr_out:.4f}")
        print(f"      Returns_out:  {Returns_out*100:.2f}%")
        print(f"\n   Overfitting:")
        print(f"      Consistency:  {Consistency:.4f}%")
        print(f"\n   Efficiency:")
        print(f"      Comp. Time:   {Comp_Time:.2f}s")
        print(f"      Assets:       {len(selected_assets)}")
        
        return {
            'Strategy': strategy_name,
            'K': len(selected_assets),
            'TE_in': TE_in,
            'TE_out': TE_out,
            'Std_TE_in': Std_TE_in,
            'Std_TE_out': Std_TE_out,
            'Mean_Corr_in': Mean_Corr_in,
            'Mean_Corr_out': Mean_Corr_out,
            'Consistency': Consistency,
            'Returns_in': Returns_in * 100,
            'Returns_out': Returns_out * 100,
            'Comp_Time': Comp_Time,
            'Selected_Assets': ','.join(selected_assets[:10]) + '...' if len(selected_assets) > 10 else ','.join(selected_assets)
        }
    
    def _create_failed_result(self, strategy_name: str, error_msg: str) -> Dict:
        """Create result dict for failed optimization."""
        return {
            'Strategy': strategy_name,
            'K': 0,
            'TE_in': np.nan,
            'TE_out': np.nan,
            'Std_TE_in': np.nan,
            'Std_TE_out': np.nan,
            'Mean_Corr_in': np.nan,
            'Mean_Corr_out': np.nan,
            'Consistency': np.nan,
            'Returns_in': np.nan,
            'Returns_out': np.nan,
            'Comp_Time': 0.0,
            'Selected_Assets': f'FAILED: {error_msg}'
        }


class OStrategyComparator:
    """Compare multiple O Strategy models."""
    
    def __init__(self, backtest_engine: OStrategyWangBacktest):
        self.engine = backtest_engine
        self.results = []
    
    def compare_models(self, models: Dict) -> pd.DataFrame:
        """
        Compare multiple O Strategy models.
        
        Args:
            models: Dict of {name: model_instance}
            
        Returns:
            DataFrame with comparison results
        """
        print(f"\n{'='*80}")
        print(f"PARADIGM 2: O STRATEGY COMPARISON")
        print(f"{'='*80}")
        
        for name, model in models.items():
            result = self.engine.backtest_strategy(model, name)
            self.results.append(result)
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*80}")
        print(f"COMPARISON TABLE")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        
        return df
