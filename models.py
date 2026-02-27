"""Portfolio selection models for index replication."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from config import PORTFOLIO_PARAMS


class SelectionModel:
    """Base class for selection models."""

    def __init__(self, K: int = PORTFOLIO_PARAMS["K"]):
        """
        Initialize selection model.

        Args:
            K: Number of assets to select
        """
        self.K = K
        self.selected_assets = None

    def select(self, *args, **kwargs) -> List[str]:
        """Select K assets. Must be implemented in subclasses."""
        raise NotImplementedError


class StratifiedSamplingModel(SelectionModel):
    """
    Stratified Sampling model - divides assets by sector and selects from each stratum.
    
    This is a structured approach that ensures sectoral representation.
    """

    def __init__(self, K: int = PORTFOLIO_PARAMS["K"], num_strata: int = None, selection_criterion: str = "correlation"):
        """
        Initialize stratified sampling model.

        Args:
            K: Number of assets to select
            num_strata: Number of strata. If None, uses PORTFOLIO_PARAMS
            selection_criterion: 'correlation', 'market_cap', or 'variance'
        """
        super().__init__(K)
        self.num_strata = num_strata or PORTFOLIO_PARAMS["num_strata"]
        self.selection_criterion = selection_criterion
        self.strata_allocations = None

    def select(
        self,
        returns: pd.DataFrame,
        correlation_matrix: pd.Series,
        sector_mapping: Dict[str, str] = None,
    ) -> List[str]:
        """
        Select assets using stratified sampling by sector.

        Args:
            returns: DataFrame of returns for all assets
            correlation_matrix: Series of correlations with index
            sector_mapping: Dictionary mapping ticker to sector

        Returns:
            List of selected ticker symbols
        """
        # If no sector mapping provided or empty, create random strata
        if sector_mapping is None or len(sector_mapping) == 0:
            return self._select_by_random_strata(returns, correlation_matrix)
        
        return self._select_by_sector_strata(returns, correlation_matrix, sector_mapping)

    def _select_by_sector_strata(
        self,
        returns: pd.DataFrame,
        correlation_matrix: pd.Series,
        sector_mapping: Dict[str, str],
    ) -> List[str]:
        """Select assets stratified by sector."""
        sectors = {}
        for ticker, sector in sector_mapping.items():
            if ticker in returns.columns:
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(ticker)

        if len(sectors) == 0:
            print("⚠ Warning: No sectors found in mapping, falling back to correlation ranking within random strata")
            return self._select_by_random_strata(returns, correlation_matrix)

        # Allocate assets to each sector proportionally to sector size
        num_sectors = len(sectors)
        total_assets = sum(len(assets) for assets in sectors.values())
        
        # Proportional allocation
        self.strata_allocations = {}
        allocated_total = 0
        for sector, assets in sectors.items():
            proportion = len(assets) / total_assets
            allocation = max(1, int(self.K * proportion))
            self.strata_allocations[sector] = allocation
            allocated_total += allocation
        
        # Adjust for rounding errors
        if allocated_total < self.K:
            # Add remaining to largest sectors
            sorted_sectors = sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True)
            for sector, _ in sorted_sectors:
                if allocated_total >= self.K:
                    break
                self.strata_allocations[sector] += 1
                allocated_total += 1
        elif allocated_total > self.K:
            # Remove from smallest allocations
            sorted_sectors = sorted(self.strata_allocations.items(), key=lambda x: x[1])
            for sector, _ in sorted_sectors:
                if allocated_total <= self.K:
                    break
                if self.strata_allocations[sector] > 1:
                    self.strata_allocations[sector] -= 1
                    allocated_total -= 1

        selected = []
        print(f"Stratified Sampling: Selecting from {num_sectors} sectors")
        for sector, assets in sectors.items():
            sector_corr = correlation_matrix[correlation_matrix.index.isin(assets)]
            k_sector = self.strata_allocations.get(sector, 0)
            
            if len(sector_corr) > 0 and k_sector > 0:
                selected_in_sector = sector_corr.nlargest(min(k_sector, len(sector_corr))).index.tolist()
                selected.extend(selected_in_sector)
                print(f"  {sector}: {len(selected_in_sector)}/{k_sector} assets")

        self.selected_assets = selected[:self.K]
        print(f"Total selected: {len(self.selected_assets)} assets")
        return self.selected_assets

    def _select_by_random_strata(
        self,
        returns: pd.DataFrame,
        correlation_matrix: pd.Series,
    ) -> List[str]:
        """Select assets stratified by random groupings."""
        all_tickers = returns.columns.tolist()
        np.random.shuffle(all_tickers)
        
        strata = [all_tickers[i::self.num_strata] for i in range(self.num_strata)]
        assets_per_stratum = max(1, self.K // self.num_strata)
        
        selected = []
        for stratum in strata:
            stratum_corr = correlation_matrix[correlation_matrix.index.isin(stratum)]
            k_stratum = min(assets_per_stratum, len(stratum_corr))
            selected_in_stratum = stratum_corr.nlargest(k_stratum).index.tolist()
            selected.extend(selected_in_stratum)

        self.selected_assets = selected[:self.K]
        return self.selected_assets


class ClusteringModel(SelectionModel):
    """
    Clustering model - data-driven selection using correlation-based clustering.
    
    Groups similar assets and selects representatives from each cluster.
    """

    def __init__(self, K: int = PORTFOLIO_PARAMS["K"], num_clusters: int = None):
        """
        Initialize clustering model.

        Args:
            K: Number of assets to select
            num_clusters: Number of clusters. If None, uses PORTFOLIO_PARAMS
        """
        super().__init__(K)
        self.num_clusters = num_clusters or PORTFOLIO_PARAMS["num_clusters"]
        self.clustering_model = None
        self.cluster_labels = None

    def select(
        self,
        returns: pd.DataFrame,
        correlation_matrix: pd.Series,
    ) -> List[str]:
        """
        Select assets using clustering on return patterns (feature space).

        Args:
            returns: DataFrame of returns for all assets
            correlation_matrix: Series of correlations with index

        Returns:
            List of selected ticker symbols
        """
        print(f"Clustering: Performing K-means with {self.num_clusters} clusters on return features")
        
        # Use returns as features for clustering (transpose to cluster assets)
        features = returns.T  # Each row is an asset, columns are time periods
        
        # Perform K-means clustering on return features
        n_clusters = min(self.num_clusters, len(returns.columns))
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.clustering_model.fit_predict(features)
        
        # Select representatives from each cluster
        selected = []
        assets = returns.columns.tolist()
        
        cluster_sizes = {}
        for cluster_id in range(n_clusters):
            cluster_assets = [assets[i] for i, label in enumerate(self.cluster_labels) if label == cluster_id]
            cluster_sizes[cluster_id] = len(cluster_assets)
            
            if cluster_assets:
                # For each cluster, select multiple assets based on cluster size
                # This ensures better representation of larger clusters
                cluster_corr = correlation_matrix[correlation_matrix.index.isin(cluster_assets)]
                if len(cluster_corr) > 0:
                    # Select top asset(s) from this cluster
                    n_select = max(1, int(len(cluster_assets) / len(assets) * self.K))
                    best_assets = cluster_corr.nlargest(min(n_select, len(cluster_corr))).index.tolist()
                    selected.extend(best_assets)
                    print(f"  Cluster {cluster_id}: {len(cluster_assets)} assets → selected {len(best_assets)}")

        # If we have fewer assets than K, take the most correlated overall
        if len(selected) < self.K:
            remaining_needed = self.K - len(selected)
            remaining_assets = [a for a in correlation_matrix.index if a not in selected]
            remaining_corr = correlation_matrix[correlation_matrix.index.isin(remaining_assets)]
            additional = remaining_corr.nlargest(remaining_needed).index.tolist()
            selected.extend(additional)
            print(f"  Added {len(additional)} additional assets to reach K={self.K}")
        elif len(selected) > self.K:
            # If we have too many, keep the most correlated ones
            selected_corr = correlation_matrix[correlation_matrix.index.isin(selected)]
            selected = selected_corr.nlargest(self.K).index.tolist()
            print(f"  Reduced to K={self.K} by keeping most correlated")

        self.selected_assets = selected[:self.K]
        print(f"Total selected: {len(self.selected_assets)} assets")
        return self.selected_assets


class CorrelationRankingModel(SelectionModel):
    """
    Correlation Ranking model - simple baseline selection.
    
    Selects the K assets with highest correlation to the index.
    """

    def __init__(self, K: int = PORTFOLIO_PARAMS["K"]):
        """
        Initialize correlation ranking model.

        Args:
            K: Number of assets to select
        """
        super().__init__(K)

    def select(
        self,
        returns: pd.DataFrame,
        correlation_matrix: pd.Series,
    ) -> List[str]:
        """
        Select assets with highest correlation to index (naive greedy approach).

        Args:
            returns: DataFrame of returns for all assets (not strictly used, kept for consistency)
            correlation_matrix: Series of correlations with index

        Returns:
            List of selected ticker symbols
        """
        print(f"Correlation Ranking: Selecting top {self.K} assets by correlation with index")
        
        # Simply take the K assets with highest correlation
        self.selected_assets = correlation_matrix.nlargest(self.K).index.tolist()
        
        print(f"  Selected asset correlations: min={correlation_matrix[self.selected_assets].min():.4f}, "
              f"max={correlation_matrix[self.selected_assets].max():.4f}, "
              f"mean={correlation_matrix[self.selected_assets].mean():.4f}")
        
        return self.selected_assets
