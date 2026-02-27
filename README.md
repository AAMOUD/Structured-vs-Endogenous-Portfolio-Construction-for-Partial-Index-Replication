# S&P 500 Index Replication: Paradigm 1 Analysis

Implementation of Wang et al. (2018) methodology for comparing portfolio construction paradigms.

## 📁 Project Structure

```
├── main.ipynb                  # Main analysis notebook
├── requirements.txt            # Python dependencies
│
├── Core Modules/
│   ├── data.py                 # Data fetching and preprocessing
│   ├── models.py               # Three selection models (S-Strategy, Clustering, Correlation Ranking)
│   ├── optimization.py         # Quadratic programming optimizer
│   ├── backtest.py             # Simple backtest engine
│   └── backtest_wang.py        # Wang et al. compliant evaluation framework
│
├── config.py                   # Configuration parameters
│
└── data/
    ├── sp500_tickers.csv       # Full S&P 500 constituent list with sectors
    ├── sp500_*.csv             # Cached price/return data (7 files)
    │
    └── exports/
        ├── wang_evaluation_results.csv           # Wang evaluation (K=50)
        ├── multi_k_wang_evaluation.csv          # Multi-K analysis (K=50-200)
        ├── sp500_constituents_info.csv          # Constituent metadata
        ├── sp500_correlations_with_sectors.csv  # Correlation analysis
        └── sp500_sector_statistics.csv          # Sector composition
```

## 🎯 Paradigm 1: Selection + Optimization

Two-stage approach:
1. **Selection**: Choose K assets using one of three methods
2. **Optimization**: Minimize tracking error via quadratic programming

### Three Models Implemented:

1. **S-Strategy (Stratified Sampling)**
   - Structured approach with proportional sector allocation
   - Ensures sectoral diversification
   - Expected: Moderate TE, good consistency

2. **Clustering (K-means)**
   - Data-driven selection on return features
   - Discovers endogenous asset groupings
   - Expected: Adaptive to market structure

3. **Correlation Ranking**
   - Greedy selection of top-K correlated assets
   - Naive baseline
   - Expected: Lowest TE_in, risk of overfitting

## 📊 Evaluation Framework (Wang et al. 2018)

### Proper Train/Test Methodology:
- **70% in-sample** (training): Asset selection and weight optimization
- **30% out-of-sample** (testing): Performance evaluation

### Key Metrics:
- **TE_in, TE_out**: Tracking Error (%) = sqrt((1/T) * ||I - R*w||²)
- **Std_TE_in, Std_TE_out**: Robustness (std of squared tracking errors)
- **Mean_Corr_in, Mean_Corr_out**: Mean correlation of selected assets with index
- **Consistency**: |TE_in - TE_out| (overfitting measure)
- **Returns**: Portfolio cumulative returns
- **Sharpe Ratio**: Risk-adjusted performance

## 🚀 Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Analysis
Open `main.ipynb` and run cells in order:
- **Section 1**: Data loading (uses cached data)
- **Section 4.5**: Wang evaluation (K=50)
- **Section 4.6**: Multi-K analysis (K=50-200)

### 3. Results
Results automatically saved to `./data/exports/`:
- `wang_evaluation_results.csv`: Single-K evaluation
- `multi_k_wang_evaluation.csv`: Multi-K evaluation

## 📈 Expected Results

### Good S-Strategy Performance:
- TE_out: 1-3% (moderate tracking error)
- Consistency: < 1% (low overfitting)
- Mean_Corr: 0.90+ (high quality assets)

### Warning Signs:
- TE_out >> TE_in (severe overfitting)
- Consistency > 2% (poor generalization)
- Mean_Corr_out << Mean_Corr_in (asset quality degradation)

## ⚠️ Important Notes

### Methodology:
- ✅ Selection and optimization on **in-sample only**
- ✅ Correlations computed **separately** for in-sample and out-of-sample
- ✅ Proper constraints: sum(w)=1, weight bounds
- ✅ Correct TE formula from Wang et al.

### Files to Use for Thesis:
- ✅ `wang_evaluation_results.csv` (with Mean_Corr fix)
- ✅ `multi_k_wang_evaluation.csv` (if generated)
- ❌ DO NOT use old `multi_k_backtest_results.csv` (deleted - invalid methodology)

## 📚 References

Wang, M., Xu H., Dai, Z., & Zaho, X. (2018). Structured vs Endogenous Portfolio Construction for Partial Index Replication. *Applied Stochastic Models in Business and Industry*.

## 🔧 Technical Details

### Data:
- **Universe**: ~500 S&P 500 constituents
- **Period**: 2020-2023
- **Source**: Yahoo Finance (yfinance)
- **Index**: Market-cap weighted returns (not ^GSPC)

### Optimization:
- **Method**: SLSQP (Sequential Least Squares Programming)
- **Objective**: min ||I - R_S @ w||²
- **Constraints**: sum(w)=1, 0 ≤ w_i ≤ 1

### Selection:
- **K**: Portfolio size (default 50)
- **Methods**: Stratified (sector-aware), Clustering (K-means), Correlation (greedy)

## 📝 Status

- ✅ Core implementation complete
- ✅ Wang et al. evaluation framework implemented
- ✅ Mean_Corr bug fixed (now computed per period)
- ✅ Multi-K analysis implemented
- ✅ Project cleaned up
- ✅ Ready for thesis results

---

**Last Updated**: February 27, 2026  
**Status**: Production-ready for thesis
