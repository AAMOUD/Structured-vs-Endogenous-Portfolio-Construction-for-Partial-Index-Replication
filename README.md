# S&P 500 Index Replication: Paradigm 1 vs Paradigm 2 Analysis

Implementation of Wang et al. (2018) methodology for comparing portfolio construction paradigms.

## 📁 Project Structure

```
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
│
├── Shared Modules/
│   ├── data.py                 # Data fetching and preprocessing
│   └── config.py               # Configuration parameters
│
├── paradigm1/                  # Paradigm 1: Selection + Optimization
│   ├── main.ipynb              # Full analysis notebook
│   ├── models.py               # Three selection models (S-Strategy, Clustering, CorrelationRanking)
│   ├── optimization.py         # Quadratic programming optimizer
│   ├── backtest.py             # Simple backtest engine
│   ├── backtest_wang.py        # Wang et al. compliant evaluation framework
│   └── results/                # Paradigm 1 results directory
│       ├── wang_evaluation_results.csv      # Wang evaluation (K=50)
│       └── multi_k_wang_evaluation.csv      # Multi-K analysis (K=50-200)
│
├── paradigm2/                  # Paradigm 2: O Strategy (Endogenous)
│   ├── main.ipynb              # O Strategy analysis
│   ├── models.py               # 4 O Strategy models (MIQP, LASSO, ElasticNet, ReweightedL1)
│   ├── optimization.py         # (unused - optimization in models.py)
│   ├── backtest.py             # Wang backtesting for O Strategy
│   └── results/                # Paradigm 2 results directory
│       └── paradigm2_wang_evaluation.csv    # O Strategy evaluation
│
└── data/
    ├── sp500_tickers.csv       # Full S&P 500 constituent list with sectors
    ├── sp500_*.csv             # Cached price/return data (7 files)
    │
    └── exports/                # Reference data exports
        ├── sp500_constituents_info.csv
        ├── sp500_correlations_with_sectors.csv
        └── sp500_sector_statistics.csv
```

## 🎯 Two Paradigms for Index Replication

### Paradigm 1: Selection + Optimization (Structured - COMPLETED ✅)

Two-stage approach:
1. **Selection**: Choose K assets using one of three methods
2. **Optimization**: Minimize tracking error via quadratic programming

**Three Models Implemented:**

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

### Paradigm 2: O Strategy (Endogenous - IMPLEMENTED ✅)

One-stage approach:
- **Simultaneous** asset selection and weight optimization
- Endogenous portfolio construction
- Expected: Better optimization but higher complexity

**Four Models Implemented:**

1. **MIQP Cardinality**
   - Mixed-Integer Quadratic Programming
   - Exact solution with binary variables: x_i ∈ {0,1}
   - Requires: SCIP (free) or CPLEX (academic license)
   - Expected: Best TE but slowest

2. **LASSO**
   - L1 regularization: min ||R*w - I||² + λ||w||₁
   - Fully convex, no integer variables
   - Fast and stable
   - Expected: Good sparsity, fast execution

3. **Elastic Net**
   - L1 + L2 regularization: λ₁||w||₁ + λ₂||w||²₂
   - More stable than pure LASSO
   - Expected: Better robustness than LASSO

4. **Reweighted L1**
   - Iterative reweighting: Σ |w_i| / (|w_i^prev| + ε)
   - Approximates L0 norm (cardinality)
   - Fully convex at each iteration
   - Expected: Best sparsity among convex methods

## � Mathematical Formulations

### Paradigm 1: Two-Stage Approach
```
Stage 1 (Selection): Choose K assets using S-Strategy / Clustering / Correlation
Stage 2 (Optimization): 
    min ||R_S * w - I||²
    s.t. sum(w) = 1
         0 ≤ w_i ≤ 1
```

### Paradigm 2: One-Stage Approach

**MIQP Cardinality:**
```
min (1/T) * ||R * w - I||²
s.t. sum(w) = 1
     sum(x) = K
     l*x_i ≤ w_i ≤ u*x_i
     x_i ∈ {0,1}
```

**LASSO:**
```
min (1/T) * ||R * w - I||² + λ * ||w||₁
s.t. sum(w) = 1
     w ≥ 0
```

**Elastic Net:**
```
min (1/T) * ||R * w - I||² + λ₁*||w||₁ + λ₂*||w||²₂
s.t. sum(w) = 1
     w ≥ 0
```

**Reweighted L1 (Iterative):**
```
For t = 1 to max_iter:
    min (1/T) * ||R * w - I||² + λ * Σ |w_i| / (|w_i^(t-1)| + ε)
    s.t. sum(w) = 1
         w ≥ 0
```

## �📊 Evaluation Framework (Wang et al. 2018)

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

### 2. Run Paradigm 1 Analysis
Open `paradigm1/main.ipynb` and run cells in order:
- **Section 1**: Data loading (uses cached data)
- **Section 4.5**: Wang evaluation (K=50)
- **Section 4.6**: Multi-K analysis (K=50-200)

### 3. Run Paradigm 2 Analysis
Open `paradigm2/main.ipynb` and run cells in order:
- **Section 1**: Setup and imports
- **Section 2**: Initialize O Strategy models
- **Section 3**: Compute correlations
- **Section 4**: Wang evaluation
- **Section 5**: Run comparison
- **Section 6**: Save results
- **Section 7**: Compare with Paradigm 1
- **Section 8**: Visualization

### 4. Results
Results automatically saved:
- `./paradigm1/results/wang_evaluation_results.csv` (Paradigm 1, K=50)
- `./paradigm1/results/multi_k_wang_evaluation.csv` (Paradigm 1, K=50-200)
- `./paradigm2/results/paradigm2_wang_evaluation.csv` (Paradigm 2)

Reference data saved to:
- `./data/exports/` (constituent info, correlations, sector stats)

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
- ✅ `paradigm1/results/wang_evaluation_results.csv` (with Mean_Corr fix)
- ✅ `paradigm1/results/multi_k_wang_evaluation.csv` (Multi-K analysis)
- ✅ `paradigm2/results/paradigm2_wang_evaluation.csv` (O Strategy results)
- 📊 `data/exports/` (reference data only)

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

### Paradigm 1 (Completed ✅):
- ✅ Core implementation complete
- ✅ Wang et al. evaluation framework implemented
- ✅ Mean_Corr bug fixed (now computed per period)
- ✅ Multi-K analysis implemented
- ✅ Project structure reorganized
- ✅ Ready for thesis results

### Paradigm 2 (Completed ✅):
- ✅ MIQP Cardinality model implemented
- ✅ LASSO model implemented
- ✅ Elastic Net model implemented
- ✅ Reweighted L1 model implemented
- ✅ Wang backtesting framework implemented
- ✅ Paradigm comparison framework implemented
- ✅ Ready for thesis comparison

### Next Steps:
1. Install cvxpy: `pip install cvxpy`
2. (Optional) Install SCIP for MIQP: `pip install pyscipopt`
3. Run paradigm2/main.ipynb to get O Strategy results
4. Compare Paradigm 1 vs Paradigm 2 using Section 7 outputs

---

**Last Updated**: February 27, 2026  
**Paradigm 1 Status**: Production-ready for thesis  
**Paradigm 2 Status**: Implementation complete, ready for evaluation
