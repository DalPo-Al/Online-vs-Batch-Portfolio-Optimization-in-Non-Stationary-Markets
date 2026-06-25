# Online vs Batch Portfolio Optimization in Non-Stationary Markets

## Overview

This repository contains the implementation and empirical analysis for the research project:

> **"Can online portfolio selection methodologies outperform traditional batch allocation frameworks under non-stationary market regimes, particularly during periods of financial stress?"**

The project compares **Batch (offline)** and **Online** portfolio selection methods across a 13-year backtest (2013–2026) covering multiple market regimes, including crisis and expansion phases.

---

## Research Question

Evaluate whether **online learning algorithms** provide advantages over **classical batch optimization methods** in terms of:

- Risk-adjusted returns (Sharpe and Calmar ratios)
- Drawdown control
- Turnover and transaction cost efficiency
- Adaptability across different market regimes

---

## Asset Universe

A universe of **10 sector ETFs** divided into two baskets:

| Defensive (S_D) | Aggressive (S_A) |
|-----------------|------------------|
| GLD, TLT, XLP, XLU, XLV | XLK, XLF, XLY, XLI, XLE |

---

## Strategies

### Batch Methods
- Mean-Variance Optimization (MVO)
- Minimum Variance (MinVar)
- Risk Parity (RP)

### Online Methods
- Exponential Gradient (EG)
- Adaptive Gradient (AdaGrad)
- Passive-Aggressive Mean Reversion (PAMR)

---

## Key Findings

### H1 – Long-Term Performance

| Strategy   | Sharpe      | Calmar      | MDD         | CAGR        | Turnover   |
|------------|------------|------------|------------|------------|------------|
| MVO        | 0.707926   | 0.469789   | -0.267619  | 0.125334   | 0.235404   |
| MIN_VAR    | 0.733259   | 0.403577   | -0.261938  | 0.105710   | 0.077824   |
| RISK_PAR   | 0.756898   | 0.385414   | -0.296657  | 0.114332   | 0.010728   |
| EXP_GRAD   | 0.751908   | 0.385115   | -0.295330  | 0.113729   | 0.003277   |
| ADA_GRAD   | 0.751387   | 0.383899   | -0.295073  | 0.113271   | 0.001075   |
| PAMR       | 0.736008   | 0.362200   | -0.309061  | 0.111923   | 0.021173   |
```

Empirical results show a relatively narrow dispersion in Sharpe ratios across strategies, suggesting that portfolio performance is largely driven by the shared constraint structure and underlying exposure rather than the specific optimization objective.

Online methods exhibit significantly lower turnover compared to batch methods, which has direct implications for transaction cost efficiency and operational feasibility in portfolio management.

---

### H2 – Crisis Period Analysis

## Liquidity Shock

| Strategy   | Sharpe      | Calmar      | MDD         | CAGR        | Turnover   |
|------------|------------|------------|------------|----------------|------------|
| ADA_GRAD   | -0.005145   | -0.262358   | -0.295374   | -0.077494   | 0.000453   |
| EXP_GRAD   | -0.011334   | -0.270993   | -0.296608   | -0.080379   | 0.001793   |
| MIN_VAR    | -0.101031   | -0.377556   | -0.303321   | -0.114521   | 0.031453   |
| MVO        | -0.575573   | -0.653903   | -0.277661   | -0.181564   | 0.095043   |
| PAMR       | 0.039484    | -0.209977   | -0.295434   | -0.062034   | 0.015668   |
| RISK_PAR   | -0.005659   | -0.263665   | -0.296144   | -0.078083   | 0.002275   |

During liquidity stress conditions, all strategies exhibit significant degradation in performance relative to normal market conditions. PAMR shows the least deterioration in Sharpe ratio, while MVO experiences the strongest decline, consistent with its sensitivity to covariance estimation instability.

Online methods display smoother adjustments, while batch methods show higher sensitivity to abrupt market changes. Risk parity appears more stable than minimum variance in this regime, although differences remain moderate.

---

## Regime Shift

| Strategy   | Sharpe      | Calmar      | MDD         | CAGR        | Turnover   |
|------------|------------|------------|------------|------------|------------|
| ADA_GRAD   | 0.030398   | 0.025224   | -0.161855  | 0.004083   | 0.000093   |
| EXP_GRAD   | 0.038177   | 0.032168   | -0.161953  | 0.005210   | 0.000430   |
| MIN_VAR    | 0.096177   | 0.082159   | -0.167885  | 0.013793   | 0.007193   |
| MVO        | -0.342785  | -0.372866  | -0.251620  | -0.093821  | 0.014508   |
| PAMR       | 0.008962   | 0.009265   | -0.159337  | 0.001476   | 0.003424   |
| RISK_PAR   | 0.033948   | 0.028558   | -0.161693  | 0.004618   | 0.000827   |

In this regime, Sharpe ratios converge toward low values across online methods, while MVO continues to underperform. Minimum variance remains the most stable batch method, suggesting robustness of covariance-driven allocation when return estimates are weak.

---

## Structural Market Rally

| Strategy   | Sharpe      | Calmar      | MDD         | CAGR        | Turnover   |
|------------|------------|------------|------------|------------|------------|
| MVO        | 1.841758   | 3.677503   | -0.113082  | 0.415859   | 0.071222   |
| MIN_VAR    | 1.644324   | 3.362831   | -0.071061  | 0.238965   | 0.026677   |
| EXP_GRAD   | 0.499329   | 1.293456   | -0.064252  | 0.083108   | 0.001512   |
| ADA_GRAD   | 0.443594   | 1.202082   | -0.064019  | 0.076956   | 0.000262   |
| RISK_PAR   | 0.437101   | 1.194503   | -0.063768  | 0.076171   | 0.002935   |
| PAMR       | -0.033481  | 0.456653   | -0.062161  | 0.028386   | 0.011629   |

During strong bullish regimes, batch methods—particularly MVO and minimum variance—benefit significantly from structural market trends. This suggests that part of their performance is driven by exposure alignment rather than pure optimization efficiency.

Online methods maintain smoother performance profiles, with lower turnover and more stable but lower peak returns.

---

## H3 – Hyperparameter Sensitivity

Grid search is performed over:
- Transaction cost: Γ ∈ {0, 0.0005, 0.001, 0.0025, 0.005}
- Rebalancing intensity: T ∈ {0.1, 0.25, 0.5, 1.0, 2.0}

Empirical results indicate that performance is highly sensitive to transaction costs. Online methods are generally more robust under low-cost regimes due to frequent small adjustments, while batch methods become more competitive as costs increase and turnover penalization reduces overtrading effects.

Overall, results suggest that transaction costs play a critical role in determining the relative performance of optimization classes.

---

## Implementation

- Language: Python
- Optimization:
  - Batch: CVXPY
  - Online: custom gradient-based solvers
- Backtest Period: 2013-01-01 to 2026-06-01
- Constraints: long-only, turnover constraints, transaction costs (10 bps)
- Metrics: Sharpe ratio, Calmar ratio, Maximum Drawdown, CAGR, Turnover

---

## Visual Outputs

- Portfolio weight evolution
- Turnover comparison across methods
- Regime-specific performance analysis
- Grid-search sensitivity surfaces

---

## References

- Markowitz (1952) – Portfolio Selection
- Helmbold et al. (1998) – Exponential Gradient Methods
- Li et al. (2012) – Passive-Aggressive Mean Reversion
- NBER, FOMC, BLS, EIA datasets for regime classification

---

## Author

Alessio Dal Pozzolo  
MSc Computational Finance – University of Padova (2025/2026)

---

## Contact

Open an issue or contact via GitHub for collaboration or discussion.
