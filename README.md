# Online vs Batch Portfolio Optimization in Non-Stationary Markets

## Overview

This repository contains the implementation and empirical analysis for the research project:

> **"Can online portfolio selection methodologies systematically outperform traditional batch allocation frameworks under non-stationary market regimes, specifically during intervals of systemic financial distress?"**

The project compares **Batch** (offline) and **Online** portfolio selection methods across a 13-year backtest (2013–2026) that includes three major financial crises.

---

## Research Question

Evaluate whether **online learning algorithms** offer advantages over **classical batch optimization** in terms of:
- Risk-adjusted returns (Sharpe & Calmar Ratios)
- Drawdown mitigation
- Turnover and transaction cost efficiency
- Adaptability during market crises

---

## Asset Universe

A universe of **10 sector ETFs** divided into two baskets:

| Defensive (S_D) | Aggressive (S_A) |
|-----------------|------------------|
| GLD, TLT, XLP, XLU, XLV | XLK, XLF, XLY, XLI, XLE |

---

## Strategies

### Batch Methods
- **Mean-Variance Optimization (MVO)**
- **Minimum Variance (MinVar)**
- **Risk Parity (RP)**

### Online Methods
- **Exponential Gradient (EG)**
- **Adaptive Gradient (AdaGrad)**
- **Passive-Aggressive Mean Reversion (PAMR)**

---

## Key Findings

### H1 – Long-Term Performance
> **Not supported.** Online methods do not consistently outperform batch methods over the full horizon.

| Metric | Batch Class | Online Class |
|--------|-------------|--------------|
| Sharpe Ratio | 0.738 | 0.746 |
| Calmar Ratio | 0.433 | 0.377 |
| Max Drawdown | -0.270 | -0.300 |
| Turnover | 3.777 | 0.265 |

**Key insight:** Online methods achieve **comparable returns with ~14x lower turnover**, offering significant implementation efficiency.

---

### H2 – Crisis Adaptability
> **Conditionally accepted.** Online methods excel in mean-reverting shocks and regime shifts, but underperform during sustained momentum rallies.

| Crisis Period | Batch Sharpe | Online Sharpe |
|---------------|--------------|---------------|
| COVID-19 (2020) | -0.259 | +0.008 |
| Inflation (2021–2022) | -0.050 | +0.026 |
| Geopolitical (2026) | +1.710 | +0.303 |

Online methods adapt faster and rebalance smoothly, but batch methods can benefit from structural inertia during strong trends.

---

### H3 – Hyperparameter Dependency
> **Supported.** The relative performance between strategy classes depends critically on transaction costs and turnover limits.

- Grid search over `Γ = {0, 0.0005, 0.001, 0.0025, 0.005}` and `T = {0.1, 0.25, 0.5, 1.0, 2.0}`
- Batch methods benefit from higher transaction costs and looser turnover constraints
- Online methods are more robust across hyperparameter settings

---

## Implementation

- **Language:** Python
- **Optimization:** CVXPY (Batch), custom solvers (Online)
- **Backtest Period:** 2013-01-01 to 2026-06-01
- **Constraints:** Long-only, turnover cap, transaction costs (10 bps)
- **Metrics:** Sharpe, Calmar, MDD, CAGR, Turnover

---

## Visual Outputs

- Portfolio weight evolution
- Turnover comparison
- Grid-search performance surfaces (Sharpe, Calmar, MDD)
- Crisis-period portfolio decomposition

---

## References

Key sources referenced in the paper:
- Markowitz (1952) – Mean-Variance Optimization
- Helmbold et al. (1998) – Exponential Gradient
- Li et al. (2012) – Passive-Aggressive Mean Reversion
- NBER, FOMC, BLS, EIA announcements for crisis period identification

---

## Author

**Alessio Dal Pozzolo**  
Master's Degree in Computational Finance  
University of Padova, 2025/2026

---

## Contact

For questions or collaborations, feel free to open an issue or reach out via GitHub.

---
