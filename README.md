# Online vs Batch Portfolio Optimization in Non-Stationary Markets

## Overview

This repository contains the implementation and empirical analysis for the research project:

> **"Can online portfolio selection methodologies outperform traditional batch allocation frameworks under non-stationary market regimes, particularly during periods of financial stress?"**

The project compares Batch (offline) and Online portfolio selection methods across a 13-year backtest (2013–2026).

---

## Research Question

Evaluate whether online learning algorithms provide advantages over classical batch optimization methods in terms of:

- Risk-adjusted returns (Sharpe and Calmar ratios)
- Drawdown control
- Turnover efficiency
- Regime adaptability

---

## Asset Universe

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

## H1 – Long-Term Performance

| Strategy | Sharpe | Calmar | MDD | CAGR | Turnover |
|----------|--------|--------|-----|------|----------|
| MVO | 0.707926 | 0.469789 | -0.267619 | 0.125334 | 0.235404 |
| MIN_VAR | 0.733259 | 0.403577 | -0.261938 | 0.105710 | 0.077824 |
| RISK_PAR | 0.756898 | 0.385414 | -0.296657 | 0.114332 | 0.010728 |
| EXP_GRAD | 0.751908 | 0.385115 | -0.295330 | 0.113729 | 0.003277 |
| ADA_GRAD | 0.751387 | 0.383899 | -0.295073 | 0.113271 | 0.001075 |
| PAMR | 0.736008 | 0.362200 | -0.309061 | 0.111923 | 0.021173 |

### Conclusion – H1

Empirical results show a relatively narrow dispersion in Sharpe ratios across strategies, indicating that performance differences are limited once identical constraints and a shared asset universe are imposed. This suggests that portfolio outcomes are primarily driven by the constraint structure and underlying market exposure rather than the specific optimization objective.

Online methods exhibit substantially lower turnover compared to batch approaches, highlighting a clear operational advantage in terms of transaction cost efficiency. However, this does not translate into a clear performance superiority, reinforcing the idea that optimizer choice plays a secondary role relative to structural constraints.

---

## H2 – Liquidity Shock

| Strategy | Sharpe | Calmar | MDD | CAGR | Turnover |
|----------|--------|--------|-----|------|----------|
| ADA_GRAD | -0.005145 | -0.262358 | -0.295374 | -0.077494 | 0.000453 |
| EXP_GRAD | -0.011334 | -0.270993 | -0.296608 | -0.080379 | 0.001793 |
| MIN_VAR | -0.101031 | -0.377556 | -0.303321 | -0.114521 | 0.031453 |
| MVO | -0.575573 | -0.653903 | -0.277661 | -0.181564 | 0.095043 |
| PAMR | 0.039484 | -0.209977 | -0.295434 | -0.062034 | 0.015668 |
| RISK_PAR | -0.005659 | -0.263665 | -0.296144 | -0.078083 | 0.002275 |

### Conclusion – H2 (Liquidity Shock)

During periods of liquidity stress, all strategies experience significant performance degradation, confirming strong sensitivity of portfolio construction methods to crisis regimes. PAMR shows relative resilience in Sharpe performance, while MVO exhibits the strongest deterioration, consistent with its sensitivity to covariance estimation error under stressed conditions.

Overall, online methods display smoother adjustment dynamics, whereas batch methods appear more vulnerable to abrupt changes in market structure. However, differences across strategies remain limited in magnitude, suggesting that crisis performance is dominated by systemic market effects rather than optimization methodology.

---

## H2 – Regime Shift

| Strategy | Sharpe | Calmar | MDD | CAGR | Turnover |
|----------|--------|--------|-----|------|----------|
| ADA_GRAD | 0.030398 | 0.025224 | -0.161855 | 0.004083 | 0.000093 |
| EXP_GRAD | 0.038177 | 0.032168 | -0.161953 | 0.005210 | 0.000430 |
| MIN_VAR | 0.096177 | 0.082159 | -0.167885 | 0.013793 | 0.007193 |
| MVO | -0.342785 | -0.372866 | -0.251620 | -0.093821 | 0.014508 |
| PAMR | 0.008962 | 0.009265 | -0.159337 | 0.001476 | 0.003424 |
| RISK_PAR | 0.033948 | 0.028558 | -0.161693 | 0.004618 | 0.000827 |

### Conclusion – H2 (Regime Shift)

In regime transition periods, Sharpe ratios compress significantly across all online methods, indicating limited exploitable signal under structural market changes. MVO remains consistently underperforming, likely due to instability in covariance inversion under shifting conditions.

Minimum variance emerges as the most robust batch method, suggesting that covariance-driven allocation provides greater stability than return-based optimization when signal quality deteriorates. Turnover remains low across all strategies, reinforcing the dominance of inertia effects in portfolio adjustments.

---

## H2 – Structural Market Rally

| Strategy | Sharpe | Calmar | MDD | CAGR | Turnover |
|----------|--------|--------|-----|------|----------|
| MVO | 1.841758 | 3.677503 | -0.113082 | 0.415859 | 0.071222 |
| MIN_VAR | 1.644324 | 3.362831 | -0.071061 | 0.238965 | 0.026677 |
| EXP_GRAD | 0.499329 | 0.385115 | -0.064252 | 0.083108 | 0.001512 |
| ADA_GRAD | 0.443594 | 1.202082 | -0.064019 | 0.076956 | 0.000262 |
| RISK_PAR | 0.437101 | 1.194503 | -0.063768 | 0.076171 | 0.002935 |
| PAMR | -0.033481 | 0.456653 | -0.062161 | 0.028386 | 0.011629 |

### Conclusion – H2 (Structural Rally)

During strong bullish regimes, batch methods benefit significantly from structural market trends, particularly MVO and minimum variance strategies. This suggests that part of their performance is driven by exposure alignment effects rather than purely optimal allocation decisions.

Online methods exhibit smoother performance trajectories with lower turnover, but do not fully capture upside during strong market expansions. PAMR underperforms due to its mean-reversion structure, which is misaligned with trending market conditions.

Overall, results indicate that performance differences across strategies are regime-dependent and heavily influenced by underlying market structure.

---

## Implementation

- Python (CVXPY + custom solvers)
- Backtest: 2013–2026
- Long-only constraints
- Transaction costs: 10 bps
