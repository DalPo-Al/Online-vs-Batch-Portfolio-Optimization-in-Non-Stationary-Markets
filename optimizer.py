from numpy.random import vonmises
import numpy as np
import pandas as pd
import cvxpy as cp
import joblib
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import time
import sys
import yfinance as yf

# Global transaction cost parameters (will be overwritten during grid search)
c = 0.001
tau = 0.50

# ---------- Optimization strategies (unchanged) ----------
def MVO(data, w_prev, N, lambda_risk=3.0):
    mu = data.mean(axis=0).values
    Sigma = data.cov().values
    if w_prev is None:
        w_prev = np.ones(N) / N
    w_t = w_prev.flatten()
    last_returns = data.iloc[-1].values
    current_price_relative = 1 + last_returns
    numerator = w_t * current_price_relative
    w_t_drift = numerator / max(np.sum(numerator), 1e-8)
    Sigma = Sigma + np.eye(N) * 1e-8
    w = cp.Variable(N)
    portfolio_return = mu @ w
    risk = cp.quad_form(w, Sigma)
    turnover = cp.norm(w - w_t_drift, 1)
    objective = cp.Maximize(portfolio_return - 0.5 * lambda_risk * risk - c * turnover)
    constraints = [cp.sum(w) == 1, w >= 0, turnover <= tau]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except:
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except:
            problem.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        return w_t_drift, 0.0
    optimized_w = np.array(w.value).squeeze()
    actual_turnover = np.linalg.norm(optimized_w - w_t_drift, ord=1)
    cost_val = c * actual_turnover
    return optimized_w, cost_val

def MIN_VAR(data, w_prev, N, lambda_risk=3.0):
    Sigma = data.cov().values
    if w_prev is None:
        w_prev = np.ones(N) / N
    w_t = w_prev.flatten()
    last_returns = data.iloc[-1].values
    current_price_relative = 1 + last_returns
    numerator = w_t * current_price_relative
    w_t_drift = numerator / max(np.sum(numerator), 1e-8)
    Sigma = Sigma + np.eye(N) * 1e-8
    w = cp.Variable(N)
    risk = cp.quad_form(w, Sigma)
    turnover = cp.norm(w - w_t_drift, 1)
    objective = cp.Maximize(-0.5 * lambda_risk * risk - c * turnover)
    constraints = [cp.sum(w) == 1, w >= 0, turnover <= tau]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except:
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except:
            problem.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        return w_t_drift, 0.0
    optimized_w = np.array(w.value).squeeze()
    actual_turnover = np.linalg.norm(optimized_w - w_t_drift, ord=1)
    cost_val = c * actual_turnover
    return optimized_w, cost_val

def sigma_p(w, Sigma):
    return np.sqrt(max(w.T @ Sigma @ w, 1e-8))

def RC(w, Sigma):
    sigma = sigma_p(w, Sigma)
    return (w * (Sigma @ w)) / sigma

# ------------------------------------------------------------
# RISK_PAR converted to convex formulation (Roncalli 2010)
# ------------------------------------------------------------
def RISK_PAR(data, w_prev, N):
    Sigma = data.cov().values
    Sigma += np.eye(N) * 1e-6          # regularization
    if w_prev is None:
        w_prev = np.ones(N) / N
    w_t = w_prev.flatten()
    last_returns = data.iloc[-1].values
    current_price_relative = 1 + last_returns
    numerator = w_t * current_price_relative
    w_t_drift = numerator / max(np.sum(numerator), 1e-8)

    # Equal risk budgets
    b = np.ones(N) / N

    w = cp.Variable(N)
    # Convex objective: 0.5 * w^T Sigma w - sum(b_i * log(w_i)) + c * ||w - w_t_drift||_1
    objective = cp.Minimize(
        0.5 * cp.quad_form(w, Sigma)
        - cp.sum(cp.multiply(b, cp.log(w)))   # elementwise multiplication
        + c * cp.norm(w - w_t_drift, 1)
    )
    constraints = [
        cp.sum(w) == 1,
        w >= 1e-8,                         # ensures log is defined
        cp.norm(w - w_t_drift, 1) <= tau
    ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except:
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except:
            pass

    if w.value is None:
        return w_t_drift, 0.0
    optimized_w = np.array(w.value).squeeze()
    actual_turnover = np.linalg.norm(optimized_w - w_t_drift, ord=1)
    cost_val = c * actual_turnover
    return optimized_w, cost_val

def EXP_GRAD(curr_pri_rel, w_t, N, eta=0.05):
    if w_t is None:
        w_t = np.ones(N) / N
    w_t = w_t.flatten()
    x_t = curr_pri_rel.flatten()
    w_t = np.clip(w_t, 1e-8, None)
    x_t = np.clip(x_t, 1e-8, None)
    numerator = w_t * x_t
    w_t_drift = numerator / max(np.sum(numerator), 1e-8)
    grad = -x_t / max(w_t.T @ x_t, 1e-8)
    w_next = cp.Variable(N)
    grad_term = eta * (grad @ w_next)
    kl_div = -cp.sum(cp.entr(w_next)) - cp.sum(cp.multiply(w_next, np.log(w_t)))
    turnover = cp.norm(w_next - w_t_drift, 1)
    penalty = eta * c * turnover
    objective = cp.Minimize(grad_term + kl_div + penalty)
    constraints = [w_next >= 0, cp.sum(w_next) == 1, turnover <= tau]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except:
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except:
            pass
    if w_next.value is None:
        return w_t_drift, 0.0
    optimized_w = np.array(w_next.value).squeeze()
    actual_turnover = np.linalg.norm(optimized_w - w_t_drift, ord=1)
    cost_val = c * actual_turnover
    return optimized_w, cost_val

def ADA_GRAD(curr_pri_rel, w_t, N, H_t_diag=None, eta=0.05, delta=1e-8):
    if w_t is None:
        w_t = np.ones(N) / N
    w_t = w_t.flatten()
    x_t = curr_pri_rel.flatten()
    w_t = np.clip(w_t, 1e-8, None)
    x_t = np.clip(x_t, 1e-8, None)
    numerator = x_t * w_t
    w_t_drift = numerator / max(np.sum(numerator), 1e-8)
    grad_t = -x_t / max(w_t @ x_t, 1e-8)
    if H_t_diag is None:
        H_t_diag_updated = np.sqrt(grad_t**2) + delta
    else:
        historical_sum_sqt = np.clip((H_t_diag - delta) ** 2, 0, None)
        H_t_diag_updated = np.sqrt(historical_sum_sqt + grad_t**2) + delta
    w_next = cp.Variable(N)
    quad_term = 0.5 * cp.sum(cp.multiply(H_t_diag_updated, cp.square(w_next - w_t)))
    grad_term = eta * (grad_t @ w_next)
    turnover = cp.norm(w_next - w_t_drift, 1)
    turnover_term = turnover * c * eta
    objective = cp.Minimize(quad_term + grad_term + turnover_term)
    constraints = [w_next >= 0, cp.sum(w_next) == 1, turnover <= tau]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except:
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except:
            pass
    if w_next.value is None:
        return w_t_drift, 0.0, H_t_diag_updated
    optimized_w = np.array(w_next.value).squeeze()
    actual_turnover = np.linalg.norm(optimized_w - w_t_drift, ord=1)
    cost_val = c * actual_turnover
    return optimized_w, cost_val, H_t_diag_updated

def PAMR(curr_pri_rel, w_t, N, epsilon=0.5, eta=0.05):
    if w_t is None:
        w_t = np.ones(N) / N
    w_t = w_t.flatten()
    x_t = curr_pri_rel.flatten()
    w_t = np.clip(w_t, 1e-8, None)
    x_t = np.clip(x_t, 1e-8, None)
    numerator = w_t * x_t
    w_t_drift = numerator / max(np.sum(numerator), 1e-8)
    w_next = cp.Variable(N)
    passive_term = 0.5 * cp.sum_squares(w_next - w_t)
    portfolio_return = w_next @ x_t
    hinge_loss = cp.max(cp.hstack([0, portfolio_return - epsilon]))
    loss_term = eta * hinge_loss
    turnover = cp.norm(w_next - w_t_drift, 1)
    turnover_term = eta * c * turnover
    objective = cp.Minimize(passive_term + loss_term + turnover_term)
    constraints = [w_next >= 0, cp.sum(w_next) == 1, turnover <= tau]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except:
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except:
            pass
    if w_next.value is None:
        return w_t_drift, 0.0
    optimized_w = np.array(w_next.value).squeeze()
    actual_turnover = np.linalg.norm(optimized_w - w_t_drift, ord=1)
    cost_val = c * actual_turnover
    return optimized_w, cost_val

optimizer = {
    "MVO": (MVO, True),
    "MIN_VAR": (MIN_VAR, True),
    "RISK_PAR": (RISK_PAR, True),
    "EXP_GRAD": (EXP_GRAD, False),
    "ADA_GRAD": (ADA_GRAD, False),
    "PAMR": (PAMR, False)
}

# ---------- Helper: load risk-free rate ----------
def load_risk_free_rate(start_date, end_date):
    """Download ^IRX (13-week T-bill) from Yahoo Finance and return daily risk-free rate (simple)."""
    try:
        rf = yf.download("^IRX", start=start_date, end=end_date, progress=False)
        if rf.empty:
            raise ValueError("No data")
        # ^IRX is annual percentage yield (e.g., 2.5 means 2.5%)
        # Use 'Close' price; if not available, fallback to 'Adj Close'
        if 'Close' in rf.columns:
            annual_yield = rf['Close']
        elif 'Adj Close' in rf.columns:
            annual_yield = rf['Adj Close']
        else:
            raise KeyError("Neither 'Close' nor 'Adj Close' found in downloaded data")
        # Convert to daily simple return: (1 + yield/100)^(1/252) - 1
        rf_daily = (1 + annual_yield / 100) ** (1/252) - 1
        # Reindex to business days and forward fill (no deprecated 'method')
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        rf_daily = rf_daily.reindex(date_range).ffill()
        return rf_daily
    except Exception as e:
        print(f"Warning: Could not load ^IRX data. Using zero risk-free rate. Error: {e}")
        return pd.Series(0.0, index=pd.date_range(start=start_date, end=end_date, freq='B'))
# ---------- Lightweight backtest (no I/O, returns results) ----------
def run_backtest(tickers, window, log_returns, start_date=None, end_date=None, verbose=False):
    """
    Run backtest for all strategies on a given log return series.
    Returns dict {strategy: (weights_df, costs_df)}.
    If verbose=True, prints start/finish for each strategy.
    """
    if start_date:
        log_returns = log_returns.loc[start_date:]
    if end_date:
        log_returns = log_returns.loc[:end_date]

    N = len(tickers)
    results = {}
    for name, (opt_meth, is_batch) in optimizer.items():
        if verbose:
            print(f"    → Running {name}...", end=' ', flush=True)
            start_time = time.time()
        weights_list = []
        costs_list = []
        dates = []
        w_prev = None
        H_t_diag = None
        if not is_batch:   # online
            for i in range(window, len(log_returns)):
                curr_log_ret = log_returns.iloc[i-1].values
                curr_price_rel = np.exp(curr_log_ret)
                if name == "ADA_GRAD":
                    w, cost, H_t_diag = opt_meth(curr_price_rel, w_prev, N, H_t_diag)
                else:
                    w, cost = opt_meth(curr_price_rel, w_prev, N)
                w_prev = w
                weights_list.append(w)
                costs_list.append(cost)
                dates.append(log_returns.index[i])
        else:   # batch
            for i in range(window, len(log_returns)):
                data_window = log_returns.iloc[i-window:i]
                w, cost = opt_meth(data_window, w_prev, N)
                w_prev = w
                weights_list.append(w)
                costs_list.append(cost)
                dates.append(log_returns.index[i])
        weights_df = pd.DataFrame(weights_list, index=dates, columns=tickers)
        costs_df = pd.DataFrame(costs_list, index=dates, columns=["Transaction cost"])
        results[name] = (weights_df, costs_df)
        if verbose:
            elapsed = time.time() - start_time
            print(f"done ({elapsed:.2f}s)", flush=True)
    return results

# ---------- Cost inspection (grid search) ----------
def cost_inspection_backtest(tickers, window=252, start="", end=""):
    from mpl_toolkits.mplot3d import Axes3D
    global c, tau

    # Create output directories
    os.makedirs("image/cost_inspection", exist_ok=True)
    os.makedirs("image/robust_weights", exist_ok=True)
    os.makedirs("joblib_output/robust_weights", exist_ok=True)

    cost_levels = [0, 0.0005, 0.0010, 0.0025, 0.0050]
    tau_levels = [0.10, 0.25, 0.50, 1.00, 2.00]

    # Pre-load data once
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    prices = prices[tickers]
    log_returns_full = np.log(prices / prices.shift(1)).dropna()
    if start and end:
        log_returns = log_returns_full.loc[start:end]
    elif start:
        log_returns = log_returns_full.loc[start:]
    elif end:
        log_returns = log_returns_full.loc[:end]
    else:
        log_returns = log_returns_full

    # Load risk-free rate for the same period
    rf_series = load_risk_free_rate(log_returns.index[0], log_returns.index[-1])
    # Align to business days of log_returns (forward fill, then fill any remaining NaN with 0)
    rf_series = rf_series.reindex(log_returns.index).ffill().fillna(0)
    # Store metrics and weights across grid
    metrics_by_strategy = {}
    all_weights = {name: [] for name in optimizer.keys()}

    total_runs = len(cost_levels) * len(tau_levels)
    run_count = 0
    print("="*80)
    print("COST INSPECTION BACKTEST (lightweight mode)")
    print(f"Evaluation window: {log_returns.index[0]} → {log_returns.index[-1]}")
    print(f"Total combinations: {total_runs}")
    print("="*80)

    overall_start = time.time()

    for cost_idx, cost_value in enumerate(cost_levels):
        for tau_idx, tau_value in enumerate(tau_levels):
            run_count += 1
            c = cost_value
            tau = tau_value
            combo_start = time.time()
            print(f"\n[{run_count}/{total_runs}] c = {cost_value:.4f}, τ = {tau_value:.2f}")
            print("  Running backtest for all strategies...")
            results = run_backtest(tickers, window, log_returns, verbose=True)
            combo_elapsed = time.time() - combo_start
            print(f"  ✓ Completed in {combo_elapsed:.2f}s")

            for strategy, (w_df, cost_df) in results.items():
                if strategy not in metrics_by_strategy:
                    metrics_by_strategy[strategy] = {
                        'CAGR': np.zeros((len(cost_levels), len(tau_levels))),
                        'Vol': np.zeros((len(cost_levels), len(tau_levels))),
                        'Neg_Vol': np.zeros((len(cost_levels), len(tau_levels))),
                        'Sharpe': np.zeros((len(cost_levels), len(tau_levels))),
                        'MDD': np.zeros((len(cost_levels), len(tau_levels))),
                        'Calmar': np.zeros((len(cost_levels), len(tau_levels))),
                        'Turnover': np.zeros((len(cost_levels), len(tau_levels)))  # annualized turnover
                    }
                all_weights[strategy].append(w_df)

                # Align dates
                common_idx = w_df.index.intersection(log_returns.index)
                if len(common_idx) == 0:
                    continue
                ret = np.exp(log_returns.loc[common_idx].values) - 1
                rf = rf_series.loc[common_idx].values
                w_aligned = w_df.loc[common_idx].values
                tc_aligned = cost_df.loc[common_idx].values.flatten()
                net_ret = np.sum(w_aligned * ret, axis=1) - tc_aligned
                excess_ret = net_ret - rf
                T = len(common_idx)

                # Volatility of excess returns
                vol = np.std(excess_ret) * np.sqrt(252)
                metrics_by_strategy[strategy]['Vol'][cost_idx, tau_idx] = vol
                neg_ret = net_ret[net_ret < 0]
                neg_vol = np.std(neg_ret) * np.sqrt(252) if len(neg_ret)>0 else 0.0
                metrics_by_strategy[strategy]['Neg_Vol'][cost_idx, tau_idx] = neg_vol

                # CAGR from net returns (not excess, as standard)
                wealth = np.cumprod(1 + net_ret)
                final = wealth[-1]
                cagr = final ** (252/T) - 1
                metrics_by_strategy[strategy]['CAGR'][cost_idx, tau_idx] = cagr

                # MDD (negative)
                running_peak = np.maximum.accumulate(wealth)
                drawdown = wealth / (running_peak + 1e-8) - 1
                mdd = drawdown.min()
                metrics_by_strategy[strategy]['MDD'][cost_idx, tau_idx] = mdd

                # Sharpe (using excess returns)
                sharpe = (np.mean(excess_ret) / (np.std(excess_ret)+1e-8)) * np.sqrt(252) if np.std(excess_ret)>0 else 0
                metrics_by_strategy[strategy]['Sharpe'][cost_idx, tau_idx] = sharpe

                # Calmar = CAGR / |MDD|
                calmar = cagr / (abs(mdd)+1e-8)
                metrics_by_strategy[strategy]['Calmar'][cost_idx, tau_idx] = calmar

                # Annualized turnover (average daily L1 norm * 252)
                daily_turn = np.sum(np.abs(np.diff(w_aligned, axis=0)), axis=1)
                avg_turn = np.mean(daily_turn) if len(daily_turn)>0 else 0.0
                annual_turn = avg_turn * 252 / len(common_idx) * 100
                metrics_by_strategy[strategy]['Turnover'][cost_idx, tau_idx] = annual_turn

    total_elapsed = time.time() - overall_start
    print(f"\n✅ Entire grid search completed in {total_elapsed:.2f} seconds.")

    # ---------- Robust weights (median across grid) ----------
    print("\nComputing robust weights (median across grid)...")
    robust_weights = {}
    for strategy, w_list in all_weights.items():
        if not w_list:
            continue
        stacked = np.stack([df.values for df in w_list], axis=-1)
        median_weights = np.median(stacked, axis=-1)
        robust_df = pd.DataFrame(median_weights, index=w_list[0].index, columns=tickers)
        robust_weights[strategy] = robust_df
        joblib.dump(robust_df, f"joblib_output/robust_weights/{strategy}_robust.pkl")
        print(f"  Saved {strategy} robust weights.")

    # ---------- Evaluate robust weights on the same window ----------
    print("\nEvaluating robust weights (same window)...")
    robust_metrics = {}
    for strategy, w_df in robust_weights.items():
        common_idx = w_df.index.intersection(log_returns.index)
        if len(common_idx) == 0:
            continue
        w = w_df.loc[common_idx].values
        ret = np.exp(log_returns.loc[common_idx].values) - 1
        rf = rf_series.loc[common_idx].values
        port_ret = np.sum(w * ret, axis=1)
        excess_ret = port_ret - rf
        # Annualized turnover
        daily_turn = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
        avg_turn = np.mean(daily_turn) if len(daily_turn)>0 else 0.0
        annual_turn = avg_turn * 252 / len(common_idx)
        # Sharpe
        sharpe = (np.mean(excess_ret) / (np.std(excess_ret)+1e-8)) * np.sqrt(252) if np.std(excess_ret)>0 else 0
        wealth = np.cumprod(1+port_ret)
        final = wealth[-1]
        T = len(common_idx)
        cagr = final ** (252/T) - 1
        running_peak = np.maximum.accumulate(wealth)
        drawdown = wealth / (running_peak+1e-8) - 1
        mdd = drawdown.min()
        calmar = cagr / (abs(mdd)+1e-8)
        robust_metrics[strategy] = {
            'sharpe': sharpe,
            'calmar': calmar,
            'mdd': mdd,
            'cagr': cagr,
            'turnover': annual_turn
        }

    # Bar plots for robust weights
    if robust_metrics:
        strategies = list(robust_metrics.keys())
        colors = plt.cm.tab10(np.linspace(0,1,len(strategies)))
        metrics_to_bar = [('sharpe','Sharpe Ratio'), ('calmar','Calmar Ratio'),
                          ('mdd','Max Drawdown'), ('cagr','CAGR'), ('turnover','Annualized Turnover')]
        for mkey, label in metrics_to_bar:
            fig, ax = plt.subplots(figsize=(10,6), dpi=300)
            vals = [robust_metrics[s][mkey] for s in strategies]
            ax.bar(strategies, vals, color=colors, edgecolor='black')
            ax.set_title(f"Robust Weights – {label} ({log_returns.index[0].date()} to {log_returns.index[-1].date()})")
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            fig.savefig(f"image/robust_weights/{label.replace(' ','_')}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        print("Saved robust weights bar plots in image/robust_weights/")

    # ---------- 3D plots (original grid metrics) ----------
    strategies_list = list(metrics_by_strategy.keys())
    colors = plt.cm.tab10(np.linspace(0,1,len(strategies_list)))
    color_map = {s: colors[i] for i,s in enumerate(strategies_list)}
    metrics_to_plot = [('Sharpe','Sharpe Ratio'), ('Calmar','Calmar Ratio'),
                       ('MDD','Maximum Drawdown'), ('CAGR','CAGR'), ('Turnover','Annualized Turnover')]

    for metric_key, metric_label in metrics_to_plot:
        fig = plt.figure(figsize=(16,10), dpi=300)
        for idx, strategy in enumerate(strategies_list):
            ax = fig.add_subplot(2,3,idx+1, projection='3d')
            Z = metrics_by_strategy[strategy][metric_key]
            bar_width = 0.35
            for i, cv in enumerate(cost_levels):
                for j, tv in enumerate(tau_levels):
                    x_pos = i + j*bar_width
                    y_pos = j
                    z_val = Z[i,j]
                    ax.bar3d(x_pos, y_pos, 0, dx=bar_width*0.9, dy=bar_width*0.9, dz=z_val,
                             color=color_map[strategy], alpha=0.85, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Transaction Cost', fontsize=11)
            ax.set_ylabel('Tau Max', fontsize=11)
            ax.set_zlabel(metric_label, fontsize=11)
            ax.set_title(strategy)
            ax.set_xticks(range(len(cost_levels)))
            ax.set_xticklabels([f'{c:.4f}' for c in cost_levels], rotation=45)
            ax.set_yticks(range(len(tau_levels)))
            ax.set_yticklabels([f'{t:.2f}' for t in tau_levels])
            ax.view_init(elev=30, azim=45)
        fig.suptitle(f'{metric_label} vs Cost & Tau', fontsize=14)
        fig.tight_layout()
        fig.savefig(f"image/cost_inspection/3D_{metric_key}_vs_Cost_and_Tau.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved 3D plot for {metric_key}")

    # ---------- Differential heatmaps (Online vs Batch) ----------
    online = ['EXP_GRAD','ADA_GRAD','PAMR']
    batch = ['MVO','MIN_VAR','RISK_PAR']
    online_exist = [s for s in online if s in metrics_by_strategy]
    batch_exist = [s for s in batch if s in metrics_by_strategy]
    if online_exist and batch_exist:
        diff_metrics = [('Sharpe','Sharpe Ratio'), ('Calmar','Calmar Ratio'),
                        ('MDD','Max Drawdown'), ('CAGR','CAGR'), ('Turnover','Annualized Turnover')]
        for metric_key, metric_label in diff_metrics:
            diff = np.zeros((len(cost_levels), len(tau_levels)))
            for s in online_exist:
                diff += metrics_by_strategy[s][metric_key]
            diff = diff / len(online_exist)
            for s in batch_exist:
                diff -= metrics_by_strategy[s][metric_key] / len(batch_exist)
            fig, ax = plt.subplots(figsize=(12,8), dpi=300)
            vmax = np.abs(diff).max()
            # For MDD (negative values) and Turnover (positive), we want green=online better.
            # For MDD, larger (less negative) is better -> positive diff means online better -> green.
            # For Turnover, lower is better -> negative diff means online better -> we want that green.
            # But we keep the colormap as RdYlGn where positive diff = green, negative = red.
            # For Turnover, we should multiply by -1 so that lower turnover gives positive diff.
            if metric_key == 'Turnover':
                diff = -diff   # so that lower turnover (better) appears green
            im = ax.imshow(diff.T, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax, origin='lower')
            ax.set_xticks(range(len(tau_levels)))
            ax.set_xticklabels([f'{t:.2f}' for t in tau_levels])
            ax.set_yticks(range(len(cost_levels)))
            ax.set_yticklabels([f'{c:.4f}' for c in cost_levels])
            ax.set_xlabel('Tau Max')
            ax.set_ylabel('Transaction Cost')
            ax.set_title(f'{metric_label} Differential (Online - Batch)\nGreen = Online better, Red = Batch better')
            plt.colorbar(im, ax=ax, label='Differential')
            for i in range(len(cost_levels)):
                for j in range(len(tau_levels)):
                    val = diff[i,j]
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            color='white' if abs(val)>vmax*0.6 else 'black')
            fig.tight_layout()
            fig.savefig(f"image/cost_inspection/Differential_{metric_key}_Online_vs_Batch.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved differential heatmap for {metric_key}")

    # Save average metrics summary
    summary = []
    for s in strategies_list:
        row = {'Strategy': s}
        for mk,_ in metrics_to_plot:
            row[mk] = metrics_by_strategy[s][mk].mean()
        summary.append(row)
    pd.DataFrame(summary).to_csv("cost_inspection_backtest_result.csv", index=False)
    print("\n✓ Cost inspection completed. Results saved.")

# ---------- Evaluate robust weights on any sub-period ----------
def evaluate_robust_weights(start, end):
    """
    Load precomputed robust weights and evaluate them on a specific date window.
    Generates:
      - Bar plots for Sharpe, Calmar, MDD, CAGR, and annualized turnover.
      - Pie charts of average allocation for each strategy.
      - CSV file with all metrics.
    """
    # Load log returns and risk-free rate
    log_ret = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)
    log_ret = log_ret.loc[start:end]

    rf_series = load_risk_free_rate(start, end)
    rf_series = rf_series.reindex(log_ret.index).ffill().fillna(0)

    robust_dir = "joblib_output/robust_weights"
    if not os.path.exists(robust_dir):
        print("Error: robust_weights directory not found. Run cost_inspection_backtest first.")
        return

    files = [f for f in os.listdir(robust_dir) if f.endswith("_robust.pkl")]
    if not files:
        print("No robust weight files found.")
        return

    metrics = {}
    avg_weights = {}   # store average weights for pie charts
    tickers = None     # will be set from the first DataFrame

    for fname in files:
        strategy = fname.replace("_robust.pkl", "")
        w_df = joblib.load(os.path.join(robust_dir, fname))
        common = w_df.index.intersection(log_ret.index)
        if len(common) == 0:
            print(f"Warning: no overlap for {strategy}")
            continue

        # Capture tickers from the first valid DataFrame
        if tickers is None:
            tickers = w_df.columns.tolist()

        w = w_df.loc[common].values
        ret = np.exp(log_ret.loc[common].values) - 1
        rf = rf_series.loc[common].values

        # Portfolio returns and excess returns
        port_ret = np.sum(w * ret, axis=1)
        excess_ret = port_ret - rf

        # Annualized turnover
        daily_turn = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
        avg_turn = np.mean(daily_turn) if len(daily_turn) > 0 else 0.0
        annual_turn = avg_turn * 252 / len(common)

        # Sharpe ratio
        sharpe = (np.mean(excess_ret) / (np.std(excess_ret) + 1e-8)) * np.sqrt(252) if np.std(excess_ret) > 0 else 0

        # Wealth, CAGR, MDD, Calmar
        wealth = np.cumprod(1 + port_ret)
        final = wealth[-1]
        T = len(common)
        cagr = final ** (252 / T) - 1
        running_peak = np.maximum.accumulate(wealth)
        drawdown = wealth / (running_peak + 1e-8) - 1
        mdd = drawdown.min()
        calmar = cagr / (abs(mdd) + 1e-8)

        metrics[strategy] = {
            'sharpe': sharpe,
            'calmar': calmar,
            'mdd': mdd,
            'cagr': cagr,
            'turnover': annual_turn
        }
        avg_weights[strategy] = np.mean(w, axis=0)  # average allocation over the period

    if not metrics:
        print("No strategies evaluated.")
        return

    # Ensure we have tickers
    if tickers is None:
        print("Error: no valid weight data found.")
        return

    # -------------------- 1. Save CSV --------------------
    os.makedirs("robust_weights_eval", exist_ok=True)
    df = pd.DataFrame(metrics).T   # rows = strategies, columns = metrics
    df.to_csv("robust_weights_eval/performance.csv")
    print("Saved performance metrics to robust_weights_eval/performance.csv")

    # -------------------- 2. Bar plots --------------------
    strategies = list(metrics.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
    for metric, ylabel, fname in [('sharpe','Sharpe Ratio','Sharpe_Ratio'),
                                  ('calmar','Calmar Ratio','Calmar_Ratio'),
                                  ('mdd','Max Drawdown','Max_Drawdown'),
                                  ('cagr','CAGR','CAGR'),
                                  ('turnover','Annualized Turnover','Turnover')]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        vals = [metrics[s][metric] for s in strategies]
        ax.bar(strategies, vals, color=colors, edgecolor='black')
        ax.set_title(f"Robust Weights – {ylabel} ({start} to {end})")
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(f"image/robust_weights_eval/{fname}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # -------------------- 3. Pie charts for average weights --------------------
    n_tickers = len(tickers)
    pie_colors = plt.cm.tab10(np.linspace(0, 1, n_tickers))

    n_strategies = len(strategies)
    n_cols = min(2, n_strategies)
    n_rows = (n_strategies + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), dpi=300)
    if n_strategies == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        avg_w = avg_weights[strategy]
        # Filter out negligible weights for cleaner labels
        threshold = 0.005
        labels = [t if w > threshold else '' for t, w in zip(tickers, avg_w)]
        # Use the fixed color map for each slice
        wedges, texts, autotexts = ax.pie(
            avg_w,
            labels=labels,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        ax.set_title(f"{strategy} - Avg Allocation", fontsize=12, fontweight='bold')

    # Remove empty subplots
    for idx in range(n_strategies, len(axes)):
        fig.delaxes(axes[idx])

    # Add a single common legend for the pie slices (ETFs)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=pie_colors[i], edgecolor='black', label=ticker)
                       for i, ticker in enumerate(tickers)]
    fig.legend(handles=legend_elements,
               title='ETFs',
               loc='upper center',
               bbox_to_anchor=(0.5, -0.01),
               ncol=n_tickers,
               frameon=True,
               fontsize=10)

    fig.suptitle(f"Average Portfolio Composition – Robust Weights ({start} to {end})", fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig("image/robust_weights_eval/Pie_Charts.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Evaluation complete. Plots saved in image/robust_weights_eval/")