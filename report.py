# report.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

from database import get_connection, DB_PATH
from backtest import (
    load_predictions, load_prices, load_benchmark,
    construct_portfolio, simulate_pnl,
    rolling_beta, max_drawdown_series,
)

import logging
logger = logging.getLogger(__name__)

# ── Metrics ───────────────────────────────────────────────────────────────────

def sharpe_ratio(returns, periods_per_year=12):
    mean = returns.mean() * periods_per_year
    std  = returns.std()  * np.sqrt(periods_per_year)
    return mean / (std + 1e-8)

def sortino_ratio(returns, periods_per_year=12):
    mean         = returns.mean() * periods_per_year
    downside_std = returns[returns < 0].std() * np.sqrt(periods_per_year)
    return mean / (downside_std + 1e-8)

def calmar_ratio(returns, nav, periods_per_year=12):
    ann_return = returns.mean() * periods_per_year
    mdd        = max_drawdown(nav)
    return ann_return / (abs(mdd) + 1e-8)

def max_drawdown(nav):
    rolling_max = nav.cummax()
    drawdown    = (nav - rolling_max) / (rolling_max + 1e-8)
    return drawdown.min()

def annualized_return(returns, periods_per_year=12):
    return returns.mean() * periods_per_year

def annualized_volatility(returns, periods_per_year=12):
    return returns.std() * np.sqrt(periods_per_year)

def hit_rate(returns):
    return (returns > 0).mean()

def avg_win_loss_ratio(returns):
    wins   = returns[returns > 0].mean()
    losses = returns[returns < 0].abs().mean()
    return wins / (losses + 1e-8)

# ── IC Metrics ────────────────────────────────────────────────────────────────

def compute_ic_series(predictions):
    def _ic(g):
        if len(g) < 5:
            return np.nan
        ic, _ = spearmanr(g["predicted_rank"], g["actual_return"])
        return ic
    return predictions.groupby("date").apply(_ic).dropna()

def compute_icir(ic_series):
    return ic_series.mean() / (ic_series.std() + 1e-8)

def annualized_turnover(pnl, periods_per_year=12):
    return pnl["turnover"].mean() * periods_per_year

# ── Full Metrics ──────────────────────────────────────────────────────────────

def compute_all_metrics(pnl, predictions, benchmark=None):
    net_ret   = pnl["net_return"]
    nav       = pnl["nav"]
    ic_series = compute_ic_series(predictions)

    metrics = {
        "Total Return":       f"{nav.iloc[-1] - 1:.2%}",
        "Ann. Return (net)":  f"{annualized_return(net_ret):.2%}",
        "Ann. Volatility":    f"{annualized_volatility(net_ret):.2%}",
        "Sharpe Ratio":       f"{sharpe_ratio(net_ret):.2f}",
        "Sortino Ratio":      f"{sortino_ratio(net_ret):.2f}",
        "Calmar Ratio":       f"{calmar_ratio(net_ret, nav):.2f}",
        "Max Drawdown":       f"{max_drawdown(nav):.2%}",
        "Hit Rate":           f"{hit_rate(net_ret):.2%}",
        "Avg Win/Loss Ratio": f"{avg_win_loss_ratio(net_ret):.2f}",
        "Mean IC":            f"{ic_series.mean():.4f}",
        "IC Std":             f"{ic_series.std():.4f}",
        "ICIR":               f"{compute_icir(ic_series):.2f}",
        "Ann. Turnover":      f"{annualized_turnover(pnl):.2%}",
    }

    if benchmark is not None:
        bmark = benchmark.pct_change().dropna().reindex(net_ret.index).fillna(0)
        beta  = rolling_beta(net_ret, bmark).mean()
        alpha = annualized_return(net_ret) - beta * annualized_return(bmark)
        metrics["Beta (vs SPY)"] = f"{beta:.2f}"
        metrics["Alpha (ann.)"]  = f"{alpha:.2%}"

    return metrics

# ── Print Tearsheet ───────────────────────────────────────────────────────────

def print_tearsheet(metrics, strategy_name="ML Alpha Engine"):
    width = 45
    print("\n" + "═" * width)
    print(f"  {strategy_name} — Performance Tearsheet")
    print("═" * width)

    sections = {
        "RETURNS":        ["Total Return", "Ann. Return (net)", "Ann. Volatility"],
        "RISK-ADJUSTED":  ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
        "DRAWDOWN":       ["Max Drawdown"],
        "WIN/LOSS":       ["Hit Rate", "Avg Win/Loss Ratio"],
        "SIGNAL QUALITY": ["Mean IC", "IC Std", "ICIR"],
        "EXECUTION":      ["Ann. Turnover"],
        "VS BENCHMARK":   ["Beta (vs SPY)", "Alpha (ann.)"],
    }

    for section, keys in sections.items():
        available = [k for k in keys if k in metrics]
        if not available:
            continue
        print(f"\n  {section}")
        print("  " + "─" * (width - 2))
        for k in available:
            print(f"  {k:<26} {metrics[k]:>10}")

    print("\n" + "═" * width + "\n")

# ── Plot Tearsheet ────────────────────────────────────────────────────────────

def plot_tearsheet(pnl, predictions, benchmark=None, save_path="tearsheet.png"):
    ic_series = compute_ic_series(predictions)
    dd_series = max_drawdown_series(pnl["nav"])

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f0f0f")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    DARK  = "#0f0f0f"
    GREEN = "#00ff88"
    RED   = "#ff4466"
    GRAY  = "#888888"
    WHITE = "#e0e0e0"

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(DARK)
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    ax1.plot(pnl.index, pnl["nav"], color=GREEN, lw=1.5, label="Strategy (net)")
    if benchmark is not None:
        bm = benchmark.reindex(pnl.index, method="ffill").bfill()
        bm = bm / bm.iloc[0]
        ax1.plot(pnl.index, bm, color=GRAY, lw=1, linestyle="--", label="SPY")
    ax1.axhline(1.0, color="#333333", lw=0.8, linestyle=":")
    ax1.set_title("Cumulative NAV", fontsize=11, fontweight="bold")
    ax1.set_ylabel("NAV", fontsize=9)
    ax1.legend(fontsize=8, facecolor=DARK, labelcolor=WHITE)
    ax1.fill_between(pnl.index, 1, pnl["nav"],
                     where=pnl["nav"] >= 1, alpha=0.15, color=GREEN)
    ax1.fill_between(pnl.index, 1, pnl["nav"],
                     where=pnl["nav"] < 1,  alpha=0.15, color=RED)

    ax2.fill_between(dd_series.index, dd_series.values, 0, color=RED, alpha=0.6)
    ax2.plot(dd_series.index, dd_series.values, color=RED, lw=0.8)
    ax2.set_title("Drawdown", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Drawdown %", fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    rolling_ic = ic_series.rolling(6).mean()
    colors = [GREEN if v >= 0 else RED for v in rolling_ic.values]
    ax3.bar(rolling_ic.index, rolling_ic.values, color=colors, alpha=0.8, width=20)
    ax3.axhline(0,    color=GRAY,  lw=0.8, linestyle="--")
    ax3.axhline(0.05, color=GREEN, lw=0.8, linestyle=":", alpha=0.5)
    ax3.set_title("Rolling 6M IC", fontsize=10, fontweight="bold")
    ax3.set_ylabel("IC", fontsize=9)

    fig.suptitle("ML Alpha Engine — Strategy Tearsheet",
                 color=WHITE, fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"Tearsheet saved to {save_path}")
    plt.show()

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictions = load_predictions()
    prices      = load_prices()
    benchmark   = load_benchmark()

    weights = construct_portfolio(predictions)
    pnl     = simulate_pnl(weights, prices, cost_bps=10.0)

    metrics = compute_all_metrics(pnl, predictions, benchmark)
    print_tearsheet(metrics)
    plot_tearsheet(pnl, predictions, benchmark)