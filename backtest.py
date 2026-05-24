# backtest.py
import numpy as np
import pandas as pd
from database import get_connection, DB_PATH
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_predictions(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        return pd.read_sql(
            "SELECT ticker, date, predicted_rank, actual_return FROM predictions ORDER BY date",
            conn, parse_dates=["date"]
        )

def load_prices(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        return pd.read_sql(
            "SELECT ticker, date, adj_close FROM equities ORDER BY date",
            conn, parse_dates=["date"]
        )

def load_benchmark(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        df = pd.read_sql(
            "SELECT date, adj_close FROM equities WHERE ticker='SPY' ORDER BY date",
            conn, parse_dates=["date"]
        )
    if df.empty:
        return None
    spy_monthly = df.set_index("date")["adj_close"].resample("ME").last()
    spy_nav = (1 + spy_monthly.pct_change().dropna()).cumprod()
    return spy_nav

def construct_portfolio(predictions, top_pct=0.2, bottom_pct=0.2):
    records = []
    for date, group in predictions.groupby("date"):
        group    = group.sort_values("predicted_rank", ascending=False)
        n        = len(group)
        n_top    = max(1, int(n * top_pct))
        n_bottom = max(1, int(n * bottom_pct))
        longs    = group.head(n_top)["ticker"].tolist()
        shorts   = group.tail(n_bottom)["ticker"].tolist()
        long_w   = +0.5 / n_top
        short_w  = -0.5 / n_bottom
        for t in longs:
            records.append({"date": date, "ticker": t, "weight": long_w})
        for t in shorts:
            records.append({"date": date, "ticker": t, "weight": short_w})
    weights = pd.DataFrame(records)
    logger.info(f"Portfolio constructed: {weights['date'].nunique()} rebalance dates")
    return weights

def compute_turnover(weights):
    wide = (
        weights.pivot(index="date", columns="ticker", values="weight")
               .fillna(0)
               .sort_index()
    )
    turnover = wide.diff().abs().sum(axis=1) * 0.5
    turnover.iloc[0] = wide.iloc[0].abs().sum() * 0.5
    return turnover

def apply_transaction_costs(gross_returns, turnover, cost_bps=10.0):
    cost = turnover * (cost_bps / 10_000)
    return gross_returns - cost

def simulate_pnl(weights, prices, cost_bps=10.0):
    # ── resample prices to monthly ──
    price_wide = prices.pivot(
        index="date", columns="ticker", values="adj_close"
    ).sort_index()
    price_monthly = price_wide.resample("ME").last()
    actual_ret    = price_monthly.pct_change(1).shift(-1)

    # ── resample weights to monthly ──
    weight_wide = weights.pivot(
        index="date", columns="ticker", values="weight"
    ).fillna(0)
    weight_wide.index = pd.to_datetime(weight_wide.index)
    weight_wide = weight_wide.resample("ME").last().fillna(0)

    # ── align ──
    common_dates   = weight_wide.index.intersection(actual_ret.index)
    common_tickers = weight_wide.columns.intersection(actual_ret.columns)

    W = weight_wide.loc[common_dates, common_tickers]
    R = actual_ret.loc[common_dates, common_tickers].fillna(0)

    gross_returns = (W * R).sum(axis=1)

    # ── turnover & costs ──
    turnover_daily = compute_turnover(weights)
    turnover_daily.index = pd.to_datetime(turnover_daily.index)
    turnover = turnover_daily.resample("ME").sum().reindex(common_dates).fillna(0)

    net_returns = apply_transaction_costs(gross_returns, turnover, cost_bps)

    # ── compound NAV ──
    nav = (1 + net_returns).cumprod()

    pnl = pd.DataFrame({
        "date":         common_dates,
        "gross_return": gross_returns.values,
        "net_return":   net_returns.values,
        "turnover":     turnover.values,
        "nav":          nav.values,
    }).set_index("date")

    logger.info(f"P&L simulated: {len(pnl)} periods")
    return pnl

def rolling_beta(portfolio_ret, benchmark_ret, window=12):
    cov = portfolio_ret.rolling(window).cov(benchmark_ret)
    var = benchmark_ret.rolling(window).var()
    return cov / (var + 1e-8)

def max_drawdown_series(nav):
    rolling_max = nav.cummax()
    return (nav - rolling_max) / (rolling_max + 1e-8)

if __name__ == "__main__":
    predictions = load_predictions()
    prices      = load_prices()
    weights     = construct_portfolio(predictions)
    pnl         = simulate_pnl(weights, prices, cost_bps=10.0)

    total_return = pnl["nav"].iloc[-1] - 1
    ann_return   = (1 + total_return) ** (12 / len(pnl)) - 1
    logger.info(f"Total return : {total_return:.2%}")
    logger.info(f"Ann. return  : {ann_return:.2%}")

    pnl.to_csv("pnl.csv")
    print(pnl.tail(10))
    print("\n✅ Backtest complete. Results saved to pnl.csv")