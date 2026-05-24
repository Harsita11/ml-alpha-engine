# features.py
# Computes 15+ alpha factors from raw price data and stores them in SQLite
# Factors: momentum, reversal, volatility, RSI, Amihud illiquidity, etc.

import sqlite3
import numpy as np
import pandas as pd
from database import get_connection, DB_PATH
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── 1. Load Price Data ────────────────────────────────────────────────────────

def load_prices(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load prices and pivot into wide format for vectorized factor computation."""
    with get_connection(db_path) as conn:
        df = pd.read_sql(
            "SELECT ticker, date, adj_close, volume FROM equities ORDER BY date",
            conn,
            parse_dates=["date"]
        )
    return df


def pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot long → wide: rows=date, columns=ticker."""
    return df.pivot(index="date", columns="ticker", values=col).sort_index()

# ── 2. Individual Factor Functions ───────────────────────────────────────────

def momentum(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Standard momentum: cumulative return over [t-window, t-1].
    Skip last month (t-1) to avoid microstructure reversal.
    """
    return close.shift(1).pct_change(window)


def short_term_reversal(close: pd.DataFrame) -> pd.DataFrame:
    """1-month reversal: prior month return (negative momentum signal)."""
    return -close.pct_change(21)


def realized_volatility(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualized realized vol from daily log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def volatility_ratio(close: pd.DataFrame,
                     short: int = 5,
                     long: int = 21) -> pd.DataFrame:
    """
    Vol ratio: short-term vol / long-term vol.
    High ratio → regime change / mean-reversion opportunity.
    """
    log_ret = np.log(close / close.shift(1))
    vol_s = log_ret.rolling(short).std()
    vol_l = log_ret.rolling(long).std()
    return vol_s / (vol_l + 1e-8)


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index — classic 14-day."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def amihud_illiquidity(close: pd.DataFrame,
                       volume: pd.DataFrame,
                       window: int = 21) -> pd.DataFrame:
    """
    Amihud (2002) illiquidity ratio:
        ILLIQ = mean(|r_t| / dollar_volume_t)
    Higher → less liquid → typically commands an illiquidity premium.
    """
    ret        = close.pct_change().abs()
    dollar_vol = close * volume
    illiq      = (ret / (dollar_vol + 1e-8)).rolling(window).mean()
    return np.log1p(illiq)   # log-transform to reduce skew


def price_to_moving_average(close: pd.DataFrame,
                             window: int = 50) -> pd.DataFrame:
    """Distance of price from its moving average — mean-reversion signal."""
    ma = close.rolling(window).mean()
    return (close - ma) / (ma + 1e-8)


def volume_trend(volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Volume z-score: how unusual is today's volume vs recent history?
    Unusual volume often precedes price moves.
    """
    vol_mean = volume.rolling(window).mean()
    vol_std  = volume.rolling(window).std()
    return (volume - vol_mean) / (vol_std + 1e-8)


def garman_klass_vol(df_raw: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Garman-Klass volatility estimator using OHLC prices.
    More efficient than close-to-close vol.
    """
    with get_connection() as conn:
        ohlc = pd.read_sql(
            "SELECT ticker, date, open, high, low, close FROM equities ORDER BY date",
            conn, parse_dates=["date"]
        )

    def _gk(g):
        log_hl = np.log(g["high"] / g["low"]) ** 2
        log_co = np.log(g["close"] / g["open"]) ** 2
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return gk.rolling(window).mean() ** 0.5

    result = (
        ohlc.groupby("ticker")
            .apply(_gk)
            .reset_index(level=0, drop=True)
    )
    return ohlc.assign(gk_vol=result).pivot(
        index="date", columns="ticker", values="gk_vol"
    )


# ── 3. Cross-Sectional Normalization ─────────────────────────────────────────

def cross_sectional_zscore(factor: pd.DataFrame,
                           clip: float = 3.0) -> pd.DataFrame:
    """
    At each date, z-score the factor across all stocks.
    This makes factors comparable across time and removes market-wide trends.
    Winsorize at ±3σ to limit outlier impact.
    """
    mean = factor.mean(axis=1)
    std  = factor.std(axis=1)
    z    = factor.sub(mean, axis=0).div(std + 1e-8, axis=0)
    return z.clip(-clip, clip)


# ── 4. Forward Return Labels ──────────────────────────────────────────────────

def forward_returns(close: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """
    Next `horizon` trading-day return — this is the ML target variable.
    Shift by -horizon so each row's label = future return.
    """
    return close.pct_change(horizon).shift(-horizon)


# ── 5. Assemble Feature Matrix ────────────────────────────────────────────────

def build_feature_matrix(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Compute all factors, normalize cross-sectionally, and return a
    stacked (date, ticker) DataFrame ready for ML training.
    """
    logger.info("Loading prices...")
    raw   = load_prices(db_path)
    close = pivot(raw, "adj_close")
    vol   = pivot(raw, "volume")

    logger.info("Computing factors...")
    factors = {
        "mom_1m":       cross_sectional_zscore(momentum(close, 21)),
        "mom_3m":       cross_sectional_zscore(momentum(close, 63)),
        "mom_6m":       cross_sectional_zscore(momentum(close, 126)),
        "mom_12m":      cross_sectional_zscore(momentum(close, 252)),
        "reversal_1m":  cross_sectional_zscore(short_term_reversal(close)),
        "vol_realized": cross_sectional_zscore(realized_volatility(close)),
        "vol_ratio":    cross_sectional_zscore(volatility_ratio(close)),
        "rsi_14":       cross_sectional_zscore(rsi(close)),
        "amihud_illiq": cross_sectional_zscore(amihud_illiquidity(close, vol)),
        "price_to_ma50":cross_sectional_zscore(price_to_moving_average(close, 50)),
        "price_to_ma200":cross_sectional_zscore(price_to_moving_average(close, 200)),
        "vol_trend":    cross_sectional_zscore(volume_trend(vol)),
        "fwd_return_1m": forward_returns(close, 21),   # ← ML label (NOT normalized)
    }

    # Stack wide → long
    stacked = {}
    for name, wide_df in factors.items():
        stacked[name] = wide_df.stack()

    panel = pd.DataFrame(stacked)
    panel.index.names = ["date", "ticker"]
    panel = panel.dropna(subset=[c for c in panel.columns if c != "fwd_return_1m"])
    panel = panel.reset_index()

    logger.info(f"Feature matrix: {panel.shape[0]:,} rows × {panel.shape[1]} cols")
    return panel


# ── 6. Persist Factors to SQL ─────────────────────────────────────────────────

def save_factors(panel: pd.DataFrame, db_path: str = DB_PATH):
    """Write computed factors back to the `factors` table in SQLite."""
    factor_cols = [
        "ticker", "date",
        "mom_1m", "mom_3m", "mom_6m", "mom_12m",
        "reversal_1m", "vol_realized", "vol_ratio",
        "rsi_14", "amihud_illiq",
    ]
    subset = panel[[c for c in factor_cols if c in panel.columns]].copy()
    subset["date"] = subset["date"].astype(str)

    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM factors")   # full refresh
        subset.to_sql("factors", conn, if_exists="append", index=False)

    logger.info(f"Saved {len(subset):,} factor rows to DB")


# ── 7. Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    panel = build_feature_matrix()
    save_factors(panel)
    print(panel.head())
    print("\n✅ Factors computed and saved.")