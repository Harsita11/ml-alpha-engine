import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime

DB_PATH = "alpha_engine.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS equities (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    adj_close   REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker          TEXT NOT NULL,
    date            TEXT NOT NULL,
    market_cap      REAL,
    pe_ratio        REAL,
    pb_ratio        REAL,
    roe             REAL,
    debt_to_equity  REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS factors (
    ticker          TEXT NOT NULL,
    date            TEXT NOT NULL,
    mom_1m          REAL,
    mom_3m          REAL,
    mom_6m          REAL,
    mom_12m         REAL,
    reversal_1m     REAL,
    vol_realized    REAL,
    vol_ratio       REAL,
    rsi_14          REAL,
    amihud_illiq    REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS predictions (
    ticker          TEXT NOT NULL,
    date            TEXT NOT NULL,
    predicted_rank  REAL,
    actual_return   REAL,
    PRIMARY KEY (ticker, date)
);
"""

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM",  "BAC",  "GS",
    "XOM",  "CVX",  "JNJ",  "PFE",  "UNH",
    "WMT",  "HD",   "MCD",  "KO",   "PEP",
    "V",    "MA",   "AMD",  "INTC", "BA",
    "CAT",  "GE",   "MMM",  "HON",  "SPY",
]

def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def initialize_db(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
    print("DB initialized")

def ingest_prices(tickers=UNIVERSE, start="2018-01-01", end=None, db_path=DB_PATH):
    end = end or datetime.today().strftime("%Y-%m-%d")
    print(f"Downloading {len(tickers)} tickers...")

    with get_connection(db_path) as conn:
        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=True, progress=False,
                )
                if raw.empty:
                    print(f"  SKIP {ticker}")
                    continue

                # Step 1: flatten MultiIndex
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [col[0] for col in raw.columns]

                # Step 2: bring Date index into columns
                raw.index.name = "date"
                raw = raw.reset_index()

                # Step 3: lowercase everything
                raw.columns = [str(c).lower().strip() for c in raw.columns]

                # Step 4: add ticker and adj_close
                raw["ticker"]    = ticker
                raw["adj_close"] = raw["close"]
                raw["date"]      = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")

                rows = raw[["ticker","date","open","high",
                            "low","close","volume","adj_close"]].values.tolist()

                conn.executemany("""
                    INSERT OR REPLACE INTO equities
                    (ticker,date,open,high,low,close,volume,adj_close)
                    VALUES (?,?,?,?,?,?,?,?)
                """, rows)
                print(f"  OK {ticker}: {len(rows)} rows")

            except Exception as e:
                print(f"  ERR {ticker}: {e}")

def ingest_fundamentals(tickers=UNIVERSE, db_path=DB_PATH):
    today = datetime.today().strftime("%Y-%m-%d")
    with get_connection(db_path) as conn:
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                conn.execute("""
                    INSERT OR REPLACE INTO fundamentals
                    (ticker,date,market_cap,pe_ratio,pb_ratio,roe,debt_to_equity)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    ticker, today,
                    info.get("marketCap"),
                    info.get("trailingPE"),
                    info.get("priceToBook"),
                    info.get("returnOnEquity"),
                    info.get("debtToEquity"),
                ))
                print(f"  Fundamentals OK {ticker}")
            except Exception as e:
                print(f"  Fundamentals ERR {ticker}: {e}")

def load_prices(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        return pd.read_sql(
            "SELECT * FROM equities ORDER BY ticker, date", conn
        )

def load_factors(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        return pd.read_sql(
            "SELECT * FROM factors ORDER BY ticker, date", conn
        )

if __name__ == "__main__":
    initialize_db()
    ingest_prices()
    ingest_fundamentals()
    print("\n✅ Database ready.")