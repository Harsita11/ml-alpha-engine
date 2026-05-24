# model.py
# Trains a LightGBM Learning-to-Rank model to predict cross-sectional
# stock return rankings. Uses walk-forward (expanding window) validation
# to strictly eliminate lookahead bias.

import sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import ndcg_score
from database import get_connection, DB_PATH
from features import build_feature_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "reversal_1m", "vol_realized", "vol_ratio",
    "rsi_14", "amihud_illiq", "price_to_ma50",
    "price_to_ma200", "vol_trend",
]
TARGET_COL = "fwd_return_1m"

# ── 1. Label Encoding ─────────────────────────────────────────────────────────

def encode_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous forward returns → integer quintile ranks (0–4)
    within each cross-section (date).

    LightGBM ranker needs non-negative integer labels where higher = better.
    Quintile 4 = top 20% stocks (longs), Quintile 0 = bottom 20% (shorts).
    """
    panel = panel.copy()
    panel["label"] = (
        panel.groupby("date")[TARGET_COL]
             .transform(lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop"))
    )
    return panel.dropna(subset=["label"])


# ── 2. Walk-Forward Split ─────────────────────────────────────────────────────

def walk_forward_splits(
    panel: pd.DataFrame,
    train_years: int = 2,
    test_months: int = 6,
):
    """
    Yield (train_df, test_df) pairs using an expanding window.

    Timeline example (train_years=2, test_months=6):
      Fold 1: train [2018-01 → 2019-12], test [2020-01 → 2020-06]
      Fold 2: train [2018-01 → 2020-06], test [2020-07 → 2020-12]
      ...

    NO data leakage — test set is always strictly after train set.
    """
    dates      = sorted(panel["date"].unique())
    train_days = train_years * 252
    test_days  = test_months * 21

    start = train_days
    while start + test_days <= len(dates):
        train_dates = dates[:start]
        test_dates  = dates[start : start + test_days]

        train_df = panel[panel["date"].isin(train_dates)]
        test_df  = panel[panel["date"].isin(test_dates)]

        yield train_df, test_df
        start += test_days   # expanding: next fold trains on more data


# ── 3. LightGBM Dataset Builder ───────────────────────────────────────────────

def make_lgb_dataset(df: pd.DataFrame) -> lgb.Dataset:
    """
    Build a LightGBM Dataset with group structure.
    Groups = number of stocks per date (required for LambdaRank).
    """
    X      = df[FEATURE_COLS].values
    y      = df["label"].astype(int).values
    groups = df.groupby("date").size().values   # stocks per rebalance date

    return lgb.Dataset(X, label=y, group=groups, feature_name=FEATURE_COLS)


# ── 4. Hyperparameters ────────────────────────────────────────────────────────

LGB_PARAMS = {
    "objective":        "lambdarank",   # Learning-to-Rank
    "metric":           "ndcg",         # Normalized Discounted Cumulative Gain
    "ndcg_eval_at":     [5, 10],        # Evaluate NDCG@5 and NDCG@10
    "learning_rate":    0.03,
    "num_leaves":       31,
    "max_depth":        5,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,            # column subsampling
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "n_jobs":           -1,
}


# ── 5. Train Single Fold ──────────────────────────────────────────────────────

def train_fold(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    params:   dict = LGB_PARAMS,
    num_rounds: int = 500,
) -> lgb.Booster:
    """Train one fold with early stopping on validation NDCG@10."""
    dtrain = make_lgb_dataset(train_df)
    dval   = make_lgb_dataset(val_df)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round    = num_rounds,
        valid_sets         = [dval],
        valid_names        = ["val"],
        callbacks          = callbacks,
    )
    return model


# ── 6. Information Coefficient ────────────────────────────────────────────────

def information_coefficient(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    IC = Spearman rank correlation between predicted scores and actual returns.
    IC > 0.05 is considered a meaningful signal at most quant funds.
    """
    from scipy.stats import spearmanr
    ic, _ = spearmanr(y_pred, y_true)
    return ic


def icir(ics: list[float]) -> float:
    """IC Information Ratio = mean(IC) / std(IC). Measures signal consistency."""
    arr = np.array(ics)
    return arr.mean() / (arr.std() + 1e-8)


# ── 7. Full Walk-Forward Training Loop ───────────────────────────────────────

def train_walk_forward(
    panel:        pd.DataFrame,
    train_years:  int = 2,
    test_months:  int = 6,
) -> dict:
    """
    Run all walk-forward folds, collect metrics, save each fold's model.

    Returns a results dict with per-fold IC and aggregate ICIR.
    """
    panel         = encode_labels(panel)
    panel["date"] = pd.to_datetime(panel["date"])

    fold_ics      = []
    fold_models   = []
    all_preds     = []

    splits = list(walk_forward_splits(panel, train_years, test_months))
    logger.info(f"Running {len(splits)} walk-forward folds...")

    for fold_idx, (train_df, test_df) in enumerate(splits):
        logger.info(f"Fold {fold_idx+1}/{len(splits)} | "
                    f"train={len(train_df):,} rows | test={len(test_df):,} rows")

        # ── train ──
        model = train_fold(train_df, test_df)

        # ── predict ──
        X_test    = test_df[FEATURE_COLS].values
        scores    = model.predict(X_test)
        test_df   = test_df.copy()
        test_df["predicted_score"] = scores

        # ── per-date IC ──
        date_ics = (
            test_df.groupby("date")
                   .apply(lambda g: information_coefficient(
                       g["predicted_score"].values,
                       g[TARGET_COL].values
                   ))
        )
        mean_ic = date_ics.mean()
        fold_ics.append(mean_ic)
        fold_models.append(model)
        all_preds.append(test_df)

        logger.info(f"  Fold IC = {mean_ic:.4f}")

        # ── save model ──
        path = os.path.join(MODEL_DIR, f"model_fold_{fold_idx+1}.lgb")
        model.save_model(path)

    # ── aggregate metrics ──
    results = {
        "fold_ics":    fold_ics,
        "mean_ic":     float(np.mean(fold_ics)),
        "icir":        icir(fold_ics),
        "models":      fold_models,
        "predictions": pd.concat(all_preds, ignore_index=True),
    }

    logger.info(f"\n{'='*40}")
    logger.info(f"Mean IC  : {results['mean_ic']:.4f}")
    logger.info(f"ICIR     : {results['icir']:.4f}")
    logger.info(f"{'='*40}")

    return results


# ── 8. Feature Importance ─────────────────────────────────────────────────────

def feature_importance(models: list[lgb.Booster]) -> pd.DataFrame:
    """
    Average feature importance across all folds.
    Useful for understanding which factors drive alpha.
    """
    imp_dfs = []
    for model in models:
        imp = pd.DataFrame({
            "feature":    model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        })
        imp_dfs.append(imp)

    avg_imp = (
        pd.concat(imp_dfs)
          .groupby("feature")["importance"]
          .mean()
          .sort_values(ascending=False)
          .reset_index()
    )
    return avg_imp


# ── 9. Persist Predictions to SQL ────────────────────────────────────────────

def save_predictions(predictions: pd.DataFrame, db_path: str = DB_PATH):
    """Store out-of-sample predictions in the `predictions` table."""
    df = predictions[["ticker", "date", "predicted_score", TARGET_COL]].copy()
    df = df.rename(columns={
        "predicted_score": "predicted_rank",
        TARGET_COL:        "actual_return",
    })
    df["date"] = df["date"].astype(str)

    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM predictions")
        df.to_sql("predictions", conn, if_exists="append", index=False)

    logger.info(f"Saved {len(df):,} prediction rows to DB")


# ── 10. Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Building feature matrix...")
    panel = build_feature_matrix()

    logger.info("Starting walk-forward training...")
    results = train_walk_forward(panel, train_years=2, test_months=6)

    imp = feature_importance(results["models"])
    print("\nTop features by gain:")
    print(imp.to_string(index=False))

    save_predictions(results["predictions"])
    print("\n✅ Model training complete.")