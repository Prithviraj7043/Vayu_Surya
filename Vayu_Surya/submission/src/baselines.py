"""
src/baselines.py
Vāyu-Sūrya — Baseline Forecasters
====================================
Three baselines on the test holdout set (2023-01-01 → 2023-03-31):
  1. Persistence
  2. Climatological mean
  3. Ridge regression (NWP only)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"

NWP_FEATURES = [
    "ghi", "cloud_cover", "temp_2m", "humidity",
    "wind_speed_80m", "wind_dir_80m", "pressure", "precipitation",
    "cloud_cover_trend", "wind_speed_variability",
    "ghi_clearsky_ratio", "temp_deviation",
    "solar_elevation", "solar_zenith",
]


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURES_DIR / "feature_store.parquet")
    train = df[df["split"] == "train"].copy()
    test  = df[df["split"] == "test"].copy()
    val   = df[df["split"] == "val"].copy()
    log.info(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
#  1. PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def persistence_forecast(test: pd.DataFrame) -> pd.DataFrame:
    """
    forecast_cf(t+h) = actual_cf(t) for all h in [1..24].
    We approximate by using the actual_cf at the same row (last known value).
    For the test set, last-known = cf_lag_1h.
    """
    pred = test.copy()
    pred["persistence_cf"] = pred["cf_lag_1h"].fillna(0).clip(0, 1)
    log.info("Persistence forecast computed.")
    return pred[["plant_id", "timestamp", "capacity_factor", "persistence_cf", "capacity_mw"]]


# ─────────────────────────────────────────────────────────────────────────────
#  2. CLIMATOLOGICAL MEAN
# ─────────────────────────────────────────────────────────────────────────────

def climatological_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    For each (plant_id, hour_of_day, month): mean CF over training set.
    """
    train = train.copy()
    train["hour"]  = pd.to_datetime(train["timestamp"]).dt.hour
    train["month"] = pd.to_datetime(train["timestamp"]).dt.month

    clim = (
        train.groupby(["plant_id", "hour", "month"])["capacity_factor"]
        .mean()
        .rename("clim_cf")
        .reset_index()
    )

    test = test.copy()
    test["hour"]  = pd.to_datetime(test["timestamp"]).dt.hour
    test["month"] = pd.to_datetime(test["timestamp"]).dt.month

    pred = test.merge(clim, on=["plant_id", "hour", "month"], how="left")
    pred["clim_cf"] = pred["clim_cf"].fillna(0).clip(0, 1)
    log.info("Climatological forecast computed.")
    return pred[["plant_id", "timestamp", "capacity_factor", "clim_cf", "capacity_mw"]]


# ─────────────────────────────────────────────────────────────────────────────
#  3. RIDGE REGRESSION (per-plant NWP only)
# ─────────────────────────────────────────────────────────────────────────────

def ridge_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    One Ridge model per plant.  Features: NWP variables only.
    Returns P50 only (no uncertainty).
    """
    feat_cols = [c for c in NWP_FEATURES if c in train.columns]
    results = []

    for plant_id in train["plant_id"].unique():
        tr = train[train["plant_id"] == plant_id]
        te = test[test["plant_id"] == plant_id].copy()

        X_train = tr[feat_cols].fillna(0).values
        y_train = tr["capacity_factor"].fillna(0).values
        X_test  = te[feat_cols].fillna(0).values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_train)

        te["ridge_cf"] = np.clip(ridge.predict(X_test_s), 0, 1)
        results.append(te[["plant_id", "timestamp", "capacity_factor", "ridge_cf", "capacity_mw"]])

    combined = pd.concat(results, ignore_index=True)
    log.info(f"Ridge forecast computed for {len(train['plant_id'].unique())} plants.")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run() -> Dict[str, pd.DataFrame]:
    log.info("═" * 60)
    log.info(" Vāyu-Sūrya — Baseline Forecasts")
    log.info("═" * 60)

    train, val, test = load_splits()

    pers = persistence_forecast(test)
    clim = climatological_forecast(train, test)
    ridge = ridge_forecast(train, test)

    return {"persistence": pers, "climatological": clim, "ridge": ridge, "test": test}


if __name__ == "__main__":
    run()
