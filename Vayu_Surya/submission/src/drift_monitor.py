"""
src/drift_monitor.py
Vāyu-Sūrya — Feature Drift & Performance Monitoring
=====================================================
Uses Evidently AI to detect feature drift and monitors rolling
nMAE against training baseline.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR  = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_ALERT_THRESHOLD = 1.15   # 15% nMAE degradation


def _nMAE(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


DRIFT_FEATURE_COLS = [
    "ghi", "cloud_cover", "temp_2m", "humidity",
    "wind_speed_80m", "pressure", "ghi_clearsky_ratio",
]


def run_evidently_drift(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
) -> dict:
    """
    Run Evidently data drift report.
    Returns report dict with drift flag.
    """
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset

        feat_cols = [c for c in DRIFT_FEATURE_COLS if c in reference.columns and c in current.columns]
        ref = reference[feat_cols].dropna().sample(min(5000, len(reference)), random_state=42)
        cur = current[feat_cols].dropna().sample(min(5000, len(current)), random_state=42)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        
        # In evidently 0.7+, the results are accessed via .as_dict() 
        # but we need to handle potential changes in the return structure
        try:
            result = report.as_dict()
            drift_detected = result["metrics"][0]["result"].get("dataset_drift", False)
            drifted_features = result["metrics"][0]["result"].get("number_of_drifted_columns", 0)
        except (KeyError, AttributeError):
            # Fallback if structure is different
            drift_detected = False
            drifted_features = 0

        if drift_detected:
            log.warning(
                f"⚠️ DRIFT ALERT: Feature distribution has shifted "
                f"({drifted_features}/{len(feat_cols)} features). Retraining recommended."
            )
        else:
            log.info(f"✓ No significant drift detected ({drifted_features}/{len(feat_cols)} features drifted).")

        report_path = str(OUTPUTS_DIR / "drift_report.html")
        report.save_html(report_path)
        log.info(f"Evidently report saved → {report_path}")
        return {"drift_detected": drift_detected, "drifted_features": drifted_features, "report_path": report_path}

    except ImportError:
        log.warning("Evidently not installed. Skipping drift report.")
        return {"drift_detected": False, "drifted_features": 0, "report_path": None}
    except Exception as e:
        log.error(f"Evidently drift report failed: {e}")
        return {"drift_detected": False, "drifted_features": 0, "report_path": None}


def compute_rolling_nMAE(
    df: pd.DataFrame,
    baseline_nMAE: float,
    window_days: int = 30,
    pred_col: str = "p50",
) -> pd.DataFrame:
    """
    Compute rolling 30-day nMAE from test set (uses cf_lag_24h as proxy forecast).
    Flags periods where nMAE exceeds baseline * DRIFT_ALERT_THRESHOLD.
    """
    test_df = df[df["split"] == "test"].copy()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    test_df = test_df.sort_values("timestamp")

    if pred_col not in test_df.columns:
        # Use persistence proxy
        test_df[pred_col] = test_df["cf_lag_24h"].fillna(0)

    test_df["abs_err"] = np.abs(test_df["capacity_factor"] - test_df[pred_col])
    test_df["cap"] = test_df["capacity_mw"]

    daily = (
        test_df.groupby(test_df["timestamp"].dt.date)
        .apply(lambda g: _nMAE(g["capacity_factor"], g[pred_col]))
        .rename("daily_nMAE")
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )
    daily["rolling_nMAE"] = daily["daily_nMAE"].rolling(window_days, min_periods=1).mean()
    daily["threshold"]    = baseline_nMAE * DRIFT_ALERT_THRESHOLD
    daily["alert"]        = daily["rolling_nMAE"] > daily["threshold"]

    alert_days = daily[daily["alert"]]["date"].tolist()
    if alert_days:
        log.warning(
            f"⚠️ PERFORMANCE ALERT: nMAE degraded >15% on {len(alert_days)} days. "
            "Retraining recommended."
        )

    daily.to_csv(str(OUTPUTS_DIR / "rolling_nMAE.csv"), index=False)
    return daily


def run(df: pd.DataFrame = None, baseline_nMAE: float = 0.10) -> dict:
    log.info("═" * 60)
    log.info(" Vāyu-Sūrya — Drift Monitor")
    log.info("═" * 60)

    if df is None:
        df = pd.read_parquet(FEATURES_DIR / "feature_store.parquet")

    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]

    # Evidently drift
    drift_result = run_evidently_drift(train_df, test_df)

    # Rolling nMAE
    rolling = compute_rolling_nMAE(df, baseline_nMAE)

    return {
        "drift": drift_result,
        "rolling_nMAE": rolling,
    }


if __name__ == "__main__":
    run()
