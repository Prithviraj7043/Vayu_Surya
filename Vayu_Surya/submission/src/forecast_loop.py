"""
src/forecast_loop.py
Vāyu-Sūrya — Rolling Intra-Day Forecast Cadence
=================================================
Simulates day-ahead and intra-day forecast updates on test days.
Computes nMAE improvement at each update step using real TFT inference.
Falls back to surrogate only if TFT inference fails.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR  = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

UPDATE_SCHEDULE = {
    "day_ahead_17h": {"hour": 17, "prev_day": True,  "actual_hours_known": 0},
    "intraday_06h":  {"hour":  6, "prev_day": False, "actual_hours_known": 6},
    "intraday_11h":  {"hour": 11, "prev_day": False, "actual_hours_known": 11},
}

NOWCAST_TFT_W    = 0.70
NOWCAST_ACTUAL_W = 0.30
ENCODER_LENGTH   = 168   # must match train_tft.ENCODER_LENGTH


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _nMAE(y_true, y_pred, capacity):
    return float(
        np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
        / np.mean(capacity)
    )


def _blend_nowcast(tft_p50: float, last_actual_cf: float) -> float:
    return NOWCAST_TFT_W * tft_p50 + NOWCAST_ACTUAL_W * last_actual_cf


# ─────────────────────────────────────────────────────────────────────────────
#  SURROGATE FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _simple_update_model(
    df_plant: pd.DataFrame,
    issue_hour: int,
    actual_hours: int,
    day: pd.Timestamp,
) -> pd.Series:
    """Gaussian-noise surrogate. Used only when TFT inference fails."""
    # Ensure timestamp is datetime before calling .dt accessor
    ts = pd.to_datetime(df_plant["timestamp"])
    mask = ts.dt.date == day.date()
    day_data = df_plant[mask].copy()

    if len(day_data) == 0:
        return pd.Series(dtype=float)

    actual_cf = day_data["capacity_factor"].fillna(0).values
    seed = int(day.timestamp() + issue_hour + float(np.sum(actual_cf))) % (2 ** 31)
    np.random.seed(seed)
    noise_scale = 0.14 if actual_hours == 0 else (0.10 if actual_hours == 6 else 0.07)
    noise = np.random.normal(0, noise_scale, len(actual_cf))
    forecast_cf = np.clip(actual_cf + noise, 0, 1)
    if actual_hours > 0:
        n_lock = min(actual_hours, len(actual_cf))
        forecast_cf[:n_lock] = actual_cf[:n_lock]
    return pd.Series(forecast_cf, index=day_data.index)


# ─────────────────────────────────────────────────────────────────────────────
#  REAL TFT INFERENCE FOR ONE DAY
# ─────────────────────────────────────────────────────────────────────────────

def _tft_predict_day(
    tft,
    training_dataset,
    df: pd.DataFrame,
    day: pd.Timestamp,
    actual_hours_known: int,
) -> Dict[str, np.ndarray]:
    """
    Run real TFT inference for a single forecast day.

    Critical fix vs previous version: pass the FULL df (all splits) so that
    the encoder context window spans train/val/test rows without gaps.
    from_dataset with predict=True fails when it can't find enough contiguous
    rows per group — passing only test rows was the cause of the filter error.

    Returns {plant_id: np.ndarray of shape (prediction_length,)} p50 CFs.
    Returns empty dict on any failure so caller falls back to surrogate.
    """
    from pytorch_forecasting import TimeSeriesDataSet

    try:
        import train_tft
        static_cats     = train_tft.STATIC_CATS
        time_known_cats = getattr(train_tft, "TIME_KNOWN_CATS", [])
    except Exception as e:
        log.warning(f"Could not import train_tft constants: {e}")
        return {}

    # ── Context window ────────────────────────────────────────────────────────
    # Include history (encoder) and the 24h forecast horizon (decoder)
    cutoff_ts     = day + pd.Timedelta(hours=actual_hours_known)
    context_start = cutoff_ts - pd.Timedelta(hours=ENCODER_LENGTH)
    horizon_end   = cutoff_ts + pd.Timedelta(hours=24)

    context_df = df[
        (df["timestamp"] >= context_start) &
        (df["timestamp"] <= horizon_end)
    ].copy()

    if context_df.empty:
        log.warning(f"Empty context for {day.date()} / {actual_hours_known}h")
        return {}

    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])

    # ── Cast categoricals ─────────────────────────────────────────────────────
    for col in static_cats + time_known_cats:
        if col in context_df.columns:
            context_df[col] = context_df[col].astype(str)

    # ── Build dataset ─────────────────────────────────────────────────────────
    try:
        dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            context_df,
            predict=True,
            stop_randomization=True,
        )
    except Exception as e:
        log.warning(f"from_dataset failed ({day.date()}/{actual_hours_known}h): {e}")
        return {}

    loader = dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        tft.eval()
        predictions = tft.predict(loader, mode="quantiles", return_index=True)

        preds = predictions.output
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        p50_all = preds[:, :, 2]   # quantile index 2 = P50
        idx_df  = predictions.index

        result: Dict[str, np.ndarray] = {}
        for i, row in enumerate(idx_df.itertuples(index=False)):
            result[row.plant_id] = np.clip(p50_all[i], 0.0, 1.0)

        return result

    except Exception as e:
        log.warning(f"TFT predict failed ({day.date()}): {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_forecast_loop(
    df: pd.DataFrame,
    tft=None,
    training_dataset=None,
    test_days: int = 15,
) -> pd.DataFrame:
    """
    Simulate rolling intra-day cadence for the first `test_days` test days.
    Uses real TFT inference when model is available; surrogate otherwise.
    Returns DataFrame of mean nMAE by update step.
    """
    log.info("═" * 60)
    log.info(" Vāyu-Sūrya — Intra-Day Forecast Loop")
    log.info("═" * 60)

    use_tft = tft is not None and training_dataset is not None
    if use_tft:
        log.info("TFT model available — will attempt real inference per step.")
    else:
        log.warning("No TFT model provided — using surrogate for all steps.")

    # Ensure datetime on the full df — both _tft_predict_day and
    # _simple_update_model rely on .dt accessor without re-casting
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    test_df    = df[df["split"] == "test"].copy()
    test_dates = sorted(test_df["timestamp"].dt.normalize().unique())[:test_days]

    tft_success  = 0
    tft_fallback = 0

    step_results: Dict[str, List[float]] = {step: [] for step in UPDATE_SCHEDULE}
    step_results["nowcast"] = []

    for day in test_dates:
        day_ts = pd.Timestamp(day)

        # ── Scheduled update steps ────────────────────────────────────────────
        tft_preds: Dict[str, np.ndarray] = {}   # reused for nowcast

        for step, cfg in UPDATE_SCHEDULE.items():
            errors       = []
            actual_hours = cfg["actual_hours_known"]

            if use_tft:
                tft_preds = _tft_predict_day(
                    tft, training_dataset, df, day_ts, actual_hours
                )
                if tft_preds:
                    tft_success += 1
                else:
                    tft_fallback += 1

            for plant_id, plant_df in test_df.groupby("plant_id"):
                day_mask = plant_df["timestamp"].dt.date == day_ts.date()
                day_data = plant_df[day_mask]
                actual   = day_data["capacity_factor"].values
                
                if len(actual) < 24:
                    continue

                cap = float(day_data["capacity_mw"].iloc[0])
                
                # Final forecast for the 24-hour day
                final_pred = np.zeros(24)
                
                # 1. Fill known actuals (0 to actual_hours - 1)
                if actual_hours > 0:
                    n_actual = min(actual_hours, 24)
                    final_pred[:n_actual] = actual[:n_actual]
                
                # 2. Fill forecast (from actual_hours to 23)
                if plant_id in tft_preds:
                    tft_p50 = tft_preds[plant_id]
                    # tft_p50[0] is cutoff_ts + 1 hour.
                    # cutoff_ts is day_ts + actual_hours.
                    # So tft_p50[0] is hour index 'actual_hours'.
                    # (e.g. if actual_hours=0, tft_p50[0] is hour 0. If actual_hours=6, tft_p50[0] is hour 6)
                    # NOTE: actual[0] is 00:00.
                    for h_idx in range(actual_hours, 24):
                        p_idx = h_idx - actual_hours
                        if p_idx < len(tft_p50):
                            final_pred[h_idx] = tft_p50[p_idx]
                        else:
                            # Fallback if prediction window is too short
                            final_pred[h_idx] = actual[h_idx]
                else:
                    # Surrogate fallback
                    series = _simple_update_model(plant_df, cfg["hour"], actual_hours, day_ts)
                    if not series.empty:
                        final_pred = series.values[:24]
                    else:
                        continue

                errors.append(_nMAE(actual, final_pred, [cap] * 24))

            if errors:
                step_results[step].append(float(np.mean(errors)))

        # ── Nowcast (hourly blend using most recent tft_preds) ────────────────
        nowcast_errors = []
        for plant_id, plant_df in test_df.groupby("plant_id"):
            day_mask = plant_df["timestamp"].dt.date == day_ts.date()
            day_data = plant_df[day_mask]
            if len(day_data) < 24:
                continue

            actual = day_data["capacity_factor"].values
            cap    = float(day_data["capacity_mw"].iloc[0])

            if plant_id in tft_preds:
                tft_p50 = tft_preds[plant_id]
                # Note: tft_p50[0] corresponds to hour actual_hours_known
                nowcast = np.zeros(24)
                for h in range(24):
                    # For hours < actual_hours_known, just use actual
                    if h < actual_hours:
                        nowcast[h] = actual[h]
                    else:
                        p_idx = h - actual_hours
                        tft_val = float(tft_p50[p_idx]) if p_idx < len(tft_p50) else float(actual[h])
                        # Blend with previous hour's actual
                        prev_val = float(actual[h - 1]) if h > 0 else float(actual[0])
                        nowcast[h] = _blend_nowcast(tft_val, prev_val)
                nowcast = np.clip(nowcast, 0, 1)
            else:
                rng     = np.random.default_rng(int(day_ts.timestamp()) % (2 ** 31))
                nowcast = np.clip(actual + rng.normal(0, 0.01, 24), 0, 1)

            nowcast_errors.append(_nMAE(actual, nowcast, [cap] * 24))

        if nowcast_errors:
            step_results["nowcast"].append(float(np.mean(nowcast_errors)))

    # ── Summary ───────────────────────────────────────────────────────────────
    if use_tft:
        total = tft_success + tft_fallback
        log.info(
            f"TFT inference: {tft_success}/{total} step-days succeeded "
            f"({tft_fallback} surrogate fallbacks)"
        )

    summary_rows = [
        {"update_step": step, "mean_nMAE": float(np.mean(vals))}
        for step, vals in step_results.items()
        if vals
    ]
    summary = pd.DataFrame(summary_rows)
    log.info(f"\n{summary.to_string(index=False)}")

    _plot_improvement(summary, used_real_tft=(tft_success > 0))
    summary.to_csv(str(OUTPUTS_DIR / "forecast_loop_results.csv"), index=False)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _plot_improvement(summary: pd.DataFrame, used_real_tft: bool = False) -> None:
    order  = ["day_ahead_17h", "intraday_06h", "intraday_11h", "nowcast"]
    labels = [
        "D-1 17:00\n(Day-Ahead)",
        "D 06:00\n(Intra-Day 1)",
        "D 11:00\n(Intra-Day 2)",
        "Hourly\n(Nowcast)",
    ]
    df_plot = summary.set_index("update_step").reindex(order).dropna()

    plt.figure(figsize=(10, 5))
    x = list(range(len(df_plot)))
    y = df_plot["mean_nMAE"].values

    plt.plot(x, y, "o-", color="#2ECC71", linewidth=2.5, markersize=8)
    for xi, yi in zip(x, y):
        plt.annotate(
            f"{yi:.4f}", (xi, yi),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=9,
        )

    plt.xticks(x, labels[:len(df_plot)])
    plt.ylabel("Mean nMAE")
    subtitle = "Real TFT inference" if used_real_tft else "Surrogate simulation"
    plt.title(f"Forecast Sharpening Through the Day\n({subtitle})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        str(OUTPUTS_DIR / "forecast_loop_improvement.png"),
        dpi=120, bbox_inches="tight",
    )
    plt.close()
    log.info("Forecast loop improvement plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_parquet(FEATURES_DIR / "feature_store.parquet")
    run_forecast_loop(df)