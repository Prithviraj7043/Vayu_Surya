"""
src/uncertainty.py
Vāyu-Sūrya — Three-Layer Uncertainty Quantification
======================================================
1. Aleatoric:  quantile spread from TFT (P10–P90)
2. Epistemic:  MC Dropout (30 forward passes, GPU)
3. Horizon:    calibrated degradation across h=1..24
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUANTILE_IDX = {
    "p10": 0, "p25": 1, "p50": 2, "p75": 3, "p90": 4,
}
N_MC_SAMPLES = 30


# ─────────────────────────────────────────────────────────────────────────────
#  MC DROPOUT PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_predict(
    model: torch.nn.Module,
    batch_x: Dict,
    n_samples: int = N_MC_SAMPLES,
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run MC Dropout over n_samples forward passes.
    Returns (mean, std) tensors on CPU. Shape: (batch, horizon).
    """
    model.train()        # keep dropout active
    model.to(device)

    # Move batch to device
    batch_x_dev = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch_x.items()
    }

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(batch_x_dev)
            # Extract P50 (index 2) from quantile output
            p50 = output["prediction"][..., QUANTILE_IDX["p50"]]  # (batch, horizon)
            
            # De-scale to capacity factor space
            if hasattr(model, "target_normalizer"):
                target_scale = batch_x_dev["target_scale"]
                p50 = model.target_normalizer(
                    dict(prediction=p50, target_scale=target_scale)
                )
            samples.append(p50.cpu())

    model.eval()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    stack = torch.stack(samples, dim=0)           # (n_samples, batch, horizon)
    mc_mean = stack.mean(dim=0)
    mc_std  = stack.std(dim=0)
    return mc_mean, mc_std


# ─────────────────────────────────────────────────────────────────────────────
#  HORIZON DEGRADATION CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def _linear_degradation(h: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * h


def fit_horizon_degradation(
    val_actuals: np.ndarray,    # shape (n_samples, horizon)
    val_p50:     np.ndarray,    # shape (n_samples, horizon)
    capacity:    float,
) -> Tuple[float, float]:
    """
    Fit linear horizon-degradation model on validation set.
    Returns (a, b) such that scale(h) = a + b*h.
    """
    horizon = val_p50.shape[1]
    h_values = np.arange(1, horizon + 1)
    mean_err = np.array([
        np.mean(np.abs(val_p50[:, h] - val_actuals[:, h])) / (capacity + 1e-6)
        for h in range(horizon)
    ])
    try:
        popt, _ = curve_fit(_linear_degradation, h_values, mean_err,
                            p0=[mean_err[0], 0.001], maxfev=5000)
        a, b = popt
    except Exception:
        a, b = float(mean_err.mean()), 0.0
    log.info(f"Horizon degradation fit: scale(h) = {a:.4f} + {b:.4f}*h")
    return a, b


# ─────────────────────────────────────────────────────────────────────────────
#  TRAFFIC LIGHT
# ─────────────────────────────────────────────────────────────────────────────

def traffic_light(aleatoric_range_pct: float) -> str:
    if aleatoric_range_pct < 15:
        return "GREEN"
    elif aleatoric_range_pct < 30:
        return "AMBER"
    else:
        return "RED"


# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED UNCERTAINTY OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def compute_uncertainty(
    tft_output: torch.Tensor,         # raw TFT output (batch, horizon, n_quantiles)
    mc_std: torch.Tensor,             # from mc_dropout_predict (batch, horizon)
    capacity_mw: float,
    deg_a: float,
    deg_b: float,
) -> List[Dict]:
    """
    Combine all three uncertainty layers into structured output dicts.
    Returns a list of dicts, one per (sample, horizon) pair.
    """
    results = []
    batch_size, horizon, n_q = tft_output.shape

    for b in range(batch_size):
        for h in range(horizon):
            p10 = float(tft_output[b, h, QUANTILE_IDX["p10"]].cpu())
            p25 = float(tft_output[b, h, QUANTILE_IDX["p25"]].cpu())
            p50 = float(tft_output[b, h, QUANTILE_IDX["p50"]].cpu())
            p75 = float(tft_output[b, h, QUANTILE_IDX["p75"]].cpu())
            p90 = float(tft_output[b, h, QUANTILE_IDX["p90"]].cpu())

            # 1. Aleatoric range
            aleatoric_range_pct = max(0.0, (p90 - p10) * 100)

            # 2. Epistemic CI-95
            mc_s = float(mc_std[b, h].cpu()) if mc_std is not None else 0.0
            epistemic_ci_95 = mc_s * 1.96

            # 3. Horizon degradation: widen bands
            scale = _linear_degradation(h + 1, deg_a, deg_b)
            p10 = max(0.0, p10 - scale)
            p90 = min(1.0, p90 + scale)
            aleatoric_range_pct = max(0.0, (p90 - p10) * 100)

            results.append({
                "p10": round(p10, 4),
                "p25": round(p25, 4),
                "p50": round(p50, 4),
                "p75": round(p75, 4),
                "p90": round(p90, 4),
                "mc_std": round(mc_s, 4),
                "aleatoric_range_pct": round(aleatoric_range_pct, 2),
                "epistemic_ci_95": round(epistemic_ci_95, 4),
                "horizon_h": h + 1,
                "confidence_band": traffic_light(aleatoric_range_pct),
                "forecast_mw": round(p50 * capacity_mw, 2),
                "p10_mw": round(p10 * capacity_mw, 2),
                "p90_mw": round(p90 * capacity_mw, 2),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def interval_coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    """Fraction of actuals within [P10, P90] interval."""
    return float(np.mean((y_true >= p10) & (y_true <= p90)))


def pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, alpha: float) -> float:
    err = y_true - y_quantile
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def calibration_summary(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    capacity: float,
) -> Dict:
    """Compute coverage and pinball for all quantiles."""
    p10 = predictions["p10"]
    p90 = predictions["p90"]
    p50 = predictions["p50"]
    return {
        "coverage_p10_p90": interval_coverage(y_true, p10, p90),
        "pinball_p10":       pinball_loss(y_true, p10, 0.10),
        "pinball_p50":       pinball_loss(y_true, p50, 0.50),
        "pinball_p90":       pinball_loss(y_true, p90, 0.90),
        "aleatoric_mean_pct": float(np.mean(np.abs(p90 - p10) * 100)),
    }
