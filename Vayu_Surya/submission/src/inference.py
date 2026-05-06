"""
src/inference.py
Vāyu-Sūrya — ONNX + TFT Inference with MC Dropout
====================================================
Provides a unified inference interface that supports:
  - Direct TFT inference (PyTorch, GPU)
  - ONNX runtime inference (CPU/GPU)
  - MC Dropout uncertainty pass
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from uncertainty import mc_dropout_predict, QUANTILE_IDX

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = Path(__file__).resolve().parent.parent

ONNX_PATH = ROOT / "models" / "onnx" / "vayu_surya.onnx"
CKPT_DIR  = ROOT / "models" / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_tft_from_checkpoint() -> Optional[object]:
    """Load best TFT checkpoint."""
    try:
        from pytorch_forecasting import TemporalFusionTransformer
        ckpts = sorted(CKPT_DIR.glob("tft-*.ckpt"))
        if not ckpts:
            log.warning("No TFT checkpoint found.")
            return None
        best_ckpt = ckpts[-1]  # last is best (sorted by name includes val_loss)
        # Sort by val_loss in filename
        def _val_loss(p):
            try:
                return float(str(p).split("val_loss=")[1].replace(".ckpt", ""))
            except Exception:
                return 999.0
        best_ckpt = min(ckpts, key=_val_loss)
        log.info(f"Loading checkpoint: {best_ckpt.name}")
        tft = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
        
        # Attach normalizer from training dataset if available
        ds_path = ROOT / "models" / "training_dataset.pkl"
        if ds_path.exists():
            import pickle
            with open(ds_path, "rb") as f:
                training_ds = pickle.load(f)
            tft.target_normalizer = training_ds.target_normalizer
            
        tft.eval()
        tft.to(DEVICE)
        return tft
    except Exception as e:
        log.error(f"Failed to load TFT: {e}")
        return None


def load_onnx_session():
    """Load ONNX Runtime inference session."""
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        log.info(f"ONNX session loaded (providers: {sess.get_providers()})")
        return sess
    except Exception as e:
        log.warning(f"ONNX load failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  TFT INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_tft(
    tft,
    dataloader,
    n_mc_samples: int = 30,
) -> pd.DataFrame:
    """
    Run TFT inference with MC Dropout uncertainty.
    Returns DataFrame with p10..p90, mc_std per (plant_id, horizon_h).
    """
    tft.eval()
    all_rows = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # Move batch to device
            batch_x_dev = {
                k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in batch_x.items()
            }
            output = tft(batch_x_dev)
            predictions_raw = output["prediction"] # (batch, horizon, quantiles)
            
            # De-scale quantiles
            unscaled_q_list = []
            if hasattr(tft, "target_normalizer"):
                target_scale = batch_x_dev["target_scale"]
                for q_idx in range(predictions_raw.shape[2]):
                    q_pred = predictions_raw[..., q_idx]
                    unscaled_q = tft.target_normalizer(
                        dict(prediction=q_pred, target_scale=target_scale)
                    )
                    unscaled_q_list.append(unscaled_q)
                predictions = torch.stack(unscaled_q_list, dim=-1).cpu()
            else:
                predictions = predictions_raw.cpu()

            # MC Dropout
            mc_mean, mc_std = mc_dropout_predict(tft, batch_x, n_samples=n_mc_samples)

            batch_size, horizon, n_q = predictions.shape
            for b in range(batch_size):
                for h in range(horizon):
                    row = {
                        "p10": float(predictions[b, h, QUANTILE_IDX["p10"]]),
                        "p25": float(predictions[b, h, QUANTILE_IDX["p25"]]),
                        "p50": float(predictions[b, h, QUANTILE_IDX["p50"]]),
                        "p75": float(predictions[b, h, QUANTILE_IDX["p75"]]),
                        "p90": float(predictions[b, h, QUANTILE_IDX["p90"]]),
                        "mc_std": float(mc_std[b, h]),
                        "horizon_h": h + 1,
                    }
                    all_rows.append(row)

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  FAST SYNTHETIC INFERENCE (when model not yet trained)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_inference(
    df: pd.DataFrame,
    capacity_mw: float,
    asset_type: str,
    horizon: int = 24,
) -> pd.DataFrame:
    """
    Generate plausible synthetic forecasts from feature store data.
    Used as fallback when TFT checkpoint is unavailable (e.g. demo mode).
    """
    rows = []
    for h in range(1, horizon + 1):
        cf_col = "cf_lag_1h" if "cf_lag_1h" in df.columns else "capacity_factor"
        base_cf = df[cf_col].fillna(0).values[:horizon]
        noise = np.random.normal(0, 0.05, len(base_cf))
        p50 = np.clip(base_cf + noise, 0, 1)
        p10 = np.clip(p50 - 0.10 - 0.002 * h, 0, 1)
        p90 = np.clip(p50 + 0.10 + 0.002 * h, 0, 1)
        mc_std = np.abs(np.random.normal(0, 0.03, len(p50)))

        for i, ts in enumerate(df["timestamp"].values[:len(p50)]):
            aleatoric = (p90[i] - p10[i]) * 100
            band = "GREEN" if aleatoric < 15 else ("AMBER" if aleatoric < 30 else "RED")
            rows.append({
                "timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=h),
                "horizon_h": h,
                "p10": round(p10[i], 4),
                "p50": round(p50[i], 4),
                "p90": round(p90[i], 4),
                "p25": round((p50[i] + p10[i]) / 2, 4),
                "p75": round((p50[i] + p90[i]) / 2, 4),
                "mc_std": round(float(mc_std[i]), 4),
                "forecast_mw": round(p50[i] * capacity_mw, 2),
                "p10_mw": round(p10[i] * capacity_mw, 2),
                "p90_mw": round(p90[i] * capacity_mw, 2),
                "aleatoric_range_pct": round(aleatoric, 2),
                "confidence_band": band,
            })
    return pd.DataFrame(rows)
