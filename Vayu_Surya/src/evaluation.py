"""
src/evaluation.py
Vāyu-Sūrya — Model Evaluation & Comparison
=============================================
Computes nMAE, nRMSE, pinball loss, P10-P90 coverage,
ramp detection F1, and skill score vs persistence.
Prints formatted comparison table.
"""

from __future__ import annotations
import logging
import sys

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def nMAE(y_true: np.ndarray, y_pred: np.ndarray, capacity: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nRMSE(y_true: np.ndarray, y_pred: np.ndarray, capacity: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, alpha: float) -> float:
    err = y_true - y_quantile
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    return float(np.mean((y_true >= p10) & (y_true <= p90)))


def ramp_detection_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.15,
) -> float:
    if len(y_true) < 2:
        return 0.0
    true_ramps = (np.abs(np.diff(y_true)) > threshold).astype(int)
    pred_ramps = (np.abs(np.diff(y_pred)) > threshold).astype(int)
    if true_ramps.sum() == 0 and pred_ramps.sum() == 0:
        return 1.0
    if true_ramps.sum() == 0 or pred_ramps.sum() == 0:
        return 0.0
    return float(f1_score(true_ramps, pred_ramps, zero_division=0))


def skill_score(model_nMAE: float, baseline_nMAE: float) -> float:
    if baseline_nMAE == 0:
        return 0.0
    return float(1 - (model_nMAE / baseline_nMAE))


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: np.ndarray,
    p10: np.ndarray = None,
    p90: np.ndarray = None,
    baseline_nMAE: float = None,
) -> Dict[str, float]:
    """Compute all metrics for a model and return dict."""
    metrics = {
        "model":    name,
        "nMAE":     nMAE(y_true, y_pred, capacity),
        "nRMSE":    nRMSE(y_true, y_pred, capacity),
        "ramp_f1":  ramp_detection_f1(y_true, y_pred),
        "coverage": coverage(y_true, p10, p90) if p10 is not None else float("nan"),
        "skill":    skill_score(nMAE(y_true, y_pred, capacity), baseline_nMAE)
                    if baseline_nMAE else 0.0,
    }
    return metrics


def print_table(rows: list) -> None:
    """Print a formatted comparison table."""
    header = f"{'Model':<20} {'nMAE':>7} {'nRMSE':>7} {'Coverage':>9} {'Ramp F1':>8} {'Skill':>7}"
    sep    = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        cov   = f"{r['coverage']:.3f}" if not np.isnan(r.get("coverage", float("nan"))) else "  —  "
        skill = f"{r['skill']:.3f}" if r.get("skill") is not None else "0.000"
        print(
            f"{r['model']:<20} "
            f"{r['nMAE']:>7.3f} "
            f"{r['nRMSE']:>7.3f} "
            f"{cov:>9} "
            f"{r['ramp_f1']:>8.3f} "
            f"{skill:>7}"
        )
    print(sep + "\n")


def run(
    baseline_results: Dict[str, pd.DataFrame],
    tft_predictions: pd.DataFrame = None,
    capacity_col: str = "capacity_mw",
) -> pd.DataFrame:
    """
    Evaluate all models.
    baseline_results: dict of {name: df with capacity_factor + forecast col + capacity_mw}
    tft_predictions: df with capacity_factor, p10, p50, p90, capacity_mw
    """
    log.info("═" * 60)
    log.info(" Vāyu-Sūrya — Model Evaluation")
    log.info("═" * 60)

    rows = []

    # ── Persistence ──────────────────────────────────────────────────────────
    pers = baseline_results.get("persistence")
    if pers is not None:
        y  = pers["capacity_factor"].values
        yp = pers["persistence_cf"].values
        cap = pers[capacity_col].values
        pers_nMAE = nMAE(y, yp, cap)
        rows.append({
            "model":    "Persistence",
            "nMAE":     pers_nMAE,
            "nRMSE":    nRMSE(y, yp, cap),
            "ramp_f1":  ramp_detection_f1(y, yp),
            "coverage": float("nan"),
            "skill":    0.0,
        })
    else:
        pers_nMAE = None

    # ── Climatological ───────────────────────────────────────────────────────
    clim = baseline_results.get("climatological")
    if clim is not None:
        y  = clim["capacity_factor"].values
        yp = clim["clim_cf"].values
        cap = clim[capacity_col].values
        rows.append(evaluate_model(
            "Climatological", y, yp, cap, baseline_nMAE=pers_nMAE
        ))

    # ── Ridge NWP ────────────────────────────────────────────────────────────
    ridge = baseline_results.get("ridge")
    if ridge is not None:
        y  = ridge["capacity_factor"].values
        yp = ridge["ridge_cf"].values
        cap = ridge[capacity_col].values
        rows.append(evaluate_model(
            "Linear NWP (Ridge)", y, yp, cap, baseline_nMAE=pers_nMAE
        ))

    # ── Vāyu-Sūrya TFT ───────────────────────────────────────────────────────
    if tft_predictions is not None:
        y   = tft_predictions["capacity_factor"].values
        yp  = tft_predictions["p50"].values
        p10 = tft_predictions["p10"].values
        p90 = tft_predictions["p90"].values
        cap = tft_predictions[capacity_col].values
        tft_nMAE = nMAE(y, yp, cap)
        rows.append({
            "model":    "Vāyu-Sūrya TFT",
            "nMAE":     tft_nMAE,
            "nRMSE":    nRMSE(y, yp, cap),
            "ramp_f1":  ramp_detection_f1(y, yp),
            "coverage": coverage(y, p10, p90),
            "skill":    skill_score(tft_nMAE, pers_nMAE) if pers_nMAE else 0.0,
        })
        # Assert 25% improvement
        if pers_nMAE is not None:
            improvement = 1 - tft_nMAE / pers_nMAE
            log.info(f"TFT skill vs persistence: {improvement*100:.1f}%")
            if improvement < 0.25:
                log.warning(
                    f"TFT nMAE improvement ({improvement*100:.1f}%) < 25% target. "
                    "Consider more training epochs or better feature engineering."
                )
            else:
                log.info("✓ TFT meets 25% improvement target over persistence.")

    # ── Vāyu-Sūrya Ensemble (DISCONNECTED) ───────────────────────────────────
    # Removed per user request as it underperformed standalone TFT

    print_table(rows)

    # ── Plot ─────────────────────────────────────────────────────────────────
    result_df = pd.DataFrame(rows)
    _plot_comparison(result_df)
    result_df.to_csv(str(OUTPUTS_DIR / "evaluation_results.csv"), index=False)
    log.info(f"✓ Results saved → {OUTPUTS_DIR / 'evaluation_results.csv'}")
    return result_df


def _plot_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # nMAE bar chart
    ax = axes[0]
    colors = ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"][:len(df)]
    ax.barh(df["model"], df["nMAE"], color=colors)
    ax.set_xlabel("nMAE")
    ax.set_title("Normalised MAE by Model")
    ax.invert_yaxis()

    # Skill score bar chart
    ax2 = axes[1]
    ax2.barh(df["model"], df["skill"].fillna(0), color=colors)
    ax2.axvline(0, color="gray", linewidth=0.8)
    ax2.set_xlabel("Skill Score (vs Persistence)")
    ax2.set_title("Skill Score Comparison")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "model_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Comparison plot saved.")


if __name__ == "__main__":
    from src.baselines import run as baseline_run
    baseline_results = baseline_run()
    run(baseline_results)
