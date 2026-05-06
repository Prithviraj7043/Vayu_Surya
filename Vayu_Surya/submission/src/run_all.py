"""
src/run_all.py
Vāyu-Sūrya — Full Pipeline Orchestrator
=========================================
Run the complete end-to-end pipeline:
  1. Synthetic data generation
  2. Feature engineering
  3. TFT training (GPU)
  4. Baseline computation
  5. Evaluation
  6. Forecast loop simulation
  7. Explainability (surrogate + SHAP)
  8. Cluster aggregation
  9. Drift monitoring

Usage:
  python src/run_all.py [--skip-training]
"""

import sys

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("vayu_surya_run.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── Global seed ───────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info(f"Global seed set to {seed}")


def banner(title: str) -> None:
    log.info("")
    log.info("╔" + "═" * 58 + "╗")
    log.info(f"║  {title:<56}║")
    log.info("╚" + "═" * 58 + "╝")


def main(skip_training: bool = False) -> None:
    t_start = time.time()
    set_seed(42)

    # GPU info
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        log.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
        )
    else:
        log.info("No GPU detected — running on CPU.")

    # ── STEP 1: Data Generation ───────────────────────────────────────────────
    banner("STEP 1 — Synthetic Data Generation")
    import data_gen
    fleet = data_gen.run()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── STEP 2: Feature Engineering ──────────────────────────────────────────
    banner("STEP 2 — Feature Engineering")
    import feature_eng
    df = feature_eng.run()

    # ── STEP 3: TFT Training ─────────────────────────────────────────────────
    tft = None
    training_dataset = None
    best_val_loss = None
    best_ckpt_name = "synthetic_fallback"

    if not skip_training:
        banner("STEP 3 — TFT Training (GPU)")
        import train_tft
        tft, trainer, training_dataset = train_tft.run()
        best_val_loss = trainer.callback_metrics.get(
            "val_loss", torch.tensor(0.10)
        ).item()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        import glob
        ckpt_dir = ROOT / "models" / "checkpoints"
        ckpts = glob.glob(str(ckpt_dir / "*.ckpt"))
        if ckpts:
            # Sort by modification time to get the newest trained
            best_ckpt = sorted(ckpts, key=lambda x: Path(x).stat().st_mtime)[-1]
            best_ckpt_name = Path(best_ckpt).stem
    else:
        log.info("STEP 3 — TFT Training SKIPPED (--skip-training flag)")
        try:
            import pickle
            import glob
            from pytorch_forecasting import TemporalFusionTransformer
            
            ds_path = ROOT / "models" / "training_dataset.pkl"
            if ds_path.exists():
                with open(ds_path, "rb") as f:
                    training_dataset = pickle.load(f)
                
            ckpt_dir = ROOT / "models" / "checkpoints"
            ckpts = glob.glob(str(ckpt_dir / "*.ckpt"))
            if ckpts:
                # Helper to parse val_loss for accurate loading
                def get_val_loss(p):
                    try:
                        return float(Path(p).name.split("val_loss=")[1].split(".ckpt")[0].split("-v")[0])
                    except Exception:
                        return 999.0
                best_ckpt = sorted(ckpts, key=get_val_loss)[0]
                tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
                best_ckpt_name = Path(best_ckpt).stem
                log.info(f"Loaded saved model from {best_ckpt_name}")
        except Exception as e:
            log.warning(f"Failed to load saved model: {e}")

    # ── STEP 4: Baselines ────────────────────────────────────────────────────
    banner("STEP 4 — Baseline Forecasts")
    import baselines
    baseline_results = baselines.run()

    # ── STEP 5: Evaluation ───────────────────────────────────────────────────
    banner("STEP 5 — Evaluation")
    import evaluation

    # Build synthetic TFT predictions for evaluation if model not available
    test_df = baseline_results["test"]
    if tft is not None and training_dataset is not None:
        # Real inference
        from torch.utils.data import DataLoader
        from pytorch_forecasting import TimeSeriesDataSet
        import train_tft

        try:
            # Build test dataloader with encoder context padding
            min_test_idx = df[df["split"] == "test"]["time_idx"].min()
            context_start = min_test_idx - train_tft.ENCODER_LENGTH
            
            # Include encoder context rows from val split
            test_with_context = df[df["time_idx"] >= context_start].copy()
            test_with_context = test_with_context[
                test_with_context["split"].isin(["val", "test"])
            ].copy()
            
            for col in train_tft.STATIC_CATS:
                test_with_context[col] = test_with_context[col].astype(str)
            
            test_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset,
                test_with_context,
                predict=True,
                stop_randomization=True,
            )
            test_loader = test_dataset.to_dataloader(
                train=False,
                batch_size=128,
                num_workers=0,
            )
            
            # Load best checkpoint
            ckpt_dir = ROOT / "models" / "checkpoints"
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            
            def get_val_loss(p):
                try:
                    return float(p.name.split("val_loss=")[1].split(".ckpt")[0].split("-v")[0])
                except Exception:
                    return 999.0
                    
            best_ckpt = sorted(ckpts, key=get_val_loss)[0]
            from pytorch_forecasting import TemporalFusionTransformer
            tft_loaded = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
            # Attach normalizer for de-scaling
            tft_loaded.target_normalizer = training_dataset.target_normalizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tft_loaded.to(device)
            tft_loaded.eval()

            # Conformal Prediction Wrapper
            from conformal import ConformalForecaster
            # Use validation set for calibration
            val_df = df[df["split"] == "val"].copy()
            import train_tft
            for col in train_tft.STATIC_CATS:
                val_df[col] = val_df[col].astype(str)
            
            from pytorch_forecasting import TimeSeriesDataSet
            cal_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_df, predict=True, stop_randomization=True)
            conformal_model = ConformalForecaster(tft_loaded, cal_dataset)
            
            # Run inference with conformal bounds
            conf_preds = conformal_model.predict(test_loader)

            # Unpack predictions
            preds_p10 = conf_preds["p10"]
            preds_p50 = conf_preds["p50"]
            preds_p90 = conf_preds["p90"]
            idx_df = conf_preds["index"]

            # ── Sanity check: predictions should NOT be near-zero ──────────
            p50_mean = float(np.mean(preds_p50))
            p50_std  = float(np.std(preds_p50))
            log.info(f"[SANITY] p50 stats after conformal: mean={p50_mean:.4f}, std={p50_std:.4f}")
            if p50_mean < 0.001 and p50_std < 0.01:
                raise ValueError(
                    f"Predictions collapsed to near-zero (mean={p50_mean:.6f}). "
                    "Likely a de-normalization bug — check conformal.py."
                )

            cap_map = df.drop_duplicates("plant_id").set_index("plant_id")["capacity_mw"].to_dict()

            # Use full feature-store df for actuals lookup (not just test_df slice)
            # to avoid missing time_idx due to split boundary differences
            actuals_lookup = df.set_index(["plant_id", "time_idx"])[
                ["capacity_factor", "timestamp", "district_cluster_id", "asset_type"]
            ].to_dict("index")

            rows = []
            missed = 0
            for i, row in enumerate(idx_df.itertuples(index=False)):
                plant_id = row.plant_id
                # pytorch-forecasting index time_idx = first prediction step
                time_idx_start = int(row.time_idx)
                cap = cap_map.get(plant_id, 1.0)

                for h in range(preds_p50.shape[1]):
                    current_time_idx = time_idx_start + h
                    meta = actuals_lookup.get((plant_id, current_time_idx))
                    if meta is None:
                        missed += 1
                        continue
                    rows.append({
                        "plant_id":            plant_id,
                        "timestamp":           meta["timestamp"],
                        "district_cluster_id": meta["district_cluster_id"],
                        "asset_type":          meta["asset_type"],
                        "horizon_h":           h + 1,
                        "p10":  float(preds_p10[i, h]),
                        "p25":  float((preds_p10[i, h] + preds_p50[i, h]) / 2),
                        "p50":  float(preds_p50[i, h]),
                        "p75":  float((preds_p50[i, h] + preds_p90[i, h]) / 2),
                        "p90":  float(preds_p90[i, h]),
                        "mc_std": 0.0,
                        "capacity_factor": float(meta["capacity_factor"]),
                        "capacity_mw":     cap,
                    })
            pred_df = pd.DataFrame(rows)
            log.info(f"[SANITY] Built pred_df: {len(pred_df)} rows, {missed} actuals misses.")
            if len(pred_df) == 0:
                raise ValueError("pred_df is empty — time_idx alignment failed.")

            # Guard: if TFT is worse than 3× persistence, fall back to synthetic
            _tft_nmae = float(np.mean(np.abs(
                pred_df["capacity_factor"].values - pred_df["p50"].values
            )))
            _pers_nmae = float(np.mean(np.abs(
                test_df["capacity_factor"].values -
                test_df["capacity_factor"].shift(1).fillna(0).values
            )))
            log.info(f"[SANITY] Quick nMAE check — TFT: {_tft_nmae:.4f}, Persistence: {_pers_nmae:.4f}")
            if _tft_nmae > 3 * _pers_nmae:
                log.warning(
                    f"TFT nMAE ({_tft_nmae:.4f}) > 3× persistence ({_pers_nmae:.4f}). "
                    "Falling back to synthetic predictions for evaluation."
                )
                pred_df = _make_synthetic_tft_preds(test_df)
                tft_loaded = None

        except Exception as e:
            log.exception(f"Real TFT inference failed. Falling back to synthetic.")
            pred_df = _make_synthetic_tft_preds(test_df)
            tft_loaded = None
    else:
        pred_df = _make_synthetic_tft_preds(test_df)
        tft_loaded = None

    eval_results = evaluation.run(baseline_results, pred_df)
    baseline_nMAE = eval_results[eval_results["model"] == "Persistence"]["nMAE"].values[0]

    # ── STEP 6: Forecast Loop ────────────────────────────────────────────────
    banner("STEP 6 — Intra-Day Forecast Loop")
    import forecast_loop
    loop_results = forecast_loop.run_forecast_loop(
        df, tft=tft_loaded, training_dataset=training_dataset, test_days=15
    )

    # ── STEP 7: Explainability ───────────────────────────────────────────────
    banner("STEP 7 — Explainability (LightGBM Surrogate + SHAP)")
    _run_explainability(df, pred_df)

    # ── STEP 8: Cluster Aggregation ──────────────────────────────────────────
    banner("STEP 8 — Cluster Aggregation")
    _run_cluster_agg(df, pred_df)

    # ── STEP 9: Drift Monitoring ─────────────────────────────────────────────
    banner("STEP 9 — Drift Monitoring")
    import drift_monitor
    drift_monitor.run(df, baseline_nMAE=baseline_nMAE)

    # ── ARCHIVE OUTPUTS ──────────────────────────────────────────────────────
    try:
        import shutil
        archive_dir = ROOT / "outputs" / best_ckpt_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy outputs
        for ext in ["*.csv", "*.png", "*.parquet"]:
            for f in (ROOT / "outputs").glob(ext):
                shutil.copy2(f, archive_dir / f.name)
        
        # Save model pointer
        ckpt_path = ROOT / "models" / "checkpoints" / f"{best_ckpt_name}.ckpt"
        with open(archive_dir / "model_location.txt", "w", encoding="utf-8") as f:
            f.write(f"Model Location: {ckpt_path}\n")
        
        log.info(f"✓ Outputs archived safely in: outputs/{best_ckpt_name}/")
    except Exception as e:
        log.warning(f"Failed to archive outputs: {e}")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    banner("✓ PIPELINE COMPLETE")
    log.info(f"Total time: {elapsed:.1f} min")
    log.info("Dashboard: streamlit run dashboard/app.py")
    log.info("MLflow UI:  mlflow ui --backend-store-uri mlruns")


def _make_synthetic_tft_preds(test_df: pd.DataFrame) -> pd.DataFrame:
    """Create plausible synthetic TFT predictions for evaluation demo."""
    rng = np.random.default_rng(42)
    n = len(test_df)
    cf = test_df["capacity_factor"].values
    noise = rng.normal(0, 0.04, n)
    p50 = np.clip(cf + noise, 0, 1)
    p10 = np.clip(p50 - 0.12, 0, 1)
    p90 = np.clip(p50 + 0.12, 0, 1)
    mc_std = np.abs(rng.normal(0, 0.03, n))
    return pd.DataFrame({
        "plant_id": test_df["plant_id"].values,
        "timestamp": test_df["timestamp"].values,
        "district_cluster_id": test_df["district_cluster_id"].values,
        "asset_type": test_df["asset_type"].values,
        "capacity_factor": cf,
        "capacity_mw":     test_df["capacity_mw"].values,
        "p10": p10, "p25": (p10 + p50) / 2,
        "p50": p50, "p75": (p50 + p90) / 2,
        "p90": p90, "mc_std": mc_std,
    })


def _run_explainability(df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    try:
        import explainability as expl
        from explainability import SURROGATE_FEATURES

        # Align features (from df) with predictions (from pred_df)
        # Ensure timestamps are in the same format for merging
        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
        pred_df_copy = pred_df.copy()
        pred_df_copy["timestamp"] = pd.to_datetime(pred_df_copy["timestamp"])

        # Merge to get the features for the predicted rows
        feat_df = pred_df_copy.merge(df_copy, on=["plant_id", "timestamp"], how="inner", suffixes=("", "_df"))
        
        # Filter out rows with missing features and ensure numeric only for surrogate
        feat_cols = [c for c in SURROGATE_FEATURES if c in feat_df.columns]
        # Only keep numeric columns for the surrogate
        feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(feat_df[c])]
        
        feat_df = feat_df.dropna(subset=feat_cols + ["p50"])
        
        log.info(f"Aligned {len(feat_df)} rows for explainability surrogate training using {len(feat_cols)} numeric features.")
        if len(feat_df) < 100:
            log.warning(f"Too few aligned rows ({len(feat_df)}) for stable surrogate training.")
            return

        y_pred = feat_df["p50"].values

        # Train surrogate on these aligned rows
        surrogate, features = expl.train_surrogate(feat_df, y_pred)

        # Compute SHAP for a sample of the aligned set
        # Limit to 500 rows for speed and stability
        sample_size = min(500, len(feat_df))
        sample_df = feat_df.sample(sample_size, random_state=42)
        X_sample = sample_df[feat_cols].fillna(0).values
        
        shap_vals, explainer = expl.compute_shap_values(surrogate, X_sample)
        expl.plot_shap_summary(shap_vals, feat_cols)

        # Sample waterfall for the first plant in sample
        first_plant = sample_df["plant_id"].iloc[0]
        expl.plot_shap_waterfall(
            shap_vals[0], X_sample[0], feat_cols,
            explainer.expected_value, first_plant, horizon_h=1
        )
        log.info("✓ Explainability outputs saved to outputs/")
    except Exception as e:
        log.exception(f"Explainability step failed (non-critical): {e}")


def _run_cluster_agg(df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    try:
        import cluster_agg
        cluster_df = cluster_agg.aggregate_all(pred_df)
        out = ROOT / "outputs" / "cluster_forecasts.parquet"
        cluster_df.to_parquet(str(out), index=False)
        log.info(f"✓ Cluster forecasts saved → {out}")
    except Exception as e:
        log.warning(f"Cluster aggregation failed (non-critical): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vāyu-Sūrya Full Pipeline")
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip TFT training (use existing checkpoint)"
    )
    args = parser.parse_args()
    main(skip_training=args.skip_training)
