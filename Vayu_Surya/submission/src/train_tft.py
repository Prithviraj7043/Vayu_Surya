"""
src/train_tft.py
Vāyu-Sūrya — Temporal Fusion Transformer Training (GPU)
=========================================================
Trains a multi-quantile TFT on the feature store, exports ONNX,
and logs to MLflow.  Optimised for NVIDIA RTX 4050 (6GB VRAM, sm_89).
"""

import logging
import time
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateMonitor
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── GPU Setup ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")   # TF32 for Ada Lovelace
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"Training on: {DEVICE} | GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f}GB")
else:
    log.info("Training on: CPU (no CUDA found)")

ROOT        = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
CKPT_DIR     = ROOT / "models" / "checkpoints"
ONNX_DIR     = ROOT / "models" / "onnx"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
ENCODER_LENGTH     = 168
PREDICTION_LENGTH  = 24
BATCH_SIZE         = 128
MAX_EPOCHS         = 50
HIDDEN_SIZE        = 128
HIDDEN_CONT_SIZE   = 64
ATTENTION_HEADS    = 4
DROPOUT            = 0.1
LEARNING_RATE      = 3e-4
GRAD_CLIP          = 0.1
QUANTILES          = [0.10, 0.25, 0.50, 0.75, 0.90]

STATIC_CATS    = ["asset_type_enc", "terrain_enc", "district_cluster_id"]
STATIC_REALS   = ["installed_capacity_mw", "lat", "lon", "hub_height_m"]

TIME_KNOWN_REALS = [
    "ghi", "cloud_cover", "temp_2m", "humidity",
    "wind_speed_80m", "wind_dir_80m", "pressure", "precipitation",
    "cloud_cover_trend", "wind_speed_variability", "ghi_clearsky_ratio",
    "temp_deviation", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month", "season", "is_weekend", "karnataka_festival_flag",
    "solar_elevation", "solar_zenith",
]

TIME_UNKNOWN_REALS = [
    "cf_lag_1h", "cf_lag_2h", "cf_lag_3h", "cf_lag_6h",
    "cf_lag_12h", "cf_lag_24h", "cf_lag_48h", "cf_lag_168h",
    "cf_roll_mean_3h", "cf_roll_mean_6h", "cf_roll_mean_24h",
    "cf_roll_std_6h", "yesterday_same_hour_cf", "capacity_factor",
]


def load_feature_store() -> pd.DataFrame:
    path = FEATURES_DIR / "feature_store.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature store not found: {path}. Run feature_eng.py first.")
    df = pd.read_parquet(path)
    # Ensure categorical columns are strings for pytorch-forecasting
    for col in STATIC_CATS:
        df[col] = df[col].astype(str)
    # Sort
    df = df.sort_values(["plant_id", "time_idx"]).reset_index(drop=True)
    log.info(f"Loaded feature store: {df.shape}")
    return df


def build_datasets(df: pd.DataFrame):
    """Build TimeSeriesDataSet for train, val splits."""
    train_df = df[df["split"] == "train"].copy()
    # Val: include enough encoder context (encoder_length rows per plant)
    max_val_idx = df[df["split"] == "val"]["time_idx"].max()
    min_val_idx = df[df["split"] == "val"]["time_idx"].min()
    context_start = min_val_idx - ENCODER_LENGTH
    val_df = df[df["time_idx"] >= context_start].copy()
    val_df = val_df[val_df["split"].isin(["train", "val"])].copy()

    log.info(f"Train rows: {len(train_df):,}  |  Val rows (with context): {len(val_df):,}")

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="capacity_factor",
        group_ids=["plant_id"],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=STATIC_CATS,
        static_reals=STATIC_REALS,
        time_varying_known_reals=TIME_KNOWN_REALS,
        time_varying_unknown_reals=TIME_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=["plant_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    return training, validation


def build_dataloaders(training: TimeSeriesDataSet, validation: TimeSeriesDataSet):
    train_loader = training.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 2,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=True,
    )
    return train_loader, val_loader


def build_model(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONT_SIZE,
        output_size=len(QUANTILES),
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=3,
        reduce_on_plateau_min_lr=1e-6,
    )
    log.info(f"TFT parameters: {sum(p.numel() for p in tft.parameters()):,}")
    return tft


class EpochTimerCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start
        log.info(f"Epoch {trainer.current_epoch} completed in {elapsed:.1f} seconds.")

def train(tft, train_loader, val_loader) -> Trainer:
    callbacks = [
        EpochTimerCallback(),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", patience=7, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=2,
            monitor="val_loss",
            mode="min",
        ),
    ]

    precision = "bf16-mixed" if DEVICE.type == "cuda" else "32"

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if DEVICE.type == "cuda" else "cpu",
        devices=1,
        precision=precision,
        gradient_clip_val=GRAD_CLIP,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    log.info("Starting TFT training…")
    t0 = time.time()
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_time_min = (time.time() - t0) / 60
    log.info(f"✓ Training complete in {train_time_min:.1f} min.")
    return trainer, train_time_min


def export_onnx(tft: TemporalFusionTransformer, training: TimeSeriesDataSet) -> None:
    """Export best model to ONNX (opset 17)."""
    try:
        # Get a sample batch for tracing
        sample_loader = training.to_dataloader(train=False, batch_size=1, num_workers=0)
        sample_x, _ = next(iter(sample_loader))

        onnx_path = str(ONNX_DIR / "vayu_surya.onnx")
        
        # Move to CPU for export to avoid device-specific ONNX issues
        tft.cpu()
        tft.eval()

        # Refined wrapper to ensure dictionary input is handled correctly by ONNX tracer
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, *args):
                # args[0] will be the input dictionary 'x'
                return self.model(args[0])

        wrapped_model = ONNXWrapper(tft)

        torch.onnx.export(
            wrapped_model,
            (sample_x,),           # Passed as a tuple containing the dict
            onnx_path,
            opset_version=17,
            input_names=["x"],
            dynamic_axes={"x": {0: "batch_size"}},
            do_constant_folding=True,
        )
        log.info(f"✓ ONNX exported → {onnx_path}")
    except Exception as e:
        log.warning(f"ONNX export failed: {e}")


def run():
    log.info("═" * 60)
    log.info(" Vāyu-Sūrya — TFT Training")
    log.info("═" * 60)

    df = load_feature_store()
    training, validation = build_datasets(df)
    train_loader, val_loader = build_dataloaders(training, validation)

    tft = build_model(training)

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow.set_experiment("vayu-surya-tft")
    with mlflow.start_run(run_name="tft-training"):
        mlflow.log_params({
            "encoder_length":    ENCODER_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "hidden_size":       HIDDEN_SIZE,
            "batch_size":        BATCH_SIZE,
            "max_epochs":        MAX_EPOCHS,
            "quantiles":         str(QUANTILES),
            "device":            str(DEVICE),
        })

        trainer, train_time_min = train(tft, train_loader, val_loader)

        best_val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("nan"))).item()
        mlflow.log_metrics({
            "val_loss":      best_val_loss,
            "train_time_min": train_time_min,
        })
        mlflow.pytorch.log_model(tft, "tft_model")
        log.info(f"MLflow: best_val_loss={best_val_loss:.4f}")

    # Clear VRAM before next phase
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        log.info("VRAM cleared.")

    # Export ONNX
    export_onnx(tft, training)

    # Save training dataset config for inference
    import pickle
    ds_path = ROOT / "models" / "training_dataset.pkl"
    with open(ds_path, "wb") as f:
        pickle.dump(training, f)
    log.info(f"Training dataset config saved → {ds_path}")

    return tft, trainer, training


if __name__ == "__main__":
    run()
