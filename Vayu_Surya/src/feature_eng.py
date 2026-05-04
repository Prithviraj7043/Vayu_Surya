"""
src/feature_eng.py
Vāyu-Sūrya — Feature Store Builder
=====================================
Loads all plant Parquets, imputes dropouts, engineers time/lag/weather
features, encodes static covariates, splits train/val/test, and saves
a merged Parquet feature store to data/features/feature_store.parquet.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from tqdm import tqdm
tqdm.pandas()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
FEATURES_DIR  = ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2022-10-31 23:00"
VAL_END   = "2022-12-31 23:00"
TEST_END  = "2023-03-31 23:00"

# Karnataka festival flags (hardcoded 2022–2023)
FESTIVAL_DATES = {
    # Dasara 2022: Oct 2–11
    **{f"2022-10-{d:02d}": True for d in range(2, 12)},
    # Ugadi 2022: Apr 2
    "2022-04-02": True,
    # Diwali 2022: Oct 24–25
    "2022-10-24": True, "2022-10-25": True,
    # Dasara 2023: Oct 22–31
    **{f"2023-10-{d:02d}": True for d in range(22, 32)},
    # Ugadi 2023: Mar 22
    "2023-03-22": True,
    # Diwali 2023: Nov 13
    "2023-11-13": True,
}


# ─────────────────────────────────────────────────────────────────────────────
#  LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_all_plants() -> pd.DataFrame:
    """Load all synthetic Parquets into a single DataFrame."""
    files = sorted(SYNTHETIC_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {SYNTHETIC_DIR}. "
            "Run src/data_gen.py first."
        )
    dfs = []
    for f in tqdm(files, desc="Loading Parquet files"):
        df = pd.read_parquet(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values(["plant_id", "timestamp"]).reset_index(drop=True)
    log.info(f"Loaded {len(files)} plants → {len(combined):,} rows.")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def impute_generation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill up to 3 consecutive NaNs in actual_gen_mw.
    Beyond 3: impute with climatological mean (same hour, same month, same plant).
    """
    df = df.copy()
    df["hour"]  = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    # Climatological mean per (plant_id, hour, month) — computed on non-NaN rows
    clim = (
        df.dropna(subset=["actual_gen_mw"])
        .groupby(["plant_id", "hour", "month"])["actual_gen_mw"]
        .mean()
        .rename("clim_mean")
    )
    df = df.join(clim, on=["plant_id", "hour", "month"])

    def _impute_group(g: pd.DataFrame) -> pd.DataFrame:
        gen = g["actual_gen_mw"].copy()
        # Forward fill ≤3 consecutive NaNs
        gen = gen.fillna(method="ffill", limit=3)
        # Remaining NaNs → climatological mean
        still_nan = gen.isna()
        gen[still_nan] = g.loc[still_nan.index, "clim_mean"]
        # Final fallback: 0.0
        gen = gen.fillna(0.0).clip(lower=0)
        g["actual_gen_mw"] = gen
        return g

    df = df.groupby("plant_id", group_keys=False).progress_apply(_impute_group)
    df = df.drop(columns=["clim_mean"], errors="ignore")
    log.info("Imputation complete.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  TIME FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _season(month: pd.Series) -> pd.Series:
    """Map month to meteorological season index (0=DJF, 1=MAM, 2=JJA, 3=SON)."""
    mapping = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
               6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    return month.map(mapping)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    hour = ts.dt.hour
    dow  = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow  / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow  / 7)
    df["month"]    = ts.dt.month
    df["season"]   = _season(df["month"])
    df["is_weekend"] = (dow >= 5).astype(int)
    date_str = ts.dt.strftime("%Y-%m-%d")
    df["karnataka_festival_flag"] = date_str.map(FESTIVAL_DATES).fillna(False).astype(int)
    log.info("Time features added.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CAPACITY FACTOR
# ─────────────────────────────────────────────────────────────────────────────

def add_capacity_factor(df: pd.DataFrame) -> pd.DataFrame:
    df["capacity_factor"] = (
        df["actual_gen_mw"] / df["capacity_mw"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  LAG & ROLLING FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-plant lag and rolling window features on capacity_factor."""
    lags = [1, 2, 3, 6, 12, 24, 48, 168]

    def _group_lags(g: pd.DataFrame) -> pd.DataFrame:
        cf = g["capacity_factor"]
        for lag in lags:
            g[f"cf_lag_{lag}h"] = cf.shift(lag)
        g["cf_roll_mean_3h"]  = cf.shift(1).rolling(3,  min_periods=1).mean()
        g["cf_roll_mean_6h"]  = cf.shift(1).rolling(6,  min_periods=1).mean()
        g["cf_roll_mean_24h"] = cf.shift(1).rolling(24, min_periods=1).mean()
        g["cf_roll_std_6h"]   = cf.shift(1).rolling(6,  min_periods=1).std().fillna(0)
        g["yesterday_same_hour_cf"] = cf.shift(24)
        return g

    df = df.groupby("plant_id", group_keys=False).progress_apply(_group_lags)
    # Fill NaNs introduced by shift with 0 (head of series)
    lag_cols = [f"cf_lag_{l}h" for l in lags] + [
        "cf_roll_mean_3h", "cf_roll_mean_6h", "cf_roll_mean_24h",
        "cf_roll_std_6h", "yesterday_same_hour_cf"
    ]
    df[lag_cols] = df[lag_cols].fillna(0)
    log.info("Lag features added.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  WEATHER-DERIVED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _clearsky_ghi(lat: float, lon: float, times: pd.DatetimeIndex) -> np.ndarray:
    """Compute Ineichen clear-sky GHI for given location and times."""
    try:
        loc = pvlib.location.Location(lat, lon, tz="Asia/Kolkata", altitude=700)
        cs = loc.get_clearsky(times, model="ineichen")
        return cs["ghi"].values
    except Exception:
        return np.ones(len(times)) * 500.0  # fallback average


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cloud_cover_trend, wind variability, clearsky ratio, temp deviation."""

    def _group_weather(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # Cloud-cover 3-hour delta
        g["cloud_cover_trend"] = g["cloud_cover"] - g["cloud_cover"].shift(3)

        # Wind speed 6-hour rolling std
        g["wind_speed_variability"] = (
            g["wind_speed_80m"].shift(1).rolling(6, min_periods=1).std().fillna(0)
        )

        # Solar position (Elevation/Zenith)
        lat = g["lat"].iloc[0]
        lon = g["lon"].iloc[0]
        times = pd.DatetimeIndex(g["timestamp"])
        sol_pos = pvlib.solarposition.get_solarposition(times, lat, lon)
        g["solar_elevation"] = sol_pos["elevation"].values
        g["solar_zenith"]    = sol_pos["zenith"].values

        # GHI clear-sky ratio
        cs_ghi = _clearsky_ghi(lat, lon, times)
        g["ghi_clearsky_ratio"] = (g["ghi"].values / (cs_ghi + 1e-6)).clip(0, 1.2)

        # Temperature deviation from monthly mean
        month_mean = g.groupby("month")["temp_2m"].transform("mean")
        g["temp_deviation"] = g["temp_2m"] - month_mean

        return g

    df = df.groupby("plant_id", group_keys=False).progress_apply(_group_weather)
    df["cloud_cover_trend"]      = df["cloud_cover_trend"].fillna(0)
    df["ghi_clearsky_ratio"]     = df["ghi_clearsky_ratio"].fillna(0)
    df["temp_deviation"]         = df["temp_deviation"].fillna(0)
    log.info("Weather-derived features added.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC ENCODINGS
# ─────────────────────────────────────────────────────────────────────────────

def add_static_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df["asset_type_enc"] = df["asset_type"].map({"solar": 0, "wind": 1}).astype(int)
    df["terrain_enc"]    = df["terrain_class"].map(
        {"flat": 0, "hilly": 1, "coastal": 2}
    ).fillna(0).astype(int)
    # hub_height_m: fill NaN for solar with 0
    df["hub_height_m"] = df["hub_height_m"].fillna(0).astype(float)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  TIME INDEX (integer hours since epoch for TFT)
# ─────────────────────────────────────────────────────────────────────────────

def add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    epoch = pd.Timestamp("2022-01-01 00:00")
    df["time_idx"] = (
        (df["timestamp"] - epoch).dt.total_seconds() // 3600
    ).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SPLIT LABELS
# ─────────────────────────────────────────────────────────────────────────────

def add_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["split"] = "train"
    df.loc[ts > TRAIN_END, "split"] = "val"
    df.loc[ts > VAL_END,   "split"] = "test"
    df.loc[ts > TEST_END,  "split"] = "future"   # beyond test horizon
    counts = df["split"].value_counts()
    log.info(f"Split counts: {counts.to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("═" * 60)
    log.info(" Vāyu-Sūrya — Feature Engineering")
    log.info("═" * 60)

    df = load_all_plants()
    df = impute_generation(df)
    df = add_capacity_factor(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_weather_features(df)
    df = add_static_encodings(df)
    df = add_time_idx(df)
    df = add_split_labels(df)

    # Final clip of capacity_factor
    df["capacity_factor"] = df["capacity_factor"].clip(0, 1)

    # Impute any remaining NaNs in features to prevent PyTorch Forecasting crashes
    fill_cols = [
        "ghi", "cloud_cover", "temp_2m", "humidity",
        "wind_speed_80m", "wind_dir_80m", "pressure", "precipitation",
        "cloud_cover_trend", "wind_speed_variability", "ghi_clearsky_ratio",
        "temp_deviation", "solar_elevation", "solar_zenith"
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    out = FEATURES_DIR / "feature_store.parquet"
    df.to_parquet(out, index=False)
    log.info(f"✓ Feature store saved → {out}  ({len(df):,} rows, {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    run()
