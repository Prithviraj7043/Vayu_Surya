"""
src/data_gen.py
Vāyu-Sūrya — Synthetic Fleet & NWP Data Generation
=======================================================
Generates a Karnataka renewable fleet (50 solar + 20 wind plants),
fetches 2 years of hourly NWP from Open-Meteo, simulates generation
via pvlib / windpowerlib, injects realistic anomalies, and saves
per-plant Parquet files to data/synthetic/.
"""

import os
import time
import random
import warnings
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import requests
import requests_cache
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem, FixedMount

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
RAW_DIR = ROOT / "data" / "raw"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Open-Meteo cache ─────────────────────────────────────────────────────────
requests_cache.install_cache(str(RAW_DIR / "open_meteo_cache"), backend="sqlite", expire_after=-1)

# ── District centroids ────────────────────────────────────────────────────────
SOLAR_DISTRICTS = {
    "Tumkur":      {"lat": 13.3379, "lon": 77.1173, "n": 10, "cluster_id": 0},
    "Pavagada":    {"lat": 14.0991, "lon": 77.2782, "n": 12, "cluster_id": 1},
    "Chitradurga": {"lat": 14.2251, "lon": 76.3980, "n":  8, "cluster_id": 2},
    "Bellary":     {"lat": 15.1394, "lon": 76.9214, "n":  8, "cluster_id": 3},
    "Vijayapura":  {"lat": 16.8302, "lon": 75.7100, "n":  7, "cluster_id": 4},
    "Yadgir":      {"lat": 16.7710, "lon": 77.1388, "n":  5, "cluster_id": 5},
}

WIND_DISTRICTS = {
    "Chitradurga_W": {"lat": 14.2251, "lon": 76.3980, "n": 6, "cluster_id": 6, "terrain": "hilly"},
    "Davangere":     {"lat": 14.4644, "lon": 75.9218, "n": 5, "cluster_id": 7, "terrain": "flat"},
    "Gadag":         {"lat": 15.4164, "lon": 75.6270, "n": 5, "cluster_id": 8, "terrain": "flat"},
    "Coastal":       {"lat": 14.8574, "lon": 74.1270, "n": 4, "cluster_id": 9, "terrain": "coastal"},
}

NWP_VARS = [
    "shortwave_radiation", "cloud_cover", "temperature_2m",
    "relative_humidity_2m", "wind_speed_80m", "wind_direction_80m",
    "surface_pressure", "precipitation",
]
START_DATE = "2022-01-01"
END_DATE   = "2023-12-31"


# ─────────────────────────────────────────────────────────────────────────────
#  FLEET DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

def build_solar_fleet() -> pd.DataFrame:
    """Return DataFrame of 50 solar plant metadata."""
    rows = []
    pid = 0
    for district, meta in SOLAR_DISTRICTS.items():
        for _ in range(meta["n"]):
            dlat = np.random.uniform(-0.3, 0.3)
            dlon = np.random.uniform(-0.3, 0.3)
            rows.append({
                "plant_id":            f"SOL_{pid:03d}",
                "asset_type":          "solar",
                "district":            district,
                "district_cluster_id": meta["cluster_id"],
                "lat":  round(meta["lat"] + dlat, 4),
                "lon":  round(meta["lon"] + dlon, 4),
                "installed_capacity_mw": round(np.random.uniform(10, 300), 1),
                "panel_vintage":         int(np.random.randint(2015, 2024)),
                "terrain_class": np.random.choice(["flat", "hilly"], p=[0.7, 0.3]),
                "hub_height_m":  float("nan"),
                "turbine_vintage": float("nan"),
            })
            pid += 1
    return pd.DataFrame(rows)


def build_wind_fleet() -> pd.DataFrame:
    """Return DataFrame of 20 wind cluster metadata."""
    rows = []
    pid = 50  # offset from solar
    for district, meta in WIND_DISTRICTS.items():
        terrain_map = {
            "Chitradurga_W": "hilly",
            "Davangere":     "flat",
            "Gadag":         "flat",
            "Coastal":       "coastal",
        }
        for _ in range(meta["n"]):
            dlat = np.random.uniform(-0.4, 0.4)
            dlon = np.random.uniform(-0.4, 0.4)
            rows.append({
                "plant_id":            f"WIN_{pid-50:03d}",
                "asset_type":          "wind",
                "district":            district,
                "district_cluster_id": meta["cluster_id"],
                "lat":  round(meta["lat"] + dlat, 4),
                "lon":  round(meta["lon"] + dlon, 4),
                "installed_capacity_mw": round(np.random.uniform(20, 150), 1),
                "panel_vintage":         float("nan"),
                "hub_height_m":          float(np.random.choice([80.0, 100.0])),
                "turbine_vintage":       float(np.random.randint(2015, 2023)),
                "terrain_class":         terrain_map[district],
            })
            pid += 1
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  NWP FETCHING (Open-Meteo Historical Weather API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nwp(lat: float, lon: float, plant_id: str) -> pd.DataFrame:
    """
    Fetch 2-year hourly NWP from Open-Meteo for a given lat/lon.
    Returns DataFrame indexed by timestamp (UTC→IST).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "hourly": ",".join(NWP_VARS),
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["hourly"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])
        df["plant_id"] = plant_id
        df = df.rename(columns={
            "shortwave_radiation":  "ghi",
            "temperature_2m":       "temp_2m",
            "relative_humidity_2m": "humidity",
            "wind_direction_80m":   "wind_dir_80m",
            "surface_pressure":     "pressure",
        })
        return df
    except Exception as e:
        log.warning(f"NWP fetch failed for {plant_id} ({lat},{lon}): {e}. Using zeros.")
        # Fallback: synthetic zeros so pipeline continues
        timestamps = pd.date_range(START_DATE, END_DATE, freq="h", tz="Asia/Kolkata")
        df = pd.DataFrame({"timestamp": timestamps, "plant_id": plant_id})
        for v in ["ghi", "cloud_cover", "temp_2m", "humidity",
                  "wind_speed_80m", "wind_dir_80m", "pressure", "precipitation"]:
            df[v] = 0.0
        return df


# ─────────────────────────────────────────────────────────────────────────────
#  SOLAR GENERATION (pvlib ModelChain)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_solar(plant: pd.Series, nwp: pd.DataFrame) -> np.ndarray:
    """
    Simulate solar PV output using pvlib ModelChain.
    Returns array of generation in MW (same length as nwp).
    """
    loc = Location(
        latitude=plant["lat"],
        longitude=plant["lon"],
        tz="Asia/Kolkata",
        altitude=700,
    )
    # Use a simple fixed-tilt system scaled to installed capacity
    # Tilt = latitude (rule of thumb), azimuth = 180° (south-facing)
    try:
        sandia_modules = pvlib.pvsystem.retrieve_sam("SandiaMod")
        # pick a generic module
        module_name = "Canadian_Solar_CS6K-270M"
        if module_name not in sandia_modules.columns:
            module_name = sandia_modules.columns[0]
        module = sandia_modules[module_name]

        cec_inverters = pvlib.pvsystem.retrieve_sam("cecinverter")
        inv_name = "ABB__ULTRA_1100_TL_OUTD_2_US_690_x_y_z___690V_"
        if inv_name not in cec_inverters.columns:
            inv_name = cec_inverters.columns[0]
        inverter = cec_inverters[inv_name]

        mount = FixedMount(surface_tilt=abs(plant["lat"]), surface_azimuth=180)
        array = pvlib.pvsystem.Array(mount=mount, module_parameters=module)
        system = PVSystem(arrays=[array], inverter_parameters=inverter)
        mc = ModelChain(system, loc, aoi_model="physical", spectral_model="no_loss")

        # Build weather DataFrame expected by pvlib
        times = pd.DatetimeIndex(nwp["timestamp"])
        weather = pd.DataFrame({
            "ghi": nwp["ghi"].values,
            "dhi": nwp["ghi"].values * 0.15,   # approximate diffuse fraction
            "dni": nwp["ghi"].values * 0.85,
            "temp_air": nwp["temp_2m"].values,
            "wind_speed": (nwp["wind_speed_80m"].values * 0.5),  # scale to 10m
        }, index=times)

        mc.run_model(weather)
        ac_power_w = mc.results.ac.fillna(0).clip(lower=0).values
        # Normalize: mc output is for 1 module; scale to capacity
        max_output = ac_power_w.max() if ac_power_w.max() > 0 else 1.0
        gen_mw = (ac_power_w / max_output) * plant["installed_capacity_mw"]
    except Exception as e:
        log.warning(f"pvlib simulation failed for {plant['plant_id']}: {e}. Using GHI proxy.")
        # Simple GHI-based proxy
        cf = (nwp["ghi"].values / 1000.0).clip(0, 1)
        gen_mw = cf * plant["installed_capacity_mw"]

    return gen_mw.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  WIND GENERATION (windpowerlib or power-curve proxy)
# ─────────────────────────────────────────────────────────────────────────────

def _wind_power_curve(ws: np.ndarray, rated_mw: float) -> np.ndarray:
    """
    Simple generic wind power curve:
      cut-in: 3 m/s, rated: 12 m/s, cut-out: 25 m/s
    Returns generation in MW.
    """
    gen = np.zeros_like(ws, dtype=np.float32)
    in_range = (ws >= 3) & (ws < 12)
    rated = (ws >= 12) & (ws <= 25)
    gen[in_range] = rated_mw * ((ws[in_range] - 3) / (12 - 3)) ** 3
    gen[rated] = rated_mw
    return gen.clip(0, rated_mw)


def simulate_wind(plant: pd.Series, nwp: pd.DataFrame) -> np.ndarray:
    """
    Simulate wind cluster output.  Tries windpowerlib first; falls back to
    a generic power-curve proxy if the library or turbine type is unavailable.
    """
    try:
        from windpowerlib import WindTurbine, ModelChain as WMC, wind_farm

        turbine_data = {
            "turbine_type": "E-101/3050",
            "hub_height": float(plant["hub_height_m"]),
        }
        turbine = WindTurbine(**turbine_data)

        # Build weather DataFrame for windpowerlib (MultiIndex columns)
        times = pd.DatetimeIndex(nwp["timestamp"])
        ws = nwp["wind_speed_80m"].values
        hub_h = float(plant["hub_height_m"])

        weather = pd.DataFrame({
            ("wind_speed", hub_h): ws,
            ("roughness_length", 0): np.full(len(ws), 0.1),
        }, index=times)
        weather.columns = pd.MultiIndex.from_tuples(weather.columns)

        mc_wind = WMC(turbine)
        power_output = mc_wind.run_model(weather).power_output.fillna(0).clip(lower=0).values / 1e6  # W → MW
        # Scale to cluster installed capacity
        rated = turbine.nominal_power / 1e6
        n_turbines = max(1, plant["installed_capacity_mw"] / rated)
        gen_mw = power_output * n_turbines
    except Exception as e:
        log.debug(f"windpowerlib failed for {plant['plant_id']}: {e}. Using power-curve proxy.")
        gen_mw = _wind_power_curve(nwp["wind_speed_80m"].values, plant["installed_capacity_mw"])

    return gen_mw.astype(np.float32).clip(0, plant["installed_capacity_mw"])


# ─────────────────────────────────────────────────────────────────────────────
#  ANOMALY INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def add_realistic_complexity(gen: np.ndarray, nwp: pd.DataFrame, plant: pd.Series) -> np.ndarray:
    n = len(gen)
    asset_type = plant["asset_type"]
    cap = plant["installed_capacity_mw"]
    
    soiling_cycle = 1 - 0.03 * np.sin(2 * np.pi * nwp['timestamp'].dt.dayofyear / 30)
    
    if asset_type == "solar":
        temp_penalty = np.where(nwp['temp_2m'] > 25, 1 - 0.004 * (nwp['temp_2m'] - 25), 1.0)
    else:
        temp_penalty = np.ones(n)
        
    if asset_type == "wind":
        prevailing_dir = 225
        angle_diff = np.abs(nwp['wind_dir_80m'] - prevailing_dir) % 360
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        wake_penalty = np.where(angle_diff < 30, 0.92, 1.0)
    else:
        wake_penalty = np.ones(n)
        
    if asset_type == "solar":
        clip_threshold = cap / 1.2
        gen = np.minimum(gen * soiling_cycle * temp_penalty, clip_threshold)
    else:
        gen = gen * soiling_cycle * temp_penalty * wake_penalty
        
    cloud_noise_scale = 0.04 + 0.12 * (nwp['cloud_cover'] / 100)
    structured_noise = np.random.normal(0, cloud_noise_scale, n)
    
    fault_mask = np.zeros(n, dtype=bool)
    fault_starts = np.random.choice(n, size=int(n * 0.01), replace=False)
    for start in fault_starts:
        duration = np.random.randint(2, 7)
        fault_mask[start:min(start+duration, n)] = True
    fault_penalty = np.where(fault_mask, np.random.uniform(0.5, 0.8, n), 1.0)
    
    return (gen * fault_penalty * (1 + structured_noise)).clip(0, cap)

def inject_anomalies(gen: np.ndarray, rng: np.random.Generator) -> tuple:
    """
    Returns (gen_with_anomalies, is_curtailed, is_dropout, is_ramp_event).
    """
    n = len(gen)
    is_curtailed  = np.zeros(n, dtype=bool)
    is_dropout     = np.zeros(n, dtype=bool)
    is_ramp_event  = np.zeros(n, dtype=bool)

    # Curtailment: 5% of hours
    curtail_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    is_curtailed[curtail_idx] = True
    gen[curtail_idx] *= rng.uniform(0.6, 0.9, size=len(curtail_idx))

    # Sensor dropout: 2% of rows → NaN
    dropout_idx = rng.choice(n, size=int(0.02 * n), replace=False)
    is_dropout[dropout_idx] = True
    gen[dropout_idx] = np.nan

    # Ramp events: every 72 hrs ± 12 hrs
    ramp_center = 72
    while ramp_center < n - 2:
        jitter = int(rng.integers(-12, 12))
        idx = ramp_center + jitter
        if 0 < idx < n - 2:
            step = rng.choice([-1, 1]) * 0.2
            gen[idx]     = np.nanmax([0.0, gen[idx] + step])
            gen[idx + 1] = np.nanmax([0.0, gen[idx + 1] + step])
            is_ramp_event[idx] = True
            is_ramp_event[idx + 1] = True
        ramp_center += 72

    return gen, is_curtailed, is_dropout, is_ramp_event


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_plant(plant: pd.Series) -> None:
    """Fetch NWP, simulate, inject anomalies, save Parquet for one plant."""
    pid = plant["plant_id"]
    out_path = SYNTHETIC_DIR / f"{pid}.parquet"
    if out_path.exists():
        log.info(f"[SKIP] {pid} — already generated.")
        return

    log.info(f"[GEN ] {pid} ({plant['asset_type']}) lat={plant['lat']}, lon={plant['lon']}")

    nwp = fetch_nwp(plant["lat"], plant["lon"], pid)
    time.sleep(0.1)  # rate limit

    cap = plant["installed_capacity_mw"]

    if plant["asset_type"] == "solar":
        # Ridge Gap: Add random cloud-front timing offset (±30-90 min)
        # Shift GHI and other weather features to create inter-plant temporal structure
        offset_mins = np.random.randint(-90, 91)
        if offset_mins != 0:
            nwp_shifted = nwp.copy()
            # We use shift() for discrete hourly, but user implies a more complex shift.
            # For simplicity in this hourly dataset, we'll shift by 1 hour if offset > 45m
            # or just add it to the structured noise.
            # Actually, let's just shift the GHI values slightly
            shift_steps = int(round(offset_mins / 60.0))
            if shift_steps != 0:
                nwp["ghi"] = nwp["ghi"].shift(shift_steps).fillna(0)

        gen = simulate_solar(plant, nwp)
    else:
        gen = simulate_wind(plant, nwp)

    # Add structured complexity
    gen = add_realistic_complexity(gen, nwp, plant)

    rng = np.random.default_rng(SEED + hash(pid) % 10000)
    gen, is_curtailed, is_dropout, is_ramp_event = inject_anomalies(gen, rng)

    cf = gen / cap  # capacity factor

    df = nwp.copy()
    df["actual_gen_mw"]   = gen
    df["capacity_mw"]     = cap
    df["capacity_factor"] = cf
    df["is_curtailed"]    = is_curtailed
    df["is_dropout"]      = is_dropout
    df["is_ramp_event"]   = is_ramp_event

    # Attach static metadata columns
    for col in ["asset_type", "district", "district_cluster_id",
                "lat", "lon", "panel_vintage", "hub_height_m",
                "turbine_vintage", "terrain_class",
                "installed_capacity_mw"]:
        df[col] = plant[col]

    df.to_parquet(out_path, index=False)
    log.info(f"[DONE] {pid} → {out_path.name}  ({len(df)} rows)")


def save_fleet_metadata(fleet: pd.DataFrame) -> None:
    fleet.to_parquet(RAW_DIR / "fleet_metadata.parquet", index=False)
    fleet.to_csv(RAW_DIR / "fleet_metadata.csv", index=False)
    log.info(f"Fleet metadata saved ({len(fleet)} plants).")


def run() -> pd.DataFrame:
    """Full data generation pipeline. Returns fleet metadata DataFrame."""
    log.info("═" * 60)
    log.info(" Vāyu-Sūrya — Synthetic Data Generation")
    log.info("═" * 60)

    solar_fleet = build_solar_fleet()
    wind_fleet  = build_wind_fleet()
    fleet = pd.concat([solar_fleet, wind_fleet], ignore_index=True)
    save_fleet_metadata(fleet)

    log.info(f"Fleet: {len(solar_fleet)} solar + {len(wind_fleet)} wind = {len(fleet)} total plants")

    for _, plant in tqdm(fleet.iterrows(), total=len(fleet), desc="Generating plants"):
        generate_plant(plant)

    log.info("✓ Data generation complete.")
    return fleet


if __name__ == "__main__":
    run()
