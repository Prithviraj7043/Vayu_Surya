"""
src/cluster_agg.py
Vāyu-Sūrya — Plant → District → Karnataka Aggregation
=========================================================
Aggregates plant-level probabilistic forecasts to cluster and
Karnataka-wide totals with uncertainty propagation.
"""

from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CROSS_PLANT_CORR = 0.3   # assumed inter-plant correlation


def _traffic_light(p10_mw: float, p90_mw: float, capacity_mw: float) -> str:
    if capacity_mw == 0:
        return "AMBER"
    range_pct = (p90_mw - p10_mw) / capacity_mw * 100
    if range_pct < 15:
        return "GREEN"
    elif range_pct < 30:
        return "AMBER"
    else:
        return "RED"


def aggregate_cluster(
    plants: pd.DataFrame,       # rows: one per plant for a given hour
    p50_col: str = "p50",
    std_col:  str = "mc_std",
    cap_col:  str = "capacity_mw",
) -> Dict:
    """
    Aggregate plant-level forecasts to cluster total with uncertainty propagation.
    Assumes cross-plant correlation of CROSS_PLANT_CORR.
    """
    plant_ids   = plants["plant_id"].values
    capacities  = plants[cap_col].values
    p50s        = plants[p50_col].values * capacities           # MW
    stds_mw     = plants[std_col].fillna(0).values * capacities  # MW std

    cluster_p50 = p50s.sum()
    total_cap   = capacities.sum()

    n = len(plants)
    # Variance = sum of plant variances + cross terms
    var = np.sum(stds_mw ** 2)
    if n > 1:
        cross = CROSS_PLANT_CORR * np.sum([
            stds_mw[i] * stds_mw[j]
            for i in range(n)
            for j in range(n)
            if i != j
        ])
        var += cross
    cluster_std = np.sqrt(var + 1e-6)

    cluster_p10 = cluster_p50 - 1.28 * cluster_std
    cluster_p90 = cluster_p50 + 1.28 * cluster_std
    cluster_p10 = max(0.0, cluster_p10)
    cluster_p90 = min(total_cap, cluster_p90)

    return {
        "cluster_p10_mw": round(cluster_p10, 2),
        "cluster_p50_mw": round(cluster_p50, 2),
        "cluster_p90_mw": round(cluster_p90, 2),
        "cluster_capacity_mw": round(total_cap, 2),
        "cluster_std_mw": round(cluster_std, 2),
        "cluster_cf": round(cluster_p50 / total_cap, 4) if total_cap > 0 else 0.0,
        "confidence_band": _traffic_light(cluster_p10, cluster_p90, total_cap),
        "n_plants": n,
    }


def aggregate_all(
    forecast_df: pd.DataFrame,   # one row per (plant_id, timestamp)
    cluster_id_col: str = "district_cluster_id",
) -> pd.DataFrame:
    """
    Returns cluster-level aggregated forecasts per timestamp.
    Input requires: plant_id, timestamp, p50, mc_std, capacity_mw, district_cluster_id
    """
    cluster_rows = []

    for ts, ts_group in forecast_df.groupby("timestamp"):
        # District clusters
        for cluster_id, cluster_plants in ts_group.groupby(cluster_id_col):
            agg = aggregate_cluster(cluster_plants)
            agg["timestamp"]  = ts
            agg["cluster_id"] = str(cluster_id)
            agg["asset_type"] = cluster_plants["asset_type"].iloc[0]
            cluster_rows.append(agg)

        # Karnataka total
        total = aggregate_cluster(ts_group)
        total["timestamp"]  = ts
        total["cluster_id"] = "KA_TOTAL"
        total["asset_type"] = "mixed"
        cluster_rows.append(total)

    cluster_df = pd.DataFrame(cluster_rows)
    log.info(
        f"Cluster aggregation: {len(cluster_df):,} rows "
        f"({forecast_df['timestamp'].nunique()} timestamps × "
        f"{forecast_df[cluster_id_col].nunique()} clusters + KA_TOTAL)"
    )
    return cluster_df


def karnataka_summary(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Return Karnataka-total rows only."""
    return cluster_df[cluster_df["cluster_id"] == "KA_TOTAL"].copy()


if __name__ == "__main__":
    # Demo: generate random forecast df and aggregate
    rng = np.random.default_rng(42)
    n_plants = 70
    n_hours  = 24
    demo = pd.DataFrame({
        "plant_id":             [f"P{i:03d}" for i in range(n_plants)] * n_hours,
        "timestamp":            pd.date_range("2023-01-01", periods=n_hours, freq="h").repeat(n_plants),
        "p50":                  rng.uniform(0, 1, n_plants * n_hours),
        "mc_std":               rng.uniform(0, 0.1, n_plants * n_hours),
        "capacity_mw":          rng.uniform(10, 300, n_plants * n_hours),
        "district_cluster_id":  rng.integers(0, 10, n_plants * n_hours),
        "asset_type":           rng.choice(["solar", "wind"], n_plants * n_hours),
    })
    result = aggregate_all(demo)
    print(result.tail())
