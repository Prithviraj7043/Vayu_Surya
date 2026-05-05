# ⚡ Vāyu-Sūrya
### Karnataka Renewable Energy Generation Forecasting
**AI for Bharat Hackathon | KREDL / KSPDCL Theme**

---

## Overview

**Vāyu-Sūrya** (*Wind-Sun*) is a probabilistic AI forecasting system for Karnataka's solar and wind assets. It predicts generation at plant level and cluster level with explicit uncertainty quantification and explainable outputs — designed to act as a forecasting layer alongside existing systems.

### Key Capabilities
| Feature | Detail |
|---------|--------|
| **Models** | Temporal Fusion Transformer (TFT) + LightGBM SHAP surrogate |
| **Horizons** | Day-ahead (24h), intra-day (6h/11h updates), hourly nowcast |
| **Fleet** | 50 solar plants + 20 wind clusters across Karnataka |
| **Uncertainty** | Aleatoric (quantile bands) + Epistemic (MC Dropout) + Horizon degradation |
| **Explainability** | SHAP values + plain-language rules engine (zero LLM) |
| **GPU** | RTX 4050 (6GB VRAM, CUDA 12.1, sm_89) |

---

## Quick Start

### 1. Create & activate conda environment
```bash
conda env create -f environment.yml
conda activate Vayu_Surya
```

### 2. Run the full pipeline
```bash
python src/run_all.py
```
> **Expected time on RTX 4050:**
> - Data generation: ~3–5 min (first run, Open-Meteo API)
> - TFT training: ~25–40 min (50 epochs, 70 plants, 2yr hourly)
> - Total: ~45–50 min

### 3. Launch dashboard
```bash
streamlit run dashboard/app.py
```
→ Opens at **http://localhost:8501**

### 4. Launch MLflow UI
```bash
mlflow ui --backend-store-uri mlruns
```
→ Opens at **http://localhost:5000**

---

## Project Structure

```
Vayu_Surya/
├── configs/config.yaml         # All hyperparameters & thresholds
├── src/
│   ├── data_gen.py             # Synthetic fleet + pvlib/windpowerlib
│   ├── feature_eng.py          # Feature store builder
│   ├── train_tft.py            # TFT training (GPU, mixed precision)
│   ├── inference.py            # ONNX + MC Dropout inference
│   ├── uncertainty.py          # 3-layer uncertainty quantification
│   ├── explainability.py       # SHAP surrogate + rules engine
│   ├── baselines.py            # Persistence / Climatological / Ridge
│   ├── evaluation.py           # nMAE, nRMSE, pinball, ramp F1
│   ├── cluster_agg.py          # Plant → district → Karnataka-wide
│   ├── forecast_loop.py        # Intra-day rolling update cadence
│   ├── drift_monitor.py        # Evidently AI drift detection
│   └── run_all.py              # Full pipeline orchestrator
├── dashboard/app.py            # Streamlit dashboard (5 tabs)
├── data/                       # raw/, synthetic/, features/
├── models/                     # checkpoints/, onnx/, surrogate/
├── outputs/                    # plots, CSVs, SHAP charts
├── environment.yml
├── requirements.txt
└── docker-compose.yml
```

---

## GPU Configuration (RTX 4050)

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True       # Ada Lovelace tuning
torch.set_float32_matmul_precision("high")  # TF32 on RTX 40xx
# Trainer: precision="16-mixed", batch_size=128
```

**Monitor VRAM:**
```bash
nvidia-smi dmon -s u   # poll every second
```

**Expected GPU utilisation during training:** ~70–85%

---

## Architecture

```
Open-Meteo NWP API
       │
       ▼
[pvlib / windpowerlib] → Synthetic Generation
       │
       ▼
[Feature Store] (Parquet, per-plant)
  - Time features (cyclical encoding)
  - Lag features (1h → 168h)
  - Weather-derived (clearsky ratio, cloud trend)
  - Static covariates (asset type, terrain, district)
       │
       ▼
[TFT Model] ──────────── GPU (RTX 4050)
  - Encoder: 168h lookback
  - Decoder: 24h prediction
  - Output: P10/P25/P50/P75/P90
       │
       ├──► [MC Dropout] → Epistemic uncertainty
       │
       ├──► [LightGBM Surrogate] → SHAP values → Rules engine
       │
       └──► [Cluster Aggregation] → District + Karnataka-wide
                    │
                    ▼
           [Streamlit Dashboard]
```

---

## Datasets Used

| Dataset | Source | Purpose |
|---------|--------|---------|
| NWP Weather | Open-Meteo Historical API | GHI, cloud, wind, temp (2022–2023) |
| Solar simulation | pvlib (Python) | Synthetic PV output from irradiance |
| Wind simulation | windpowerlib (Python) | Turbine power curve simulation |
| Fleet metadata | KREDL 2024 Annual Report | District capacities & coordinates |

> **Note:** No real SCADA data is used. All generation data is synthetically simulated.

---

## Evaluation Targets

| Metric | Target |
|--------|--------|
| nMAE improvement vs persistence | ≥ 25% (target: 30%) |
| P10–P90 coverage | 75–85% |
| Ramp detection F1 | > 0.65 |
| Surrogate R² | > 0.92 |

---

## Skip Training (Demo Mode)

If you want to run the dashboard without waiting for TFT training:
```bash
python src/run_all.py --skip-training
```
The dashboard will show synthetic forecast data.

---

## Non-Negotiables Compliance

| Requirement | Implementation |
|-------------|----------------|
| No existing system modification | Acts as overlay forecasting layer |
| Works with synthetic/masked data | pvlib + windpowerlib simulation |
| Explainable forecasts | SHAP + rules engine (no LLM) |
| Explicit uncertainty | P10/P25/P50/P75/P90 + MC Dropout |
| No hosted LLM on sensitive data | Zero LLM — pure Python rules engine |

---

*Built for AI for Bharat Hackathon | KREDL/KSPDCL Theme*
