# Vāyu-Sūrya: AI-Powered Renewable Energy Forecasting

**Vāyu-Sūrya** is a high-fidelity forecasting and monitoring platform designed for the Karnataka renewable energy grid. Built for the AI for Bharat Hackathon, it utilizes a Probabilistic Temporal Fusion Transformer (TFT) to provide plant-level and cluster-level generation forecasts with uncertainty quantification.

---

## 🚀 Quick Start (Reviewer Mode)

To explore the dashboard immediately without retraining the models:

### 1. Prerequisites
* Python 3.9 or 3.10 is recommended.
* A stable internet connection for tile map loading.

### 2. Installation
```bash
# Clone the repository (if applicable)
git clone https://github.com/Prithviraj7043/Vayu_Surya.git
cd Vayu_Surya

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
# Run the Streamlit application
python -m streamlit run dashboard/app.py
```

---

## 🛠️ Advanced: Running the Full Pipeline

If you wish to regenerate the synthetic data, retrain the surrogate models, and update all metrics:

```bash
# Execute the end-to-end pipeline
python src/run_all.py
```
*Note: Full retraining of the TFT model requires a GPU and significant time. The project includes pre-trained checkpoints in `models/checkpoints/` for immediate evaluation.*

---

## 📂 Project Structure

- `dashboard/`: Streamlit UI code and dark-mode styling.
- `src/`: Core logic (TFT model, Feature Engineering, Drift Monitoring, Explainability).
- `models/`: Pre-trained checkpoints and surrogate SHAP models.
- `data/`: Synthetic generation metadata and feature stores.
- `outputs/`: Pre-calculated metrics, SHAP plots, and drift reports.

---

## ✨ Key Features
- **Probabilistic Forecasting:** Day-Ahead and Intra-Day P10/P50/P90 bands.
- **Explainable AI:** SHAP-based feature importance for every plant.
- **Fleet Monitoring:** Interactive map with confidence indicators.
- **Drift Detection:** Monitoring for data distribution shifts to ensure model reliability.
