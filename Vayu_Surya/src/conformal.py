"""
src/conformal.py
Vāyu-Sūrya — Conformal Prediction Wrapper
==========================================
Calibrates TFT probabilistic bands to achieve exact coverage targets.
"""

import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet

class ConformalForecaster:
    """
    Wraps TFT and calibrates P10/P90 to achieve exactly 80% coverage.
    Replaces MC Dropout as the primary uncertainty mechanism.
    """
    def __init__(self, tft_model, calibration_dataset: TimeSeriesDataSet):
        self.model = tft_model
        self.target_normalizer = calibration_dataset.target_normalizer
        self.alpha = 0.20  # target 80% coverage
        self.q_hat = 0.0
        self._calibrate(calibration_dataset)
    
    def _calibrate(self, cal_dataset: TimeSeriesDataSet):
        """Calibrate conformal correction using validation set.
        
        NOTE: model.predict(mode="quantiles") already returns de-normalized
        predictions (pytorch-forecasting applies the inverse transform internally
        via transform_output). Do NOT call target_normalizer again.
        """
        loader = cal_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

        # output is already de-normalized (capacity_factor scale, 0-1)
        predictions = self.model.predict(loader, mode="quantiles", return_y=True)

        preds = predictions.output  # (N, horizon, n_quantiles) — already in CF scale
        actuals = predictions.y[0]  # (N, horizon) — already in CF scale

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(actuals, torch.Tensor):
            actuals = actuals.cpu().numpy()

        # Reshape actuals if needed
        if actuals.ndim == 1:
            actuals = actuals.reshape(preds.shape[0], preds.shape[1])

        p10 = preds[:, :, 0]  # Index 0 = P10
        p90 = preds[:, :, 4]  # Index 4 = P90

        # Non-conformity score: how far the interval needs to expand to cover actual
        scores = np.maximum(p10 - actuals, actuals - p90).flatten()
        scores = scores[np.isfinite(scores)]  # remove any NaN/inf

        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = float(np.quantile(scores, np.clip(level, 0, 1)))
        print(f"Conformal calibration complete. q_hat={self.q_hat:.6f} "
              f"(mean_score={np.mean(scores):.6f})")
    
    def predict(self, loader):
        """Returns P10, P50, P90 with conformal adjustment.
        
        model.predict(mode="quantiles") already returns de-normalized values.
        We only apply the conformal q_hat correction on top.
        """
        predictions = self.model.predict(
            loader, mode="quantiles", return_index=True
        )

        preds = predictions.output  # (N, horizon, n_quantiles) — already in CF scale
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        p10 = preds[:, :, 0] - self.q_hat  # expand lower bound
        p50 = preds[:, :, 2]               # median unchanged
        p90 = preds[:, :, 4] + self.q_hat  # expand upper bound

        # Enforce monotonicity and clip to valid CF range
        p10_final = np.minimum(p10, p90)
        p90_final = np.maximum(p10, p90)

        return {
            "p10":   np.clip(p10_final, 0, 1),
            "p50":   np.clip(p50,       0, 1),
            "p90":   np.clip(p90_final, 0, 1),
            "index": predictions.index,
        }
