"""PredictionAgent: a lightweight forecast using linear trend on recent prices.

Provides a point forecast for the next period and a simple confidence estimate based on residuals.
"""
from typing import Dict, Any
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


class PredictionAgent:
    name = "PredictionAgent"

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        history = context.get("history", [])
        if np is None:
            return {"ticker": ticker, "error": "numpy not installed. Install with 'pip install numpy'"}

        if not history or len(history) < 5:
            return {"ticker": ticker, "error": "not enough history to forecast (need >=5 days)"}

        # Use last N points
        N = min(30, len(history))
        recent = history[-N:]
        ys = np.array([h["close"] for h in recent], dtype=float)
        xs = np.arange(len(ys))
        # Fit linear trend
        coeffs = np.polyfit(xs, ys, 1)
        slope, intercept = coeffs[0], coeffs[1]
        next_x = len(ys)
        forecast = float(slope * next_x + intercept)

        # residuals for simple std dev
        preds = slope * xs + intercept
        resid = ys - preds
        sigma = float(np.std(resid))

        return {
            "ticker": ticker,
            "forecast_next_close": forecast,
            "trend_slope": float(slope),
            "residual_std": sigma,
            "confidence_interval_approx": [forecast - 1.96 * sigma, forecast + 1.96 * sigma],
        }
