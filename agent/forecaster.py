"""24-hour iterative forecast using the Milestone 1 Random Forest model."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from agent.schemas import ForecastPoint, ForecastState

RF_MODEL_PATH = Path("models/random_forest_model.pkl")
FEATURES_PATH = Path("models/feature_list.pkl")
POWER_CAP_KW = 1500.0


@lru_cache(maxsize=1)
def _load_rf():
    with open(RF_MODEL_PATH, "rb") as f:
        rf = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    return rf, features


def _irradiation_curve(pattern: str) -> list[float]:
    """Preset hourly irradiation (kW/m2) for 24h by weather pattern."""
    base = [0, 0, 0, 0, 0, 0,
            0.05, 0.15, 0.30, 0.50, 0.70, 0.85,
            0.90, 0.85, 0.70, 0.50, 0.30, 0.15,
            0.05, 0, 0, 0, 0, 0]
    factor = {"sunny": 1.0, "partly_cloudy": 0.7, "overcast": 0.35}.get(pattern, 1.0)
    return [v * factor for v in base]


def _module_temp_curve(ambient: float, irradiation: list[float]) -> list[float]:
    """Simple module temp model: ambient + k * irradiation."""
    return [ambient + 20.0 * i for i in irradiation]


def generate_24h_forecast(
    date: str,
    pattern: str = "sunny",
    ambient_temp: float = 28.0,
    seed_lag: float = 0.0,
) -> ForecastState:
    """Produce a 24-point ForecastState by iterating the RF model hour by hour.

    lag_1 and rolling_mean_3 are updated from the model's own prior predictions,
    which is the standard one-step-ahead approach for autoregressive forecasts.
    """
    rf, features = _load_rf()

    irradiation = _irradiation_curve(pattern)
    module_temps = _module_temp_curve(ambient_temp, irradiation)

    predictions: list[float] = []
    points: list[ForecastPoint] = []
    recent: list[float] = [seed_lag, seed_lag, seed_lag]

    for hour in range(24):
        lag_1 = recent[-1]
        rolling_mean_3 = float(np.mean(recent[-3:]))
        row = {
            "AMBIENT_TEMPERATURE": ambient_temp,
            "MODULE_TEMPERATURE": module_temps[hour],
            "IRRADIATION": irradiation[hour],
            "HOUR": hour,
            "MONTH": 6,
            "DAY_OF_WEEK": 2,
            "lag_1": lag_1,
            "rolling_mean_3": rolling_mean_3,
        }
        x = np.array([[row[f] for f in features]])
        y_hat = float(rf.predict(x)[0])
        y_hat = float(np.clip(y_hat, 0.0, POWER_CAP_KW))

        predictions.append(y_hat)
        recent.append(y_hat)
        points.append(
            ForecastPoint(
                hour=hour,
                ac_power_kw=y_hat,
                irradiation=irradiation[hour],
                module_temp=module_temps[hour],
            )
        )

    peak = max(predictions) if predictions else 0.0
    total_kwh = float(sum(predictions))
    low_threshold = 0.2 * peak if peak > 0 else 0.0
    low_power_hours = [
        h for h, p in enumerate(predictions)
        if 6 <= h <= 18 and p < low_threshold
    ]
    high_var = _find_high_variability_windows(predictions)

    return ForecastState(
        date=date,
        points=points,
        peak_kw=peak,
        total_kwh=total_kwh,
        low_power_hours=low_power_hours,
        high_variability_windows=high_var,
    )


def _find_high_variability_windows(predictions: list[float]) -> list[tuple[int, int]]:
    """Flag consecutive-hour windows where |delta| > 30% of peak."""
    if not predictions:
        return []
    peak = max(predictions)
    if peak == 0:
        return []
    threshold = 0.30 * peak
    windows: list[tuple[int, int]] = []
    start = None
    for h in range(1, len(predictions)):
        delta = abs(predictions[h] - predictions[h - 1])
        if delta > threshold:
            if start is None:
                start = h - 1
            end = h
        else:
            if start is not None:
                windows.append((start, end))
                start = None
    if start is not None:
        windows.append((start, len(predictions) - 1))
    return windows
