"""Training pipeline for solar power generation forecasting.

Loads generation and weather CSVs, engineers temporal and lag features,
trains Linear Regression (scaled) and Random Forest baselines, and saves
both models along with their evaluation metrics.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

GEN_PATH = "data/Plant_1_Generation_Data.csv"
WEATHER_PATH = "data/Plant_1_Weather_Sensor_Data.csv"
MODELS_DIR = Path("models")

FEATURES = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "HOUR",
    "MONTH",
    "DAY_OF_WEEK",
    "lag_1",
    "rolling_mean_3",
]
TARGET = "AC_POWER"


def load_data(gen_path: str, weather_path: str) -> pd.DataFrame:
    """Load generation + weather CSVs, sort by DATE_TIME, and merge nearest-in-time."""
    gen_df = pd.read_csv(gen_path)
    weather_df = pd.read_csv(weather_path)

    gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])
    weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

    gen_df = gen_df.sort_values("DATE_TIME")
    weather_df = weather_df.sort_values("DATE_TIME")

    merged = pd.merge_asof(gen_df, weather_df, on="DATE_TIME", direction="nearest")
    merged = merged[merged[TARGET] > 0]
    merged = merged.drop_duplicates()
    return merged


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add HOUR/MONTH/DAY_OF_WEEK, lag_1, rolling_mean_3; drop rows with NaN lags."""
    df = df.copy()
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["MONTH"] = df["DATE_TIME"].dt.month
    df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek
    df["lag_1"] = df[TARGET].shift(1)
    df["rolling_mean_3"] = df[TARGET].rolling(window=3).mean()
    df = df.dropna()
    return df, FEATURES


def time_series_split(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    """Chronological split (no shuffle) preserving temporal order."""
    split_idx = int(len(X) * (1 - test_frac))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def _evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_linear(X_train, y_train, X_test, y_test):
    """Fit LinearRegression on scaled features; return (model, scaler, metrics)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    metrics = _evaluate(y_test, model.predict(X_test_scaled))
    return model, scaler, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Fit RandomForestRegressor(n_estimators=100); return (model, metrics)."""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    metrics = _evaluate(y_test, model.predict(X_test))
    return model, metrics


def save_artifacts(lr_model, rf_model, scaler, features, metrics) -> None:
    """Pickle models/scaler/features and dump metrics.json to MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "linear_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(MODELS_DIR / "feature_list.pkl", "wb") as f:
        pickle.dump(features, f)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    df = load_data(GEN_PATH, WEATHER_PATH)
    df, features = engineer_features(df)

    X_train, X_test, y_train, y_test = time_series_split(df[features], df[TARGET])

    lr_model, scaler, lr_metrics = train_linear(X_train, y_train, X_test, y_test)
    print(f"Linear Regression - MAE: {lr_metrics['mae']:.2f}, RMSE: {lr_metrics['rmse']:.2f}")

    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"Random Forest    - MAE: {rf_metrics['mae']:.2f}, RMSE: {rf_metrics['rmse']:.2f}")

    metrics = {"linear_regression": lr_metrics, "random_forest": rf_metrics}
    save_artifacts(lr_model, rf_model, scaler, features, metrics)
    print("\nModels + metrics.json saved to ./models/")


if __name__ == "__main__":
    main()
