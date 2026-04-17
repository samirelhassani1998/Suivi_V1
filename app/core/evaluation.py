"""Évaluation robuste des prévisions (walk-forward)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(denom == 0, 0, 2 * np.abs(y_pred - y_true) / denom)) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    return float((np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).mean() * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "mape": mape,
        "smape": smape(y_true, y_pred),
        "biais": float(np.mean(y_pred - y_true)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def baseline_predictions(train: pd.Series, test_len: int, kind: str) -> np.ndarray:
    if kind == "last":
        return np.repeat(train.iloc[-1], test_len)
    if kind == "ma7":
        return np.repeat(train.tail(7).mean(), test_len)
    if kind == "ma14":
        return np.repeat(train.tail(14).mean(), test_len)
    slope = (train.iloc[-1] - train.iloc[0]) / max(len(train) - 1, 1)
    return np.array([train.iloc[-1] + slope * (i + 1) for i in range(test_len)])


def walk_forward_backtest(series: pd.Series, model_fn, n_splits: int = 5) -> dict[str, float]:
    if len(series) < 8:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "smape": np.nan, "biais": np.nan, "directional_accuracy": np.nan}

    splitter = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(series) // 5)))
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in splitter.split(series):
        train = series.iloc[train_idx]
        test = series.iloc[test_idx]
        pred = model_fn(train, len(test))
        y_true_all.extend(test.values)
        y_pred_all.extend(pred)
    return compute_metrics(np.asarray(y_true_all), np.asarray(y_pred_all))


def evaluate_baselines(series: pd.Series) -> pd.DataFrame:
    rows = []
    mapping = {"last": "Dernière valeur", "ma7": "Moyenne mobile 7j", "ma14": "Moyenne mobile 14j", "drift": "Tendance linéaire"}
    for key, label in mapping.items():
        metrics = walk_forward_backtest(series, lambda tr, h, k=key: baseline_predictions(tr, h, k))
        rows.append({"modèle": label, **metrics})
    return pd.DataFrame(rows)
