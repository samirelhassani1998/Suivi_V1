"""Prévisions multi-modèles avec intervalles de confiance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.core.features import build_features
from app.core.models import get_quantile_models, get_regression_models


def _training_matrix(df: pd.DataFrame, height_m: float) -> tuple[pd.DataFrame, pd.Series]:
    feat = build_features(df, height_m=height_m).dropna().copy()
    X = feat.drop(columns=["Date", "Poids (Kgs)", "Notes", "Condition de mesure"], errors="ignore")
    y = feat["Poids (Kgs)"]
    return X, y


def forecast_with_ml(df: pd.DataFrame, horizon: int, height_m: float) -> pd.DataFrame:
    if len(df) < 20:
        return pd.DataFrame()
    X, y = _training_matrix(df, height_m)
    if len(X) < 10:
        return pd.DataFrame()

    models = get_quantile_models()
    for m in models.values():
        m.fit(X, y)

    future_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    last_row = df.iloc[-1:].copy()
    rows = []
    for d in future_dates:
        row = last_row.copy()
        row.loc[row.index[0], "Date"] = d
        feat = build_features(pd.concat([df, pd.DataFrame(rows + [row.iloc[0]])], ignore_index=True), height_m=height_m).tail(1)
        Xf = feat.drop(columns=["Date", "Poids (Kgs)", "Notes", "Condition de mesure"], errors="ignore")
        p10 = float(models["q10"].predict(Xf)[0])
        p50 = float(models["q50"].predict(Xf)[0])
        p90 = float(models["q90"].predict(Xf)[0])
        rows.append({"Date": d, "prevision": p50, "borne_basse": p10, "borne_haute": p90, "confiance": 0.8})
    return pd.DataFrame(rows)


def forecast_with_sarimax(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if len(df) < 14:
        return pd.DataFrame()
    model = SARIMAX(df["Poids (Kgs)"], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.get_forecast(steps=horizon)
    ci = pred.conf_int(alpha=0.1)
    dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "prevision": pred.predicted_mean.values,
        "borne_basse": ci.iloc[:, 0].values,
        "borne_haute": ci.iloc[:, 1].values,
        "confiance": 0.9,
    })
