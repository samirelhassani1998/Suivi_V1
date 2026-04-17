"""Prévisions multi-modèles avec intervalles de confiance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.core.features import build_features
from app.core.models import get_quantile_models


def _training_matrix(df: pd.DataFrame, height_m: float) -> tuple[pd.DataFrame, pd.Series]:
    feat = build_features(df, height_m=height_m).dropna().copy()
    X = feat.drop(columns=["Date", "Poids (Kgs)", "Notes", "Condition de mesure"], errors="ignore")
    X = X.select_dtypes(include=["number"]).fillna(0.0)
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
    base = df[["Date", "Poids (Kgs)"]].copy()
    rows: list[dict[str, object]] = []

    for d in future_dates:
        history_aug = pd.concat([base, pd.DataFrame(rows)[["Date", "Poids (Kgs)"]]], ignore_index=True) if rows else base.copy()
        last_row = history_aug.iloc[-1:].copy()
        last_row.loc[last_row.index[0], "Date"] = d
        history_aug = pd.concat([history_aug, last_row], ignore_index=True)

        feat = build_features(history_aug, height_m=height_m).tail(1)
        Xf = feat.drop(columns=["Date", "Poids (Kgs)", "Notes", "Condition de mesure"], errors="ignore")
        Xf = Xf.select_dtypes(include=["number"]).fillna(0.0)
        Xf = Xf.reindex(columns=X.columns, fill_value=0.0)

        p10 = float(models["q10"].predict(Xf)[0])
        p50 = float(models["q50"].predict(Xf)[0])
        p90 = float(models["q90"].predict(Xf)[0])
        rows.append({"Date": d, "Poids (Kgs)": p50, "prevision": p50, "borne_basse": p10, "borne_haute": p90, "confiance": 0.8})

    return pd.DataFrame(rows)[["Date", "prevision", "borne_basse", "borne_haute", "confiance"]]


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
