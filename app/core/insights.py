"""Insights analytiques: plateau, anomalies, ETA objectif."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_plateau(df: pd.DataFrame, window: int = 14) -> dict[str, float | str]:
    if len(df) < window:
        return {"status": "données insuffisantes", "slope": 0.0, "volatility": 0.0}
    recent = df.sort_values("Date").tail(window)
    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent["Poids (Kgs)"], 1)[0])
    vol = float(recent["Poids (Kgs)"].std())
    if abs(slope) < 0.03 and vol < 0.5:
        status = "plateau probable"
    elif slope <= -0.03:
        status = "baisse active"
    elif slope >= 0.03:
        status = "reprise de poids probable"
    else:
        status = "signal mixte"
    return {"status": status, "slope": slope, "volatility": vol}


def detect_anomalies_robust(df: pd.DataFrame, use_iforest: bool = False) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["anomalie"] = False
        out["raison"] = "aucune donnée"
        return out
    median = out["Poids (Kgs)"].median()
    mad = np.median(np.abs(out["Poids (Kgs)"] - median)) + 1e-9
    z = 0.6745 * (out["Poids (Kgs)"] - median) / mad
    out["z_robuste"] = z
    out["anomalie"] = np.abs(z) > 3.5
    out["raison"] = np.where(out["anomalie"], "z-score robuste > 3.5", "normal")

    if use_iforest and len(out) >= 10:
        iso = IsolationForest(contamination=0.1, random_state=42)
        out["iforest"] = iso.fit_predict(out[["Poids (Kgs)"]]) == -1
        out["anomalie"] = out["anomalie"] | out["iforest"]
        out.loc[out["iforest"], "raison"] = "IsolationForest"
    out["decision"] = "à revoir"
    return out


def estimate_target_eta(df: pd.DataFrame, target_weight: float) -> dict[str, object]:
    if len(df) < 7:
        return {"credible": False, "message": "Données insuffisantes"}
    recent = df.sort_values("Date").tail(30)
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent["Poids (Kgs)"], 1)
    if slope >= -0.01:
        return {"credible": False, "message": "La tendance actuelle ne permet pas d'estimation crédible"}
    days = int((target_weight - intercept) / slope)
    start = recent["Date"].iloc[0]
    eta = start + pd.Timedelta(days=max(days, 0))
    return {
        "credible": True,
        "eta": eta,
        "eta_min": eta - pd.Timedelta(days=14),
        "eta_max": eta + pd.Timedelta(days=21),
        "confidence": 0.6,
    }
