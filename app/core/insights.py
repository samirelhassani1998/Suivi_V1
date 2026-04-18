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
    data = df.sort_values("Date")
    current = float(data["Poids (Kgs)"].iloc[-1])
    last_date = data["Date"].max()

    if current <= target_weight:
        return {"credible": True, "message": "Objectif déjà atteint !", "eta": last_date,
                "eta_min": last_date, "eta_max": last_date, "confidence": 1.0, "scenarios": {}}

    # Multi-scenario ETA based on different windows
    scenarios = {}
    for name, window in [("optimiste", 7), ("réaliste", 30), ("pessimiste", 90)]:
        subset = data.tail(min(window, len(data)))
        if len(subset) < 3:
            continue
        x = np.arange(len(subset))
        slope, intercept = np.polyfit(x, subset["Poids (Kgs)"], 1)
        if slope >= -0.005:
            scenarios[name] = {"slope": round(slope, 4), "credible": False,
                               "message": "Tendance insuffisante sur cette fenêtre"}
            continue
        remaining = current - target_weight
        days_needed = int(remaining / abs(slope))
        eta = last_date + pd.Timedelta(days=max(days_needed, 0))
        scenarios[name] = {
            "slope": round(slope, 4),
            "credible": True,
            "eta": eta,
            "days_remaining": days_needed,
            "kg_per_week": round(slope * 7, 3),
        }

    # Primary estimate from 30-day trend
    recent = data.tail(30)
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent["Poids (Kgs)"], 1)
    if slope >= -0.005:
        return {
            "credible": False,
            "message": f"La tendance 30j est de {slope*7:+.3f} kg/sem — insuffisante pour une estimation fiable.",
            "slope_30d": round(slope, 4),
            "scenarios": scenarios,
        }
    remaining = current - target_weight
    days = int(remaining / abs(slope))
    eta = last_date + pd.Timedelta(days=max(days, 0))
    return {
        "credible": True,
        "eta": eta,
        "eta_min": eta - pd.Timedelta(days=14),
        "eta_max": eta + pd.Timedelta(days=21),
        "confidence": 0.6,
        "slope_30d": round(slope, 4),
        "scenarios": scenarios,
    }
