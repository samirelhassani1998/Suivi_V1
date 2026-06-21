"""Insights analytiques: plateau, anomalies, ETA objectif."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.business import FINAL_TARGET_WEIGHT_KG, StagnationConfig
from app.core.plateau import evaluate_plateau_window, prepare_plateau_series


def _prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_plateau_series(df)


def _slope_kg_per_day(data: pd.DataFrame) -> float | None:
    data = prepare_plateau_series(data)
    if len(data) < 2:
        return None
    x_days = (data["Date"] - data["Date"].min()).dt.total_seconds() / 86400
    if float(x_days.max()) <= 0:
        return None
    return float(np.polyfit(x_days, data["Poids (Kgs)"], 1)[0])


def detect_plateau(df: pd.DataFrame, window: int = 14) -> dict[str, object]:
    """Détecte stagnation/plateau sur une fenêtre calendaire réelle."""
    config = StagnationConfig(window_days=window)
    data = _prepare_time_series(df)
    if data.empty:
        return evaluate_plateau_window(data, config)
    cutoff = data["Date"].max() - pd.Timedelta(days=config.window_days)
    return evaluate_plateau_window(data[data["Date"] >= cutoff], config)

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


def estimate_target_eta(df: pd.DataFrame, target_weight: float, effort_df: pd.DataFrame | None = None) -> dict[str, object]:
    if len(_prepare_time_series(df)) < 7:
        return {"credible": False, "message": "Données insuffisantes"}
    data = _prepare_time_series(df)
    target_weight = max(float(target_weight), FINAL_TARGET_WEIGHT_KG)
    current = float(data["Poids (Kgs)"].iloc[-1])
    last_date = data["Date"].max()

    if current <= target_weight:
        return {"credible": True, "message": "Objectif déjà atteint !", "eta": last_date,
                "eta_min": last_date, "eta_max": last_date, "confidence": 1.0, "scenarios": {}}

    # Multi-scenario ETA based on different calendar-day windows
    scenarios = {}
    for name, window in [("optimiste", 7), ("réaliste", 30), ("pessimiste", 90)]:
        cutoff = last_date - pd.Timedelta(days=window)
        subset = data[data["Date"] >= cutoff]
        if len(subset) < 3:
            continue
        slope = _slope_kg_per_day(subset)
        if slope is None:
            continue
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

    # Primary estimate: prefer effort period if available, else 30 calendar days
    primary_data = None
    primary_source = "30j"
    if effort_df is not None and len(_prepare_time_series(effort_df)) >= 3:
        primary_data = _prepare_time_series(effort_df)
        primary_source = "effort"
    else:
        cutoff_30 = last_date - pd.Timedelta(days=30)
        recent = data[data["Date"] >= cutoff_30]
        if len(recent) >= 3:
            primary_data = recent
        else:
            primary_data = data.tail(10)  # fallback

    if primary_data is None or len(primary_data) < 3:
        return {
            "credible": False,
            "message": "Données insuffisantes pour l'estimation primaire.",
            "scenarios": scenarios,
        }

    slope = _slope_kg_per_day(primary_data)
    if slope is None:
        return {"credible": False, "message": "Durée insuffisante pour calculer une projection.", "scenarios": scenarios}

    # Garde-fou 1 : tendance insuffisante
    if slope >= -0.005:
        return {
            "credible": False,
            "message": f"La tendance ({primary_source}) est de {slope*7:+.3f} kg/sem — insuffisante pour une estimation fiable.",
            "slope_30d": round(slope, 4),
            "scenarios": scenarios,
        }

    # Garde-fou 2 : si < 7 mesures dans l'effort, signal trop fragile
    if effort_df is not None and len(_prepare_time_series(effort_df)) < 7:
        return {
            "credible": False,
            "message": f"Phase de démarrage ({len(_prepare_time_series(effort_df))} mesures) — l'ETA sera fiable à partir de 7 mesures.",
            "slope_30d": round(slope, 4),
            "source": primary_source,
            "scenarios": scenarios,
        }

    # Garde-fou 3 : si pente > 2 kg/sem, c'est du bruit (perte hydrique)
    kg_per_week = abs(slope * 7)
    if kg_per_week > 2.0:
        # Plafonner à 0.75 kg/sem pour un ETA réaliste
        capped_slope = -0.75 / 7
        remaining = current - target_weight
        days = int(remaining / abs(capped_slope))
        eta = last_date + pd.Timedelta(days=max(days, 0))
        return {
            "credible": True,
            "eta": eta,
            "eta_min": eta - pd.Timedelta(days=30),
            "eta_max": eta + pd.Timedelta(days=60),
            "confidence": 0.3,
            "message": f"Vitesse actuelle ({kg_per_week:.1f} kg/sem) probablement temporaire. ETA basé sur un rythme réaliste de 0.75 kg/sem.",
            "slope_30d": round(slope, 4),
            "source": primary_source,
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
        "source": primary_source,
        "scenarios": scenarios,
    }


