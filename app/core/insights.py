"""Insights analytiques: plateau, anomalies, ETA objectif."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.business import FINAL_TARGET_WEIGHT_KG, StagnationConfig
from app.core.plateau import evaluate_plateau_window, prepare_plateau_series
from app.core.trend import MIN_POINTS_TREND, trend_weight


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

ANOMALY_Z_THRESHOLD = 3.5


def _trend_reference(out: pd.DataFrame) -> pd.Series:
    """Valeur de référence par ligne : la tendance robuste quand elle existe, sinon la médiane.

    Comparer chaque pesée à la médiane de toute la série signalait comme
    « anormales » les premières et dernières pesées d'une perte régulière : sur
    six mois de baisse, les extrêmes sont la tendance elle-même, pas un écart.
    """
    series = pd.to_numeric(out["Poids (Kgs)"], errors="coerce")
    reference = pd.Series(float(series.median()), index=out.index)
    if "Date" not in out.columns or len(out) < MIN_POINTS_TREND:
        return reference
    frame = trend_weight(out)
    if frame.empty:
        return reference
    lookup = frame.set_index("Date")["Tendance"]
    dates = pd.to_datetime(out["Date"], errors="coerce")
    mapped = dates.map(lookup)
    return mapped.where(mapped.notna(), reference).astype(float)


def detect_anomalies_robust(df: pd.DataFrame, use_iforest: bool = False, threshold: float = ANOMALY_Z_THRESHOLD) -> pd.DataFrame:
    """Pesées atypiques : z-score robuste (Iglewicz & Hoaglin) des écarts à la tendance."""
    out = df.copy()
    if out.empty:
        out["anomalie"] = False
        out["raison"] = "aucune donnée"
        out["decision"] = "conservée"
        return out
    series = pd.to_numeric(out["Poids (Kgs)"], errors="coerce")
    residual = series - _trend_reference(out)
    center = float(np.nanmedian(residual))
    mad = float(np.nanmedian(np.abs(residual - center))) + 1e-9
    z = 0.6745 * (residual - center) / mad
    out["ecart_tendance"] = residual
    out["z_robuste"] = z
    out["anomalie"] = np.abs(z) > threshold
    out["raison"] = np.where(out["anomalie"], f"écart à la tendance : z robuste > {threshold}", "normal")

    if use_iforest and len(out) >= 10:
        iso = IsolationForest(contamination=0.1, random_state=42)
        features = residual.fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
        out["iforest"] = iso.fit_predict(features) == -1
        out["anomalie"] = out["anomalie"] | out["iforest"]
        out.loc[out["iforest"] & (out["raison"] == "normal"), "raison"] = "IsolationForest (écart à la tendance)"
    out["decision"] = np.where(out["anomalie"], "à revoir", "conservée")
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


