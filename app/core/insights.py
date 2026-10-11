"""Insights analytiques: plateau, anomalies, ETA objectif."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.business import FINAL_TARGET_WEIGHT_KG, INTERRUPTION_THRESHOLD_DAYS, StagnationConfig
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
    """Extrapolations par fenêtre observée, sans probabilité de réussite inventée.

    Les libellés décrivent la durée analysée : une pente de 7 jours n'est pas
    systématiquement plus optimiste qu'une pente de 90 jours. Aucun historique
    ancien ne remplace une fenêtre récente trop courte, et aucune vitesse
    supposée ne remplace une vitesse qui dépasse le garde-fou existant.
    """
    data = _prepare_time_series(df)
    if data.empty:
        return {"credible": False, "message": "Données insuffisantes", "scenarios": {}}
    target_weight = max(float(target_weight), FINAL_TARGET_WEIGHT_KG)
    current = float(data["Poids (Kgs)"].iloc[-1])
    last_date = data["Date"].iloc[-1]
    if current <= target_weight:
        return {"credible": True, "message": "Objectif déjà atteint !", "eta": last_date, "scenarios": {}}

    # A restart is a new observation period. Earlier slopes cannot describe it.
    restarts = data.index[data["Date"].diff().dt.days > INTERRUPTION_THRESHOLD_DAYS]
    if len(restarts):
        data = data.loc[restarts[-1]:].copy()
    if len(data) < 7:
        return {"credible": False, "message": "Au moins 7 mesures dans la période de suivi actuelle sont nécessaires.", "scenarios": {}}

    def estimate_window(subset: pd.DataFrame) -> dict[str, object]:
        n = len(subset)
        span = (subset["Date"].max() - subset["Date"].min()).days if n else 0
        base = {"credible": False, "measurements": n, "span_days": span}
        if n < 3 or span < 7:
            return {**base, "message": "Recul insuffisant : au moins 3 mesures réparties sur 7 jours."}
        slope = _slope_kg_per_day(subset)
        if slope is None or not np.isfinite(slope):
            return {**base, "message": "Pente indisponible."}
        base["slope"] = round(slope, 4)
        if slope >= -0.005:
            return {**base, "message": "La tendance ne va pas clairement vers l'objectif."}
        if abs(slope * 7) > 2.0:
            return {**base, "message": "Rythme supérieur au garde-fou de projection (2 kg/sem) : non extrapolé."}
        days = math.ceil((current - target_weight) / abs(slope))
        if days > 3 * 365:
            return {**base, "message": "Échéance au-delà de trois ans : non affichée."}
        return {**base, "credible": True, "eta": last_date + pd.Timedelta(days=days),
                "days_remaining": days, "kg_per_week": round(slope * 7, 3),
                "message": "Extrapolation de la pente observée si elle se maintient ; ce n'est pas une prédiction."}

    scenarios = {
        f"{window} jours": estimate_window(data[data["Date"] >= last_date - pd.Timedelta(days=window)])
        for window in (7, 30, 90)
    }
    primary_source = "30j"
    recent = data[data["Date"] >= last_date - pd.Timedelta(days=30)]
    if effort_df is not None:
        effort = _prepare_time_series(effort_df)
        effort = effort[(effort["Date"] >= data["Date"].iloc[0]) & (effort["Date"] <= last_date)]
        if len(effort) < 7:
            return {"credible": False, "message": "Phase de démarrage : au moins 7 mesures dans l'effort sont nécessaires.", "scenarios": scenarios}
        # Use the same recent horizon and last observation as the main series.
        recent = effort[effort["Date"] >= last_date - pd.Timedelta(days=30)]
        if recent.empty or recent["Date"].iloc[-1] != last_date:
            return {"credible": False, "message": "La période d'effort ne contient pas la dernière mesure.", "scenarios": scenarios}
        primary_source = "effort (30j max)"

    primary = estimate_window(recent)
    return {**primary, "slope_30d": primary.get("slope"), "source": primary_source, "scenarios": scenarios}
