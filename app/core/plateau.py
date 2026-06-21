from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.business import StagnationConfig
from app.core.time_utils import normalize_datetime_series


def prepare_plateau_series(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Date" not in df.columns or "Poids (Kgs)" not in df.columns:
        return pd.DataFrame(columns=["Date", "Poids (Kgs)"])
    out = df[["Date", "Poids (Kgs)"]].copy()
    out["Date"] = normalize_datetime_series(out["Date"], normalize_day=False)
    out["Poids (Kgs)"] = pd.to_numeric(out["Poids (Kgs)"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out = out.dropna(subset=["Date", "Poids (Kgs)"]).sort_values("Date")
    return out.reset_index(drop=True)


def evaluate_plateau_window(df: pd.DataFrame, config: StagnationConfig | None = None) -> dict[str, Any]:
    config = config or StagnationConfig()
    data = prepare_plateau_series(df)
    base = {"status": "indisponible", "is_plateau": False, "parameters": config.__dict__}
    if data.empty:
        return base | {"reason": "aucune mesure", "nb_mesures": 0, "slope": 0.0, "slope_kg_week": 0.0, "slope_kg_day": 0.0}
    if len(data) < config.min_measurements:
        return base | {"reason": "mesures insuffisantes", "nb_mesures": len(data), "slope": 0.0, "slope_kg_week": 0.0, "slope_kg_day": 0.0}
    duration = (data["Date"].iloc[-1] - data["Date"].iloc[0]).total_seconds() / 86400
    if duration <= 0:
        return base | {"reason": "durée insuffisante", "nb_mesures": len(data), "slope": 0.0, "slope_kg_week": 0.0, "slope_kg_day": 0.0}
    x_days = (data["Date"] - data["Date"].min()).dt.total_seconds() / 86400
    slope_day = float(np.polyfit(x_days, data["Poids (Kgs)"], 1)[0])
    slope_week = slope_day * 7
    variation = float(data["Poids (Kgs)"].iloc[-1] - data["Poids (Kgs)"].iloc[0])
    amplitude = float(data["Poids (Kgs)"].max() - data["Poids (Kgs)"].min())
    if slope_week <= -config.max_abs_slope_kg_per_week or variation <= -config.max_amplitude_kg:
        status = "baisse active"
    elif abs(slope_week) <= config.max_abs_slope_kg_per_week and amplitude <= config.max_amplitude_kg:
        status = "plateau probable"
    elif slope_week >= config.max_abs_slope_kg_per_week:
        status = "reprise de poids probable"
    else:
        status = "signal mixte"
    return {
        "status": status,
        "is_plateau": status == "plateau probable",
        "reason": None,
        "period_start": data["Date"].iloc[0],
        "period_end": data["Date"].iloc[-1],
        "period_days": duration,
        "nb_mesures": len(data),
        "variation_kg": variation,
        "amplitude": amplitude,
        "volatility": float(data["Poids (Kgs)"].std()),
        "slope_kg_day": slope_day,
        "slope": slope_week,
        "slope_kg_week": slope_week,
        "confidence": "moyenne" if status != "signal mixte" else "faible",
        "parameters": config.__dict__,
    }
