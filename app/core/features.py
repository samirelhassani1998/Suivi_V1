"""Feature engineering pour les modèles temporels."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame, height_m: float | None = None) -> pd.DataFrame:
    data = df.sort_values("Date").copy()
    data["jour_semaine"] = data["Date"].dt.weekday
    data["jours_depuis_derniere_mesure"] = data["Date"].diff().dt.days.fillna(0)
    data["variation_journaliere"] = data["Poids (Kgs)"].diff().fillna(0)

    for lag in (1, 3, 7, 14, 30):
        data[f"lag_{lag}"] = data["Poids (Kgs)"].shift(lag)

    for w in (7, 14, 30):
        data[f"roll_mean_{w}"] = data["Poids (Kgs)"].rolling(w, min_periods=1).mean()
        data[f"roll_std_{w}"] = data["Poids (Kgs)"].rolling(w, min_periods=2).std().fillna(0)

    if height_m and height_m > 0:
        data["imc"] = data["Poids (Kgs)"] / (height_m**2)

    if {"Calories consommées", "Calories brûlées"}.issubset(data.columns):
        data["bilan_calorique"] = (
            pd.to_numeric(data["Calories consommées"], errors="coerce").fillna(0)
            - pd.to_numeric(data["Calories brûlées"], errors="coerce").fillna(0)
        )
    return data
