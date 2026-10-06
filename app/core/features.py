"""Feature engineering pour les modèles temporels.

Toutes les variables dérivées du poids sont **décalées d'au moins une mesure** :
une moyenne glissante qui inclut la pesée du jour, ou l'IMC du jour, contient
la cible elle-même et donne des modèles au R² de 1,000 qui ne prédisent rien.
Seules les variables issues du poids passé et du calendrier sont autorisées.
Les colonnes importées (IMC courant, calories, notes…) ne sont pas des variables
prédictives tant que leur disponibilité avant la prévision n'est pas définie.
"""

from __future__ import annotations

import pandas as pd

LAGS: tuple[int, ...] = (1, 3, 7, 14, 30)
WINDOWS: tuple[int, ...] = (7, 14, 30)


def build_features(df: pd.DataFrame, height_m: float | None = None) -> pd.DataFrame:
    # Une colonne numérique importée peut être la cible elle-même (IMC courant,
    # copie du poids…) : ne jamais l'admettre implicitement dans les modèles.
    data = df[["Date", "Poids (Kgs)"]].sort_values("Date", kind="mergesort").copy()
    weight = pd.to_numeric(data["Poids (Kgs)"], errors="coerce")
    previous = weight.shift(1)

    data["jour_semaine"] = data["Date"].dt.weekday
    calendar_days = data["Date"].dt.tz_localize(None).dt.normalize()
    data["jours_depuis_derniere_mesure"] = calendar_days.diff().dt.days.fillna(0)
    # Variation entre les deux pesées précédentes : connue au moment de prédire.
    data["variation_precedente"] = weight.diff().shift(1).fillna(0)

    for lag in LAGS:
        data[f"lag_{lag}"] = weight.shift(lag)

    for window in WINDOWS:
        data[f"roll_mean_{window}"] = previous.rolling(window, min_periods=1).mean()
        data[f"roll_std_{window}"] = previous.rolling(window, min_periods=2).std().fillna(0)

    if height_m and height_m > 0:
        # IMC de la pesée précédente : le même jour, ce serait la cible divisée par une constante.
        data["imc_precedent"] = previous / (height_m**2)

    return data
