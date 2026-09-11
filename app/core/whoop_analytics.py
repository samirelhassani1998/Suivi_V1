"""Analyses croisées WHOOP × poids.

L'application WHOOP décrit la récupération, le sommeil et la charge isolément.
Ce module exploite ce qu'elle ne possède pas : la série de poids. Il en tire un
bilan énergétique, des corrélations décalées, une charge d'entraînement
aigüe/chronique et un récapitulatif hebdomadaire.

Toutes les fonctions sont pures et refusent explicitement de conclure sur un
échantillon trop faible : chaque résultat porte son effectif et son statut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from app.core.whoop import available_metrics

# Équivalent énergétique d'un kilogramme de masse corporelle perdue ou prise.
# Valeur de référence usuelle en nutrition (~7700 kcal pour 1 kg de tissu adipeux).
KCAL_PER_KG = 7700.0

# Zones de récupération telles que définies par WHOOP.
RECOVERY_ZONES: tuple[tuple[str, float, float, str], ...] = (
    ("Rouge", 0.0, 34.0, "récupération insuffisante"),
    ("Jaune", 34.0, 67.0, "récupération modérée"),
    ("Vert", 67.0, 100.01, "récupération élevée"),
)

# Effectifs minimaux avant d'afficher une analyse.
MIN_DAYS_CORRELATION = 10
MIN_DAYS_ENERGY_BALANCE = 14
MIN_DAYS_TRAINING_LOAD = 7
MIN_DAYS_REGRESSION = 12
CHRONIC_LOAD_DAYS = 28
ACUTE_LOAD_DAYS = 7


@dataclass(frozen=True)
class Availability:
    """Disponibilité d'une analyse et nombre de jours encore nécessaires."""

    name: str
    available: int
    required: int

    @property
    def ready(self) -> bool:
        return self.available >= self.required

    @property
    def missing(self) -> int:
        return max(0, self.required - self.available)


def _clean_daily(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "Date" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy(deep=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    return data.dropna(subset=["Date"]).sort_values("Date", kind="mergesort").reset_index(drop=True)


def daily_grid(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Réindexe sur un calendrier continu, en laissant les jours manquants vides.

    Sans cette étape, un graphique relie deux mesures espacées de six jours par
    une droite qui donne l'illusion d'une tendance mesurée.
    """
    data = _clean_daily(frame)
    if data.empty:
        return data
    full_range = pd.date_range(data["Date"].min(), data["Date"].max(), freq="D")
    return (
        data.set_index("Date")
        .reindex(full_range)
        .rename_axis("Date")
        .reset_index()
    )


def coverage_report(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Couverture réelle de la période importée."""
    data = _clean_daily(frame)
    if data.empty:
        return {"days_with_data": 0, "span_days": 0, "coverage_pct": 0.0, "gaps": 0, "start": None, "end": None}
    start, end = data["Date"].min(), data["Date"].max()
    span = int((end - start).days) + 1
    measured = int(data["Date"].nunique())
    return {
        "days_with_data": measured,
        "span_days": span,
        "coverage_pct": round(measured / span * 100, 1) if span else 0.0,
        "gaps": max(0, span - measured),
        "start": start,
        "end": end,
    }


def analysis_availability(daily: pd.DataFrame | None, merged: pd.DataFrame | None = None) -> list[Availability]:
    """Ce qui est déjà calculable et ce qui attend encore des jours de mesure."""
    whoop_days = coverage_report(daily)["days_with_data"]
    common_days = int(len(_clean_daily(merged))) if merged is not None else 0
    return [
        Availability("Charge d'entraînement", whoop_days, MIN_DAYS_TRAINING_LOAD),
        Availability("Moteurs de la récupération", whoop_days, MIN_DAYS_REGRESSION),
        Availability("Corrélations poids", common_days, MIN_DAYS_CORRELATION),
        Availability("Bilan énergétique", common_days, MIN_DAYS_ENERGY_BALANCE),
    ]


def recovery_zones(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Répartition dans les zones de récupération WHOOP (rouge / jaune / vert)."""
    data = _clean_daily(frame)
    empty = {"counts": pd.DataFrame(columns=["Zone", "Jours", "Part (%)", "Lecture"]), "latest": None, "latest_zone": None, "days": 0}
    if data.empty or "Récupération (%)" not in data.columns:
        return empty
    values = data["Récupération (%)"].dropna()
    if values.empty:
        return empty

    rows = []
    for label, low, high, reading in RECOVERY_ZONES:
        count = int(((values >= low) & (values < high)).sum())
        rows.append({"Zone": label, "Jours": count, "Part (%)": round(count / len(values) * 100, 1), "Lecture": reading})

    latest_value = float(values.iloc[-1])
    latest_zone = next((label for label, low, high, _ in RECOVERY_ZONES if low <= latest_value < high), None)
    return {"counts": pd.DataFrame(rows), "latest": latest_value, "latest_zone": latest_zone, "days": int(len(values))}


def training_load(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Charge aigüe (7 j) rapportée à la charge chronique (28 j).

    Ce rapport est un indicateur classique de gestion de charge en préparation
    physique : au-delà de 1,5 la progression est brutale, en dessous de 0,8 la
    charge s'effondre. WHOOP affiche le strain du jour, pas cette dynamique.
    """
    data = _clean_daily(frame)
    result = {"acute": float("nan"), "chronic": float("nan"), "ratio": float("nan"), "status": "indisponible", "days": 0}
    if data.empty or "Strain" not in data.columns:
        return result
    grid = daily_grid(data)
    strain = grid["Strain"]
    measured = int(strain.notna().sum())
    result["days"] = measured
    if measured < MIN_DAYS_TRAINING_LOAD:
        return result

    acute = float(strain.tail(ACUTE_LOAD_DAYS).mean(skipna=True))
    chronic = float(strain.tail(CHRONIC_LOAD_DAYS).mean(skipna=True))
    result["acute"] = acute
    result["chronic"] = chronic
    if not np.isfinite(chronic) or chronic <= 0:
        return result

    ratio = acute / chronic
    result["ratio"] = ratio
    if ratio > 1.5:
        result["status"] = "montée en charge brutale"
    elif ratio >= 1.3:
        result["status"] = "montée en charge soutenue"
    elif ratio >= 0.8:
        result["status"] = "charge maîtrisée"
    else:
        result["status"] = "charge en retrait"
    return result


def sleep_debt_summary(frame: pd.DataFrame | None, days: int = 7) -> dict[str, Any]:
    """Dette de sommeil cumulée sur la période récente."""
    data = _clean_daily(frame)
    result = {"nights": 0, "cumulative_debt": float("nan"), "mean_debt": float("nan"), "mean_sleep": float("nan"), "mean_need": float("nan")}
    if data.empty or "Dette de sommeil (heures)" not in data.columns:
        return result
    recent = data.tail(max(1, int(days)))
    debt = recent["Dette de sommeil (heures)"].dropna()
    if debt.empty:
        return result
    result["nights"] = int(len(debt))
    result["cumulative_debt"] = float(debt.sum())
    result["mean_debt"] = float(debt.mean())
    if "Sommeil (heures)" in recent.columns:
        result["mean_sleep"] = float(recent["Sommeil (heures)"].mean(skipna=True))
    if "Besoin de sommeil (heures)" in recent.columns:
        result["mean_need"] = float(recent["Besoin de sommeil (heures)"].mean(skipna=True))
    return result


def weight_trend(merged: pd.DataFrame | None) -> dict[str, Any]:
    """Pente du poids estimée par moindres carrés sur les jours communs."""
    data = _clean_daily(merged)
    result = {"slope_kg_per_day": float("nan"), "slope_kg_per_week": float("nan"), "days": 0, "span_days": 0, "r_squared": float("nan")}
    if data.empty or "Poids (Kgs)" not in data.columns:
        return result
    series = data[["Date", "Poids (Kgs)"]].dropna()
    if len(series) < 3:
        result["days"] = int(len(series))
        return result

    x = (series["Date"] - series["Date"].min()).dt.days.to_numpy(dtype=float)
    y = series["Poids (Kgs)"].to_numpy(dtype=float)
    if np.ptp(x) == 0:
        result["days"] = int(len(series))
        return result

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    result.update(
        {
            "slope_kg_per_day": float(slope),
            "slope_kg_per_week": float(slope) * 7.0,
            "days": int(len(series)),
            "span_days": int(np.ptp(x)) + 1,
            "r_squared": 1.0 - residual / total if total > 0 else float("nan"),
        }
    )
    return result


def energy_balance(merged: pd.DataFrame | None, *, kcal_per_kg: float = KCAL_PER_KG) -> dict[str, Any]:
    """Apport calorique implicite, déduit de la dépense WHOOP et de la pente du poids.

    WHOOP mesure la dépense quotidienne mais ignore le poids ; la balance ignore
    la dépense. Croiser les deux donne une estimation de l'apport moyen, que ni
    l'une ni l'autre ne peut produire seule.
    """
    data = _clean_daily(merged)
    result = {
        "ready": False,
        "days": int(len(data)),
        "required_days": MIN_DAYS_ENERGY_BALANCE,
        "mean_burn": float("nan"),
        "imbalance_per_day": float("nan"),
        "estimated_intake": float("nan"),
        "slope_kg_per_week": float("nan"),
    }
    if data.empty or "Calories (kcal)" not in data.columns:
        return result

    burn = data["Calories (kcal)"].dropna()
    trend = weight_trend(data)
    usable_days = int(min(len(burn), trend["days"]))
    result["days"] = usable_days
    result["mean_burn"] = float(burn.mean()) if not burn.empty else float("nan")
    result["slope_kg_per_week"] = trend["slope_kg_per_week"]

    if usable_days < MIN_DAYS_ENERGY_BALANCE or not np.isfinite(trend["slope_kg_per_day"]) or not np.isfinite(result["mean_burn"]):
        return result

    imbalance = trend["slope_kg_per_day"] * float(kcal_per_kg)
    result["imbalance_per_day"] = imbalance
    result["estimated_intake"] = result["mean_burn"] + imbalance
    result["ready"] = True
    return result


def lagged_correlations(
    merged: pd.DataFrame | None,
    *,
    target: str = "Variation poids (kg)",
    lags: Sequence[int] = (0, 1, 2),
    min_pairs: int = MIN_DAYS_CORRELATION,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Corrélations entre une métrique décalée de *k* jours et la cible.

    Un entraînement intense pèse rarement sur la balance le jour même : tester
    plusieurs décalages évite de conclure à une absence de lien trop vite.
    """
    columns = ["Métrique", "Décalage (jours)", "Corrélation", "Observations", "Lecture"]
    grid = daily_grid(merged)
    if grid.empty or target not in grid.columns:
        return pd.DataFrame(columns=columns)

    candidates = list(metrics) if metrics is not None else available_metrics(grid)
    rows = []
    for metric in candidates:
        if metric not in grid.columns:
            continue
        for lag in lags:
            shifted = grid[metric].shift(int(lag))
            pair = pd.DataFrame({"target": grid[target], "metric": shifted}).dropna()
            if len(pair) < max(3, int(min_pairs)):
                continue
            if pair["target"].nunique() < 2 or pair["metric"].nunique() < 2:
                continue
            correlation = float(pair["target"].corr(pair["metric"]))
            if not np.isfinite(correlation):
                continue
            magnitude = abs(correlation)
            strength = "forte" if magnitude >= 0.5 else "modérée" if magnitude >= 0.3 else "faible"
            direction = "même sens" if correlation > 0 else "sens opposé"
            rows.append(
                {
                    "Métrique": metric,
                    "Décalage (jours)": int(lag),
                    "Corrélation": round(correlation, 3),
                    "Observations": int(len(pair)),
                    "Lecture": f"association {strength}, {direction}",
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame.reindex(frame["Corrélation"].abs().sort_values(ascending=False).index).reset_index(drop=True)[columns]


def recovery_drivers(frame: pd.DataFrame | None, *, min_days: int = MIN_DAYS_REGRESSION) -> dict[str, Any]:
    """Régression de la récupération sur le sommeil et la charge de la veille.

    Répond à une question que l'application WHOOP laisse à l'intuition :
    combien de points de récupération rapporte une heure de sommeil de plus.
    """
    result = {"ready": False, "days": 0, "required_days": int(min_days), "coefficients": {}, "r_squared": float("nan")}
    grid = daily_grid(frame)
    if grid.empty or "Récupération (%)" not in grid.columns:
        return result

    predictors: dict[str, pd.Series] = {}
    if "Sommeil (heures)" in grid.columns:
        predictors["Sommeil (heures)"] = grid["Sommeil (heures)"]
    if "Strain" in grid.columns:
        predictors["Strain de la veille"] = grid["Strain"].shift(1)
    if not predictors:
        return result

    design = pd.DataFrame({"Récupération (%)": grid["Récupération (%)"], **predictors}).dropna()
    result["days"] = int(len(design))
    # Il faut nettement plus d'observations que de paramètres pour que les
    # coefficients aient un sens ; sinon la régression interpole le bruit.
    if len(design) < max(int(min_days), len(predictors) + 3):
        return result

    x = np.column_stack([np.ones(len(design))] + [design[name].to_numpy(dtype=float) for name in predictors])
    y = design["Récupération (%)"].to_numpy(dtype=float)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return result

    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    predicted = x @ coefficients
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    result["coefficients"] = {name: float(coefficients[index + 1]) for index, name in enumerate(predictors)}
    result["r_squared"] = 1.0 - residual / total if total > 0 else float("nan")
    result["ready"] = True
    return result


def weekly_rollup(merged: pd.DataFrame | None) -> pd.DataFrame:
    """Synthèse hebdomadaire : récupération, sommeil, charge et variation de poids."""
    columns = ["Semaine", "Jours", "Récupération (%)", "Sommeil (heures)", "Strain cumulé", "Poids moyen (kg)", "Variation (kg)"]
    data = _clean_daily(merged)
    if data.empty:
        return pd.DataFrame(columns=columns)

    data = data.assign(_week=data["Date"].dt.to_period("W").dt.start_time)
    rows = []
    for week, chunk in data.groupby("_week", sort=True):
        weights = chunk["Poids (Kgs)"].dropna() if "Poids (Kgs)" in chunk.columns else pd.Series(dtype=float)
        rows.append(
            {
                "Semaine": week,
                "Jours": int(len(chunk)),
                "Récupération (%)": float(chunk["Récupération (%)"].mean(skipna=True)) if "Récupération (%)" in chunk.columns else float("nan"),
                "Sommeil (heures)": float(chunk["Sommeil (heures)"].mean(skipna=True)) if "Sommeil (heures)" in chunk.columns else float("nan"),
                "Strain cumulé": float(chunk["Strain"].sum(skipna=True)) if "Strain" in chunk.columns else float("nan"),
                "Poids moyen (kg)": float(weights.mean()) if not weights.empty else float("nan"),
                # Écart entre la première et la dernière pesée de la semaine.
                "Variation (kg)": float(weights.iloc[-1] - weights.iloc[0]) if len(weights) >= 2 else float("nan"),
            }
        )
    return pd.DataFrame(rows)[columns]


def weekday_profile(frame: pd.DataFrame | None, metric: str) -> pd.DataFrame:
    """Moyenne d'une métrique par jour de semaine, pour repérer les creux récurrents."""
    columns = ["Jour", "Moyenne", "Observations"]
    data = _clean_daily(frame)
    if data.empty or metric not in data.columns:
        return pd.DataFrame(columns=columns)
    values = data[["Date", metric]].dropna()
    if values.empty:
        return pd.DataFrame(columns=columns)

    labels = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    values = values.assign(_dow=values["Date"].dt.dayofweek)
    grouped = values.groupby("_dow")[metric].agg(["mean", "count"]).reindex(range(7))
    return pd.DataFrame(
        {
            "Jour": labels,
            "Moyenne": grouped["mean"].to_numpy(),
            "Observations": grouped["count"].fillna(0).astype(int).to_numpy(),
        }
    )[columns]


def strain_recovery_balance(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Repère les jours où une charge élevée suit une récupération basse.

    C'est le signal que l'application WHOOP donne a posteriori ; le calculer ici
    permet de le relier ensuite à la courbe de poids.
    """
    columns = ["Date", "Récupération (%)", "Strain", "Signal"]
    grid = daily_grid(frame)
    if grid.empty or "Strain" not in grid.columns or "Récupération (%)" not in grid.columns:
        return pd.DataFrame(columns=columns)

    usable = grid[["Date", "Récupération (%)", "Strain"]].dropna()
    if usable.empty:
        return pd.DataFrame(columns=columns)

    median_strain = float(usable["Strain"].median())
    signals = []
    for _, row in usable.iterrows():
        recovery, strain = float(row["Récupération (%)"]), float(row["Strain"])
        if recovery < 34 and strain > median_strain:
            signal = "charge élevée sur récupération basse"
        elif recovery >= 67 and strain < median_strain:
            signal = "récupération élevée sous-exploitée"
        else:
            signal = "cohérent"
        signals.append(signal)
    return usable.assign(Signal=signals)[columns].reset_index(drop=True)
