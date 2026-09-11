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


# ──────────────────────────────────────────────────────────────────────────────
# Repères personnels, lissage et mise en base commune
# ──────────────────────────────────────────────────────────────────────────────

# Fenêtre servant de référence personnelle, comme le fait WHOOP pour la HRV.
BASELINE_DAYS = 30
MIN_DAYS_BASELINE = 10


def personal_baseline(frame: pd.DataFrame | None, metric: str, *, days: int = BASELINE_DAYS) -> dict[str, Any]:
    """Compare la dernière valeur au repère personnel plutôt qu'à une norme.

    Une HRV de 40 ms ne veut rien dire dans l'absolu : ce qui compte est l'écart
    à votre propre habitude. La médiane est préférée à la moyenne car une seule
    nuit blanche ne doit pas déplacer le repère.
    """
    result = {"ready": False, "metric": metric, "latest": float("nan"), "baseline": float("nan"), "deviation_pct": float("nan"), "days": 0}
    data = _clean_daily(frame)
    if data.empty or metric not in data.columns:
        return result

    series = data[metric].dropna()
    result["days"] = int(len(series))
    if len(series) < MIN_DAYS_BASELINE:
        return result

    # Le repère exclut la dernière mesure : sinon elle se compare à elle-même.
    history = series.iloc[:-1].tail(max(1, int(days)))
    baseline = float(history.median())
    latest = float(series.iloc[-1])
    result.update({"latest": latest, "baseline": baseline})
    if not np.isfinite(baseline) or baseline == 0:
        return result
    result["deviation_pct"] = (latest - baseline) / abs(baseline) * 100.0
    result["ready"] = True
    return result


def rolling_trend(frame: pd.DataFrame | None, metric: str, *, window: int = 7) -> pd.DataFrame:
    """Moyenne glissante calendaire, pour lire la tendance sous le bruit quotidien."""
    grid = daily_grid(frame)
    if grid.empty or metric not in grid.columns:
        return pd.DataFrame(columns=["Date", metric])
    smoothed = grid[metric].rolling(window=max(2, int(window)), min_periods=max(2, int(window) // 2)).mean()
    return pd.DataFrame({"Date": grid["Date"], metric: smoothed})


def indexed_series(frame: pd.DataFrame | None, metrics: Sequence[str], *, base: float = 100.0) -> pd.DataFrame:
    """Ramène plusieurs séries à une base commune pour les lire sur un seul axe.

    Superposer un poids (kg) et une récupération (%) sur deux axes verticaux
    fabrique une corrélation visuelle arbitraire : le calage des deux échelles
    est un choix, pas une donnée. La mise en base commune supprime ce biais.
    """
    grid = daily_grid(frame)
    columns = ["Date", *metrics]
    if grid.empty:
        return pd.DataFrame(columns=columns)

    output = pd.DataFrame({"Date": grid["Date"]})
    for metric in metrics:
        if metric not in grid.columns:
            continue
        series = grid[metric]
        first_valid = series.first_valid_index()
        if first_valid is None:
            continue
        reference = float(series.loc[first_valid])
        if not np.isfinite(reference) or reference == 0:
            continue
        output[metric] = series / reference * float(base)
    return output


def calendar_matrix(frame: pd.DataFrame | None, metric: str) -> dict[str, Any]:
    """Matrice semaine × jour de semaine, pour lire un mois d'un seul coup d'œil."""
    empty = {"values": [], "weeks": [], "weekdays": [], "dates": []}
    grid = daily_grid(frame)
    if grid.empty or metric not in grid.columns or not grid[metric].notna().any():
        return empty

    data = grid[["Date", metric]].copy()
    data["_week"] = data["Date"].dt.to_period("W").dt.start_time
    data["_dow"] = data["Date"].dt.dayofweek

    weeks = sorted(data["_week"].unique())
    labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    lookup: dict[tuple[Any, int], tuple[float, pd.Timestamp]] = {
        (week, dow): (value, date)
        for week, dow, value, date in zip(data["_week"], data["_dow"], data[metric], data["Date"])
    }

    values, dates = [], []
    for week in weeks:
        value_row, date_row = [], []
        for dow in range(7):
            value, date = lookup.get((week, dow), (float("nan"), None))
            numeric = float(value) if value is not None else float("nan")
            # Plotly attend None (et non NaN) pour laisser une case vide.
            value_row.append(numeric if np.isfinite(numeric) else None)
            date_row.append(pd.Timestamp(date) if date is not None else None)
        values.append(value_row)
        dates.append(date_row)

    return {
        "values": values,
        "weeks": [pd.Timestamp(week) for week in weeks],
        "weekdays": labels,
        "dates": dates,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Lecture narrative
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Insight:
    """Constat rédigé, porteur de son chiffre et de son niveau d'alerte."""

    title: str
    body: str
    tone: str = "info"
    icon: str = "💡"
    priority: int = 50


def _fr(value: float, decimals: int = 1, *, sign: bool = False) -> str:
    """Nombre au format français, sans dépendre de la couche d'affichage."""
    if value is None or not np.isfinite(value):
        return "—"
    prefix = ""
    if sign:
        prefix = "+" if value > 0 else "−" if value < 0 else ""
        value = abs(value)
    return f"{prefix}{value:.{decimals}f}".replace(".", ",")


def _coverage_insight(daily: pd.DataFrame | None) -> Insight | None:
    coverage = coverage_report(daily)
    if coverage["span_days"] < 7 or coverage["coverage_pct"] >= 80:
        return None
    return Insight(
        "Port du bracelet irrégulier",
        f"{coverage['gaps']} jour(s) sans mesure sur {coverage['span_days']}, soit "
        f"{_fr(coverage['coverage_pct'], 0)} % de couverture. Les moyennes et les tendances "
        "portent donc sur une base incomplète.",
        tone="warning",
        icon="📡",
        priority=70,
    )


def _recovery_insight(daily: pd.DataFrame | None) -> Insight | None:
    grid = daily_grid(daily)
    if grid.empty or "Récupération (%)" not in grid.columns:
        return None
    series = grid["Récupération (%)"].dropna()
    if len(series) < 10:
        return None

    recent = float(series.tail(7).mean())
    previous = float(series.iloc[:-7].tail(7).mean()) if len(series) > 7 else float("nan")
    if not np.isfinite(previous):
        return None

    delta = recent - previous
    if abs(delta) < 5:
        return Insight(
            "Récupération stable",
            f"Moyenne de {_fr(recent, 0)} % sur les 7 derniers jours, contre {_fr(previous, 0)} % "
            "sur les 7 précédents. Aucune dérive notable.",
            tone="success",
            icon="🟢",
            priority=40,
        )
    improving = delta > 0
    return Insight(
        "Récupération en hausse" if improving else "Récupération en baisse",
        f"{_fr(recent, 0)} % en moyenne sur 7 jours, soit {_fr(delta, 0, sign=True)} points "
        f"par rapport aux 7 jours précédents ({_fr(previous, 0)} %).",
        tone="success" if improving else "warning",
        icon="📈" if improving else "📉",
        priority=75 if not improving else 55,
    )


def _sleep_debt_insight(daily: pd.DataFrame | None) -> Insight | None:
    debt = sleep_debt_summary(daily, days=7)
    if debt["nights"] < 3 or not np.isfinite(debt["cumulative_debt"]):
        return None
    cumulative = debt["cumulative_debt"]
    if cumulative <= 0:
        return Insight(
            "Besoin de sommeil couvert",
            f"Sur {debt['nights']} nuit(s), vous dormez en moyenne {_fr(debt['mean_sleep'])} h "
            f"pour un besoin estimé à {_fr(debt['mean_need'])} h. Aucune dette accumulée.",
            tone="success",
            icon="🛌",
            priority=35,
        )
    severity = "warning" if cumulative >= 3 else "info"
    return Insight(
        "Dette de sommeil accumulée",
        f"{_fr(cumulative)} h de retard sur {debt['nights']} nuit(s), soit {_fr(debt['mean_debt'])} h "
        f"par nuit. WHOOP estime votre besoin à {_fr(debt['mean_need'])} h, vous en obtenez "
        f"{_fr(debt['mean_sleep'])} h.",
        tone=severity,
        icon="😴",
        priority=80 if cumulative >= 3 else 45,
    )


def _training_load_insight(daily: pd.DataFrame | None) -> Insight | None:
    load = training_load(daily)
    if not np.isfinite(load["ratio"]):
        return None
    ratio = load["ratio"]
    if ratio > 1.5:
        return Insight(
            "Montée en charge brutale",
            f"Votre charge des 7 derniers jours ({_fr(load['acute'])}) vaut {_fr(ratio, 2)} fois "
            f"celle des 28 derniers ({_fr(load['chronic'])}). Au-delà de 1,5, l'augmentation "
            "dépasse nettement vos habitudes récentes.",
            tone="warning",
            icon="⚠️",
            priority=85,
        )
    if ratio < 0.8:
        return Insight(
            "Charge en retrait",
            f"Votre charge récente ({_fr(load['acute'])}) ne représente que {_fr(ratio, 2)} fois "
            f"votre charge habituelle ({_fr(load['chronic'])}).",
            tone="info",
            icon="🔽",
            priority=45,
        )
    return Insight(
        "Charge maîtrisée",
        f"Rapport aigu/chronique de {_fr(ratio, 2)}, dans la plage généralement considérée "
        "comme soutenable (0,8 à 1,3).",
        tone="success",
        icon="⚖️",
        priority=30,
    )


def _baseline_insight(daily: pd.DataFrame | None, metric: str, label: str, *, lower_is_better: bool = False) -> Insight | None:
    baseline = personal_baseline(daily, metric)
    if not baseline["ready"] or abs(baseline["deviation_pct"]) < 10:
        return None
    below = baseline["deviation_pct"] < 0
    favourable = below if lower_is_better else not below
    return Insight(
        f"{label} {'sous' if below else 'au-dessus de'} votre repère",
        f"Dernière valeur {_fr(baseline['latest'])} contre {_fr(baseline['baseline'])} "
        f"habituellement, soit {_fr(baseline['deviation_pct'], 0, sign=True)} %. "
        "Le repère est la médiane de vos 30 derniers jours, pas une norme générale.",
        tone="success" if favourable else "warning",
        icon="🫀",
        priority=60 if not favourable else 35,
    )


def _energy_insight(merged: pd.DataFrame | None) -> Insight | None:
    balance = energy_balance(merged)
    if not balance["ready"]:
        return None
    deficit = balance["imbalance_per_day"]
    direction = "déficit" if deficit < 0 else "excédent"
    return Insight(
        f"Apport estimé à {_fr(balance['estimated_intake'], 0)} kcal/jour",
        f"Votre dépense mesurée par WHOOP est de {_fr(balance['mean_burn'], 0)} kcal/jour et votre "
        f"poids évolue de {_fr(balance['slope_kg_per_week'], 2, sign=True)} kg/semaine, ce qui "
        f"correspond à un {direction} d'environ {_fr(abs(deficit), 0)} kcal/jour. "
        "Estimation sensible aux variations d'eau et au bruit de pesée.",
        tone="info",
        icon="🔥",
        priority=90,
    )


def _correlation_insight(merged: pd.DataFrame | None) -> Insight | None:
    table = lagged_correlations(merged)
    if table.empty:
        return None
    best = table.iloc[0]
    if abs(float(best["Corrélation"])) < 0.4:
        return None
    lag = int(best["Décalage (jours)"])
    when = "le jour même" if lag == 0 else f"avec {lag} jour(s) de décalage"
    direction = "augmente" if float(best["Corrélation"]) > 0 else "diminue"
    return Insight(
        f"{best['Métrique']} suit votre variation de poids",
        f"Corrélation de {_fr(float(best['Corrélation']), 2)} {when}, sur "
        f"{int(best['Observations'])} observations : quand cette métrique monte, votre poids "
        f"{direction}. Association statistique, pas une relation de cause à effet.",
        tone="info",
        icon="🔗",
        priority=65,
    )


def _weekday_insight(daily: pd.DataFrame | None) -> Insight | None:
    profile = weekday_profile(daily, "Récupération (%)")
    if profile.empty:
        return None
    usable = profile.dropna(subset=["Moyenne"])
    # Deux mesures par jour de semaine au minimum, sinon un mauvais lundi isolé
    # se transformerait en « pattern du lundi ».
    usable = usable[usable["Observations"] >= 2]
    if len(usable) < 5:
        return None

    worst = usable.loc[usable["Moyenne"].idxmin()]
    overall = float(usable["Moyenne"].mean())
    gap = overall - float(worst["Moyenne"])
    if gap < 8:
        return None
    return Insight(
        f"Creux récurrent le {str(worst['Jour']).lower()}",
        f"Récupération moyenne de {_fr(float(worst['Moyenne']), 0)} % ce jour-là, contre "
        f"{_fr(overall, 0)} % en moyenne sur la semaine, sur {int(worst['Observations'])} "
        "occurrences.",
        tone="warning",
        icon="📆",
        priority=55,
    )


def _drivers_insight(daily: pd.DataFrame | None) -> Insight | None:
    drivers = recovery_drivers(daily)
    if not drivers["ready"] or drivers["r_squared"] < 0.25:
        return None
    coefficient = drivers["coefficients"].get("Sommeil (heures)")
    if coefficient is None or abs(coefficient) < 1:
        return None
    return Insight(
        "Ce qu'une heure de sommeil vous rapporte",
        f"Sur vos {drivers['days']} jours de données, chaque heure de sommeil supplémentaire "
        f"s'accompagne de {_fr(coefficient, 1, sign=True)} point(s) de récupération. Le modèle "
        f"explique {_fr(drivers['r_squared'] * 100, 0)} % des variations.",
        tone="success" if coefficient > 0 else "info",
        icon="🔬",
        priority=70,
    )


def generate_insights(daily: pd.DataFrame | None, merged: pd.DataFrame | None = None, *, limit: int = 6) -> list[Insight]:
    """Constats rédigés, classés par importance décroissante.

    Chaque règle reste muette tant que ses conditions d'effectif ne sont pas
    réunies : une page sans constat vaut mieux qu'un constat inventé.
    """
    candidates = [
        _coverage_insight(daily),
        _energy_insight(merged),
        _training_load_insight(daily),
        _sleep_debt_insight(daily),
        _recovery_insight(daily),
        _drivers_insight(daily),
        _correlation_insight(merged),
        _weekday_insight(daily),
        _baseline_insight(daily, "HRV (ms)", "Variabilité cardiaque"),
        _baseline_insight(daily, "FC repos (bpm)", "Fréquence au repos", lower_is_better=True),
    ]
    found = [insight for insight in candidates if insight is not None]
    found.sort(key=lambda insight: insight.priority, reverse=True)
    return found[: max(1, int(limit))]
