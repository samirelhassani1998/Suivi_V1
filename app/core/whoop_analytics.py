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
from typing import Any, Iterable, Mapping, Sequence

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
# Jours réellement mesurés exigés dans la fenêtre aigüe avant de conclure.
MIN_ACUTE_COVERAGE = 4
# Jours que la fenêtre chronique doit compter en plus de la fenêtre aigüe
# pour que le rapport compare deux périodes distinctes.
MIN_CHRONIC_MARGIN = 7


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
    # Les convertisseurs dédupliquent déjà, mais une date en double venue d'une
    # autre source ferait échouer la réindexation et emporterait toute la page :
    # cette fonction est trop en aval pour se permettre de lever une exception.
    if data["Date"].duplicated().any():
        data = data.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    full_range = pd.date_range(data["Date"].min(), data["Date"].max(), freq="D")
    return (
        data.set_index("Date")
        .reindex(full_range)
        .rename_axis("Date")
        .reset_index()
    )


def last_days(frame: pd.DataFrame | None, days: int, *, today: Any = None) -> pd.DataFrame:
    """Lignes des *days* derniers jours calendaires, et non les *days* dernières lignes.

    La nuance est invisible sur un historique quotidien complet et décisive dès
    qu'il y a des trous : ``tail(7)`` sur un bracelet porté trois fois en six
    semaines renvoie des mesures vieilles d'un mois tout en les présentant comme
    « les 7 derniers jours ».
    """
    data = _clean_daily(frame)
    if data.empty:
        return data
    reference = pd.Timestamp(today).normalize() if today is not None else data["Date"].max()
    cutoff = reference - pd.Timedelta(days=max(1, int(days)) - 1)
    return data[(data["Date"] >= cutoff) & (data["Date"] <= reference)].reset_index(drop=True)


def previous_days(frame: pd.DataFrame | None, days: int, *, today: Any = None) -> pd.DataFrame:
    """Fenêtre calendaire immédiatement antérieure à celle de :func:`last_days`."""
    data = _clean_daily(frame)
    if data.empty:
        return data
    reference = pd.Timestamp(today).normalize() if today is not None else data["Date"].max()
    window = max(1, int(days))
    end = reference - pd.Timedelta(days=window)
    start = end - pd.Timedelta(days=window - 1)
    return data[(data["Date"] >= start) & (data["Date"] <= end)].reset_index(drop=True)


def coverage_report(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Couverture réelle de la période importée."""
    data = _clean_daily(frame)
    if data.empty:
        return {
            "days_with_data": 0,
            "complete_days": 0,
            "span_days": 0,
            "coverage_pct": 0.0,
            "complete_pct": 0.0,
            "gaps": 0,
            "start": None,
            "end": None,
        }
    start, end = data["Date"].min(), data["Date"].max()
    span = int((end - start).days) + 1
    measured = int(data["Date"].nunique())
    # Une ligne existe dès qu'une seule des trois sources a renvoyé quelque chose.
    # Compter ces jours comme couverts annonçait 100 % de couverture alors que
    # ni la récupération ni le sommeil n'étaient disponibles.
    core_columns = [column for column in ("Récupération (%)", "Sommeil (heures)", "Strain") if column in data.columns]
    if core_columns:
        complete = int(data.loc[data[core_columns].notna().all(axis=1), "Date"].nunique())
    else:
        complete = 0
    return {
        "days_with_data": measured,
        "complete_days": complete,
        "span_days": span,
        "coverage_pct": round(measured / span * 100, 1) if span else 0.0,
        "complete_pct": round(complete / span * 100, 1) if span else 0.0,
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
    result = {
        "acute": float("nan"),
        "chronic": float("nan"),
        "ratio": float("nan"),
        "status": "indisponible",
        "days": 0,
        "acute_days_measured": 0,
        "chronic_days_measured": 0,
    }
    if data.empty or "Strain" not in data.columns:
        return result
    grid = daily_grid(data)
    strain = grid["Strain"]
    measured = int(strain.notna().sum())
    result["days"] = measured
    if measured < MIN_DAYS_TRAINING_LOAD:
        return result

    acute_window = strain.tail(ACUTE_LOAD_DAYS)
    chronic_window = strain.tail(CHRONIC_LOAD_DAYS)
    acute_measured = int(acute_window.notna().sum())
    result["acute_days_measured"] = acute_measured
    result["chronic_days_measured"] = int(chronic_window.notna().sum())

    acute = float(acute_window.mean(skipna=True))
    chronic = float(chronic_window.mean(skipna=True))
    result["acute"] = acute
    result["chronic"] = chronic
    if not np.isfinite(chronic) or chronic <= 0:
        return result
    # Deux journées intenses isolées dans une semaine peu portée donneraient une
    # « charge aigüe » élevée qui ne décrit pas une semaine de travail réelle.
    if acute_measured < MIN_ACUTE_COVERAGE:
        result["status"] = "couverture insuffisante"
        return result
    # Tant que l'historique ne dépasse pas la fenêtre aigüe, les deux moyennes
    # portent sur les mêmes jours : le rapport vaut alors 1,00 par construction
    # et afficherait « charge maîtrisée » quelles que soient les données.
    if result["chronic_days_measured"] <= acute_measured + MIN_CHRONIC_MARGIN:
        result["status"] = "historique trop court"
        result["ratio"] = float("nan")
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
    recent = last_days(data, days)
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
    result = {
        "slope_kg_per_day": float("nan"),
        "slope_kg_per_week": float("nan"),
        "slope_std_error": float("nan"),
        "days": 0,
        "span_days": 0,
        "r_squared": float("nan"),
    }
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
    # Erreur-type de la pente : sans elle, une tendance estimée sur douze jours
    # de pesées bruitées se lirait avec la même assurance qu'une tendance longue.
    degrees = len(series) - 2
    variance_x = float(np.sum((x - x.mean()) ** 2))
    slope_error = (
        float(np.sqrt(residual / degrees / variance_x)) if degrees > 0 and variance_x > 0 else float("nan")
    )
    result.update(
        {
            "slope_kg_per_day": float(slope),
            "slope_kg_per_week": float(slope) * 7.0,
            "slope_std_error": slope_error,
            "days": int(len(series)),
            "span_days": int(np.ptp(x)) + 1,
            "r_squared": 1.0 - residual / total if total > 0 else float("nan"),
        }
    )
    return result


def energy_balance(
    merged: pd.DataFrame | None,
    *,
    kcal_per_kg: float = KCAL_PER_KG,
    weight_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
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
        "intake_margin": float("nan"),
        "trend_r_squared": float("nan"),
        "slope_kg_per_week": float("nan"),
    }
    if data.empty or "Calories (kcal)" not in data.columns:
        return result

    burn = data["Calories (kcal)"].dropna()
    # La pente gagne à s'appuyer sur toutes les pesées de la période, y compris
    # celles des jours où le bracelet n'a rien enregistré : la jointure interne
    # en écartait une partie et rendait la tendance plus bruitée qu'utile.
    trend_source = data
    if weight_history is not None and not weight_history.empty:
        window = _clean_daily(weight_history)
        if not window.empty:
            inside = window[(window["Date"] >= data["Date"].min()) & (window["Date"] <= data["Date"].max())]
            if len(inside) > trend_source["Poids (Kgs)"].notna().sum():
                trend_source = inside
    trend = weight_trend(trend_source)
    usable_days = int(min(len(burn), trend["days"]))
    result["days"] = usable_days
    result["mean_burn"] = float(burn.mean()) if not burn.empty else float("nan")
    result["slope_kg_per_week"] = trend["slope_kg_per_week"]

    if usable_days < MIN_DAYS_ENERGY_BALANCE or not np.isfinite(trend["slope_kg_per_day"]) or not np.isfinite(result["mean_burn"]):
        return result

    imbalance = trend["slope_kg_per_day"] * float(kcal_per_kg)
    result["imbalance_per_day"] = imbalance
    result["estimated_intake"] = result["mean_burn"] + imbalance
    # Marge issue de l'incertitude sur la pente du poids, convertie en calories.
    margin = trend["slope_std_error"] * 1.96 * float(kcal_per_kg)
    result["intake_margin"] = float(margin) if np.isfinite(margin) else float("nan")
    result["trend_r_squared"] = trend["r_squared"]
    result["ready"] = True
    return result


# Seuil de signification avant correction pour tests multiples.
ALPHA = 0.05


def _correlation_p_value(correlation: float, observations: int) -> float:
    """Probabilité d'obtenir une corrélation au moins aussi forte par hasard."""
    if observations < 3 or not np.isfinite(correlation) or abs(correlation) >= 1.0:
        return 0.0 if abs(correlation) >= 1.0 else 1.0
    from scipy import stats

    degrees = observations - 2
    statistic = abs(correlation) * np.sqrt(degrees / (1.0 - correlation**2))
    return float(2.0 * stats.t.sf(statistic, degrees))


def _correlation_reading(correlation: float, significant: bool, tests: int) -> str:
    """Formule la lecture en tenant compte du nombre de tests effectués."""
    direction = "même sens" if correlation > 0 else "sens opposé"
    if not significant:
        return f"non distinguable du hasard sur {tests} tests"
    magnitude = abs(correlation)
    strength = "forte" if magnitude >= 0.5 else "modérée" if magnitude >= 0.3 else "faible"
    return f"association {strength}, {direction}"


DEFAULT_CORRELATION_TARGET = "Variation poids (kg/jour)"


# Repères d'apport énergétique couramment cités pour un adulte. Ce ne sont pas
# des seuils médicaux : ils servent uniquement à qualifier l'exigence d'un rythme.
INTAKE_DEMANDING_KCAL = 1800.0
INTAKE_VERY_DEMANDING_KCAL = 1400.0


def target_pace_feasibility(
    merged: pd.DataFrame | None,
    *,
    required_daily_kg: float,
    kcal_per_kg: float = KCAL_PER_KG,
) -> dict[str, Any]:
    """Traduit le rythme visé par la trajectoire cible en apport calorique implicite.

    L'objectif de poids est exprimé en kilogrammes et la dépense mesurée par WHOOP
    en kilocalories. Les rapprocher indique ce que le rythme visé suppose de manger
    chaque jour — un chiffre que ni la balance ni le bracelet ne produisent seuls,
    et qui dit si l'objectif est atteignable ou seulement souhaitable.
    """
    result = {
        "ready": False,
        "required_daily_kg": float(required_daily_kg),
        "required_weekly_kg": float(required_daily_kg) * 7.0,
        "required_deficit": float(required_daily_kg) * float(kcal_per_kg),
        "mean_burn": float("nan"),
        "implied_intake": float("nan"),
        "current_daily_kg": float("nan"),
        "verdict": "indisponible",
        "days": 0,
    }
    # Un rythme nul ou négatif ne décrit pas une perte de poids : sans cette
    # garde, le déficit requis vaudrait zéro et l'apport implicite égalerait la
    # dépense, ce qui se lirait comme un objectif trivialement atteignable.
    if not np.isfinite(required_daily_kg) or required_daily_kg <= 0:
        return result

    data = _clean_daily(merged)
    if data.empty or "Calories (kcal)" not in data.columns:
        return result

    burn = data["Calories (kcal)"].dropna()
    result["days"] = int(len(burn))
    if burn.empty:
        return result
    result["mean_burn"] = float(burn.mean())

    trend = weight_trend(data)
    result["current_daily_kg"] = -trend["slope_kg_per_day"] if np.isfinite(trend["slope_kg_per_day"]) else float("nan")

    if result["days"] < MIN_DAYS_TRAINING_LOAD:
        return result

    implied = result["mean_burn"] - result["required_deficit"]
    result["implied_intake"] = implied
    if implied >= INTAKE_DEMANDING_KCAL:
        result["verdict"] = "exigeant"
    elif implied >= INTAKE_VERY_DEMANDING_KCAL:
        result["verdict"] = "très exigeant"
    else:
        result["verdict"] = "sous les repères usuels"
    result["ready"] = True
    return result


def lagged_correlations(
    merged: pd.DataFrame | None,
    *,
    target: str = DEFAULT_CORRELATION_TARGET,
    lags: Sequence[int] = (0, 1, 2),
    min_pairs: int = MIN_DAYS_CORRELATION,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Corrélations entre une métrique décalée de *k* jours et la cible.

    Un entraînement intense pèse rarement sur la balance le jour même : tester
    plusieurs décalages évite de conclure à une absence de lien trop vite.

    La cible par défaut est le rythme quotidien (kg/jour) et non la variation
    brute : sans cette normalisation, deux pesées espacées de dix jours pèseraient
    dix fois plus lourd qu'une variation d'un jour à l'autre.
    """
    columns = ["Métrique", "Décalage (jours)", "Corrélation", "Observations", "Significatif", "Lecture"]
    grid = daily_grid(merged)
    if grid.empty or target not in grid.columns:
        return pd.DataFrame(columns=columns)

    candidates = list(metrics) if metrics is not None else available_metrics(grid)
    # Les colonnes dérivées du poids corréleraient trivialement avec la cible.
    candidates = [name for name in candidates if name != target]
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
            rows.append(
                {
                    "Métrique": metric,
                    "Décalage (jours)": int(lag),
                    "Corrélation": round(correlation, 3),
                    "Observations": int(len(pair)),
                    "_p": _correlation_p_value(correlation, len(pair)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    # Chaque métrique est testée à chaque décalage : avec une vingtaine de
    # métriques et trois décalages, une corrélation « forte » apparaît presque
    # sûrement par hasard. Le seuil est donc divisé par le nombre de tests.
    tests = len(frame)
    threshold = ALPHA / max(1, tests)
    frame["Significatif"] = frame["_p"] <= threshold
    frame["Lecture"] = [
        _correlation_reading(row.Corrélation, row.Significatif, tests)
        for row in frame.itertuples(index=False)
    ]
    frame = frame.drop(columns=["_p"])
    ordered = frame.reindex(frame["Corrélation"].abs().sort_values(ascending=False).index)
    # Les associations qui survivent à la correction passent devant.
    ordered = ordered.sort_values("Significatif", ascending=False, kind="mergesort")
    return ordered.reset_index(drop=True)[columns]


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

    # Un R² global faible n'interdit pas qu'un prédicteur porte un effet réel :
    # la physiologie quotidienne est bruitée, et exiger 25 % de variance
    # expliquée fait taire des coefficients pourtant nettement non nuls. Chaque
    # coefficient reçoit donc son erreur-type et sa p-value, testés séparément.
    degrees = len(design) - x.shape[1]
    standard_errors: dict[str, float] = {}
    p_values: dict[str, float] = {}
    if degrees > 0 and residual >= 0:
        from scipy import stats

        variance = residual / degrees
        try:
            covariance = variance * np.linalg.inv(x.T @ x)
        except np.linalg.LinAlgError:
            covariance = None
        if covariance is not None:
            for index, name in enumerate(predictors):
                error = float(np.sqrt(abs(covariance[index + 1, index + 1])))
                standard_errors[name] = error
                if error > 0:
                    statistic = abs(float(coefficients[index + 1])) / error
                    p_values[name] = float(2.0 * stats.t.sf(statistic, degrees))
                else:
                    p_values[name] = float("nan")
    result["standard_errors"] = standard_errors
    result["p_values"] = p_values
    result["observations"] = int(len(design))
    raw_r2 = 1.0 - residual / total if total > 0 else float("nan")
    # Le R² brut augmente mécaniquement avec le nombre de variables : sur douze
    # observations, il dépasse souvent 0,2 sur des données sans aucun lien.
    observations, parameters = len(design), len(predictors)
    if np.isfinite(raw_r2) and observations > parameters + 1:
        adjusted = 1.0 - (1.0 - raw_r2) * (observations - 1) / (observations - parameters - 1)
    else:
        adjusted = float("nan")
    result["r_squared"] = float(adjusted) if np.isfinite(adjusted) else float("nan")
    result["raw_r_squared"] = float(raw_r2) if np.isfinite(raw_r2) else float("nan")
    result["ready"] = True
    return result


def weekly_rollup(merged: pd.DataFrame | None) -> pd.DataFrame:
    """Synthèse hebdomadaire : récupération, sommeil, charge et variation de poids."""
    columns = [
        "Semaine",
        "Jours",
        "Récupération (%)",
        "Sommeil (heures)",
        "Strain cumulé",
        "Poids moyen (kg)",
        "Variation (kg)",
        "Sur (jours)",
    ]
    data = _clean_daily(merged)
    if data.empty:
        return pd.DataFrame(columns=columns)

    data = data.assign(_week=data["Date"].dt.to_period("W").dt.start_time)
    rows = []
    for week, chunk in data.groupby("_week", sort=True):
        weighed = chunk.dropna(subset=["Poids (Kgs)"]) if "Poids (Kgs)" in chunk.columns else chunk.iloc[0:0]
        weights = weighed["Poids (Kgs)"] if not weighed.empty else pd.Series(dtype=float)
        # L'écart entre la première et la dernière pesée ne couvre la semaine que
        # si ces deux pesées en sont éloignées : le dire évite de lire « -1 kg
        # cette semaine » pour deux pesées consécutives.
        span_days = (
            int((weighed["Date"].iloc[-1] - weighed["Date"].iloc[0]).days) + 1 if len(weighed) >= 2 else float("nan")
        )
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
                "Sur (jours)": span_days,
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
    columns = ["Date", "Récupération (%)", "Strain", "Signal", "Type"]
    grid = daily_grid(frame)
    if grid.empty or "Strain" not in grid.columns or "Récupération (%)" not in grid.columns:
        return pd.DataFrame(columns=columns)

    usable = grid[["Date", "Récupération (%)", "Strain"]].dropna()
    if usable.empty:
        return pd.DataFrame(columns=columns)

    median_strain = float(usable["Strain"].median())
    signals: list[str] = []
    kinds: list[str] = []
    for _, row in usable.iterrows():
        recovery, strain = float(row["Récupération (%)"]), float(row["Strain"])
        if recovery < 34 and strain > median_strain:
            signal, kind = "charge élevée sur récupération basse", "alerte"
        elif recovery >= 67 and strain < median_strain:
            signal, kind = "récupération élevée sous-exploitée", "occasion"
        else:
            signal, kind = "cohérent", "cohérent"
        signals.append(signal)
        kinds.append(kind)
    # Ranger une bonne journée parmi les alertes brouille la lecture : le type
    # distingue un risque d'une occasion manquée.
    return usable.assign(Signal=signals, Type=kinds)[columns].reset_index(drop=True)


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
    measured = data.loc[series.index, ["Date", metric]]
    latest_date = measured["Date"].iloc[-1]
    history = measured.iloc[:-1]
    history = history[history["Date"] >= latest_date - pd.Timedelta(days=max(1, int(days)))]
    if history.empty:
        return result
    baseline = float(history[metric].median())
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
        if not np.isfinite(reference) or reference <= 0:
            continue
        # Une série qui traverse zéro ou passe en négatif inverse le sens de
        # l'indice : se coucher plus tôt (-2 h) donnerait un indice de 200.
        values = series.dropna()
        if not values.empty and float(values.min()) <= 0:
            continue
        output[metric] = series / reference * float(base)
    return output


def indexable_metrics(frame: pd.DataFrame | None, metrics: Sequence[str]) -> list[str]:
    """Métriques pour lesquelles une base 100 reste interprétable."""
    indexed = indexed_series(frame, metrics)
    return [metric for metric in metrics if metric in indexed.columns]


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
    # Un constat épinglé survit à la coupure d'affichage : il est réservé aux
    # avertissements qu'il serait fautif de masquer derrière une limite de place.
    pinned: bool = False


def _fr(value: float, decimals: int = 1, *, sign: bool = False) -> str:
    """Nombre au format français, sans dépendre de la couche d'affichage."""
    if value is None or not np.isfinite(value):
        return "—"
    prefix = ""
    if sign:
        prefix = "+" if value > 0 else "−" if value < 0 else ""
        value = abs(value)
    elif value < 0:
        # Sans cette branche, un négatif passé sans ``sign`` sortait avec un
        # trait d'union ASCII quand le reste de l'application affiche le signe
        # moins typographique : deux graphies pour une même grandeur.
        prefix = "−"
        value = abs(value)
    return f"{prefix}{value:.{decimals}f}".replace(".", ",")


def _plural(count: Any, singular: str, plural: str | None = None) -> str:
    """Accord du nom sur le nombre, plutôt qu'un « (s) » systématique.

    La forme plurielle doit être fournie pour tout nom qui ne prend pas un
    simple « s » : « signe vital » devient « signes vitaux », pas « signe vitals ».
    """
    try:
        numeric = int(count)
    except (TypeError, ValueError):
        return singular
    return singular if abs(numeric) <= 1 else (plural or singular + "s")


def _coverage_insight(daily: pd.DataFrame | None) -> Insight | None:
    coverage = coverage_report(daily)
    if coverage["span_days"] < 7 or coverage["coverage_pct"] >= 80:
        return None
    return Insight(
        "Port du bracelet irrégulier",
        f"{coverage['gaps']} {_plural(coverage['gaps'], 'jour')} sans mesure sur {coverage['span_days']}, soit "
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
    if grid["Récupération (%)"].notna().sum() < 10:
        return None

    recent_window = last_days(grid, 7)["Récupération (%)"].dropna()
    previous_window = previous_days(grid, 7)["Récupération (%)"].dropna()
    # Sous trois nuits par fenêtre, l'écart décrit le hasard des jours portés.
    if len(recent_window) < 3 or len(previous_window) < 3:
        return None
    recent, previous = float(recent_window.mean()), float(previous_window.mean())
    if not np.isfinite(recent) or not np.isfinite(previous):
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
            f"Sur {debt['nights']} {_plural(debt['nights'], 'nuit')}, vous dormez en moyenne {_fr(debt['mean_sleep'])} h "
            f"pour un besoin estimé à {_fr(debt['mean_need'])} h. Aucune dette accumulée.",
            tone="success",
            icon="🛌",
            priority=35,
        )
    severity = "warning" if cumulative >= 3 else "info"
    return Insight(
        "Dette de sommeil accumulée",
        f"{_fr(cumulative)} h de retard sur {debt['nights']} {_plural(debt['nights'], 'nuit')}, soit {_fr(debt['mean_debt'])} h "
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
        icon="🏋️",
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
    # Sans cette condition, le constat se déclenchait sur presque toutes les
    # séries de bruit : une vingtaine de métriques testées à trois décalages
    # produit mécaniquement une corrélation forte.
    if not bool(best.get("Significatif", False)):
        return None
    correlation = float(best["Corrélation"])
    lag = int(best["Décalage (jours)"])
    when = "le jour même" if lag == 0 else "le lendemain" if lag == 1 else f"{lag} jours plus tard"
    direction = "plus vite" if correlation < 0 else "moins vite"
    return Insight(
        f"{best['Métrique']} accompagne votre rythme de perte",
        f"Corrélation de {_fr(correlation, 2)} sur {int(best['Observations'])} jours : quand cette "
        f"métrique est élevée, votre poids baisse {direction} {when}. L'association survit à la "
        "correction pour tests multiples, mais reste une association, pas une cause.",
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
        f"{_plural(worst['Observations'], 'occurrence')}.",
        tone="warning",
        icon="📆",
        priority=55,
    )


def _significant_coefficient(drivers: Mapping[str, Any], name: str) -> float | None:
    """Coefficient d'un prédicteur, seulement s'il se distingue de zéro.

    L'ancien garde-fou portait sur le R² de l'ensemble du modèle. Un R² élevé
    grâce à un prédicteur n'autorise pourtant rien à affirmer sur un autre : un
    coefficient du sommeil mesuré à p = 0,80 — donc indiscernable de zéro —
    était présenté comme un fait dès que la charge, elle, expliquait assez de
    variance. Chaque coefficient répond désormais de sa propre p-value.
    """
    if not drivers.get("ready"):
        return None
    coefficients = drivers.get("coefficients", {})
    coefficient = coefficients.get(name)
    p_value = drivers.get("p_values", {}).get(name)
    if coefficient is None or p_value is None:
        return None
    # Les deux coefficients du modèle sont interrogés au cours d'un même appel à
    # generate_insights. Les accepter chacun à 5 % laisse près de 10 % de chances
    # qu'au moins un passe alors qu'aucun lien n'existe — 8,2 % mesurés sur
    # 600 séries sans lien. Le seuil est donc divisé par le nombre de
    # coefficients testés, comme il l'est déjà pour les corrélations décalées.
    threshold = ALPHA / max(1, len(coefficients))
    if not np.isfinite(coefficient) or not np.isfinite(p_value) or p_value > threshold:
        return None
    return float(coefficient)


def _drivers_insight(daily: pd.DataFrame | None) -> Insight | None:
    drivers = recovery_drivers(daily)
    coefficient = _significant_coefficient(drivers, "Sommeil (heures)")
    if coefficient is None or abs(coefficient) < 1:
        return None
    return Insight(
        "Ce qu'une heure de sommeil vous rapporte",
        f"Sur vos {drivers['days']} jours de données, chaque heure de sommeil supplémentaire "
        f"s'accompagne de {_fr(coefficient, 1, sign=True)} {_plural(round(coefficient), 'point')} de récupération"
        # Le modèle n'inclut la charge que si le bracelet a renvoyé des cycles.
        # Annoncer « à charge égale » sans l'avoir ajustée serait faux.
        + (", à charge d'entraînement égale." if "Strain de la veille" in drivers["coefficients"] else "."),
        tone="success" if coefficient > 0 else "info",
        icon="🔬",
        priority=70,
    )


def _observed_strain_contrast(frame: pd.DataFrame | None) -> float | None:
    """Écart de charge réellement observé, pour illustrer sans extrapoler."""
    grid = daily_grid(frame)
    if grid.empty or "Strain" not in grid.columns:
        return None
    values = grid["Strain"].dropna()
    if len(values) < 4:
        return None
    spread = float(values.quantile(0.75) - values.quantile(0.25))
    if not np.isfinite(spread) or spread < 1.0:
        return None
    return round(spread)


def _load_cost_insight(daily: pd.DataFrame | None) -> Insight | None:
    """Ce qu'une journée chargée coûte à la récupération du lendemain.

    La régression estimait déjà ce coefficient sans jamais l'afficher, alors
    qu'il porte la seule question d'arbitrage que pose l'entraînement : combien
    me coûtera demain l'effort d'aujourd'hui. L'application WHOOP affiche le
    strain et la récupération côte à côte, mais ne chiffre pas le lien.
    """
    drivers = recovery_drivers(daily)
    coefficient = _significant_coefficient(drivers, "Strain de la veille")
    if coefficient is None or abs(coefficient) < 0.3:
        return None
    if coefficient < 0:
        body = (
            f"Sur vos {drivers['days']} jours de données, chaque point de strain supplémentaire "
            f"s'accompagne de {_fr(abs(coefficient))} {_plural(round(abs(coefficient)) or 1, 'point')} "
            f"de récupération en moins le lendemain matin"
            + (", à sommeil égal." if "Sommeil (heures)" in drivers["coefficients"] else ".")
        )
        # L'exemple chiffré était figé à cinq points de strain. Chez qui varie
        # moins que cela, il extrapole hors de la plage observée et présente le
        # résultat comme une prévision. L'écart illustré est donc repris de la
        # dispersion réellement mesurée, et omis lorsqu'elle est trop faible.
        contrast = _observed_strain_contrast(daily)
        if contrast is not None:
            body += (
                f" Entre une journée ordinaire et une journée {_fr(contrast, 0)} points plus chargée — "
                f"un écart courant chez vous — l'écart de récupération observé le lendemain est "
                f"d'environ {_fr(abs(coefficient) * contrast, 0)} points."
            )
        # « coûte donc » transformerait une association en cause. Le strain peut
        # accompagner un facteur non mesuré — maladie, jour de repos planifié,
        # récupération de la veille — auquel l'écart serait en réalité imputable.
        body += (
            " Il s'agit d'une association mesurée sur vos données, pas d'un effet établi :"
            " d'autres facteurs non mesurés varient avec la charge."
        )
        tone = "warning"
    else:
        # Conclure « votre charge reste dans ce que vous encaissez » ferait de
        # cette branche une recommandation. Or si vous vous entraînez plus fort
        # après une bonne nuit, et que la récupération est autocorrélée, un
        # coefficient positif apparaît sans que la charge y soit pour rien. La
        # branche positive reste donc aussi prudente que la négative.
        body = (
            f"Sur vos {drivers['days']} jours de données, vos journées chargées ne sont pas suivies "
            f"d'une récupération dégradée : le lien mesuré est de {_fr(coefficient, 1, sign=True)} point "
            f"par point de strain"
            + (", à sommeil égal." if "Sommeil (heures)" in drivers["coefficients"] else ".")
            + " C'est une association mesurée sur vos données, et non "
            f"la preuve que votre charge est bien tolérée : s'entraîner davantage les matins où l'on se "
            f"réveille frais produit le même signe."
        )
        tone = "info"
    return Insight(
        "Ce qu'une journée chargée coûte au lendemain",
        body,
        tone=tone,
        icon="⚡",
        priority=72,
    )


def _factor_unit(factor: str) -> str:
    """Unité à afficher derrière la valeur d'un facteur de contraste."""
    if "heures" in factor:
        return "h"
    if "coucher" in factor:
        return "h"
    if "kcal" in factor:
        return "kcal"
    if "%" in factor:
        return "%"
    return ""


def _factor_phrase(factor: str) -> str:
    """Groupe nominal articlé, utilisable tel quel en milieu de phrase."""
    return _FACTOR_PHRASES.get(factor, factor.split(" (")[0].lower())


def _factor_verb(factor: str) -> str:
    """Accorde le verbe au nombre du groupe nominal.

    « les calories brûlées la veille vaut » : un gabarit figé au singulier
    produit une faute d'accord dès qu'un facteur pluriel arrive en tête.
    """
    return "valent" if _factor_phrase(factor).startswith("les ") else "vaut"


def _contrast_insight(daily: pd.DataFrame | None) -> Insight | None:
    contrast = contrast_best_worst_days(daily)
    if not contrast["ready"] or contrast["table"].empty:
        return None
    top = contrast["table"].iloc[0]
    factor, gap = str(top["Facteur"]), float(top["Écart"])
    effect = float(top.get("Écart normalisé", float("nan")))
    # Un écart normalisé sous 0,5 est un effet modeste : le publier comme « le
    # facteur qui sépare le plus » lui donnerait un poids qu'il n'a pas.
    if not np.isfinite(effect) or abs(effect) < 0.5:
        return None
    unit = _factor_unit(factor)
    # « soit moins de 2,7 » se lit en français « moins que 2,7 », un tout autre
    # sens que « 2,7 de moins ». Le complément se place donc après la valeur.
    direction = "de plus" if gap > 0 else "de moins"
    return Insight(
        f"Vos meilleurs jours : {factor.split(' (')[0]}",
        f"Sur vos {contrast['best_days']} meilleurs jours de récupération, "
        f"{_factor_phrase(factor)} {_factor_verb(factor)} {_fr(float(top['Meilleurs jours']))} {unit}".rstrip()
        + f" contre {_fr(float(top['Pires jours']))} {unit}".rstrip()
        + f" sur les {contrast['worst_days']} pires, soit {_fr(abs(gap))} {unit} {direction}".replace("  ", " ")
        + ". C'est le facteur qui sépare le plus vos bons et vos mauvais jours.",
        tone="info",
        icon="🔍",
        priority=78,
    )


def _sport_insight(daily: pd.DataFrame | None, workouts: pd.DataFrame | None) -> Insight | None:
    impact = sport_recovery_impact(daily, workouts)
    if impact.empty:
        return None
    hardest = impact.iloc[0]
    gap = float(hardest["Écart à votre moyenne"])
    if gap > -5:
        return None
    return Insight(
        f"{str(hardest['Sport']).capitalize()} pèse sur votre lendemain",
        f"Après vos {int(hardest['Séances'])} séances de {str(hardest['Sport']).lower()}, votre "
        f"récupération du lendemain atteint {_fr(float(hardest['Récupération du lendemain (%)']), 0)} %, "
        # « soit −6 points sous votre moyenne » énonce deux fois la négation :
        # littéralement six points AU-DESSUS. Le sens est porté par « sous ».
        f"soit {_fr(abs(gap), 0)} points sous votre moyenne. Prévoir une journée plus légère ensuite "
        "est une piste, pas une prescription.",
        tone="warning",
        icon="🥊",
        priority=72,
    )


def _target_pace_insight(merged: pd.DataFrame | None, required_daily_kg: float | None) -> Insight | None:
    """Confronte le rythme visé à la dépense réellement mesurée."""
    if required_daily_kg is None or not np.isfinite(required_daily_kg) or required_daily_kg <= 0:
        return None
    feasibility = target_pace_feasibility(merged, required_daily_kg=required_daily_kg)
    if not feasibility["ready"]:
        return None

    intake = feasibility["implied_intake"]
    tone = "info" if feasibility["verdict"] == "exigeant" else "warning"
    closing = (
        " Un apport à ce niveau se discute avec un professionnel de santé plutôt qu'avec un tableau de bord."
        if intake < INTAKE_DEMANDING_KCAL
        else ""
    )
    return Insight(
        f"Votre objectif suppose environ {_fr(intake, 0)} kcal/jour",
        f"Atteindre la cible demande {_fr(feasibility['required_weekly_kg'], 2)} kg par semaine, soit un déficit "
        f"d'environ {_fr(feasibility['required_deficit'], 0)} kcal/jour. Votre dépense mesurée par WHOOP étant de "
        f"{_fr(feasibility['mean_burn'], 0)} kcal/jour, il resterait {_fr(intake, 0)} kcal/jour à consommer — "
        f"un rythme {feasibility['verdict']}." + closing,
        tone=tone,
        icon="🎯",
        # Volontairement sous les constats mesurés : ce chiffre découle des
        # paramètres de l'objectif et bouge à peine d'un jour sur l'autre.
        priority=74,
    )


def generate_insights(
    daily: pd.DataFrame | None,
    merged: pd.DataFrame | None = None,
    workouts: pd.DataFrame | None = None,
    *,
    required_daily_kg: float | None = None,
    target_status: Mapping[str, Any] | None = None,
    target_weight: float | None = None,
    target_date: Any = None,
    limit: int = 6,
) -> list[Insight]:
    """Constats rédigés, classés par importance décroissante.

    Chaque règle reste muette tant que ses conditions d'effectif ne sont pas
    réunies : une page sans constat vaut mieux qu'un constat inventé.
    """
    candidates = [
        _coverage_insight(daily),
        _vitals_insight(daily),
        # Où va le poids passe avant tout le reste : c'est la question que pose
        # l'application. Sans ces deux règles, une prise de deux kilos pouvait
        # rester invisible derrière un constat sur le sommeil du lundi.
        _weight_direction_insight(merged),
        _target_progress_insight(merged, target_status, target_weight=target_weight, target_date=target_date),
        _target_pace_insight(merged, required_daily_kg),
        _energy_insight(merged),
        _training_load_insight(daily),
        _streak_insight(daily),
        _sleep_debt_insight(daily),
        _recovery_insight(daily),
        _drivers_insight(daily),
        _load_cost_insight(daily),
        _tolerance_insight(daily),
        _architecture_insight(daily),
        _correlation_insight(merged),
        _contrast_insight(daily),
        _sport_insight(daily, workouts),
        _weekday_insight(daily),
        _baseline_insight(daily, "HRV (ms)", "Variabilité cardiaque"),
        _baseline_insight(daily, "FC repos (bpm)", "Fréquence au repos", lower_is_better=True),
    ]
    found = [insight for insight in candidates if insight is not None]
    # La veille physiologique nomme déjà les signes vitaux qui dévient : les
    # répéter en cartes séparées ferait dire trois fois la même chose.
    if any(insight.icon == "🩺" for insight in found):
        found = [insight for insight in found if insight.icon != "🫀"]
    # Le plafond de charge et le coût marginal d'un point de strain décrivent le
    # même effet, l'un par son seuil, l'autre par sa pente : les afficher tous
    # deux occupe deux cartes voisines pour un seul constat. Le seuil l'emporte,
    # parce qu'il nomme un nombre sur lequel agir pendant la séance.
    if any(insight.icon == "🧗" for insight in found):
        found = [insight for insight in found if insight.icon != "⚡"]
    found.sort(key=lambda insight: insight.priority, reverse=True)
    kept = max(1, int(limit))
    shown = found[:kept]
    # Les constats épinglés absents de la coupure y sont réintégrés : masquer un
    # avertissement de santé parce que six autres cartes se sont déclenchées
    # serait le pire comportement possible de cette liste.
    missing = [insight for insight in found[kept:] if insight.pinned]
    if missing:
        shown = sorted(shown + missing, key=lambda insight: insight.priority, reverse=True)
    return shown


# ──────────────────────────────────────────────────────────────────────────────
# Ce qui distingue les bons jours des mauvais
# ──────────────────────────────────────────────────────────────────────────────

MIN_DAYS_CONTRAST = 15
MIN_SESSIONS_PER_SPORT = 3

# Métriques susceptibles d'expliquer un écart de récupération.
# Le score de récupération WHOOP est calculé au réveil, à partir de la nuit
# écoulée : il est donc déjà fixé avant la première minute d'effort de la
# journée. Comparer la récupération du matin au strain du même jour inverse la
# flèche du temps — on mesure alors « je m'entraîne plus les jours où je me
# réveille frais », que le lecteur interprète en « m'entraîner me fait
# récupérer ». Chaque facteur porte donc son décalage explicite, et seuls des
# antécédents figurent ici.
#
# (libellé du tableau, colonne source, décalage en jours, groupe nominal)
# Le groupe nominal porte son article : recomposer « le »/« la »/« les » à
# partir du libellé demanderait de deviner le genre de chaque intitulé.
CONTRAST_FACTORS: tuple[tuple[str, str, int, str], ...] = (
    ("Sommeil (heures)", "Sommeil (heures)", 0, "le sommeil"),
    ("Dette de sommeil (heures)", "Dette de sommeil (heures)", 0, "la dette de sommeil"),
    ("Heure de coucher", "Heure de coucher", 0, "l'heure de coucher"),
    ("Perturbations sommeil", "Perturbations sommeil", 0, "les perturbations du sommeil"),
    ("Régularité sommeil (%)", "Régularité sommeil (%)", 0, "la régularité du sommeil"),
    ("Strain de la veille", "Strain", 1, "le strain de la veille"),
    ("Calories de la veille (kcal)", "Calories (kcal)", 1, "les calories brûlées la veille"),
)

_FACTOR_PHRASES: dict[str, str] = {label: phrase for label, _, _, phrase in CONTRAST_FACTORS}

# Conservé pour la compatibilité : la liste des colonnes réellement lues.
CONTRAST_METRICS: tuple[str, ...] = tuple(dict.fromkeys(column for _, column, _, _ in CONTRAST_FACTORS))


def contrast_best_worst_days(
    frame: pd.DataFrame | None,
    *,
    metric: str = "Récupération (%)",
    min_days: int = MIN_DAYS_CONTRAST,
) -> dict[str, Any]:
    """Compare le tiers de vos meilleurs jours au tiers des pires.

    L'application WHOOP indique votre score ; elle ne dit pas ce que vous
    faisiez différemment les jours où il était bon. La comparaison par tiers
    répond à cette question sans supposer de relation linéaire.
    """
    columns = ["Facteur", "Meilleurs jours", "Pires jours", "Écart", "Écart normalisé"]
    result = {
        "ready": False,
        "days": 0,
        "required_days": int(min_days),
        "table": pd.DataFrame(columns=columns),
        "best_threshold": float("nan"),
        "worst_threshold": float("nan"),
    }
    grid = daily_grid(frame)
    if grid.empty or metric not in grid.columns:
        return result

    # Les décalages sont appliqués sur la grille complète, avant tout filtrage :
    # calculés après, ils prendraient « la ligne précédente » au lieu de « la
    # veille », ce qui n'est pas la même chose dès qu'un jour manque.
    for label, column, lag, _phrase in CONTRAST_FACTORS:
        if column in grid.columns:
            grid[label] = grid[column].shift(int(lag)) if lag else grid[column]

    scored = grid.dropna(subset=[metric])
    result["days"] = int(len(scored))
    if len(scored) < max(6, int(min_days)):
        return result

    worst_threshold = float(scored[metric].quantile(1 / 3))
    best_threshold = float(scored[metric].quantile(2 / 3))
    if not np.isfinite(worst_threshold) or not np.isfinite(best_threshold) or best_threshold <= worst_threshold:
        return result

    best = scored[scored[metric] >= best_threshold]
    worst = scored[scored[metric] <= worst_threshold]
    if len(best) < 3 or len(worst) < 3:
        return result

    rows = []
    for factor, source, _lag, _phrase in CONTRAST_FACTORS:
        if factor not in scored.columns or source not in grid.columns:
            continue
        best_values, worst_values = best[factor].dropna(), worst[factor].dropna()
        if len(best_values) < 3 or len(worst_values) < 3:
            continue
        best_mean, worst_mean = float(best_values.mean()), float(worst_values.mean())
        if not np.isfinite(best_mean) or not np.isfinite(worst_mean):
            continue
        # Comparer des heures de sommeil à des points de strain par leur écart
        # brut revient à classer par unité de mesure : c'est l'échelle qui
        # gagnerait, pas l'effet. L'écart est donc rapporté à la dispersion.
        spread = float(np.sqrt((best_values.var(ddof=1) + worst_values.var(ddof=1)) / 2.0))
        effect = (best_mean - worst_mean) / spread if np.isfinite(spread) and spread > 0 else float("nan")
        rows.append(
            {
                "Facteur": factor,
                "Meilleurs jours": round(best_mean, 2),
                "Pires jours": round(worst_mean, 2),
                "Écart": round(best_mean - worst_mean, 2),
                "Écart normalisé": round(effect, 2) if np.isfinite(effect) else float("nan"),
            }
        )

    if not rows:
        return result

    table = pd.DataFrame(rows)
    ranking = table["Écart normalisé"].abs().fillna(-1)
    table = table.reindex(ranking.sort_values(ascending=False).index).reset_index(drop=True)
    result.update(
        {
            "ready": True,
            "table": table[columns],
            "best_threshold": best_threshold,
            "worst_threshold": worst_threshold,
            "best_days": int(len(best)),
            "worst_days": int(len(worst)),
        }
    )
    return result


def sport_recovery_impact(
    daily: pd.DataFrame | None,
    workouts: pd.DataFrame | None,
    *,
    min_sessions: int = MIN_SESSIONS_PER_SPORT,
) -> pd.DataFrame:
    """Récupération du lendemain, sport par sport.

    Savoir quelle activité vous coûte le plus le jour suivant est exactement ce
    qu'un score quotidien isolé ne dit pas.
    """
    columns = ["Sport", "Séances", "Récupération du lendemain (%)", "Écart à votre moyenne"]
    grid = daily_grid(daily)
    if grid.empty or "Récupération (%)" not in grid.columns or workouts is None or workouts.empty:
        return pd.DataFrame(columns=columns)
    if "Sport" not in workouts.columns or "Date" not in workouts.columns:
        return pd.DataFrame(columns=columns)

    recovery_by_day = {
        pd.Timestamp(date).normalize(): value
        for date, value in zip(grid["Date"], grid["Récupération (%)"])
        if pd.notna(value)
    }
    overall = float(np.mean(list(recovery_by_day.values()))) if recovery_by_day else float("nan")
    if not np.isfinite(overall):
        return pd.DataFrame(columns=columns)

    sessions = workouts.copy()
    sessions["Date"] = pd.to_datetime(sessions["Date"], errors="coerce").dt.normalize()
    sessions = sessions.dropna(subset=["Date", "Sport"])

    collected: dict[str, list[float]] = {}
    for sport, date in zip(sessions["Sport"], sessions["Date"]):
        # Le lendemain porte la trace de l'effort : le score du jour même a été
        # calculé avant la séance.
        following = recovery_by_day.get(pd.Timestamp(date) + pd.Timedelta(days=1))
        if following is None:
            continue
        collected.setdefault(str(sport), []).append(float(following))

    rows = []
    for sport, values in collected.items():
        if len(values) < max(2, int(min_sessions)):
            continue
        mean_value = float(np.mean(values))
        rows.append(
            {
                "Sport": sport,
                "Séances": int(len(values)),
                "Récupération du lendemain (%)": round(mean_value, 1),
                "Écart à votre moyenne": round(mean_value - overall, 1),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("Écart à votre moyenne").reset_index(drop=True)[columns]


MIN_PAIRS_TOLERANCE = 21


def strain_tolerance(frame: pd.DataFrame | None, *, min_pairs: int = MIN_PAIRS_TOLERANCE) -> dict[str, Any]:
    """À partir de quelle charge votre récupération du lendemain décroche.

    WHOOP affiche un strain visé pour la journée en cours, calculé sur sa
    population de référence. Il ne dit pas où se situe *votre* plafond : la
    charge au-delà de laquelle vos propres lendemains virent au rouge. Les
    journées sont donc rangées par tiers de charge, et chaque tiers est jugé
    sur la récupération qu'il a effectivement produite le lendemain matin.
    """
    columns = ["Charge de la veille", "Jours", "Strain moyen", "Récupération du lendemain (%)", "Journées rouges (%)"]
    result: dict[str, Any] = {
        "ready": False,
        # Un déclin mesuré, distinct de « assez de données » : trois bandes
        # peuplées n'impliquent pas que les journées chargées coûtent quoi que
        # ce soit.
        "declines": False,
        # La significativité ne dit rien du SENS : un écart inverse peut être
        # parfaitement établi. Les confondre fait annoncer « indistinguable
        # du hasard » sur un résultat mesuré à p = 10⁻¹³.
        "significant": False,
        "pairs": 0,
        "required_pairs": int(min_pairs),
        # Pourquoi l'analyse ne s'affiche pas : « effectif » se comble avec le
        # temps, « charge trop uniforme » non.
        "reason": "effectif",
        "table": pd.DataFrame(columns=columns),
        # Bord inférieur du tiers haut de VOTRE distribution de charge, et non
        # un point de rupture estimé sur la récupération : il se déplace si vous
        # vous mettez à vous entraîner davantage, à réponse physiologique
        # inchangée. Le libellé affiché doit donc rester descriptif.
        "high_band_floor": float("nan"),
        "heavy_recovery": float("nan"),
        "calm_recovery": float("nan"),
        "heavy_red_share": float("nan"),
        "gap": float("nan"),
        "gap_low": float("nan"),
        "gap_high": float("nan"),
        "p_value": float("nan"),
        "inference": "testée",
    }
    grid = daily_grid(frame)
    if grid.empty or "Strain" not in grid.columns or "Récupération (%)" not in grid.columns:
        return result

    # La charge d'un jour se juge sur le matin suivant : c'est le seul moment où
    # son effet est mesurable par un score de récupération.
    paired = pd.DataFrame(
        {"strain": grid["Strain"], "next_recovery": grid["Récupération (%)"].shift(-1)}
    ).dropna()
    result["pairs"] = int(len(paired))
    if len(paired) < max(9, int(min_pairs)):
        result["reason"] = "effectif"
        return result

    low = float(paired["strain"].quantile(1 / 3))
    high = float(paired["strain"].quantile(2 / 3))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        # Assez de jours, mais une charge trop uniforme pour former trois tiers.
        # Réclamer « plus de jours » serait une consigne qu'aucune journée
        # supplémentaire de ce type ne pourrait satisfaire.
        result["reason"] = "charge trop uniforme"
        return result

    red_ceiling = RECOVERY_ZONES[0][2]
    bands = (
        ("Journées calmes", paired[paired["strain"] <= low]),
        ("Journées moyennes", paired[(paired["strain"] > low) & (paired["strain"] < high)]),
        ("Journées chargées", paired[paired["strain"] >= high]),
    )
    rows = []
    for label, band in bands:
        if band.empty:
            continue
        rows.append(
            {
                "Charge de la veille": label,
                "Jours": int(len(band)),
                "Strain moyen": round(float(band["strain"].mean()), 1),
                "Récupération du lendemain (%)": round(float(band["next_recovery"].mean()), 1),
                "Journées rouges (%)": round(float((band["next_recovery"] < red_ceiling).mean() * 100), 1),
            }
        )
    if len(rows) < 3:
        # Les bornes de tiers peuvent coïncider au point de vider la bande
        # centrale : là encore, c'est la diversité de la charge qui manque.
        result["reason"] = "charge trop uniforme"
        return result

    heavy, calm = rows[-1], rows[0]
    calm_values = bands[0][1]["next_recovery"]
    heavy_values = bands[-1][1]["next_recovery"]

    # Au minimum d'effectif, chaque bande ne compte qu'une poignée de jours :
    # un écart brut de quelques points y naît facilement du hasard. L'écart est
    # donc testé (Welch, variances inégales) et assorti de son intervalle de
    # confiance, plutôt que comparé à un seuil fixe.
    gap = float(calm_values.mean() - heavy_values.mean())
    gap_low = gap_high = p_value = float("nan")
    # Si les deux bandes sont chacune constantes, leurs variances sont nulles :
    # le test de Welch n'est pas défini. Le dire, plutôt que de laisser l'écran
    # conclure « ne se distingue pas du hasard » sur une séparation parfaite.
    inference = "testée"
    if len(calm_values) >= 2 and len(heavy_values) >= 2:
        from scipy import stats

        variance_calm = float(calm_values.var(ddof=1))
        variance_heavy = float(heavy_values.var(ddof=1))
        error = float(np.sqrt(variance_calm / len(calm_values) + variance_heavy / len(heavy_values)))
        if np.isfinite(error) and error > 0:
            numerator = (variance_calm / len(calm_values) + variance_heavy / len(heavy_values)) ** 2
            denominator = (variance_calm / len(calm_values)) ** 2 / (len(calm_values) - 1) + (
                variance_heavy / len(heavy_values)
            ) ** 2 / (len(heavy_values) - 1)
            degrees = numerator / denominator if denominator > 0 else float("nan")
            if np.isfinite(degrees) and degrees > 0:
                p_value = float(2.0 * stats.t.sf(abs(gap) / error, degrees))
                margin = float(stats.t.ppf(1.0 - ALPHA / 2.0, degrees)) * error
                gap_low, gap_high = gap - margin, gap + margin
            else:
                inference = "indisponible"
        elif abs(gap) > 0:
            inference = "indisponible"
    else:
        inference = "indisponible"

    result.update(
        {
            "ready": True,
            "significant": bool(np.isfinite(p_value) and p_value <= ALPHA),
            "declines": bool(gap > 0 and np.isfinite(p_value) and p_value <= ALPHA),
            "table": pd.DataFrame(rows)[columns],
            "high_band_floor": round(high, 1),
            "heavy_recovery": heavy["Récupération du lendemain (%)"],
            "calm_recovery": calm["Récupération du lendemain (%)"],
            "heavy_red_share": heavy["Journées rouges (%)"],
            "heavy_days": heavy["Jours"],
            "calm_days": calm["Jours"],
            "gap": round(gap, 1),
            "gap_low": round(gap_low, 1) if np.isfinite(gap_low) else float("nan"),
            "gap_high": round(gap_high, 1) if np.isfinite(gap_high) else float("nan"),
            "p_value": p_value,
            "inference": inference,
        }
    )
    return result


def _tolerance_insight(daily: pd.DataFrame | None) -> Insight | None:
    tolerance = strain_tolerance(daily)
    # « declines » exige que l'écart survive à un test de Welch. Un écart brut
    # de trois points suffisait auparavant : mesuré sur des séries sans aucun
    # lien entre charge et récupération, il se déclenchait dans 31,8 % des cas,
    # contre 2,8 % avec le test.
    if not tolerance["declines"]:
        return None
    gap = tolerance["gap"]
    low, high = tolerance["gap_low"], tolerance["gap_high"]
    interval = (
        f" L'écart est estimé entre {_fr(low, 0)} et {_fr(high, 0)} points."
        if np.isfinite(low) and np.isfinite(high)
        else ""
    )
    return Insight(
        "Vos journées les plus chargées se paient le lendemain",
        f"Au-dessus de {_fr(tolerance['high_band_floor'])} de strain — le tiers le plus chargé de vos "
        f"journées — votre récupération du lendemain tombe à {_fr(tolerance['heavy_recovery'], 0)} % en "
        f"moyenne sur {tolerance['heavy_days']} jours, contre {_fr(tolerance['calm_recovery'], 0)} % après "
        f"vos {tolerance['calm_days']} journées les plus calmes, soit {_fr(gap, 0)} points d'écart."
        + interval
        # Annoncer « rouges 0 % du temps » occupe une phrase pour ne rien
        # apprendre : la mention n'a de sens que si le cas s'est produit.
        + (
            f" Ces lendemains sont rouges {_fr(tolerance['heavy_red_share'], 0)} % du temps."
            if np.isfinite(tolerance["heavy_red_share"]) and tolerance["heavy_red_share"] > 0
            else ""
        ),
        tone="warning" if gap >= 8 else "info",
        icon="🧗",
        priority=71,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Signes vitaux nocturnes
# ──────────────────────────────────────────────────────────────────────────────

# Les cinq grandeurs que WHOOP mesure pendant le sommeil. Pour chacune, le sens
# de l'écart qui mérite attention : une fréquence cardiaque de repos qui monte et
# une variabilité qui descend sont les signatures classiques d'une charge mal
# absorbée ou d'une infection qui débute.
NOCTURNAL_VITALS: tuple[tuple[str, str, str], ...] = (
    ("FC repos (bpm)", "haut", "élevée"),
    ("HRV (ms)", "bas", "basse"),
    ("Fréquence respiratoire (resp/min)", "haut", "élevée"),
    ("Température peau (°C)", "haut", "élevée"),
    ("SpO2 (%)", "bas", "basse"),
)

# Nuits minimales avant de parler d'écart : un repère construit sur moins que
# cela décrirait surtout la dernière nuit elle-même.
MIN_NIGHTS_VITALS = 10
# Seuil d'écart, en unités d'écart robuste. Mesuré sur 200 séries de bruit :
# à 2,0 un signal apparaît une nuit sur cinq sans rien à signaler, à 2,5 une
# nuit sur treize, et deux signaux simultanés n'apparaissent jamais. Un
# indicateur de santé qui crie au loup finit ignoré : c'est 2,5 qui est retenu.
VITAL_DEVIATION_THRESHOLD = 2.5
# Facteur ramenant l'écart absolu médian à une échelle comparable à un
# écart-type sous une distribution normale.
MAD_TO_SIGMA = 1.4826


def _robust_scale(values: pd.Series) -> float:
    """Dispersion résistante aux valeurs isolées.

    Sur dix à trente nuits, une seule nuit aberrante gonfle l'écart-type et
    masque ensuite tout écart réel. L'écart absolu médian ne bouge pas.
    """
    if len(values) < 3:
        return float("nan")
    median = float(values.median())
    mad = float((values - median).abs().median())
    return mad * MAD_TO_SIGMA if mad > 0 else float("nan")


def vital_deviations(
    frame: pd.DataFrame | None,
    *,
    days: int = BASELINE_DAYS,
    min_nights: int = MIN_NIGHTS_VITALS,
    threshold: float = VITAL_DEVIATION_THRESHOLD,
) -> pd.DataFrame:
    """Écart de la dernière nuit au repère personnel, signe vital par signe vital.

    Les valeurs absolues de ces grandeurs varient énormément d'une personne à
    l'autre : seule la comparaison à sa propre habitude est interprétable.
    """
    columns = ["Signe vital", "Dernière nuit", "Repère habituel", "Écart", "Sens", "Inhabituel"]
    data = _clean_daily(frame)
    if data.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for metric, concerning_direction, concerning_label in NOCTURNAL_VITALS:
        if metric not in data.columns:
            continue
        measured = data[["Date", metric]].dropna()
        if len(measured) < max(3, int(min_nights)):
            continue

        latest = float(measured[metric].iloc[-1])
        latest_date = measured["Date"].iloc[-1]
        history = measured.iloc[:-1]
        history = history[history["Date"] >= latest_date - pd.Timedelta(days=max(1, int(days)))]
        if len(history) < 3:
            continue

        baseline = float(history[metric].median())
        scale = _robust_scale(history[metric])
        if not np.isfinite(scale):
            continue

        deviation = (latest - baseline) / scale
        above = deviation > 0
        unusual = abs(deviation) >= float(threshold) and (
            (concerning_direction == "haut" and above) or (concerning_direction == "bas" and not above)
        )
        rows.append(
            {
                "Signe vital": metric,
                "Dernière nuit": round(latest, 2),
                "Repère habituel": round(baseline, 2),
                "Écart": round(deviation, 2),
                "Sens": concerning_label if unusual else ("au-dessus" if above else "en dessous"),
                "Inhabituel": bool(unusual),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    frame_out = pd.DataFrame(rows)
    ranking = frame_out["Écart"].abs()
    return frame_out.reindex(ranking.sort_values(ascending=False).index).reset_index(drop=True)[columns]


def physiological_watch(frame: pd.DataFrame | None, **kwargs: Any) -> dict[str, Any]:
    """Combien de signes vitaux sortent simultanément de leur habitude.

    Plusieurs signes qui dévient ensemble pèsent davantage qu'un seul, sans que
    cela constitue pour autant un diagnostic : ce sont des mesures de bracelet,
    et leur interprétation appartient à un professionnel de santé.
    """
    table = vital_deviations(frame, **kwargs)
    result = {
        "ready": not table.empty,
        "table": table,
        "flagged": [],
        "count": 0,
        "level": "indisponible",
        "date": None,
    }
    if table.empty:
        return result

    data = _clean_daily(frame)
    result["date"] = data["Date"].max() if not data.empty else None
    flagged = table[table["Inhabituel"]]
    result["flagged"] = [
        {"Signe vital": row["Signe vital"], "Sens": row["Sens"], "Écart": row["Écart"]}
        for _, row in flagged.iterrows()
    ]
    result["count"] = int(len(flagged))
    if result["count"] == 0:
        result["level"] = "aucun signal"
    elif result["count"] == 1:
        result["level"] = "un signal isolé"
    else:
        result["level"] = "plusieurs signaux concordants"
    return result


def _vitals_insight(daily: pd.DataFrame | None) -> Insight | None:
    """Signale une nuit physiologiquement atypique, sans jamais conclure."""
    watch = physiological_watch(daily)
    if not watch["ready"] or watch["count"] == 0:
        return None

    names = ", ".join(f"{item['Signe vital'].split(' (')[0].lower()} {item['Sens']}" for item in watch["flagged"])
    several = watch["count"] > 1
    return Insight(
        f"{watch['count']} {_plural(watch['count'], 'signe vital', 'signes vitaux')} hors de votre habitude",
        f"La dernière nuit montre : {names}. Ces écarts sont mesurés par rapport à votre propre "
        "repère des trente derniers jours, pas à une norme. Un bracelet ne diagnostique rien ; "
        + (
            "plusieurs signes qui dévient ensemble méritent toutefois d'être signalés à un "
            "professionnel de santé s'ils persistent."
            if several
            else "un signe isolé s'explique souvent par une soirée tardive, un repas copieux ou l'alcool."
        ),
        tone="warning" if several else "info",
        icon="🩺",
        priority=88 if several else 58,
        pinned=several,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Journal jour par jour
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DaySession:
    """Une séance, située dans sa journée par son heure de début."""

    start: Any
    sport: str
    duration_min: float
    strain: float
    calories: float
    average_hr: float
    max_hr: float


@dataclass(frozen=True)
class DayEntry:
    """Tout ce qui s'est passé un jour donné, réuni sur une seule ligne de lecture."""

    date: Any
    recovery: float
    zone: str | None
    hrv: float
    resting_hr: float
    sleep_hours: float
    sleep_debt: float
    bedtime: float
    strain: float
    calories: float
    weight: float
    weight_change: float
    sessions: tuple[DaySession, ...] = ()

    @property
    def has_measurement(self) -> bool:
        return any(
            np.isfinite(value)
            for value in (self.recovery, self.sleep_hours, self.strain, self.weight)
        ) or bool(self.sessions)


def _value(row: Mapping[str, Any] | pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    try:
        numeric = float(row[column])
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _zone_of(recovery: float) -> str | None:
    if not np.isfinite(recovery):
        return None
    return next((label for label, low, high, _ in RECOVERY_ZONES if low <= recovery < high), None)


def daily_log(
    daily: pd.DataFrame | None,
    workouts: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    *,
    newest_first: bool = True,
    limit: int | None = None,
) -> list[DayEntry]:
    """Journal chronologique : une entrée par jour, séances rattachées à leur date.

    Les graphiques et les moyennes répondent à « comment ça évolue » ; ils ne
    répondent jamais à « que s'est-il passé mardi ». C'est ce que ce journal
    rétablit, en réunissant récupération, sommeil, charge, poids et séances du
    jour sur une même ligne de lecture.
    """
    grid = daily_grid(daily)
    if grid.empty:
        return []

    weight_by_day: dict[Any, tuple[float, float]] = {}
    if weights is not None and not weights.empty and "Poids (Kgs)" in weights.columns:
        weight_frame = _clean_daily(weights)[["Date", "Poids (Kgs)"]].dropna()
        if not weight_frame.empty:
            averaged = weight_frame.groupby("Date", as_index=False)["Poids (Kgs)"].mean()
            averaged["_change"] = averaged["Poids (Kgs)"].diff()
            weight_by_day = {
                pd.Timestamp(date): (float(value), float(change) if pd.notna(change) else float("nan"))
                for date, value, change in zip(averaged["Date"], averaged["Poids (Kgs)"], averaged["_change"])
            }

    sessions_by_day: dict[Any, list[DaySession]] = {}
    if workouts is not None and not workouts.empty and "Date" in workouts.columns:
        session_frame = workouts.copy()
        session_frame["Date"] = pd.to_datetime(session_frame["Date"], errors="coerce").dt.normalize()
        session_frame = session_frame.dropna(subset=["Date"])
        if "Début" in session_frame.columns:
            session_frame = session_frame.sort_values(["Date", "Début"], kind="mergesort")
        for row in session_frame.to_dict("records"):
            sessions_by_day.setdefault(pd.Timestamp(row["Date"]), []).append(
                DaySession(
                    start=row.get("Début"),
                    sport=str(row.get("Sport") or "Séance"),
                    duration_min=_value(row, "Durée (min)"),
                    strain=_value(row, "Strain séance"),
                    calories=_value(row, "Calories séance (kcal)"),
                    average_hr=_value(row, "FC moyenne (bpm)"),
                    max_hr=_value(row, "FC max (bpm)"),
                )
            )

    entries: list[DayEntry] = []
    for row in grid.to_dict("records"):
        date = pd.Timestamp(row["Date"])
        recovery = _value(row, "Récupération (%)")
        weight, change = weight_by_day.get(date, (float("nan"), float("nan")))
        entries.append(
            DayEntry(
                date=date,
                recovery=recovery,
                zone=_zone_of(recovery),
                hrv=_value(row, "HRV (ms)"),
                resting_hr=_value(row, "FC repos (bpm)"),
                sleep_hours=_value(row, "Sommeil (heures)"),
                sleep_debt=_value(row, "Dette de sommeil (heures)"),
                bedtime=_value(row, "Heure de coucher"),
                strain=_value(row, "Strain"),
                calories=_value(row, "Calories (kcal)"),
                weight=weight,
                weight_change=change,
                sessions=tuple(sessions_by_day.get(date, ())),
            )
        )

    if newest_first:
        entries.reverse()
    if limit is not None:
        entries = entries[: max(1, int(limit))]
    return entries


def daily_log_table(entries: Sequence[DayEntry]) -> pd.DataFrame:
    """Contrepartie tabulaire du journal, exportable et lisible au clavier."""
    columns = [
        "Date",
        "Récupération (%)",
        "Sommeil (heures)",
        "Dette de sommeil (heures)",
        "Strain",
        "Poids (Kgs)",
        "Séances",
    ]
    if not entries:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [
            {
                "Date": entry.date,
                "Récupération (%)": entry.recovery,
                "Sommeil (heures)": entry.sleep_hours,
                "Dette de sommeil (heures)": entry.sleep_debt,
                "Strain": entry.strain,
                "Poids (Kgs)": entry.weight,
                "Séances": len(entry.sessions),
            }
            for entry in entries
        ]
    )[columns]


# ──────────────────────────────────────────────────────────────────────────────
# Architecture du sommeil
# ──────────────────────────────────────────────────────────────────────────────

# Plages usuellement citées chez l'adulte pour la part de chaque stade dans une
# nuit. Ce sont des repères de population, pas des objectifs personnels : une
# nuit hors plage n'a pas de signification isolée.
SLEEP_STAGE_REFERENCES: tuple[tuple[str, str, float, float], ...] = (
    ("Sommeil profond (heures)", "Sommeil profond", 13.0, 23.0),
    ("Sommeil REM (heures)", "Sommeil REM", 20.0, 25.0),
)

MIN_NIGHTS_ARCHITECTURE = 5


def sleep_architecture(
    frame: pd.DataFrame | None,
    *,
    days: int = BASELINE_DAYS,
    min_nights: int = MIN_NIGHTS_ARCHITECTURE,
) -> dict[str, Any]:
    """Part de sommeil profond et de sommeil REM, rapportée aux repères adultes.

    WHOOP affiche des heures de stades ; c'est leur *proportion* dans la nuit
    qui se compare d'une nuit à l'autre et à une référence, une nuit courte
    réduisant mécaniquement les heures de chaque stade.
    """
    columns = ["Stade", "Votre part (%)", "Plage usuelle", "Position"]
    result = {
        "ready": False,
        "nights": 0,
        "required_nights": int(min_nights),
        "table": pd.DataFrame(columns=columns),
        "mean_sleep": float("nan"),
    }
    data = _clean_daily(frame)
    if data.empty or "Sommeil (heures)" not in data.columns:
        return result

    window = last_days(data, days)
    usable = window.dropna(subset=["Sommeil (heures)"])
    usable = usable[usable["Sommeil (heures)"] > 0]
    result["nights"] = int(len(usable))
    if len(usable) < max(3, int(min_nights)):
        return result
    result["mean_sleep"] = float(usable["Sommeil (heures)"].mean())

    rows = []
    for column, label, low, high in SLEEP_STAGE_REFERENCES:
        if column not in usable.columns:
            continue
        pair = usable[["Sommeil (heures)", column]].dropna()
        if len(pair) < 3:
            continue
        # Part moyenne des nuits, et non part de la somme : une nuit très longue
        # ne doit pas peser davantage qu'une nuit courte dans la moyenne.
        shares = (pair[column] / pair["Sommeil (heures)"] * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
        if shares.empty:
            continue
        share = float(shares.mean())
        if share < low:
            position = "sous la plage"
        elif share > high:
            position = "au-dessus de la plage"
        else:
            position = "dans la plage"
        rows.append(
            {
                "Stade": label,
                "Votre part (%)": round(share, 1),
                "Plage usuelle": f"{low:.0f} à {high:.0f} %",
                "Position": position,
            }
        )

    if not rows:
        return result
    result["table"] = pd.DataFrame(rows)[columns]
    result["ready"] = True
    return result


def _architecture_insight(daily: pd.DataFrame | None) -> Insight | None:
    """Signale un stade durablement hors plage, sans en faire un objectif."""
    architecture = sleep_architecture(daily)
    if not architecture["ready"]:
        return None
    outside = architecture["table"][architecture["table"]["Position"] != "dans la plage"]
    if outside.empty:
        return None

    first = outside.iloc[0]
    return Insight(
        f"{first['Stade']} {first['Position']}",
        f"Sur {architecture['nights']} {_plural(architecture['nights'], 'nuit')}, ce stade représente "
        f"{_fr(float(first['Votre part (%)']))} % de votre sommeil, contre {first['Plage usuelle']} "
        "couramment cités chez l'adulte. Ces plages décrivent une population, pas un objectif "
        f"personnel ; allonger la nuit (actuellement {_fr(architecture['mean_sleep'])} h en moyenne) "
        "augmente généralement les deux stades en valeur absolue.",
        tone="info",
        icon="🌙",
        priority=50,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Où va le poids, et quand la cible serait atteinte
# ──────────────────────────────────────────────────────────────────────────────

# Sous ce rythme hebdomadaire, la courbe ne se distingue pas du bruit de pesée.
PLATEAU_KG_PER_WEEK = 0.1
# Qualité minimale de l'ajustement avant d'extrapoler une date d'arrivée : une
# droite qui n'explique rien ne peut pas fixer d'échéance.
MIN_TREND_FIT = 0.3
MIN_DAYS_PROJECTION = 14


def projected_goal_date(
    merged: pd.DataFrame | None,
    *,
    target_weight: float,
    today: Any = None,
) -> dict[str, Any]:
    """Date d'arrivée à la cible si le rythme actuel se maintenait.

    Une extrapolation linéaire n'est pas une prédiction : elle répond à « et si
    ça continuait ainsi », ce qui suffit à situer un objectif comme proche ou
    hors d'atteinte.
    """
    result = {
        "ready": False,
        "date": None,
        "days": float("nan"),
        "slope_kg_per_week": float("nan"),
        "current_weight": float("nan"),
        "target_weight": float(target_weight),
        "fit": float("nan"),
        "reason": "indisponible",
    }
    data = _clean_daily(merged)
    if data.empty or "Poids (Kgs)" not in data.columns:
        return result

    weighed = data.dropna(subset=["Poids (Kgs)"])
    if len(weighed) < MIN_DAYS_PROJECTION:
        result["reason"] = "historique trop court"
        return result

    trend = weight_trend(weighed)
    result["slope_kg_per_week"] = trend["slope_kg_per_week"]
    result["fit"] = trend["r_squared"]
    current = float(weighed["Poids (Kgs)"].iloc[-1])
    result["current_weight"] = current

    slope = trend["slope_kg_per_day"]
    if not np.isfinite(slope) or not np.isfinite(trend["r_squared"]) or trend["r_squared"] < MIN_TREND_FIT:
        result["reason"] = "tendance trop irrégulière"
        return result

    remaining = current - float(target_weight)
    # Le signe de la pente doit aller vers la cible, sinon l'échéance n'existe pas.
    if remaining <= 0:
        result["reason"] = "cible atteinte"
        result["ready"] = True
        result["days"] = 0.0
        result["date"] = pd.Timestamp(today).normalize() if today is not None else weighed["Date"].max()
        return result
    if slope >= 0:
        result["reason"] = "le poids ne va pas vers la cible"
        return result

    days = remaining / abs(slope)
    if days > 3650:
        result["reason"] = "échéance au-delà de dix ans"
        return result

    reference = pd.Timestamp(today).normalize() if today is not None else weighed["Date"].max()
    result.update({"ready": True, "days": float(days), "date": reference + pd.Timedelta(days=round(days)), "reason": "estimée"})
    return result


def _weight_direction_insight(merged: pd.DataFrame | None) -> Insight | None:
    """Dit d'abord où va le poids : c'est la question que pose l'application.

    Sans cette règle, une prise de poids de deux kilos pouvait passer inaperçue
    derrière des constats sur le sommeil du lundi.
    """
    data = _clean_daily(merged)
    if data.empty or "Poids (Kgs)" not in data.columns:
        return None
    weighed = data.dropna(subset=["Poids (Kgs)"])
    if len(weighed) < 7:
        return None

    trend = weight_trend(weighed)
    weekly = trend["slope_kg_per_week"]
    if not np.isfinite(weekly):
        return None

    span = trend["span_days"]
    reliability = (
        "" if not np.isfinite(trend["r_squared"]) else
        " La tendance est nette." if trend["r_squared"] >= 0.6 else
        " Les pesées sont dispersées autour de cette tendance : le chiffre est indicatif."
    )

    if abs(weekly) < PLATEAU_KG_PER_WEEK:
        return Insight(
            "Votre poids stagne",
            f"Sur {span} {_plural(span, 'jour')}, la tendance est de {_fr(weekly, 2, sign=True)} kg par semaine, "
            f"soit un poids stable autour de {_fr(float(weighed['Poids (Kgs)'].iloc[-1]))} kg." + reliability,
            tone="warning",
            icon="⚖️",
            priority=92,
        )
    gaining = weekly > 0
    return Insight(
        "Votre poids augmente" if gaining else "Votre poids baisse",
        f"Sur {span} {_plural(span, 'jour')}, la tendance est de {_fr(weekly, 2, sign=True)} kg par semaine "
        f"(actuellement {_fr(float(weighed['Poids (Kgs)'].iloc[-1]))} kg)." + reliability,
        tone="warning" if gaining else "success",
        # La balance désigne le poids sans ambiguïté : les flèches de tendance
        # servent déjà à la récupération, et deux cartes partageant une icône
        # ne se distinguent plus d'un coup d'œil.
        icon="⚖️",
        # Devance le bilan énergétique : ce dernier est une estimation dérivée
        # de cette même pente, elle ne peut pas la précéder.
        priority=94 if gaining else 91,
    )


def _target_progress_insight(
    merged: pd.DataFrame | None,
    target_status: Mapping[str, Any] | None,
    *,
    target_weight: float | None = None,
    target_date: Any = None,
) -> Insight | None:
    """Situe le poids face à la trajectoire cible, et projette la date d'arrivée."""
    if not target_status or not target_status.get("status"):
        return None
    gap = target_status.get("gap_kg")
    status = str(target_status.get("status"))
    if gap is None or not np.isfinite(float(gap)):
        return None

    projection = (
        projected_goal_date(merged, target_weight=float(target_weight))
        if target_weight is not None
        else {"ready": False, "reason": "indisponible"}
    )
    if projection["ready"] and projection["date"] is not None:
        projected = pd.Timestamp(projection["date"])
        horizon = (
            f" Au rythme actuel, la cible de {_fr(float(target_weight))} kg serait atteinte vers le "
            f"{projected.strftime('%d/%m/%Y')}"
        )
        # Une date d'arrivée ne devient actionnable que confrontée à l'échéance.
        deadline = pd.Timestamp(target_date) if target_date is not None else None
        if deadline is not None and pd.notna(deadline):
            gap_days = int((projected - deadline.normalize()).days)
            if gap_days > 7:
                horizon += f", soit {gap_days} {_plural(gap_days, 'jour')} après l'échéance visée."
            elif gap_days < -7:
                horizon += f", soit {abs(gap_days)} {_plural(abs(gap_days), 'jour')} avant l'échéance visée."
            else:
                horizon += ", soit à peu près à l'échéance visée."
        else:
            horizon += "."
    elif projection.get("reason") == "le poids ne va pas vers la cible":
        horizon = " Au rythme actuel, la cible ne serait jamais atteinte."
    elif projection.get("reason") == "tendance trop irrégulière":
        horizon = " Les pesées sont trop dispersées pour projeter une date d'arrivée."
    else:
        horizon = ""

    behind = status == "en retard"
    return Insight(
        f"Trajectoire : {status}",
        f"Vous êtes à {_fr(abs(float(gap)), 2)} kg "
        + ("au-dessus" if float(gap) > 0 else "en dessous")
        + " du poids prévu par votre trajectoire cible à cette date."
        + horizon,
        tone="warning" if behind else "success",
        icon="🧭",
        priority=93 if behind else 84,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Part de l'entraînement et séries de journées
# ──────────────────────────────────────────────────────────────────────────────

MIN_DAYS_TRAINING_SHARE = 7


def training_energy_share(
    daily: pd.DataFrame | None,
    workouts: pd.DataFrame | None,
    *,
    min_days: int = MIN_DAYS_TRAINING_SHARE,
) -> dict[str, Any]:
    """Part de la dépense quotidienne attribuable aux séances.

    Les calories s'additionnent, contrairement au strain qui est une échelle
    logarithmique : cette part-là se calcule honnêtement. Elle répond à une
    question que WHOOP ne pose pas — « mon sport pèse-t-il vraiment dans ma
    dépense, ou est-ce mon quotidien qui la porte ? »
    """
    result = {
        "ready": False,
        "days": 0,
        "required_days": int(min_days),
        "mean_daily_burn": float("nan"),
        "mean_session_burn": float("nan"),
        "share_pct": float("nan"),
        "session_days": 0,
    }
    grid = daily_grid(daily)
    if grid.empty or "Calories (kcal)" not in grid.columns:
        return result

    burn = grid[["Date", "Calories (kcal)"]].dropna()
    result["days"] = int(len(burn))
    if len(burn) < max(3, int(min_days)):
        return result
    result["mean_daily_burn"] = float(burn["Calories (kcal)"].mean())

    session_total = pd.Series(0.0, index=burn["Date"].to_numpy(), dtype=float)
    if workouts is not None and not workouts.empty and "Calories séance (kcal)" in workouts.columns:
        sessions = workouts.copy()
        sessions["Date"] = pd.to_datetime(sessions["Date"], errors="coerce").dt.normalize()
        sessions = sessions.dropna(subset=["Date", "Calories séance (kcal)"])
        if not sessions.empty:
            grouped = sessions.groupby("Date")["Calories séance (kcal)"].sum()
            session_total = session_total.add(grouped.reindex(session_total.index).fillna(0.0), fill_value=0.0)
            result["session_days"] = int((grouped > 0).sum())

    result["mean_session_burn"] = float(session_total.mean())
    if result["mean_daily_burn"] > 0:
        result["share_pct"] = result["mean_session_burn"] / result["mean_daily_burn"] * 100.0
        result["ready"] = True
    return result


def recovery_streaks(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Séries de journées consécutives dans la même zone de récupération.

    Une succession de journées rouges ne se lit pas sur une moyenne ; elle se
    lit sur la série, qui dit combien de temps l'état dure.
    """
    result = {
        "ready": False,
        "current_zone": None,
        "current_length": 0,
        "longest_green": 0,
        "longest_red": 0,
        "days": 0,
    }
    grid = daily_grid(frame)
    if grid.empty or "Récupération (%)" not in grid.columns:
        return result

    scored = grid.dropna(subset=["Récupération (%)"])
    result["days"] = int(len(scored))
    if scored.empty:
        return result

    zones = [_zone_of(float(value)) for value in scored["Récupération (%)"]]
    longest = {"Vert": 0, "Rouge": 0}
    run_zone, run_length = None, 0
    for zone in zones:
        if zone == run_zone:
            run_length += 1
        else:
            run_zone, run_length = zone, 1
        if run_zone in longest:
            longest[run_zone] = max(longest[run_zone], run_length)

    result.update(
        {
            "ready": True,
            "current_zone": run_zone,
            "current_length": run_length,
            "longest_green": longest["Vert"],
            "longest_red": longest["Rouge"],
        }
    )
    return result


def _streak_insight(daily: pd.DataFrame | None) -> Insight | None:
    """Signale une série de journées rouges, qu'aucune moyenne ne fait ressortir."""
    streaks = recovery_streaks(daily)
    if not streaks["ready"] or streaks["current_zone"] != "Rouge" or streaks["current_length"] < 3:
        return None
    length = streaks["current_length"]
    return Insight(
        f"{length} {_plural(length, 'journée')} de suite en zone rouge",
        f"Votre récupération reste sous 34 % depuis {length} {_plural(length, 'jour')}. "
        "Une moyenne hebdomadaire lisse ce genre de série ; la durée, elle, indique un état qui "
        "s'installe plutôt qu'une mauvaise nuit isolée.",
        tone="warning",
        icon="🔻",
        priority=87,
    )
