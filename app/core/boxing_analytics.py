"""Analyses de la boxe à partir des données WHOOP.

L'application WHOOP range une séance de boxe parmi toutes les autres activités
et en affiche le strain, la fréquence cardiaque et les calories. Elle ne dit
pas ce qui intéresse le boxeur au fil des semaines :

- quelle intensité la séance a réellement demandée, rapportée à SA réserve de
  fréquence cardiaque (méthode de Karvonen, celle dont WHOOP tire ses zones) ;
- ce que la boxe coûte à la récupération du lendemain, comparé aux autres jours ;
- s'il boxe plus fort les matins verts, et si les séances tardives abîment la nuit ;
- comment évoluent sa charge, sa régularité et son intensité ;
- ce que la boxe pèse dans la dépense et dans le déficit que vise l'objectif de poids.

Toutes les fonctions sont pures et refusent de conclure sous leur effectif
minimal ; un écart n'est dit « établi » que s'il résiste à un test de Welch
corrigé pour le nombre de mesures comparées (correction de Bonferroni).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from app.core.formatting import format_fr_number
from app.core.whoop import HR_ZONE_COLUMNS, sport_label
from app.core.whoop_analytics import (
    ALPHA,
    KCAL_PER_KG,
    RECOVERY_ZONES,
    Insight,
    daily_grid,
    plural,
    welch_comparison,
)

BOXING_LABEL = "Boxe"
# Sports de combat proposés d'emblée quand aucune séance n'est étiquetée
# « Boxe » : une séance de boxe enregistrée sous « Kickboxing » ou « Arts
# martiaux » resterait sinon invisible.
COMBAT_SPORT_LABELS: tuple[str, ...] = ("Boxe", "Kickboxing", "Muay-thaï", "Arts martiaux")

# Effectifs minimaux.
MIN_SESSIONS_SUMMARY = 1
MIN_GROUP_SIZE = 4
MIN_SESSIONS_PROGRESSION = 6
MIN_SPAN_PROGRESSION_DAYS = 21
MIN_CHRONIC_SESSIONS = 3
MIN_REST_DAYS_BASELINE = 3
MIN_HABIT_SESSIONS = 3

ACUTE_DAYS = 7
CHRONIC_DAYS = 21

# Une séance qui se termine moins de quatre heures avant le coucher est dite
# « tardive » : seuil tiré de Leota et al., « Dose-response relationship
# between evening exercise and sleep », Nature Communications 2025
# (https://doi.org/10.1038/s41467-025-58271-x), étude menée sur 14 689
# porteurs de WHOOP.
LATE_SESSION_HOURS = 4.0
# Sous 80 % de séance captée, strain, calories et zones sont sous-estimés.
MIN_PERCENT_RECORDED = 80.0
INACTIVITY_DAYS = 10

# Plage plausible d'une FC maximale : en dehors, la valeur renvoyée est une erreur.
MAX_HR_PLAUSIBLE = (120.0, 230.0)

# Poids du TRIMP par zones (Edwards) : minutes en zone × numéro de zone. Les
# zones WHOOP sont exprimées en % de réserve de FC et non en % de FC maximale
# comme chez Edwards : le score est une adaptation, comparable d'une séance à
# l'autre chez une même personne, pas d'une personne à l'autre.
TRIMP_WEIGHTS: Mapping[str, int] = {column: index for index, column in enumerate(HR_ZONE_COLUMNS)}

SESSION_COLUMNS: tuple[str, ...] = (
    "Date",
    "Début",
    "Fin",
    "Sport",
    "Durée (min)",
    "Strain séance",
    "Calories séance (kcal)",
    "kcal/min",
    "FC moyenne (bpm)",
    "FC max (bpm)",
    "Intensité (% FCR)",
    "Pic (% FC max)",
    "Zones 4–5 (min)",
    "Part en zones 4–5 (%)",
    "TRIMP",
    "Part enregistrée (%)",
    "Récupération du matin (%)",
    "Zone du matin",
    "Récupération du lendemain (%)",
    "Zone du lendemain",
    "HRV du lendemain (ms)",
    "FC repos du lendemain (bpm)",
    "Sommeil suivant (heures)",
    "Efficacité sommeil suivant (%)",
    "Coucher suivant",
    "Délai avant coucher (h)",
    "Marge avant coucher habituel (h)",
) + HR_ZONE_COLUMNS

WEEKDAY_SHORT: tuple[str, ...] = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")
WEEKDAY_LONG: tuple[str, ...] = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
TIME_SLOTS: tuple[tuple[str, float, float], ...] = (
    ("Matin (avant 12 h)", 0.0, 12.0),
    ("Midi (12–14 h)", 12.0, 14.0),
    ("Après-midi (14–18 h)", 14.0, 18.0),
    ("Soir (18–21 h)", 18.0, 21.0),
    ("Tard (après 21 h)", 21.0, 24.01),
)


def _fr(value: Any, decimals: int = 0, *, sign: bool = False) -> str:
    return format_fr_number(value, decimals=decimals, sign=sign)


def zone_of(recovery: Any) -> str | None:
    """Zone WHOOP d'un score de récupération (rouge < 34, jaune < 67, vert)."""
    try:
        value = float(recovery)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return next((label for label, low, high, _ in RECOVERY_ZONES if low <= value < high), None)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _lookup(grid: pd.DataFrame, column: str) -> dict[pd.Timestamp, float]:
    if grid is None or grid.empty or column not in grid.columns:
        return {}
    values = pd.to_numeric(grid[column], errors="coerce")
    return {pd.Timestamp(date): float(value) for date, value in zip(grid["Date"], values) if pd.notna(value)}


def _clean_values(values: Any) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _today(today: Any = None) -> pd.Timestamp:
    return (pd.Timestamp(today) if today is not None else pd.Timestamp.now()).normalize()


# ──────────────────────────────────────────────────────────────────────────────
# Séances de boxe
# ──────────────────────────────────────────────────────────────────────────────


def available_sports(workouts: pd.DataFrame | None) -> list[str]:
    """Sports présents dans les séances, du plus pratiqué au moins pratiqué, en français."""
    if workouts is None or workouts.empty or "Sport" not in workouts.columns:
        return []
    counts = workouts["Sport"].map(sport_label).value_counts()
    return [str(sport) for sport in counts.index]


def default_sport_selection(sports: Sequence[str]) -> list[str]:
    """La boxe si elle est présente, sinon les sports de combat détectés."""
    if BOXING_LABEL in sports:
        return [BOXING_LABEL]
    return [sport for sport in sports if sport in COMBAT_SPORT_LABELS]


def boxing_sessions(workouts: pd.DataFrame | None, sports: Sequence[str] = (BOXING_LABEL,)) -> pd.DataFrame:
    """Séances dont le sport, rendu en français, figure dans *sports*."""
    if workouts is None or workouts.empty or "Sport" not in workouts.columns or "Date" not in workouts.columns:
        return pd.DataFrame(columns=["Date", "Sport"])
    data = workouts.copy(deep=True)
    data["Sport"] = data["Sport"].map(sport_label)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["Date"])
    selected = data[data["Sport"].isin(list(sports))]
    order = ["Date", "Début"] if "Début" in selected.columns else ["Date"]
    return selected.sort_values(order, kind="mergesort").reset_index(drop=True)


def reference_max_hr(
    body: Mapping[str, Any] | None,
    workouts: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
) -> tuple[float, str]:
    """FC maximale de référence et sa provenance.

    WHOOP calcule ses zones à partir de la FC maximale de votre profil
    (``max_heart_rate`` des mesures corporelles). À défaut, la plus haute FC
    jamais observée sert de borne basse : elle sous-estime la vraie FC
    maximale, et surestime donc légèrement l'intensité.
    """
    low, high = MAX_HR_PLAUSIBLE
    try:
        profile_value = float((body or {}).get("max_heart_rate"))
    except (TypeError, ValueError):
        profile_value = float("nan")
    if np.isfinite(profile_value) and low <= profile_value <= high:
        return profile_value, "profil WHOOP"
    observed = []
    for frame in (workouts, daily):
        if frame is not None and not frame.empty and "FC max (bpm)" in frame.columns:
            values = _clean_values(frame["FC max (bpm)"])
            values = values[(values >= low) & (values <= high)]
            if values.size:
                observed.append(float(values.max()))
    if observed:
        return max(observed), "plus haute FC observée"
    return float("nan"), "indisponible"


def session_table(
    daily: pd.DataFrame | None,
    sessions: pd.DataFrame | None,
    *,
    max_hr: float = float("nan"),
    newest_first: bool = False,
) -> pd.DataFrame:
    """Une ligne par séance, enrichie de ce que WHOOP ne met jamais en regard.

    - **Intensité (% FCR)** : FC moyenne rapportée à la réserve de fréquence
      cardiaque du jour, (FC moy. − FC repos) / (FC max − FC repos), méthode de
      Karvonen et al. (1957) dont WHOOP tire ses zones.
    - **TRIMP** : minutes en zones pondérées par le numéro de zone (Edwards).
    - **Récupération du matin** : le score calculé au réveil, avant la séance,
      et **celle du lendemain**, premier moment où la séance devient mesurable.
    - **Délai avant coucher** : entre la fin de la séance et l'endormissement
      de la nuit suivante.
    """
    if sessions is None or sessions.empty or "Date" not in sessions.columns:
        return pd.DataFrame(columns=list(SESSION_COLUMNS))

    data = sessions.copy(deep=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["Date"]).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(columns=list(SESSION_COLUMNS))

    grid = daily_grid(daily)
    recovery = _lookup(grid, "Récupération (%)")
    hrv = _lookup(grid, "HRV (ms)")
    resting = _lookup(grid, "FC repos (bpm)")
    sleep = _lookup(grid, "Sommeil (heures)")
    efficiency = _lookup(grid, "Efficacité sommeil (%)")
    bedtime = _lookup(grid, "Heure de coucher")
    typical_resting = float(np.median(list(resting.values()))) if resting else float("nan")

    table = pd.DataFrame({"Date": data["Date"]})
    table["Début"] = pd.to_datetime(data["Début"], errors="coerce") if "Début" in data.columns else pd.NaT
    table["Sport"] = data["Sport"].map(sport_label) if "Sport" in data.columns else BOXING_LABEL
    duration = _numeric(data, "Durée (min)")
    table["Durée (min)"] = duration
    table["Fin"] = table["Début"] + pd.to_timedelta(duration, unit="m")
    table["Strain séance"] = _numeric(data, "Strain séance")
    calories = _numeric(data, "Calories séance (kcal)")
    table["Calories séance (kcal)"] = calories
    table["kcal/min"] = calories / duration.where(duration > 0)
    average_hr = _numeric(data, "FC moyenne (bpm)")
    peak_hr = _numeric(data, "FC max (bpm)")
    table["FC moyenne (bpm)"] = average_hr
    table["FC max (bpm)"] = peak_hr

    day_resting = pd.Series([resting.get(pd.Timestamp(date), typical_resting) for date in table["Date"]], index=table.index, dtype=float)
    if np.isfinite(max_hr):
        reserve = (max_hr - day_resting).where(max_hr - day_resting > 0)
        # Une FC moyenne sous la FC de repos est un défaut de capteur, pas une
        # intensité négative : elle est ramenée à zéro.
        table["Intensité (% FCR)"] = ((average_hr - day_resting) / reserve * 100.0).clip(lower=0.0)
        table["Pic (% FC max)"] = peak_hr / max_hr * 100.0
    else:
        table["Intensité (% FCR)"] = np.nan
        table["Pic (% FC max)"] = np.nan

    zones = pd.DataFrame({column: _numeric(data, column) for column in HR_ZONE_COLUMNS})
    for column in HR_ZONE_COLUMNS:
        table[column] = zones[column]
    zone_total = zones.sum(axis=1, min_count=1)
    hard = zones[["Zone 4 (min)", "Zone 5 (min)"]].sum(axis=1, min_count=1)
    # Une séance sans zones reste vide : zéro minute en zone haute est une
    # information, l'absence de mesure n'en est pas une.
    table["Zones 4–5 (min)"] = hard
    table["Part en zones 4–5 (%)"] = hard / zone_total.where(zone_total > 0) * 100.0
    weights = pd.Series(TRIMP_WEIGHTS, dtype=float)
    table["TRIMP"] = (zones[list(weights.index)].fillna(0.0) * weights).sum(axis=1).where(zones.notna().any(axis=1))
    table["Part enregistrée (%)"] = _numeric(data, "Part enregistrée (%)")

    next_days = [pd.Timestamp(date) + pd.Timedelta(days=1) for date in table["Date"]]
    morning = [recovery.get(pd.Timestamp(date), float("nan")) for date in table["Date"]]
    following = [recovery.get(day, float("nan")) for day in next_days]
    table["Récupération du matin (%)"] = morning
    table["Zone du matin"] = [zone_of(value) or "—" for value in morning]
    table["Récupération du lendemain (%)"] = following
    table["Zone du lendemain"] = [zone_of(value) or "—" for value in following]
    table["HRV du lendemain (ms)"] = [hrv.get(day, float("nan")) for day in next_days]
    table["FC repos du lendemain (bpm)"] = [resting.get(day, float("nan")) for day in next_days]
    # La nuit est rattachée au jour du réveil : celle qui suit la séance du
    # mardi porte la date du mercredi.
    table["Sommeil suivant (heures)"] = [sleep.get(day, float("nan")) for day in next_days]
    table["Efficacité sommeil suivant (%)"] = [efficiency.get(day, float("nan")) for day in next_days]
    bedtimes = [bedtime.get(day, float("nan")) for day in next_days]
    table["Coucher suivant"] = bedtimes
    gaps = []
    for day, hour, end in zip(next_days, bedtimes, table["Fin"]):
        if not np.isfinite(hour) or pd.isna(end):
            gaps.append(float("nan"))
            continue
        # L'heure de coucher est stockée autour de minuit (22 h 30 → −1,5) :
        # minuit du jour du réveil plus cette heure donne l'instant du coucher.
        asleep_at = day + pd.Timedelta(hours=float(hour))
        gap = (asleep_at - pd.Timestamp(end)).total_seconds() / 3600.0
        # Un coucher antérieur à la fin de séance, ou distant de plus d'un jour,
        # signale une nuit qui n'est pas celle qui suit la séance.
        gaps.append(gap if 0.0 <= gap <= 24.0 else float("nan"))
    table["Délai avant coucher (h)"] = gaps
    # Classer une séance « tardive » selon le coucher réel qui la suit rendrait
    # le test circulaire : un coucher avancé raccourcit le délai et range la
    # séance parmi les tardives. La marge se mesure donc au coucher HABITUEL
    # (médiane de la période), indépendant de la nuit que l'on veut juger.
    habitual = float(np.median(list(bedtime.values()))) if bedtime else float("nan")
    margins = []
    for day, end in zip(next_days, table["Fin"]):
        if not np.isfinite(habitual) or pd.isna(end):
            margins.append(float("nan"))
            continue
        margin = (day + pd.Timedelta(hours=habitual) - pd.Timestamp(end)).total_seconds() / 3600.0
        margins.append(margin if -12.0 <= margin <= 24.0 else float("nan"))
    table["Marge avant coucher habituel (h)"] = margins

    order = ["Date", "Début"]
    table = table.sort_values(order, kind="mergesort", ascending=not newest_first).reset_index(drop=True)
    return table[list(SESSION_COLUMNS)]


# ──────────────────────────────────────────────────────────────────────────────
# Volume, régularité, habitudes
# ──────────────────────────────────────────────────────────────────────────────


def boxing_summary(table: pd.DataFrame | None, *, window_days: int, today: Any = None) -> dict[str, Any]:
    """Volume et régularité sur la période : séances, fréquence, durée, intensité."""
    result: dict[str, Any] = {
        "ready": False,
        "sessions": 0,
        "per_week": float("nan"),
        "total_minutes": float("nan"),
        "mean_minutes": float("nan"),
        "mean_strain": float("nan"),
        "mean_calories": float("nan"),
        "total_calories": float("nan"),
        "mean_intensity": float("nan"),
        "mean_hard_minutes": float("nan"),
        "mean_hard_share": float("nan"),
        "mean_trimp": float("nan"),
        "last_date": None,
        "days_since_last": None,
        "median_rest_days": float("nan"),
        "partial_sessions": 0,
    }
    if table is None or table.empty:
        return result
    count = int(len(table))
    window = max(1, int(window_days))
    dates = pd.to_datetime(table["Date"]).dt.normalize()
    unique_days = dates.drop_duplicates().sort_values()

    def _mean(column: str) -> float:
        values = _clean_values(table[column]) if column in table.columns else np.array([])
        return float(values.mean()) if values.size else float("nan")

    def _sum(column: str) -> float:
        values = _clean_values(table[column]) if column in table.columns else np.array([])
        return float(values.sum()) if values.size else float("nan")

    last = pd.Timestamp(unique_days.iloc[-1])
    rests = unique_days.diff().dt.days.dropna() - 1
    partial = _clean_values(table["Part enregistrée (%)"]) if "Part enregistrée (%)" in table.columns else np.array([])
    result.update(
        {
            "ready": count >= MIN_SESSIONS_SUMMARY,
            "sessions": count,
            "per_week": count / (window / 7.0),
            "total_minutes": _sum("Durée (min)"),
            "mean_minutes": _mean("Durée (min)"),
            "mean_strain": _mean("Strain séance"),
            "mean_calories": _mean("Calories séance (kcal)"),
            "total_calories": _sum("Calories séance (kcal)"),
            "mean_intensity": _mean("Intensité (% FCR)"),
            "mean_hard_minutes": _mean("Zones 4–5 (min)"),
            "mean_hard_share": _mean("Part en zones 4–5 (%)"),
            "mean_trimp": _mean("TRIMP"),
            "last_date": last,
            "days_since_last": int((_today(today) - last).days),
            # Jours sans boxe entre deux journées de boxe : la médiane résiste
            # à une coupure de vacances, contrairement à la moyenne.
            "median_rest_days": float(rests.median()) if not rests.empty else float("nan"),
            "partial_sessions": int((partial < MIN_PERCENT_RECORDED).sum()),
        }
    )
    return result


def weekly_sessions(table: pd.DataFrame | None, *, start: Any, end: Any) -> pd.DataFrame:
    """Une ligne par semaine (du lundi), semaines sans séance comprises.

    Une semaine à zéro séance est ici une information : la laisser hors du
    tableau ferait croire à une pratique régulière.
    """
    columns = ["Semaine", "Séances", "Durée totale (min)", "TRIMP", "Calories (kcal)", "Strain moyen"]
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if last < first:
        return pd.DataFrame(columns=columns)
    weeks = pd.date_range(first - pd.Timedelta(days=first.dayofweek), last - pd.Timedelta(days=last.dayofweek), freq="7D")
    frame = pd.DataFrame({"Semaine": weeks})
    if table is None or table.empty:
        for column in columns[1:]:
            frame[column] = 0.0 if column != "Strain moyen" else np.nan
        frame["Séances"] = 0
        return frame[columns]
    data = table.copy()
    data["Semaine"] = pd.to_datetime(data["Date"]).dt.normalize() - pd.to_timedelta(pd.to_datetime(data["Date"]).dt.dayofweek, unit="D")
    grouped = data.groupby("Semaine").agg(
        Séances=("Date", "size"),
        **{
            "Durée totale (min)": ("Durée (min)", "sum"),
            "TRIMP": ("TRIMP", lambda values: float(values.sum(min_count=1)) if values.notna().any() else np.nan),
            "Calories (kcal)": ("Calories séance (kcal)", "sum"),
            "Strain moyen": ("Strain séance", "mean"),
        },
    )
    frame = frame.merge(grouped, left_on="Semaine", right_index=True, how="left")
    frame["Séances"] = frame["Séances"].fillna(0).astype(int)
    for column in ("Durée totale (min)", "Calories (kcal)"):
        frame[column] = frame[column].fillna(0.0)
    frame.loc[frame["Séances"] == 0, "TRIMP"] = 0.0
    return frame[columns]


def boxing_habits(table: pd.DataFrame | None) -> dict[str, Any]:
    """Quand vous boxez : jour de semaine × moment de la journée."""
    slots = [label for label, _, _ in TIME_SLOTS]
    empty = pd.DataFrame(0, index=slots, columns=list(WEEKDAY_SHORT))
    result: dict[str, Any] = {"ready": False, "matrix": empty, "favorite_day": None, "favorite_slot": None, "sessions": 0}
    if table is None or table.empty or "Début" not in table.columns:
        return result
    starts = pd.to_datetime(table["Début"], errors="coerce").dropna()
    if starts.empty:
        return result
    matrix = empty.copy()
    for start in starts:
        hour = start.hour + start.minute / 60.0
        slot = next(label for label, low, high in TIME_SLOTS if low <= hour < high)
        matrix.loc[slot, WEEKDAY_SHORT[start.dayofweek]] += 1
    by_day = matrix.sum(axis=0)
    by_slot = matrix.sum(axis=1)
    result.update(
        {
            "ready": len(starts) >= MIN_HABIT_SESSIONS,
            "matrix": matrix,
            "favorite_day": WEEKDAY_LONG[list(WEEKDAY_SHORT).index(by_day.idxmax())],
            "favorite_slot": str(by_slot.idxmax()),
            "sessions": int(len(starts)),
        }
    )
    return result


RECORD_SPECS: tuple[tuple[str, str, str, int], ...] = (
    ("Plus longue séance", "Durée (min)", "min", 0),
    ("Strain record", "Strain séance", "", 1),
    ("FC max record", "FC max (bpm)", "bpm", 0),
    ("Zones 4–5 record", "Zones 4–5 (min)", "min", 0),
    ("Calories record", "Calories séance (kcal)", "kcal", 0),
)


def boxing_records(table: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Meilleures marques de la période, chacune avec sa date."""
    if table is None or table.empty:
        return []
    records = []
    for label, column, unit, decimals in RECORD_SPECS:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if not values.notna().any():
            continue
        index = values.idxmax()
        records.append(
            {"label": label, "value": float(values.loc[index]), "unit": unit, "decimals": decimals, "date": table.loc[index, "Date"]}
        )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Charge
# ──────────────────────────────────────────────────────────────────────────────


def boxing_load(table: pd.DataFrame | None, *, reference: Any, data_start: Any) -> dict[str, Any]:
    """Charge boxe des 7 derniers jours rapportée à la semaine type des 21 précédents.

    La charge est le TRIMP quand les zones sont disponibles, la durée sinon :
    le strain, échelle logarithmique, ne s'additionne pas. La fenêtre chronique
    est découplée de la semaine aigüe (Windt & Gabbett, Br J Sports Med 2019),
    sans quoi le rapport est borné par construction. Avec deux ou trois
    séances par semaine, une séance de plus suffit à faire bondir le rapport :
    il décrit une rupture d'habitude, il ne prédit pas la blessure.
    """
    reference_day = pd.Timestamp(reference).normalize()
    acute_start = reference_day - pd.Timedelta(days=ACUTE_DAYS - 1)
    chronic_end = acute_start - pd.Timedelta(days=1)
    chronic_start = chronic_end - pd.Timedelta(days=CHRONIC_DAYS - 1)
    result: dict[str, Any] = {
        "status": "indisponible",
        "ratio": float("nan"),
        "acute": float("nan"),
        "chronic_weekly": float("nan"),
        "unit": "TRIMP",
        "acute_sessions": 0,
        "chronic_sessions": 0,
        "acute_start": acute_start,
        "acute_end": reference_day,
        "chronic_start": chronic_start,
        "chronic_end": chronic_end,
    }
    if table is None or table.empty:
        return result
    data = table.copy()
    data["Date"] = pd.to_datetime(data["Date"]).dt.normalize()
    use_trimp = "TRIMP" in data.columns and data["TRIMP"].notna().any()
    unit = "TRIMP" if use_trimp else "minutes"
    load = pd.to_numeric(data["TRIMP" if use_trimp else "Durée (min)"], errors="coerce").fillna(0.0)
    result["unit"] = unit
    acute_mask = (data["Date"] >= acute_start) & (data["Date"] <= reference_day)
    chronic_mask = (data["Date"] >= chronic_start) & (data["Date"] <= chronic_end)
    result["acute_sessions"] = int(acute_mask.sum())
    result["chronic_sessions"] = int(chronic_mask.sum())
    result["acute"] = float(load[acute_mask].sum())
    # Une absence de séance n'est un zéro que si le bracelet couvrait déjà ces
    # jours : avant le début de l'historique, le zéro serait inventé.
    if pd.Timestamp(data_start).normalize() > chronic_start:
        result["status"] = "historique trop court"
        return result
    chronic_weekly = float(load[chronic_mask].sum()) / (CHRONIC_DAYS / 7.0)
    result["chronic_weekly"] = chronic_weekly
    if result["chronic_sessions"] < MIN_CHRONIC_SESSIONS or chronic_weekly <= 0:
        result["status"] = "habitude trop mince"
        return result
    ratio = result["acute"] / chronic_weekly
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


# ──────────────────────────────────────────────────────────────────────────────
# Comparaisons testées
# ──────────────────────────────────────────────────────────────────────────────


def _group_tests(
    rows: Sequence[tuple[str, Any, Any]],
    labels: tuple[str, str],
    *,
    min_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Welch par mesure, seuil divisé par le nombre de mesures testées.

    Comparer cinq mesures entre deux groupes, c'est tirer cinq fois : sans
    correction, l'une d'elles « diffère » par hasard une fois sur quatre.
    """
    columns = ["Mesure", labels[0], labels[1], "Écart", "IC 95 % bas", "IC 95 % haut", "Effectifs", "Écart établi"]
    records = []
    for name, first, second in rows:
        a, b = _clean_values(first), _clean_values(second)
        if a.size < min_size or b.size < min_size:
            continue
        test = welch_comparison(a, b)
        records.append(
            {
                "Mesure": name,
                labels[0]: float(a.mean()),
                labels[1]: float(b.mean()),
                "Écart": float(test["gap"]),
                "IC 95 % bas": float(test["low"]),
                "IC 95 % haut": float(test["high"]),
                "Effectifs": f"{a.size} / {b.size}",
                "_p": float(test["p_value"]),
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(records)
    threshold = ALPHA / max(1, len(table))
    table["Écart établi"] = table["_p"].notna() & (table["_p"] <= threshold)
    return table[columns]


def _row(table: pd.DataFrame, name: str) -> pd.Series | None:
    if table is None or table.empty:
        return None
    match = table[table["Mesure"] == name]
    return None if match.empty else match.iloc[0]


def next_day_cost(daily: pd.DataFrame | None, table: pd.DataFrame | None, *, min_size: int = MIN_GROUP_SIZE) -> dict[str, Any]:
    """Le lendemain d'une journée de boxe, comparé au lendemain des autres journées.

    Le score du matin même précède la séance ; seul le lendemain en porte la
    trace. Deux séances le même jour ne comptent qu'un lendemain.
    """
    labels = ("Après boxe", "Autres jours")
    result: dict[str, Any] = {
        "ready": False,
        "boxing_days": 0,
        "other_days": 0,
        "table": pd.DataFrame(columns=["Mesure", *labels, "Écart", "IC 95 % bas", "IC 95 % haut", "Effectifs", "Écart établi"]),
        "morning_boxing": float("nan"),
        "morning_other": float("nan"),
        "required": min_size,
    }
    grid = daily_grid(daily)
    if grid.empty or "Récupération (%)" not in grid.columns or table is None or table.empty:
        return result
    boxing_days = set(pd.to_datetime(table["Date"]).dt.normalize())
    recovery = _lookup(grid, "Récupération (%)")
    hrv = _lookup(grid, "HRV (ms)")
    resting = _lookup(grid, "FC repos (bpm)")

    groups: dict[bool, dict[str, list[float]]] = {
        True: {"rec": [], "hrv": [], "rhr": [], "morning": []},
        False: {"rec": [], "hrv": [], "rhr": [], "morning": []},
    }
    for date in grid["Date"]:
        day = pd.Timestamp(date)
        following = day + pd.Timedelta(days=1)
        if following not in recovery:
            continue
        bucket = groups[day in boxing_days]
        bucket["rec"].append(recovery[following])
        bucket["hrv"].append(hrv.get(following, float("nan")))
        bucket["rhr"].append(resting.get(following, float("nan")))
        bucket["morning"].append(recovery.get(day, float("nan")))

    result["boxing_days"] = len(groups[True]["rec"])
    result["other_days"] = len(groups[False]["rec"])
    morning_boxing = _clean_values(groups[True]["morning"])
    morning_other = _clean_values(groups[False]["morning"])
    result["morning_boxing"] = float(morning_boxing.mean()) if morning_boxing.size else float("nan")
    result["morning_other"] = float(morning_other.mean()) if morning_other.size else float("nan")
    tests = _group_tests(
        [
            ("Récupération du lendemain (%)", groups[True]["rec"], groups[False]["rec"]),
            ("HRV du lendemain (ms)", groups[True]["hrv"], groups[False]["hrv"]),
            ("FC repos du lendemain (bpm)", groups[True]["rhr"], groups[False]["rhr"]),
        ],
        labels,
        min_size=min_size,
    )
    result["table"] = tests if not tests.empty else result["table"]
    result["ready"] = _row(tests, "Récupération du lendemain (%)") is not None
    return result


def recovery_profile(daily: pd.DataFrame | None, table: pd.DataFrame | None, *, horizon: int = 3) -> dict[str, Any]:
    """Récupération moyenne le matin de la séance, puis J+1 à J+*horizon*.

    Une moyenne du lendemain dit combien la séance coûte ; le profil dit en
    combien de jours vous revenez à votre niveau habituel.
    """
    columns = ["Jour", "Décalage", "Récupération moyenne (%)", "IC 95 % bas", "IC 95 % haut", "Observations"]
    result: dict[str, Any] = {"ready": False, "table": pd.DataFrame(columns=columns), "overall": float("nan")}
    grid = daily_grid(daily)
    if grid.empty or "Récupération (%)" not in grid.columns or table is None or table.empty:
        return result
    recovery = _lookup(grid, "Récupération (%)")
    if not recovery:
        return result
    result["overall"] = float(np.mean(list(recovery.values())))
    boxing_days = sorted(set(pd.to_datetime(table["Date"]).dt.normalize()))
    names = {0: "Matin de la séance", 1: "Lendemain"}
    rows = []
    from scipy import stats

    for offset in range(0, max(1, int(horizon)) + 1):
        values = _clean_values([recovery.get(day + pd.Timedelta(days=offset), float("nan")) for day in boxing_days])
        if values.size == 0:
            continue
        mean = float(values.mean())
        if values.size >= 2:
            margin = float(stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size))
        else:
            margin = float("nan")
        rows.append(
            {
                "Jour": names.get(offset, f"J+{offset}"),
                "Décalage": offset,
                "Récupération moyenne (%)": mean,
                "IC 95 % bas": mean - margin,
                "IC 95 % haut": mean + margin,
                "Observations": int(values.size),
            }
        )
    if not rows:
        return result
    profile = pd.DataFrame(rows)[columns]
    result["table"] = profile
    result["ready"] = bool((profile["Observations"] >= MIN_GROUP_SIZE).any())
    return result


def readiness_effect(table: pd.DataFrame | None, *, min_size: int = MIN_GROUP_SIZE) -> dict[str, Any]:
    """Boxez-vous plus fort les matins verts ?

    Le score du matin est calculé avant la séance : la comparaison va de la
    récupération vers l'intensité, dans le sens du temps.
    """
    labels = ("Matins verts", "Autres matins")
    by_zone_columns = ["Zone du matin", "Séances", "Strain moyen", "Intensité (% FCR)", "Zones 4–5 (min)", "Récupération du lendemain (%)"]
    result: dict[str, Any] = {
        "ready": False,
        "by_zone": pd.DataFrame(columns=by_zone_columns),
        "table": pd.DataFrame(columns=["Mesure", *labels, "Écart", "IC 95 % bas", "IC 95 % haut", "Effectifs", "Écart établi"]),
        "green": 0,
        "other": 0,
        "required": min_size,
    }
    if table is None or table.empty or "Récupération du matin (%)" not in table.columns:
        return result
    scored = table[pd.to_numeric(table["Récupération du matin (%)"], errors="coerce").notna()].copy()
    if scored.empty:
        return result
    rows = []
    for zone in ("Vert", "Jaune", "Rouge"):
        chunk = scored[scored["Zone du matin"] == zone]
        if chunk.empty:
            continue
        rows.append(
            {
                "Zone du matin": zone,
                "Séances": int(len(chunk)),
                "Strain moyen": float(pd.to_numeric(chunk["Strain séance"], errors="coerce").mean()),
                "Intensité (% FCR)": float(pd.to_numeric(chunk["Intensité (% FCR)"], errors="coerce").mean()),
                "Zones 4–5 (min)": float(pd.to_numeric(chunk["Zones 4–5 (min)"], errors="coerce").mean()),
                "Récupération du lendemain (%)": float(pd.to_numeric(chunk["Récupération du lendemain (%)"], errors="coerce").mean()),
            }
        )
    result["by_zone"] = pd.DataFrame(rows, columns=by_zone_columns)
    green = scored[scored["Zone du matin"] == "Vert"]
    other = scored[scored["Zone du matin"] != "Vert"]
    result["green"], result["other"] = int(len(green)), int(len(other))
    tests = _group_tests(
        [
            ("Strain séance", green["Strain séance"], other["Strain séance"]),
            ("Intensité (% FCR)", green["Intensité (% FCR)"], other["Intensité (% FCR)"]),
            ("Zones 4–5 (min)", green["Zones 4–5 (min)"], other["Zones 4–5 (min)"]),
        ],
        labels,
        min_size=min_size,
    )
    if not tests.empty:
        result["table"] = tests
        result["ready"] = True
    return result


def late_session_sleep(
    table: pd.DataFrame | None,
    *,
    cutoff_hours: float = LATE_SESSION_HOURS,
    min_size: int = MIN_GROUP_SIZE,
) -> dict[str, Any]:
    """Les nuits qui suivent une séance tardive, face à celles qui suivent une séance plus tôt.

    Sur 14 689 porteurs de WHOOP, Leota et al. (Nature Communications 2025)
    associent un effort qui s'achève moins de quatre heures avant le coucher à
    un endormissement plus tardif, une nuit plus courte, une FC nocturne plus
    haute et une HRV plus basse. Ce test vérifie si c'est vrai chez vous.

    La séance est jugée tardive par rapport à votre heure de coucher
    habituelle, et non au coucher de la nuit suivante : sans quoi un coucher
    avancé suffirait à la rendre « tardive » et le test se mordrait la queue.
    """
    labels = ("Séances tardives", "Séances plus tôt")
    result: dict[str, Any] = {
        "ready": False,
        "late": 0,
        "early": 0,
        "median_gap": float("nan"),
        "table": pd.DataFrame(columns=["Mesure", *labels, "Écart", "IC 95 % bas", "IC 95 % haut", "Effectifs", "Écart établi"]),
        "cutoff": float(cutoff_hours),
        "required": min_size,
    }
    if table is None or table.empty or "Marge avant coucher habituel (h)" not in table.columns:
        return result
    gaps = pd.to_numeric(table["Marge avant coucher habituel (h)"], errors="coerce")
    known = table[gaps.notna()]
    if known.empty:
        return result
    late = known[gaps[gaps.notna()] < cutoff_hours]
    early = known[gaps[gaps.notna()] >= cutoff_hours]
    result["late"], result["early"] = int(len(late)), int(len(early))
    result["median_gap"] = float(gaps.dropna().median())
    tests = _group_tests(
        [
            ("Coucher suivant", late["Coucher suivant"], early["Coucher suivant"]),
            ("Sommeil suivant (heures)", late["Sommeil suivant (heures)"], early["Sommeil suivant (heures)"]),
            ("Efficacité sommeil suivant (%)", late["Efficacité sommeil suivant (%)"], early["Efficacité sommeil suivant (%)"]),
            ("FC repos du lendemain (bpm)", late["FC repos du lendemain (bpm)"], early["FC repos du lendemain (bpm)"]),
            ("HRV du lendemain (ms)", late["HRV du lendemain (ms)"], early["HRV du lendemain (ms)"]),
        ],
        labels,
        min_size=min_size,
    )
    if not tests.empty:
        result["table"] = tests
        result["ready"] = True
    return result


def next_morning_weight(weights: pd.DataFrame | None, table: pd.DataFrame | None, *, min_size: int = MIN_GROUP_SIZE) -> dict[str, Any]:
    """Variation de poids d'un matin au suivant, selon qu'une séance a eu lieu entre les deux.

    Une séance fait transpirer : la pesée du lendemain peut baisser sans
    qu'aucune graisse n'ait été perdue, puis remonter dès la réhydratation.
    Seules les paires de pesées espacées d'exactement un jour sont comparées.
    """
    result: dict[str, Any] = {
        "ready": False,
        "boxing_pairs": 0,
        "other_pairs": 0,
        "after_boxing": float("nan"),
        "other": float("nan"),
        "gap": float("nan"),
        "low": float("nan"),
        "high": float("nan"),
        "p_value": float("nan"),
        "significant": False,
        "required": min_size,
    }
    if weights is None or weights.empty or "Poids (Kgs)" not in weights.columns or table is None or table.empty:
        return result
    data = weights[["Date", "Poids (Kgs)"]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    data["Poids (Kgs)"] = pd.to_numeric(data["Poids (Kgs)"], errors="coerce")
    data = data.dropna().groupby("Date", as_index=False)["Poids (Kgs)"].mean().sort_values("Date")
    if len(data) < 2:
        return result
    boxing_days = set(pd.to_datetime(table["Date"]).dt.normalize())
    change = data["Poids (Kgs)"].diff()
    gap_days = data["Date"].diff().dt.days
    after, other = [], []
    for previous_day, delta, gap in zip(data["Date"].shift(1), change, gap_days):
        if gap != 1 or pd.isna(delta):
            continue
        (after if pd.Timestamp(previous_day) in boxing_days else other).append(float(delta))
    result["boxing_pairs"], result["other_pairs"] = len(after), len(other)
    if len(after) < min_size or len(other) < min_size:
        return result
    test = welch_comparison(after, other)
    result.update(
        {
            "ready": True,
            "after_boxing": float(np.mean(after)),
            "other": float(np.mean(other)),
            "gap": float(test["gap"]),
            "low": float(test["low"]),
            "high": float(test["high"]),
            "p_value": float(test["p_value"]),
            "significant": bool(np.isfinite(test["p_value"]) and test["p_value"] <= ALPHA),
        }
    )
    return result


PROGRESSION_METRICS: tuple[tuple[str, str, int], ...] = (
    ("Intensité (% FCR)", "points de % FCR", 1),
    ("Strain séance", "point de strain", 2),
    ("Durée (min)", "min", 1),
    ("Zones 4–5 (min)", "min", 1),
    ("kcal/min", "kcal/min", 2),
)


def boxing_progression(
    table: pd.DataFrame | None,
    *,
    min_sessions: int = MIN_SESSIONS_PROGRESSION,
    min_span_days: int = MIN_SPAN_PROGRESSION_DAYS,
) -> dict[str, Any]:
    """Pente de chaque mesure de séance par tranche de 30 jours, avec son intervalle.

    Une pente n'est dite « en hausse » ou « en baisse » que si elle résiste à
    un test corrigé pour le nombre de mesures examinées ; sinon elle est
    « stable » — ce qui veut dire que le hasard suffit à la produire, pas
    qu'elle est nulle.
    """
    columns = ["Mesure", "Séances", "Pente / 30 jours", "IC 95 % bas", "IC 95 % haut", "Lecture"]
    result: dict[str, Any] = {"ready": False, "table": pd.DataFrame(columns=columns), "span_days": 0, "required": min_sessions}
    if table is None or table.empty:
        return result
    dates = pd.to_datetime(table["Date"]).dt.normalize()
    span = int((dates.max() - dates.min()).days)
    result["span_days"] = span
    if len(table) < min_sessions or span < min_span_days:
        return result
    from scipy import stats

    elapsed = (dates - dates.min()).dt.days.to_numpy(dtype=float)
    rows = []
    for metric, unit, _ in PROGRESSION_METRICS:
        if metric not in table.columns:
            continue
        values = pd.to_numeric(table[metric], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < min_sessions or np.ptp(elapsed[mask]) < min_span_days or np.nanstd(values[mask]) == 0:
            continue
        fit = stats.linregress(elapsed[mask], values[mask])
        degrees = int(mask.sum()) - 2
        margin = float(stats.t.ppf(0.975, degrees) * fit.stderr) if degrees > 0 else float("nan")
        rows.append(
            {
                "Mesure": metric,
                "Séances": int(mask.sum()),
                "Pente / 30 jours": float(fit.slope * 30.0),
                "IC 95 % bas": float((fit.slope - margin) * 30.0),
                "IC 95 % haut": float((fit.slope + margin) * 30.0),
                "_p": float(fit.pvalue),
                "_unit": unit,
            }
        )
    if not rows:
        return result
    frame = pd.DataFrame(rows)
    threshold = ALPHA / max(1, len(frame))
    frame["Lecture"] = [
        ("en hausse" if slope > 0 else "en baisse") if np.isfinite(p) and p <= threshold else "stable (non établi)"
        for slope, p in zip(frame["Pente / 30 jours"], frame["_p"])
    ]
    result["units"] = dict(zip(frame["Mesure"], frame["_unit"]))
    result["table"] = frame[columns]
    result["ready"] = True
    return result


def boxing_energy(
    daily: pd.DataFrame | None,
    all_workouts: pd.DataFrame | None,
    table: pd.DataFrame | None,
    *,
    window_days: int,
    required_daily_kg: float | None = None,
    kcal_per_kg: float = KCAL_PER_KG,
) -> dict[str, Any]:
    """Ce que la boxe pèse dans la dépense, et dans le déficit que vise l'objectif.

    Les calories d'une séance WHOOP sont brutes : elles contiennent ce que le
    corps aurait dépensé de toute façon pendant ce temps. L'excédent net est
    estimé en retranchant, minute pour minute, le rythme de dépense moyen des
    jours sans aucune séance.
    """
    result: dict[str, Any] = {
        "ready": False,
        "gross_weekly": float("nan"),
        "net_weekly": float("nan"),
        "baseline_per_minute": float("nan"),
        "share_of_burn": float("nan"),
        "share_of_target": float("nan"),
        "required_weekly_deficit": float("nan"),
        "kg_per_month": float("nan"),
        "rest_days": 0,
    }
    if table is None or table.empty:
        return result
    window = max(1, int(window_days))
    calories = pd.to_numeric(table["Calories séance (kcal)"], errors="coerce")
    duration = pd.to_numeric(table["Durée (min)"], errors="coerce")
    if not calories.notna().any():
        return result
    weeks = window / 7.0
    result["gross_weekly"] = float(calories.sum()) / weeks

    grid = daily_grid(daily)
    if not grid.empty and "Calories (kcal)" in grid.columns:
        workout_days: set = set()
        if all_workouts is not None and not all_workouts.empty and "Date" in all_workouts.columns:
            workout_days = set(pd.to_datetime(all_workouts["Date"], errors="coerce").dt.normalize().dropna())
        burn = pd.to_numeric(grid["Calories (kcal)"], errors="coerce")
        rest = burn[[pd.Timestamp(date) not in workout_days for date in grid["Date"]]].dropna()
        result["rest_days"] = int(len(rest))
        if len(rest) >= MIN_REST_DAYS_BASELINE:
            per_minute = float(rest.median()) / 1440.0
            result["baseline_per_minute"] = per_minute
            net = (calories - duration * per_minute).clip(lower=0.0)
            result["net_weekly"] = float(net.sum()) / weeks
        measured = grid.loc[burn.notna(), "Date"].map(pd.Timestamp)
        measured_days = set(measured)
        on_measured = calories[[pd.Timestamp(date) in measured_days for date in table["Date"]]]
        total_burn = float(burn.sum())
        if total_burn > 0 and on_measured.notna().any():
            result["share_of_burn"] = float(on_measured.sum()) / total_burn * 100.0

    reference_weekly = result["net_weekly"] if np.isfinite(result["net_weekly"]) else result["gross_weekly"]
    result["kg_per_month"] = reference_weekly * (30.0 / 7.0) / float(kcal_per_kg)
    if required_daily_kg is not None and np.isfinite(required_daily_kg) and required_daily_kg > 0:
        required_weekly = float(required_daily_kg) * float(kcal_per_kg) * 7.0
        result["required_weekly_deficit"] = required_weekly
        result["share_of_target"] = reference_weekly / required_weekly * 100.0
    result["ready"] = True
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Repère du jour et constats
# ──────────────────────────────────────────────────────────────────────────────


def todays_guidance(
    daily: pd.DataFrame | None,
    table: pd.DataFrame | None,
    load: Mapping[str, Any] | None = None,
    *,
    today: Any = None,
) -> dict[str, Any]:
    """Quelle séance la récupération du jour autorise, d'après les zones WHOOP.

    WHOOP associe la zone verte à un organisme prêt pour un effort soutenu, la
    jaune à un effort modéré, la rouge à un besoin de repos. Ce repère traduit
    ces zones en séance de boxe ; ce n'est pas une prescription.
    """
    day = _today(today)
    grid = daily_grid(daily)
    recovery = _lookup(grid, "Récupération (%)").get(day, float("nan"))
    zone = zone_of(recovery)
    dates = sorted(set(pd.to_datetime(table["Date"]).dt.normalize())) if table is not None and not table.empty else []
    boxed_today = bool(dates) and dates[-1] == day
    previous = [date for date in dates if date < day]
    days_since = int((day - previous[-1]).days) if previous else None
    status = str((load or {}).get("status", ""))
    ratio = float((load or {}).get("ratio", float("nan")))

    if boxed_today:
        title = "Séance du jour enregistrée"
        body = "Le score de demain matin dira ce qu'elle vous a coûté : c'est le premier moment où elle devient mesurable."
        tone = "success"
    elif zone is None:
        title = "Score du jour pas encore disponible"
        body = "Synchronisez après votre réveil : le repère s'appuie sur la récupération calculée ce matin."
        tone = "info"
    elif zone == "Rouge":
        title = "Journée rouge : la récupération d'abord"
        body = (
            f"Récupération à {_fr(recovery)} %. WHOOP associe la zone rouge à un organisme qui a surtout besoin "
            "de repos. Si vous boxez, une séance technique légère — shadow, travail de pieds, mobilité — "
            "charge bien moins qu'un sparring ou des rounds au sac."
        )
        tone = "warning"
    elif zone == "Jaune":
        title = "Journée jaune : une séance modérée"
        body = (
            f"Récupération à {_fr(recovery)} %. Technique, pattes d'ours ou sac à intensité contrôlée ; les "
            "rounds les plus durs trouvent mieux leur place un jour vert."
        )
        tone = "info"
    else:
        title = "Feu vert pour une séance intense"
        body = f"Récupération à {_fr(recovery)} %. Sparring, rounds au sac ou travail de puissance : c'est le jour."
        tone = "success"
        if status == "montée en charge brutale" and np.isfinite(ratio):
            title = "Feu vert, mais volume à surveiller"
            body += (
                f" Votre charge boxe des 7 derniers jours atteint toutefois {_fr(ratio, 1)} fois votre semaine "
                "habituelle : l'intensité, oui ; un volume maîtrisé, aussi."
            )
            tone = "info"
    if not boxed_today and days_since is not None and days_since >= INACTIVITY_DAYS:
        body += f" Dernière séance il y a {days_since} jours : une reprise progressive ménage les épaules et les poignets."
    return {
        "title": title,
        "body": body,
        "tone": tone,
        "zone": zone,
        "recovery": recovery,
        "days_since": days_since,
        "boxed_today": boxed_today,
    }


def _measure_gap(tests: pd.DataFrame, name: str) -> tuple[float, bool]:
    row = _row(tests, name)
    if row is None:
        return float("nan"), False
    return float(row["Écart"]), bool(row["Écart établi"])


def boxing_insights(
    *,
    summary: Mapping[str, Any],
    load: Mapping[str, Any],
    cost: Mapping[str, Any],
    readiness: Mapping[str, Any],
    late: Mapping[str, Any],
    weight: Mapping[str, Any],
    progression: Mapping[str, Any],
    energy: Mapping[str, Any],
    limit: int = 6,
) -> list[Insight]:
    """Constats rédigés, classés par importance ; chacun reste muet sous son effectif."""
    found: list[Insight] = []
    sessions = int(summary.get("sessions", 0) or 0)
    if sessions == 0:
        return found

    days_since = summary.get("days_since_last")
    if days_since is not None and days_since >= INACTIVITY_DAYS:
        found.append(
            Insight(
                f"Pas de boxe depuis {days_since} jours",
                "Après une coupure, une reprise progressive — durée et intensité en hausse sur deux ou trois "
                "séances — ménage les articulations et laisse le cœur retrouver ses repères.",
                tone="warning",
                icon="⏸️",
                priority=82,
            )
        )

    status = str(load.get("status", ""))
    if status == "montée en charge brutale":
        found.append(
            Insight(
                "Charge boxe en hausse brutale",
                f"Les 7 derniers jours représentent {_fr(load.get('ratio'), 1)} fois votre semaine type des 21 jours "
                f"précédents ({load.get('acute_sessions', 0)} {plural(load.get('acute_sessions', 0), 'séance')} contre "
                f"{_fr(float(load.get('chronic_sessions', 0)) / 3.0, 1)} par semaine). Avec peu de séances par semaine, "
                "une seule en plus suffit à franchir ce seuil : c'est une rupture d'habitude, pas un verdict.",
                tone="warning",
                icon="📈",
                priority=78,
            )
        )

    gap, established = _measure_gap(cost.get("table", pd.DataFrame()), "Récupération du lendemain (%)")
    if established and np.isfinite(gap):
        if gap < 0:
            found.append(
                Insight(
                    f"La boxe vous coûte {_fr(abs(gap))} points le lendemain",
                    f"Après une journée de boxe, votre récupération du lendemain est en moyenne {_fr(abs(gap))} points "
                    f"sous celle qui suit vos autres journées ({cost.get('boxing_days')} lendemains de boxe comparés). "
                    "L'écart résiste au test : prévoir une journée plus calme après une grosse séance est une piste.",
                    tone="warning",
                    icon="🥊",
                    priority=76,
                )
            )
        else:
            found.append(
                Insight(
                    "Vos lendemains de boxe sont meilleurs que les autres",
                    f"+{_fr(gap)} points de récupération le lendemain d'une séance. Cela arrive quand on boxe surtout "
                    "les jours où l'on est déjà en forme : la séance suit la récupération plus qu'elle ne la crée.",
                    tone="success",
                    icon="🥊",
                    priority=60,
                )
            )

    late_tests = late.get("table", pd.DataFrame())
    sleep_gap, sleep_established = _measure_gap(late_tests, "Sommeil suivant (heures)")
    if sleep_established and np.isfinite(sleep_gap) and sleep_gap < 0:
        found.append(
            Insight(
                f"Vos séances tardives raccourcissent votre nuit de {_fr(abs(sleep_gap) * 60)} min",
                f"Quand la séance se termine moins de {_fr(late.get('cutoff', LATE_SESSION_HOURS))} heures avant le "
                "coucher, la nuit qui suit est plus courte, et l'écart résiste au test. Une étude sur 14 689 porteurs "
                "de WHOOP fait le même constat (Leota et al., Nature Communications 2025).",
                tone="warning",
                icon="🌙",
                priority=74,
            )
        )
    elif late.get("ready") and not late_tests["Écart établi"].any():
        found.append(
            Insight(
                "Vos séances tardives n'abîment pas visiblement vos nuits",
                f"{late.get('late')} séances terminées moins de {_fr(late.get('cutoff', LATE_SESSION_HOURS))} h avant "
                "le coucher, comparées aux autres : aucun écart de sommeil, de FC de repos ou de HRV ne se distingue "
                "du hasard sur vos données.",
                tone="success",
                icon="🌙",
                priority=48,
            )
        )

    strain_gap, strain_established = _measure_gap(readiness.get("table", pd.DataFrame()), "Strain séance")
    if strain_established and np.isfinite(strain_gap):
        found.append(
            Insight(
                "Vous boxez plus fort les matins verts" if strain_gap > 0 else "Vous boxez plus fort les matins non verts",
                f"Écart de {_fr(strain_gap, 1, sign=True)} de strain par séance entre vos matins verts et les autres, "
                "un écart qui résiste au test. Le score du matin précède la séance : c'est lui qui annonce "
                "l'intensité, pas l'inverse.",
                tone="info",
                icon="🚦",
                priority=66,
            )
        )

    progression_table = progression.get("table", pd.DataFrame())
    if progression.get("ready") and not progression_table.empty:
        moving = progression_table[progression_table["Lecture"] != "stable (non établi)"]
        if not moving.empty:
            first = moving.iloc[0]
            unit = progression.get("units", {}).get(first["Mesure"], "")
            found.append(
                Insight(
                    f"{first['Mesure']} {first['Lecture']}",
                    f"{_fr(first['Pente / 30 jours'], 1, sign=True)} {unit} par tranche de 30 jours sur "
                    f"{int(first['Séances'])} séances, une pente qui résiste au test corrigé pour le nombre de "
                    "mesures examinées.",
                    tone="info",
                    icon="📊",
                    priority=58,
                )
            )

    if weight.get("ready") and weight.get("significant") and float(weight.get("gap", 0.0)) < 0:
        found.append(
            Insight(
                f"Le lendemain d'une séance, la balance affiche {_fr(abs(weight['gap']), 2)} kg de moins",
                "Comparé aux autres paires de pesées espacées d'un jour. Une séance fait perdre de l'eau, que la "
                "réhydratation rend : ne lisez pas ce creux comme de la graisse perdue, ni sa remontée comme une reprise.",
                tone="info",
                icon="⚖️",
                priority=62,
            )
        )

    share = float(energy.get("share_of_target", float("nan")))
    weekly = float(energy.get("net_weekly", float("nan")))
    if not np.isfinite(weekly):
        weekly = float(energy.get("gross_weekly", float("nan")))
    if energy.get("ready") and np.isfinite(share) and np.isfinite(weekly):
        found.append(
            Insight(
                f"La boxe couvre environ {_fr(share)} % du déficit visé",
                f"Environ {_fr(weekly)} kcal par semaine attribuables à la boxe, face à un déficit hebdomadaire "
                f"d'environ {_fr(energy.get('required_weekly_deficit'))} kcal que suppose la trajectoire cible. "
                "Le reste se joue dans l'assiette et dans l'activité quotidienne.",
                tone="info",
                icon="🔥",
                priority=56,
            )
        )

    per_week = float(summary.get("per_week", float("nan")))
    found.append(
        Insight(
            f"{sessions} {plural(sessions, 'séance')} de boxe sur la période",
            f"Soit {_fr(per_week, 1)} par semaine, {_fr(summary.get('mean_minutes'))} min en moyenne"
            + (
                f", dont {_fr(summary.get('mean_hard_minutes'))} min au-dessus de 80 % de votre réserve cardiaque"
                if np.isfinite(float(summary.get("mean_hard_minutes", float("nan"))))
                else ""
            )
            + ".",
            tone="info",
            icon="🗓️",
            priority=50,
        )
    )
    found.sort(key=lambda insight: insight.priority, reverse=True)
    return found[: max(1, int(limit))]
