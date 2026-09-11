"""Libellés de dates en français, lisibles par un humain.

Les noms de mois et de jours sont codés en dur volontairement : dépendre de la
locale système rendrait l'affichage tributaire des paquets installés sur la
machine de déploiement, et une locale `fr_FR` absente ferait silencieusement
réapparaître « Sep » au lieu de « sept. ».
"""

from __future__ import annotations

from typing import Any

import pandas as pd

MISSING_LABEL = "—"

MONTHS_SHORT: tuple[str, ...] = (
    "janv.", "févr.", "mars", "avr.", "mai", "juin",
    "juil.", "août", "sept.", "oct.", "nov.", "déc.",
)

MONTHS_LONG: tuple[str, ...] = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)

WEEKDAYS_LONG: tuple[str, ...] = (
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
)

WEEKDAYS_SHORT: tuple[str, ...] = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize(None) if stamp.tz is not None else stamp


def format_day_month(value: Any) -> str:
    """``3 sept.`` — format court des axes et des étiquettes denses."""
    stamp = _timestamp(value)
    if stamp is None:
        return MISSING_LABEL
    return f"{stamp.day} {MONTHS_SHORT[stamp.month - 1]}"


def format_long_date(value: Any, *, with_weekday: bool = True) -> str:
    """``mercredi 3 septembre 2026`` — pour un titre ou une phrase."""
    stamp = _timestamp(value)
    if stamp is None:
        return MISSING_LABEL
    day_part = f"{WEEKDAYS_LONG[stamp.dayofweek]} " if with_weekday else ""
    return f"{day_part}{stamp.day} {MONTHS_LONG[stamp.month - 1]} {stamp.year}"


def format_datetime(value: Any) -> str:
    """``3 sept. à 18:42`` — une heure de synchronisation sans heure ne sert à rien."""
    stamp = _timestamp(value)
    if stamp is None:
        return MISSING_LABEL
    return f"{format_day_month(stamp)} à {stamp.strftime('%H:%M')}"


def format_relative_day(value: Any, *, today: Any = None) -> str:
    """``aujourd'hui``, ``hier``, ``il y a 3 jours``, puis la date au-delà d'une semaine.

    Au-delà de sept jours, un décompte cesse d'être parlant : la date absolue
    situe mieux le lecteur.
    """
    stamp = _timestamp(value)
    if stamp is None:
        return MISSING_LABEL
    reference = _timestamp(today) or pd.Timestamp.now()
    delta = (reference.normalize() - stamp.normalize()).days
    if delta == 0:
        return "aujourd'hui"
    if delta == 1:
        return "hier"
    if delta == 2:
        return "avant-hier"
    if 2 < delta <= 7:
        return f"il y a {delta} jours"
    if delta < 0:
        return format_day_month(stamp)
    return format_day_month(stamp)


def describe_freshness(value: Any, *, today: Any = None) -> dict[str, Any]:
    """Ancienneté de la dernière mesure, avec son niveau d'alerte."""
    stamp = _timestamp(value)
    if stamp is None:
        return {"days": None, "label": MISSING_LABEL, "tone": "info", "stale": False}
    reference = _timestamp(today) or pd.Timestamp.now()
    days = int((reference.normalize() - stamp.normalize()).days)
    # Deux jours sans donnée peuvent tenir à l'heure de synchronisation ; au-delà,
    # le bracelet n'a probablement pas été porté ou synchronisé.
    stale = days > 2
    # Ici le décompte prime sur la date : mesurer l'ancienneté est justement le
    # rôle de cette fonction, alors que l'affichage courant bascule sur la date
    # absolue au-delà d'une semaine.
    label = f"il y a {days} jours" if days > 7 else format_relative_day(stamp, today=reference)
    return {
        "days": days,
        "label": label,
        "tone": "warning" if stale else "success",
        "stale": stale,
    }


def format_date_range(start: Any, end: Any) -> str:
    """``3 sept. → 11 sept. 2026``, l'année n'apparaissant qu'une fois."""
    first, last = _timestamp(start), _timestamp(end)
    if first is None or last is None:
        return MISSING_LABEL
    if first.year == last.year:
        return f"{format_day_month(first)} → {format_day_month(last)} {last.year}"
    return f"{format_day_month(first)} {first.year} → {format_day_month(last)} {last.year}"


def format_week_label(value: Any) -> str:
    """``Semaine du 3 août`` — plus parlant qu'une date isolée en tête de ligne."""
    stamp = _timestamp(value)
    if stamp is None:
        return MISSING_LABEL
    return f"Semaine du {format_day_month(stamp)}"


def format_duration_minutes(minutes: Any) -> str:
    """``1 h 30`` plutôt que ``90,0`` : une durée se lit en heures et minutes."""
    try:
        numeric = float(minutes)
    except (TypeError, ValueError):
        return MISSING_LABEL
    if not pd.notna(numeric) or numeric < 0:
        return MISSING_LABEL
    total = int(round(numeric))
    hours, remainder = divmod(total, 60)
    if hours == 0:
        return f"{remainder} min"
    return f"{hours} h {remainder:02d}" if remainder else f"{hours} h"


def format_clock_hour(value: Any) -> str:
    """Heure décimale recentrée autour de minuit, rendue en horloge (``23:30``)."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return MISSING_LABEL
    if not pd.notna(numeric):
        return MISSING_LABEL
    # Les couchers avant minuit sont stockés en négatif pour rester continus.
    hour = numeric + 24.0 if numeric < 0 else numeric
    hours = int(hour) % 24
    minutes = int(round((hour - int(hour)) * 60))
    if minutes == 60:
        hours, minutes = (hours + 1) % 24, 0
    return f"{hours:02d}:{minutes:02d}"


def axis_tick_labels(dates: Any) -> list[str]:
    """Étiquettes d'axe en français, pour remplacer les mois anglais de Plotly."""
    return [format_day_month(date) for date in pd.to_datetime(pd.Series(list(dates)), errors="coerce")]
