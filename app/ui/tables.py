"""Mise en forme des tableaux WHOOP, partagée par les onglets Whoop et Boxe.

Un tableau brut affiche « 43.4038 », « True » et « 2026-09-08 00:00:00 » :
ce module rend des décimales maîtrisées, des dates en français avec leur jour
de semaine, des durées en heures et minutes et des vides explicites.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from app.core.date_labels import (
    format_clock_hour,
    format_datetime,
    format_duration_minutes,
    format_short_date,
    format_week_label,
)
from app.core.formatting import MISSING_VALUE as MISSING_TEXT, format_fr_number

DURATION_COLUMNS = frozenset({"Durée (min)", "Durée totale (min)"})
CLOCK_COLUMNS = frozenset({"Heure de coucher", "Coucher suivant"})


def format_table(
    frame: pd.DataFrame | None,
    decimals: Mapping[str, int] | None = None,
    *,
    default_decimals: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Rend un tableau lisible : décimales maîtrisées, dates courtes, vides explicites."""
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame()
    rules = dict(decimals or {})
    defaults = dict(default_decimals or {})
    display = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if column == "Semaine":
            # Testé avant le cas général : cette colonne est un horodatage, et la
            # branche datetime la réduisait à « 3 août » sans dire que c'est une semaine.
            display[column] = series.apply(format_week_label)
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Une colonne d'horodatages perd tout son sens réduite au seul jour ;
            # et une date de tableau sans son jour de semaine oblige à le
            # retrouver de tête, alors que c'est lui qui explique la mesure.
            formatter = format_datetime if column == "Début" else format_short_date
            display[column] = series.apply(formatter)
        elif column in DURATION_COLUMNS:
            display[column] = series.apply(format_duration_minutes)
        elif column in CLOCK_COLUMNS:
            display[column] = series.apply(format_clock_hour)
        elif pd.api.types.is_bool_dtype(series):
            # « True » au milieu d'un tableau français se lit mal : oui / non.
            display[column] = series.map({True: "oui", False: "non"}).fillna(MISSING_TEXT)
        elif pd.api.types.is_numeric_dtype(series):
            places = rules.get(column, defaults.get(column, 1))
            display[column] = series.apply(lambda value, places=places: format_fr_number(value, decimals=places))
        else:
            display[column] = series.fillna(MISSING_TEXT).astype(str)
    return display
