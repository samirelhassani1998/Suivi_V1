"""Tests des libellés de dates en français."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.date_labels import (
    MISSING_LABEL,
    describe_freshness,
    format_clock_hour,
    format_date_range,
    format_datetime,
    format_day_month,
    format_duration_minutes,
    format_long_date,
    format_short_date,
    format_relative_day,
    format_week_label,
)

REFERENCE = pd.Timestamp("2026-09-11")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-01-03", "3 janv."),
        ("2026-02-28", "28 févr."),
        ("2026-08-15", "15 août"),
        ("2026-09-03", "3 sept."),
        ("2026-12-25", "25 déc."),
    ],
)
def test_format_day_month_uses_french_abbreviations(value, expected):
    assert format_day_month(pd.Timestamp(value)) == expected


def test_format_short_date_names_the_weekday_before_the_day():
    assert format_short_date(pd.Timestamp("2026-09-08")) == "mar. 8 sept."
    assert format_short_date(None) == "—"


def test_format_long_date_spells_the_weekday_and_month():
    assert format_long_date(pd.Timestamp("2026-09-03")) == "jeudi 3 septembre 2026"
    assert format_long_date(pd.Timestamp("2026-09-03"), with_weekday=False) == "3 septembre 2026"


def test_format_datetime_keeps_the_hour():
    """Une synchronisation datée du seul jour ne dit pas si elle vient d'avoir lieu."""
    assert format_datetime(pd.Timestamp("2026-09-11 18:42")) == "11 sept. à 18:42"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-11", "aujourd'hui"),
        ("2026-09-10", "hier"),
        ("2026-09-09", "avant-hier"),
        ("2026-09-06", "il y a 5 jours"),
        ("2026-09-04", "il y a 7 jours"),
    ],
)
def test_format_relative_day_reads_like_speech(value, expected):
    assert format_relative_day(pd.Timestamp(value), today=REFERENCE) == expected


def test_format_relative_day_falls_back_to_a_date_beyond_a_week():
    """Au-delà de sept jours, un décompte cesse d'être parlant."""
    assert format_relative_day(pd.Timestamp("2026-08-20"), today=REFERENCE) == "20 août"


def test_describe_freshness_flags_data_that_stopped_days_ago():
    fresh = describe_freshness(pd.Timestamp("2026-09-10"), today=REFERENCE)
    assert fresh["days"] == 1 and not fresh["stale"] and fresh["tone"] == "success"

    stale = describe_freshness(pd.Timestamp("2026-09-05"), today=REFERENCE)
    assert stale["days"] == 6 and stale["stale"] and stale["tone"] == "warning"
    assert stale["label"] == "il y a 6 jours"


def test_describe_freshness_without_any_date_stays_neutral():
    assert describe_freshness(None)["days"] is None


def test_format_date_range_mentions_the_year_once():
    assert format_date_range("2026-09-03", "2026-09-11") == "3 sept. → 11 sept. 2026"
    assert format_date_range("2025-12-28", "2026-01-04") == "28 déc. 2025 → 4 janv. 2026"


def test_format_week_label_names_the_week_by_its_monday():
    assert format_week_label(pd.Timestamp("2026-08-03")) == "Semaine du 3 août"


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(43.4038, "43 min"), (90, "1 h 30"), (120, "2 h"), (0, "0 min"), (125.6, "2 h 06")],
)
def test_format_duration_minutes_reads_in_hours_and_minutes(minutes, expected):
    assert format_duration_minutes(minutes) == expected


def test_format_duration_rejects_nonsense():
    assert format_duration_minutes(-5) == MISSING_LABEL
    assert format_duration_minutes("abc") == MISSING_LABEL


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.5, "22:30"), (0.5, "00:30"), (-0.25, "23:45"), (1.0, "01:00")],
)
def test_format_clock_hour_turns_the_continuous_scale_back_into_a_clock(value, expected):
    """L'heure de coucher est stockée en négatif avant minuit pour rester continue."""
    assert format_clock_hour(value) == expected


def test_every_formatter_survives_a_missing_value():
    for formatter in (format_day_month, format_long_date, format_datetime, format_relative_day, format_week_label, format_clock_hour):
        assert formatter(None) == MISSING_LABEL
    assert format_date_range(None, None) == MISSING_LABEL


def test_describe_freshness_keeps_counting_days_beyond_a_week():
    """Mesurer l'ancienneté est le rôle de cette fonction : la date n'y suffit pas."""
    stale = describe_freshness(pd.Timestamp("2026-08-25"), today=REFERENCE)
    assert stale["days"] == 17
    assert stale["label"] == "il y a 17 jours"
    assert stale["stale"]
    # L'affichage courant, lui, bascule bien sur la date absolue.
    assert format_relative_day(pd.Timestamp("2026-08-25"), today=REFERENCE) == "25 août"
