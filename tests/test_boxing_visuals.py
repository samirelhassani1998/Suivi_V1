"""Tests des figures de l'onglet Boxe : encodage, couleurs et garde-fous de lisibilité."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.core.boxing_analytics import boxing_habits, boxing_sessions, recovery_profile, session_table, weekly_sessions
from app.ui.boxing_visuals import (
    DAY_BAR_WIDTH_MS,
    SEQUENTIAL_BLUES,
    boxing_sessions_chart,
    habits_heatmap,
    late_sessions_chart,
    progression_chart,
    readiness_scatter,
    recovery_profile_chart,
    session_intensity_chart,
    weekly_frequency_chart,
)
from app.ui.whoop_visuals import SERIES_COLORS, ZONE_COLORS

TODAY = pd.Timestamp("2026-10-03")


def _spec(figure) -> dict:
    return json.loads(figure.to_json())


def _table(days: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(end=TODAY, periods=days, freq="D")
    daily = pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": rng.uniform(20, 95, days),
            "FC repos (bpm)": 56.0,
            "Sommeil (heures)": rng.normal(7, 0.5, days),
            "Heure de coucher": rng.normal(-0.5, 0.5, days),
        }
    )
    rows = []
    for index, date in enumerate(dates[::2]):
        rows.append(
            {
                "Date": date,
                "Début": date + pd.Timedelta(hours=19 if index % 2 else 12),
                "Sport": "boxing",
                "Durée (min)": 60.0,
                "Strain séance": float(rng.uniform(10, 16)),
                "Calories séance (kcal)": 600.0,
                "FC moyenne (bpm)": 145.0,
                "FC max (bpm)": 182.0,
                **{f"Zone {zone} (min)": float(rng.uniform(2, 15)) for zone in range(6)},
            }
        )
    return daily, session_table(daily, boxing_sessions(pd.DataFrame(rows)), max_hr=190.0)


def test_sessions_chart_paints_each_bar_with_the_status_color_of_its_morning():
    _, table = _table()

    traces = _spec(boxing_sessions_chart(table))["data"]

    by_name = {trace["name"]: trace for trace in traces}
    for zone, label in (("Vert", "Matin vert"), ("Jaune", "Matin jaune"), ("Rouge", "Matin rouge")):
        if label in by_name:
            assert by_name[label]["marker"]["color"] == ZONE_COLORS[zone]


def test_sessions_chart_gives_every_bar_the_same_width():
    """Laissée à Plotly, la largeur se calcule par trace : une zone rare s'étalait sur des semaines."""
    _, table = _table()

    traces = _spec(boxing_sessions_chart(table))["data"]

    assert traces and all(trace["width"] == DAY_BAR_WIDTH_MS for trace in traces)


def test_charts_return_none_without_usable_data():
    empty = pd.DataFrame()
    assert boxing_sessions_chart(empty) is None
    assert weekly_frequency_chart(empty) is None
    assert recovery_profile_chart(empty, 60.0) is None
    assert readiness_scatter(empty) is None
    assert late_sessions_chart(empty) is None
    assert habits_heatmap(pd.DataFrame(0, index=["a"], columns=["lun."])) is None
    assert session_intensity_chart(empty) is None
    assert progression_chart(empty, "Strain séance") is None


def test_no_boxing_chart_uses_a_second_vertical_axis():
    daily, table = _table()
    figures = [
        boxing_sessions_chart(table),
        weekly_frequency_chart(weekly_sessions(table, start=table["Date"].min(), end=TODAY)),
        recovery_profile_chart(recovery_profile(daily, table)["table"], 60.0),
        readiness_scatter(table),
        late_sessions_chart(table),
        session_intensity_chart(table),
        progression_chart(table, "Strain séance"),
    ]
    for figure in figures:
        assert figure is not None
        layout = _spec(figure)["layout"]
        assert "yaxis2" not in layout


def test_single_series_charts_hide_their_legend_and_use_the_first_slot():
    _, table = _table()
    weekly = weekly_frequency_chart(weekly_sessions(table, start=table["Date"].min(), end=TODAY))

    spec = _spec(weekly)

    assert spec["layout"]["showlegend"] is False
    assert spec["data"][0]["marker"]["color"] == SERIES_COLORS[0]


def test_weekly_chart_axis_is_labelled_in_french():
    _, table = _table()

    spec = _spec(weekly_frequency_chart(weekly_sessions(table, start=table["Date"].min(), end=TODAY)))

    ticks = " ".join(spec["layout"]["xaxis"].get("ticktext", []))
    assert ticks and not any(month in ticks for month in ("Sep", "Oct", "Aug"))


def test_habits_heatmap_leaves_empty_cells_blank_and_uses_one_hue():
    _, table = _table()

    spec = _spec(habits_heatmap(boxing_habits(table)["matrix"]))

    values = spec["data"][0]["z"]
    assert any(value is None for row in values for value in row)
    assert [color for _, color in spec["data"][0]["colorscale"]] == list(SEQUENTIAL_BLUES)


def test_intensity_chart_stacks_three_bands_with_a_surface_gap():
    _, table = _table()

    spec = _spec(session_intensity_chart(table))

    assert spec["layout"]["barmode"] == "stack"
    assert [trace["name"] for trace in spec["data"]] == ["Facile (zones 0–2)", "Modéré (zone 3)", "Dur (zones 4–5)"]
    assert all(trace["marker"]["line"]["width"] == 2 for trace in spec["data"])


def test_late_sessions_chart_marks_the_four_hour_threshold():
    _, table = _table()

    spec = _spec(late_sessions_chart(table))

    shapes = spec["layout"].get("shapes", [])
    assert any(shape.get("x0") == 4.0 and shape.get("x1") == 4.0 for shape in shapes)


def test_recovery_profile_draws_the_overall_mean_as_a_reference():
    daily, table = _table()

    spec = _spec(recovery_profile_chart(recovery_profile(daily, table)["table"], 61.0))

    assert any(shape.get("y0") == 61.0 and shape.get("y1") == 61.0 for shape in spec["layout"].get("shapes", []))
    assert spec["data"][0]["error_y"]["type"] == "data"
