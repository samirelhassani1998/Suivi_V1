"""Une barre par séance et aucun total partiel présenté comme complet."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.boxing_analytics import boxing_summary, session_table
from app.core.whoop_analytics import session_log
from app.ui.boxing_visuals import boxing_sessions_chart, session_intensity_chart
from app.ui.whoop_visuals import sessions_timeline_chart


def _sessions():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-09-10", "2026-09-10"]),
        "Sport": ["boxing", "boxing"],
        "Durée (min)": [60., 50.], "Strain séance": [10., 15.],
        "Calories séance (kcal)": [500., 400.],
        **{f"Zone {n} (min)": [10., 8.] for n in range(6)},
    })


@pytest.mark.parametrize("column,total", [("Durée (min)", "total_minutes"), ("Calories séance (kcal)", "total_calories")])
@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_summary_does_not_label_a_partial_sum_as_a_total(column, total, missing):
    workouts = _sessions()
    workouts.loc[1, column] = missing
    summary = boxing_summary(session_table(None, workouts), window_days=7)
    assert summary["sessions"] == 2
    assert np.isnan(summary[total])


def test_summary_retains_complete_totals():
    summary = boxing_summary(session_table(None, _sessions()), window_days=7)
    assert summary["total_minutes"] == 110
    assert summary["total_calories"] == 900


@pytest.mark.parametrize("start_case", ["absent", "null", "identical"])
@pytest.mark.parametrize("chart", [boxing_sessions_chart, session_intensity_chart, sessions_timeline_chart])
def test_sessions_with_ambiguous_start_times_get_distinct_honest_categories(chart, start_case):
    workouts = _sessions()
    if start_case == "null":
        workouts["Début"] = pd.NaT
    elif start_case == "identical":
        workouts["Début"] = pd.Timestamp("2026-09-10 08:00")
    table = session_log(None, workouts) if chart is sessions_timeline_chart else session_table(None, workouts)
    figure = chart(table)
    assert figure.layout.xaxis.type == "category"
    positions = list(figure.data[0].x)
    assert len(set(positions)) == 2
    assert set(positions) == {"10/09/2026 · séance 1", "10/09/2026 · séance 2"}
    assert "heures absentes ou identiques" in figure.layout.xaxis.title.text
    hover = " ".join(str(item) for item in figure.data[0].customdata)
    if start_case == "identical":
        assert "08:00" in hover
    else:
        assert "00:00" not in hover
        assert "08:00" not in hover
    if chart is session_intensity_chart:
        # Les trois composantes d'une séance partagent sa catégorie ; les deux
        # séances ne peuvent plus s'additionner en une seule barre de 108 min.
        assert all(list(trace.x) == positions for trace in figure.data)
        assert [sum(float(trace.y[i]) for trace in figure.data) for i in range(2)] == [60., 48.]
