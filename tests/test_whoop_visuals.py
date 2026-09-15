"""Tests des figures WHOOP : encodage, couleurs et garde-fous de lisibilité.

Ces règles sont vérifiables automatiquement ; les laisser à la relecture
visuelle, c'est accepter qu'elles se dégradent au fil des modifications.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.core.whoop_analytics import calendar_matrix, hr_zone_profile, indexed_series, rolling_trend, session_log, smoothed_baseline
from app.ui.whoop_visuals import (
    RECOVERY_BANDS,
    STRAIN_BANDS,
    SERIES_COLORS,
    intensity_profile_chart,
    recovery_bars_chart,
    sessions_timeline_chart,
    smoothed_baseline_chart,
    STATUS_CRITICAL,
    STATUS_GOOD,
    STATUS_WARNING,
    ZONE_COLORS,
    indexed_comparison_chart,
    recovery_calendar,
    recovery_gauge,
    series_chart,
    sleep_stages_chart,
    sparkline,
    weekday_chart,
    zone_distribution_chart,
)


def _spec(figure) -> dict:
    return json.loads(figure.to_json())


def _grid(days: int = 12, *, gaps: tuple[int, ...] = ()) -> pd.DataFrame:
    dates = pd.date_range("2026-08-01", periods=days, freq="D")
    values = np.linspace(40, 80, days)
    recovery = pd.Series(values)
    sleep = pd.Series(np.linspace(6.0, 8.0, days))
    for index in gaps:
        recovery.iloc[index] = np.nan
        sleep.iloc[index] = np.nan
    return pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": recovery,
            "Sommeil (heures)": sleep,
            "Sommeil profond (heures)": sleep / 5,
            "Sommeil REM (heures)": sleep / 4,
        }
    )


# ── Séries temporelles ────────────────────────────────────────────────────────


def test_series_chart_never_bridges_a_day_without_measurement():
    figure = series_chart(_grid(gaps=(3, 4)), ["Récupération (%)"], "Titre", "%")

    traces = _spec(figure)["data"]
    assert traces
    assert all(trace["connectgaps"] is False for trace in traces)
    assert any(value is None for value in traces[0]["y"])


def test_series_trend_line_also_breaks_on_gaps():
    """Une tendance lissée qui enjambe les trous ment autant qu'une série brute."""
    grid = _grid(20, gaps=(5, 6, 7))
    trend = pd.DataFrame({"Date": grid["Date"], "Récupération (%)": grid["Récupération (%)"].rolling(3, min_periods=2).mean()})

    figure = series_chart(grid, ["Récupération (%)"], "Titre", "%", trend=trend)

    traces = {trace["name"]: trace for trace in _spec(figure)["data"]}
    assert "Tendance 7 jours" in traces
    assert traces["Tendance 7 jours"]["connectgaps"] is False


def test_series_chart_hides_the_legend_for_a_single_series():
    """Le titre nomme déjà la série ; une légende d'un seul élément est du bruit."""
    single = _spec(series_chart(_grid(), ["Récupération (%)"], "Titre", "%"))
    assert single["layout"]["showlegend"] is False

    multiple = _spec(series_chart(_grid(), ["Récupération (%)", "Sommeil (heures)"], "Titre", "u"))
    assert multiple["layout"]["showlegend"] is True


def test_series_chart_assigns_palette_colors_in_fixed_order():
    """La couleur suit l'entité, jamais son rang d'affichage."""
    figure = _spec(series_chart(_grid(), ["Récupération (%)", "Sommeil (heures)"], "Titre", "u"))

    colors = [trace["line"]["color"] for trace in figure["data"]]
    assert colors == [SERIES_COLORS[0], SERIES_COLORS[1]]


def test_series_chart_labels_its_axis_in_french_not_in_plotly_english():
    """Plotly ne connaît que les mois anglais : « Sep 3 » dans une interface française."""
    figure = _spec(series_chart(_grid(20), ["Récupération (%)"], "Titre", "%"))

    axis = figure["layout"]["xaxis"]
    assert axis["tickmode"] == "array"
    assert axis["ticktext"], "l'axe doit porter des étiquettes explicites"
    assert all("Sep" not in label and "Aug" not in label for label in axis["ticktext"])
    assert any("août" in label or "sept." in label for label in axis["ticktext"])


def test_series_chart_axis_never_degrades_to_hourly_ticks():
    """Avec peu de points, Plotly graduerait sinon en heures des mesures quotidiennes."""
    axis = _spec(series_chart(_grid(3), ["Récupération (%)"], "Titre", "%"))["layout"]["xaxis"]
    assert axis.get("tickmode") == "array" or axis.get("tickformat") == "%d/%m"


def test_series_chart_returns_none_when_no_metric_is_usable():
    grid = _grid()
    grid["Récupération (%)"] = np.nan
    assert series_chart(grid, ["Récupération (%)"], "Titre", "%") is None
    assert series_chart(grid, ["Métrique absente"], "Titre", "%") is None


def test_series_chart_grid_is_solid_and_recessive():
    """Une grille en pointillés se lit comme un seuil ou une projection."""
    layout = _spec(series_chart(_grid(), ["Récupération (%)"], "Titre", "%"))["layout"]
    assert layout["yaxis"]["griddash"] == "solid"
    assert layout["xaxis"]["griddash"] == "solid"


# ── Comparaison poids / métrique ──────────────────────────────────────────────


def test_indexed_comparison_uses_a_single_axis():
    """Deux échelles verticales fabriqueraient une corrélation visuelle arbitraire."""
    merged = _grid(15).assign(**{"Poids (Kgs)": np.linspace(104, 102, 15)})
    indexed = indexed_series(merged, ["Poids (Kgs)", "Récupération (%)"])

    figure = _spec(indexed_comparison_chart(indexed, ["Poids (Kgs)", "Récupération (%)"], "Titre"))

    assert "yaxis2" not in figure["layout"]
    assert all("yaxis" not in trace or trace["yaxis"] == "y" for trace in figure["data"])
    # Les deux séries partent bien de la même base.
    for trace in figure["data"]:
        first = next(value for value in trace["y"] if value is not None)
        assert first == pytest.approx(100.0)


def test_indexed_comparison_returns_none_without_usable_metric():
    empty = pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=3)})
    assert indexed_comparison_chart(empty, ["Poids (Kgs)"], "Titre") is None


# ── Zones, calendrier, profils ────────────────────────────────────────────────


def test_zone_chart_uses_status_colors_matching_each_zone():
    counts = pd.DataFrame({"Zone": ["Rouge", "Jaune", "Vert"], "Jours": [2, 5, 3], "Part (%)": [20.0, 50.0, 30.0]})

    figure = _spec(zone_distribution_chart(counts))

    assert figure["data"][0]["marker"]["color"] == [STATUS_CRITICAL, STATUS_WARNING, STATUS_GOOD]
    assert ZONE_COLORS["Vert"] == STATUS_GOOD


def test_zone_chart_on_empty_counts_returns_none():
    assert zone_distribution_chart(pd.DataFrame()) is None


def test_recovery_gauge_marks_the_whoop_thresholds():
    figure = _spec(recovery_gauge(72.0, "Vert"))

    gauge = figure["data"][0]["gauge"]
    ranges = [step["range"] for step in gauge["steps"]]
    assert ranges == [[0, 34], [34, 67], [67, 100]]
    assert gauge["bar"]["color"] == STATUS_GOOD
    assert figure["data"][0]["value"] == 72.0


def test_recovery_gauge_tolerates_a_missing_score():
    assert _spec(recovery_gauge(float("nan"), None))["data"][0]["value"] == 0.0


def test_recovery_calendar_orders_weeks_top_down_and_leaves_gaps_blank():
    frame = _grid(14, gaps=(2, 3))
    matrix = calendar_matrix(frame, "Récupération (%)")

    figure = _spec(recovery_calendar(matrix))

    heatmap = figure["data"][0]
    assert heatmap["x"] == ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    # La semaine la plus ancienne se lit en haut, comme dans un calendrier.
    assert figure["layout"]["yaxis"]["autorange"] == "reversed"
    assert any(value is None for row in heatmap["z"] for value in row)
    # Une échelle de chaleur colorée exige sa légende chiffrée.
    assert heatmap["colorbar"]["title"]["text"] == "%"


def test_recovery_calendar_on_empty_matrix_returns_none():
    assert recovery_calendar({}) is None
    assert recovery_calendar(calendar_matrix(pd.DataFrame(), "Récupération (%)")) is None


def test_weekday_chart_paints_one_series_in_one_color():
    """Colorer chaque barre selon sa hauteur redoublerait l'information déjà lue."""
    profile = pd.DataFrame(
        {
            "Jour": ["Lundi", "Mardi", "Mercredi"],
            "Moyenne": [45.0, 60.0, 72.0],
            "Observations": [3, 3, 3],
        }
    )

    figure = _spec(weekday_chart(profile, "Récupération (%)"))

    assert figure["data"][0]["marker"]["color"] == SERIES_COLORS[0]
    assert figure["layout"]["showlegend"] is False


def test_weekday_chart_skips_days_without_any_measurement():
    profile = pd.DataFrame({"Jour": ["Lundi", "Mardi"], "Moyenne": [50.0, np.nan], "Observations": [2, 0]})
    figure = _spec(weekday_chart(profile, "Récupération (%)"))
    assert figure["data"][0]["x"] == ["Lundi"]


def test_sleep_stages_are_stacked_with_a_surface_gap():
    figure = _spec(sleep_stages_chart(_grid()))

    assert figure["layout"]["barmode"] == "stack"
    # Un filet de surface sépare les segments, plutôt qu'un contour tracé.
    assert all(trace["marker"]["line"]["width"] == 2 for trace in figure["data"])


def test_sleep_stages_without_stage_columns_returns_none():
    assert sleep_stages_chart(pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=3)})) is None


def test_sparkline_carries_no_axes_legend_or_hover():
    figure = _spec(sparkline([1.0, 2.0, 1.5]))

    assert figure["layout"]["xaxis"]["visible"] is False
    assert figure["layout"]["yaxis"]["visible"] is False
    assert figure["layout"]["showlegend"] is False
    assert figure["data"][0]["hoverinfo"] == "skip"


@pytest.mark.parametrize(("value", "expected"), [(-5.0, 0.0), (140.0, 100.0), (float("nan"), 0.0)])
def test_recovery_gauge_clamps_values_to_its_dial(value, expected):
    """Une valeur hors 0-100 dessinerait une aiguille hors du cadran."""
    assert _spec(recovery_gauge(value, "Vert"))["data"][0]["value"] == expected


# ── Lecture par date et repères de fond ───────────────────────────────────────


def test_recovery_bars_take_the_color_of_their_whoop_zone():
    frame = pd.DataFrame(
        {"Date": pd.date_range("2026-08-01", periods=3, freq="D"), "Récupération (%)": [20.0, 50.0, 80.0]}
    )

    figure = _spec(recovery_bars_chart(frame, trend=rolling_trend(frame, "Récupération (%)")))

    bars = figure["data"][0]
    assert bars["type"] == "bar"
    assert bars["marker"]["color"] == [STATUS_CRITICAL, STATUS_WARNING, STATUS_GOOD]
    # Les zones sont dessinées en fond, les seuils 34 et 67 matérialisés.
    shapes = figure["layout"]["shapes"]
    assert sorted((shape["y0"], shape["y1"]) for shape in shapes) == [(0.0, 34.0), (34.0, 67.0), (67.0, 100.0)]
    # Chaque barre nomme sa date en toutes lettres au survol.
    assert "août 2026" in bars["customdata"][0]


def test_recovery_bars_return_none_without_scores():
    assert recovery_bars_chart(pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=2), "Récupération (%)": [np.nan, np.nan]})) is None


def test_series_chart_draws_the_whoop_strain_levels_in_the_background():
    frame = pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=4, freq="D"), "Strain": [5.0, 12.0, 15.0, 19.0]})

    figure = _spec(series_chart(frame, ["Strain"], "Charge", "Strain", bands=STRAIN_BANDS))

    shapes = figure["layout"]["shapes"]
    assert [shape["y1"] for shape in shapes] == [10.0, 14.0, 18.0, 21.0]
    assert all(shape["layer"] == "below" for shape in shapes)
    labels = [annotation["text"] for annotation in figure["layout"]["annotations"]]
    assert labels == ["Léger", "Modéré", "Élevé", "Maximal"]


def test_sleep_stages_stack_light_sleep_and_awake_time_on_top():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=2, freq="D"),
            "Sommeil profond (heures)": [1.5, 1.4],
            "Sommeil REM (heures)": [1.8, 1.7],
            "Sommeil léger (heures)": [3.6, 3.9],
            "Éveil (heures)": [0.4, 0.5],
        }
    )

    figure = _spec(sleep_stages_chart(frame))

    names = [trace["name"] for trace in figure["data"]]
    # Du plus profond au plus léger, l'éveil en dernier : l'empilement se lit comme une nuit.
    assert names == ["Sommeil profond", "Sommeil REM", "Sommeil léger", "Éveil"]
    assert figure["layout"]["barmode"] == "stack"


def test_smoothed_baseline_chart_shows_nights_smoothing_and_the_usual_range():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {"Date": pd.date_range("2026-07-01", periods=50, freq="D"), "HRV (ms)": np.exp(np.log(45) + rng.normal(0, 0.15, 50))}
    )
    trend = smoothed_baseline(frame, "HRV (ms)", log_scale=True)

    figure = _spec(smoothed_baseline_chart(trend["series"], "HRV (ms)", "Variabilité cardiaque", "ms"))

    modes = [trace["mode"] for trace in figure["data"]]
    assert modes == ["markers", "lines"]
    band = figure["layout"]["shapes"][0]
    assert band["y0"] == pytest.approx(trend["low"])
    assert band["y1"] == pytest.approx(trend["high"])
    assert figure["layout"]["annotations"][0]["text"] == "Plage habituelle"


def test_sessions_timeline_names_the_next_morning_in_its_hover():
    dates = pd.date_range("2026-09-01", periods=3, freq="D")
    daily = pd.DataFrame({"Date": dates, "Récupération (%)": [60.0, 30.0, 80.0]})
    workouts = pd.DataFrame(
        {
            "Date": [dates[0], dates[1]],
            "Début": [dates[0] + pd.Timedelta(hours=18), dates[1] + pd.Timedelta(hours=7)],
            "Sport": ["boxing", "running"],
            "Strain séance": [12.0, 8.0],
        }
    )

    figure = _spec(sessions_timeline_chart(session_log(daily, workouts)))

    names = [trace["name"] for trace in figure["data"]]
    assert set(names) == {"Boxe", "Course à pied"}
    boxing = next(trace for trace in figure["data"] if trace["name"] == "Boxe")
    assert "Lendemain : 30 %" in boxing["customdata"][0]
    assert "mardi 1 septembre 2026" in boxing["customdata"][0]


def test_sessions_timeline_without_sessions_returns_none():
    assert sessions_timeline_chart(pd.DataFrame()) is None


def test_intensity_profile_stacks_to_one_hundred_percent_per_sport():
    workouts = pd.DataFrame(
        {
            "Sport": ["boxing", "running"],
            "Zone 0 (min)": [0.0, 10.0],
            "Zone 1 (min)": [5.0, 20.0],
            "Zone 2 (min)": [5.0, 20.0],
            "Zone 3 (min)": [10.0, 5.0],
            "Zone 4 (min)": [20.0, 0.0],
            "Zone 5 (min)": [10.0, 0.0],
        }
    )

    figure = _spec(intensity_profile_chart(hr_zone_profile(workouts, min_sessions=1)))

    assert figure["layout"]["barmode"] == "stack"
    assert figure["layout"]["xaxis"]["range"] == [0, 100]
    totals = np.sum([trace["x"] for trace in figure["data"]], axis=0)
    assert all(abs(total - 100.0) < 0.2 for total in totals)
