"""Smoke tests des pages revues : Insights, Journal (ajout rapide), Prévisions, Paramètres, Dashboard."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _weights(days: int = 150, gap: tuple[int, int] | None = (40, 70)) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2026-04-01", periods=days, freq="D")
    values = 108 - 0.09 * np.arange(days) + rng.normal(0, 0.4, days) + np.where(dates.weekday == 0, 0.5, 0.0)
    df = pd.DataFrame({"Date": dates, "Poids (Kgs)": values})
    if gap:
        df = df.drop(index=list(range(*gap))).reset_index(drop=True)
    return df


def _state(at: AppTest, df: pd.DataFrame | None = None) -> None:
    df = _weights() if df is None else df
    for key in ("source_data", "working_data", "raw_data"):
        at.session_state[key] = df.copy()
    at.session_state["filtered_data"] = pd.DataFrame()
    at.session_state["filter_active"] = False
    at.session_state["target_weights"] = (100.0, 95.0, 90.0, 85.0, 80.0)
    at.session_state["fast_mode"] = True
    at.session_state["data_quality"] = {
        "source": "test",
        "raw_rows": len(df),
        "valid_rows": len(df),
        "invalid_rows": 0,
        "duplicate_dates": 0,
        "columns_kept": len(df.columns),
        "extra_columns": [],
    }


def _markdown(at: AppTest) -> str:
    return " ".join(str(m.value) for m in at.markdown)


def test_insights_page_renders_every_section_and_tests_the_weekday_effect():
    at = AppTest.from_file("app/pages/Insights.py", default_timeout=120)
    _state(at)
    at.run()
    assert not at.exception
    rendered = _markdown(at)
    for section in ("Qualité des données", "Phases du parcours", "Effet du jour de la semaine", "Séries consécutives", "Pesées atypiques"):
        assert section in rendered
    # Le constat sur le jour de semaine est testé, pas seulement affiché.
    assert "p ajustée" in rendered
    assert len(at.get("plotly_chart")) >= 3
    # Le périmètre « effort actuel » existe grâce au trou de 30 jours.
    scope = next(radio for radio in at.radio if "Périmètre" in radio.label)
    scope.set_value("Historique complet").run()
    assert not at.exception


def test_insights_anomalies_use_the_trend_not_the_global_median():
    at = AppTest.from_file("app/pages/Insights.py", default_timeout=120)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-04-01", periods=120, freq="D")
    df = pd.DataFrame({"Date": dates, "Poids (Kgs)": np.linspace(108, 97, 120) + rng.normal(0, 0.25, 120)})
    df.loc[60, "Poids (Kgs)"] += 5.0
    _state(at, df)
    at.run()
    assert not at.exception
    tables = [frame.value for frame in at.dataframe]
    flagged = [table for table in tables if "z robuste" in table.columns]
    assert flagged, "le tableau des pesées atypiques doit être rendu"
    # Une seule pesée aberrante ; les extrêmes de la perte régulière ne le sont pas.
    assert len(flagged[0]) == 1
    toggle = next(item for item in at.toggle if "IsolationForest" in item.label)
    assert toggle.value is False


def test_journal_quick_add_appends_a_validated_row_and_keeps_dtypes():
    at = AppTest.from_file("app/pages/Journal.py", default_timeout=60)
    _state(at)
    at.run()
    assert not at.exception
    before = len(at.session_state["working_data"])
    next(widget for widget in at.number_input if widget.label == "Poids (kg)").set_value(94.3)
    next(button for button in at.button if button.label == "Ajouter").click().run()
    assert not at.exception
    working = at.session_state["working_data"]
    assert len(working) == before + 1
    assert str(working["Date"].dtype).startswith("datetime64")
    assert float(working["Poids (Kgs)"].iloc[-1]) == 94.3
    assert any("ajoutée" in str(message.value) for message in at.success)
    # Les libellés historiques restent en place.
    assert "Enregistrer les modifications" in [button.label for button in at.button]
    assert "Exporter les données enregistrées" in [button.label for button in at.get("download_button")]


def test_predictions_page_shows_a_validated_projection_and_a_leaderboard_verdict():
    at = AppTest.from_file("app/pages/Predictions.py", default_timeout=180)
    _state(at)
    at.run()
    assert not at.exception
    rendered = _markdown(at)
    assert "Projection selon vos mesures" in rendered
    assert "Leaderboard" in rendered
    # Le tableau porte le verdict face à la dernière valeur.
    tables = [frame.value for frame in at.dataframe]
    leaderboard = next(table for table in tables if "Verdict" in table.columns)
    assert "Dernière valeur" in set(leaderboard["Modèle"])
    assert leaderboard["Verdict"].str.contains("dernière valeur").all()
    # Le cône d'incertitude est dessiné autour de la projection principale.
    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    names = {trace.get("name", "") for spec in specs for trace in spec.get("data", [])}
    assert any("Cône" in name for name in names)
    assert any("Poids de tendance" in name for name in names)
    labels = [metric.label for metric in at.metric]
    assert any(label.startswith("Rythme") for label in labels)
    assert "Arrivée à l'objectif" in labels


def test_settings_page_groups_sections_and_saves_without_dead_options():
    at = AppTest.from_file("app/pages/Settings.py", default_timeout=60)
    _state(at)
    at.run()
    assert not at.exception
    labels = [widget.label for widget in at.selectbox]
    assert "Gestion des doublons journaliers" in labels
    # Les réglages sans effet (modèle par défaut, thème Plotly) ont disparu.
    assert not any("Modèle par défaut" in label or "Thème Plotly" in label for label in labels)
    assert [metric.label for metric in at.metric] == ["Streamlit", "Source", "Lignes en session", "WHOOP"]
    next(button for button in at.button if button.label == "Enregistrer").click().run()
    assert not at.exception
    assert any("enregistrés" in str(message.value) for message in at.success)
    assert at.session_state["height_m"] == 1.82


def test_dashboard_shows_trend_weight_rate_interval_and_noise_band():
    at = AppTest.from_file("app/pages/Dashboard.py", default_timeout=120)
    _state(at)
    at.run()
    assert not at.exception
    labels = [metric.label for metric in at.metric]
    assert "Poids de tendance" in labels
    assert any(label.startswith("Rythme") for label in labels)
    rendered = _markdown(at)
    assert "IC 95 %" in rendered
    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    names = {trace.get("name", "") for spec in specs for trace in spec.get("data", [])}
    assert any(name.startswith("Poids de tendance") for name in names)
    assert any(name.startswith("Bruit habituel") for name in names)
    # Les deltas des KPI portent un signe ASCII, seul lu par Streamlit.
    deltas = [metric.delta for metric in at.metric if metric.delta]
    assert deltas
    assert all(not str(delta).startswith("−") for delta in deltas)
    assert "Objectifs & paliers" in [tab.label for tab in at.tabs]
