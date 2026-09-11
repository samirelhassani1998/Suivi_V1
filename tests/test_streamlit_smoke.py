from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.core.formatting import format_fr_kg
from app.core.data_editing import has_unsaved_changes
from streamlit.testing.v1 import AppTest


def _state(at: AppTest) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=80),
            "Poids (Kgs)": [95 - i * 0.08 for i in range(80)],
            "Extra Col": [f"v{i}" for i in range(80)],
            "Moment": ["08:00" for _ in range(80)],
        }
    )
    at.session_state["source_data"] = df.copy()
    at.session_state["working_data"] = df.copy()
    at.session_state["filtered_data"] = df.copy()
    at.session_state["raw_data"] = df.copy()
    at.session_state["target_weights"] = (100.0, 95.0, 90.0, 85.0, 80.0)
    at.session_state["fast_mode"] = True
    at.session_state["data_quality"] = {"source": "test", "raw_rows": 80, "valid_rows": 80, "invalid_rows": 0, "duplicate_dates": 0, "columns_kept": len(df.columns), "extra_columns": ["Extra Col", "Moment"]}


def _active_trajectory_state(at: AppTest, *, offset_kg: float = 0.0) -> None:
    dates = pd.date_range(
        "2026-09-03",
        periods=40,
        freq="D",
    )

    weights = [
        106.1 - index * 0.22
        for index in range(len(dates))
    ]
    weights[-1] += offset_kg

    df = pd.DataFrame(
        {
            "Date": dates,
            "Poids (Kgs)": weights,
            "Moment": ["08:00"] * len(dates),
            "Extra Col": [
                f"v{index}"
                for index in range(len(dates))
            ],
        }
    )

    at.session_state["source_data"] = df.copy()
    at.session_state["working_data"] = df.copy()
    at.session_state["filtered_data"] = pd.DataFrame()
    at.session_state["filter_active"] = False
    at.session_state["raw_data"] = df.copy()
    at.session_state["target_weights"] = (
        100.0,
        95.0,
        90.0,
        85.0,
        80.0,
    )
    at.session_state["fast_mode"] = True
    at.session_state["data_quality"] = {
        "source": "test",
        "raw_rows": len(df),
        "valid_rows": len(df),
        "invalid_rows": 0,
        "duplicate_dates": 0,
        "columns_kept": len(df.columns),
        "extra_columns": ["Moment", "Extra Col"],
    }


def test_dashboard_renders_active_target_trajectory_without_exception():
    at = AppTest.from_file(
        "app/pages/Dashboard.py"
    )

    _active_trajectory_state(at)
    at.run(timeout=15)

    assert not at.exception

    rendered = " ".join(
        str(element.value)
        for element in at.markdown
    )

    assert "Trajectoire cible" in rendered
    assert "Rythme moyen requis" in rendered
    assert "Écart" in rendered
    assert "statut" in rendered


@pytest.mark.parametrize(
    ("offset_kg", "expected_status"),
    [
        (-2.0, "en avance"),
        (0.0, "aligné"),
    ],
)
def test_dashboard_renders_active_target_trajectory_status_variants_without_exception(offset_kg, expected_status):
    at = AppTest.from_file("app/pages/Dashboard.py")

    _active_trajectory_state(at, offset_kg=offset_kg)
    at.run(timeout=15)

    assert not at.exception

    rendered = " ".join(str(element.value) for element in at.markdown)
    assert "Trajectoire cible" in rendered
    assert f"statut : {expected_status}" in rendered

def test_dashboard_renders_kpis_and_sections():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    markdown_values = [str(m.value) for m in at.markdown]
    assert any("Dashboard" in value for value in markdown_values)
    assert len(at.metric) >= 3
    plotly_elements = at.get("plotly_chart")
    assert len(plotly_elements) >= 2
    trace_names = []
    for chart in plotly_elements:
        spec = json.loads(chart.proto.spec)
        trace_names.extend(trace.get("name", "") for trace in spec.get("data", []))
    normalized_trace_names = " ".join(trace_names).lower()
    assert "poids" in normalized_trace_names
    assert "objectif" in normalized_trace_names or "cible" in normalized_trace_names
    target_traces = []
    for chart in plotly_elements:
        spec = json.loads(chart.proto.spec)
        target_traces.extend(
            trace for trace in spec.get("data", [])
            if "Trajectoire cible vers 80 kg au 16/12/2026" in trace.get("name", "")
        )
    assert target_traces
    trace = next(trace for trace in target_traces if trace["x"][0].startswith("2026-09-03"))
    assert trace["x"][0].startswith("2026-09-03")
    assert trace["y"][0] == 106.1
    assert trace["x"][-1].startswith("2026-12-16")
    assert trace["y"][-1] == 80.0
    assert len(trace["x"]) == 105
    assert all(not x.startswith("2026-12-17") for x in trace["x"])
    assert any("objectif" in str(c.value).lower() for c in at.caption)


def test_journal_loads_and_keeps_state_on_navigation():
    at = AppTest.from_file("app/pages/Journal.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    assert len(at.dataframe) >= 1
    assert "Extra Col" in at.session_state["working_data"].columns
    assert any("Qualité données" in str(info.value) for info in at.info)
    button_labels = [b.label for b in at.button]
    assert "Enregistrer les modifications" in button_labels
    downloads = at.get("download_button")
    download_labels = [d.label for d in downloads]
    assert "Exporter les données enregistrées" in download_labels

    # navigation simulée vers une autre page puis retour
    at_dash = AppTest.from_file("app/pages/Dashboard.py")
    for k in ["source_data", "working_data", "filtered_data", "raw_data", "target_weights", "fast_mode"]:
        at_dash.session_state[k] = at.session_state[k]
    at_dash.run()
    assert not at_dash.exception

    at_back = AppTest.from_file("app/pages/Journal.py")
    for k in ["source_data", "working_data", "filtered_data", "raw_data", "target_weights", "fast_mode"]:
        at_back.session_state[k] = at_dash.session_state[k]
    at_back.run()
    assert not at_back.exception
    assert len(at_back.session_state["working_data"]) == 80
    assert "Extra Col" in at_back.session_state["working_data"].columns


def test_predictions_render_multiple_sections_even_if_submodel_fails():
    at = AppTest.from_file("app/pages/Predictions.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    rendered_text = " ".join([str(s.value) for s in at.subheader] + [str(m.value) for m in at.markdown])
    assert "Leaderboard" in rendered_text
    assert "Projection selon vos mesures" in rendered_text
    tab_labels = [t.label for t in at.tabs]
    assert any("SARIMA" in t for t in tab_labels)
    assert any("Auto-ARIMA" in t for t in tab_labels)


def test_settings_exposes_five_goals():
    at = AppTest.from_file("app/pages/Settings.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    labels = [n.label for n in at.number_input]
    assert "Objectif 1 (kg)" in labels
    assert "Objectif 2 (kg)" in labels
    assert "Objectif 3 (kg)" in labels
    assert "Objectif 4 (kg)" in labels
    assert "Objectif 5 (kg)" in labels
    date_labels = [d.label for d in at.date_input]
    assert "Début du zoom trajectoire" in date_labels
    assert "Fin du zoom trajectoire" in date_labels
    assert at.session_state.get("zoom_target_start_date") == pd.Timestamp("2026-09-03")
    assert at.session_state.get("zoom_target_end_date") == pd.Timestamp("2026-12-16")


def test_dashboard_zoom_chart_uses_configured_period_and_handles_empty_data():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.run(timeout=10)

    assert not at.exception
    assert any("Aucune donnée de poids disponible sur cette période." in str(info.value) for info in at.info)

    plotly_elements = at.get("plotly_chart")
    assert len(plotly_elements) >= 2
    zoom_spec = json.loads(plotly_elements[1].proto.spec)
    xaxis = zoom_spec.get("layout", {}).get("xaxis", {})
    assert xaxis.get("range", [])[0].startswith("2026-09-03")
    assert xaxis.get("range", [])[1].startswith("2026-12-16")
    measured_traces = [trace for trace in zoom_spec.get("data", []) if trace.get("name") == "Poids mesuré"]
    assert measured_traces == []


def test_dashboard_zoom_chart_filters_measured_data_to_configured_period():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _active_trajectory_state(at)
    at.session_state["zoom_target_start_date"] = pd.Timestamp("2026-09-10")
    at.session_state["zoom_target_end_date"] = pd.Timestamp("2026-09-15")
    at.run(timeout=15)

    assert not at.exception
    plotly_elements = at.get("plotly_chart")
    assert len(plotly_elements) >= 2
    zoom_spec = json.loads(plotly_elements[1].proto.spec)
    measured_trace = next(trace for trace in zoom_spec.get("data", []) if trace.get("name") == "Poids mesuré")
    assert measured_trace["x"][0].startswith("2026-09-10")
    assert measured_trace["x"][-1].startswith("2026-09-15")
    assert all("2026-09-03" not in x for x in measured_trace["x"])


def test_dashboard_zoom_invalid_period_warns_without_exception():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _active_trajectory_state(at)
    at.session_state["zoom_target_start_date"] = pd.Timestamp("2026-12-16")
    at.session_state["zoom_target_end_date"] = pd.Timestamp("2026-09-03")
    at.run(timeout=15)

    assert not at.exception
    assert any("La date de début du zoom" in str(w.value) for w in at.warning)


def test_dashboard_migrates_legacy_four_goals_to_requested_five_goals():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.session_state["target_weights"] = (100.0, 95.0, 90.0, 85.0)
    at.session_state["target_weight"] = 85.0
    at.run(timeout=10)
    assert not at.exception
    assert at.session_state["target_weights"] == (100.0, 95.0, 90.0, 85.0, 80.0)
    assert at.session_state["target_weight"] == 80.0
    assert any(f"Objectif 5: {format_fr_kg(80.0)}" in str(c.value) for c in at.caption)


def test_has_unsaved_changes_detects_real_dataframe_differences():
    saved = pd.DataFrame({"Date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")], "Poids (Kgs)": [80.0, 79.8]})
    assert has_unsaved_changes(saved.copy(), saved) is False
    changed = saved.copy(); changed.loc[0, "Poids (Kgs)"] = 80.2
    assert has_unsaved_changes(changed, saved) is True
    added = pd.concat([saved, pd.DataFrame({"Date": [pd.Timestamp("2026-01-03")], "Poids (Kgs)": [79.6]})], ignore_index=True)
    assert has_unsaved_changes(added, saved) is True
    assert has_unsaved_changes(saved.iloc[:1], saved) is True
    typed = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "Poids (Kgs)": ["80", "79,8"]})
    assert has_unsaved_changes(typed, saved) is False
    typed_decimal = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "Poids (Kgs)": ["80,0", "79,8"]})
    assert has_unsaved_changes(typed_decimal, saved) is False
    french_date = pd.DataFrame({"Date": ["01/01/2026", "02/01/2026"], "Poids (Kgs)": [80, 79.8]})
    assert has_unsaved_changes(french_date, saved) is False
    november_date = pd.DataFrame({"Date": ["01/11/2026"], "Poids (Kgs)": [80]})
    november_ts = pd.DataFrame({"Date": [pd.Timestamp("2026-11-01")], "Poids (Kgs)": [80.0]})
    assert has_unsaved_changes(november_date, november_ts) is False
    reindexed = saved.copy(); reindexed.index = [10, 11]
    assert has_unsaved_changes(reindexed, saved) is False
    reordered = saved.iloc[::-1].reset_index(drop=True)
    assert has_unsaved_changes(reordered, saved) is True
    custom_saved = saved.assign(Note=["a", "b"])
    custom_changed = custom_saved.copy(); custom_changed.loc[1, "Note"] = "c"
    assert has_unsaved_changes(custom_changed, custom_saved) is True
    different_date = saved.copy(); different_date.loc[1, "Date"] = pd.Timestamp("2026-01-03")
    assert has_unsaved_changes(different_date, saved) is True


def _whoop_daily_frame(days: int = 14) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": [55 + (i % 6) * 5 for i in range(days)],
            "HRV (ms)": [40 + (i % 5) * 2.5 for i in range(days)],
            "FC repos (bpm)": [54 + (i % 3) for i in range(days)],
            "Sommeil (heures)": [6.8 + (i % 4) * 0.3 for i in range(days)],
            "Performance sommeil (%)": [80 + (i % 5) * 3 for i in range(days)],
            "Sommeil profond (heures)": [1.2 + (i % 3) * 0.2 for i in range(days)],
            "Sommeil REM (heures)": [1.6 + (i % 4) * 0.15 for i in range(days)],
            "Strain": [9 + (i % 7) for i in range(days)],
            "Calories (kcal)": [2400 + (i % 6) * 90 for i in range(days)],
        }
    )


def _whoop_connected_state(at: AppTest) -> None:
    _state(at)
    at.session_state["whoop_token"] = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_at": (pd.Timestamp.utcnow() + pd.Timedelta(hours=2)).isoformat(),
        "scopes": ["offline", "read:recovery"],
        "token_type": "Bearer",
    }
    at.session_state["whoop_daily"] = _whoop_daily_frame()
    at.session_state["whoop_workouts"] = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-07")],
            "Sport": ["Running", "Weightlifting"],
            "Durée (min)": [45.0, 60.0],
            "Strain séance": [9.4, 7.2],
            "Calories séance (kcal)": [520.0, 430.0],
            "FC moyenne (bpm)": [142.0, 118.0],
            "FC max (bpm)": [178.0, 150.0],
            "Distance (km)": [8.2, float("nan")],
        }
    )
    at.session_state["whoop_profile"] = {"first_name": "Test", "last_name": "Utilisateur"}
    at.session_state["whoop_last_sync"] = pd.Timestamp("2026-01-14")


def test_whoop_page_renders_connection_panel_without_credentials():
    at = AppTest.from_file("app/pages/Whoop.py")
    _state(at)
    at.run(timeout=15)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "WHOOP" in rendered
    assert "Connexion WHOOP" in rendered
    # Sans identifiants, aucune donnée WHOOP n'est inventée.
    assert at.session_state["whoop_daily"].empty


def test_whoop_page_renders_tabs_and_charts_when_session_holds_data():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    at.run(timeout=20)

    assert not at.exception
    tab_labels = [t.label for t in at.tabs]
    for expected in ["Vue d'ensemble", "Récupération", "Sommeil", "Effort", "Poids × WHOOP"]:
        assert expected in tab_labels
    assert len(at.get("plotly_chart")) >= 3
    assert len(at.metric) >= 3


def test_whoop_page_crosses_weight_and_whoop_without_touching_weight_data():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    before = at.session_state["working_data"].copy()
    at.run(timeout=20)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption)
    assert "Corrélations" in rendered
    # Non-régression : l'onglet WHOOP est en lecture seule sur les données de poids.
    pd.testing.assert_frame_equal(at.session_state["working_data"], before)
    assert len(at.session_state["working_data"]) == 80


def test_whoop_page_handles_connected_account_without_synced_data():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    at.session_state["whoop_daily"] = pd.DataFrame()
    at.session_state["whoop_workouts"] = pd.DataFrame()
    at.run(timeout=15)

    assert not at.exception
    assert any("Lancez une synchronisation" in str(info.value) for info in at.info)


def test_main_navigation_exposes_whoop_page_without_dropping_existing_pages():
    source = Path("Suivi_V1.py").read_text(encoding="utf-8")
    for page in ["Dashboard.py", "Journal.py", "Predictions.py", "Insights.py", "Settings.py", "Whoop.py"]:
        assert f"app/pages/{page}" in source
    assert 'title="Whoop"' in source
