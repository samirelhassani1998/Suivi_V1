from __future__ import annotations

import json

import pandas as pd

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


def test_dashboard_renders_kpis_and_sections():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    markdown_values = [str(m.value) for m in at.markdown]
    assert any("Dashboard" in value for value in markdown_values)
    assert len(at.metric) >= 3
    plotly_elements = at.get("plotly_chart")
    assert len(plotly_elements) >= 1
    trace_names = []
    for chart in plotly_elements:
        spec = json.loads(chart.proto.spec)
        trace_names.extend(trace.get("name", "") for trace in spec.get("data", []))
    normalized_trace_names = " ".join(trace_names).lower()
    assert "poids" in normalized_trace_names
    assert "objectif" in normalized_trace_names or "cible" in normalized_trace_names
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
    assert "Estimation de date objectif" in rendered_text
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
    typed = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "Poids (Kgs)": ["80,0", "79,8"]})
    assert has_unsaved_changes(typed, saved) is False
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
