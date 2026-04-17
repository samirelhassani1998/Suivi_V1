from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest


def _state(at: AppTest) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=80),
            "Poids (Kgs)": [95 - i * 0.08 for i in range(80)],
            "Extra Col": [f"v{i}" for i in range(80)],
        }
    )
    at.session_state["source_data"] = df.copy()
    at.session_state["working_data"] = df.copy()
    at.session_state["filtered_data"] = df.copy()
    at.session_state["raw_data"] = df.copy()
    at.session_state["target_weights"] = (93.0, 90.0, 87.0, 84.0)
    at.session_state["fast_mode"] = True


def test_dashboard_renders_kpis_and_sections():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    assert any("Dashboard" in h.value for h in at.title)
    assert len(at.metric) >= 3
    assert any("objectif" in str(c.value).lower() for c in at.caption)


def test_journal_loads_and_keeps_state_on_navigation():
    at = AppTest.from_file("app/pages/Journal.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    assert len(at.dataframe) >= 1
    button_labels = [b.label for b in at.button]
    assert "Enregistrer les modifications" in button_labels

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


def test_predictions_render_multiple_sections_even_if_submodel_fails():
    at = AppTest.from_file("app/pages/Predictions.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("Leaderboard" in s for s in subheaders)
    assert any("Estimation de date objectif" in s for s in subheaders)
    tab_labels = [t.label for t in at.tabs]
    assert any("SARIMA" in t for t in tab_labels)
    assert any("Auto-ARIMA" in t for t in tab_labels)


def test_settings_exposes_four_goals():
    at = AppTest.from_file("app/pages/Settings.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    labels = [n.label for n in at.number_input]
    assert "Objectif 1 (kg)" in labels
    assert "Objectif 2 (kg)" in labels
    assert "Objectif 3 (kg)" in labels
    assert "Objectif 4 (kg)" in labels
