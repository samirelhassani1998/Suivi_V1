from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest


def _state(at: AppTest) -> None:
    at.session_state["filtered_data"] = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=20), "Poids (Kgs)": [90 - i * 0.1 for i in range(20)]})
    at.session_state["raw_data"] = at.session_state["filtered_data"].copy()


def test_dashboard_smoke():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.run()
    assert not at.exception


def test_journal_smoke():
    at = AppTest.from_file("app/pages/Journal.py")
    _state(at)
    at.run()
    assert not at.exception


def test_predictions_smoke():
    at = AppTest.from_file("app/pages/Predictions.py")
    _state(at)
    at.run()
    assert not at.exception


def test_insights_smoke():
    at = AppTest.from_file("app/pages/Insights.py")
    _state(at)
    at.run()
    assert not at.exception


def test_settings_smoke():
    at = AppTest.from_file("app/pages/Settings.py")
    _state(at)
    at.run()
    assert not at.exception
