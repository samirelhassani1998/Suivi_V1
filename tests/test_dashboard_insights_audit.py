"""Régressions des périodes, périmètres et projections de Dashboard/Insights."""
from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.core.analytics import discipline_score, segment_phases
from app.core.insights import detect_plateau
from app.core.trend import eta_to_target


def _weights(dates, weights) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Poids (Kgs)": weights})


def _page(name: str, data: pd.DataFrame) -> AppTest:
    at = AppTest.from_file(f"app/pages/{name}.py", default_timeout=120)
    for key in ("source_data", "working_data", "raw_data"):
        at.session_state[key] = data.copy()
    at.session_state["filtered_data"] = pd.DataFrame()
    at.session_state["filter_active"] = False
    at.session_state["target_weights"] = (100.0, 95.0, 90.0, 85.0, 80.0)
    return at.run()


def _chart(at: AppTest, title: str) -> dict:
    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    return next(spec for spec in specs if spec["layout"]["title"]["text"] == title)


def test_followup_coverage_uses_thirty_distinct_calendar_days():
    data = _weights(pd.date_range("2026-01-01", periods=31), np.repeat(100.0, 31))
    # Une seconde pesée le dernier jour ne doit pas augmenter la couverture.
    data = pd.concat([data, _weights(["2026-01-31 18:00"], [99.9])], ignore_index=True)
    result = discipline_score(data, window_days=30)
    assert result["measured_days"] == 30
    assert result["expected_days"] == 30
    assert result["rate"] == 100.0
    assert result["score"] == 100


def test_phases_preserve_interruptions_even_in_the_same_direction():
    dates = pd.date_range("2026-01-01", periods=14).append(pd.date_range("2026-04-01", periods=14))
    data = _weights(dates, np.r_[100 - np.arange(14) * 0.1, 95 - np.arange(14) * 0.1])
    phases = segment_phases(data)
    assert len(phases) == 2
    assert [phase.phase_type for phase in phases] == ["perte", "perte"]
    assert [(phase.start, phase.end) for phase in phases] == [(dates[0], dates[13]), (dates[14], dates[-1])]
    assert [phase.duration_days for phase in phases] == [14, 14]


def test_short_phases_are_not_joined_across_interruptions():
    dates = pd.DatetimeIndex([])
    for month in (1, 3, 5, 7):
        dates = dates.append(pd.date_range(f"2026-{month:02d}-01", periods=4))
    assert segment_phases(_weights(dates, 100 - np.arange(len(dates)) * 0.1)) == []


@pytest.mark.parametrize("window", [14, 30])
def test_four_recent_weigh_ins_cannot_establish_a_long_plateau(window):
    result = detect_plateau(_weights(pd.date_range("2026-01-01", periods=4), [100.0] * 4), window)
    assert result["status"] == "indisponible"
    assert result["is_plateau"] is False
    assert "recul" in result["reason"]


def test_plateau_requires_the_selected_calendar_span():
    data = _weights(pd.date_range("2026-01-01", periods=4, freq="4D"), [100.0] * 4)
    assert detect_plateau(data, 14)["status"] == "plateau probable"
    assert detect_plateau(data, 30)["status"] == "indisponible"
    longer = _weights(pd.date_range("2026-01-01", periods=5, freq="7D"), [100.0] * 5)
    assert detect_plateau(longer, 30)["status"] == "plateau probable"


def test_dashboard_does_not_overflow_on_a_tiny_significant_slope():
    data = _weights(pd.date_range("2026-10-01", periods=30), 100 - np.arange(30) * 0.000001)
    estimate = eta_to_target(data, 80.0)
    assert estimate["ready"] is False
    assert "trois ans" in estimate["reason"]
    at = _page("Dashboard", data)
    assert not at.exception
    assert any("trois ans" in message.value for message in at.info)


def test_insights_scope_applies_to_anomalies_and_fluctuations():
    dates = pd.date_range("2026-04-01", periods=28).append(pd.date_range("2026-09-01", periods=28))
    values = 106 - np.arange(56) * 0.1
    values[10] += 8.0
    data = _weights(dates, values)
    at = _page("Insights", data)
    assert not at.exception
    scope = next(radio for radio in at.radio if radio.label == "Périmètre d'analyse")
    assert scope.value == "Effort actuel"

    anomalies = _chart(at, "Pesées atypiques face à la tendance")
    observed_dates = [pd.Timestamp(x) for trace in anomalies["data"] for x in trace.get("x", [])]
    assert observed_dates
    assert min(observed_dates) >= pd.Timestamp("2026-09-01")
    # Sur le bloc récent parfaitement linéaire, les résidus sont nuls.
    # Les variations entre pesées seraient -0,1 kg : ce sont deux variables distinctes.
    histogram = _chart(at, "Écarts entre les pesées et la tendance")
    residuals = histogram["data"][0]["x"]
    if isinstance(residuals, dict):
        residuals = np.frombuffer(base64.b64decode(residuals["bdata"]), dtype=residuals["dtype"])
    assert len(residuals) == 28
    assert np.allclose(residuals, 0.0, atol=1e-7)

    scope.set_value("Historique complet").run()
    assert not at.exception
    anomalies = _chart(at, "Pesées atypiques face à la tendance")
    observed_dates = [pd.Timestamp(x) for trace in anomalies["data"] for x in trace.get("x", [])]
    assert min(observed_dates) == pd.Timestamp("2026-04-01")
