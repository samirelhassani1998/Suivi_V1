from __future__ import annotations

import pandas as pd

from app.core.weight_summary import (
    delta_since_days,
    generate_daily_insights,
    projection_to_target,
    summarize_weight_journey,
)


def test_weight_summary_handles_invalid_rows_and_core_metrics():
    df = pd.DataFrame(
        {
            "Date": ["01/01/2026", "bad date", "08/01/2026", "31/01/2026"],
            "Poids (Kgs)": ["90,0", "oops", 89.0, 87.0],
        }
    )
    summary = summarize_weight_journey(df, target_weight=85.0)

    assert summary["valid"] is True
    assert summary["current"] == 87.0
    assert summary["delta_start"] == -3.0
    assert summary["min_weight"] == 87.0
    assert summary["target_gap"] == 2.0


def test_delta_since_days_is_prudent_with_short_history():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "Poids (Kgs)": [90.0, 89.7],
        }
    )

    delta = delta_since_days(df, 30)

    assert delta.value is None
    assert delta.reason is not None


def test_projection_requires_clear_recent_trend():
    stable = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=10),
            "Poids (Kgs)": [90.0, 90.1, 90.0, 90.05, 90.0, 90.05, 90.0, 90.05, 90.0, 90.05],
        }
    )
    falling = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=20),
            "Poids (Kgs)": [95 - i * 0.1 for i in range(20)],
        }
    )

    assert projection_to_target(stable, 85.0)["available"] is False
    assert projection_to_target(falling, 90.0)["available"] is True


def test_daily_insights_are_simple_strings():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=35),
            "Poids (Kgs)": [95 - i * 0.05 for i in range(35)],
        }
    )

    insights = generate_daily_insights(df, target_weight=90.0)

    assert insights
    assert all(isinstance(item, str) for item in insights)
    assert any("30" in item for item in insights)
