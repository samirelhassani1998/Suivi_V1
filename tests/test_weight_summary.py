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


def test_summarize_weight_journey_generates_insights_without_nameerror():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-05-20", periods=10, freq="D"),
            "Poids (Kgs)": [106.2, 106.0, 105.8, 105.5, 105.2, 105.0, 104.8, 104.5, 104.2, 104.0],
        }
    )

    result = summarize_weight_journey(df, 80.0)

    assert result["valid"] is True
    assert "insights" in result
    assert isinstance(result["insights"], list)
    assert len(result["insights"]) >= 1
    assert all(isinstance(item, str) for item in result["insights"])


def test_summarize_weight_journey_empty_dataframe_is_invalid_without_insights():
    result = summarize_weight_journey(pd.DataFrame(columns=["Date", "Poids (Kgs)"]), 80.0)

    assert result["valid"] is False


def test_summarize_weight_journey_single_measure_generates_limited_history_insight():
    df = pd.DataFrame({"Date": [pd.Timestamp("2026-05-20")], "Poids (Kgs)": [106.2]})

    result = summarize_weight_journey(df, 80.0)

    assert result["valid"] is True
    assert result["insights"] == ["Ajoutez au moins une deuxième mesure pour calculer les variations."]


def test_summarize_weight_journey_two_measures_generates_prudent_insights():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-20", "2026-05-27"]),
            "Poids (Kgs)": [106.2, 105.8],
        }
    )

    result = summarize_weight_journey(df, 80.0)

    assert result["valid"] is True
    assert result["insights"]
    assert any("7 jours" in item for item in result["insights"])


def test_summarize_weight_journey_sorts_unsorted_data_before_insights():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-29", "2026-05-20", "2026-05-28"]),
            "Poids (Kgs)": [104.0, 106.2, 104.2],
        }
    )

    result = summarize_weight_journey(df, 80.0)

    assert result["valid"] is True
    assert result["last_date"] == pd.Timestamp("2026-05-29")
    assert result["current"] == 104.0
    assert result["insights"]


def test_summarize_weight_journey_accepts_comma_weight_values():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-05-20", periods=2, freq="D"),
            "Poids (Kgs)": ["106,2", "105,8"],
        }
    )

    result = summarize_weight_journey(df, 80.0)

    assert result["valid"] is True
    assert result["current"] == 105.8
    assert result["insights"]


def test_summarize_weight_journey_reports_detected_stagnation():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-05-01", periods=15, freq="D"),
            "Poids (Kgs)": [100.0, 100.1, 100.0, 100.2, 100.1, 100.0, 100.2, 100.1, 100.0, 100.2, 100.1, 100.0, 100.2, 100.1, 100.0],
        }
    )

    result = summarize_weight_journey(df, 80.0)

    assert result["stagnations"]
    assert any("Stabilité possible" in item for item in result["insights"])


def test_summarize_weight_journey_significant_drop_does_not_report_stagnation_insight():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-05-01", periods=15, freq="D"),
            "Poids (Kgs)": [106.0, 105.8, 105.6, 105.4, 105.2, 105.0, 104.8, 104.6, 104.4, 104.2, 104.0, 103.8, 103.6, 103.4, 103.2],
        }
    )

    result = summarize_weight_journey(df, 80.0)

    assert result["valid"] is True
    assert any("baisse" in item.lower() for item in result["insights"])
    assert not any("Stabilité possible" in item for item in result["insights"])
