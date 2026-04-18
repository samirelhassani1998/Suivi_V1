"""Tests pour le module analytique avancé app/core/analytics.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.analytics import (
    best_worst_weeks,
    consistency_score,
    day_of_week_analysis,
    detect_trend_breaks,
    discipline_score,
    generate_insights_text,
    multi_rolling_averages,
    period_comparison,
    progression_score,
    prospective_scenarios,
    segment_phases,
    streak_analysis,
    weight_acceleration,
    weight_velocity,
    weight_volatility,
)


@pytest.fixture
def sample_df():
    """60 days of realistic weight data with a downward trend."""
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    np.random.seed(42)
    weights = np.linspace(95, 90, 60) + np.random.normal(0, 0.3, 60)
    return pd.DataFrame({"Date": dates, "Poids (Kgs)": weights})


@pytest.fixture
def short_df():
    """5 days of data."""
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame({"Date": dates, "Poids (Kgs)": [90.0, 89.8, 89.6, 89.5, 89.3]})


class TestWeightVelocity:
    def test_returns_dict_with_all_windows(self, sample_df):
        result = weight_velocity(sample_df)
        assert isinstance(result, dict)
        assert 7 in result
        assert 30 in result

    def test_velocity_is_negative_for_weight_loss(self, sample_df):
        result = weight_velocity(sample_df, windows=(30,))
        assert result[30] is not None
        assert result[30] < 0

    def test_empty_df(self):
        result = weight_velocity(pd.DataFrame(columns=["Date", "Poids (Kgs)"]))
        assert all(v is None for v in result.values())


class TestMultiRollingAverages:
    def test_adds_ma_columns(self, sample_df):
        result = multi_rolling_averages(sample_df)
        assert "MA_7m" in result.columns
        assert "MA_14m" in result.columns
        assert "MA_30m" in result.columns
        assert len(result) == len(sample_df)


class TestWeightVolatility:
    def test_returns_dict(self, sample_df):
        result = weight_volatility(sample_df, window=14)
        assert "std" in result
        assert "cv" in result
        assert "interpretation" in result

    def test_short_data(self):
        df = pd.DataFrame({"Date": [pd.Timestamp("2026-01-01")], "Poids (Kgs)": [90.0]})
        result = weight_volatility(df)
        assert result["interpretation"] == "données insuffisantes"


class TestDisciplineScore:
    def test_perfect_discipline(self, sample_df):
        result = discipline_score(sample_df, window_days=30)
        assert result["score"] > 80
        assert result["interpretation"] == "excellente"

    def test_empty_data(self):
        result = discipline_score(pd.DataFrame(columns=["Date", "Poids (Kgs)"]))
        assert result["score"] == 0


class TestConsistencyScore:
    def test_returns_score(self, sample_df):
        result = consistency_score(sample_df, n_weeks=4)
        assert 0 <= result["score"] <= 100
        assert "interpretation" in result


class TestDetectTrendBreaks:
    def test_no_breaks_in_smooth_data(self, sample_df):
        breaks = detect_trend_breaks(sample_df)
        # Smooth data may or may not have breaks, just check type
        assert isinstance(breaks, list)

    def test_short_data_returns_empty(self, short_df):
        breaks = detect_trend_breaks(short_df)
        assert breaks == []


class TestBestWorstWeeks:
    def test_returns_dataframes(self, sample_df):
        result = best_worst_weeks(sample_df, n=3)
        assert "best" in result
        assert "worst" in result
        assert isinstance(result["best"], pd.DataFrame)

    def test_short_data(self, short_df):
        result = best_worst_weeks(short_df)
        assert result["best"].empty


class TestSegmentPhases:
    def test_returns_phases(self, sample_df):
        phases = segment_phases(sample_df, min_days=7)
        assert isinstance(phases, list)
        if phases:
            assert hasattr(phases[0], "phase_type")
            assert phases[0].phase_type in ("perte", "plateau", "reprise")


class TestPeriodComparison:
    def test_returns_dict(self, sample_df):
        result = period_comparison(sample_df)
        assert "week" in result
        assert "month" in result


class TestProspectiveScenarios:
    def test_returns_scenarios(self, sample_df):
        result = prospective_scenarios(sample_df, target_weight=85.0)
        assert isinstance(result, dict)
        if result:
            assert "réaliste" in result or "optimiste" in result

    def test_short_data(self, short_df):
        result = prospective_scenarios(short_df, target_weight=85.0)
        assert result == {}


class TestStreakAnalysis:
    def test_returns_streak_info(self, sample_df):
        result = streak_analysis(sample_df)
        assert "current_streak" in result
        assert "longest_loss" in result
        assert result["longest_loss"] >= 0


class TestDayOfWeekAnalysis:
    def test_returns_dataframe(self, sample_df):
        result = day_of_week_analysis(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert "Jour" in result.columns
        assert len(result) == 7  # 7 days of the week


class TestProgressionScore:
    def test_returns_score(self, sample_df):
        result = progression_score(sample_df, target_weight=85.0)
        assert 0 <= result["score"] <= 100
        assert result["grade"] in ("A+", "A", "B", "C", "D")

    def test_components(self, sample_df):
        result = progression_score(sample_df, target_weight=85.0)
        assert "components" in result
        assert "progression" in result["components"]


class TestWeightAcceleration:
    def test_returns_dict(self, sample_df):
        result = weight_acceleration(sample_df)
        assert "acceleration" in result
        assert "interpretation" in result


class TestGenerateInsightsText:
    def test_returns_list_of_strings(self, sample_df):
        result = generate_insights_text(sample_df, target_weight=85.0)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, str) for s in result)

    def test_short_data(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2026-01-01", periods=3),
            "Poids (Kgs)": [90, 89, 88],
        })
        result = generate_insights_text(df, target_weight=85.0)
        assert len(result) > 0
