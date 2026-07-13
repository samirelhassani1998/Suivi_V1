import math

import pandas as pd
import pytest

from app.core.target_trajectory import (
    DEFAULT_TARGET_TRAJECTORY_END_DATE,
    DEFAULT_TARGET_TRAJECTORY_START_DATE,
    TargetTrajectoryConfig,
    build_target_trajectory,
    compare_to_target_trajectory,
    required_daily_loss,
    required_weekly_loss,
    target_weight_on_date,
)


def test_target_trajectory_exact_fixed_contract():
    result = build_target_trajectory(pd.DataFrame())
    trajectory = result["trajectory"]
    assert trajectory["Date"].iloc[0] == pd.Timestamp("2026-07-12")
    assert trajectory["Poids cible (kg)"].iloc[0] == 106.1
    assert trajectory["Date"].iloc[-1] == pd.Timestamp("2026-11-11")
    assert trajectory["Poids cible (kg)"].iloc[-1] == 80.0
    assert len(trajectory) == 123
    assert result["start_date"] == DEFAULT_TARGET_TRAJECTORY_START_DATE
    assert result["end_date"] == DEFAULT_TARGET_TRAJECTORY_END_DATE
    assert result["total_duration_days"] == 122


def test_required_weekly_loss():
    expected = (106.1 - 80.0) / (122 / 7)
    assert required_weekly_loss() == pytest.approx(expected, abs=1e-12)
    assert TargetTrajectoryConfig().required_weekly_loss == pytest.approx(expected, abs=1e-12)


def test_required_daily_loss():
    assert required_daily_loss() == pytest.approx((106.1 - 80.0) / 122, abs=1e-12)


@pytest.mark.parametrize("data_weight", [90.0, 100.0, 110.0])
def test_source_data_does_not_change_target_anchor(data_weight):
    df = pd.DataFrame({"Date": [pd.Timestamp("2026-07-12")], "Poids (Kgs)": [data_weight]})
    result = build_target_trajectory(df)
    assert result["start_weight"] == 106.1
    assert result["start_weight_source"] == "fixed_business_rule"
    assert result["trajectory"]["Poids cible (kg)"].iloc[0] == 106.1


def test_target_weight_on_date_fixed_bounds():
    assert target_weight_on_date(pd.Timestamp("2026-07-11")) is None
    assert target_weight_on_date(pd.Timestamp("2026-07-12")) == 106.1
    mid = target_weight_on_date(pd.Timestamp("2026-09-11"))
    expected = 106.1 + (61 / 122) * (80.0 - 106.1)
    assert mid == pytest.approx(expected, abs=1e-9)
    assert target_weight_on_date(pd.Timestamp("2026-11-11")) == 80.0
    assert target_weight_on_date(pd.Timestamp("2026-11-12")) is None


def test_trajectory_strictly_decreasing_and_never_below_floor():
    trajectory = build_target_trajectory(pd.DataFrame())["trajectory"]
    values = trajectory["Poids cible (kg)"]
    assert values.is_monotonic_decreasing
    assert (values.diff().dropna() < 0).all()
    assert (values >= 80.0).all()
    assert (values == 80.0).sum() == 1
    assert trajectory["Date"].max() == pd.Timestamp("2026-11-11")


def test_compare_to_target_trajectory_reports_gap_status_and_progress():
    scheduled = target_weight_on_date(pd.Timestamp("2026-09-11"))
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-07-12", "2026-09-11"]), "Poids (Kgs)": [106.1, scheduled - 1.0]})
    result = compare_to_target_trajectory(df)
    assert result["scheduled_weight"] == pytest.approx(scheduled, abs=1e-9)
    assert result["gap_kg"] == pytest.approx(-1.0, abs=1e-9)
    assert result["status"] == "en avance"
    assert result["days_delta"] == pytest.approx(-1.0 / required_daily_loss(), abs=1e-9)
    assert math.isclose(result["progress_pct"], (106.1 - (scheduled - 1.0)) / 26.1 * 100)


@pytest.mark.parametrize(("offset_kg", "expected_status"), [(-0.40, "en avance"), (-0.30, "aligné"), (0.00, "aligné"), (0.30, "aligné"), (0.40, "en retard")])
def test_alignment_status_uses_configurable_tolerance(offset_kg, expected_status):
    comparison_date = pd.Timestamp("2026-07-13")
    scheduled = target_weight_on_date(comparison_date)
    df = pd.DataFrame({"Date": [pd.Timestamp("2026-07-12"), comparison_date], "Poids (Kgs)": [106.1, scheduled + offset_kg]})
    result = compare_to_target_trajectory(df)
    assert result["gap_kg"] == pytest.approx(offset_kg, abs=1e-9)
    assert result["status"] == expected_status


def test_post_end_measurement_marks_trajectory_completed_without_exception():
    df = pd.DataFrame({"Date": [pd.Timestamp("2026-11-12")], "Poids (Kgs)": [79.5]})
    result = compare_to_target_trajectory(df)
    assert result["available"] is False
    assert result["trajectory_completed"] is True
    assert "terminée" in result["message"]
    assert len(result["trajectory"]) == 123
