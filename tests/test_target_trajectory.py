import math

import pandas as pd
import pytest

from app.core.target_trajectory import (
    DEFAULT_FINAL_TARGET_WEIGHT,
    DEFAULT_TARGET_TRAJECTORY_END_DATE,
    DEFAULT_TARGET_TRAJECTORY_START_DATE,
    TargetTrajectoryConfig,
    build_target_trajectory,
    compare_to_target_trajectory,
    required_weekly_loss,
    target_weight_on_date,
)


def _df(weights=(102.0,)):
    return pd.DataFrame({"Date": [pd.Timestamp("2026-07-11") for _ in weights], "Poids (Kgs)": list(weights)})


def test_target_trajectory_exact_bounds_floor_and_no_horizontal_tail():
    result = build_target_trajectory(_df())
    trajectory = result["trajectory"]
    assert trajectory["Date"].iloc[0] == pd.Timestamp("2026-07-11")
    assert trajectory["Date"].iloc[-1] == pd.Timestamp("2026-11-11")
    assert trajectory["Poids cible (kg)"].iloc[-1] == 80.0
    assert len(trajectory) == 124
    assert (trajectory["Poids cible (kg)"] >= 80.0).all()
    assert (trajectory["Poids cible (kg)"] == 80.0).sum() == 1
    assert trajectory["Date"].max() == pd.Timestamp("2026-11-11")
    assert result["start_date"] == DEFAULT_TARGET_TRAJECTORY_START_DATE
    assert result["end_date"] == DEFAULT_TARGET_TRAJECTORY_END_DATE


def test_required_weekly_loss_is_derived_from_start_weight_and_duration():
    start_weight = 102.0
    result = build_target_trajectory(_df([start_weight]))
    expected_rate = (start_weight - 80.0) / (123 / 7)
    assert result["required_weekly_loss"] == pytest.approx(expected_rate, abs=1e-9)
    assert required_weekly_loss(start_weight) == pytest.approx(expected_rate, abs=1e-9)


def test_target_weight_on_date_uses_exact_time_proportion_and_stops_outside_bounds():
    start = pd.Timestamp("2026-07-11")
    end = pd.Timestamp("2026-11-11")
    assert target_weight_on_date(pd.Timestamp("2026-07-10"), start, end, 102.0, 80.0) is None
    assert target_weight_on_date(start, start, end, 102.0, 80.0) == 102.0
    mid = target_weight_on_date(pd.Timestamp("2026-09-11"), start, end, 102.0, 80.0)
    expected = 102.0 + (62 * 86400 / (123 * 86400)) * (80.0 - 102.0)
    assert mid == pytest.approx(expected, abs=1e-9)
    assert target_weight_on_date(end, start, end, 102.0, 80.0) == 80.0
    assert target_weight_on_date(pd.Timestamp("2026-11-12"), start, end, 102.0, 80.0) is None


def test_build_target_trajectory_uses_previous_measurement_when_start_date_missing():
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-07-01", "2026-07-12"]), "Poids (Kgs)": [103.0, 101.0]})
    result = build_target_trajectory(df)
    assert result["start_date"] == pd.Timestamp("2026-07-11")
    assert result["start_measurement_date"] == pd.Timestamp("2026-07-01")
    assert result["start_weight"] == 103.0
    assert result["start_weight_source"] == "previous"
    assert result["trajectory"]["Date"].iloc[0] == pd.Timestamp("2026-07-11")


def test_multiple_start_measurements_respect_duplicate_strategy_without_mutating_source():
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-07-11", "2026-07-11"]), "Poids (Kgs)": [102.4, 101.8], "Moment": ["matin", "soir"]})
    before = df.copy(deep=True)
    result = build_target_trajectory(df, TargetTrajectoryConfig(duplicate_strategy="garder_la_derniere"))
    assert result["start_weight"] == 101.8
    pd.testing.assert_frame_equal(df, before)


def test_compare_to_target_trajectory_reports_gap_status_and_progress():
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-07-11", "2026-09-11"]), "Poids (Kgs)": [102.0, 90.0]})
    result = compare_to_target_trajectory(df)
    expected_scheduled = 102.0 + (62 / 123) * (80.0 - 102.0)
    assert result["scheduled_weight"] == pytest.approx(expected_scheduled, abs=1e-9)
    assert result["gap_kg"] == pytest.approx(90.0 - expected_scheduled, abs=1e-9)
    assert result["status"] == "en avance"
    assert math.isclose(result["progress_pct"], (102.0 - 90.0) / 22.0 * 100)


@pytest.mark.parametrize(
    ("offset_kg", "expected_status"),
    [
        (-0.40, "en avance"),
        (-0.30, "aligné"),
        (-0.20, "aligné"),
        (0.00, "aligné"),
        (0.20, "aligné"),
        (0.30, "aligné"),
        (0.40, "en retard"),
    ],
)
def test_alignment_status_uses_configurable_tolerance(offset_kg, expected_status):
    # La tolérance de ±0,30 kg est inclusive: les bornes exactes restent alignées.
    start_date = pd.Timestamp("2026-07-11")
    comparison_date = pd.Timestamp("2026-07-12")
    start_weight = 102.0

    scheduled = target_weight_on_date(
        comparison_date,
        start_date,
        pd.Timestamp("2026-11-11"),
        start_weight,
        80.0,
    )

    df = pd.DataFrame(
        {
            "Date": [
                start_date,
                comparison_date,
            ],
            "Poids (Kgs)": [
                start_weight,
                scheduled + offset_kg,
            ],
        }
    )

    result = compare_to_target_trajectory(df)

    assert result["gap_kg"] == pytest.approx(offset_kg, abs=1e-9)
    assert result["status"] == expected_status
