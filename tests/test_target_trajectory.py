import math

import pandas as pd

from app.core.target_trajectory import (
    DEFAULT_FINAL_TARGET_WEIGHT,
    DEFAULT_TARGET_TRAJECTORY_START_DATE,
    DEFAULT_TARGET_TRAJECTORY_START_WEIGHT,
    DEFAULT_WEEKLY_LOSS_TARGET,
    TargetTrajectoryConfig,
    build_target_trajectory,
    compare_to_target_trajectory,
)


def test_target_trajectory_starts_on_fixed_reference_point_and_stops_at_80():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01", "2026-06-01", "2026-06-10"]),
            "Poids (Kgs)": [120.0, 105.0, 102.0],
        }
    )

    result = build_target_trajectory(df)
    trajectory = result["trajectory"]

    assert result["available"] is True
    assert result["start_date"] == DEFAULT_TARGET_TRAJECTORY_START_DATE
    assert result["start_measurement_date"] == DEFAULT_TARGET_TRAJECTORY_START_DATE
    assert result["start_weight"] == DEFAULT_TARGET_TRAJECTORY_START_WEIGHT
    assert trajectory["Date"].iloc[0] == DEFAULT_TARGET_TRAJECTORY_START_DATE
    assert trajectory["Poids cible (kg)"].iloc[0] == DEFAULT_TARGET_TRAJECTORY_START_WEIGHT
    assert trajectory["Poids cible (kg)"].min() == DEFAULT_FINAL_TARGET_WEIGHT
    assert trajectory["Poids cible (kg)"].iloc[-1] == DEFAULT_FINAL_TARGET_WEIGHT
    assert math.isclose(result["target_days"], 91.7)
    assert result["eta_date"] == DEFAULT_TARGET_TRAJECTORY_START_DATE + pd.Timedelta(days=91.7)
    assert trajectory["Date"].iloc[-1] == result["eta_date"]


def test_target_trajectory_does_not_infer_start_from_nearby_measurements():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-29", "2026-06-04"]),
            "Poids (Kgs)": [106.0, 104.0],
        }
    )

    result = build_target_trajectory(df)

    assert result["start_date"] == pd.Timestamp("2026-05-26")
    assert result["start_measurement_date"] == pd.Timestamp("2026-05-26")
    assert result["start_weight"] == 106.2


def test_compare_to_target_trajectory_reports_gap_status_and_progress():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-26", "2026-06-02"]),
            "Poids (Kgs)": [106.2, 105.0],
        }
    )

    result = compare_to_target_trajectory(df)

    assert result["scheduled_weight"] == 104.2
    assert math.isclose(result["gap_kg"], 0.8)
    assert result["status"] == "retard"
    assert math.isclose(result["progress_pct"], (106.2 - 105.0) / 26.2 * 100)


def test_target_trajectory_supports_explicit_business_parameters():
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-07-01"]), "Poids (Kgs)": [90.0]})
    config = TargetTrajectoryConfig.from_values(
        "2026-07-01",
        weekly_loss_target=1.0,
        final_target_weight=85.0,
        start_weight=90.0,
    )

    result = build_target_trajectory(df, config)

    assert result["start_date"] == pd.Timestamp("2026-07-01")
    assert result["start_weight"] == 90.0
    assert result["weekly_loss_target"] == 1.0
    assert result["final_target_weight"] == 85.0
    assert math.isclose(result["target_days"], 35.0)
    assert result["eta_date"] == pd.Timestamp("2026-08-05")
    assert DEFAULT_WEEKLY_LOSS_TARGET == 2.0
