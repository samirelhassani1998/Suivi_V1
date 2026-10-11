"""Regression cases from the calculation audit: calendar, gaps and honest ETAs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.analytics import (
    consistency_score, generate_action_summary, generate_insights_text,
    next_milestone, pace_comparison, prospective_scenarios, streak_analysis, weight_velocity,
)
from app.core.insights import detect_plateau, estimate_target_eta
from app.core.plateau import prepare_plateau_series
from app.core.projection_constraints import constrain_interval_dataframe, truncate_projection_at_floor
from app.core.target_trajectory import target_weight_on_date
from app.core.trend import noise_level, rate_of_change, trend_weight
from app.core.weight_summary import (
    generate_daily_insights, prepare_weight_series, projection_to_target,
)


def _weights(dates, weights):
    return pd.DataFrame({"Date": pd.to_datetime(dates, format="mixed"), "Poids (Kgs)": weights})


@pytest.mark.parametrize("prepare", [prepare_weight_series, prepare_plateau_series])
def test_weight_analytics_exclude_nonfinite_nonpositive_and_duplicate_days(prepare):
    frame = _weights(
        ["2026-01-01 18:00", "2026-01-01 08:00", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        [99, 100, np.inf, -np.inf, 0, -1],
    )
    result = prepare(frame)
    assert list(result.columns) == ["Date", "Poids (Kgs)"]
    assert result["Date"].tolist() == [pd.Timestamp("2026-01-01")]
    assert result["Poids (Kgs)"].tolist() == [99]
    assert len(frame) == 6


@pytest.mark.parametrize("prepare", [prepare_weight_series, prepare_plateau_series])
def test_weight_dates_remain_local_calendar_days(prepare):
    frame = _weights(pd.date_range("2026-10-24", periods=4, tz="Europe/Paris"), [100, 99, 98, 97])
    assert prepare(frame)["Date"].tolist() == list(pd.date_range("2026-10-24", periods=4))


def test_iso_week_remains_one_week_across_new_year():
    frame = _weights(pd.date_range("2020-12-28", periods=7), [100] * 4 + [101] * 3)
    result = consistency_score(frame, n_weeks=1)
    assert result["avg_weekly_std"] == round(frame["Poids (Kgs)"].std(), 3)
    assert result["score"] < 100


def test_recent_velocity_does_not_use_a_measurement_before_the_window():
    frame = _weights(["2026-01-01", "2026-05-01", "2026-05-05"], [110, 100, 101])
    assert weight_velocity(frame, windows=(7,))[7] == pytest.approx(1.75)
    assert weight_velocity(frame.iloc[[0, 2]], windows=(7,))[7] is None


def test_stable_latest_measurement_ends_the_current_streak():
    frame = _weights(pd.date_range("2026-01-01", periods=4), [100, 99, 98, 98])
    result = streak_analysis(frame)
    assert result["current_streak"] == 0
    assert result["current_type"] == "neutre"
    assert result["longest_loss"] == 2


def test_one_measurement_after_restart_does_not_inherit_old_loss_or_eta():
    frame = _weights(
        pd.date_range("2026-01-01", periods=8).append(pd.DatetimeIndex(["2026-05-01"])),
        [100 - i * .1 for i in range(8)] + [95],
    )
    assert projection_to_target(frame, 90)["available"] is False
    assert estimate_target_eta(frame, 90)["credible"] is False
    summary = generate_action_summary(frame, 80)
    assert "01/05/2026" in summary["situation"]
    assert "perdu" not in summary["situation"]
    assert "une mesure" in summary["situation"]
    assert not any("kg/sem" in text for text in generate_insights_text(frame, 80))


def test_old_plateau_is_not_described_as_current_after_restart():
    old = _weights(pd.date_range("2026-01-01", periods=15), [100] * 15)
    recent = _weights(pd.date_range("2026-05-01", periods=2), [99, 99])
    insights = generate_daily_insights(pd.concat([old, recent]), 80)
    assert not any("Stabilité possible" in text for text in insights)


def test_multiple_readings_on_two_days_cannot_establish_a_plateau():
    frame = _weights(["2026-01-01", "2026-01-01", "2026-01-14", "2026-01-14"], [100] * 4)
    result = detect_plateau(frame)
    assert result["status"] == "indisponible"
    assert result["nb_mesures"] == 2


def test_next_milestone_never_suggests_losing_more_below_final_goal():
    result = next_milestone(79, (100, 95, 90, 85, 80), velocity=-.5, measurements=30)
    assert result["target"] == 80
    assert result["remaining"] == 0
    assert result["reached"] is True
    assert result["eta_days"] is None


def test_elapsed_target_deadline_is_not_replaced_by_tomorrow():
    frame = _weights(pd.date_range("2026-01-01", periods=10), 100 - np.arange(10) * .1)
    result = pace_comparison(frame, 80, pd.Timestamp("2026-01-05"))
    assert result["required_pace"] is None
    assert result["days_left"] == -5


def test_eta_windows_do_not_fabricate_confidence_or_future_pace():
    frame = _weights(pd.date_range("2026-01-01", periods=40), 100 - np.arange(40) * .1)
    result = estimate_target_eta(frame, 90)
    assert result["credible"] is True
    assert set(result["scenarios"]) == {"7 jours", "30 jours", "90 jours"}
    assert not {"confidence", "eta_min", "eta_max"}.intersection(result)
    fast = frame.assign(**{"Poids (Kgs)": 110 - np.arange(40) * .5})
    result = estimate_target_eta(fast, 80)
    assert result["credible"] is False
    assert "garde-fou" in result["message"]
    assert all(not scenario["credible"] for scenario in result["scenarios"].values())


def test_projection_does_not_round_down_before_the_target_is_reached():
    frame = _weights(pd.date_range("2026-01-01", periods=10), 100 - np.arange(10) * .1)
    result = projection_to_target(frame, 98.85)
    assert result["days_needed"] == 3  # 2.5 calendar days, not 2


def test_smoothing_and_signal_readiness_restart_after_a_tracking_gap():
    dates = pd.date_range("2026-01-01", periods=6).append(pd.date_range("2026-01-29", periods=2))
    frame = _weights(dates, [100, 99, 98, 97, 96, 95, 110, 109])
    result = trend_weight(frame)
    independent = trend_weight(frame.tail(2))
    np.testing.assert_allclose(result["Tendance"].tail(2), independent["Tendance"])
    assert rate_of_change(frame, 30)["ready"] is False
    assert noise_level(frame)["ready"] is False
    assert projection_to_target(frame, 80)["available"] is False


def test_projection_constraints_preserve_dst_calendar_dates_and_fractional_crossing():
    dates = pd.date_range("2026-10-24", periods=4, tz="Europe/Paris")
    raw = pd.DataFrame({"Date": dates, "prevision": [83, 82, 81, 79],
                        "borne_basse": [82, 81, 80, 78], "borne_haute": [84, 83, 82, 80]})
    out = constrain_interval_dataframe(raw)
    assert out["Date"].iloc[0] == pd.Timestamp("2026-10-24")
    assert out["Date"].iloc[-1] == pd.Timestamp("2026-10-26 12:00")
    repeated = truncate_projection_at_floor(out["Date"], out["prevision"])
    assert repeated.stop_date == pd.Timestamp("2026-10-26 12:00")
    assert (out["borne_basse"] <= out["prevision"]).all()
    assert (out["borne_haute"] >= out["prevision"]).all()


def test_fixed_target_uses_the_selected_local_date():
    assert target_weight_on_date(pd.Timestamp("2026-10-01", tz="Europe/Paris")) == 106.0


def test_progress_score_does_not_count_an_old_reached_target_as_still_reached():
    from app.core.analytics import progression_score
    frame = _weights(pd.date_range("2026-01-01", periods=8), [79, 79.5, 80, 80.5, 81, 81.5, 82, 82.5])
    assert progression_score(frame, 80)["components"]["progression"] == 0


def test_prospective_scenarios_need_enough_measurements_since_the_restart():
    old = _weights(pd.date_range("2026-01-01", periods=14), 100 - np.arange(14) * .1)
    restart = _weights(pd.date_range("2026-02-08", periods=14), 95 - np.arange(14) * .1)
    assert prospective_scenarios(pd.concat([old, restart.head(1)]), 80) == {}
    assert prospective_scenarios(pd.concat([old, restart.head(13)]), 80) == {}
    actual = prospective_scenarios(pd.concat([old, restart]), 80)
    expected = prospective_scenarios(restart, 80)
    assert actual == expected
    assert len(actual) == 3
