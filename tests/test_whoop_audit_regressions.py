"""Regression cases discovered by auditing WHOOP API semantics and boxing totals."""
import numpy as np
import pandas as pd
import pytest

from app.core.whoop import HttpResponse, WhoopToken, mask_in_progress_cycle, recoveries_to_frame
from app.core.whoop_sync import fetch_whoop_data
from app.core.whoop_analytics import daily_grid, filter_period, training_energy_share
from app.core.boxing_analytics import session_table, weekly_sessions, todays_guidance
from app.ui.boxing_visuals import boxing_sessions_chart, session_intensity_chart


def test_sync_aligns_cycle_and_delayed_recovery_with_linked_sleep_wake_date():
    sleep = {"id": "sleep-1", "cycle_id": 42, "start": "2026-09-09T21:00:00Z",
             "end": "2026-09-10T05:00:00Z", "timezone_offset": "+02:00", "nap": False,
             "score_state": "SCORED", "score": {"stage_summary": {
                 "total_in_bed_time_milli": 28800000, "total_awake_time_milli": 0}}}
    cycle = {"id": 42, "start": sleep["start"], "end": "2026-09-10T21:00:00Z",
             "timezone_offset": "+02:00", "score_state": "SCORED", "score": {"strain": 14}}
    recovery = {"cycle_id": 42, "sleep_id": "sleep-1", "created_at": "2026-09-12T09:00:00Z",
                "score_state": "SCORED", "score": {"recovery_score": 75}}
    payloads = {"cycle": [cycle], "recovery": [recovery], "sleep": [sleep], "workout": []}
    def transport(method, url, **kwargs):
        return HttpResponse(200, {"records": payloads.get(url.rsplit("/", 1)[-1], [])})
    result = fetch_whoop_data(WhoopToken("at"), start="2026-09-01", end="2026-09-12", transport=transport)
    assert result.daily["Date"].tolist() == [pd.Timestamp("2026-09-10")]
    assert result.daily.loc[0, "Récupération (%)"] == 75
    assert result.daily.loc[0, "Strain"] == 14


def test_recovery_keeps_latest_revision_independent_of_api_response_order():
    records = [{"created_at": "2026-09-10T08:00:00Z", "updated_at": updated,
                "score": {"recovery_score": score}}
               for updated, score in [("2026-09-10T12:00:00Z", 90), ("2026-09-10T09:00:00Z", 20)]]
    assert recoveries_to_frame(records).loc[0, "Récupération (%)"] == 90


def test_cleaning_open_cycles_twice_preserves_the_provisional_value():
    raw = pd.DataFrame({"Date": [pd.Timestamp("2026-09-10")], "Strain": [5.2], "Cycle en cours": [True]})
    first = mask_in_progress_cycle(raw)
    second = daily_grid(first)
    assert second.loc[0, "Strain provisoire"] == pytest.approx(5.2)
    assert pd.isna(second.loc[0, "Strain"])
    assert raw.loc[0, "Strain"] == 5.2


def test_future_measurements_are_excluded_from_a_current_period():
    frame = pd.DataFrame({"Date": pd.to_datetime(["2026-09-08", "2026-09-10", "2026-09-11"])})
    assert len(filter_period(frame, 7, today="2026-09-10")) == 2


def test_training_energy_excludes_unknown_session_calories_from_both_totals():
    daily = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=10), "Calories (kcal)": 3000.0})
    workouts = pd.DataFrame({"Date": pd.to_datetime(["2026-09-02", "2026-09-02", "2026-09-03", "2026-08-15"]),
                             "Calories séance (kcal)": [600, np.nan, 900, 800]})
    result = training_energy_share(daily, workouts)
    assert result["days"] == 9  # The day with an unknown second session is incomplete.
    assert result["session_days"] == 1  # The out-of-period session is not counted.
    assert result["share_pct"] == pytest.approx(900 / (9 * 3000) * 100)


def _sessions():
    return pd.DataFrame({"Date": pd.to_datetime(["2026-09-10", "2026-09-10"]),
        "Début": pd.to_datetime(["2026-09-10 08:00", "2026-09-10 19:00"]),
        "Durée (min)": [60, 60], "Strain séance": [10, 15], "Calories séance (kcal)": [500, np.nan],
        **{f"Zone {n} (min)": [10.0, 10.0] for n in range(6)}})


def test_unknown_high_zone_cannot_be_reported_as_complete_minutes_or_share():
    sessions = _sessions().iloc[[0]].copy()
    sessions["Zone 5 (min)"] = np.nan
    table = session_table(None, sessions)
    assert np.isnan(table.loc[0, "Zones 4–5 (min)"])
    assert np.isnan(table.loc[0, "Part en zones 4–5 (%)"])


def test_weekly_calories_are_unknown_when_one_session_has_no_calorie_score():
    table = session_table(None, _sessions())
    week = weekly_sessions(table, start="2026-09-07", end="2026-09-13")
    assert week.loc[0, "Séances"] == 2
    assert np.isnan(week.loc[0, "Calories (kcal)"])


def test_weekly_aggregation_respects_bounds_inside_a_calendar_week():
    table = session_table(None, _sessions())
    week = weekly_sessions(table, start="2026-09-11", end="2026-09-13")
    assert week.loc[0, "Séances"] == 0


@pytest.mark.parametrize("chart", [boxing_sessions_chart, session_intensity_chart])
def test_two_boxing_sessions_on_one_day_have_distinct_plot_positions(chart):
    figure = chart(session_table(None, _sessions()))
    assert len(set(figure.data[0].x)) == 2


def test_calibration_score_does_not_authorize_an_intense_boxing_session():
    daily = pd.DataFrame({"Date": [pd.Timestamp("2026-09-10")], "Récupération (%)": [95], "Calibration": [True]})
    guidance = todays_guidance(daily, None, today="2026-09-10")
    assert guidance["zone"] is None
    assert "calibration" in guidance["body"].lower()


@pytest.mark.parametrize("state", ["PENDING_SCORE", "UNSCORABLE"])
def test_unscored_workout_still_counts_as_exercise_with_unknown_calories(state):
    from app.core.whoop import workouts_to_frame
    workouts = workouts_to_frame([{"id": "workout-1", "start": "2026-09-10T18:00:00Z",
        "end": "2026-09-10T19:00:00Z", "timezone_offset": "+02:00", "sport_name": "boxing",
        "score_state": state, "score": {"strain": 0, "kilojoule": 0}}])
    assert len(workouts) == 1
    assert workouts.loc[0, "Durée (min)"] == 60
    assert np.isnan(workouts.loc[0, "Calories séance (kcal)"])
    assert np.isnan(workouts.loc[0, "Strain séance"])


def test_whoop_zone_summary_does_not_normalize_an_incomplete_distribution():
    from app.core.whoop_analytics import hr_zone_profile, session_log
    sessions = _sessions()
    sessions["Sport"] = "boxing"
    sessions["Zone 5 (min)"] = np.nan
    assert hr_zone_profile(sessions, min_sessions=1).empty
    assert session_log(None, sessions)["Zones 4–5 (min)"].isna().all()


def test_boxing_energy_cannot_present_a_partial_calorie_total_as_weekly_total():
    from app.core.boxing_analytics import boxing_energy
    table = session_table(None, _sessions())
    result = boxing_energy(None, _sessions(), table, window_days=7)
    assert not result["ready"]
    assert np.isnan(result["gross_weekly"])


def test_whoop_timeline_never_adds_strain_scores_from_distinct_sessions():
    from app.core.whoop_analytics import session_log
    from app.ui.whoop_visuals import sessions_timeline_chart
    sessions = _sessions()
    sessions["Sport"] = "boxing"
    figure = sessions_timeline_chart(session_log(None, sessions))
    assert figure.layout.barmode != "stack"
    assert len(set(figure.data[0].x)) == 2


def test_old_measurements_cannot_unlock_a_sparse_recent_personal_baseline():
    from app.core.whoop_analytics import personal_baseline, vital_deviations
    dates = list(pd.date_range("2025-01-01", periods=20)) + list(pd.date_range("2026-09-01", periods=4))
    frame = pd.DataFrame({"Date": dates, "HRV (ms)": [40 + n % 3 for n in range(23)] + [10]})
    assert not personal_baseline(frame, "HRV (ms)")["ready"]
    assert vital_deviations(frame).empty


def test_sync_excludes_unmatched_recovery_and_cycle_instead_of_guessing_their_dates():
    records = {
        "cycle": [{"id": 42, "start": "2026-09-09T21:00:00Z", "end": "2026-09-10T21:00:00Z",
                   "timezone_offset": "+02:00", "score": {"strain": 14}}],
        "recovery": [{"cycle_id": 42, "sleep_id": "missing", "created_at": "2026-09-12T09:00:00Z",
                      "score": {"recovery_score": 75}}],
        "sleep": [], "workout": [],
    }
    def fake(method, url, **kwargs):
        return HttpResponse(200, {"records": records.get(url.rsplit("/", 1)[-1], [])})
    result = fetch_whoop_data(WhoopToken("test"), start="2026-09-01", end="2026-09-12", transport=fake)
    assert result.daily.empty
    assert result.counts["recovery"] == result.counts["cycle"] == 0
    assert result.counts["recovery_unaligned"] == result.counts["cycle_unaligned"] == 1
    assert result.warnings and "nuit principale" in result.summary()
    # Direct converter users still have the historical timestamp fallback.
    assert recoveries_to_frame(records["recovery"]).loc[0, "Date"] == pd.Timestamp("2026-09-12")


def test_sync_does_not_treat_a_nap_as_the_missing_main_sleep():
    records = {
        "cycle": [{"id": 42, "start": "2026-09-09T21:00:00Z", "score": {"strain": 14}}],
        "recovery": [],
        "sleep": [{"id": "nap", "cycle_id": 42, "start": "2026-09-10T13:00:00Z",
                   "end": "2026-09-10T14:00:00Z", "nap": True}],
    }
    def fake(method, url, **kwargs):
        return HttpResponse(200, {"records": records.get(url.rsplit("/", 1)[-1], [])})
    result = fetch_whoop_data(WhoopToken("test"), start="2026-09-01", end="2026-09-12", transport=fake)
    assert result.daily.empty
    assert result.counts["cycle_unaligned"] == 1


def test_sync_warnings_survive_rerun_and_clear_on_complete_sync_or_disconnect(monkeypatch):
    from app.core import session_state
    from app.core.whoop_sync import WhoopSyncResult
    state = {}
    monkeypatch.setattr(session_state.st, "session_state", state)
    incomplete = WhoopSyncResult(pd.DataFrame(), pd.DataFrame(), warnings=("Nuit principale absente",))
    session_state.store_whoop_sync(incomplete)
    session_state.ensure_whoop_defaults()
    assert state["whoop_sync_warnings"] == ["Nuit principale absente"]
    session_state.store_whoop_sync(WhoopSyncResult(pd.DataFrame(), pd.DataFrame()))
    assert not state["whoop_sync_warnings"]
    session_state.store_whoop_sync(incomplete)
    session_state.clear_whoop_session()
    assert not state["whoop_sync_warnings"]


@pytest.mark.parametrize("page", ["app/pages/Whoop.py", "app/pages/Boxe.py"])
def test_missing_sleep_warning_remains_visible_on_empty_pages_after_rerun(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(page)
    at.session_state["whoop_token"] = {"access_token": "fixture-token"}
    warning = "1 cycle exclu : aucune nuit principale associée."
    at.session_state["whoop_sync_warnings"] = [warning]
    at.run(timeout=20)
    assert not at.exception
    assert warning in [element.value for element in at.warning]
    at.run(timeout=20)
    assert not at.exception
    assert warning in [element.value for element in at.warning]
