"""Tests hors ligne de l'intégration WHOOP (aucun appel réseau réel)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.core.whoop import (
    API_BASE_URL,
    AUTHORIZE_URL,
    DEFAULT_SCOPES,
    TOKEN_URL,
    HttpResponse,
    WhoopCredentials,
    WhoopError,
    WhoopToken,
    available_metrics,
    build_authorization_url,
    build_daily_frame,
    build_scopes,
    credentials_from_sources,
    cycles_to_frame,
    ensure_fresh_token,
    exchange_code_for_token,
    fetch_collection,
    generate_state,
    merge_with_weight,
    parse_timezone_offset,
    recoveries_to_frame,
    refresh_access_token,
    sleeps_to_frame,
    summarise_daily,
    workouts_to_frame,
)

CREDENTIALS = WhoopCredentials(
    client_id="client-id",
    client_secret="client-secret",
    redirect_uri="https://exemple.test/callback",
)

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


class FakeTransport:
    """Transport scriptable : enregistre les appels et renvoie des réponses fixes."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, method, url, *, params=None, data=None, headers=None, timeout=30):
        self.calls.append(
            {"method": method, "url": url, "params": dict(params or {}), "data": dict(data or {}), "headers": dict(headers or {})}
        )
        if not self.responses:
            raise AssertionError(f"Appel non prévu : {method} {url}")
        return self.responses.pop(0)


def _recovery_record(created_at: str, recovery_score: float, hrv: float = 45.0) -> dict:
    return {
        "cycle_id": 1,
        "sleep_id": "11111111-1111-1111-1111-111111111111",
        "user_id": 7,
        "created_at": created_at,
        "updated_at": created_at,
        "score_state": "SCORED",
        "score": {
            "user_calibrating": False,
            "recovery_score": recovery_score,
            "resting_heart_rate": 54.0,
            "hrv_rmssd_milli": hrv,
            "spo2_percentage": 96.4,
            "skin_temp_celsius": 33.2,
        },
    }


def _sleep_record(start: str, end: str, *, nap: bool = False, in_bed_milli: int = 30_600_000, awake_milli: int = 1_800_000) -> dict:
    return {
        "id": "22222222-2222-2222-2222-222222222222",
        "user_id": 7,
        "start": start,
        "end": end,
        "timezone_offset": "+02:00",
        "nap": nap,
        "score_state": "SCORED",
        "score": {
            "stage_summary": {
                "total_in_bed_time_milli": in_bed_milli,
                "total_awake_time_milli": awake_milli,
                "total_no_data_time_milli": 0,
                "total_light_sleep_time_milli": 14_400_000,
                "total_slow_wave_sleep_time_milli": 5_400_000,
                "total_rem_sleep_time_milli": 7_200_000,
                "sleep_cycle_count": 4,
                "disturbance_count": 6,
            },
            "sleep_needed": {"baseline_milli": 27_000_000},
            "respiratory_rate": 15.1,
            "sleep_performance_percentage": 88.0,
            "sleep_consistency_percentage": 70.0,
            "sleep_efficiency_percentage": 94.0,
        },
    }


def _cycle_record(start: str, strain: float, kilojoule: float = 11_000.0) -> dict:
    return {
        "id": 42,
        "user_id": 7,
        "created_at": start,
        "updated_at": start,
        "start": start,
        "end": None,
        "timezone_offset": "+02:00",
        "score_state": "SCORED",
        "score": {"strain": strain, "kilojoule": kilojoule, "average_heart_rate": 72, "max_heart_rate": 165},
    }


# ── OAuth ─────────────────────────────────────────────────────────────────────


def test_build_authorization_url_contains_required_oauth_parameters():
    url = build_authorization_url(CREDENTIALS, "state-long-enough")

    assert url.startswith(AUTHORIZE_URL + "?")
    assert "response_type=code" in url
    assert "client_id=client-id" in url
    assert "redirect_uri=https%3A%2F%2Fexemple.test%2Fcallback" in url
    assert "state=state-long-enough" in url
    for scope in DEFAULT_SCOPES:
        assert scope.replace(":", "%3A") in url
    # Le secret client ne doit jamais transiter par l'URL d'autorisation.
    assert "client-secret" not in url


def test_build_authorization_url_rejects_short_state():
    with pytest.raises(WhoopError):
        build_authorization_url(CREDENTIALS, "court")


def test_generate_state_is_long_enough_and_unique():
    first, second = generate_state(), generate_state()
    assert len(first) >= 8 and first != second


def test_exchange_code_for_token_posts_expected_payload():
    transport = FakeTransport([
        HttpResponse(200, {"access_token": "at", "refresh_token": "rt", "expires_in": 3600, "scope": "offline read:recovery"})
    ])

    token = exchange_code_for_token(CREDENTIALS, "auth-code", transport=transport, now=NOW)

    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == TOKEN_URL
    assert call["data"]["grant_type"] == "authorization_code"
    assert call["data"]["code"] == "auth-code"
    assert call["data"]["redirect_uri"] == CREDENTIALS.redirect_uri
    assert token.access_token == "at"
    assert token.refresh_token == "rt"
    assert token.expires_at == NOW + timedelta(seconds=3600)
    assert token.scopes == ("offline", "read:recovery")


def test_exchange_code_for_token_raises_readable_error_on_http_failure():
    transport = FakeTransport([HttpResponse(400, {"error": "invalid_grant"})])

    with pytest.raises(WhoopError) as excinfo:
        exchange_code_for_token(CREDENTIALS, "bad-code", transport=transport)

    assert "400" in str(excinfo.value)
    assert "client-secret" not in str(excinfo.value)


def test_refresh_access_token_keeps_previous_refresh_token_when_absent():
    transport = FakeTransport([HttpResponse(200, {"access_token": "at2", "expires_in": 60})])
    token = WhoopToken(access_token="at1", refresh_token="rt1", expires_at=NOW)

    refreshed = refresh_access_token(CREDENTIALS, token, transport=transport, now=NOW)

    assert transport.calls[0]["data"]["grant_type"] == "refresh_token"
    assert refreshed.access_token == "at2"
    assert refreshed.refresh_token == "rt1"


def test_ensure_fresh_token_refreshes_only_when_expired():
    valid = WhoopToken(access_token="at", refresh_token="rt", expires_at=NOW + timedelta(hours=1))
    untouched = FakeTransport([])
    assert ensure_fresh_token(CREDENTIALS, valid, transport=untouched, now=NOW) is valid
    assert untouched.calls == []

    expired = WhoopToken(access_token="old", refresh_token="rt", expires_at=NOW)
    transport = FakeTransport([HttpResponse(200, {"access_token": "new", "expires_in": 3600})])
    assert ensure_fresh_token(CREDENTIALS, expired, transport=transport, now=NOW).access_token == "new"


def test_ensure_fresh_token_without_refresh_token_raises():
    expired = WhoopToken(access_token="old", refresh_token=None, expires_at=NOW)
    with pytest.raises(WhoopError):
        ensure_fresh_token(CREDENTIALS, expired, transport=FakeTransport([]), now=NOW)


def test_token_round_trips_through_session_dict():
    token = WhoopToken(access_token="at", refresh_token="rt", expires_at=NOW, scopes=("offline",))
    restored = WhoopToken.from_dict(token.to_dict())
    assert restored.access_token == "at"
    assert restored.refresh_token == "rt"
    assert restored.scopes == ("offline",)
    assert restored.expires_at == NOW


# ── Appels API ────────────────────────────────────────────────────────────────


def test_fetch_collection_follows_pagination_and_sends_bearer_token():
    transport = FakeTransport([
        HttpResponse(200, {"records": [_recovery_record("2026-09-09T12:00:00.000Z", 60)], "next_token": "page-2"}),
        HttpResponse(200, {"records": [_recovery_record("2026-09-10T12:00:00.000Z", 70)], "next_token": None}),
    ])
    token = WhoopToken(access_token="at")

    records = fetch_collection("recovery", token, start="2026-09-01", end="2026-09-10", transport=transport)

    assert len(records) == 2
    assert transport.calls[0]["url"] == f"{API_BASE_URL}/v2/recovery"
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer at"
    assert transport.calls[0]["params"]["limit"] == 25
    assert transport.calls[0]["params"]["start"].endswith("Z")
    assert "nextToken" not in transport.calls[0]["params"]
    assert transport.calls[1]["params"]["nextToken"] == "page-2"


def test_fetch_collection_stops_on_repeated_next_token():
    page = HttpResponse(200, {"records": [_recovery_record("2026-09-09T12:00:00.000Z", 60)], "next_token": "same"})
    transport = FakeTransport([page, page])
    records = fetch_collection("recovery", WhoopToken(access_token="at"), start="2026-09-01", end="2026-09-10", transport=transport)
    assert len(records) == 2
    assert len(transport.calls) == 2


def test_fetch_collection_caps_limit_to_api_maximum():
    transport = FakeTransport([HttpResponse(200, {"records": [], "next_token": None})])
    fetch_collection("cycle", WhoopToken(access_token="at"), start="2026-09-01", end="2026-09-10", transport=transport, limit=500)
    assert transport.calls[0]["params"]["limit"] == 25


def test_fetch_collection_surfaces_expired_and_throttled_tokens():
    with pytest.raises(WhoopError, match="401"):
        fetch_collection("sleep", WhoopToken(access_token="at"), start="2026-09-01", end="2026-09-10", transport=FakeTransport([HttpResponse(401, {})]))
    with pytest.raises(WhoopError, match="429"):
        fetch_collection("sleep", WhoopToken(access_token="at"), start="2026-09-01", end="2026-09-10", transport=FakeTransport([HttpResponse(429, {})]))


def test_fetch_collection_rejects_unknown_resource():
    with pytest.raises(WhoopError):
        fetch_collection("inconnu", WhoopToken(access_token="at"), start="2026-09-01", end="2026-09-10", transport=FakeTransport([]))


# ── Normalisation ─────────────────────────────────────────────────────────────


def test_parse_timezone_offset_handles_whoop_formats():
    assert parse_timezone_offset("+02:00") == pd.Timedelta(hours=2)
    assert parse_timezone_offset("-0430") == -pd.Timedelta(hours=4, minutes=30)
    assert parse_timezone_offset(None) == pd.Timedelta(0)
    assert parse_timezone_offset("bizarre") == pd.Timedelta(0)


def test_recoveries_to_frame_extracts_scores_and_skips_unparsable_rows():
    frame = recoveries_to_frame([
        _recovery_record("2026-09-10T12:00:00.000Z", 62, hrv=45.2),
        {"score": {"recovery_score": 50}},  # sans date : ignorée
    ])

    assert len(frame) == 1
    assert frame.loc[0, "Date"] == pd.Timestamp("2026-09-10")
    assert frame.loc[0, "Récupération (%)"] == 62.0
    assert frame.loc[0, "HRV (ms)"] == 45.2
    assert frame.loc[0, "FC repos (bpm)"] == 54.0


def test_recoveries_to_frame_returns_typed_empty_frame():
    frame = recoveries_to_frame([])
    assert frame.empty
    assert "Récupération (%)" in frame.columns


def test_sleeps_to_frame_attaches_night_to_wake_up_day_and_prefers_main_night():
    frame = sleeps_to_frame([
        _sleep_record("2026-09-09T22:00:00.000Z", "2026-09-10T06:30:00.000Z"),
        _sleep_record("2026-09-10T12:00:00.000Z", "2026-09-10T13:00:00.000Z", nap=True, in_bed_milli=3_600_000, awake_milli=0),
    ])

    assert len(frame) == 1
    assert frame.loc[0, "Date"] == pd.Timestamp("2026-09-10")
    # 30 600 000 ms au lit - 1 800 000 ms éveillé = 8 h de sommeil.
    assert frame.loc[0, "Sommeil (heures)"] == pytest.approx(8.0)
    assert frame.loc[0, "Performance sommeil (%)"] == 88.0
    assert frame.loc[0, "Sommeil profond (heures)"] == pytest.approx(1.5)


def test_cycles_to_frame_converts_kilojoules_to_kilocalories():
    frame = cycles_to_frame([_cycle_record("2026-09-10T04:00:00.000Z", 12.4, kilojoule=11_000.0)])

    assert frame.loc[0, "Strain"] == 12.4
    assert frame.loc[0, "Calories (kcal)"] == pytest.approx(2629.06, abs=0.1)


def test_workouts_to_frame_computes_duration_and_distance():
    frame = workouts_to_frame([
        {
            "id": "33333333-3333-3333-3333-333333333333",
            "user_id": 7,
            "start": "2026-09-10T17:00:00.000Z",
            "end": "2026-09-10T18:05:00.000Z",
            "timezone_offset": "+02:00",
            "sport_name": "Running",
            "score_state": "SCORED",
            "score": {"strain": 9.1, "average_heart_rate": 140, "max_heart_rate": 180, "kilojoule": 2500.0, "percent_recorded": 100, "distance_meter": 8500.0},
        }
    ])

    assert frame.loc[0, "Sport"] == "Running"
    assert frame.loc[0, "Durée (min)"] == pytest.approx(65.0)
    assert frame.loc[0, "Distance (km)"] == pytest.approx(8.5)


def test_build_daily_frame_merges_sources_on_calendar_day():
    recovery = recoveries_to_frame([_recovery_record("2026-09-10T12:00:00.000Z", 62)])
    sleep = sleeps_to_frame([_sleep_record("2026-09-09T22:00:00.000Z", "2026-09-10T06:30:00.000Z")])
    cycle = cycles_to_frame([_cycle_record("2026-09-10T04:00:00.000Z", 12.4)])

    daily = build_daily_frame(recovery, sleep, cycle)

    assert len(daily) == 1
    assert {"Récupération (%)", "Sommeil (heures)", "Strain"} <= set(daily.columns)
    # Les colonnes techniques ne polluent pas la vue journalière.
    assert "Calibration" not in daily.columns and "Sieste" not in daily.columns


def test_build_daily_frame_without_sources_returns_empty_typed_frame():
    daily = build_daily_frame()
    assert daily.empty
    assert "Date" in daily.columns
    assert available_metrics(daily) == []


# ── Croisement avec le poids ──────────────────────────────────────────────────


def _daily_fixture(days: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2026-09-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": [50 + (index % 5) * 4 for index in range(days)],
            "Sommeil (heures)": [6.5 + (index % 4) * 0.4 for index in range(days)],
            "Strain": [10 + (index % 3) for index in range(days)],
        }
    )


def test_merge_with_weight_keeps_common_days_and_computes_variation():
    weights = pd.DataFrame({"Date": pd.date_range("2026-09-05", periods=6, freq="D"), "Poids (Kgs)": [100.0, 99.6, 99.8, 99.2, 99.0, 98.7]})

    merged = merge_with_weight(weights, _daily_fixture())

    assert len(merged) == 6
    assert merged["Date"].min() == pd.Timestamp("2026-09-05")
    assert "Variation poids (kg)" in merged.columns
    assert merged.loc[1, "Variation poids (kg)"] == pytest.approx(-0.4)


def test_merge_with_weight_averages_multiple_measures_of_the_same_day():
    weights = pd.DataFrame({"Date": [pd.Timestamp("2026-09-05")] * 2, "Poids (Kgs)": [100.0, 101.0]})
    merged = merge_with_weight(weights, _daily_fixture())
    assert len(merged) == 1
    assert merged.loc[0, "Poids (Kgs)"] == pytest.approx(100.5)


def test_merge_with_weight_without_any_source_returns_empty_frame():
    assert merge_with_weight(pd.DataFrame(), _daily_fixture()).empty
    assert merge_with_weight(pd.DataFrame({"Date": [pd.Timestamp("2026-09-05")], "Poids (Kgs)": [100.0]}), pd.DataFrame()).empty


def test_summarise_daily_compares_last_window_to_previous_one():
    summary = summarise_daily(_daily_fixture(days=14), days=7)
    assert "Récupération (%)" in summary
    stats = summary["Récupération (%)"]
    assert pd.notna(stats["current"]) and pd.notna(stats["previous"])
    assert stats["delta"] == pytest.approx(stats["current"] - stats["previous"])


def test_summarise_daily_on_empty_frame_returns_empty_mapping():
    assert summarise_daily(pd.DataFrame()) == {}


# ── Résolution des identifiants ───────────────────────────────────────────────


def test_credentials_from_sources_gives_priority_to_session_overrides():
    credentials = credentials_from_sources(
        {"whoop": {"client_id": "secrets-id", "client_secret": "secrets-secret", "redirect_uri": "https://secrets"}},
        {"WHOOP_CLIENT_ID": "env-id"},
        {"client_id": "ui-id"},
    )
    assert credentials.client_id == "ui-id"
    assert credentials.client_secret == "secrets-secret"
    assert credentials.redirect_uri == "https://secrets"
    assert credentials.is_complete


def test_credentials_from_sources_supports_flat_secrets_and_environment():
    credentials = credentials_from_sources(
        {"whoop_client_id": "flat-id"},
        {"WHOOP_CLIENT_SECRET": "env-secret", "WHOOP_REDIRECT_URI": "https://env"},
    )
    assert (credentials.client_id, credentials.client_secret, credentials.redirect_uri) == ("flat-id", "env-secret", "https://env")


def test_credentials_from_sources_without_anything_is_incomplete():
    credentials = credentials_from_sources()
    assert not credentials.is_complete
    assert credentials.scopes == DEFAULT_SCOPES


def test_build_scopes_toggles_only_the_offline_scope():
    with_offline = build_scopes(offline=True)
    without_offline = build_scopes(offline=False)

    assert with_offline[0] == "offline"
    assert with_offline == DEFAULT_SCOPES
    assert "offline" not in without_offline
    # Les scopes de lecture restent identiques dans les deux cas.
    assert set(with_offline) - {"offline"} == set(without_offline)


def test_authorization_url_reflects_a_scope_set_without_offline():
    credentials = WhoopCredentials(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://exemple.test/",
        scopes=build_scopes(offline=False),
    )
    url = build_authorization_url(credentials, "state-long-enough")
    assert "offline" not in url
    assert "read%3Arecovery" in url
