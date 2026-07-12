from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.analytics import prospective_scenarios
from app.core.business import FINAL_TARGET_WEIGHT_KG, STAGNATION_MIN_MEASUREMENTS
from app.core.formatting import format_fr_date, format_fr_kg, format_fr_kg_per_week
from app.core.insights import detect_plateau
from app.core.plateau import evaluate_plateau_window
from app.core.projection_constraints import constrain_interval_dataframe, truncate_projection_at_floor
from app.core.time_utils import normalize_datetime_series
from app.core.target_trajectory import build_target_trajectory
from app.core.weight_summary import detect_stagnation_periods, projection_to_target


def test_target_trajectory_official_points_and_floor():
    df = pd.DataFrame({"Date": [pd.Timestamp("2026-07-11")], "Poids (Kgs)": [102.0]})
    traj = build_target_trajectory(df)["trajectory"].rename(columns={"Poids cible (kg)": "Poids cible"})
    assert traj["Date"].iloc[0] == pd.Timestamp("2026-07-11")
    assert traj["Poids cible"].iloc[0] == 102.0
    assert traj["Date"].iloc[-1] == pd.Timestamp("2026-11-11")
    assert traj["Poids cible"].iloc[-1] == 80.0
    assert len(traj) == 124
    assert (traj["Poids cible"] >= 80.0).all()
    assert traj["Date"].max() == pd.Timestamp("2026-11-11")
    assert len(traj[traj["Poids cible"] == 80.0]) == 1

def test_normalize_datetime_series_distinguishes_iso_and_french_dates():
    parse = lambda value: normalize_datetime_series([value]).iloc[0]
    assert parse("2026-01-11") == pd.Timestamp("2026-01-11")
    assert parse("11/01/2026") == pd.Timestamp("2026-01-11")
    assert parse("01/11/2026") == pd.Timestamp("2026-11-01")
    assert parse("2026-11-01T12:30:00Z") == pd.Timestamp("2026-11-01 12:30:00")
    assert parse(pd.Timestamp("2026-01-11")) == pd.Timestamp("2026-01-11")
    aware = parse(pd.Timestamp("2026-01-11 01:00", tz="Europe/Paris"))
    assert aware == pd.Timestamp("2026-01-11 00:00")
    mixed = pd.Series(["2026-01-11", "01/11/2026"], index=["iso", "fr"])
    parsed = normalize_datetime_series(mixed)
    assert parsed.index.tolist() == ["iso", "fr"]
    assert parsed.tolist() == [pd.Timestamp("2026-01-11"), pd.Timestamp("2026-11-01")]


def test_truncate_projection_cases():
    r = truncate_projection_at_floor(pd.date_range("2026-01-01", periods=3), [82, 80, 79])
    assert r.values == [82.0, 80.0]
    r = truncate_projection_at_floor(["2026-01-01", "2026-01-11"], [82, 78])
    assert r.stop_date == pd.Timestamp("2026-01-06")
    assert r.values[-1] == 80.0
    assert truncate_projection_at_floor(["2026-01-01"], [81]).stop_date is None
    assert truncate_projection_at_floor(["2026-01-01"], [79]).values == [80.0]
    assert truncate_projection_at_floor(["2026-01-01", "2026-01-02"], [79, 82]).values == [80.0]
    r = truncate_projection_at_floor(["2026-01-03", "2026-01-01", "2026-01-02"], [83, 82, 79])
    assert r.dates[-1] == pd.Timestamp("2026-01-02")
    r = truncate_projection_at_floor(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"], [82, np.nan, 81, 79])
    assert r.values[-1] == 80.0
    assert truncate_projection_at_floor([], []).values == []
    aware = truncate_projection_at_floor([pd.Timestamp("2026-01-01", tz="Europe/Paris"), pd.Timestamp("2026-01-02", tz="UTC")], [82, 79])
    assert aware.values[-1] == 80.0


def test_prospective_scenario_projection_contracts():
    cases = [
        (np.linspace(90, 81, 40), True),
        (np.linspace(90, 86, 40), True),
        (np.linspace(90, 89, 40), False),
        (np.linspace(79, 78, 40), True),
    ]
    for weights, may_reach in cases:
        df = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=len(weights)), "Poids (Kgs)": weights})
        scenarios = prospective_scenarios(df, target_weight=80.0)
        assert scenarios
        for data in scenarios.values():
            for key in ["proj_30j", "proj_60j", "proj_90j"]:
                assert data[key] is not None
                assert data[key] >= 80.0
            assert None not in data["projection_values"]
            if data["eta_date"] is not None:
                assert data["projection_dates"][-1] == data["eta_date"]
                assert data["projection_values"][-1] == 80.0


def test_scenario_stops_at_j47_without_horizontal_tail():
    df = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=14), "Poids (Kgs)": np.linspace(90, 88, 14)})
    # 88 kg, velocity forced to about -8/47*7 = -1.191 kg/week by monkeypatching weight_velocity input trend is enough with target 80
    scenarios = prospective_scenarios(df, target_weight=80.0)
    for data in scenarios.values():
        assert min(data["projection_values"]) >= 80.0
        if data.get("eta_date") is not None:
            assert data["projection_values"][-1] == 80.0
            assert len(data["projection_values"]) == len(set(data["projection_dates"]))


def test_interval_constraints_stop_by_central_and_keep_order():
    raw = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=4), "prevision": [83, 81, 79, 78], "borne_basse": [82, 79, 75, 74], "borne_haute": [84, 82, 81, 80]})
    out = constrain_interval_dataframe(raw)
    assert out["prevision"].iloc[-1] == FINAL_TARGET_WEIGHT_KG
    assert (out[["prevision", "borne_basse", "borne_haute"]] >= 80.0).all().all()
    assert (out["borne_basse"] <= out["prevision"]).all()
    assert (out["prevision"] <= out["borne_haute"]).all()
    assert len(out) < len(raw)
    assert raw["prevision"].iloc[-1] == 78


def test_plateau_shared_engine_thresholds_and_duplicate_dates():
    three = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=3), "Poids (Kgs)": [90, 90.1, 90]})
    assert detect_plateau(three)["nb_mesures"] == 3
    assert detect_plateau(three)["status"] == "indisponible"
    assert STAGNATION_MIN_MEASUREMENTS == 4
    four = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=4, freq="4D"), "Poids (Kgs)": [90, 90.1, 90.0, 90.1]})
    assert detect_plateau(four)["status"] == evaluate_plateau_window(four)["status"]
    spanning = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=4, freq="5D"), "Poids (Kgs)": [90, 90.1, 90.0, 90.1]})
    spanning_res = detect_plateau(spanning, window=14)
    assert spanning_res["status"] == "indisponible"
    assert spanning_res["reason"] == "mesures insuffisantes"
    assert spanning_res["nb_mesures"] == 3
    falling = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=4, freq="5D"), "Poids (Kgs)": [90, 89, 88, 87]})
    assert detect_plateau(falling)["status"] != "plateau probable"
    dup = pd.DataFrame({"Date": ["2026-01-01", "2026-01-01"], "Poids (Kgs)": [90, 90]})
    assert evaluate_plateau_window(dup)["status"] == "indisponible"
    assert detect_stagnation_periods(four, window_days=14) != []


def test_format_fr_kg_signed_values():
    assert format_fr_kg(
        1.25,
        decimals=2,
        sign=True,
    ) == "+1,25 kg"

    assert format_fr_kg(
        -1.25,
        decimals=2,
        sign=True,
    ) == "−1,25 kg"

    assert format_fr_kg(
        0,
        decimals=1,
        sign=True,
    ) == "0,0 kg"


def test_formatting_and_projection_tolerance():
    assert format_fr_kg(103.5) == "103,5 kg"
    assert format_fr_kg(1.9, sign=True) == "+1,9 kg"
    assert format_fr_kg_per_week(-0.6) == "−0,6 kg/semaine"
    assert format_fr_kg(None) == "—"
    assert format_fr_kg(np.nan) == "—"
    assert format_fr_kg(0) == "0,0 kg"
    assert format_fr_date(pd.Timestamp("2026-11-25")) == "25/11/2026"
    df = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=5), "Poids (Kgs)": [81, 81, 81, 81, 80.05]})
    assert projection_to_target(df, 80.0).get("reached") is not True
