"""Tests du module de tendance robuste (app/core/trend.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.trend import (
    bmi_category,
    eta_to_target,
    noise_level,
    project_trend,
    rate_of_change,
    reading_vs_trend,
    trend_weight,
    weight_for_bmi,
)


def _series(days: int = 60, start: float = 95.0, slope_per_day: float = -0.08, noise: float = 0.4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-03-01", periods=days, freq="D")
    values = start + slope_per_day * np.arange(days) + rng.normal(0.0, noise, days)
    return pd.DataFrame({"Date": dates, "Poids (Kgs)": values})


def test_trend_weight_tracks_the_underlying_line_better_than_raw_readings():
    df = _series(days=60, slope_per_day=-0.08, noise=0.5)
    frame = trend_weight(df)
    truth = 95.0 - 0.08 * np.arange(60)
    raw_error = float(np.mean(np.abs(frame["Poids (Kgs)"].to_numpy() - truth)))
    trend_error = float(np.mean(np.abs(frame["Tendance"].to_numpy() - truth)))
    assert list(frame.columns) == ["Date", "Poids (Kgs)", "Tendance", "Résidu"]
    assert len(frame) == 60
    assert trend_error < raw_error * 0.6


def test_trend_weight_is_robust_to_a_single_outlier():
    df = _series(days=40, noise=0.2)
    spiked = df.copy()
    spiked.loc[20, "Poids (Kgs)"] += 6.0
    clean = trend_weight(df)["Tendance"].to_numpy()
    with_spike = trend_weight(spiked)["Tendance"].to_numpy()
    # La tendance au point aberrant bouge de bien moins que l'aberration elle-même.
    assert abs(with_spike[20] - clean[20]) < 1.5


def test_trend_weight_falls_back_gracefully_on_short_or_empty_series():
    empty = trend_weight(pd.DataFrame(columns=["Date", "Poids (Kgs)"]))
    assert empty.empty
    short = trend_weight(_series(days=3))
    assert len(short) == 3
    assert short["Tendance"].notna().all()


def test_noise_level_recovers_the_injected_daily_noise():
    df = _series(days=90, noise=0.5, seed=3)
    noise = noise_level(df)
    assert noise["ready"]
    assert 0.3 < noise["sigma"] < 0.8
    assert noise["band"] == pytest.approx(1.96 * noise["sigma"])


def test_rate_of_change_confidence_interval_contains_the_true_slope():
    df = _series(days=40, slope_per_day=-0.1, noise=0.3, seed=11)
    rate = rate_of_change(df, window_days=28)
    assert rate["ready"]
    true_weekly = -0.7
    assert rate["ci_low"] <= true_weekly <= rate["ci_high"]
    assert rate["ci_low"] < rate["slope_kg_week"] < rate["ci_high"]
    assert rate["significant"] is True
    assert rate["direction"] == "baisse"
    assert rate["p_value"] < 0.05


def test_rate_of_change_declares_a_flat_noisy_series_stable():
    df = _series(days=40, slope_per_day=0.0, noise=0.4, seed=5)
    rate = rate_of_change(df, window_days=28)
    assert rate["ready"]
    assert rate["direction"] == "stable"
    assert rate["ci_low"] < 0 < rate["ci_high"]


def test_rate_of_change_refuses_short_histories_with_a_reason():
    rate = rate_of_change(_series(days=3), window_days=14)
    assert rate["ready"] is False
    assert "mesures" in rate["reason"]
    dense = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=6, freq="D"), "Poids (Kgs)": [90, 89.9, 89.8, 89.7, 89.6, 89.5]})
    rate = rate_of_change(dense, window_days=14)
    assert rate["ready"] is False
    assert "recul" in rate["reason"]


def test_reading_vs_trend_flags_a_reading_outside_the_usual_noise():
    df = _series(days=50, noise=0.25, seed=2)
    ordinary = reading_vs_trend(df)
    assert ordinary["ready"]
    assert ordinary["verdict"] == "dans le bruit habituel"
    spiked = df.copy()
    spiked.loc[spiked.index[-1], "Poids (Kgs)"] += 3.0
    outlier = reading_vs_trend(spiked)
    assert outlier["verdict"] == "au-dessus du bruit habituel"
    assert outlier["deviation"] > outlier["band"]


def test_project_trend_produces_an_ordered_widening_cone():
    df = _series(days=40, slope_per_day=-0.08, noise=0.3, seed=9)
    projection = project_trend(df, horizon_days=30)
    assert len(projection) == 30
    assert (projection["borne_basse"] <= projection["prevision"]).all()
    assert (projection["prevision"] <= projection["borne_haute"]).all()
    widths = (projection["borne_haute"] - projection["borne_basse"]).to_numpy()
    assert widths[-1] > widths[0]
    assert projection["Date"].iloc[0] == df["Date"].max() + pd.Timedelta(days=1)


def test_project_trend_is_empty_without_a_usable_rate():
    assert project_trend(_series(days=3), horizon_days=10).empty


def test_eta_to_target_gives_a_date_and_an_ordered_range_for_a_declining_series():
    df = _series(days=40, slope_per_day=-0.1, noise=0.2, seed=4)
    eta = eta_to_target(df, target_weight=85.0)
    assert eta["ready"]
    assert eta["eta"] is not None
    assert eta["eta_early"] is not None and eta["eta_early"] <= eta["eta"]
    if eta["eta_late"] is not None:
        assert eta["eta_late"] >= eta["eta"]


def test_eta_to_target_declines_when_the_slope_is_not_established():
    df = _series(days=40, slope_per_day=0.0, noise=0.4, seed=8)
    eta = eta_to_target(df, target_weight=80.0)
    assert eta["ready"] is False
    assert "stagnation" in eta["reason"]


def test_eta_to_target_reports_a_reached_target():
    df = _series(days=20, start=79.0, slope_per_day=-0.05, noise=0.1)
    eta = eta_to_target(df, target_weight=80.0)
    assert eta["ready"] and eta["reached"]
    assert eta["days"] == 0


@pytest.mark.parametrize(
    ("bmi", "label"),
    [
        (17.0, "insuffisance pondérale"),
        (22.0, "corpulence normale"),
        (27.5, "surpoids"),
        (32.0, "obésité (classe I)"),
        (37.0, "obésité (classe II)"),
        (42.0, "obésité (classe III)"),
        (float("nan"), "indisponible"),
    ],
)
def test_bmi_category_follows_who_thresholds(bmi, label):
    assert bmi_category(bmi)[0] == label


def test_weight_for_bmi_inverts_the_formula():
    assert weight_for_bmi(25.0, 1.82) == pytest.approx(25.0 * 1.82**2)
