"""Tests des analyses croisées WHOOP × poids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.whoop_analytics import (
    ACUTE_LOAD_DAYS,
    KCAL_PER_KG,
    MIN_DAYS_CORRELATION,
    MIN_DAYS_ENERGY_BALANCE,
    analysis_availability,
    coverage_report,
    daily_grid,
    energy_balance,
    lagged_correlations,
    recovery_drivers,
    recovery_zones,
    sleep_debt_summary,
    strain_recovery_balance,
    training_load,
    weekday_profile,
    weekly_rollup,
    weight_trend,
)


def _daily(days: int = 30, *, seed: int = 7, strain: float | None = None) -> pd.DataFrame:
    dates = pd.date_range("2026-08-01", periods=days, freq="D")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": np.clip(60 + rng.normal(0, 15, days), 5, 99),
            "Sommeil (heures)": 7 + rng.normal(0, 0.8, days),
            "Besoin de sommeil (heures)": np.full(days, 8.2),
            "Strain": np.full(days, strain) if strain is not None else np.clip(10 + rng.normal(0, 4, days), 0, 21),
            "Calories (kcal)": 2900 + rng.normal(0, 250, days),
        }
    )
    frame["Dette de sommeil (heures)"] = frame["Besoin de sommeil (heures)"] - frame["Sommeil (heures)"]
    return frame


def _merged(days: int = 30, *, kg_per_day: float = -0.09, seed: int = 3) -> pd.DataFrame:
    daily = _daily(days, seed=seed)
    rng = np.random.default_rng(seed)
    weights = pd.DataFrame(
        {
            "Date": daily["Date"],
            "Poids (Kgs)": 104 + np.arange(days) * kg_per_day + rng.normal(0, 0.1, days),
        }
    )
    merged = weights.merge(daily, on="Date", how="inner")
    merged["Variation poids (kg)"] = merged["Poids (Kgs)"].diff()
    return merged


# ── Couverture et grille calendaire ───────────────────────────────────────────


def test_daily_grid_keeps_missing_days_empty_instead_of_interpolating():
    """Sans grille continue, un graphique relie deux points distants par une droite."""
    frame = _daily(10).drop(index=[4, 5, 6]).reset_index(drop=True)

    grid = daily_grid(frame)

    assert len(grid) == 10
    assert grid["Date"].is_monotonic_increasing
    assert grid["Récupération (%)"].isna().sum() == 3
    # Les jours réellement mesurés gardent leur valeur.
    assert grid["Récupération (%)"].notna().sum() == 7


def test_daily_grid_on_empty_frame_returns_empty():
    assert daily_grid(pd.DataFrame()).empty
    assert daily_grid(None).empty


def test_coverage_report_counts_real_gaps():
    frame = _daily(10).drop(index=[4, 5]).reset_index(drop=True)

    coverage = coverage_report(frame)

    assert coverage["days_with_data"] == 8
    assert coverage["span_days"] == 10
    assert coverage["gaps"] == 2
    assert coverage["coverage_pct"] == 80.0


def test_analysis_availability_reports_what_is_still_missing():
    items = {item.name: item for item in analysis_availability(_daily(5), _merged(5))}

    energy = items["Bilan énergétique"]
    assert not energy.ready
    assert energy.missing == MIN_DAYS_ENERGY_BALANCE - 5

    ready = {item.name: item for item in analysis_availability(_daily(40), _merged(40))}
    assert all(item.ready and item.missing == 0 for item in ready.values())


# ── Zones de récupération ─────────────────────────────────────────────────────


def test_recovery_zones_follows_whoop_thresholds():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=6, freq="D"),
            "Récupération (%)": [10.0, 33.9, 34.0, 66.9, 67.0, 95.0],
        }
    )

    zones = recovery_zones(frame)
    counts = zones["counts"].set_index("Zone")["Jours"].to_dict()

    assert counts == {"Rouge": 2, "Jaune": 2, "Vert": 2}
    assert zones["latest"] == 95.0
    assert zones["latest_zone"] == "Vert"
    assert zones["days"] == 6


def test_recovery_zones_without_scores_returns_empty_structure():
    zones = recovery_zones(pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=3), "Strain": [1.0, 2.0, 3.0]}))
    assert zones["days"] == 0
    assert zones["latest_zone"] is None


# ── Charge d'entraînement ─────────────────────────────────────────────────────


def test_training_load_ratio_is_one_when_load_is_constant():
    load = training_load(_daily(30, strain=12.0))

    assert load["ratio"] == pytest.approx(1.0)
    assert load["status"] == "charge maîtrisée"


def test_training_load_detects_a_sharp_ramp_up():
    frame = _daily(30, strain=5.0)
    frame.loc[frame.index[-ACUTE_LOAD_DAYS:], "Strain"] = 18.0

    load = training_load(frame)

    assert load["ratio"] > 1.5
    assert load["status"] == "montée en charge brutale"


def test_training_load_stays_silent_below_the_minimum_history():
    load = training_load(_daily(4, strain=10.0))
    assert not np.isfinite(load["ratio"])
    assert load["status"] == "indisponible"


# ── Sommeil ───────────────────────────────────────────────────────────────────


def test_sleep_debt_summary_sums_the_recent_nights():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=3, freq="D"),
            "Sommeil (heures)": [6.0, 7.0, 8.0],
            "Besoin de sommeil (heures)": [8.0, 8.0, 8.0],
            "Dette de sommeil (heures)": [2.0, 1.0, 0.0],
        }
    )

    debt = sleep_debt_summary(frame, days=7)

    assert debt["nights"] == 3
    assert debt["cumulative_debt"] == pytest.approx(3.0)
    assert debt["mean_debt"] == pytest.approx(1.0)
    assert debt["mean_sleep"] == pytest.approx(7.0)


def test_sleep_debt_summary_limits_itself_to_the_requested_window():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=10, freq="D"),
            "Dette de sommeil (heures)": [5.0] * 7 + [1.0, 1.0, 1.0],
        }
    )

    assert sleep_debt_summary(frame, days=3)["cumulative_debt"] == pytest.approx(3.0)


# ── Tendance du poids et bilan énergétique ────────────────────────────────────


def test_weight_trend_recovers_a_known_slope():
    merged = _merged(30, kg_per_day=-0.1, seed=11)

    trend = weight_trend(merged)

    assert trend["slope_kg_per_day"] == pytest.approx(-0.1, abs=0.01)
    assert trend["slope_kg_per_week"] == pytest.approx(-0.7, abs=0.07)
    assert trend["r_squared"] > 0.9


def test_weight_trend_refuses_to_extrapolate_from_two_points():
    frame = pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=2), "Poids (Kgs)": [100.0, 99.0]})
    assert not np.isfinite(weight_trend(frame)["slope_kg_per_day"])


def test_energy_balance_derives_intake_from_burn_and_weight_slope():
    merged = _merged(30, kg_per_day=-0.1, seed=5)

    balance = energy_balance(merged)

    assert balance["ready"]
    # Perdre 0,1 kg/jour correspond à un déficit d'environ 0,1 x 7700 kcal.
    assert balance["imbalance_per_day"] == pytest.approx(-0.1 * KCAL_PER_KG, rel=0.15)
    # L'apport estimé est la dépense diminuée du déficit.
    assert balance["estimated_intake"] == pytest.approx(balance["mean_burn"] + balance["imbalance_per_day"])
    assert balance["estimated_intake"] < balance["mean_burn"]


def test_energy_balance_reports_a_surplus_when_weight_increases():
    balance = energy_balance(_merged(30, kg_per_day=0.05, seed=9))

    assert balance["ready"]
    assert balance["imbalance_per_day"] > 0
    assert balance["estimated_intake"] > balance["mean_burn"]


def test_energy_balance_withholds_its_estimate_on_a_short_history():
    balance = energy_balance(_merged(6))

    assert not balance["ready"]
    assert balance["required_days"] == MIN_DAYS_ENERGY_BALANCE
    assert not np.isfinite(balance["estimated_intake"])
    # La dépense déjà mesurée reste affichable même sans estimation complète.
    assert np.isfinite(balance["mean_burn"])


# ── Corrélations décalées ─────────────────────────────────────────────────────


def test_lagged_correlations_detects_a_planted_one_day_lag():
    days = 40
    dates = pd.date_range("2026-08-01", periods=days, freq="D")
    rng = np.random.default_rng(21)
    strain = rng.uniform(5, 18, days)
    # La variation de poids du jour reproduit le strain de la veille.
    variation = np.concatenate([[np.nan], strain[:-1] * 0.05 + rng.normal(0, 0.01, days - 1)])
    merged = pd.DataFrame({"Date": dates, "Strain": strain, "Variation poids (kg)": variation, "Poids (Kgs)": 100.0})

    table = lagged_correlations(merged, lags=(0, 1, 2), metrics=["Strain"])

    best = table.iloc[0]
    assert best["Métrique"] == "Strain"
    assert best["Décalage (jours)"] == 1
    assert best["Corrélation"] > 0.9


def test_lagged_correlations_requires_enough_common_days():
    merged = _merged(6)
    assert lagged_correlations(merged, min_pairs=MIN_DAYS_CORRELATION).empty


def test_lagged_correlations_are_sorted_by_absolute_strength():
    table = lagged_correlations(_merged(40), lags=(0, 1))
    if not table.empty:
        absolutes = table["Corrélation"].abs().tolist()
        assert absolutes == sorted(absolutes, reverse=True)


def test_lagged_correlations_on_empty_input_returns_typed_frame():
    table = lagged_correlations(pd.DataFrame())
    assert table.empty
    assert "Décalage (jours)" in table.columns


# ── Régression des moteurs de la récupération ─────────────────────────────────


def test_recovery_drivers_recovers_a_planted_sleep_effect():
    days = 40
    dates = pd.date_range("2026-08-01", periods=days, freq="D")
    rng = np.random.default_rng(4)
    sleep = rng.uniform(5, 9, days)
    strain = rng.uniform(5, 18, days)
    # Construction : chaque heure de sommeil vaut 8 points de récupération.
    recovery = 10 + 8 * sleep - 0.5 * np.concatenate([[10.0], strain[:-1]]) + rng.normal(0, 1.0, days)
    frame = pd.DataFrame({"Date": dates, "Récupération (%)": recovery, "Sommeil (heures)": sleep, "Strain": strain})

    drivers = recovery_drivers(frame)

    assert drivers["ready"]
    assert drivers["coefficients"]["Sommeil (heures)"] == pytest.approx(8.0, abs=0.6)
    assert drivers["coefficients"]["Strain de la veille"] == pytest.approx(-0.5, abs=0.4)
    assert drivers["r_squared"] > 0.9


def test_recovery_drivers_does_not_fit_noise_on_a_short_history():
    drivers = recovery_drivers(_daily(6))
    assert not drivers["ready"]
    assert drivers["coefficients"] == {}


def test_recovery_drivers_reports_a_low_explanatory_power_on_random_data():
    """Garde-fou : sur du bruit, le modèle doit avouer qu'il n'explique rien."""
    drivers = recovery_drivers(_daily(60, seed=17))

    assert drivers["ready"]
    assert drivers["r_squared"] < 0.3


# ── Synthèses ─────────────────────────────────────────────────────────────────


def test_weekly_rollup_groups_by_calendar_week():
    rollup = weekly_rollup(_merged(21))

    assert not rollup.empty
    assert int(rollup["Jours"].sum()) == 21
    assert list(rollup.columns)[:2] == ["Semaine", "Jours"]
    assert rollup["Semaine"].is_monotonic_increasing


def test_weekly_rollup_variation_uses_first_and_last_weighing():
    merged = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-03", periods=4, freq="D"),
            "Poids (Kgs)": [100.0, 99.5, 99.2, 99.0],
            "Récupération (%)": [50.0, 55.0, 60.0, 65.0],
            "Sommeil (heures)": [7.0, 7.0, 7.0, 7.0],
            "Strain": [10.0, 10.0, 10.0, 10.0],
        }
    )

    rollup = weekly_rollup(merged)

    assert len(rollup) == 1
    assert rollup.loc[0, "Variation (kg)"] == pytest.approx(-1.0)
    assert rollup.loc[0, "Strain cumulé"] == pytest.approx(40.0)


def test_weekday_profile_averages_each_day_of_week():
    profile = weekday_profile(_daily(28), "Récupération (%)")

    assert list(profile["Jour"]) == ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    assert int(profile["Observations"].sum()) == 28


def test_weekday_profile_on_missing_metric_returns_empty():
    assert weekday_profile(_daily(14), "Métrique absente").empty


def test_strain_recovery_balance_flags_hard_days_on_low_recovery():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=4, freq="D"),
            "Récupération (%)": [20.0, 80.0, 50.0, 25.0],
            "Strain": [18.0, 4.0, 10.0, 3.0],
        }
    )

    balance = strain_recovery_balance(frame)
    signals = dict(zip(balance["Date"].dt.day, balance["Signal"]))

    assert signals[1] == "charge élevée sur récupération basse"
    assert signals[2] == "récupération élevée sous-exploitée"
    assert signals[4] == "cohérent"
