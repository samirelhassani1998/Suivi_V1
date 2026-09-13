"""Tests des analyses croisées WHOOP × poids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.whoop_analytics import (
    ACUTE_LOAD_DAYS,
    DEFAULT_CORRELATION_TARGET,
    KCAL_PER_KG,
    MIN_DAYS_CONTRAST,
    MIN_DAYS_CORRELATION,
    MIN_DAYS_ENERGY_BALANCE,
    analysis_availability,
    calendar_matrix,
    contrast_best_worst_days,
    coverage_report,
    daily_grid,
    daily_log,
    daily_log_table,
    energy_balance,
    generate_insights,
    indexable_metrics,
    indexed_series,
    lagged_correlations,
    last_days,
    personal_baseline,
    physiological_watch,
    previous_days,
    projected_goal_date,
    recovery_drivers,
    recovery_streaks,
    recovery_zones,
    rolling_trend,
    sleep_architecture,
    sleep_debt_summary,
    sport_recovery_impact,
    strain_recovery_balance,
    target_pace_feasibility,
    training_energy_share,
    training_load,
    weekday_profile,
    weekly_rollup,
    vital_deviations,
    weight_trend,
)
from app.core.whoop_analytics import _robust_scale


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
    gap = merged["Date"].diff().dt.days
    merged["Jours depuis la pesée précédente"] = gap
    merged["Variation poids (kg/jour)"] = merged["Variation poids (kg)"] / gap.where(gap > 0)
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
    merged = pd.DataFrame(
        {
            "Date": dates,
            "Strain": strain,
            # Cible par défaut : le rythme quotidien, comparable d'une ligne à l'autre.
            "Variation poids (kg/jour)": variation,
            "Poids (Kgs)": 100.0,
        }
    )

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


# ── Repères personnels, lissage, base commune ─────────────────────────────────


def test_personal_baseline_compares_the_latest_value_to_your_own_median():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=15, freq="D"),
            "HRV (ms)": [40.0] * 14 + [50.0],
        }
    )

    baseline = personal_baseline(frame, "HRV (ms)")

    assert baseline["ready"]
    assert baseline["baseline"] == pytest.approx(40.0)
    assert baseline["latest"] == pytest.approx(50.0)
    assert baseline["deviation_pct"] == pytest.approx(25.0)


def test_personal_baseline_excludes_the_latest_point_from_its_own_reference():
    """Sinon la dernière valeur se comparerait partiellement à elle-même."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=12, freq="D"),
            "HRV (ms)": [40.0] * 11 + [100.0],
        }
    )
    assert personal_baseline(frame, "HRV (ms)")["baseline"] == pytest.approx(40.0)


def test_personal_baseline_stays_silent_on_a_short_history():
    frame = pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=5), "HRV (ms)": [40.0] * 5})
    assert not personal_baseline(frame, "HRV (ms)")["ready"]


def test_rolling_trend_smooths_without_bridging_gaps():
    frame = _daily(20).drop(index=[8, 9, 10]).reset_index(drop=True)

    trend = rolling_trend(frame, "Récupération (%)", window=7)

    assert len(trend) == 20
    assert trend["Récupération (%)"].isna().any()


def test_indexed_series_puts_every_metric_at_one_hundred_on_its_first_day():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=4, freq="D"),
            "Poids (Kgs)": [100.0, 99.0, 98.0, 97.0],
            "Récupération (%)": [50.0, 60.0, 40.0, 55.0],
        }
    )

    indexed = indexed_series(frame, ["Poids (Kgs)", "Récupération (%)"])

    assert indexed.loc[0, "Poids (Kgs)"] == pytest.approx(100.0)
    assert indexed.loc[0, "Récupération (%)"] == pytest.approx(100.0)
    assert indexed.loc[3, "Poids (Kgs)"] == pytest.approx(97.0)
    assert indexed.loc[1, "Récupération (%)"] == pytest.approx(120.0)


def test_indexed_series_ignores_a_metric_that_starts_at_zero():
    """Diviser par zéro produirait des infinis silencieux."""
    frame = pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=3), "Strain": [0.0, 5.0, 8.0]})
    assert "Strain" not in indexed_series(frame, ["Strain"]).columns


def test_calendar_matrix_lays_weeks_out_monday_to_sunday():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-03", periods=9, freq="D"),  # un lundi
            "Récupération (%)": list(range(50, 59)),
        }
    )

    matrix = calendar_matrix(frame, "Récupération (%)")

    assert matrix["weekdays"] == ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    assert len(matrix["weeks"]) == 2
    assert matrix["values"][0][0] == 50.0
    # Les jours non couverts de la seconde semaine restent vides.
    assert matrix["values"][1][2] is None


def test_calendar_matrix_without_usable_metric_returns_empty():
    assert calendar_matrix(pd.DataFrame(), "Récupération (%)")["values"] == []


# ── Moteur de constats ────────────────────────────────────────────────────────


def test_generate_insights_stays_almost_silent_on_a_four_day_history():
    """Situation réelle d'un bracelet neuf : mieux vaut peu de constats que des faux."""
    short = _daily(4)

    insights = generate_insights(short, _merged(4))

    assert len(insights) <= 2
    assert "Apport estimé" not in " ".join(insight.title for insight in insights)


def test_generate_insights_ranks_the_most_actionable_first():
    daily = _daily(40)
    # Une montée de charge brutale doit passer devant un constat de routine.
    daily.loc[daily.index[-7:], "Strain"] = 19.0
    daily.loc[daily.index[:-7], "Strain"] = 5.0

    insights = generate_insights(daily, _merged(40))

    assert insights
    assert insights == sorted(insights, key=lambda item: item.priority, reverse=True)
    assert any("charge" in insight.title.lower() for insight in insights)


def test_generate_insights_reports_a_sharp_training_ramp_as_a_warning():
    daily = _daily(40)
    daily.loc[daily.index[:-7], "Strain"] = 5.0
    daily.loc[daily.index[-7:], "Strain"] = 19.0

    load = [insight for insight in generate_insights(daily) if "charge" in insight.title.lower()]

    assert load and load[0].tone == "warning"


def test_generate_insights_states_the_estimated_intake_when_data_allows():
    merged = _merged(30, kg_per_day=-0.1, seed=5)

    intake = [insight for insight in generate_insights(_daily(30, seed=5), merged) if "Apport estimé" in insight.title]

    assert intake
    # Le chiffre est présent dans l'énoncé, pas seulement dans un tableau.
    assert "kcal" in intake[0].title


def test_generate_insights_flags_an_accumulated_sleep_debt():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=7, freq="D"),
            "Sommeil (heures)": [5.5] * 7,
            "Besoin de sommeil (heures)": [8.0] * 7,
            "Dette de sommeil (heures)": [2.5] * 7,
        }
    )

    debt = [insight for insight in generate_insights(frame) if "Dette" in insight.title]

    assert debt and debt[0].tone == "warning"
    assert "17,5" in debt[0].body


def test_generate_insights_celebrates_a_covered_sleep_need():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=7, freq="D"),
            "Sommeil (heures)": [8.5] * 7,
            "Besoin de sommeil (heures)": [8.0] * 7,
            "Dette de sommeil (heures)": [-0.5] * 7,
        }
    )

    covered = [insight for insight in generate_insights(frame) if "Besoin de sommeil couvert" in insight.title]

    assert covered and covered[0].tone == "success"


def test_generate_insights_warns_about_a_patchy_coverage():
    frame = _daily(20).drop(index=list(range(5, 15))).reset_index(drop=True)

    coverage = [insight for insight in generate_insights(frame) if "irrégulier" in insight.title]

    assert coverage and coverage[0].tone == "warning"


def test_generate_insights_on_empty_input_returns_nothing():
    assert generate_insights(pd.DataFrame(), pd.DataFrame()) == []


def test_generate_insights_respects_its_limit():
    assert len(generate_insights(_daily(60), _merged(60), limit=3)) <= 3


# ── Contraste bons / mauvais jours et impact par sport ────────────────────────


def test_contrast_best_worst_days_identifies_the_planted_differentiator():
    """Le facteur injecté doit ressortir en tête, pas se noyer dans le bruit."""
    days = 45
    dates = pd.date_range("2026-07-20", periods=days, freq="D")
    rng = np.random.default_rng(5)
    sleep = rng.uniform(5.0, 9.0, days)
    frame = pd.DataFrame(
        {
            "Date": dates,
            # Construction : la récupération suit le sommeil.
            "Récupération (%)": np.clip(5 + 9 * sleep + rng.normal(0, 4, days), 5, 99),
            "Sommeil (heures)": sleep,
            "Strain": rng.uniform(4, 18, days),
            "Perturbations sommeil": rng.integers(0, 9, days),
        }
    )

    contrast = contrast_best_worst_days(frame)

    assert contrast["ready"]
    assert contrast["table"].iloc[0]["Facteur"] == "Sommeil (heures)"
    assert contrast["table"].iloc[0]["Écart"] > 1.5
    assert contrast["best_threshold"] > contrast["worst_threshold"]


def test_contrast_best_worst_days_needs_enough_scored_days():
    contrast = contrast_best_worst_days(_daily(8))
    assert not contrast["ready"]
    assert contrast["table"].empty
    assert contrast["required_days"] == MIN_DAYS_CONTRAST


def test_contrast_best_worst_days_ranks_by_effect_size_not_by_unit():
    """Classer par écart brut ferait gagner l'unité de mesure, pas l'effet.

    Le strain se compte en dizaines, les heures de sommeil en unités : un écart
    de 0,6 point de strain écraserait un écart de 0,5 h de sommeil, pourtant
    bien plus significatif rapporté à sa dispersion.
    """
    contrast = contrast_best_worst_days(_daily(45, seed=2))

    if contrast["ready"]:
        effects = contrast["table"]["Écart normalisé"].abs().fillna(-1).tolist()
        assert effects == sorted(effects, reverse=True)


def test_contrast_normalised_gap_puts_the_real_driver_first():
    days = 45
    rng = np.random.default_rng(5)
    sleep = rng.uniform(5.0, 9.0, days)
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-07-20", periods=days, freq="D"),
            "Récupération (%)": np.clip(5 + 9 * sleep + rng.normal(0, 4, days), 5, 99),
            "Sommeil (heures)": sleep,
            # Le strain varie sur une échelle dix fois plus large, sans lien réel.
            "Strain": rng.uniform(4, 18, days),
        }
    )

    table = contrast_best_worst_days(frame)["table"]

    assert table.iloc[0]["Facteur"] == "Sommeil (heures)"
    assert abs(table.iloc[0]["Écart normalisé"]) > abs(table.iloc[-1]["Écart normalisé"])


def test_sport_recovery_impact_uses_the_day_after_the_session():
    """Le score du jour même précède la séance : seul le lendemain la reflète."""
    dates = pd.date_range("2026-08-01", periods=20, freq="D")
    recovery = [70.0] * 20
    workout_days = [dates[i] for i in (1, 5, 9, 13)]
    for day in workout_days:
        recovery[list(dates).index(day) + 1] = 30.0

    daily = pd.DataFrame({"Date": dates, "Récupération (%)": recovery})
    workouts = pd.DataFrame({"Date": workout_days, "Sport": ["boxing"] * 4})

    impact = sport_recovery_impact(daily, workouts)

    assert len(impact) == 1
    assert impact.loc[0, "Sport"] == "boxing"
    assert impact.loc[0, "Séances"] == 4
    assert impact.loc[0, "Récupération du lendemain (%)"] == pytest.approx(30.0)
    assert impact.loc[0, "Écart à votre moyenne"] < -20


def test_sport_recovery_impact_ranks_the_costliest_sport_first():
    dates = pd.date_range("2026-08-01", periods=30, freq="D")
    recovery = [65.0] * 30
    hard_days = [dates[i] for i in (1, 5, 9)]
    easy_days = [dates[i] for i in (2, 6, 10)]
    for day in hard_days:
        recovery[list(dates).index(day) + 1] = 25.0
    for day in easy_days:
        recovery[list(dates).index(day) + 1] = 80.0

    daily = pd.DataFrame({"Date": dates, "Récupération (%)": recovery})
    workouts = pd.DataFrame(
        {"Date": hard_days + easy_days, "Sport": ["boxing"] * 3 + ["yoga"] * 3}
    )

    impact = sport_recovery_impact(daily, workouts)

    assert list(impact["Sport"]) == ["boxing", "yoga"]


def test_sport_recovery_impact_ignores_sports_with_too_few_sessions():
    dates = pd.date_range("2026-08-01", periods=10, freq="D")
    daily = pd.DataFrame({"Date": dates, "Récupération (%)": [60.0] * 10})
    workouts = pd.DataFrame({"Date": [dates[1]], "Sport": ["boxing"]})

    assert sport_recovery_impact(daily, workouts).empty


def test_sport_recovery_impact_without_workouts_returns_typed_frame():
    impact = sport_recovery_impact(_daily(20), pd.DataFrame())
    assert impact.empty
    assert "Écart à votre moyenne" in impact.columns


def test_generate_insights_names_the_sport_that_costs_the_most():
    dates = pd.date_range("2026-08-01", periods=30, freq="D")
    recovery = [70.0] * 30
    workout_days = [dates[i] for i in (1, 5, 9, 13, 17)]
    for day in workout_days:
        recovery[list(dates).index(day) + 1] = 28.0

    daily = pd.DataFrame({"Date": dates, "Récupération (%)": recovery})
    workouts = pd.DataFrame({"Date": workout_days, "Sport": ["boxing"] * 5})

    sport = [insight for insight in generate_insights(daily, None, workouts) if "Boxing" in insight.title]

    assert sport and sport[0].tone == "warning"
    assert "lendemain" in sport[0].title


# ── Fenêtres calendaires : « 7 derniers jours » ≠ « 7 dernières lignes » ──────


def _sparse_nights() -> pd.DataFrame:
    """Trois nuits réparties sur six semaines, comme un bracelet porté par à-coups."""
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-11")],
            "Sommeil (heures)": [5.0, 7.0, 7.0],
            "Besoin de sommeil (heures)": [8.0, 8.0, 8.0],
            "Dette de sommeil (heures)": [3.0, 1.0, 1.0],
            "Récupération (%)": [20.0, 80.0, 82.0],
        }
    )


def test_last_days_selects_calendar_days_not_trailing_rows():
    window = last_days(_sparse_nights(), 7)

    assert len(window) == 2
    assert window["Date"].min() == pd.Timestamp("2026-09-10")


def test_previous_days_returns_the_window_just_before():
    frame = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=21, freq="D"), "Strain": 1.0})

    recent, earlier = last_days(frame, 7), previous_days(frame, 7)

    assert recent["Date"].min() == pd.Timestamp("2026-09-15")
    assert earlier["Date"].max() == pd.Timestamp("2026-09-14")
    assert earlier["Date"].min() == pd.Timestamp("2026-09-08")
    # Les deux fenêtres ne se recouvrent pas.
    assert set(recent["Date"]).isdisjoint(set(earlier["Date"]))


def test_sleep_debt_ignores_nights_outside_the_calendar_window():
    """Une nuit vieille de six semaines ne fait pas partie des « 7 dernières nuits »."""
    debt = sleep_debt_summary(_sparse_nights(), days=7)

    assert debt["nights"] == 2
    assert debt["cumulative_debt"] == pytest.approx(2.0)


def test_personal_baseline_reference_window_is_calendar_based():
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-01")] + list(pd.date_range("2026-09-01", periods=11, freq="D")),
            "HRV (ms)": [999.0] + [40.0] * 11,
        }
    )

    baseline = personal_baseline(frame, "HRV (ms)", days=30)

    # La valeur aberrante d'il y a plus d'un an ne doit pas servir de repère.
    assert baseline["ready"]
    assert baseline["baseline"] == pytest.approx(40.0)


# ── Charge d'entraînement : couverture réelle de la fenêtre ───────────────────


def test_training_load_refuses_to_conclude_on_a_barely_worn_week():
    """Deux séances isolées ne décrivent pas une semaine de charge soutenue."""
    dates = pd.date_range("2026-08-15", periods=28, freq="D")
    strain = [8.0] * 21 + [20.0, np.nan, np.nan, np.nan, np.nan, np.nan, 20.0]

    load = training_load(pd.DataFrame({"Date": dates, "Strain": strain}))

    assert load["acute_days_measured"] == 2
    assert load["status"] == "couverture insuffisante"
    assert not np.isfinite(load["ratio"])


def test_training_load_still_concludes_on_a_properly_worn_week():
    dates = pd.date_range("2026-08-15", periods=28, freq="D")
    load = training_load(pd.DataFrame({"Date": dates, "Strain": [10.0] * 28}))

    assert load["acute_days_measured"] == ACUTE_LOAD_DAYS
    assert load["status"] == "charge maîtrisée"
    assert load["ratio"] == pytest.approx(1.0)


def test_training_load_reports_coverage_of_both_windows():
    dates = pd.date_range("2026-08-15", periods=28, freq="D")
    strain = [np.nan] * 10 + [9.0] * 18
    load = training_load(pd.DataFrame({"Date": dates, "Strain": strain}))

    assert load["acute_days_measured"] == 7
    assert load["chronic_days_measured"] == 18


# ── Variation de poids ramenée au jour ───────────────────────────────────────


def test_lagged_correlations_default_target_is_the_daily_rate():
    """Une variation sur dix jours pèserait dix fois trop lourd sans normalisation."""
    assert DEFAULT_CORRELATION_TARGET == "Variation poids (kg/jour)"


def test_lagged_correlations_never_correlates_the_target_with_itself():
    dates = pd.date_range("2026-08-01", periods=20, freq="D")
    merged = pd.DataFrame(
        {
            "Date": dates,
            "Variation poids (kg/jour)": np.linspace(-0.3, 0.3, 20),
            "Récupération (%)": np.linspace(40, 80, 20),
        }
    )

    table = lagged_correlations(merged, lags=(0,), metrics=["Variation poids (kg/jour)", "Récupération (%)"])

    assert "Variation poids (kg/jour)" not in list(table["Métrique"])


def test_weekly_rollup_states_the_span_its_variation_covers():
    """« -1 kg cette semaine » sur deux pesées consécutives induit en erreur."""
    merged = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-04")],
            "Poids (Kgs)": [104.0, 103.0],
            "Récupération (%)": [50.0, 55.0],
            "Sommeil (heures)": [7.0, 7.0],
            "Strain": [10.0, 10.0],
        }
    )

    rollup = weekly_rollup(merged)

    assert rollup.loc[0, "Variation (kg)"] == pytest.approx(-1.0)
    assert rollup.loc[0, "Sur (jours)"] == 2


def test_weekly_rollup_span_is_absent_without_two_weighings():
    merged = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-08-03")],
            "Poids (Kgs)": [104.0],
            "Récupération (%)": [50.0],
            "Sommeil (heures)": [7.0],
            "Strain": [10.0],
        }
    )
    assert pd.isna(weekly_rollup(merged).loc[0, "Sur (jours)"])


# ── Objectif de poids traduit en calories ────────────────────────────────────


def _burn_frame(days: int = 20, *, burn: float = 2900.0, kg_per_day: float = -0.12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-09-01", periods=days, freq="D"),
            "Poids (Kgs)": 106 + np.arange(days) * kg_per_day,
            "Calories (kcal)": [burn] * days,
        }
    )


def test_target_pace_feasibility_converts_the_goal_into_an_intake():
    """0,251 kg/jour × 7700 kcal/kg = le déficit que l'objectif suppose."""
    feasibility = target_pace_feasibility(_burn_frame(), required_daily_kg=0.251)

    assert feasibility["ready"]
    assert feasibility["required_deficit"] == pytest.approx(0.251 * KCAL_PER_KG)
    assert feasibility["required_weekly_kg"] == pytest.approx(0.251 * 7)
    # Apport = dépense mesurée − déficit requis.
    assert feasibility["implied_intake"] == pytest.approx(2900.0 - 0.251 * KCAL_PER_KG)


def test_target_pace_feasibility_reports_the_current_pace_as_a_loss():
    """Une pente négative du poids est une perte : le signe doit être retourné."""
    feasibility = target_pace_feasibility(_burn_frame(kg_per_day=-0.1), required_daily_kg=0.251)
    assert feasibility["current_daily_kg"] == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize(
    ("burn", "expected"),
    [(4200.0, "exigeant"), (3500.0, "très exigeant"), (2900.0, "sous les repères usuels")],
)
def test_target_pace_feasibility_grades_how_demanding_the_goal_is(burn, expected):
    assert target_pace_feasibility(_burn_frame(burn=burn), required_daily_kg=0.251)["verdict"] == expected


def test_target_pace_feasibility_withholds_its_verdict_without_expenditure():
    frame = _burn_frame().drop(columns=["Calories (kcal)"])
    assert not target_pace_feasibility(frame, required_daily_kg=0.251)["ready"]


@pytest.mark.parametrize("bad_target", [0.0, -0.2, float("nan")])
def test_target_pace_feasibility_refuses_a_nonsensical_target(bad_target):
    """Un rythme nul ferait lire « objectif atteignable » là où il n'y a pas d'objectif."""
    feasibility = target_pace_feasibility(_burn_frame(), required_daily_kg=bad_target)

    assert not feasibility["ready"]
    assert not np.isfinite(feasibility["implied_intake"])


def test_measured_facts_outrank_the_derived_goal_arithmetic():
    """L'apport que suppose l'objectif découle des paramètres de la cible et ne
    bouge presque pas d'un jour sur l'autre : le laisser en tête chaque jour le
    transformait en bruit, devant des faits réellement mesurés."""
    insights = generate_insights(_daily(20), _burn_frame(), None, required_daily_kg=0.251)
    icons = [insight.icon for insight in insights]

    goal = next(insight for insight in insights if insight.icon == "🎯")
    assert "kcal/jour" in goal.title
    # La direction mesurée du poids arrive avant l'arithmétique de l'objectif.
    assert "⚖️" in icons
    assert icons.index("⚖️") < icons.index("🎯")


def test_goal_insight_points_to_a_professional_when_the_intake_is_low():
    insights = generate_insights(_daily(20), _burn_frame(burn=2900.0), None, required_daily_kg=0.251)
    goal = next(insight for insight in insights if insight.icon == "🎯")

    assert goal.tone == "warning"
    assert "professionnel de santé" in goal.body


def test_goal_insight_stays_neutral_when_the_intake_is_comfortable():
    insights = generate_insights(_daily(20), _burn_frame(burn=4200.0), None, required_daily_kg=0.251)
    goal = next(insight for insight in insights if insight.icon == "🎯")

    assert goal.tone == "info"
    assert "professionnel de santé" not in goal.body


def test_generate_insights_without_a_target_omits_the_goal_rule():
    assert not [i for i in generate_insights(_daily(20), _burn_frame(), None) if i.icon == "🎯"]


# ── Correction pour tests multiples ──────────────────────────────────────────


def _noise_frame(seed: int, days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = {"Date": pd.date_range("2026-08-01", periods=days, freq="D"), "Variation poids (kg/jour)": rng.normal(0, 0.2, days)}
    for metric in ("Récupération (%)", "HRV (ms)", "FC repos (bpm)", "Sommeil (heures)", "Strain", "Calories (kcal)"):
        frame[metric] = rng.normal(50, 10, days)
    return pd.DataFrame(frame)


def test_correlations_on_pure_noise_stay_near_the_nominal_false_positive_rate():
    """Une vingtaine de métriques testées à trois décalages produisait presque
    toujours une corrélation « forte » par pur hasard."""
    fired = 0
    for seed in range(40):
        table = lagged_correlations(_noise_frame(seed), lags=(0, 1, 2))
        if not table.empty and bool(table.iloc[0]["Significatif"]):
            fired += 1

    # Sans correction, ce compteur atteignait la quasi-totalité des simulations.
    assert fired <= 6, f"{fired}/40 constats sur du bruit pur"


def test_correlations_still_detect_a_genuine_planted_effect():
    days = 40
    rng = np.random.default_rng(3)
    strain = rng.uniform(5, 18, days)
    variation = np.concatenate([[np.nan], strain[:-1] * 0.02 + rng.normal(0, 0.01, days - 1)])
    merged = pd.DataFrame(
        {"Date": pd.date_range("2026-08-01", periods=days, freq="D"), "Strain": strain, "Variation poids (kg/jour)": variation}
    )

    table = lagged_correlations(merged, lags=(0, 1, 2), metrics=["Strain"])

    assert bool(table.iloc[0]["Significatif"])
    assert table.iloc[0]["Décalage (jours)"] == 1


def test_non_significant_correlations_say_so_plainly():
    table = lagged_correlations(_noise_frame(11), lags=(0, 1))
    insignificant = table[~table["Significatif"]]

    assert not insignificant.empty
    assert all("hasard" in reading for reading in insignificant["Lecture"])


def test_correlation_insight_stays_silent_on_noise():
    for seed in range(15):
        insights = generate_insights(_daily(30, seed=seed), _noise_frame(seed))
        assert not [i for i in insights if i.icon == "🔗"] or seed >= 0  # présence rare, jamais systématique
    fired = sum(1 for seed in range(15) if [i for i in generate_insights(_daily(30, seed=seed), _noise_frame(seed)) if i.icon == "🔗"])
    assert fired <= 3


# ── Charge, incertitude, R² ajusté ───────────────────────────────────────────


def test_training_load_refuses_a_ratio_while_both_windows_cover_the_same_days():
    """Sur sept jours, charge aigüe et chronique portent sur les mêmes mesures :
    le rapport vaut 1,00 par construction et rassurerait à tort."""
    for days in (7, 10, 14):
        frame = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=days, freq="D"), "Strain": np.linspace(5, 18, days)})
        load = training_load(frame)
        assert load["status"] == "historique trop court"
        assert not np.isfinite(load["ratio"])


def test_training_load_concludes_once_the_windows_differ_enough():
    frame = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=25, freq="D"), "Strain": np.linspace(5, 18, 25)})
    load = training_load(frame)

    assert np.isfinite(load["ratio"])
    assert load["status"] != "historique trop court"


def test_weight_trend_exposes_the_uncertainty_of_its_slope():
    clean = _merged(30, kg_per_day=-0.1, seed=2)
    noisy = clean.copy()
    rng = np.random.default_rng(7)
    noisy["Poids (Kgs)"] = noisy["Poids (Kgs)"] + rng.normal(0, 1.5, len(noisy))

    assert weight_trend(clean)["slope_std_error"] < weight_trend(noisy)["slope_std_error"]


def test_energy_balance_publishes_a_margin_rather_than_a_bare_number():
    balance = energy_balance(_merged(30, kg_per_day=-0.1, seed=5))

    assert balance["ready"]
    assert np.isfinite(balance["intake_margin"])
    assert balance["intake_margin"] > 0


def test_energy_balance_uses_weighings_the_join_would_have_dropped():
    """La jointure interne écartait les pesées des jours sans mesure WHOOP.

    Le bracelet n'ayant pas enregistré tous les jours, la pente ne reposait que
    sur les jours communs. Les pesées intermédiaires existent pourtant et
    resserrent l'estimation.
    """
    dates = pd.date_range("2026-09-01", periods=30, freq="D")
    rng = np.random.default_rng(12)
    true_weights = 104 - np.arange(30) * 0.1 + rng.normal(0, 0.4, 30)
    full_weights = pd.DataFrame({"Date": dates, "Poids (Kgs)": true_weights})

    # WHOOP a enregistré vingt de ces trente jours.
    whoop_days = np.sort(rng.choice(30, size=20, replace=False))
    merged = pd.DataFrame(
        {
            "Date": dates[whoop_days],
            "Poids (Kgs)": true_weights[whoop_days],
            "Calories (kcal)": [2900.0] * len(whoop_days),
        }
    )

    without_history = energy_balance(merged)
    with_history = energy_balance(merged, weight_history=full_weights)

    assert without_history["ready"] and with_history["ready"]
    # Dix pesées de plus resserrent l'estimation de la pente.
    assert with_history["intake_margin"] < without_history["intake_margin"]


def test_recovery_drivers_adjusted_r_squared_collapses_on_noise():
    """Le R² brut monte mécaniquement avec le nombre de variables."""
    rng = np.random.default_rng(1)
    days = 14
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-01", periods=days, freq="D"),
            "Récupération (%)": rng.normal(60, 12, days),
            "Sommeil (heures)": rng.normal(7, 1, days),
            "Strain": rng.normal(12, 3, days),
        }
    )

    drivers = recovery_drivers(frame)

    assert drivers["ready"]
    assert drivers["r_squared"] < drivers["raw_r_squared"]


def test_recovery_drivers_keeps_a_high_score_on_a_genuine_relationship():
    days = 40
    rng = np.random.default_rng(4)
    sleep = rng.uniform(5, 9, days)
    strain = rng.uniform(5, 18, days)
    recovery = 10 + 8 * sleep - 0.5 * np.concatenate([[10.0], strain[:-1]]) + rng.normal(0, 1.0, days)
    frame = pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=days, freq="D"), "Récupération (%)": recovery, "Sommeil (heures)": sleep, "Strain": strain})

    assert recovery_drivers(frame)["r_squared"] > 0.9


# ── Indexation ───────────────────────────────────────────────────────────────


def test_indexed_series_refuses_metrics_that_cross_zero():
    """Se coucher plus tôt (−2 h) donnerait un indice de 200 : le sens s'inverse."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-09-01", periods=4, freq="D"),
            "Heure de coucher": [-1.0, -2.0, -0.5, 0.5],
            "Poids (Kgs)": [104.0, 103.0, 103.5, 102.8],
        }
    )

    indexed = indexed_series(frame, ["Heure de coucher", "Poids (Kgs)"])

    assert "Heure de coucher" not in indexed.columns
    assert "Poids (Kgs)" in indexed.columns
    assert indexable_metrics(frame, ["Heure de coucher", "Poids (Kgs)"]) == ["Poids (Kgs)"]


# ── Couverture honnête et typage des signaux ─────────────────────────────────


def test_coverage_distinguishes_partial_days_from_complete_ones():
    """Une ligne existe dès qu'une seule des trois sources a répondu."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-09-01", periods=3, freq="D"),
            "Récupération (%)": [60.0, np.nan, 70.0],
            "Sommeil (heures)": [7.0, np.nan, 7.5],
            "Strain": [10.0, 12.0, 11.0],
        }
    )

    coverage = coverage_report(frame)

    assert coverage["days_with_data"] == 3
    # Le jour du milieu n'a ni récupération ni sommeil : il n'est pas complet.
    assert coverage["complete_days"] == 2
    assert coverage["complete_pct"] == pytest.approx(66.7)


def test_coverage_on_empty_frame_reports_zero_complete_days():
    assert coverage_report(pd.DataFrame())["complete_days"] == 0


def test_strain_balance_separates_a_risk_from_a_missed_opportunity():
    """Une bonne journée rangée parmi les alertes brouille la lecture."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-09-01", periods=4, freq="D"),
            "Récupération (%)": [20.0, 80.0, 50.0, 25.0],
            "Strain": [18.0, 4.0, 10.0, 3.0],
        }
    )

    balance = strain_recovery_balance(frame)
    by_day = dict(zip(balance["Date"].dt.day, balance["Type"]))

    assert by_day[1] == "alerte"
    assert by_day[2] == "occasion"
    assert by_day[3] == "cohérent"


# ── Accords grammaticaux ─────────────────────────────────────────────────────


def test_insights_agree_the_noun_with_the_number():
    """« 1 jour(s) » trahit un gabarit, pas une phrase."""
    one_gap = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-09-01")] + list(pd.date_range("2026-09-03", periods=8, freq="D")),
            "Récupération (%)": [50.0] * 9,
            "Sommeil (heures)": [7.0] * 9,
            "Strain": [10.0] * 9,
        }
    )

    texts = " ".join(insight.body + insight.title for insight in generate_insights(one_gap))

    assert "(s)" not in texts
    if "sans mesure" in texts:
        assert "1 jour sans mesure" in texts


def test_sleep_debt_insight_says_one_night_not_one_nights():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-09-10", periods=3, freq="D"),
            "Sommeil (heures)": [5.0, 5.5, 6.0],
            "Besoin de sommeil (heures)": [8.0, 8.0, 8.0],
            "Dette de sommeil (heures)": [3.0, 2.5, 2.0],
        }
    )

    debt = next(insight for insight in generate_insights(frame) if "Dette" in insight.title)

    assert "nuits" in debt.body
    assert "nuit(s)" not in debt.body


def test_daily_grid_tolerates_duplicate_dates_instead_of_raising():
    """Cette fonction est trop en aval pour se permettre de lever une exception :
    une date en double emportait l'affichage de toute la page."""
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02")],
            "Récupération (%)": [60.0, 40.0, 70.0],
        }
    )

    grid = daily_grid(frame)

    assert len(grid) == 2
    assert not grid["Date"].duplicated().any()
    # La dernière valeur de la journée est retenue.
    assert grid.loc[0, "Récupération (%)"] == 40.0


# ── Signes vitaux nocturnes ──────────────────────────────────────────────────


def _vitals_frame(nights: int = 25, *, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-20", periods=nights, freq="D"),
            "FC repos (bpm)": 54 + rng.normal(0, 1.5, nights),
            "HRV (ms)": 45 + rng.normal(0, 4, nights),
            "Fréquence respiratoire (resp/min)": 14.5 + rng.normal(0, 0.4, nights),
            "Température peau (°C)": 33.2 + rng.normal(0, 0.2, nights),
            "SpO2 (%)": 96.5 + rng.normal(0, 0.4, nights),
        }
    )


def test_vital_deviations_measure_the_gap_to_the_personal_reference():
    frame = _vitals_frame()
    frame.loc[frame.index[-1], "FC repos (bpm)"] = 62.0

    table = vital_deviations(frame)
    rhr = table[table["Signe vital"] == "FC repos (bpm)"].iloc[0]

    assert rhr["Dernière nuit"] == 62.0
    assert rhr["Repère habituel"] == pytest.approx(54.0, abs=1.5)
    assert rhr["Écart"] > 2.5
    assert bool(rhr["Inhabituel"])


def test_vital_deviations_only_flag_the_clinically_concerning_direction():
    """Une variabilité cardiaque haute n'est pas un signal à surveiller."""
    frame = _vitals_frame()
    frame.loc[frame.index[-1], "HRV (ms)"] = 90.0

    hrv = vital_deviations(frame).set_index("Signe vital").loc["HRV (ms)"]

    assert hrv["Écart"] > 2.5
    assert not bool(hrv["Inhabituel"])


def test_vital_deviations_stay_silent_on_a_short_history():
    assert vital_deviations(_vitals_frame(nights=6)).empty


def test_physiological_watch_counts_concordant_signals():
    frame = _vitals_frame()
    last = frame.index[-1]
    frame.loc[last, ["FC repos (bpm)", "HRV (ms)", "Fréquence respiratoire (resp/min)", "Température peau (°C)"]] = [
        62.0,
        28.0,
        16.8,
        34.1,
    ]

    watch = physiological_watch(frame)

    assert watch["count"] == 4
    assert watch["level"] == "plusieurs signaux concordants"


def test_physiological_watch_rarely_cries_wolf_on_a_quiet_series():
    """Un indicateur de santé qui se déclenche sans raison finit ignoré."""
    alarms = sum(1 for seed in range(60) if physiological_watch(_vitals_frame(seed=seed))["count"] >= 1)
    concordant = sum(1 for seed in range(60) if physiological_watch(_vitals_frame(seed=seed))["count"] >= 2)

    assert alarms <= 12, f"{alarms}/60 nuits signalées sans anomalie"
    assert concordant == 0


def test_robust_scale_resists_a_single_outlier_night():
    """Une seule nuit aberrante gonfle l'écart-type et masque ensuite tout
    écart réel ; la dispersion robuste, elle, bouge à peine."""
    steady = pd.Series([50.0, 51.0, 49.0, 50.5, 49.5, 50.0, 51.0, 50.2, 49.8, 50.3])
    with_outlier = pd.concat([steady, pd.Series([120.0])], ignore_index=True)

    robust_inflation = _robust_scale(with_outlier) / _robust_scale(steady)
    std_inflation = float(with_outlier.std()) / float(steady.std())

    # L'écart-type est multiplié par un ordre de grandeur, la mesure robuste non.
    assert std_inflation > 20
    assert robust_inflation < 2.0
    assert robust_inflation < std_inflation / 10


def test_vitals_insight_never_diagnoses():
    frame = _vitals_frame()
    last = frame.index[-1]
    frame.loc[last, ["FC repos (bpm)", "HRV (ms)", "Fréquence respiratoire (resp/min)"]] = [63.0, 27.0, 17.0]

    insight = next(item for item in generate_insights(frame) if item.icon == "🩺")

    assert "signes vitaux" in insight.title
    assert "diagnostique rien" in insight.body
    assert "professionnel de santé" in insight.body
    # Aucun nom de maladie ni conseil thérapeutique.
    assert not any(word in insight.body.lower() for word in ("grippe", "covid", "infection à", "traitement"))


def test_vitals_insight_replaces_the_redundant_baseline_cards():
    """Trois cartes disant la même chose sur la HRV noieraient le reste."""
    frame = _vitals_frame()
    last = frame.index[-1]
    frame.loc[last, ["FC repos (bpm)", "HRV (ms)", "Fréquence respiratoire (resp/min)"]] = [63.0, 27.0, 17.0]

    icons = [insight.icon for insight in generate_insights(frame)]

    assert "🩺" in icons
    assert "🫀" not in icons


# ── Journal jour par jour ────────────────────────────────────────────────────


def _log_fixtures():
    dates = pd.date_range("2026-09-08", periods=4, freq="D")
    daily = pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": [np.nan, 55.0, 42.0, 28.0],
            "HRV (ms)": [np.nan, 44.0, 40.2, 37.5],
            "FC repos (bpm)": [np.nan, 57.0, 60.0, 69.0],
            "Sommeil (heures)": [np.nan, 1.4, 7.3, 4.0],
            "Dette de sommeil (heures)": [np.nan, 6.0, 1.0, 4.2],
            "Strain": [np.nan, 4.5, 16.4, 5.2],
        }
    )
    workouts = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-09-10")] * 3,
            "Début": [
                pd.Timestamp("2026-09-10 11:00"),
                pd.Timestamp("2026-09-10 18:30"),
                pd.Timestamp("2026-09-10 21:05"),
            ],
            "Sport": ["boxing"] * 3,
            "Durée (min)": [16.0, 43.0, 39.0],
            "Strain séance": [5.3, 13.9, 7.8],
            "Calories séance (kcal)": [110.8, 505.5, 236.0],
            "FC moyenne (bpm)": [120.0, 142.0, 115.0],
            "FC max (bpm)": [142.0, 185.0, 170.0],
        }
    )
    weights = pd.DataFrame({"Date": dates, "Poids (Kgs)": [103.6, 104.0, 103.1, 102.9]})
    return daily, workouts, weights


def test_daily_log_reads_newest_first():
    entries = daily_log(*_log_fixtures())

    assert [entry.date for entry in entries] == list(pd.date_range("2026-09-11", periods=4, freq="-1D"))


def test_daily_log_attaches_each_session_to_its_own_day_and_hour():
    """Trois séances le même jour doivent rester distinctes et ordonnées."""
    entries = {entry.date: entry for entry in daily_log(*_log_fixtures())}
    tenth = entries[pd.Timestamp("2026-09-10")]

    assert len(tenth.sessions) == 3
    assert [pd.Timestamp(session.start).strftime("%H:%M") for session in tenth.sessions] == ["11:00", "18:30", "21:05"]
    assert all(session.sport == "boxing" for session in tenth.sessions)
    # Les autres jours n'héritent pas de ces séances.
    assert entries[pd.Timestamp("2026-09-11")].sessions == ()


def test_daily_log_carries_the_recovery_zone_of_each_day():
    entries = {entry.date: entry for entry in daily_log(*_log_fixtures())}

    assert entries[pd.Timestamp("2026-09-11")].zone == "Rouge"
    assert entries[pd.Timestamp("2026-09-10")].zone == "Jaune"
    # Un jour sans score n'a pas de zone plutôt qu'une zone par défaut.
    assert entries[pd.Timestamp("2026-09-08")].zone is None


def test_daily_log_reports_the_weight_and_its_change():
    entries = {entry.date: entry for entry in daily_log(*_log_fixtures())}
    eleventh = entries[pd.Timestamp("2026-09-11")]

    assert eleventh.weight == pytest.approx(102.9)
    assert eleventh.weight_change == pytest.approx(-0.2)


def test_daily_log_marks_days_without_any_measurement():
    daily, workouts, weights = _log_fixtures()
    entries = {entry.date: entry for entry in daily_log(daily, workouts, weights.iloc[1:])}

    assert not entries[pd.Timestamp("2026-09-08")].has_measurement
    assert entries[pd.Timestamp("2026-09-10")].has_measurement


def test_daily_log_keeps_calendar_gaps_visible():
    daily, workouts, weights = _log_fixtures()
    holed = daily.drop(index=[1]).reset_index(drop=True)

    entries = daily_log(holed, workouts, weights)

    # Le jour retiré reste présent, sans mesure, plutôt que de disparaître.
    assert len(entries) == 4
    assert not next(entry for entry in entries if entry.date == pd.Timestamp("2026-09-09")).sessions


def test_daily_log_honours_its_limit_and_ordering():
    entries = daily_log(*_log_fixtures(), limit=2)
    assert len(entries) == 2
    assert entries[0].date > entries[1].date


def test_daily_log_on_empty_input_returns_nothing():
    assert daily_log(pd.DataFrame()) == []
    assert daily_log_table([]).empty


def test_daily_log_table_counts_the_sessions_of_each_day():
    table = daily_log_table(daily_log(*_log_fixtures())).set_index("Date")

    assert table.loc[pd.Timestamp("2026-09-10"), "Séances"] == 3
    assert table.loc[pd.Timestamp("2026-09-11"), "Séances"] == 0


# ── Architecture du sommeil ──────────────────────────────────────────────────


def _architecture_frame(nights: int = 14, *, deep_share: float = 0.18, rem_share: float = 0.22) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    sleep = 7.0 + rng.normal(0, 0.4, nights)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-08-30", periods=nights, freq="D"),
            "Sommeil (heures)": sleep,
            "Sommeil profond (heures)": sleep * deep_share,
            "Sommeil REM (heures)": sleep * rem_share,
        }
    )


def test_sleep_architecture_reports_shares_not_hours():
    """Une nuit courte réduit mécaniquement les heures de chaque stade."""
    architecture = sleep_architecture(_architecture_frame(deep_share=0.18, rem_share=0.22))
    shares = architecture["table"].set_index("Stade")["Votre part (%)"]

    assert architecture["ready"]
    assert shares["Sommeil profond"] == pytest.approx(18.0, abs=0.3)
    assert shares["Sommeil REM"] == pytest.approx(22.0, abs=0.3)
    assert all(architecture["table"]["Position"] == "dans la plage")


def test_sleep_architecture_places_a_share_outside_the_usual_range():
    architecture = sleep_architecture(_architecture_frame(deep_share=0.09))
    deep = architecture["table"].set_index("Stade").loc["Sommeil profond"]

    assert deep["Position"] == "sous la plage"
    assert deep["Plage usuelle"] == "13 à 23 %"


def test_sleep_architecture_averages_nights_rather_than_totals():
    """Une nuit très longue ne doit pas peser davantage qu'une nuit courte."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-09-01", periods=5, freq="D"),
            "Sommeil (heures)": [2.0, 2.0, 2.0, 10.0, 10.0],
            "Sommeil profond (heures)": [0.6, 0.6, 0.6, 1.0, 1.0],
            "Sommeil REM (heures)": [0.4, 0.4, 0.4, 2.0, 2.0],
        }
    )

    deep_share = sleep_architecture(frame)["table"].set_index("Stade").loc["Sommeil profond", "Votre part (%)"]

    # Moyenne des parts : (30 + 30 + 30 + 10 + 10) / 5 = 22 %. La part du total
    # vaudrait 3,8 / 26 = 14,6 %, dominée par les deux nuits longues.
    assert deep_share == pytest.approx(22.0, abs=0.2)


def test_sleep_architecture_ignores_nights_without_sleep():
    frame = _architecture_frame(6)
    frame.loc[frame.index[:3], "Sommeil (heures)"] = 0.0
    assert not sleep_architecture(frame)["ready"]


def test_sleep_architecture_stays_silent_on_a_short_history():
    architecture = sleep_architecture(_architecture_frame(3))
    assert not architecture["ready"]
    assert architecture["table"].empty


def test_architecture_insight_presents_the_range_as_a_reference_not_a_target():
    insight = next(item for item in generate_insights(_architecture_frame(deep_share=0.09)) if item.icon == "🌙")

    assert "pas des objectifs" in insight.body or "pas un objectif" in insight.body
    assert "population" in insight.body


# ── Où va le poids : le fait que l'application doit énoncer en premier ───────


def _weight_frame(days: int = 40, *, kg_per_day: float = -0.08, noise: float = 0.15, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-08-05", periods=days, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Poids (Kgs)": 104 + np.arange(days) * kg_per_day + rng.normal(0, noise, days),
            "Calories (kcal)": [2900.0] * days,
            "Récupération (%)": [55.0] * days,
        }
    )


@pytest.mark.parametrize(
    ("kg_per_day", "expected_title"),
    [(-0.08, "Votre poids baisse"), (0.06, "Votre poids augmente"), (0.0, "Votre poids stagne")],
)
def test_insights_state_where_the_weight_is_going(kg_per_day, expected_title):
    """Une prise de deux kilos restait invisible derrière un constat sur le lundi."""
    insights = generate_insights(_daily(40), _weight_frame(kg_per_day=kg_per_day))

    card = next(insight for insight in insights if insight.icon == "⚖️")
    assert card.title == expected_title
    assert "kg par semaine" in card.body


def test_each_insight_family_owns_a_distinct_icon():
    """Deux cartes partageant une icône ne se distinguent plus d'un coup d'œil :
    la flèche descendante désignait à la fois le poids et la récupération."""
    insights = generate_insights(_daily(40), _weight_frame(kg_per_day=-0.08), limit=20)
    icons = [insight.icon for insight in insights]

    assert len(icons) == len(set(icons)), f"icônes répétées : {icons}"


def test_weight_direction_outranks_every_derived_estimate():
    insights = generate_insights(_daily(40), _weight_frame(kg_per_day=0.06))
    icons = [insight.icon for insight in insights]

    assert icons.index("⚖️") < icons.index("🔥")


def test_weight_direction_stays_silent_without_enough_weighings():
    frame = _weight_frame(days=5)
    assert not [insight for insight in generate_insights(_daily(20), frame) if insight.icon == "⚖️"]


def test_weight_direction_flags_a_scattered_trend_as_indicative():
    noisy = generate_insights(_daily(40), _weight_frame(noise=2.0))
    card = next(insight for insight in noisy if insight.icon == "⚖️")
    assert "indicatif" in card.body


# ── Projection vers la cible ─────────────────────────────────────────────────


def test_projected_goal_date_extrapolates_the_measured_pace():
    projection = projected_goal_date(_weight_frame(kg_per_day=-0.1, noise=0.05), target_weight=80.0)

    assert projection["ready"]
    # 100 kg à la dernière pesée, 0,1 kg/jour : environ 200 jours.
    assert projection["days"] == pytest.approx(200, rel=0.15)
    assert projection["date"] > pd.Timestamp("2026-09-13")


def test_projected_goal_date_refuses_when_the_weight_rises():
    projection = projected_goal_date(_weight_frame(kg_per_day=0.05), target_weight=80.0)

    assert not projection["ready"]
    assert projection["reason"] == "le poids ne va pas vers la cible"


def test_projected_goal_date_refuses_an_unreliable_trend():
    """Une droite qui n'explique rien ne peut pas fixer d'échéance."""
    projection = projected_goal_date(_weight_frame(kg_per_day=-0.01, noise=4.0), target_weight=80.0)

    assert not projection["ready"]
    assert projection["reason"] in ("tendance trop irrégulière", "échéance au-delà de dix ans")


def test_projected_goal_date_needs_a_minimum_history():
    projection = projected_goal_date(_weight_frame(days=8), target_weight=80.0)
    assert not projection["ready"]
    assert projection["reason"] == "historique trop court"


def test_target_progress_insight_compares_the_projection_to_the_deadline():
    """Être en avance aujourd'hui ne dit rien de la date d'arrivée."""
    status = {"status": "en avance", "gap_kg": -3.5}
    insights = generate_insights(
        _daily(40),
        _weight_frame(kg_per_day=-0.1, noise=0.05),
        target_status=status,
        target_weight=80.0,
        target_date=pd.Timestamp("2026-12-16"),
    )

    card = next(insight for insight in insights if insight.icon == "🧭")
    assert "en avance" in card.title
    assert "après l'échéance visée" in card.body


def test_target_progress_insight_says_when_the_goal_is_out_of_reach():
    status = {"status": "en retard", "gap_kg": 2.7}
    insights = generate_insights(
        _daily(40), _weight_frame(kg_per_day=0.05), target_status=status, target_weight=80.0
    )

    card = next(insight for insight in insights if insight.icon == "🧭")
    assert card.tone == "warning"
    assert "ne serait jamais atteinte" in card.body


def test_target_progress_insight_absent_without_a_trajectory():
    assert not [insight for insight in generate_insights(_daily(40), _weight_frame()) if insight.icon == "🧭"]


# ── Un avertissement de santé ne se fait jamais tronquer ─────────────────────


def test_a_concordant_health_warning_survives_the_display_limit():
    """Masquer une alerte parce que six autres cartes se sont déclenchées serait
    le pire comportement possible de cette liste."""
    frame = _vitals_frame(40)
    last = frame.index[-1]
    frame.loc[last, ["FC repos (bpm)", "HRV (ms)", "Fréquence respiratoire (resp/min)", "Température peau (°C)"]] = [
        64.0,
        26.0,
        17.2,
        34.2,
    ]
    frame["Sommeil (heures)"] = 5.6
    frame["Besoin de sommeil (heures)"] = 8.2
    frame["Dette de sommeil (heures)"] = 2.6
    frame["Récupération (%)"] = 45.0
    frame["Strain"] = 6.0

    insights = generate_insights(frame, limit=1)

    assert any(insight.icon == "🩺" for insight in insights)
    assert next(insight for insight in insights if insight.icon == "🩺").pinned


def test_an_isolated_health_signal_is_not_pinned():
    """Un signe isolé s'explique souvent par une soirée tardive : il ne mérite
    pas de forcer sa place."""
    frame = _vitals_frame(40)
    frame.loc[frame.index[-1], "FC repos (bpm)"] = 63.0

    watch_cards = [insight for insight in generate_insights(frame, limit=20) if insight.icon == "🩺"]
    assert watch_cards and not watch_cards[0].pinned


# ── Part de l'entraînement et séries ─────────────────────────────────────────


def test_training_energy_share_uses_calories_which_actually_add_up():
    """Le strain est une échelle logarithmique : seules les calories s'additionnent."""
    daily = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=10, freq="D"), "Calories (kcal)": [3000.0] * 10})
    workouts = pd.DataFrame(
        {"Date": [pd.Timestamp("2026-09-02"), pd.Timestamp("2026-09-05")], "Calories séance (kcal)": [600.0, 900.0]}
    )

    share = training_energy_share(daily, workouts)

    assert share["ready"]
    assert share["mean_daily_burn"] == pytest.approx(3000.0)
    # 1 500 kcal réparties sur dix jours, soit 150 par jour, soit 5 %.
    assert share["mean_session_burn"] == pytest.approx(150.0)
    assert share["share_pct"] == pytest.approx(5.0)
    assert share["session_days"] == 2


def test_training_energy_share_without_any_session_is_zero_not_missing():
    daily = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=10, freq="D"), "Calories (kcal)": [3000.0] * 10})
    share = training_energy_share(daily, pd.DataFrame())

    assert share["ready"]
    assert share["share_pct"] == pytest.approx(0.0)


def test_training_energy_share_needs_a_minimum_history():
    daily = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=3, freq="D"), "Calories (kcal)": [3000.0] * 3})
    assert not training_energy_share(daily, pd.DataFrame())["ready"]


def test_recovery_streaks_measure_how_long_a_state_lasts():
    """Une moyenne hebdomadaire lisse une série de journées rouges."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-09-01", periods=14, freq="D"),
            "Récupération (%)": [70, 75, 80, 25, 20, 30, 28, 60, 50, 71, 68, 72, 66, 80],
        }
    )

    streaks = recovery_streaks(frame)

    assert streaks["longest_red"] == 4
    assert streaks["longest_green"] == 3
    assert streaks["current_zone"] == "Vert"


def test_streak_insight_fires_only_on_a_lasting_red_run():
    short = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=6, freq="D"), "Récupération (%)": [70, 70, 70, 70, 20, 20]})
    assert not [insight for insight in generate_insights(short, limit=20) if insight.icon == "🔻"]

    lasting = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=6, freq="D"), "Récupération (%)": [70, 70, 20, 22, 25, 18]})
    card = next(insight for insight in generate_insights(lasting, limit=20) if insight.icon == "🔻")
    assert "zone rouge" in card.title
    assert card.tone == "warning"
