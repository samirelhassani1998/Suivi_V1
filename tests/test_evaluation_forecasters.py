"""Tests du leaderboard walk-forward (app/core/evaluation.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.evaluation import (
    NAIVE_MODEL,
    TREND_MODEL,
    best_model,
    evaluate_forecasters,
    trend_forecast_for_dates,
    walk_forward_splits,
)


def _series(days: int, slope: float = -0.08, noise: float = 0.4, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=days, freq="D"),
            "Poids (Kgs)": 95 + slope * np.arange(days) + rng.normal(0, noise, days),
        }
    )


def test_walk_forward_splits_keep_a_minimum_training_set_and_weekly_test_blocks():
    splits = walk_forward_splits(80)
    assert 2 <= len(splits) <= 8
    first_train, first_test = splits[0]
    assert len(first_train) >= 20
    assert all(len(test) == 7 for _train, test in splits)
    # Chronologie stricte : le test suit toujours l'apprentissage.
    assert all(train.max() < test.min() for train, test in splits)
    assert walk_forward_splits(5) == []


def test_evaluate_forecasters_ranks_models_and_scores_them_against_the_naive_reference():
    table = evaluate_forecasters(_series(80), include_sarimax=True, include_auto_arima=False)
    assert not table.empty
    assert {"Modèle", "MAE", "Gain vs dernière valeur (%)", "Couverture IC 95 % (%)", "Verdict", "Disponible"} <= set(table.columns)
    available = table[table["Disponible"]]
    # Trié par MAE croissante parmi les modèles disponibles.
    assert available["MAE"].is_monotonic_increasing
    naive = table[table["Modèle"] == NAIVE_MODEL].iloc[0]
    assert naive["Gain vs dernière valeur (%)"] == 0.0
    assert naive["Verdict"] == "équivalent à la dernière valeur"
    # Sur une perte régulière, prolonger la tendance bat la dernière valeur.
    trend = table[table["Modèle"] == TREND_MODEL].iloc[0]
    assert trend["Gain vs dernière valeur (%)"] > 5
    assert trend["Verdict"] == "bat la dernière valeur"
    assert 0 <= trend["Couverture IC 95 % (%)"] <= 100


def test_evaluate_forecasters_marks_models_that_cannot_run_instead_of_failing():
    table = evaluate_forecasters(_series(25), include_sarimax=True, include_auto_arima=True)
    unavailable = table[~table["Disponible"]]
    assert not unavailable.empty
    assert (unavailable["Verdict"] == "non évalué").all()
    assert unavailable["Détail"].str.contains("trop court").all()
    # Les modèles disponibles restent classés en tête.
    assert table["Disponible"].iloc[0]


def test_evaluate_forecasters_is_empty_on_too_short_a_series():
    assert evaluate_forecasters(_series(5)).empty
    assert evaluate_forecasters(pd.DataFrame()).empty


def test_best_model_returns_the_top_available_row_or_none():
    table = evaluate_forecasters(_series(60), include_sarimax=False)
    best = best_model(table)
    assert best is not None
    assert best["Disponible"] is True or best["Disponible"] == True  # noqa: E712 — numpy bool
    assert best_model(pd.DataFrame()) is None


def test_trend_forecast_for_dates_returns_bounds_around_the_central_path():
    train = _series(40)
    dates = pd.DatetimeIndex(pd.date_range(train["Date"].max() + pd.Timedelta(days=1), periods=7))
    central, low, high = trend_forecast_for_dates(train, dates)
    assert len(central) == 7
    assert low is not None and high is not None
    assert (low <= central).all() and (central <= high).all()
    # Le chemin central descend, comme la série.
    assert central[-1] < central[0]
