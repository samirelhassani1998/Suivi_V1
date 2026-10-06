"""Régressions : calendrier, fuites de cible et stabilité des prévisions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core import forecasting
from app.core.evaluation import (
    AUTO_ARIMA_MODEL,
    NAIVE_MODEL,
    SARIMAX_MODEL,
    default_forecasters,
    evaluate_forecasters,
    walk_forward_backtest,
)
from app.core.features import build_features
from app.core.forecasting import MIN_ML_MEASUREMENTS, forecast_with_ml, forecast_with_sarimax


def _weights(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=n),
        "Poids (Kgs)": 100 + np.random.default_rng(42).normal(0, 0.5, n),
    })


def test_imported_target_copies_and_empty_notes_do_not_change_model_inputs_or_forecast():
    source = _weights()
    imported = source.assign(IMC=source["Poids (Kgs)"] / 1.82**2, copie_cible=source["Poids (Kgs)"], Notes=None)
    pd.testing.assert_frame_equal(build_features(source, 1.82), build_features(imported, 1.82))
    pd.testing.assert_frame_equal(forecast_with_ml(source, 7, 1.82), forecast_with_ml(imported, 7, 1.82))


def test_small_stationary_series_does_not_explode_into_a_false_target_arrival():
    # Avant régularisation : arrivée à 80 kg en 6,7 jours et bande [80;80].
    pred = forecast_with_ml(_weights(MIN_ML_MEASUREMENTS), 30, 1.82)
    assert len(pred) == 30
    assert pred["prevision"].between(97, 103).all()
    assert np.isfinite(pred[["prevision", "borne_basse", "borne_haute"]]).all().all()
    assert (pred["borne_basse"] <= pred["prevision"]).all()
    assert (pred["prevision"] <= pred["borne_haute"]).all()


def test_ml_minimum_counts_lag_warmup_and_effective_training_rows():
    assert forecast_with_ml(_weights(MIN_ML_MEASUREMENTS - 1), 7, 1.82).empty
    assert len(forecast_with_ml(_weights(MIN_ML_MEASUREMENTS), 7, 1.82)) == 7


def test_daily_calendar_accepts_the_daylight_saving_transition():
    df = _weights()
    df["Date"] = pd.date_range("2026-03-01", periods=len(df), tz="Europe/Paris")
    forecasting.require_daily_sampling(df)
    features = build_features(df)
    assert features["jours_depuis_derniere_mesure"].iloc[1:].eq(1).all()


@pytest.mark.parametrize("cadence", ["weekly", "missing_day", "duplicate_day"])
@pytest.mark.parametrize("method", ["sarimax", "ml"])
def test_daily_models_refuse_to_relabel_measurement_steps_as_calendar_days(cadence, method):
    df = _weights()
    if cadence == "weekly":
        df["Date"] = pd.date_range("2026-01-01", periods=len(df), freq="7D")
    elif cadence == "missing_day":
        df = df.drop(index=20)
    else:
        df.loc[20, "Date"] = df.loc[19, "Date"]
    with pytest.raises(ValueError, match="une seule pesée par jour"):
        if method == "sarimax":
            forecast_with_sarimax(df, 7)
        else:
            forecast_with_ml(df, 7, 1.82)


def test_leaderboard_marks_arima_unavailable_on_irregular_calendar_and_keeps_baselines():
    df = _weights()
    df["Date"] = pd.date_range("2026-01-01", periods=len(df), freq="7D")
    table = evaluate_forecasters(df, include_sarimax=True, include_auto_arima=True).set_index("Modèle")
    assert table.loc[NAIVE_MODEL, "Disponible"]
    for model in (SARIMAX_MODEL, AUTO_ARIMA_MODEL):
        assert not table.loc[model, "Disponible"]
        assert "une seule pesée par jour" in table.loc[model, "Détail"]


def test_drift_predicts_for_actual_requested_dates():
    train = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-22"]),
        "Poids (Kgs)": [100.0, 99.0, 97.0],
    })
    dates = pd.to_datetime(["2026-01-29", "2026-02-12"])
    predict = default_forecasters(include_sarimax=False)["Tendance linéaire (drift)"]
    central, _, _ = predict(train, dates)
    np.testing.assert_allclose(central, [96.0, 94.0])


def test_direction_accuracy_never_counts_jumps_between_refitted_forecasts():
    df = _weights(40)
    df["Poids (Kgs)"] = 100 - np.arange(len(df)) * 0.1
    table = evaluate_forecasters(df, include_sarimax=False).set_index("Modèle")
    assert table.loc[NAIVE_MODEL, "Précision directionnelle (%)"] == 0
    metrics = walk_forward_backtest(df["Poids (Kgs)"], lambda train, h: np.repeat(train.iloc[-1], h))
    assert metrics["directional_accuracy"] == 0


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.repeat(self.value, len(X))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1e12])
def test_unstable_quantile_model_is_refused_before_display_clipping(monkeypatch, invalid):
    monkeypatch.setattr(forecasting, "get_quantile_models", lambda: {key: _ConstantModel(invalid) for key in ("q10", "q50", "q90")})
    with pytest.raises(ValueError, match="Prévision ML instable"):
        forecast_with_ml(_weights(), 7, 1.82)


def test_crossed_quantiles_are_reordered_without_collapsing_the_interval(monkeypatch):
    monkeypatch.setattr(forecasting, "get_quantile_models", lambda: {
        "q10": _ConstantModel(101), "q50": _ConstantModel(100), "q90": _ConstantModel(99),
    })
    pred = forecast_with_ml(_weights(), 7, 1.82)
    np.testing.assert_allclose(pred["borne_basse"], 99)
    np.testing.assert_allclose(pred["prevision"], 100)
    np.testing.assert_allclose(pred["borne_haute"], 101)


def test_malformed_interval_does_not_crash_the_leaderboard_or_claim_coverage():
    def invalid_bounds(train, dates):
        return np.repeat(100.0, len(dates)), np.array([90.0]), np.repeat(110.0, len(dates))

    table = evaluate_forecasters(_weights(), forecasters={"Mauvais intervalle": invalid_bounds})
    assert not table.iloc[0]["Disponible"]
    assert table.iloc[0]["Détail"] == "intervalle incomplet"


def test_predictions_page_runs_sarimax_backtest_only_when_requested(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.core import evaluation

    calls = []

    def counted_forecast(train, dates):
        calls.append(len(train))
        return np.repeat(float(train["Poids (Kgs)"].iloc[-1]), len(dates)), None, None

    monkeypatch.setattr(evaluation, "_sarimax_forecaster", counted_forecast)
    at = AppTest.from_file("app/pages/Predictions.py", default_timeout=30)
    df = _weights(47)
    for key in ("source_data", "working_data", "raw_data"):
        at.session_state[key] = df.copy()
    at.session_state["filtered_data"] = pd.DataFrame()
    at.session_state["filter_active"] = False
    at.session_state["fast_mode"] = True
    at.run()
    assert not at.exception
    assert calls == []
    tables = [frame.value for frame in at.dataframe]
    initial = next(table for table in tables if "Verdict" in table.columns)
    assert SARIMAX_MODEL not in set(initial["Modèle"])

    checkbox = next(item for item in at.checkbox if item.label == "Évaluer SARIMAX dans le classement")
    checkbox.check().run()
    assert not at.exception
    assert calls
    tables = [frame.value for frame in at.dataframe]
    evaluated = next(table for table in tables if "Verdict" in table.columns)
    assert SARIMAX_MODEL in set(evaluated["Modèle"])
