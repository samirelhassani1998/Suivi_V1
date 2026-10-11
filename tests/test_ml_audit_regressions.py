"""Backtests vérifiables : forme, calendrier, effectifs et référence naïve."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core import forecasting
from app.core.evaluation import NAIVE_MODEL, evaluate_forecasters, evaluate_regressors, walk_forward_splits


def _weights(n=60):
    return pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=n),
                         "Poids (Kgs)": 100 + np.random.default_rng(17).normal(0, 0.3, n)})


@pytest.mark.parametrize("bad", [100.0, "invalid", np.array([[100.0, 101.0]]), None])
def test_malformed_prediction_is_unavailable_instead_of_crashing(bad):
    table = evaluate_forecasters(_weights(), forecasters={"incorrect": lambda train, dates: (bad, None, None)})
    assert not table.iloc[0]["Disponible"]
    assert table.iloc[0]["Pesées testées"] == 0
    assert "prédiction" in table.iloc[0]["Détail"]


def test_matrix_prediction_with_the_right_row_count_is_still_invalid():
    table = evaluate_forecasters(_weights(), forecasters={
        "incorrect": lambda train, dates: (np.full((len(dates), 2), 100), None, None)
    })
    assert not table.iloc[0]["Disponible"]


def test_one_missing_interval_bound_cannot_claim_a_valid_model():
    table = evaluate_forecasters(_weights(), forecasters={
        "incorrect": lambda train, dates: (np.full(len(dates), 100), None, np.full(len(dates), 101))
    })
    assert not table.iloc[0]["Disponible"]
    assert table.iloc[0]["Détail"] == "intervalle incomplet"


@pytest.mark.parametrize("value", [np.inf, -np.inf, "invalide"])
def test_nonfinite_truth_is_excluded_before_scoring(value):
    data = _weights()
    data["Poids (Kgs)"] = data["Poids (Kgs)"].astype(object)
    data.loc[40, "Poids (Kgs)"] = value
    expected = evaluate_forecasters(data.drop(index=40), include_sarimax=False)
    actual = evaluate_forecasters(data, include_sarimax=False)
    pd.testing.assert_frame_equal(actual, expected)


def test_perfect_reference_is_evaluated_and_a_worse_model_is_not_equivalent():
    data = _weights().assign(**{"Poids (Kgs)": 100.0})
    table = evaluate_forecasters(data, forecasters={
        NAIVE_MODEL: lambda train, dates: (np.repeat(100., len(dates)), None, None),
        "Décalé": lambda train, dates: (np.repeat(101., len(dates)), None, None),
    }).set_index("Modèle")
    assert table.loc[NAIVE_MODEL, "Verdict"] == "équivalent à la dernière valeur"
    assert table.loc[NAIVE_MODEL, "Gain vs dernière valeur (%)"] == 0
    assert table.loc["Décalé", "Verdict"] == "moins bon que la dernière valeur"
    assert np.isnan(table.loc["Décalé", "Gain vs dernière valeur (%)"])
    assert "erreur nulle" in table.loc["Décalé", "Détail"]


def test_walk_forward_never_splits_measurements_from_the_same_day():
    dates = pd.date_range("2026-01-01", periods=40).repeat(3)
    for train, test in walk_forward_splits(len(dates), dates=dates):
        assert dates[train].max() < dates[test].min()
    too_few_days = _weights(7).loc[lambda frame: frame.index.repeat(4)].reset_index(drop=True)
    assert evaluate_forecasters(too_few_days, include_sarimax=False).empty


def test_coverage_is_reported_with_sample_count_width_and_calendar_reach():
    data = _weights()
    data["Date"] = pd.date_range("2026-01-01", periods=len(data), freq="7D")
    data["Poids (Kgs)"] = 100.
    table = evaluate_forecasters(data, forecasters={
        NAIVE_MODEL: lambda train, dates: (np.repeat(100., len(dates)), np.repeat(99., len(dates)), np.repeat(101., len(dates)))
    })
    row = table.iloc[0]
    assert row["Couverture IC 95 % (%)"] == 100
    assert row["Largeur moyenne IC (kg)"] == 2
    assert row["Pesées testées"] == 7 * row["Blocs évalués"]
    assert row["Horizon min (j)"] == 7
    assert row["Horizon max (j)"] == 49


def test_backtest_horizon_is_calendar_based_across_dst():
    data = _weights(40)
    data["Date"] = pd.date_range("2026-03-01", periods=len(data), tz="Europe/Paris")
    table = evaluate_forecasters(data, include_sarimax=False)
    assert table["Disponible"].all()
    assert table["Horizon min (j)"].eq(1).all()
    assert table["Horizon max (j)"].eq(7).all()


def test_regression_comparison_requires_usable_training_and_test_days():
    assert evaluate_regressors(_weights(54)).empty
    table = evaluate_regressors(_weights(55))
    assert table.attrs["train_days"] == 20
    assert table.attrs["test_days"] == 5
    assert NAIVE_MODEL in set(table["Modèle"])


def test_regression_reference_uses_the_previous_observed_test_weight():
    data = _weights(80)
    table = evaluate_regressors(data, models={})
    n_test = table.attrs["test_rows"]
    expected_mae = data["Poids (Kgs)"].diff().abs().tail(n_test).mean()
    assert table.iloc[0]["MAE"] == pytest.approx(expected_mae)
    assert table.iloc[0]["Gain vs dernière valeur (%)"] == 0
    assert table.attrs["train_end"] < table.attrs["test_start"]


def test_regression_holdout_keeps_repeated_days_together():
    data = _weights(60).loc[lambda frame: frame.index.repeat(3)].reset_index(drop=True)
    table = evaluate_regressors(data, models={})
    assert not table.empty
    assert table.attrs["train_end"] < table.attrs["test_start"]


def test_regression_constant_test_has_no_fabricated_r_squared():
    table = evaluate_regressors(_weights(60).assign(**{"Poids (Kgs)": 100.0}), models={})
    assert table["R²"].isna().all()
    assert table["Verdict"].eq("équivalent à la dernière valeur").all()


class _ConstantModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.repeat(100., len(X))


@pytest.mark.parametrize("last_date", ["2026-03-28", "2026-10-24"])
def test_ml_forecast_keeps_local_days_across_daylight_saving(monkeypatch, last_date):
    monkeypatch.setattr(forecasting, "get_quantile_models", lambda: {key: _ConstantModel() for key in ("q10", "q50", "q90")})
    data = _weights()
    data["Date"] = pd.date_range(end=last_date, periods=len(data), tz="Europe/Paris")
    pred = forecasting.forecast_with_ml(data, 3, 1.82)
    expected = pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=3)
    pd.testing.assert_index_equal(pd.DatetimeIndex(pred["Date"]).rename(None), expected)


@pytest.mark.parametrize("value", [np.nan, np.inf, "invalide"])
def test_daily_forecast_refuses_invalid_weight_instead_of_imputing_it(value):
    data = _weights().astype({"Poids (Kgs)": object})
    data.loc[40, "Poids (Kgs)"] = value
    with pytest.raises(ValueError, match="poids numériques finis"):
        forecasting.forecast_with_ml(data, 7, 1.82)


def test_sarimax_refuses_nonconverged_fit(monkeypatch):
    monkeypatch.setattr(forecasting, "SARIMAX", lambda *args, **kwargs: SimpleNamespace(
        fit=lambda **options: SimpleNamespace(mle_retvals={"converged": False})
    ))
    with pytest.raises(ValueError, match="non convergé"):
        forecasting.forecast_with_sarimax(_weights(), 7)


def test_sarimax_validates_raw_interval_before_display_clipping(monkeypatch):
    prediction = SimpleNamespace(predicted_mean=pd.Series([100., 100.]),
                                 conf_int=lambda **kwargs: pd.DataFrame({0: [99., np.nan], 1: [101., 101.]}))
    fitted = SimpleNamespace(mle_retvals={"converged": True}, get_forecast=lambda **kwargs: prediction)
    monkeypatch.setattr(forecasting, "SARIMAX", lambda *args, **kwargs: SimpleNamespace(fit=lambda **options: fitted))
    with pytest.raises(ValueError, match="intervalle incomplet"):
        forecasting.forecast_with_sarimax(_weights(), 2)


def _page(data, strategy="garder_la_derniere"):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app/pages/Predictions.py", default_timeout=30)
    for key in ("source_data", "working_data", "raw_data"):
        at.session_state[key] = data.copy()
    at.session_state["filtered_data"] = pd.DataFrame()
    at.session_state["filter_active"] = False
    at.session_state["fast_mode"] = True
    at.session_state["duplicate_strategy"] = strategy
    return at


def test_predictions_page_applies_selected_daily_aggregation(monkeypatch):
    from app.core import evaluation
    observed = []
    original = evaluation.evaluate_forecasters

    def capture(data, **kwargs):
        observed.append(data.copy())
        return original(data, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_forecasters", capture)
    data = _weights(61)
    data = pd.concat([data, data.assign(**{"Poids (Kgs)": data["Poids (Kgs)"] + 2})], ignore_index=True)
    at = _page(data, "moyenne_journaliere").run()
    assert not at.exception
    assert observed
    assert len(observed[-1]) == 61
    np.testing.assert_allclose(observed[-1]["Poids (Kgs)"], _weights(61)["Poids (Kgs)"] + 1)
    assert any("Portée réelle" in caption.value for caption in at.caption)


def test_predictions_ml_ui_shows_naive_reference_and_test_size():
    at = _page(_weights(65)).run()
    assert not at.exception
    next(item for item in at.checkbox if item.label == "Calculer les modèles ML").check().run()
    assert not at.exception
    table = next(item.value for item in at.dataframe if "R²" in item.value.columns)
    assert NAIVE_MODEL in set(table["Modèle"])
    assert "Gain vs dernière valeur (%)" in table.columns
    assert any("Apprentissage :" in caption.value and "Test :" in caption.value for caption in at.caption)
