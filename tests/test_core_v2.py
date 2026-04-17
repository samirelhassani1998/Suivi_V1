from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.data import clean_weight_dataframe, data_quality_report, resolve_duplicates, validate_journal
from app.core.evaluation import evaluate_baselines, walk_forward_backtest
from app.core.forecasting import forecast_with_ml, forecast_with_sarimax
from app.core.insights import detect_anomalies_robust, detect_plateau, estimate_target_eta


def test_clean_and_validate_columns_with_extra_columns():
    raw = pd.DataFrame(
        {
            "dates": ["01/01/2026", "02/01/2026"],
            "poids": ["80,2", "79,9"],
            "Colonne perso": ["A", "B"],
        }
    )
    cleaned = clean_weight_dataframe(raw)
    assert list(cleaned.columns[:2]) == ["Date", "Poids (Kgs)"]
    assert "Colonne perso" in cleaned.columns

    res = validate_journal(raw)
    assert len(res.errors) == 0
    assert len(res.cleaned) == 2
    assert "Colonne perso" in res.cleaned.columns


def test_import_preserves_row_count_and_columns():
    raw = pd.DataFrame(
        {
            "Date": ["01/01/2026", "02/01/2026", "03/01/2026"],
            "Poids (Kgs)": [81, 80.5, 80],
            "Extra 1": [1, 2, 3],
            "Extra 2": ["x", "y", "z"],
        }
    )
    out = clean_weight_dataframe(raw)
    assert len(out) == 3
    assert ["Extra 1", "Extra 2"] == [c for c in out.columns if c.startswith("Extra")]


def test_duplicate_strategy_works_with_extra_columns():
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "Poids (Kgs)": [80.0, 79.0, 78.5],
            "Note perso": ["a", "b", "c"],
        }
    )
    out = resolve_duplicates(raw, "moyenne_journaliere")
    assert len(out) == 2
    assert "Note perso" in out.columns
    assert np.isclose(float(out.loc[out["Date"] == pd.Timestamp("2026-01-01"), "Poids (Kgs)"].iloc[0]), 79.5)


def test_backtesting_outputs(synthetic_df):
    table = evaluate_baselines(synthetic_df["Poids (Kgs)"])
    assert {"modèle", "mae", "rmse", "directional_accuracy"}.issubset(table.columns)
    assert len(table) == 4


def test_walk_forward_small_series():
    s = pd.Series([80.0, 79.8, 79.7])
    out = walk_forward_backtest(s, lambda tr, h: np.repeat(tr.iloc[-1], h))
    assert np.isnan(out["mae"])


def test_eta_objectif_simple_dataset():
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    weights = np.linspace(95, 90, len(dates))
    df = pd.DataFrame({"Date": dates, "Poids (Kgs)": weights})
    eta = estimate_target_eta(df, target_weight=88.0)
    assert "credible" in eta


def test_plateau_detection(synthetic_df):
    res = detect_plateau(synthetic_df, window=14)
    assert "status" in res and "slope" in res


def test_anomalies_detection(synthetic_df):
    df = synthetic_df.copy()
    df.loc[df.index[-1], "Poids (Kgs)"] = 99
    out = detect_anomalies_robust(df)
    assert out["anomalie"].any()


def test_data_quality_report(synthetic_df):
    r = data_quality_report(synthetic_df)
    assert 0 <= r["score"] <= 100


def test_forecast_functions_return_well_formed_dataframes(synthetic_df):
    sarimax = forecast_with_sarimax(synthetic_df, horizon=14)
    ml = forecast_with_ml(synthetic_df, horizon=14, height_m=1.82)

    for frame in [sarimax, ml]:
        assert isinstance(frame, pd.DataFrame)
        if not frame.empty:
            assert {"Date", "prevision", "borne_basse", "borne_haute"}.issubset(frame.columns)
