from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.data import clean_weight_dataframe, data_quality_report, validate_journal
from app.core.evaluation import evaluate_baselines, walk_forward_backtest
from app.core.features import build_features
from app.core.insights import detect_anomalies_robust, detect_plateau, estimate_target_eta


def test_clean_and_validate_columns():
    raw = pd.DataFrame({"dates": ["01/01/2026", "bad"], "poids": ["80,2", "x"]})
    cleaned = clean_weight_dataframe(raw)
    assert list(cleaned.columns[:2]) == ["Date", "Poids (Kgs)"]
    res = validate_journal(raw)
    assert len(res.errors) == 0
    assert len(res.cleaned) == 1


def test_feature_engineering(synthetic_df):
    feat = build_features(synthetic_df, height_m=1.82)
    for col in ["lag_1", "lag_30", "roll_mean_7", "variation_journaliere", "jour_semaine", "imc"]:
        assert col in feat.columns


def test_backtesting_outputs(synthetic_df):
    table = evaluate_baselines(synthetic_df["Poids (Kgs)"])
    assert {"modèle", "mae", "rmse", "directional_accuracy"}.issubset(table.columns)
    assert len(table) == 4


def test_walk_forward_small_series():
    s = pd.Series([80.0, 79.8, 79.7])
    out = walk_forward_backtest(s, lambda tr, h: np.repeat(tr.iloc[-1], h))
    assert np.isnan(out["mae"])


def test_eta_objectif(synthetic_df):
    eta = estimate_target_eta(synthetic_df, target_weight=84.0)
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
