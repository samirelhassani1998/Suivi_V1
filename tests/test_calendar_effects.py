"""Effet du jour de la semaine et pesées atypiques sur les écarts à la tendance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.analytics import weekday_effect
from app.core.insights import detect_anomalies_robust


def _declining(days: int = 70, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-05", periods=days, freq="D")
    return pd.DataFrame({"Date": dates, "Poids (Kgs)": np.linspace(96, 89, days) + rng.normal(0, 0.25, days)})


def test_weekday_effect_detects_an_injected_monday_bump():
    df = _declining()
    df.loc[df["Date"].dt.weekday == 0, "Poids (Kgs)"] += 0.7
    effect = weekday_effect(df)
    assert effect["ready"]
    assert effect["day"] == "Lundi"
    assert effect["mean"] > 0.4
    assert effect["significant"] is True
    assert effect["p_adjusted"] < 0.05
    assert list(effect["table"]["Jour"]) == ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def test_weekday_effect_does_not_invent_an_effect_on_a_plain_decline():
    effect = weekday_effect(_declining(seed=3))
    assert effect["ready"]
    # Le jour le plus extrême existe toujours ; il ne doit pas être déclaré établi.
    assert effect["significant"] is False


def test_weekday_effect_refuses_short_histories():
    effect = weekday_effect(_declining(days=10))
    assert effect["ready"] is False
    assert "mesures" in effect["reason"]


def test_anomalies_ignore_the_extremes_of_a_regular_decline():
    out = detect_anomalies_robust(_declining())
    # Comparées à la médiane globale, la première et la dernière pesée d'une
    # perte de 7 kg seraient signalées ; comparées à la tendance, non.
    assert not out["anomalie"].any()
    assert (out["decision"] == "conservée").all()
    assert {"ecart_tendance", "z_robuste"} <= set(out.columns)


def test_anomalies_flag_a_single_aberrant_reading_in_the_middle_of_the_series():
    df = _declining()
    df.loc[30, "Poids (Kgs)"] += 4.0
    out = detect_anomalies_robust(df)
    assert bool(out.loc[30, "anomalie"])
    assert out.loc[30, "decision"] == "à revoir"
    assert int(out["anomalie"].sum()) == 1


def test_anomalies_keep_working_with_duplicate_dates_and_isolation_forest():
    df = _declining(days=30)
    df = pd.concat([df, df.tail(1)], ignore_index=True)
    out = detect_anomalies_robust(df, use_iforest=True)
    assert len(out) == 31
    assert "iforest" in out.columns
