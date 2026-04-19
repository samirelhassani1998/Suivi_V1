"""Tests unitaires pour les garde-fous V3.

Couvre :
- Score plafonné à 60 pour < 7 mesures (+ confidence: fragile)
- Score complet retourne confidence: solide
- ETA refusé si < 7 mesures dans l'effort
- Milestone sans ETA si < 7 mesures
- Milestone ETA réaliste si vitesse > 2 kg/sem
- Résumé actionnable adapté à la phase de démarrage
"""

import pandas as pd
import pytest

from app.core.analytics import (
    generate_action_summary,
    next_milestone,
    progression_score,
)
from app.core.insights import estimate_target_eta


def _make_df(n: int, start_weight: float = 100.0, daily_loss: float = 0.3) -> pd.DataFrame:
    """Crée un DataFrame de test avec n mesures en perte régulière."""
    dates = pd.date_range("2026-04-10", periods=n, freq="D")
    weights = [start_weight - i * daily_loss for i in range(n)]
    return pd.DataFrame({"Date": dates, "Poids (Kgs)": weights})


# ── Test 1: Score plafonné à 60 pour < 7 mesures ──

def test_score_fallback_capped_at_60():
    df = _make_df(3, start_weight=100.0, daily_loss=1.0)  # -3 kg en 3j
    result = progression_score(df, target_weight=80.0)
    assert result["score"] <= 60, f"Score fallback doit être <= 60, got {result['score']}"
    assert result["confidence"] == "fragile"


# ── Test 2: Score complet retourne confidence: solide ──

def test_score_full_has_confidence_solide():
    df = _make_df(14, start_weight=100.0, daily_loss=0.2)
    result = progression_score(df, target_weight=80.0)
    assert "confidence" in result, "Le score complet doit retourner un champ 'confidence'"
    assert result["confidence"] == "solide"


# ── Test 3: ETA refusé si < 7 mesures dans l'effort ──

def test_eta_refused_if_effort_too_short():
    df = _make_df(20, start_weight=100.0, daily_loss=0.1)  # historique long
    effort_df = _make_df(3, start_weight=100.0, daily_loss=1.0)  # effort court
    result = estimate_target_eta(df, target_weight=80.0, effort_df=effort_df)
    assert result["credible"] is False, "ETA ne doit pas être crédible avec < 7 mesures d'effort"
    assert "démarrage" in result.get("message", "").lower() or "mesures" in result.get("message", "").lower()


# ── Test 4: Milestone sans ETA si < 7 mesures ──

def test_milestone_no_eta_if_few_measurements():
    result = next_milestone(
        current_weight=100.0,
        targets=(95.0, 90.0, 85.0, 80.0),
        velocity=-1.5,  # perte active
        measurements=3,  # trop peu
    )
    assert result["eta_days"] is None, "Pas d'ETA avec < 7 mesures"
    assert result["eta_confidence"] == "fragile"


# ── Test 5: Milestone ETA réaliste si vitesse > 2 kg/sem ──

def test_milestone_capped_eta_if_velocity_too_fast():
    result = next_milestone(
        current_weight=100.0,
        targets=(95.0, 90.0, 85.0, 80.0),
        velocity=-9.0,  # vitesse absurde (perte hydrique)
        measurements=14,  # assez de mesures
    )
    assert result["eta_days"] is not None, "ETA doit exister avec 14 mesures"
    assert result["eta_confidence"] == "optimiste"
    # ETA doit être basé sur 0.75 kg/sem, pas 9 kg/sem
    # remaining = 5 kg, à 0.75 kg/sem = 5/0.75*7 = 46.7j
    assert result["eta_days"] >= 40, f"ETA doit être réaliste (>= 40j), got {result['eta_days']}"


# ── Test 6: Résumé actionnable adapté à la phase de démarrage ──

def test_action_summary_startup_phase():
    df = _make_df(3, start_weight=100.0, daily_loss=1.0)
    result = generate_action_summary(df, target_weight=80.0)
    assert "situation" in result
    assert "interpretation" in result
    assert "action" in result
    # En phase de démarrage avec > 1.5 kg de perte, doit mentionner l'eau
    assert "eau" in result["interpretation"].lower() or "graisse" in result["interpretation"].lower()
    # L'action doit mentionner "cette semaine" ou "jour 7"
    assert "semaine" in result["action"].lower() or "jour 7" in result["action"].lower()
