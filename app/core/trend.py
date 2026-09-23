"""Tendance robuste du poids : poids de tendance, bruit quotidien et rythme avec incertitude.

Une pesée quotidienne bouge de plusieurs centaines de grammes d'un jour à
l'autre sous l'effet de l'eau, du glycogène et du contenu digestif, sans que la
masse grasse ait changé. Lire la courbe brute revient donc à lire du bruit.
Ce module fournit trois lectures qui résistent à ce bruit :

- un **poids de tendance** (régression locale robuste LOWESS, Cleveland 1979),
  qui suit la série sans être tirée par une pesée isolée ;
- le **bruit quotidien typique** autour de cette tendance, pour dire si la
  pesée du jour est un vrai mouvement ou une fluctuation ordinaire ;
- un **rythme en kg/semaine avec son intervalle de confiance à 95 %**, estimé
  par moindres carrés sur les jours calendaires, pour distinguer une pente
  établie d'une pente que le hasard suffit à produire.

Toutes les fonctions sont pures, sans dépendance Streamlit, et refusent de
conclure sur un effectif insuffisant : chaque résultat porte son effectif et,
le cas échéant, la raison pour laquelle il n'est pas disponible.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from app.core.weight_summary import moving_average_by_days, prepare_weight_series

DATE_COL = "Date"
WEIGHT_COL = "Poids (Kgs)"
TREND_COL = "Tendance"
RESIDUAL_COL = "Résidu"

# Largeur de la fenêtre locale du lissage, en jours calendaires. Deux semaines
# absorbent les cycles hebdomadaires (week-end, sel, alcool) sans effacer un
# changement de régime réel.
TREND_WINDOW_DAYS = 14
# En deçà, le lissage local n'a pas assez de points pour être robuste : on
# retombe sur une moyenne mobile calendaire.
MIN_POINTS_TREND = 5
# Nombre minimal de points dans la fenêtre LOWESS (fraction × n).
MIN_LOCAL_POINTS = 4
# Itérations de repondération robuste : une pesée aberrante perd son poids.
LOWESS_ITERATIONS = 2

# Rythme : effectif et recul minimaux avant d'estimer une pente.
MIN_POINTS_RATE = 5
MIN_SPAN_RATE_DAYS = 7
RATE_WINDOW_DAYS = 14
LONG_RATE_WINDOW_DAYS = 28
CONFIDENCE_LEVEL = 0.95
# Facteur de conversion de l'écart absolu médian en écart-type sous normalité.
MAD_TO_SIGMA = 1.4826
# Au-delà de ce facteur du bruit habituel, une pesée est jugée hors norme.
Z_BAND = 1.96


def _t_quantile(dof: int) -> float:
    return float(stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, max(1, dof)))


def trend_weight(df: pd.DataFrame, window_days: int = TREND_WINDOW_DAYS) -> pd.DataFrame:
    """Série de poids dédoublonnée par jour, avec sa tendance robuste et le résidu.

    Le lissage est une régression locale pondérée (LOWESS) sur les jours
    calendaires : elle s'adapte aux pesées irrégulières, contrairement à une
    moyenne sur N mesures, et ses itérations robustes retirent leur influence
    aux pesées aberrantes. Sous ``MIN_POINTS_TREND`` mesures, la tendance est
    une moyenne mobile calendaire, plus stable qu'un lissage sous-alimenté.
    """
    data = prepare_weight_series(df)
    if data.empty:
        return pd.DataFrame(columns=[DATE_COL, WEIGHT_COL, TREND_COL, RESIDUAL_COL])

    y = data[WEIGHT_COL].to_numpy(dtype=float)
    n = len(data)
    span_days = float((data[DATE_COL].iloc[-1] - data[DATE_COL].iloc[0]).total_seconds() / 86400)

    if n < MIN_POINTS_TREND or span_days <= 0:
        trend = moving_average_by_days(data, window_days).to_numpy(dtype=float)
    else:
        x = ((data[DATE_COL] - data[DATE_COL].iloc[0]).dt.total_seconds() / 86400).to_numpy(dtype=float)
        frac = float(window_days) / span_days
        frac = max(frac, MIN_LOCAL_POINTS / n)
        frac = min(1.0, frac)
        trend = lowess(y, x, frac=frac, it=LOWESS_ITERATIONS, return_sorted=False)
        trend = np.asarray(trend, dtype=float)
        if not np.all(np.isfinite(trend)):
            trend = moving_average_by_days(data, window_days).to_numpy(dtype=float)

    out = data.copy()
    out[TREND_COL] = trend
    out[RESIDUAL_COL] = y - trend
    return out.reset_index(drop=True)


def noise_level(df: pd.DataFrame, window_days: int = TREND_WINDOW_DAYS) -> dict[str, Any]:
    """Dispersion robuste des pesées autour de la tendance.

    ``sigma`` est l'écart-type robuste (MAD × 1,4826) des résidus, ``band`` la
    demi-largeur à 95 % : une pesée qui s'écarte de la tendance de moins de
    ``band`` est une fluctuation ordinaire, pas un changement.
    """
    frame = trend_weight(df, window_days)
    residuals = frame[RESIDUAL_COL].dropna().to_numpy(dtype=float) if not frame.empty else np.array([])
    result = {"ready": False, "n": int(len(residuals)), "sigma": float("nan"), "band": float("nan")}
    if len(residuals) < MIN_POINTS_TREND:
        return result
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    sigma = MAD_TO_SIGMA * mad
    if sigma <= 1e-9:
        # Résidus quasi nuls (série lisse ou très courte) : l'écart-type
        # classique reste une borne raisonnable.
        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    result.update({"ready": True, "sigma": float(sigma), "band": float(Z_BAND * sigma)})
    return result


def reading_vs_trend(df: pd.DataFrame, window_days: int = TREND_WINDOW_DAYS) -> dict[str, Any]:
    """Situe la dernière pesée par rapport à la tendance et au bruit habituel."""
    frame = trend_weight(df, window_days)
    noise = noise_level(df, window_days)
    result = {
        "ready": False,
        "reading": float("nan"),
        "trend": float("nan"),
        "deviation": float("nan"),
        "band": noise["band"],
        "verdict": "indisponible",
        "date": None,
    }
    if frame.empty:
        return result
    latest = frame.iloc[-1]
    deviation = float(latest[WEIGHT_COL] - latest[TREND_COL])
    result.update(
        {
            "reading": float(latest[WEIGHT_COL]),
            "trend": float(latest[TREND_COL]),
            "deviation": deviation,
            "date": latest[DATE_COL],
        }
    )
    if not noise["ready"]:
        return result
    band = noise["band"]
    if abs(deviation) <= band:
        verdict = "dans le bruit habituel"
    elif deviation > 0:
        verdict = "au-dessus du bruit habituel"
    else:
        verdict = "au-dessous du bruit habituel"
    result.update({"ready": True, "verdict": verdict})
    return result


def rate_of_change(df: pd.DataFrame, window_days: int = RATE_WINDOW_DAYS) -> dict[str, Any]:
    """Pente du poids sur les derniers jours calendaires, avec intervalle de confiance.

    Moindres carrés ordinaires sur les jours écoulés ; l'intervalle et la
    valeur p sont ceux de la loi de Student à n − 2 degrés de liberté. Une
    pente dont l'intervalle contient zéro n'est pas distinguable d'une
    stagnation : le résultat le dit plutôt que d'afficher un chiffre signé.
    """
    data = prepare_weight_series(df)
    result: dict[str, Any] = {
        "ready": False,
        "reason": None,
        "window_days": int(window_days),
        "n": int(len(data)),
        "span_days": 0,
        "start_date": None,
        "end_date": None,
        "slope_kg_week": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "std_error_kg_week": float("nan"),
        "p_value": float("nan"),
        "r_squared": float("nan"),
        "significant": False,
        "direction": "indéterminée",
    }
    if data.empty:
        result["reason"] = "aucune mesure"
        return result

    last_date = data[DATE_COL].iloc[-1]
    cutoff = last_date - pd.Timedelta(days=int(window_days))
    window = data[data[DATE_COL] >= cutoff]
    result["n"] = int(len(window))
    if len(window) < MIN_POINTS_RATE:
        result["reason"] = f"moins de {MIN_POINTS_RATE} mesures sur {int(window_days)} jours"
        return result

    x = ((window[DATE_COL] - window[DATE_COL].iloc[0]).dt.total_seconds() / 86400).to_numpy(dtype=float)
    y = window[WEIGHT_COL].to_numpy(dtype=float)
    span = float(x.max())
    result.update({"span_days": int(round(span)), "start_date": window[DATE_COL].iloc[0], "end_date": last_date})
    if span < MIN_SPAN_RATE_DAYS:
        result["reason"] = f"recul de {int(round(span))} jour(s), {MIN_SPAN_RATE_DAYS} requis"
        return result

    dof = len(window) - 2
    x_centered = x - x.mean()
    sxx = float(np.sum(x_centered**2))
    if sxx <= 0 or dof < 1:
        result["reason"] = "durée insuffisante"
        return result
    slope = float(np.sum(x_centered * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    fitted = intercept + slope * x
    residual_ss = float(np.sum((y - fitted) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    sigma2 = residual_ss / dof
    se_slope = float(np.sqrt(sigma2 / sxx))
    t_crit = _t_quantile(dof)
    if se_slope > 0:
        t_stat = slope / se_slope
        p_value = float(2.0 * stats.t.sf(abs(t_stat), dof))
    else:
        p_value = 0.0 if abs(slope) > 0 else 1.0

    slope_week = slope * 7.0
    half_width = t_crit * se_slope * 7.0
    ci_low = slope_week - half_width
    ci_high = slope_week + half_width
    significant = bool(p_value < (1.0 - CONFIDENCE_LEVEL))
    if significant and slope_week < 0:
        direction = "baisse"
    elif significant and slope_week > 0:
        direction = "hausse"
    else:
        direction = "stable"

    result.update(
        {
            "ready": True,
            "slope_kg_week": slope_week,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "std_error_kg_week": se_slope * 7.0,
            "p_value": p_value,
            "r_squared": (1.0 - residual_ss / total_ss) if total_ss > 0 else float("nan"),
            "significant": significant,
            "direction": direction,
        }
    )
    return result


def trend_cone(
    df: pd.DataFrame,
    days_ahead: np.ndarray | Sequence[float],
    *,
    rate_window_days: int = LONG_RATE_WINDOW_DAYS,
    trend_window_days: int = TREND_WINDOW_DAYS,
) -> dict[str, Any]:
    """Poids de tendance prolongé à ``days_ahead`` jours, avec son cône à 95 %.

    Trois sources d'incertitude sont combinées en quadrature : l'erreur sur la
    pente (qui croît avec l'horizon), l'erreur sur le niveau de la tendance au
    dernier jour (σ / √(points de la fenêtre locale)) et le bruit quotidien
    d'une pesée. Sans pente estimable, la tendance est prolongée à plat avec le
    seul bruit : c'est le prolongement le plus prudent, et il reste évaluable.
    """
    days = np.asarray(list(days_ahead), dtype=float)
    frame = trend_weight(df, trend_window_days)
    result: dict[str, Any] = {
        "ready": False,
        "reason": None,
        "start": float("nan"),
        "last_date": None,
        "central": np.array([]),
        "low": np.array([]),
        "high": np.array([]),
        "rate": None,
        "noise": None,
        "has_bounds": False,
    }
    if frame.empty or len(days) == 0:
        result["reason"] = "aucune mesure"
        return result
    start = float(frame[TREND_COL].iloc[-1])
    last_date = frame[DATE_COL].iloc[-1]
    rate = rate_of_change(df, rate_window_days)
    noise = noise_level(df, trend_window_days)
    band = noise["band"] if noise["ready"] and np.isfinite(noise["band"]) else float("nan")
    local = frame[frame[DATE_COL] >= last_date - pd.Timedelta(days=int(trend_window_days))]
    level_se = (noise["sigma"] / np.sqrt(max(1, len(local)))) if noise["ready"] else float("nan")
    result.update({"start": start, "last_date": last_date, "rate": rate, "noise": noise})

    if not rate["ready"]:
        central = np.repeat(start, len(days))
        result["central"] = central
        if np.isfinite(band):
            half = np.sqrt(band**2 + (Z_BAND * level_se) ** 2) if np.isfinite(level_se) else band
            result.update({"low": central - half, "high": central + half, "has_bounds": True})
        result.update({"ready": True, "reason": rate["reason"]})
        return result

    slope_day = rate["slope_kg_week"] / 7.0
    se_day = rate["std_error_kg_week"] / 7.0
    t_crit = _t_quantile(max(1, rate["n"] - 2))
    central = start + slope_day * days
    slope_part = t_crit * se_day * days
    level_part = Z_BAND * level_se if np.isfinite(level_se) else 0.0
    noise_part = band if np.isfinite(band) else 0.0
    half = np.sqrt(slope_part**2 + level_part**2 + noise_part**2)
    result.update({"ready": True, "central": central, "low": central - half, "high": central + half, "has_bounds": True})
    return result


def project_trend(
    df: pd.DataFrame,
    horizon_days: int,
    *,
    rate_window_days: int = LONG_RATE_WINDOW_DAYS,
    trend_window_days: int = TREND_WINDOW_DAYS,
) -> pd.DataFrame:
    """Prolonge la tendance au rythme récent, avec un cône d'incertitude.

    Le point de départ est le poids de tendance (et non la dernière pesée, qui
    porte le bruit du jour). Ce n'est pas une prévision : c'est la réponse à
    « et si cela continuait ainsi ». Vide tant que la pente n'est pas estimable.
    """
    columns = [DATE_COL, "prevision", "borne_basse", "borne_haute"]
    if int(horizon_days) <= 0:
        return pd.DataFrame(columns=columns)
    days = np.arange(1, int(horizon_days) + 1, dtype=float)
    cone = trend_cone(df, days, rate_window_days=rate_window_days, trend_window_days=trend_window_days)
    if not cone["ready"] or cone["rate"] is None or not cone["rate"]["ready"] or not cone["has_bounds"]:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            DATE_COL: [cone["last_date"] + pd.Timedelta(days=int(d)) for d in days],
            "prevision": cone["central"],
            "borne_basse": cone["low"],
            "borne_haute": cone["high"],
        }
    )


def eta_to_target(
    df: pd.DataFrame,
    target_weight: float,
    *,
    rate_window_days: int = LONG_RATE_WINDOW_DAYS,
    trend_window_days: int = TREND_WINDOW_DAYS,
    max_days: int = 3 * 365,
) -> dict[str, Any]:
    """Date d'arrivée à la cible au rythme récent, encadrée par l'intervalle de la pente.

    ``eta_early`` suit la borne la plus rapide de l'intervalle, ``eta_late`` la
    plus lente ; cette dernière est absente quand l'intervalle contient une
    pente nulle ou montante, auquel cas la cible peut ne jamais être atteinte
    à ce rythme.
    """
    rate = rate_of_change(df, rate_window_days)
    frame = trend_weight(df, trend_window_days)
    result: dict[str, Any] = {
        "ready": False,
        "reason": None,
        "reached": False,
        "remaining_kg": float("nan"),
        "eta": None,
        "eta_early": None,
        "eta_late": None,
        "days": None,
        "rate": rate,
    }
    if frame.empty:
        result["reason"] = "aucune mesure"
        return result
    start = float(frame[TREND_COL].iloc[-1])
    last_date = frame[DATE_COL].iloc[-1]
    remaining = start - float(target_weight)
    result["remaining_kg"] = remaining
    if remaining <= 0:
        result.update({"ready": True, "reached": True, "eta": last_date, "days": 0})
        return result
    if not rate["ready"]:
        result["reason"] = rate["reason"]
        return result
    slope_day = rate["slope_kg_week"] / 7.0
    if slope_day >= 0 or not rate["significant"]:
        result["reason"] = "la pente récente ne se distingue pas d'une stagnation"
        return result

    days = remaining / abs(slope_day)
    if days > max_days:
        result["reason"] = "échéance au-delà de trois ans"
        return result
    result.update({"ready": True, "days": int(round(days)), "eta": last_date + pd.Timedelta(days=int(round(days)))})

    fast_day = rate["ci_low"] / 7.0
    slow_day = rate["ci_high"] / 7.0
    if fast_day < 0:
        early = remaining / abs(fast_day)
        result["eta_early"] = last_date + pd.Timedelta(days=int(round(early)))
    if slow_day < 0:
        late = remaining / abs(slow_day)
        if late <= max_days:
            result["eta_late"] = last_date + pd.Timedelta(days=int(round(late)))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Indice de masse corporelle
# ──────────────────────────────────────────────────────────────────────────────

# Classes de l'OMS pour l'adulte (https://www.who.int/europe/news-room/fact-sheets/item/a-healthy-lifestyle---who-recommendations).
BMI_CLASSES: tuple[tuple[float, str, str], ...] = (
    (18.5, "insuffisance pondérale", "warning"),
    (25.0, "corpulence normale", "success"),
    (30.0, "surpoids", "warning"),
    (35.0, "obésité (classe I)", "warning"),
    (40.0, "obésité (classe II)", "warning"),
    (float("inf"), "obésité (classe III)", "warning"),
)


def bmi_category(bmi: float) -> tuple[str, str]:
    """Libellé OMS de l'IMC et tonalité d'affichage. Repère de population, pas un diagnostic."""
    try:
        value = float(bmi)
    except (TypeError, ValueError):
        return ("indisponible", "info")
    if not np.isfinite(value) or value <= 0:
        return ("indisponible", "info")
    for upper, label, tone in BMI_CLASSES:
        if value < upper:
            return (label, tone)
    return (BMI_CLASSES[-1][1], BMI_CLASSES[-1][2])


def weight_for_bmi(bmi: float, height_m: float) -> float:
    """Poids correspondant à un IMC donné pour une taille donnée."""
    try:
        return float(bmi) * float(height_m) ** 2
    except (TypeError, ValueError):
        return float("nan")
