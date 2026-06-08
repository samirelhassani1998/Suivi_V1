"""Synthèses robustes et lisibles pour le suivi de poids quotidien."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

WEIGHT_COL = "Poids (Kgs)"
DATE_COL = "Date"


@dataclass(frozen=True)
class PeriodDelta:
    """Variation entre la dernière mesure et une référence passée."""

    label: str
    days: int
    value: float | None
    reference_date: pd.Timestamp | None
    measurements: int
    reason: str | None = None


def _empty_period(label: str, days: int, reason: str) -> PeriodDelta:
    return PeriodDelta(label=label, days=days, value=None, reference_date=None, measurements=0, reason=reason)


def prepare_weight_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a minimal, sorted and valid dataframe for analytics.

    This function intentionally does not replace the import cleaner. It is a
    final defensive layer for dashboard calculations so malformed in-session
    edits do not produce misleading metrics or Streamlit errors.
    """
    if df is None or df.empty or DATE_COL not in df.columns or WEIGHT_COL not in df.columns:
        return pd.DataFrame(columns=[DATE_COL, WEIGHT_COL])

    out = df[[DATE_COL, WEIGHT_COL]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", dayfirst=True)
    out[WEIGHT_COL] = pd.to_numeric(
        out[WEIGHT_COL].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    out = out.dropna(subset=[DATE_COL, WEIGHT_COL])
    out = out[out[WEIGHT_COL] > 0]
    if out.empty:
        return out.reset_index(drop=True)

    # Keep the last edited measure for a duplicated day, matching the app import policy.
    out = out.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL], keep="last")
    return out.reset_index(drop=True)


def format_fr_number(value: float, decimals: int = 1, *, trim_zeros: bool = True) -> str:
    formatted = f"{float(value):.{decimals}f}"
    if trim_zeros:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted.replace(".", ",")


def format_fr_kg(value: float, decimals: int = 1, *, trim_zeros: bool = True) -> str:
    return f"{format_fr_number(value, decimals=decimals, trim_zeros=trim_zeros)} kg"


def _format_delta_text(delta: float) -> str:
    if delta < 0:
        return f"baisse de {format_fr_kg(abs(delta))}"
    if delta > 0:
        return f"hausse de {format_fr_kg(delta)}"
    return "variation stable"


def _current_tracking_period(df: pd.DataFrame, gap_threshold_days: int = 21) -> pd.DataFrame:
    data = prepare_weight_series(df)
    if len(data) < 2:
        return data
    gaps = data[DATE_COL].diff().dt.days.fillna(0)
    restart_positions = gaps[gaps > gap_threshold_days].index
    if len(restart_positions) == 0:
        return data
    return data.iloc[int(restart_positions[-1]) :].reset_index(drop=True)


def moving_average_by_days(df: pd.DataFrame, days: int) -> pd.Series:
    """Calendar-aware moving average, robust to irregular measurements."""
    if df.empty:
        return pd.Series(dtype=float)
    ordered = prepare_weight_series(df)
    if ordered.empty:
        return pd.Series(dtype=float)

    indexed = ordered.set_index(DATE_COL)[WEIGHT_COL]
    return indexed.rolling(f"{int(days)}D", min_periods=1).mean().reset_index(drop=True)


def delta_since_days(df: pd.DataFrame, days: int, min_measurements: int = 2) -> PeriodDelta:
    """Compute current weight minus the closest available measure near a past cutoff."""
    data = prepare_weight_series(df)
    label = f"{days} jours"
    if len(data) < min_measurements:
        return _empty_period(label, days, "pas assez de mesures")

    current_date = data[DATE_COL].iloc[-1]
    current_weight = float(data[WEIGHT_COL].iloc[-1])
    cutoff = current_date - pd.Timedelta(days=days)
    window = data[data[DATE_COL] >= cutoff]
    if len(window) < min_measurements:
        return _empty_period(label, days, f"moins de {min_measurements} mesures sur la période")

    reference = window.iloc[0]
    actual_span = max((current_date - reference[DATE_COL]).days, 0)
    if actual_span < max(1, days // 2):
        return PeriodDelta(
            label=label,
            days=days,
            value=None,
            reference_date=reference[DATE_COL],
            measurements=len(window),
            reason="recul historique insuffisant",
        )

    return PeriodDelta(
        label=label,
        days=days,
        value=current_weight - float(reference[WEIGHT_COL]),
        reference_date=reference[DATE_COL],
        measurements=len(window),
    )


def classify_trend(delta_30d: PeriodDelta, delta_7d: PeriodDelta, threshold: float = 0.25) -> tuple[str, str]:
    """Return a compact trend label and a simple explanation."""
    source = delta_30d if delta_30d.value is not None else delta_7d
    if source.value is None:
        return "À confirmer", "Ajoutez quelques mesures pour fiabiliser la tendance."

    if source.value <= -threshold:
        return "Baisse", f"Le poids est en baisse sur les {source.label} disponibles."
    if source.value >= threshold:
        return "Hausse", f"Le poids est en hausse sur les {source.label} disponibles."
    return "Stable", f"La variation sur les {source.label} reste faible."


def average_weekly_pace(df: pd.DataFrame) -> float | None:
    """Average kg/week pace from first to last measurement."""
    data = prepare_weight_series(df)
    if len(data) < 2:
        return None
    days = (data[DATE_COL].iloc[-1] - data[DATE_COL].iloc[0]).days
    if days < 7:
        return None
    delta = float(data[WEIGHT_COL].iloc[-1] - data[WEIGHT_COL].iloc[0])
    return delta / days * 7


def projection_to_target(df: pd.DataFrame, target_weight: float, max_years: int = 3) -> dict[str, Any]:
    """Prudent linear projection based on the recent 30-day trend.

    The projection is intentionally conservative: it is hidden when data are too
    sparse, when the recent slope goes away from the goal, or when the result is
    unrealistically far in the future.
    """
    data = prepare_weight_series(df)
    if len(data) < 5:
        return {"available": False, "message": "Projection disponible après au moins 5 mesures."}

    current = float(data[WEIGHT_COL].iloc[-1])
    last_date = data[DATE_COL].iloc[-1]
    if math.isclose(current, target_weight, abs_tol=0.1) or current <= target_weight:
        return {"available": True, "reached": True, "message": "Objectif atteint ou très proche."}

    recent = data[data[DATE_COL] >= last_date - pd.Timedelta(days=30)]
    if len(recent) < 5:
        recent = data.tail(min(len(data), 14))
    span_days = (recent[DATE_COL].iloc[-1] - recent[DATE_COL].iloc[0]).days
    if len(recent) < 5 or span_days < 7:
        return {"available": False, "message": "Projection non affichée : recul récent insuffisant."}

    x = (recent[DATE_COL] - recent[DATE_COL].iloc[0]).dt.days.to_numpy(dtype=float)
    y = recent[WEIGHT_COL].to_numpy(dtype=float)
    slope_per_day = float(np.polyfit(x, y, 1)[0])
    if slope_per_day >= -0.01:
        return {"available": False, "message": "Projection non affichée : la tendance récente ne va pas clairement vers l'objectif."}

    days_needed = int((current - target_weight) / abs(slope_per_day))
    if days_needed > max_years * 365:
        return {"available": False, "message": "Projection trop lointaine pour être fiable."}

    return {
        "available": True,
        "reached": False,
        "eta": last_date + pd.Timedelta(days=max(days_needed, 0)),
        "days_needed": max(days_needed, 0),
        "pace_kg_week": slope_per_day * 7,
        "message": "Projection indicative basée sur la tendance récente, à interpréter prudemment.",
    }


def detect_stagnation_periods(df: pd.DataFrame, window_days: int = 14, tolerance_kg: float = 0.4) -> list[dict[str, Any]]:
    """Detect recent calendar windows where the amplitude remains small."""
    data = prepare_weight_series(df)
    if len(data) < 4:
        return []

    periods: list[dict[str, Any]] = []
    start_idx = 0
    for idx in range(len(data)):
        while (data[DATE_COL].iloc[idx] - data[DATE_COL].iloc[start_idx]).days > window_days:
            start_idx += 1
        window = data.iloc[start_idx : idx + 1]
        if len(window) >= 4:
            amplitude = float(window[WEIGHT_COL].max() - window[WEIGHT_COL].min())
            span = int((window[DATE_COL].iloc[-1] - window[DATE_COL].iloc[0]).days)
            if span >= window_days - 2 and amplitude <= tolerance_kg:
                candidate = {
                    "start": window[DATE_COL].iloc[0],
                    "end": window[DATE_COL].iloc[-1],
                    "days": span,
                    "amplitude": amplitude,
                    "measurements": len(window),
                }
                if not periods or candidate["start"] > periods[-1]["end"]:
                    periods.append(candidate)
                else:
                    periods[-1] = candidate
    return periods[-3:]


def generate_daily_insights(df: pd.DataFrame, target_weight: float) -> list[str]:
    """Generate short, non-alarmist insights for the dashboard."""
    data = prepare_weight_series(df)
    if data.empty:
        return ["Aucune mesure valide n’est disponible pour générer des insights."]
    if len(data) == 1:
        return ["Ajoutez au moins une deuxième mesure pour calculer les variations."]

    current = float(data[WEIGHT_COL].iloc[-1])
    delta_7 = delta_since_days(data, 7)
    delta_30 = delta_since_days(data, 30)
    trend, explanation = classify_trend(delta_30, delta_7)
    target_weight = float(target_weight)

    insights: list[str] = []
    recent_period = _current_tracking_period(data)
    if len(recent_period) >= 2:
        recent_delta = float(recent_period[WEIGHT_COL].iloc[-1] - recent_period[WEIGHT_COL].iloc[0])
        recent_days = int((recent_period[DATE_COL].iloc[-1] - recent_period[DATE_COL].iloc[0]).days)
        start_str = recent_period[DATE_COL].iloc[0].strftime("%d/%m/%Y")
        if recent_period[DATE_COL].iloc[0] > data[DATE_COL].iloc[0]:
            insights.append(
                f"Depuis la reprise du {start_str}, votre poids affiche une {_format_delta_text(recent_delta)} "
                f"en {recent_days} jours."
            )

    if not insights:
        if delta_7.value is not None:
            insights.append(f"Sur 7 jours, votre poids affiche une {_format_delta_text(delta_7.value)}.")
        else:
            insights.append(explanation)

    if delta_30.value is not None:
        if delta_30.value < -0.3:
            insights.append(f"Sur 30 jours, la tendance reste orientée à la baisse ({format_fr_kg(abs(delta_30.value))}).")
        elif delta_30.value > 0.3:
            insights.append(f"Sur 30 jours, les dernières mesures montrent une hausse de {format_fr_kg(delta_30.value)}.")
        else:
            insights.append("Sur 30 jours, votre poids reste globalement stable.")
    elif delta_7.value is not None and abs(delta_7.value) <= 0.25:
        insights.append("Sur 7 jours, la variation reste faible : continuez le suivi pour confirmer la tendance.")

    next_step = math.floor(current / 5) * 5
    if current > next_step and next_step > target_weight:
        insights.append(f"Prochain palier : passer sous {format_fr_kg(next_step, trim_zeros=True)}.")

    gap = current - target_weight
    if gap > 0:
        insights.append(f"Il reste {format_fr_kg(gap)} avant l’objectif final configuré.")
    else:
        insights.append("L’objectif final configuré est atteint ou dépassé.")

    has_significant_drop = (delta_7.value is not None and delta_7.value <= -0.3) or (
        delta_30.value is not None and delta_30.value <= -0.5
    )
    periods = detect_stagnation_periods(data)
    if periods and trend != "Baisse" and not has_significant_drop:
        last = periods[-1]
        insights.append(
            f"Stabilité possible : {last['days']} jours avec une amplitude limitée à {format_fr_kg(last['amplitude'])}."
        )

    if trend == "À confirmer":
        insights.append("Les calculs restent prudents car l’historique récent est encore limité.")
    return insights[:5]


def summarize_weight_journey(df: pd.DataFrame, target_weight: float) -> dict[str, Any]:
    """Build all daily dashboard metrics in one maintainable place."""
    data = prepare_weight_series(df)
    if data.empty:
        return {"valid": False, "message": "Aucune donnée de poids valide."}

    current = float(data[WEIGHT_COL].iloc[-1])
    previous = float(data[WEIGHT_COL].iloc[-2]) if len(data) > 1 else None
    first = float(data[WEIGHT_COL].iloc[0])
    delta_7 = delta_since_days(data, 7)
    delta_30 = delta_since_days(data, 30)
    delta_90 = delta_since_days(data, 90)
    ma_7 = moving_average_by_days(data, 7)
    ma_30 = moving_average_by_days(data, 30)
    trend_label, trend_explanation = classify_trend(delta_30, delta_7)
    pace = average_weekly_pace(data)
    projection = projection_to_target(data, target_weight)
    target_weight = float(target_weight)
    target_gap = current - target_weight
    total_to_goal = first - target_weight
    if total_to_goal > 0:
        progress_pct = (first - current) / total_to_goal * 100
    else:
        progress_pct = 100.0 if current <= target_weight else 0.0
    progress_pct = float(max(0.0, min(100.0, progress_pct)))
    days_tracked = int((data[DATE_COL].iloc[-1] - data[DATE_COL].iloc[0]).days)

    return {
        "valid": True,
        "rows": len(data),
        "start_date": data[DATE_COL].iloc[0],
        "last_date": data[DATE_COL].iloc[-1],
        "current": current,
        "previous_delta": None if previous is None else current - previous,
        "delta_start": current - first,
        "delta_7": delta_7,
        "delta_30": delta_30,
        "delta_90": delta_90,
        "ma_7_current": float(ma_7.iloc[-1]) if not ma_7.empty else None,
        "ma_30_current": float(ma_30.iloc[-1]) if not ma_30.empty else None,
        "min_weight": float(data[WEIGHT_COL].min()),
        "max_weight": float(data[WEIGHT_COL].max()),
        "best_weight": float(data[WEIGHT_COL].min()),
        "target_gap": target_gap,
        "target_remaining": max(target_gap, 0.0),
        "target_progress_pct": progress_pct,
        "days_tracked": days_tracked,
        "measurements": len(data),
        "trend_label": trend_label,
        "trend_explanation": trend_explanation,
        "weekly_pace": pace,
        "projection": projection,
        "insights": generate_daily_insights(data, target_weight),
        "stagnations": detect_stagnation_periods(data),
    }
