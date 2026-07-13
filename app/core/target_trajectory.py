from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.business import (
    ALIGNMENT_TOLERANCE_KG,
    FINAL_TARGET_WEIGHT_KG,
    TARGET_TRAJECTORY_END_DATE,
    TARGET_TRAJECTORY_START_DATE,
    TARGET_TRAJECTORY_START_WEIGHT_KG,
    TARGET_TRAJECTORY_TOTAL_DURATION_DAYS,
)
from app.core.data import prepare_analysis_data
from app.core.time_utils import normalize_datetime_series

DEFAULT_TARGET_TRAJECTORY_START_DATE = TARGET_TRAJECTORY_START_DATE
DEFAULT_TARGET_TRAJECTORY_END_DATE = TARGET_TRAJECTORY_END_DATE
DEFAULT_TARGET_TRAJECTORY_START_WEIGHT = TARGET_TRAJECTORY_START_WEIGHT_KG
DEFAULT_FINAL_TARGET_WEIGHT = FINAL_TARGET_WEIGHT_KG
DEFAULT_TOTAL_DURATION_DAYS = TARGET_TRAJECTORY_TOTAL_DURATION_DAYS

DATE_COL = "Date"
WEIGHT_COL = "Poids (Kgs)"
TARGET_WEIGHT_COL = "Poids cible (kg)"
COMPAT_TARGET_WEIGHT_COL = "Poids cible"
TRAJECTORY_COMPLETED_MESSAGE = "La trajectoire cible s’est terminée le 11/11/2026."


@dataclass(frozen=True)
class TargetTrajectoryConfig:
    """Fixed business contract for the main target trajectory.

    The target line is always anchored at 106.1 kg on 2026-07-12 and reaches
    80.0 kg on 2026-11-11. Source measurements are never used to alter the
    start point, slope, duration, or final target.
    """

    start_date: pd.Timestamp = pd.Timestamp("2026-07-12")
    end_date: pd.Timestamp = pd.Timestamp("2026-11-11")
    start_weight: float = 106.1
    final_target_weight: float = 80.0

    @property
    def required_weekly_loss(self) -> float:
        total_days = (self.end_date - self.start_date).days
        return (self.start_weight - self.final_target_weight) / (total_days / 7)

    @classmethod
    def from_values(
        cls,
        start_date: Any = DEFAULT_TARGET_TRAJECTORY_START_DATE,
        weekly_loss_target: float | None = None,
        final_target_weight: float = DEFAULT_FINAL_TARGET_WEIGHT,
        *,
        start_weight: float = DEFAULT_TARGET_TRAJECTORY_START_WEIGHT,
        end_date: Any = DEFAULT_TARGET_TRAJECTORY_END_DATE,
        duplicate_strategy: str = "garder_la_derniere",
    ) -> "TargetTrajectoryConfig":
        """Return the fixed config; deprecated arguments are ignored.

        ``weekly_loss_target`` and duplicate/source-data parameters are retained
        only for call-site compatibility and never influence the calculation.
        """
        return cls()


def _prepare_weight_data(df: pd.DataFrame, duplicate_strategy: str = "garder_la_derniere") -> pd.DataFrame:
    if df.empty or DATE_COL not in df.columns or WEIGHT_COL not in df.columns:
        return pd.DataFrame(columns=[DATE_COL, WEIGHT_COL])
    data = prepare_analysis_data(df.copy(deep=True), duplicate_strategy)
    data[DATE_COL] = normalize_datetime_series(data[DATE_COL], dayfirst=True, normalize_day=True)
    data[WEIGHT_COL] = pd.to_numeric(data[WEIGHT_COL], errors="coerce")
    return data.dropna(subset=[DATE_COL, WEIGHT_COL]).sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)


def resolve_start_measurement(df: pd.DataFrame, config: TargetTrajectoryConfig) -> tuple[float, pd.Timestamp, str]:
    """Compatibility helper returning the fixed business anchor."""
    return float(config.start_weight), pd.Timestamp(config.start_date).normalize(), "fixed_business_rule"


def required_weekly_loss(
    start_weight: float = DEFAULT_TARGET_TRAJECTORY_START_WEIGHT,
    final_target_weight: float = DEFAULT_FINAL_TARGET_WEIGHT,
    total_duration_days: int = DEFAULT_TOTAL_DURATION_DAYS,
) -> float:
    """Return the required pace derived from the fixed trajectory contract."""
    return (DEFAULT_TARGET_TRAJECTORY_START_WEIGHT - DEFAULT_FINAL_TARGET_WEIGHT) / (DEFAULT_TOTAL_DURATION_DAYS / 7.0)


def required_daily_loss() -> float:
    """Return the theoretical daily loss; negative days_delta means ahead."""
    return (DEFAULT_TARGET_TRAJECTORY_START_WEIGHT - DEFAULT_FINAL_TARGET_WEIGHT) / DEFAULT_TOTAL_DURATION_DAYS


def target_weight_on_date(
    date: pd.Timestamp,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    start_weight: float | None = None,
    final_weight: float | None = None,
) -> float | None:
    """Return the fixed scheduled weight for a date, or None outside bounds."""
    current = normalize_datetime_series([date], dayfirst=True, normalize_day=True).iloc[0]
    start = DEFAULT_TARGET_TRAJECTORY_START_DATE
    end = DEFAULT_TARGET_TRAJECTORY_END_DATE
    if pd.isna(current) or current < start or current > end:
        return None
    if current == start:
        return float(DEFAULT_TARGET_TRAJECTORY_START_WEIGHT)
    if current == end:
        return float(DEFAULT_FINAL_TARGET_WEIGHT)
    total_seconds = (end - start).total_seconds()
    elapsed_seconds = (current - start).total_seconds()
    progress = elapsed_seconds / total_seconds
    scheduled = DEFAULT_TARGET_TRAJECTORY_START_WEIGHT + progress * (DEFAULT_FINAL_TARGET_WEIGHT - DEFAULT_TARGET_TRAJECTORY_START_WEIGHT)
    return max(float(scheduled), float(DEFAULT_FINAL_TARGET_WEIGHT))


def _fixed_trajectory_dataframe() -> pd.DataFrame:
    dates = pd.date_range(start=DEFAULT_TARGET_TRAJECTORY_START_DATE, end=DEFAULT_TARGET_TRAJECTORY_END_DATE, freq="D")
    values = [target_weight_on_date(date) for date in dates]
    trajectory = pd.DataFrame({DATE_COL: dates, TARGET_WEIGHT_COL: values})
    trajectory.loc[trajectory.index[0], DATE_COL] = DEFAULT_TARGET_TRAJECTORY_START_DATE
    trajectory.loc[trajectory.index[0], TARGET_WEIGHT_COL] = DEFAULT_TARGET_TRAJECTORY_START_WEIGHT
    trajectory.loc[trajectory.index[-1], DATE_COL] = DEFAULT_TARGET_TRAJECTORY_END_DATE
    trajectory.loc[trajectory.index[-1], TARGET_WEIGHT_COL] = DEFAULT_FINAL_TARGET_WEIGHT
    return trajectory


def build_target_trajectory(df: pd.DataFrame | None = None, config: TargetTrajectoryConfig | None = None) -> dict[str, Any] | pd.DataFrame:
    """Build the fixed target trajectory.

    ``df`` is accepted for backward compatibility with existing call sites, but
    source data never changes the fixed anchor, slope, duration, or endpoint.
    """
    config = config or TargetTrajectoryConfig()
    trajectory = _fixed_trajectory_dataframe()
    weekly_loss = required_weekly_loss()
    result = {
        "available": True,
        "config": config,
        "trajectory": trajectory,
        "start_date": DEFAULT_TARGET_TRAJECTORY_START_DATE,
        "end_date": DEFAULT_TARGET_TRAJECTORY_END_DATE,
        "start_weight": DEFAULT_TARGET_TRAJECTORY_START_WEIGHT,
        "start_measurement_date": DEFAULT_TARGET_TRAJECTORY_START_DATE,
        "start_weight_source": "fixed_business_rule",
        "weekly_loss_target": weekly_loss,
        "required_weekly_loss": weekly_loss,
        "required_daily_loss": required_daily_loss(),
        "final_target_weight": DEFAULT_FINAL_TARGET_WEIGHT,
        "target_days": float(DEFAULT_TOTAL_DURATION_DAYS),
        "total_duration_days": DEFAULT_TOTAL_DURATION_DAYS,
        "eta_date": DEFAULT_TARGET_TRAJECTORY_END_DATE,
    }
    return result


def compare_to_target_trajectory(df: pd.DataFrame, config: TargetTrajectoryConfig | None = None) -> dict[str, Any]:
    config = config or TargetTrajectoryConfig()
    trajectory = build_target_trajectory(df, config)
    data = _prepare_weight_data(df)
    if data.empty:
        return {**trajectory, "available": False, "message": "Aucune mesure disponible."}
    latest = data.iloc[-1]
    current_date = pd.Timestamp(latest[DATE_COL])
    current_weight = float(latest[WEIGHT_COL])
    scheduled_weight = target_weight_on_date(current_date)
    if scheduled_weight is None:
        if current_date > DEFAULT_TARGET_TRAJECTORY_END_DATE:
            return {**trajectory, "available": False, "trajectory_completed": True, "current_date": current_date, "current_weight": current_weight, "message": TRAJECTORY_COMPLETED_MESSAGE}
        return {**trajectory, "available": False, "current_date": current_date, "current_weight": current_weight, "message": "Aucune trajectoire cible disponible pour cette date."}
    gap_kg = current_weight - scheduled_weight
    total_to_lose = DEFAULT_TARGET_TRAJECTORY_START_WEIGHT - DEFAULT_FINAL_TARGET_WEIGHT
    progress_pct = (DEFAULT_TARGET_TRAJECTORY_START_WEIGHT - current_weight) / total_to_lose * 100
    progress_pct = max(0.0, min(100.0, progress_pct))
    daily_loss = required_daily_loss()
    days_delta = gap_kg / daily_loss if daily_loss else None
    if gap_kg < -ALIGNMENT_TOLERANCE_KG:
        status = "en avance"
    elif gap_kg > ALIGNMENT_TOLERANCE_KG:
        status = "en retard"
    else:
        status = "aligné"
    return {**trajectory, "current_date": current_date, "current_weight": current_weight, "scheduled_weight": scheduled_weight, "gap_kg": gap_kg, "days_delta": days_delta, "status": status, "alignment_tolerance_kg": ALIGNMENT_TOLERANCE_KG, "progress_pct": progress_pct}
