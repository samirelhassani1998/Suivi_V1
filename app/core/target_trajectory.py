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

DEFAULT_TARGET_TRAJECTORY_START_DATE = TARGET_TRAJECTORY_START_DATE
DEFAULT_TARGET_TRAJECTORY_END_DATE = TARGET_TRAJECTORY_END_DATE
DEFAULT_TARGET_TRAJECTORY_START_WEIGHT = TARGET_TRAJECTORY_START_WEIGHT_KG
DEFAULT_FINAL_TARGET_WEIGHT = FINAL_TARGET_WEIGHT_KG
DEFAULT_TOTAL_DURATION_DAYS = TARGET_TRAJECTORY_TOTAL_DURATION_DAYS

DATE_COL = "Date"
WEIGHT_COL = "Poids (Kgs)"
TARGET_WEIGHT_COL = "Poids cible (kg)"
COMPAT_TARGET_WEIGHT_COL = "Poids cible"


@dataclass(frozen=True)
class TargetTrajectoryConfig:
    """Business parameters for the main target trajectory."""

    start_date: pd.Timestamp = DEFAULT_TARGET_TRAJECTORY_START_DATE
    end_date: pd.Timestamp = DEFAULT_TARGET_TRAJECTORY_END_DATE
    start_weight: float = DEFAULT_TARGET_TRAJECTORY_START_WEIGHT
    final_target_weight: float = DEFAULT_FINAL_TARGET_WEIGHT
    duplicate_strategy: str = "garder_la_derniere"

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
        # weekly_loss_target is kept only as a backward-compatible ignored argument:
        # the required pace is derived from start/end weights and dates.
        return cls(
            start_date=pd.Timestamp(start_date).normalize(),
            end_date=pd.Timestamp(end_date).normalize(),
            start_weight=float(start_weight),
            final_target_weight=float(final_target_weight),
            duplicate_strategy=duplicate_strategy,
        )


def _prepare_weight_data(df: pd.DataFrame, duplicate_strategy: str = "garder_la_derniere") -> pd.DataFrame:
    if df.empty or DATE_COL not in df.columns or WEIGHT_COL not in df.columns:
        return pd.DataFrame(columns=[DATE_COL, WEIGHT_COL])
    data = prepare_analysis_data(df.copy(deep=True), duplicate_strategy)
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce").dt.normalize()
    data[WEIGHT_COL] = pd.to_numeric(data[WEIGHT_COL], errors="coerce")
    return data.dropna(subset=[DATE_COL, WEIGHT_COL]).sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)


def resolve_start_measurement(df: pd.DataFrame, config: TargetTrajectoryConfig) -> tuple[float, pd.Timestamp, str]:
    """Resolve the configured start-date reference without mutating source data."""
    data = _prepare_weight_data(df, config.duplicate_strategy)
    start_date = pd.Timestamp(config.start_date).normalize()
    if not data.empty:
        on_start = data[data[DATE_COL] == start_date]
        if not on_start.empty:
            row = on_start.iloc[-1]
            return float(row[WEIGHT_COL]), pd.Timestamp(row[DATE_COL]), "exact"
        previous = data[data[DATE_COL] <= start_date]
        if not previous.empty:
            row = previous.iloc[-1]
            return float(row[WEIGHT_COL]), pd.Timestamp(row[DATE_COL]), "previous"
    return float(config.start_weight), start_date, "configured"


def required_weekly_loss(start_weight: float, final_target_weight: float = DEFAULT_FINAL_TARGET_WEIGHT, total_duration_days: int = DEFAULT_TOTAL_DURATION_DAYS) -> float:
    return (float(start_weight) - float(final_target_weight)) / (float(total_duration_days) / 7.0)


def target_weight_on_date(
    date: pd.Timestamp,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    start_weight: float,
    final_weight: float,
) -> float | None:
    current = pd.Timestamp(date).normalize()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if current < start or current > end:
        return None
    if current == end:
        return float(final_weight)
    total_seconds = int((end - start).total_seconds())
    elapsed_seconds = int((current - start).total_seconds())
    progress = elapsed_seconds / total_seconds if total_seconds else 1.0
    scheduled = float(start_weight) + progress * (float(final_weight) - float(start_weight))
    return max(scheduled, float(final_weight))


def build_target_trajectory(df: pd.DataFrame | None = None, config: TargetTrajectoryConfig | None = None) -> dict[str, Any] | pd.DataFrame:
    config = config or TargetTrajectoryConfig()
    compat_dataframe_return = df is None
    source = pd.DataFrame(columns=[DATE_COL, WEIGHT_COL]) if df is None else df
    start_date = pd.Timestamp(config.start_date).normalize()
    end_date = pd.Timestamp(config.end_date).normalize()
    final_target = float(config.final_target_weight)
    start_weight, measurement_date, source_kind = resolve_start_measurement(source, config)
    total_duration_days = int((end_date - start_date).days)
    weekly_loss = required_weekly_loss(start_weight, final_target, total_duration_days)
    dates = pd.date_range(start_date, end_date, freq="D")
    values = [target_weight_on_date(date, start_date, end_date, start_weight, final_target) for date in dates]
    trajectory = pd.DataFrame({DATE_COL: dates, TARGET_WEIGHT_COL: values})
    trajectory.iloc[-1] = {DATE_COL: end_date, TARGET_WEIGHT_COL: final_target}
    if compat_dataframe_return:
        return trajectory.rename(columns={TARGET_WEIGHT_COL: COMPAT_TARGET_WEIGHT_COL})
    return {
        "available": True,
        "config": config,
        "trajectory": trajectory,
        "start_date": start_date,
        "end_date": end_date,
        "start_weight": start_weight,
        "start_measurement_date": measurement_date,
        "start_weight_source": source_kind,
        "weekly_loss_target": weekly_loss,
        "required_weekly_loss": weekly_loss,
        "final_target_weight": final_target,
        "target_days": float(total_duration_days),
        "total_duration_days": total_duration_days,
        "eta_date": end_date,
    }


def compare_to_target_trajectory(df: pd.DataFrame, config: TargetTrajectoryConfig | None = None) -> dict[str, Any]:
    config = config or TargetTrajectoryConfig()
    trajectory = build_target_trajectory(df, config)
    if not trajectory.get("available"):
        return trajectory
    data = _prepare_weight_data(df, config.duplicate_strategy)
    if data.empty:
        return {**trajectory, "available": False, "message": "Aucune mesure disponible."}
    latest = data.iloc[-1]
    current_date = pd.Timestamp(latest[DATE_COL])
    current_weight = float(latest[WEIGHT_COL])
    scheduled_weight = target_weight_on_date(current_date, trajectory["start_date"], trajectory["end_date"], trajectory["start_weight"], trajectory["final_target_weight"])
    if scheduled_weight is None:
        return {**trajectory, "available": False, "message": "Aucune trajectoire cible disponible pour cette date."}
    gap_kg = current_weight - scheduled_weight
    total_to_lose = max(float(trajectory["start_weight"]) - float(trajectory["final_target_weight"]), 0.0)
    progress_pct = 100.0 if total_to_lose == 0 else (float(trajectory["start_weight"]) - current_weight) / total_to_lose * 100
    progress_pct = max(0.0, min(100.0, progress_pct))
    daily_loss = trajectory["required_weekly_loss"] / 7.0 if trajectory["required_weekly_loss"] else 0.0
    days_delta = gap_kg / daily_loss if daily_loss else None
    if gap_kg < -ALIGNMENT_TOLERANCE_KG:
        status = "en avance"
    elif gap_kg > ALIGNMENT_TOLERANCE_KG:
        status = "en retard"
    else:
        status = "aligné"
    return {**trajectory, "current_date": current_date, "current_weight": current_weight, "scheduled_weight": scheduled_weight, "gap_kg": gap_kg, "days_delta": days_delta, "status": status, "alignment_tolerance_kg": ALIGNMENT_TOLERANCE_KG, "progress_pct": progress_pct}
