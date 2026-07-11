from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.business import (
    ALIGNMENT_TOLERANCE_KG,
    FINAL_TARGET_WEIGHT_KG,
    TARGET_TRAJECTORY_START_DATE,
    TARGET_TRAJECTORY_START_WEIGHT_KG,
    WEEKLY_LOSS_TARGET_KG,
)


DEFAULT_TARGET_TRAJECTORY_START_DATE = TARGET_TRAJECTORY_START_DATE
DEFAULT_TARGET_TRAJECTORY_START_WEIGHT = TARGET_TRAJECTORY_START_WEIGHT_KG
DEFAULT_WEEKLY_LOSS_TARGET = WEEKLY_LOSS_TARGET_KG
DEFAULT_FINAL_TARGET_WEIGHT = FINAL_TARGET_WEIGHT_KG

DATE_COL = "Date"
WEIGHT_COL = "Poids (Kgs)"


@dataclass(frozen=True)
class TargetTrajectoryConfig:
    """Business parameters for the main target trajectory."""

    start_date: pd.Timestamp = DEFAULT_TARGET_TRAJECTORY_START_DATE
    start_weight: float = DEFAULT_TARGET_TRAJECTORY_START_WEIGHT
    weekly_loss_target: float = DEFAULT_WEEKLY_LOSS_TARGET
    final_target_weight: float = DEFAULT_FINAL_TARGET_WEIGHT

    @classmethod
    def from_values(
        cls,
        start_date: Any = DEFAULT_TARGET_TRAJECTORY_START_DATE,
        weekly_loss_target: float = DEFAULT_WEEKLY_LOSS_TARGET,
        final_target_weight: float = DEFAULT_FINAL_TARGET_WEIGHT,
        *,
        start_weight: float = DEFAULT_TARGET_TRAJECTORY_START_WEIGHT,
    ) -> "TargetTrajectoryConfig":
        return cls(
            start_date=pd.Timestamp(start_date).normalize(),
            start_weight=float(start_weight),
            weekly_loss_target=float(weekly_loss_target),
            final_target_weight=float(final_target_weight),
        )


def _prepare_weight_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or DATE_COL not in df.columns or WEIGHT_COL not in df.columns:
        return pd.DataFrame(columns=[DATE_COL, WEIGHT_COL])

    data = df[[DATE_COL, WEIGHT_COL]].copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce").dt.normalize()
    data[WEIGHT_COL] = pd.to_numeric(data[WEIGHT_COL], errors="coerce")
    return data.dropna(subset=[DATE_COL, WEIGHT_COL]).sort_values(DATE_COL).reset_index(drop=True)


def fixed_start_measurement(_df: pd.DataFrame, config: TargetTrajectoryConfig) -> pd.Series:
    """Return the explicit reference point used to anchor the trajectory.

    The business rule intentionally does not infer this point from the data set:
    it must always be 26/05/2026 at 106.2 kg by default.
    """
    return pd.Series({DATE_COL: config.start_date, WEIGHT_COL: config.start_weight})


def build_target_trajectory(
    df: pd.DataFrame | None = None,
    config: TargetTrajectoryConfig | None = None,
) -> dict[str, Any]:
    """Build a capped target trajectory from a fixed date to the final target.

    The curve starts at the explicit business reference point in ``config``
    (26/05/2026 at 106.2 kg by default), descends at
    ``weekly_loss_target`` kg/week, and stops exactly when
    ``final_target_weight`` is reached. It never goes below the final target
    and never extends past it.
    """
    config = config or TargetTrajectoryConfig()
    if config.weekly_loss_target <= 0:
        return {"available": False, "message": "Le rythme cible doit être positif."}

    compat_dataframe_return = df is None
    if df is None:
        df = pd.DataFrame(columns=[DATE_COL, WEIGHT_COL])
    measurement = fixed_start_measurement(df, config)

    start_date = pd.Timestamp(config.start_date).normalize()
    start_weight = float(config.start_weight)
    final_target = float(config.final_target_weight)
    weekly_loss = float(config.weekly_loss_target)
    daily_loss = weekly_loss / 7.0

    if start_weight <= final_target:
        dates = pd.DatetimeIndex([start_date])
        values = [final_target]
        target_days = 0.0
        eta_date = start_date
    else:
        target_days = (start_weight - final_target) / daily_loss
        full_days = int(target_days)
        dates = pd.date_range(start_date, periods=full_days + 1, freq="D")
        values = [max(start_weight - daily_loss * day, final_target) for day in range(full_days + 1)]

        eta_date = start_date + pd.Timedelta(days=target_days)
        if target_days > full_days:
            dates = dates.append(pd.DatetimeIndex([eta_date]))
            values.append(final_target)
        else:
            values[-1] = final_target

    trajectory = pd.DataFrame({DATE_COL: dates, "Poids cible (kg)": values})
    if compat_dataframe_return:
        return trajectory.rename(columns={"Poids cible (kg)": "Poids cible"})
    return {
        "available": True,
        "config": config,
        "trajectory": trajectory,
        "start_date": start_date,
        "start_weight": start_weight,
        "start_measurement_date": pd.Timestamp(measurement[DATE_COL]),
        "weekly_loss_target": weekly_loss,
        "final_target_weight": final_target,
        "target_days": float(target_days),
        "eta_date": eta_date,
    }


def target_weight_on_date(start_weight: float, date: pd.Timestamp, config: TargetTrajectoryConfig) -> float | None:
    """Return the scheduled target weight for a date, capped at final target.

    No trajectory value exists before the fixed business start date.
    """
    start_date = pd.Timestamp(config.start_date).normalize()
    current_date = pd.Timestamp(date).normalize()
    if current_date < start_date:
        return None
    elapsed_days = (current_date - start_date).total_seconds() / 86400
    scheduled = float(start_weight) - (elapsed_days / 7.0 * config.weekly_loss_target)
    return max(scheduled, config.final_target_weight)


def compare_to_target_trajectory(
    df: pd.DataFrame,
    config: TargetTrajectoryConfig | None = None,
) -> dict[str, Any]:
    """Compare the latest measured weight with the corrected target trajectory."""
    config = config or TargetTrajectoryConfig()
    trajectory = build_target_trajectory(df, config)
    if not trajectory.get("available"):
        return trajectory

    data = _prepare_weight_data(df)
    if data.empty:
        return {"available": False, "message": "Aucune mesure disponible."}

    latest = data.iloc[-1]
    current_date = pd.Timestamp(latest[DATE_COL])
    current_weight = float(latest[WEIGHT_COL])
    scheduled_weight = target_weight_on_date(float(trajectory["start_weight"]), current_date, config)
    if scheduled_weight is None:
        return {**trajectory, "available": False, "message": "Aucune trajectoire avant le 26/05/2026."}
    gap_kg = current_weight - scheduled_weight

    start_weight = float(trajectory["start_weight"])
    final_target = float(config.final_target_weight)
    total_to_lose = max(start_weight - final_target, 0.0)
    progress_pct = 100.0 if total_to_lose == 0 else (start_weight - current_weight) / total_to_lose * 100
    progress_pct = max(0.0, min(100.0, progress_pct))

    days_delta = gap_kg / (config.weekly_loss_target / 7.0) if config.weekly_loss_target > 0 else None
    if gap_kg > ALIGNMENT_TOLERANCE_KG:
        status = "au-dessus de la trajectoire"
    elif gap_kg < -ALIGNMENT_TOLERANCE_KG:
        status = "en dessous de la trajectoire"
    else:
        status = "aligné avec la trajectoire"

    return {
        **trajectory,
        "current_date": current_date,
        "current_weight": current_weight,
        "scheduled_weight": scheduled_weight,
        "gap_kg": gap_kg,
        "days_delta": days_delta,
        "status": status,
        "alignment_tolerance_kg": ALIGNMENT_TOLERANCE_KG,
        "progress_pct": progress_pct,
    }
