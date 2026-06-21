from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

TARGET_TRAJECTORY_START_DATE: Final[pd.Timestamp] = pd.Timestamp("2026-05-26")
TARGET_TRAJECTORY_START_WEIGHT_KG: Final[float] = 106.2
WEEKLY_LOSS_TARGET_KG: Final[float] = 1.0
FINAL_TARGET_WEIGHT_KG: Final[float] = 80.0
ALIGNMENT_TOLERANCE_KG: Final[float] = 0.5
INTERRUPTION_THRESHOLD_DAYS: Final[int] = 21
MIN_MEASUREMENTS_REQUIRED: Final[int] = 2

STAGNATION_WINDOW_DAYS: Final[int] = 14
STAGNATION_MIN_MEASUREMENTS: Final[int] = 4
STAGNATION_MAX_AMPLITUDE_KG: Final[float] = 0.5
STAGNATION_MAX_ABS_SLOPE_KG_PER_WEEK: Final[float] = 0.15


@dataclass(frozen=True)
class StagnationConfig:
    window_days: int = STAGNATION_WINDOW_DAYS
    min_measurements: int = STAGNATION_MIN_MEASUREMENTS
    max_amplitude_kg: float = STAGNATION_MAX_AMPLITUDE_KG
    max_abs_slope_kg_per_week: float = STAGNATION_MAX_ABS_SLOPE_KG_PER_WEEK
