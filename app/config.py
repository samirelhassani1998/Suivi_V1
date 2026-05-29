"""Configuration centralisée de l'application Suivi V2."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
import math
from typing import Any, Final

DATA_URL: Final[str] = "https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv"
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("Date", "Poids (Kgs)")
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "Calories consommées",
    "Calories brûlées",
    "Pas",
    "Sommeil (heures)",
    "Hydratation",
    "Tour de taille",
    "Body fat %",
    "Notes",
    "Condition de mesure",
)
ALL_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

DUPLICATE_STRATEGIES: Final[tuple[str, ...]] = (
    "garder_la_derniere",
    "moyenne_journaliere",
    "mediane_journaliere",
)

DEFAULT_TARGETS: Final[tuple[float, float, float, float, float]] = (100.0, 95.0, 90.0, 85.0, 80.0)
LEGACY_FOUR_TARGETS: Final[tuple[float, float, float, float]] = (95.0, 90.0, 85.0, 80.0)
TARGET_COUNT: Final[int] = len(DEFAULT_TARGETS)


def _as_finite_float(value: Any, default: float) -> float:
    """Coerce ``value`` to a finite float, falling back to ``default``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def _target_values(target_weights: Any = None) -> list[Any]:
    if target_weights is None or isinstance(target_weights, (str, bytes)):
        return []
    if isinstance(target_weights, Iterable):
        return list(target_weights)
    return [target_weights]


def _is_legacy_four_target_list(values: list[Any]) -> bool:
    """Detect only true four-value legacy payloads, not explicit five-goal forms."""
    if len(values) != len(LEGACY_FOUR_TARGETS):
        return False

    rounded_values = tuple(round(_as_finite_float(value, math.nan), 3) for value in values)
    missing_final = tuple(round(target, 3) for target in DEFAULT_TARGETS[:-1])
    missing_initial = tuple(round(target, 3) for target in LEGACY_FOUR_TARGETS)
    return rounded_values in (missing_final, missing_initial)


def normalise_target_weights(target_weights: Any = None) -> tuple[float, float, float, float, float]:
    """Return exactly five finite numeric targets, migrating only real 4-goal sessions."""
    values = _target_values(target_weights)
    if _is_legacy_four_target_list(values):
        return DEFAULT_TARGETS

    normalised = [
        _as_finite_float(values[idx], default) if idx < len(values) else float(default)
        for idx, default in enumerate(DEFAULT_TARGETS)
    ]
    return tuple(normalised)  # type: ignore[return-value]


def get_target_weights(session_state: MutableMapping[str, Any]) -> tuple[float, float, float, float, float]:
    """Read, migrate and persist the five configured target weights."""
    targets = normalise_target_weights(session_state.get("target_weights"))
    session_state["target_weights"] = targets
    session_state["target_weight"] = float(targets[-1])
    return targets


@dataclass(frozen=True)
class AppDefaults:
    """Valeurs par défaut modifiables via l'UI."""

    height_cm: int = 182
    target_weight: float = 80.0
    confidence_level: float = 0.9
    duplicate_strategy: str = "garder_la_derniere"
    default_model: str = "ridge"
