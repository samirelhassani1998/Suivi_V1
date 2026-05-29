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


DEFAULT_TARGETS: Final[tuple[float, float, float, float, float]] = (100.0, 95.0, 90.0, 85.0, 80.0)
LEGACY_FOUR_TARGETS: Final[tuple[float, float, float, float]] = DEFAULT_TARGETS[1:]
TARGET_COUNT: Final[int] = len(DEFAULT_TARGETS)


def _as_finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def normalise_target_weights(target_weights: Any = None) -> tuple[float, float, float, float, float]:
    """Return the five configured weight targets, migrating legacy four-goal sessions."""
    if target_weights is None or isinstance(target_weights, (str, bytes)):
        values: list[Any] = []
    elif isinstance(target_weights, Iterable):
        values = list(target_weights)
    else:
        values = [target_weights]

    parsed_values = [_as_finite_float(value, float("nan")) for value in values]
    finite_values = [value for value in parsed_values if math.isfinite(value)]

    # Older deployed sessions stored only 95/90/85/80.  Prepend the new 100 kg
    # objective instead of padding at the end, otherwise the graph still starts
    # at 95 kg and appears unchanged to existing users.
    rounded_legacy = tuple(round(value, 3) for value in LEGACY_FOUR_TARGETS)
    rounded_values = tuple(round(value, 3) for value in finite_values)
    legacy_with_duplicated_final = rounded_legacy + (round(DEFAULT_TARGETS[-1], 3),)
    if rounded_values == rounded_legacy or rounded_values[:TARGET_COUNT] == legacy_with_duplicated_final:
        return DEFAULT_TARGETS

    normalised: list[float] = []
    for idx, default in enumerate(DEFAULT_TARGETS):
        candidate = values[idx] if idx < len(values) else default
        normalised.append(_as_finite_float(candidate, default))

    return tuple(normalised)  # type: ignore[return-value]


def get_target_weights(session_state: MutableMapping[str, Any]) -> tuple[float, float, float, float, float]:
    """Read, migrate and persist the five configured target weights in session state."""
    targets = normalise_target_weights(session_state.get("target_weights"))
    session_state["target_weights"] = targets
    session_state["target_weight"] = float(targets[-1])
    return targets


DUPLICATE_STRATEGIES: Final[tuple[str, ...]] = (
    "garder_la_derniere",
    "moyenne_journaliere",
    "mediane_journaliere",
)


@dataclass(frozen=True)
class AppDefaults:
    """Valeurs par défaut modifiables via l'UI."""

    height_cm: int = 182
    target_weight: float = 80.0
    confidence_level: float = 0.9
    duplicate_strategy: str = "garder_la_derniere"
    default_model: str = "ridge"
