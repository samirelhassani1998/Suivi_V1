"""Compatibility helpers for user weight targets stored in Streamlit session state."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
import math
from typing import Any, Final

from app import config as _config

DEFAULT_TARGETS: Final[tuple[float, float, float, float, float]] = tuple(
    float(value) for value in getattr(_config, "DEFAULT_TARGETS", (100.0, 95.0, 90.0, 85.0, 80.0))
)  # type: ignore[assignment]
LEGACY_FOUR_TARGETS: Final[tuple[float, float, float, float]] = tuple(
    float(value) for value in getattr(_config, "LEGACY_FOUR_TARGETS", (95.0, 90.0, 85.0, 80.0))
)  # type: ignore[assignment]
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


def get_target_weights(session_state: MutableMapping[str, Any] | None = None) -> tuple[float, float, float, float, float]:
    """Read, migrate and persist the five configured target weights."""
    if session_state is None:
        import streamlit as st

        session_state = st.session_state

    targets = normalise_target_weights(session_state.get("target_weights"))
    session_state["target_weights"] = targets
    session_state["target_weight"] = float(targets[-1])
    return targets
