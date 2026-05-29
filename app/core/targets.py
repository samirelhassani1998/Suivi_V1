"""Helpers for user weight targets stored in Streamlit session state."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
import math
from typing import Any


DEFAULT_TARGETS = (100.0, 95.0, 90.0, 85.0, 80.0)
LEGACY_FOUR_TARGETS = DEFAULT_TARGETS[1:]
TARGET_COUNT = len(DEFAULT_TARGETS)


def _as_finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def normalise_target_weights(target_weights: Any = None) -> tuple[float, float, float, float, float]:
    """Return exactly five finite numeric targets, migrating legacy sessions."""
    if target_weights is None or isinstance(target_weights, (str, bytes)):
        values: list[Any] = []
    elif isinstance(target_weights, Iterable):
        values = list(target_weights)
    else:
        values = [target_weights]

    finite_values = [_as_finite_float(value, float("nan")) for value in values]
    rounded_values = tuple(round(value, 3) for value in finite_values if math.isfinite(value))
    rounded_legacy = tuple(round(value, 3) for value in LEGACY_FOUR_TARGETS)
    legacy_with_duplicated_final = rounded_legacy + (round(DEFAULT_TARGETS[-1], 3),)
    if rounded_values == rounded_legacy or rounded_values[:TARGET_COUNT] == legacy_with_duplicated_final:
        return DEFAULT_TARGETS

    normalised: list[float] = []
    for idx, default in enumerate(DEFAULT_TARGETS):
        candidate = values[idx] if idx < len(values) else default
        normalised.append(_as_finite_float(candidate, default))

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
