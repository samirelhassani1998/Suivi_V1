"""Helpers for user weight targets stored in Streamlit session state."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
import math
from typing import Any


DEFAULT_TARGETS = (100.0, 95.0, 90.0, 85.0, 80.0)
TARGET_COUNT = len(DEFAULT_TARGETS)


def normalise_target_weights(target_weights: Any = None) -> tuple[float, float, float, float, float]:
    """Return exactly five finite numeric targets, padding legacy sessions with defaults."""
    if target_weights is None or isinstance(target_weights, (str, bytes)):
        values: list[Any] = []
    elif isinstance(target_weights, Iterable):
        values = list(target_weights)
    else:
        values = [target_weights]

    normalised: list[float] = []
    for idx, default in enumerate(DEFAULT_TARGETS):
        candidate = values[idx] if idx < len(values) else default
        try:
            numeric = float(candidate)
        except (TypeError, ValueError):
            numeric = default
        if not math.isfinite(numeric):
            numeric = default
        normalised.append(numeric)

    return tuple(normalised)  # type: ignore[return-value]


def get_target_weights(session_state: MutableMapping[str, Any] | None = None) -> tuple[float, float, float, float, float]:
    """Read, migrate and persist the five configured target weights.

    ``streamlit`` is imported lazily so importing this pure helper module cannot
    fail while Streamlit is initialising a page.  Passing ``session_state`` keeps
    the function testable and lets pages reuse the same migration logic.
    """
    if session_state is None:
        import streamlit as st

        session_state = st.session_state

    targets = normalise_target_weights(session_state.get("target_weights"))
    session_state["target_weights"] = targets
    session_state["target_weight"] = float(targets[-1])
    return targets
