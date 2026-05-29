"""Compatibility helpers for user weight targets stored in Streamlit session state."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from app.config import (
    DEFAULT_TARGETS,
    LEGACY_FOUR_TARGETS,
    TARGET_COUNT,
    get_target_weights as _get_target_weights,
    normalise_target_weights,
)


def get_target_weights(session_state: MutableMapping[str, Any] | None = None) -> tuple[float, float, float, float, float]:
    """Read target weights using the legacy no-argument Streamlit API when needed."""
    if session_state is None:
        import streamlit as st

        session_state = st.session_state
    return _get_target_weights(session_state)
