"""Compatibility re-exports for user weight target helpers.

The implementation lives in :mod:`app.config` so Streamlit pages can import the
helpers from a module they already load, avoiding page-level ImportError when the
separate ``app.core.targets`` module is stale during deployment.
"""

from __future__ import annotations

from app.config import DEFAULT_TARGETS, TARGET_COUNT, get_target_weights, normalise_target_weights

__all__ = [
    "DEFAULT_TARGETS",
    "TARGET_COUNT",
    "get_target_weights",
    "normalise_target_weights",
]
