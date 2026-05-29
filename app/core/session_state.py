"""Gestion du cycle de vie des données en session Streamlit."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import pandas as pd
import streamlit as st

from app.config import ALL_COLUMNS


DEFAULT_TARGETS = (100.0, 95.0, 90.0, 85.0, 80.0)
DEFAULT_WEIGHT_COLUMNS = ["Date", "Poids (Kgs)"]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ALL_COLUMNS))


def normalise_target_weights(target_weights: Any = None) -> tuple[float, float, float, float, float]:
    """Return exactly five numeric targets, padding legacy sessions with defaults."""
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


def get_target_weights() -> tuple[float, float, float, float, float]:
    """Read, migrate and persist the five configured target weights."""
    targets = normalise_target_weights(st.session_state.get("target_weights"))
    st.session_state["target_weights"] = targets
    st.session_state["target_weight"] = float(targets[-1])
    return targets


def get_filtered_or_working_data() -> pd.DataFrame:
    return st.session_state.get(
        "filtered_data",
        st.session_state.get("working_data", pd.DataFrame(columns=DEFAULT_WEIGHT_COLUMNS)),
    )


def ensure_session_defaults() -> None:
    """Initialise les clés de session une seule fois."""
    st.session_state.setdefault("source_data", _empty_df())
    st.session_state.setdefault("working_data", _empty_df())
    st.session_state.setdefault("filtered_data", _empty_df())

    st.session_state.setdefault("target_weights", DEFAULT_TARGETS)
    get_target_weights()
    st.session_state.setdefault("ma_type", "Simple")
    st.session_state.setdefault("window_size", 7)
    st.session_state.setdefault("theme", "plotly")


def set_source_data(df: pd.DataFrame, source_name: str) -> None:
    clean = df.copy()
    st.session_state["source_data"] = clean
    st.session_state["working_data"] = clean.copy()
    st.session_state["filtered_data"] = clean.copy()
    # compat héritage V2
    st.session_state["raw_data"] = clean.copy()
    st.session_state["data_source"] = source_name


def reset_working_to_source() -> None:
    source = st.session_state.get("source_data", _empty_df())
    st.session_state["working_data"] = source.copy()
    st.session_state["filtered_data"] = source.copy()
    st.session_state["raw_data"] = source.copy()


def set_working_data(df: pd.DataFrame) -> None:
    work = df.copy()
    st.session_state["working_data"] = work
    st.session_state["filtered_data"] = work.copy()
    st.session_state["raw_data"] = work.copy()


def get_working_data() -> pd.DataFrame:
    return st.session_state.get("working_data", _empty_df()).copy()
