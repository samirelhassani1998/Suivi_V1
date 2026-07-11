"""Gestion du cycle de vie des données en session Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import ALL_COLUMNS
from app.core.targets import DEFAULT_TARGETS, get_target_weights

DEFAULT_WEIGHT_COLUMNS = ["Date", "Poids (Kgs)"]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ALL_COLUMNS))


def get_filtered_or_working_data() -> pd.DataFrame:
    """Retourne une copie de la vue filtrée ou des données de travail."""
    if "filtered_data" in st.session_state and not st.session_state["filtered_data"].empty:
        return st.session_state["filtered_data"].copy(deep=True)
    return st.session_state.get("working_data", _empty_df()).copy(deep=True)


def ensure_session_defaults() -> None:
    """Initialise les clés de session une seule fois sans écraser les edits."""
    st.session_state.setdefault("source_data", _empty_df())
    st.session_state.setdefault("working_data", _empty_df())
    st.session_state.setdefault("filtered_data", _empty_df())
    st.session_state.setdefault("analysis_data", _empty_df())
    st.session_state.setdefault("data_quality", {})

    st.session_state.setdefault("target_weights", DEFAULT_TARGETS)
    get_target_weights(st.session_state)
    st.session_state.setdefault("ma_type", "Simple")
    st.session_state.setdefault("window_size", 7)
    st.session_state.setdefault("theme", "plotly")


def set_source_data(df: pd.DataFrame, source_name: str, quality: dict | None = None) -> None:
    """Remplace explicitement la source et réinitialise la copie éditable."""
    clean = df.copy(deep=True)
    st.session_state["source_data"] = clean.copy(deep=True)
    st.session_state["working_data"] = clean.copy(deep=True)
    st.session_state["filtered_data"] = clean.copy(deep=True)
    st.session_state["analysis_data"] = _empty_df()
    st.session_state["raw_data"] = clean.copy(deep=True)
    st.session_state["data_source"] = source_name
    if quality is not None:
        q = dict(quality)
        q["source"] = source_name
        st.session_state["data_quality"] = q


def reset_working_to_source() -> None:
    source = st.session_state.get("source_data", _empty_df()).copy(deep=True)
    st.session_state["working_data"] = source.copy(deep=True)
    st.session_state["filtered_data"] = source.copy(deep=True)
    st.session_state["analysis_data"] = _empty_df()
    st.session_state["raw_data"] = source.copy(deep=True)


def set_working_data(df: pd.DataFrame) -> None:
    work = df.copy(deep=True)
    st.session_state["working_data"] = work.copy(deep=True)
    st.session_state["filtered_data"] = work.copy(deep=True)
    st.session_state["analysis_data"] = _empty_df()
    st.session_state["raw_data"] = work.copy(deep=True)


def set_filtered_data(df: pd.DataFrame) -> None:
    """Stocke une vue temporaire sans modifier source_data ni working_data."""
    st.session_state["filtered_data"] = df.copy(deep=True)


def get_working_data() -> pd.DataFrame:
    return st.session_state.get("working_data", _empty_df()).copy(deep=True)
