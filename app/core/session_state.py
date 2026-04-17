"""Gestion du cycle de vie des données en session Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import ALL_COLUMNS


DEFAULT_TARGETS = (95.0, 90.0, 85.0, 80.0)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ALL_COLUMNS))


def ensure_session_defaults() -> None:
    """Initialise les clés de session une seule fois."""
    st.session_state.setdefault("source_data", _empty_df())
    st.session_state.setdefault("working_data", _empty_df())
    st.session_state.setdefault("filtered_data", _empty_df())

    st.session_state.setdefault("target_weights", DEFAULT_TARGETS)
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
