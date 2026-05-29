"""Gestion du cycle de vie des données en session Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import ALL_COLUMNS


DEFAULT_TARGETS = (100.0, 95.0, 90.0, 85.0, 90.0)
DEFAULT_WEIGHT_COLUMNS = ["Date", "Poids (Kgs)"]


def normalise_target_weights(targets: object | None) -> tuple[float, ...]:
    """Retourne toujours les 5 objectifs attendus pour les graphiques.

    Les anciennes sessions Streamlit peuvent encore contenir les 4 anciens
    objectifs (95, 90, 85, 80). Dans ce cas, on force les nouveaux paliers
    demandés pour éviter de réafficher l'ancien graphique après déploiement.
    """
    if targets is None:
        return DEFAULT_TARGETS

    try:
        values = tuple(float(target) for target in targets)
    except (TypeError, ValueError):
        return DEFAULT_TARGETS

    if len(values) != len(DEFAULT_TARGETS):
        return DEFAULT_TARGETS

    return values


def get_target_weights() -> tuple[float, ...]:
    """Lit, corrige et persiste les objectifs de poids de la session."""
    targets = normalise_target_weights(st.session_state.get("target_weights"))
    st.session_state["target_weights"] = targets
    st.session_state["target_weight"] = float(targets[-1])
    return targets


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ALL_COLUMNS))


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

    targets = normalise_target_weights(st.session_state.get("target_weights"))
    st.session_state["target_weights"] = targets
    st.session_state["target_weight"] = float(targets[-1])
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
