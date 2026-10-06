"""Gestion du cycle de vie des données en session Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import ALL_COLUMNS
from app.core.targets import DEFAULT_TARGETS, get_target_weights

DEFAULT_WEIGHT_COLUMNS = ["Date", "Poids (Kgs)"]
DEFAULT_ZOOM_TARGET_START_DATE = pd.Timestamp("2026-10-01")
DEFAULT_ZOOM_TARGET_END_DATE = pd.Timestamp("2026-12-31")
DEFAULT_WHOOP_SYNC_DAYS = 30


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ALL_COLUMNS))


def get_filtered_or_working_data() -> pd.DataFrame:
    """Retourne une copie de la vue filtrée ou des données de travail."""
    if st.session_state.get("filter_active", False):
        return st.session_state.get("filtered_data", _empty_df()).copy(deep=True)
    return st.session_state.get("working_data", _empty_df()).copy(deep=True)


def ensure_session_defaults() -> None:
    """Initialise les clés de session une seule fois sans écraser les edits."""
    st.session_state.setdefault("source_data", _empty_df())
    st.session_state.setdefault("working_data", _empty_df())
    st.session_state.setdefault("filtered_data", _empty_df())
    st.session_state.setdefault("filter_active", False)
    st.session_state.setdefault("analysis_data", _empty_df())
    st.session_state.setdefault("data_quality", {})

    st.session_state.setdefault("target_weights", DEFAULT_TARGETS)
    get_target_weights(st.session_state)
    st.session_state.setdefault("ma_type", "Simple")
    st.session_state.setdefault("window_size", 7)
    st.session_state.setdefault("theme", "plotly")
    st.session_state.setdefault("zoom_target_start_date", DEFAULT_ZOOM_TARGET_START_DATE)
    st.session_state.setdefault("zoom_target_end_date", DEFAULT_ZOOM_TARGET_END_DATE)

    ensure_whoop_defaults()


def ensure_whoop_defaults() -> None:
    """Initialise les clés WHOOP sans toucher aux données de poids existantes."""
    st.session_state.setdefault("whoop_token", None)
    st.session_state.setdefault("whoop_oauth_state", None)
    st.session_state.setdefault("whoop_daily", pd.DataFrame())
    st.session_state.setdefault("whoop_workouts", pd.DataFrame())
    st.session_state.setdefault("whoop_profile", {})
    st.session_state.setdefault("whoop_body", {})
    st.session_state.setdefault("whoop_last_sync", None)
    st.session_state.setdefault("whoop_sync_days", DEFAULT_WHOOP_SYNC_DAYS)
    st.session_state.setdefault("whoop_manual_credentials", {})
    st.session_state.setdefault("whoop_request_offline", True)


def clear_whoop_session() -> None:
    """Déconnecte WHOOP et purge les données importées de la session."""
    st.session_state["whoop_token"] = None
    st.session_state["whoop_oauth_state"] = None
    st.session_state["whoop_daily"] = pd.DataFrame()
    st.session_state["whoop_workouts"] = pd.DataFrame()
    st.session_state["whoop_profile"] = {}
    st.session_state["whoop_body"] = {}
    st.session_state["whoop_last_sync"] = None
    for key in ("whoop_callback_to_resume", "whoop_callback_url", "whoop_pending_auth"):
        st.session_state.pop(key, None)


def store_whoop_sync(result) -> None:
    """Range le résultat d'une synchronisation (voir ``app.core.whoop_sync``) en session."""
    st.session_state["whoop_daily"] = result.daily
    st.session_state["whoop_workouts"] = result.workouts
    st.session_state["whoop_profile"] = result.profile
    st.session_state["whoop_body"] = result.body
    st.session_state["whoop_last_sync"] = pd.Timestamp.utcnow().tz_localize(None)


def _reset_journal_editor() -> None:
    """Une nouvelle base de données ne doit jamais réappliquer les anciens edits."""
    st.session_state["journal_editor_revision"] = int(st.session_state.get("journal_editor_revision", 0)) + 1
    st.session_state["journal_has_unsaved_changes"] = False


def set_source_data(df: pd.DataFrame, source_name: str, quality: dict | None = None) -> None:
    """Remplace explicitement la source et réinitialise la copie éditable."""
    clean = df.copy(deep=True)
    st.session_state["source_data"] = clean.copy(deep=True)
    st.session_state["working_data"] = clean.copy(deep=True)
    st.session_state["filtered_data"] = _empty_df()
    st.session_state["filter_active"] = False
    st.session_state["analysis_data"] = _empty_df()
    st.session_state["raw_data"] = clean.copy(deep=True)
    st.session_state["data_source"] = source_name
    st.session_state["data_initialized"] = True
    _reset_journal_editor()
    if quality is not None:
        q = dict(quality)
        q["source"] = source_name
        st.session_state["data_quality"] = q


def reset_working_to_source() -> None:
    source = st.session_state.get("source_data", _empty_df()).copy(deep=True)
    st.session_state["working_data"] = source.copy(deep=True)
    st.session_state["filtered_data"] = _empty_df()
    st.session_state["filter_active"] = False
    st.session_state["analysis_data"] = _empty_df()
    st.session_state["raw_data"] = source.copy(deep=True)
    _reset_journal_editor()


def set_working_data(df: pd.DataFrame) -> None:
    work = df.copy(deep=True)
    st.session_state["working_data"] = work.copy(deep=True)
    st.session_state["filtered_data"] = _empty_df()
    st.session_state["filter_active"] = False
    st.session_state["analysis_data"] = _empty_df()
    st.session_state["raw_data"] = work.copy(deep=True)
    _reset_journal_editor()


def set_filtered_data(df: pd.DataFrame) -> None:
    """Stocke une vue temporaire sans modifier source_data ni working_data."""
    st.session_state["filtered_data"] = df.copy(deep=True)
    st.session_state["filter_active"] = True


def clear_filter() -> None:
    """Désactive explicitement la vue filtrée."""
    st.session_state["filtered_data"] = _empty_df()
    st.session_state["filter_active"] = False


def get_working_data() -> pd.DataFrame:
    return st.session_state.get("working_data", _empty_df()).copy(deep=True)
