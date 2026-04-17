"""Entrée principale Streamlit - Suivi V2."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.auth import check_password
from app.config import DATA_URL
from app.core.data import clean_weight_dataframe, validate_journal
from app.ui.theme import apply_global_theme


@st.cache_data(ttl=300)
def load_remote_csv(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url)
    return clean_weight_dataframe(raw)


def load_dataset() -> None:
    """Charge Google Sheets, puis fallback CSV local si indisponible."""
    data_url = st.secrets.get("data_url", DATA_URL)
    st.session_state["data_url"] = data_url

    try:
        df = load_remote_csv(data_url)
        st.session_state["data_source"] = "google_sheets"
    except Exception:
        st.warning("Source Google Sheets indisponible. Utilisez un CSV local pour continuer.")
        uploaded = st.sidebar.file_uploader("Fallback CSV local", type=["csv"])
        if uploaded is None:
            df = st.session_state.get("raw_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"]))
            st.session_state["data_source"] = "session"
        else:
            local = pd.read_csv(uploaded)
            report = validate_journal(local)
            df = report.cleaned
            st.session_state["data_source"] = "csv_local"
            for warning in report.warnings:
                st.info(warning)

    st.session_state["raw_data"] = df.copy()
    st.session_state["filtered_data"] = df.copy()


def sidebar_controls() -> None:
    st.sidebar.header("Contrôles")
    if st.sidebar.button("Recharger les données"):
        load_remote_csv.clear()
        st.rerun()

    st.sidebar.caption(f"Source active: {st.session_state.get('data_source', 'n/a')}")


st.set_page_config(page_title="Suivi V2", layout="wide")
apply_global_theme()

if not check_password():
    st.stop()

load_dataset()
sidebar_controls()

pages = [
    st.Page("app/pages/Dashboard.py", title="Dashboard", icon="📊"),
    st.Page("app/pages/Journal.py", title="Journal", icon="🧾"),
    st.Page("app/pages/Predictions.py", title="Prévisions", icon="📈"),
    st.Page("app/pages/Insights.py", title="Insights", icon="🔍"),
    st.Page("app/pages/Settings.py", title="Paramètres", icon="⚙️"),
]
st.navigation(pages).run()
