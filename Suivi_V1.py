"""Entrée principale Streamlit - Suivi V2."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.auth import check_password
from app.config import DATA_URL
from app.core.data import clean_weight_dataframe_with_report
from app.core.session_state import ensure_session_defaults, reset_working_to_source, set_source_data
from app.ui.theme import apply_global_theme


@st.cache_data(ttl=300)
def load_remote_csv(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url)
    cleaned, _ = clean_weight_dataframe_with_report(raw, source="google_sheets")
    # Conserver toutes les mesures source, y compris plusieurs lignes le même jour.
    return cleaned



def _show_quality_message() -> None:
    q = st.session_state.get("data_quality", {})
    if not q:
        return
    st.info(
        f"Qualité données ({q.get('source', 'n/a')}): "
        f"{q.get('raw_rows', 0)} lues, {q.get('valid_rows', 0)} valides conservées, "
        f"{q.get('invalid_rows', 0)} invalides, {q.get('duplicate_dates', 0)} dates dupliquées, "
        f"{q.get('columns_kept', 0)} colonnes conservées. "
        f"Colonnes additionnelles: {', '.join(q.get('extra_columns', [])) or 'aucune'}."
    )

def _load_from_source() -> None:
    data_url = st.secrets.get("data_url", DATA_URL)
    st.session_state["data_url"] = data_url
    raw = pd.read_csv(data_url)
    df, quality = clean_weight_dataframe_with_report(raw, source="google_sheets")
    set_source_data(df, "google_sheets", quality.to_dict())


def _import_local_csv(uploaded_file) -> None:
    imported = pd.read_csv(uploaded_file)
    df, quality = clean_weight_dataframe_with_report(imported, source="csv_local")
    set_source_data(df, "csv_local", quality.to_dict())


def init_data_once() -> None:
    ensure_session_defaults()
    if st.session_state.get("source_data", pd.DataFrame()).empty:
        try:
            _load_from_source()
        except Exception:
            st.warning("Source Google Sheets indisponible. Importez un CSV pour continuer.")


def sidebar_controls() -> None:
    st.sidebar.markdown(
        """
        <div class="suivi-sidebar-card">
            <span class="suivi-sidebar-eyebrow">Source active</span>
            <strong>{source}</strong>
        </div>
        """.format(source=st.session_state.get("data_source", "n/a")),
        unsafe_allow_html=True,
    )

    _show_quality_message()

    with st.sidebar.expander("Données", expanded=True):
        st.caption("Synchronisez la source principale ou annulez les modifications de session.")
        if st.button("Recharger Google Sheets", use_container_width=True):
            load_remote_csv.clear()
            try:
                _load_from_source()
                st.success("Données rechargées depuis Google Sheets.")
                _show_quality_message()
            except Exception as exc:
                st.error(f"Échec du rechargement source: {exc}")

        if st.button("Réinitialiser la session", use_container_width=True):
            reset_working_to_source()
            st.info("Les données de travail ont été réinitialisées depuis la source.")

    with st.sidebar.expander("Import CSV", expanded=False):
        st.caption("Option secondaire : remplace uniquement les données de la session courante.")
        uploaded = st.file_uploader("Fichier CSV à importer", type=["csv"], key="sidebar_csv_import")
        if uploaded is not None and st.button("Valider l’import CSV", use_container_width=True):
            _import_local_csv(uploaded)
            st.success("CSV importé dans la session.")
            _show_quality_message()



st.set_page_config(page_title="Suivi V2", layout="wide")
apply_global_theme()

if not check_password():
    st.stop()

init_data_once()
sidebar_controls()

pages = [
    st.Page("app/pages/Dashboard.py", title="Dashboard", icon="📊"),
    st.Page("app/pages/Journal.py", title="Journal", icon="🧾"),
    st.Page("app/pages/Predictions.py", title="Prévisions", icon="📈"),
    st.Page("app/pages/Insights.py", title="Insights", icon="🔍"),
    st.Page("app/pages/Settings.py", title="Paramètres", icon="⚙️"),
]
st.navigation(pages).run()
