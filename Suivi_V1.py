"""Entrée principale Streamlit - Suivi V1."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import requests
import streamlit as st

from app.auth import check_password
from app.config import DATA_URL
from app.core.data import clean_weight_dataframe_with_report, read_weight_csv
from app.core.data_editing import has_unsaved_changes
from app.core.session_state import ensure_session_defaults, reset_working_to_source, set_source_data
from app.core.whoop_session import capture_oauth_callback, switch_to_whoop_page_if_pending
from app.ui.theme import apply_global_theme


@st.cache_data(ttl=300)
def load_remote_csv_with_report(url: str) -> tuple[pd.DataFrame, dict]:
    response = requests.get(url, timeout=(5, 20))
    response.raise_for_status()
    raw = read_weight_csv(BytesIO(response.content))
    cleaned, quality = clean_weight_dataframe_with_report(raw, source="google_sheets")
    # Conserver toutes les mesures source, y compris plusieurs lignes le même jour.
    return cleaned, quality.to_dict()


def load_remote_csv(url: str) -> pd.DataFrame:
    df, _ = load_remote_csv_with_report(url)
    return df.copy(deep=True)


def _show_quality_message() -> None:
    """Résumé compact de l'import dans la barre latérale ; le détail vit dans Journal."""
    q = st.session_state.get("data_quality", {})
    if not q:
        return
    valid = q.get("valid_rows", 0)
    invalid = q.get("invalid_rows", 0)
    duplicates = q.get("duplicate_dates", 0)
    icon = "✅" if not invalid else "⚠️"
    st.sidebar.caption(
        f"{icon} {valid} mesure(s) valide(s) · {invalid} rejetée(s) · {duplicates} date(s) en double conservée(s). "
        "Détail dans Journal."
    )


def _load_from_source() -> None:
    data_url = st.secrets.get("data_url", DATA_URL)
    st.session_state["data_url"] = data_url
    df, quality = load_remote_csv_with_report(data_url)
    if df.empty:
        raise ValueError("La source ne contient aucune mesure valide. Les données en session sont conservées.")
    set_source_data(df, "google_sheets", quality)


def _import_local_csv(uploaded_file) -> None:
    imported = read_weight_csv(uploaded_file)
    df, quality = clean_weight_dataframe_with_report(imported, source="csv_local")
    if df.empty:
        raise ValueError("Le CSV ne contient aucune mesure valide. Les données en session sont conservées.")
    set_source_data(df, "csv_local", quality.to_dict())


def init_data_once() -> None:
    ensure_session_defaults()
    if not st.session_state.get("data_initialized", False) and st.session_state.get("source_data", pd.DataFrame()).empty and st.session_state.get("working_data", pd.DataFrame()).empty:
        try:
            _load_from_source()
        except Exception:
            st.warning("Source Google Sheets indisponible. Importez un CSV pour continuer.")


def sidebar_controls() -> None:
    confirmation_revision = int(st.session_state.get("replacement_confirmation_revision", 0))
    st.sidebar.markdown(
        """
        <div class="suivi-sidebar-card">
            <span class="suivi-sidebar-eyebrow">Source active</span>
            <strong>{source}</strong>
        </div>
        """.format(source=st.session_state.get("data_source", "n/a")),
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Données", expanded=True):
        st.caption("Synchronisez la source principale ou annulez les modifications de session.")
        changed = has_unsaved_changes(st.session_state.get("working_data", pd.DataFrame()), st.session_state.get("source_data", pd.DataFrame())) or st.session_state.get("journal_has_unsaved_changes", False)
        replace_allowed = True
        if changed:
            st.warning("Des modifications locales sont présentes. Exportez-les avant de remplacer les données.")
            replace_allowed = st.checkbox("Remplacer mes modifications de session", key=f"allow_replace_session_data_{confirmation_revision}")
        if st.button("Recharger Google Sheets", use_container_width=True, disabled=not replace_allowed) and replace_allowed:
            st.session_state["replacement_confirmation_revision"] = confirmation_revision + 1
            load_remote_csv_with_report.clear()
            try:
                _load_from_source()
                st.success("Données rechargées depuis Google Sheets.")
            except Exception as exc:
                st.error(f"Échec du rechargement source: {exc}")

        if st.button("Réinitialiser la session", use_container_width=True, disabled=not replace_allowed) and replace_allowed:
            st.session_state["replacement_confirmation_revision"] = confirmation_revision + 1
            reset_working_to_source()
            st.info("Les données de travail ont été réinitialisées depuis la source.")

    with st.sidebar.expander("Import CSV", expanded=False):
        st.caption("Option secondaire : remplace uniquement les données de la session courante.")
        uploaded = st.file_uploader("Fichier CSV à importer", type=["csv"], key="sidebar_csv_import")
        if uploaded is not None and st.button("Valider l’import CSV", use_container_width=True, disabled=not replace_allowed) and replace_allowed:
            st.session_state["replacement_confirmation_revision"] = confirmation_revision + 1
            try:
                _import_local_csv(uploaded)
            except (ValueError, UnicodeError) as exc:
                st.error(f"Import impossible : {exc}")
            else:
                st.success("CSV importé dans la session.")

    _show_quality_message()



st.set_page_config(page_title="Suivi V1", page_icon="📊", layout="wide")
apply_global_theme()

# WHOOP redirige vers la Redirect URI déclarée, qui est le plus souvent la racine
# de l'application : le code d'autorisation est donc capté avant la porte
# d'authentification et avant tout rendu de page, sinon il serait perdu.
capture_oauth_callback()

if not check_password():
    st.stop()

init_data_once()
sidebar_controls()

whoop_page = st.Page("app/pages/Whoop.py", title="Whoop", icon="⌚")
pages = [
    st.Page("app/pages/Dashboard.py", title="Dashboard", icon="📊"),
    st.Page("app/pages/Journal.py", title="Journal", icon="🧾"),
    st.Page("app/pages/Predictions.py", title="Prévisions", icon="📈"),
    st.Page("app/pages/Insights.py", title="Insights", icon="🔍"),
    whoop_page,
    st.Page("app/pages/Boxe.py", title="Boxe", icon="🥊"),
    st.Page("app/pages/Settings.py", title="Paramètres", icon="⚙️"),
]
navigation = st.navigation(pages)
switch_to_whoop_page_if_pending(whoop_page)
navigation.run()
