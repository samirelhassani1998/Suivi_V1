"""Entry point for the Streamlit multi-page application."""

from __future__ import annotations

import subprocess
import pandas as pd
import streamlit as st

from app.utils import (
    DATA_URL,
    filter_by_dates,
    get_date_range,
    load_data,
)
from app.auth import check_password


def _get_commit_sha() -> str:
    """Get the short commit SHA for version tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


st.set_page_config(page_title="Suivi & Analyses du Poids", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container { font-family: 'Segoe UI', sans-serif; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf, #2e7bcf); color: white; }
    .stButton>button { background-color: #2e7bcf; color: white; border: none; }
    .stMetric { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_dataset() -> None:
    """Load the dataset and store both raw and filtered versions."""
    if "data_url" not in st.session_state:
        st.session_state["data_url"] = st.secrets.get("data_url", DATA_URL)

    if st.session_state.get("reload_requested"):
        load_data.clear()
        st.session_state.pop("reload_requested")

    with st.spinner("Chargement des données de poids..."):
        try:
            df = load_data(st.session_state["data_url"])
        except Exception as error:
            st.error(f"Erreur de chargement : {error}")
            st.exception(error)
            empty_df = pd.DataFrame(columns=["Date", "Poids (Kgs)"])
            st.session_state["raw_data"] = empty_df
            st.session_state["filtered_data"] = empty_df
            return  # Don't st.stop() - let the page handle empty data

    st.session_state["raw_data"] = df
    st.session_state["filtered_data"] = df


def _configure_sidebar() -> None:
    """Setup sidebar controls and apply filtering to the dataset."""
    st.sidebar.header("Paramètres Généraux")

    if st.sidebar.button("Recharger les données"):
        st.session_state["reload_requested"] = True
        load_data.clear()
        st.rerun()

    st.sidebar.selectbox(
        "Choisir un thème",
        ["Default", "Dark", "Light", "Solar", "Seaborn"],
        key="theme",
    )
    st.sidebar.selectbox(
        "Type de moyenne mobile",
        ["Simple", "Exponentielle"],
        key="ma_type",
    )
    st.sidebar.slider(
        "Taille de la moyenne mobile (jours)",
        1, 30,
        st.session_state.get("window_size", 7),
        key="window_size",
    )

    df = st.session_state.get("raw_data")
    if df is not None and not df.empty:
        try:
            date_min, date_max = get_date_range(df)
            date_range = st.sidebar.date_input(
                "Sélectionnez une plage de dates",
                (date_min, date_max),
                key="date_range",
            )
            st.session_state["filtered_data"] = filter_by_dates(df, date_range)
        except Exception as error:
            st.sidebar.error(f"Erreur dates : {error}")
            st.session_state["filtered_data"] = df
    else:
        st.session_state["filtered_data"] = df

    st.sidebar.markdown("---")
    st.sidebar.subheader("Objectifs et Infos Personnelles")
    target1 = st.sidebar.number_input("Objectif 1 (Kgs)", value=st.session_state.get("target_weight_1", 95.0), key="target_weight_1")
    target2 = st.sidebar.number_input("Objectif 2 (Kgs)", value=st.session_state.get("target_weight_2", 90.0), key="target_weight_2")
    target3 = st.sidebar.number_input("Objectif 3 (Kgs)", value=st.session_state.get("target_weight_3", 85.0), key="target_weight_3")
    target4 = st.sidebar.number_input("Objectif 4 (Kgs)", value=st.session_state.get("target_weight_4", 80.0), key="target_weight_4")
    st.session_state["target_weights"] = (target1, target2, target3, target4)

    height_cm = st.sidebar.number_input("Votre taille (cm)", value=st.session_state.get("height_cm", 182), key="height_cm")
    st.session_state["height_m"] = height_cm / 100.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("Anomalies & Activité")
    st.sidebar.selectbox("Méthode de détection", ["Z-score", "IsolationForest"], key="anomaly_method")
    st.sidebar.slider("Seuil Z-score", 1.0, 5.0, st.session_state.get("z_threshold", 2.0), step=0.5, key="z_threshold")

    calories = st.sidebar.number_input("Calories consommées", min_value=0, value=st.session_state.get("calories", 2000), key="calories")
    calories_burned = st.sidebar.number_input("Calories brûlées", min_value=0, value=st.session_state.get("calories_burned", 500), key="calories_burned")
    st.sidebar.write("Bilan calorique :", calories - calories_burned, "kcal")

    st.sidebar.markdown("---")
    with st.sidebar.expander("État du système", expanded=False):
        st.write(f"Streamlit: {st.__version__}")
        st.write(f"Commit: {_get_commit_sha()}")
        if "auth" in st.secrets or "password" in st.secrets:
            st.success("Secrets: Configuré")
        else:
            st.warning("Secrets: Non configuré")


# ============================================================
# MAIN APPLICATION FLOW
# ============================================================

# 1. Check auth FIRST
if not check_password():
    st.stop()

# 2. Load data
_load_dataset()

# 3. Setup sidebar
_configure_sidebar()

# 4. CRITICAL: Run navigation - this MUST be the main content
#    Pages will render their own content. Do NOT add content after pg.run()!
if hasattr(st, "Page") and hasattr(st, "navigation"):
    pages = [
        st.Page("app/pages/Overview.py", title="Vue d'ensemble", icon="📊"),
        st.Page("app/pages/Modeles.py", title="Modèles", icon="🤖"),
        st.Page("app/pages/Predictions.py", title="Prédictions", icon="📈"),
    ]
    pg = st.navigation(pages)
    pg.run()  # <-- This executes the selected page script
else:
    st.error("Streamlit >= 1.31 requis pour st.navigation.")
    st.stop()
