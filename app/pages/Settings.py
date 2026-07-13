from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.components import page_hero, section_header

from app.config import AppDefaults, DUPLICATE_STRATEGIES
from app.core.session_state import (
    DEFAULT_ZOOM_TARGET_END_DATE,
    DEFAULT_ZOOM_TARGET_START_DATE,
    ensure_session_defaults,
)
from app.core.targets import get_target_weights, normalise_target_weights


def main() -> None:
    ensure_session_defaults()
    defaults = AppDefaults()
    page_hero(
        "Configuration",
        "Paramètres / Qualité",
        "Ajustez vos objectifs, votre taille, les moyennes mobiles et quelques options d’analyse.",
        meta="Les changements sont enregistrés dans la session Streamlit",
    )
    section_header("Objectifs et préférences", "Gardez une cible finale claire tout en suivant des paliers intermédiaires.", "⚙️")

    with st.form("settings_form"):
        padded_goals = get_target_weights(st.session_state)
        st.number_input(
            "Poids objectif final (kg)",
            value=float(padded_goals[-1]),
            disabled=True,
            help="Synchronisé avec Objectif 5 pour garder une cible finale unique.",
        )
        height_cm = st.number_input("Taille (cm)", value=float(st.session_state.get("height_cm", defaults.height_cm)))
        g1 = st.number_input("Objectif 1 (kg)", value=float(padded_goals[0]))
        g2 = st.number_input("Objectif 2 (kg)", value=float(padded_goals[1]))
        g3 = st.number_input("Objectif 3 (kg)", value=float(padded_goals[2]))
        g4 = st.number_input("Objectif 4 (kg)", value=float(padded_goals[3]))
        g5 = st.number_input("Objectif 5 (kg)", value=float(padded_goals[4]))

        ma_type = st.selectbox("Type de moyenne mobile", ["Simple", "Exponentielle"], index=0 if st.session_state.get("ma_type", "Simple") == "Simple" else 1)
        window_size = st.slider("Fenêtre moyenne mobile", min_value=3, max_value=60, value=int(st.session_state.get("window_size", 7)))

        zoom_start = st.date_input(
            "Début du zoom trajectoire",
            value=pd.Timestamp(
                st.session_state.get("zoom_target_start_date", DEFAULT_ZOOM_TARGET_START_DATE)
            ).date(),
        )
        zoom_end = st.date_input(
            "Fin du zoom trajectoire",
            value=pd.Timestamp(
                st.session_state.get("zoom_target_end_date", DEFAULT_ZOOM_TARGET_END_DATE)
            ).date(),
        )

        duplicate = st.selectbox("Gestion doublons journaliers", DUPLICATE_STRATEGIES, index=DUPLICATE_STRATEGIES.index(st.session_state.get("duplicate_strategy", defaults.duplicate_strategy)))
        default_model = st.selectbox("Modèle par défaut", ["linear", "ridge", "elasticnet", "random_forest", "boosting"], index=0)
        plotly_theme = st.selectbox("Thème Plotly", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"], index=0)
        submitted = st.form_submit_button("Enregistrer")

    if submitted:
        target_weights = normalise_target_weights((g1, g2, g3, g4, g5))
        st.session_state["target_weights"] = target_weights
        st.session_state["target_weight"] = float(target_weights[-1])
        st.session_state["height_cm"] = float(height_cm)
        st.session_state["height_m"] = float(height_cm) / 100
        st.session_state["duplicate_strategy"] = duplicate
        st.session_state["default_model"] = default_model
        st.session_state["ma_type"] = ma_type
        zoom_start_ts = pd.Timestamp(zoom_start)
        zoom_end_ts = pd.Timestamp(zoom_end)
        if zoom_start_ts > zoom_end_ts:
            st.warning("La date de début du zoom doit être antérieure ou égale à la date de fin.")
        else:
            st.session_state["window_size"] = int(window_size)
            st.session_state["zoom_target_start_date"] = zoom_start_ts
            st.session_state["zoom_target_end_date"] = zoom_end_ts
            st.session_state["theme"] = plotly_theme
            st.success("Paramètres enregistrés.")

    section_header("Diagnostic système", "Informations utiles pour vérifier la session active et le déploiement.", "🧪")
    st.write({
        "streamlit": st.__version__,
        "source": st.session_state.get("data_source", "n/a"),
        "rows_working_data": len(st.session_state.get("working_data", [])),
        "session_keys": sorted(list(st.session_state.keys()))[:20],
    })


main()
