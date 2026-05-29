from __future__ import annotations

import streamlit as st

from app.config import AppDefaults, DUPLICATE_STRATEGIES
from app.core.session_state import get_target_weights, normalise_target_weights


def main() -> None:
    st.title("Paramètres / Qualité")
    defaults = AppDefaults()

    with st.form("settings_form"):
        padded_goals = get_target_weights()
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
        st.session_state["window_size"] = int(window_size)
        st.session_state["theme"] = plotly_theme
        st.success("Paramètres enregistrés.")

    st.caption("Diagnostic système")
    st.write({
        "streamlit": st.__version__,
        "source": st.session_state.get("data_source", "n/a"),
        "rows_working_data": len(st.session_state.get("working_data", [])),
        "session_keys": sorted(list(st.session_state.keys()))[:20],
    })


main()
