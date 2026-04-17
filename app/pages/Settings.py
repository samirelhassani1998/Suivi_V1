from __future__ import annotations

import streamlit as st

from app.config import AppDefaults, DUPLICATE_STRATEGIES


def main() -> None:
    st.title("Paramètres / Qualité")
    defaults = AppDefaults()

    with st.form("settings_form"):
        target = st.number_input("Poids objectif final (kg)", value=float(st.session_state.get("target_weight", defaults.target_weight)))
        height_cm = st.number_input("Taille (cm)", value=float(st.session_state.get("height_cm", defaults.height_cm)))

        goals = st.session_state.get("target_weights", (95.0, 90.0, 85.0, float(target)))
        g1 = st.number_input("Objectif 1 (kg)", value=float(goals[0]))
        g2 = st.number_input("Objectif 2 (kg)", value=float(goals[1]))
        g3 = st.number_input("Objectif 3 (kg)", value=float(goals[2]))
        g4 = st.number_input("Objectif 4 (kg)", value=float(goals[3] if len(goals) > 3 else target))

        ma_type = st.selectbox("Type de moyenne mobile", ["Simple", "Exponentielle"], index=0 if st.session_state.get("ma_type", "Simple") == "Simple" else 1)
        window_size = st.slider("Fenêtre moyenne mobile", min_value=3, max_value=60, value=int(st.session_state.get("window_size", 7)))

        duplicate = st.selectbox("Gestion doublons journaliers", DUPLICATE_STRATEGIES, index=DUPLICATE_STRATEGIES.index(st.session_state.get("duplicate_strategy", defaults.duplicate_strategy)))
        default_model = st.selectbox("Modèle par défaut", ["linear", "ridge", "elasticnet", "random_forest", "boosting"], index=0)
        plotly_theme = st.selectbox("Thème Plotly", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"], index=0)
        submitted = st.form_submit_button("Enregistrer")

    if submitted:
        st.session_state["target_weight"] = float(target)
        st.session_state["target_weights"] = (float(g1), float(g2), float(g3), float(g4))
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
