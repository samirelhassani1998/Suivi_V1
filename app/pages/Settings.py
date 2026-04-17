from __future__ import annotations

import streamlit as st

from app.config import AppDefaults, DUPLICATE_STRATEGIES


def main() -> None:
    st.title("Paramètres / Qualité")
    defaults = AppDefaults()
    with st.form("settings_form"):
        target = st.number_input("Poids objectif (kg)", value=float(st.session_state.get("target_weight", defaults.target_weight)))
        height_cm = st.number_input("Taille (cm)", value=float(st.session_state.get("height_cm", defaults.height_cm)))
        duplicate = st.selectbox("Gestion doublons journaliers", DUPLICATE_STRATEGIES, index=DUPLICATE_STRATEGIES.index(st.session_state.get("duplicate_strategy", defaults.duplicate_strategy)))
        default_model = st.selectbox("Modèle par défaut", ["ridge", "elasticnet", "random_forest", "boosting"], index=0)
        submitted = st.form_submit_button("Enregistrer")

    if submitted:
        st.session_state["target_weight"] = float(target)
        st.session_state["height_cm"] = float(height_cm)
        st.session_state["height_m"] = float(height_cm) / 100
        st.session_state["duplicate_strategy"] = duplicate
        st.session_state["default_model"] = default_model
        st.success("Paramètres enregistrés.")

    st.caption("Diagnostic système")
    st.write({"streamlit": st.__version__, "session_keys": sorted(list(st.session_state.keys()))[:10]})


main()
