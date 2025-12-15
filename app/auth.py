"""Authentication module for the Streamlit app."""

from __future__ import annotations

import hmac
import streamlit as st


def check_password() -> bool:
    """Return `True` if the user had the correct password."""
    
    password_conf = st.secrets.get("auth", {})
    correct_password = str(password_conf.get("password", ""))

    if not correct_password:
        # Fallback pour compatibilité ou message explicite
        if "password" in st.secrets:
             correct_password = str(st.secrets["password"])
        else:
            st.error("⚠️ Secret '[auth] password' introuvable.")
            with st.expander("Comment configurer l'accès ?"):
                st.markdown(
                    """
                    1. Allez dans les **Settings** de votre app sur Streamlit Cloud.
                    2. Ouvrez l'onglet **Secrets**.
                    3. Ajoutez la configuration suivante :
                    ```toml
                    [auth]
                    password = "votre_mot_de_passe_secret"
                    ```
                    """
                )
            # Mode "Démo" restreint ou blocage (ici blocage propre)
            return False

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], correct_password):
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Mot de passe",
        type="password",
        on_change=password_entered,
        key="password",
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Mot de passe incorrect.")

    return False
