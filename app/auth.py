"""Authentication module for the Streamlit app."""

from __future__ import annotations

import hmac
import streamlit as st


def check_password() -> bool:
    """Return `True` if the user had the correct password."""
    
    # Récupération sécurisée du mot de passe
    if "password" in st.secrets:
        correct_password = str(st.secrets["password"])
    else:
        st.error("⚠️ Secret 'password' introuvable. Veuillez configurer les secrets de l'application.")
        # On empêche l'accès par sécurité si pas de secret configuré
        return False

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], correct_password):
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)  # Don't store the password
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
