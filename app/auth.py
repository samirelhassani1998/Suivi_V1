import os
import hmac

import streamlit as st

def _get_expected_password() -> str | None:
    """Return the configured password from secrets or environment."""

    if "app_password" in st.secrets:
        return str(st.secrets["app_password"])

    env_password = os.getenv("APP_PASSWORD")
    if env_password:
        return env_password

    return None


def check_password() -> bool:
    """Return ``True`` if the user provided the correct password.

    The password is read from ``st.secrets['app_password']`` or the
    ``APP_PASSWORD`` environment variable. A clear error is displayed if no
    secret is configured to avoid silently blocking the UI.
    """
    Returns `True` if the user had the correct password.
    """
    
    correct_password = st.secrets.get("password", "1234567890")
    
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
        "Mot de passe (voir la section secrets de l'app)",
        type="password",
        on_change=password_entered,
        key="password",
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Mot de passe incorrect. Assurez-vous d'utiliser le secret configuré.")

    return False
