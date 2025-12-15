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

    expected_password = _get_expected_password()
    if expected_password is None:
        st.error(
            "Configuration manquante : définissez `app_password` dans les "
            "secrets Streamlit ou la variable d'environnement `APP_PASSWORD`."
        )
        return False

    def password_entered() -> None:
        """Check whether the user provided the correct password."""

        if hmac.compare_digest(st.session_state.get("password", ""), expected_password):
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
