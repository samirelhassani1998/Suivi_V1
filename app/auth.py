"""Authentication module for the Streamlit app."""

from __future__ import annotations

import hmac
import streamlit as st


def check_password() -> bool:
    """Return `True` if the user had the correct password."""
    
    # Récupération de la configuration
    auth_config = st.secrets.get("auth", {})
    password_required = auth_config.get("required", True)  # Par défaut, auth requise
    correct_password = str(auth_config.get("password", ""))

    # Mode "Démo" ou non requis
    if not password_required:
        st.warning("⚠️ Mode DÉMO actif : l'authentification est désactivée.")
        return True

    # Si mot de passe requis mais absent
    if not correct_password:
        if "password" in st.secrets:
             correct_password = str(st.secrets["password"])
        else:
            st.warning("🔒 Authentification requise mais non configurée.")
            with st.expander("Comment configurer l'accès ?", expanded=True):
                st.markdown(
                    """
                    Cette application nécessite un mot de passe pour accéder aux données.
                    
                    **Administrateur :**
                    1. Allez dans les **Settings** de votre app sur Streamlit Cloud.
                    2. Ouvrez l'onglet **Secrets**.
                    3. Ajoutez la configuration suivante :
                    ```toml
                    [auth]
                    required = true
                    password = "votre_mot_de_passe_secret"
                    ```
                    
                    *Pour activer le mode démo sans mot de passe : set `required = false`.*
                    """
                )
            st.stop()  # Arrêt propre, pas d'erreur python
            return False

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        entered_password = str(st.session_state.get("password", ""))
        if hmac.compare_digest(entered_password, correct_password):
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
