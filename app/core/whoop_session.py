"""Glue Streamlit du flux OAuth WHOOP : détection d'URL et capture du retour.

Le retour d'autorisation WHOOP (``?code=...``) arrive sur la *Redirect URI*
déclarée côté WHOOP, qui n'est pas forcément l'URL de la page Whoop : si elle
pointe vers la racine, Streamlit affiche le Dashboard et le code serait perdu.
Ces fonctions sont donc appelées dès le point d'entrée de l'application, avant
tout rendu de page, puis consommées par la page Whoop.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit.errors import NoSessionContext, StreamlitAPIException

from app.core.whoop import DEFAULT_REDIRECT_URI

PENDING_KEY = "whoop_pending_auth"
AUTO_SWITCH_KEY = "whoop_auto_switch_done"
WHOOP_PAGE_PATH = "app/pages/Whoop.py"
WHOOP_PAGE_SLUG = "Whoop"
OAUTH_PARAM_KEYS = ("code", "state", "scope", "error", "error_description")


def _headers() -> dict[str, str]:
    """En-têtes de la requête courante ; vides hors runtime Streamlit."""
    try:
        return {str(key): str(value) for key, value in st.context.headers.items()}
    except Exception:
        return {}


def detect_base_url(default: str = DEFAULT_REDIRECT_URI) -> str:
    """Devine l'URL publique de l'application à partir des en-têtes HTTP.

    Streamlit Cloud place le domaine réel dans ``Host`` et le protocole dans
    ``X-Forwarded-Proto``. Sans en-tête exploitable, on retombe sur ``default``.
    """
    headers = _headers()
    host = (headers.get("X-Forwarded-Host") or headers.get("Host") or "").split(",")[0].strip()
    if not host:
        return default
    scheme = (headers.get("X-Forwarded-Proto") or "").split(",")[0].strip().lower()
    if scheme not in {"http", "https"}:
        scheme = "http" if host.startswith("localhost") or host.startswith("127.0.0.1") else "https"
    return f"{scheme}://{host}".rstrip("/")


def redirect_uri_candidates(base_url: str | None = None) -> list[str]:
    """URLs de redirection gérées par l'application, la racine en premier.

    WHOOP compare la Redirect URI caractère pour caractère : la variante avec
    barre oblique finale est donc une URL distincte, et non un détail cosmétique.
    L'application sait traiter le retour sur chacune de ces formes.
    """
    base = (base_url or detect_base_url()).rstrip("/")
    return [f"{base}/", base, f"{base}/{WHOOP_PAGE_SLUG}"]


def capture_oauth_callback() -> None:
    """Stocke un éventuel retour WHOOP en session et nettoie l'URL.

    Appelée à chaque exécution depuis le point d'entrée : quelle que soit la
    page affichée au retour de WHOOP, le code d'autorisation est conservé.
    """
    try:
        params = st.query_params
        code = params.get("code")
        state = params.get("state")
        error = params.get("error")
        description = params.get("error_description")
    except Exception:
        return

    if not code and not error:
        return

    st.session_state[PENDING_KEY] = {
        "code": str(code) if code else "",
        "state": str(state) if state else "",
        "error": str(error) if error else "",
        "error_description": str(description) if description else "",
    }
    st.session_state[AUTO_SWITCH_KEY] = False
    clear_oauth_params()


def clear_oauth_params() -> None:
    """Retire les paramètres OAuth de l'URL pour éviter un rejeu au rafraîchissement."""
    try:
        for key in OAUTH_PARAM_KEYS:
            if key in st.query_params:
                del st.query_params[key]
    except Exception:
        pass


def pending_callback() -> dict[str, Any] | None:
    pending = st.session_state.get(PENDING_KEY)
    return dict(pending) if isinstance(pending, dict) and (pending.get("code") or pending.get("error")) else None


def clear_pending_callback() -> None:
    st.session_state.pop(PENDING_KEY, None)


def switch_to_whoop_page_if_pending(page: Any = None) -> None:
    """Bascule une seule fois vers l'onglet Whoop quand un retour OAuth est en attente.

    ``st.switch_page`` signale le changement de page en levant une exception de
    contrôle : elle doit remonter intacte. Seules les erreurs d'API sont
    absorbées, auquel cas l'utilisateur rejoint l'onglet manuellement — le code
    d'autorisation reste en session et sera consommé à ce moment-là. Le garde-fou
    ``AUTO_SWITCH_KEY`` interdit toute boucle de redirection.
    """
    if pending_callback() is None or st.session_state.get(AUTO_SWITCH_KEY):
        return
    st.session_state[AUTO_SWITCH_KEY] = True
    try:
        st.switch_page(page or WHOOP_PAGE_PATH)
    except StreamlitAPIException:
        return
    except NoSessionContext:
        return
