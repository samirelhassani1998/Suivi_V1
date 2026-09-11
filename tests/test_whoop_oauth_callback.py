"""Tests du retour d'autorisation WHOOP, y compris quand il arrive hors de l'onglet Whoop."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.core import whoop_session
from app.core.whoop import WhoopError


class FakeQueryParams(dict):
    """Substitut de ``st.query_params`` : dictionnaire supportant la suppression."""


class FakeSessionState(dict):
    pass


@pytest.fixture
def stub_streamlit(monkeypatch):
    """Remplace l'espace de noms Streamlit utilisé par ``whoop_session``."""

    class StubStreamlit:
        def __init__(self):
            self.query_params = FakeQueryParams()
            self.session_state = FakeSessionState()
            self.switched_to = None

        def switch_page(self, page):
            self.switched_to = page

    stub = StubStreamlit()
    monkeypatch.setattr(whoop_session, "st", stub)
    return stub


def test_capture_oauth_callback_stores_code_and_cleans_url(stub_streamlit):
    stub_streamlit.query_params.update({"code": "abc123", "state": "xyz", "other": "keep"})

    whoop_session.capture_oauth_callback()

    pending = whoop_session.pending_callback()
    assert pending["code"] == "abc123"
    assert pending["state"] == "xyz"
    # L'URL est nettoyée pour éviter un rejeu du code au rafraîchissement.
    assert "code" not in stub_streamlit.query_params
    assert "state" not in stub_streamlit.query_params
    assert stub_streamlit.query_params["other"] == "keep"


def test_capture_oauth_callback_stores_provider_error(stub_streamlit):
    stub_streamlit.query_params.update({"error": "invalid_request", "error_description": "redirect_uri mismatch"})

    whoop_session.capture_oauth_callback()

    pending = whoop_session.pending_callback()
    assert pending["error"] == "invalid_request"
    assert "redirect_uri" in pending["error_description"]


def test_capture_oauth_callback_ignores_unrelated_query_strings(stub_streamlit):
    stub_streamlit.query_params.update({"page": "dashboard"})

    whoop_session.capture_oauth_callback()

    assert whoop_session.pending_callback() is None
    assert stub_streamlit.query_params["page"] == "dashboard"


def test_switch_to_whoop_page_happens_once_and_never_loops(stub_streamlit):
    stub_streamlit.query_params.update({"code": "abc123"})
    whoop_session.capture_oauth_callback()

    whoop_session.switch_to_whoop_page_if_pending("app/pages/Whoop.py")
    assert stub_streamlit.switched_to == "app/pages/Whoop.py"

    # Un second passage ne doit pas relancer la bascule : sinon l'application boucle.
    stub_streamlit.switched_to = None
    whoop_session.switch_to_whoop_page_if_pending("app/pages/Whoop.py")
    assert stub_streamlit.switched_to is None


def test_switch_to_whoop_page_does_nothing_without_pending_callback(stub_streamlit):
    whoop_session.switch_to_whoop_page_if_pending("app/pages/Whoop.py")
    assert stub_streamlit.switched_to is None


def test_detect_base_url_builds_public_url_from_proxy_headers(monkeypatch):
    monkeypatch.setattr(
        whoop_session,
        "_headers",
        lambda: {"Host": "mon-app.streamlit.app", "X-Forwarded-Proto": "https"},
    )
    assert whoop_session.detect_base_url() == "https://mon-app.streamlit.app"


def test_detect_base_url_assumes_http_for_localhost(monkeypatch):
    monkeypatch.setattr(whoop_session, "_headers", lambda: {"Host": "localhost:8501"})
    assert whoop_session.detect_base_url() == "http://localhost:8501"


def test_detect_base_url_falls_back_when_headers_are_unavailable(monkeypatch):
    monkeypatch.setattr(whoop_session, "_headers", lambda: {})
    assert whoop_session.detect_base_url(default="https://defaut") == "https://defaut"


def test_redirect_uri_candidates_cover_the_forms_whoop_treats_as_distinct(monkeypatch):
    monkeypatch.setattr(whoop_session, "detect_base_url", lambda *a, **k: "https://mon-app.streamlit.app")

    candidates = whoop_session.redirect_uri_candidates()

    # WHOOP compare caractère pour caractère : la barre oblique finale compte.
    assert candidates[0] == "https://mon-app.streamlit.app/"
    assert "https://mon-app.streamlit.app" in candidates
    assert "https://mon-app.streamlit.app/Whoop" in candidates


def test_entry_point_captures_callback_before_authentication_gate():
    """Non-régression : un retour sur la racine ne doit pas être perdu par st.stop()."""
    source = Path("Suivi_V1.py").read_text(encoding="utf-8")
    capture_position = source.index("capture_oauth_callback()")
    auth_position = source.index("if not check_password():")
    assert capture_position < auth_position


def test_whoop_page_exchanges_a_callback_received_on_another_page(monkeypatch):
    """Le code capté sur la racine est consommé dès l'ouverture de l'onglet Whoop.

    L'échange est simulé : faire dépendre l'assertion d'un échec réseau réel
    enverrait un code bidon aux serveurs WHOOP sur toute machine connectée, et
    rendrait le test tributaire de la connectivité.
    """
    attempts: list[str] = []

    def _refuse(credentials, code, **kwargs):
        attempts.append(code)
        raise WhoopError("Échec d'authentification WHOOP (HTTP 400).")

    monkeypatch.setattr("app.core.whoop.exchange_code_for_token", _refuse)

    at = AppTest.from_file("app/pages/Whoop.py")
    at.session_state["whoop_manual_credentials"] = {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "redirect_uri": "https://mon-app.streamlit.app/",
    }
    at.session_state[whoop_session.PENDING_KEY] = {
        "code": "code-recu-sur-la-racine",
        "state": "",
        "error": "",
        "error_description": "",
    }
    at.run(timeout=15)

    assert not at.exception
    # L'échange est bien tenté avec le code capté sur une autre page…
    assert attempts == ["code-recu-sur-la-racine"]
    # …le code en attente est consommé plutôt qu'ignoré…
    assert whoop_session.PENDING_KEY not in at.session_state or not at.session_state[whoop_session.PENDING_KEY]
    # …et le refus est présenté sans faire planter la page.
    assert any("WHOOP" in str(error.value) for error in at.error)


def test_whoop_page_explains_a_redirect_uri_rejection():
    at = AppTest.from_file("app/pages/Whoop.py")
    at.session_state["whoop_manual_credentials"] = {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "redirect_uri": "https://mon-app.streamlit.app/",
    }
    at.session_state[whoop_session.PENDING_KEY] = {
        "code": "",
        "state": "",
        "error": "invalid_request",
        "error_description": 'The "redirect_uri" parameter does not match any of the pre-registered redirect urls.',
    }
    at.run(timeout=15)

    assert not at.exception
    assert any("invalid_request" in str(error.value) for error in at.error)
    rendered = " ".join(str(w.value) for w in at.warning)
    assert "caractère pour caractère" in rendered
    # L'URL exacte à déclarer côté WHOOP est proposée telle quelle.
    assert any("https://mon-app.streamlit.app/" in str(code.value) for code in at.code)


def test_whoop_page_reports_access_denied_without_crashing():
    at = AppTest.from_file("app/pages/Whoop.py")
    at.session_state[whoop_session.PENDING_KEY] = {
        "code": "",
        "state": "",
        "error": "access_denied",
        "error_description": "The resource owner denied the request.",
    }
    at.run(timeout=15)

    assert not at.exception
    assert any("access_denied" in str(error.value) for error in at.error)


def test_whoop_page_rejects_a_state_that_does_not_match_the_request():
    at = AppTest.from_file("app/pages/Whoop.py")
    at.session_state["whoop_manual_credentials"] = {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "redirect_uri": "https://mon-app.streamlit.app/",
    }
    at.session_state["whoop_oauth_state"] = "etat-attendu"
    at.session_state[whoop_session.PENDING_KEY] = {
        "code": "code",
        "state": "etat-force-par-un-tiers",
        "error": "",
        "error_description": "",
    }
    at.run(timeout=15)

    assert not at.exception
    assert any("État OAuth inattendu" in str(error.value) for error in at.error)
    # Aucun jeton ne doit être créé à partir d'un retour non sollicité.
    assert not at.session_state["whoop_token"]


def test_entry_point_keeps_the_callback_even_when_authentication_blocks_the_app():
    """Scénario réel : WHOOP redirige vers la racine, protégée par mot de passe.

    ``check_password`` appelle ``st.stop()`` tant que l'accès n'est pas accordé.
    Le code d'autorisation doit malgré tout être mis de côté, sinon il disparaît
    de l'URL au premier rerun et la connexion ne peut jamais aboutir.
    """
    at = AppTest.from_file("Suivi_V1.py")
    at.query_params["code"] = "code-de-la-racine"
    at.query_params["state"] = "etat-emis"
    at.run(timeout=30)

    pending = at.session_state[whoop_session.PENDING_KEY]
    assert pending["code"] == "code-de-la-racine"
    assert pending["state"] == "etat-emis"
