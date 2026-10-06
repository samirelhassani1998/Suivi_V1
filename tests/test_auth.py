from __future__ import annotations

import pytest
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest

from app import auth


def _auth_app() -> AppTest:
    return AppTest.from_string(
        "from app.auth import check_password\n"
        "import streamlit as st\n"
        "if check_password():\n"
        "    st.success('Accès autorisé')\n",
        default_timeout=10,
    )


@pytest.mark.parametrize("password", ["correct-password", "café-sécurisé-🔒"])
def test_auth_accepts_correct_password_and_removes_input_from_session(password):
    at = _auth_app()
    at.secrets["auth"] = {"required": True, "password": password}
    at.run()
    at.text_input[0].input(password).run()

    assert not at.exception
    assert [message.value for message in at.success] == ["Accès autorisé"]
    assert at.session_state["password_correct"] is True
    assert "password" not in at.session_state


@pytest.mark.parametrize("password", ["wrong-password", "café-incorrect-🔒"])
def test_auth_rejects_incorrect_password_without_exception(password):
    at = _auth_app()
    at.secrets["auth"] = {"required": True, "password": "correct-password"}
    at.run()
    at.text_input[0].input(password).run()

    assert not at.exception
    assert not at.success
    assert at.session_state["password_correct"] is False
    assert [message.value for message in at.error] == ["Mot de passe incorrect."]


def test_auth_without_secret_file_stays_blocked_with_french_setup_message(monkeypatch):
    secrets = Secrets()
    monkeypatch.setattr(secrets, "_parse_file_path", lambda path: ({}, False))
    monkeypatch.setattr(auth.st, "secrets", secrets)
    at = _auth_app()
    at.session_state["password_correct"] = True
    at.run()

    assert not at.exception
    assert not at.error
    assert not at.success
    assert [message.value for message in at.warning] == [
        "🔒 Authentification requise mais non configurée."
    ]
    assert "Comment configurer l'accès ?" in [expander.label for expander in at.expander]


def test_auth_with_no_configured_password_does_not_trust_old_session():
    at = _auth_app()
    at.secrets["auth"] = {"required": True}
    at.session_state["password_correct"] = True
    at.run()

    assert not at.exception
    assert not at.success
    assert any("non configurée" in message.value for message in at.warning)


def test_auth_accepts_legacy_top_level_password():
    at = _auth_app()
    at.secrets["password"] = "ancien-café"
    at.run()
    at.text_input[0].input("ancien-café").run()

    assert not at.exception
    assert [message.value for message in at.success] == ["Accès autorisé"]


def test_auth_demo_access_requires_explicit_configuration():
    at = _auth_app()
    at.secrets["auth"] = {"required": False}
    at.run()

    assert not at.exception
    assert [message.value for message in at.success] == ["Accès autorisé"]
    assert any("Mode DÉMO actif" in message.value for message in at.warning)
