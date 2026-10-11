"""Connexion durable : isolation OAuth, reprise de session et révocation partagée.

Le stockage est simulé : aucun appel PostgreSQL, WHOOP ou accès aux secrets réels.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.core import session_state, whoop_session
from app.core.whoop import WhoopCredentials, WhoopError, WhoopToken
from app.core.whoop_sync import WhoopSyncResult
from app.core.whoop_token_store import SavedConnection, TokenStoreError


CREDENTIALS = WhoopCredentials("server-id", "server-secret", "https://app.example/")


def _token(access="stored-access", refresh="stored-refresh", *, expired=False):
    expiry = datetime.now(timezone.utc) + timedelta(hours=-1 if expired else 2)
    return WhoopToken(access, refresh, expiry, ("offline", "read:workout"))


class MemoryStore:
    """Substitut du contrat stockage, sans tester à nouveau le backend SQL."""
    def __init__(self, token=None):
        self.connection = SavedConnection(token, "generation-1") if token else None
        self.sequence = 1
        self.loads = 0
        self.events = []
        self.failure = None

    def _check(self):
        if self.failure:
            raise self.failure

    def read(self):
        self._check()
        self.loads += 1
        return self.connection

    def save_new(self, token):
        self._check()
        self.sequence += 1
        self.connection = SavedConnection(token, f"generation-{self.sequence}")
        self.events.append("save")
        return self.connection

    def delete(self):
        self._check()
        self.connection = None
        self.events.append("delete")

    def freshen(self, refresh):
        self._check()
        if self.connection is None:
            return None
        token = refresh(self.connection.token)
        self.connection = SavedConnection(token, self.connection.generation)
        self.events.append("refresh-saved")
        return self.connection


class StreamlitStub:
    def __init__(self):
        self.session_state = {}
        self.messages = []

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def _message(self, text, *args, **kwargs):
        self.messages.append(str(text))

    error = warning = success = caption = info = write = _message


@pytest.fixture
def isolated_session(monkeypatch):
    stub = StreamlitStub()
    monkeypatch.setattr(whoop_session, "st", stub)
    monkeypatch.setattr(session_state, "st", stub)
    session_state.ensure_whoop_defaults()
    return stub


def _seed_data(state):
    state["whoop_daily"] = pd.DataFrame({"Date": [pd.Timestamp("2026-10-01")], "Strain": [10.]})
    state["whoop_workouts"] = pd.DataFrame({"Date": [pd.Timestamp("2026-10-01")], "Sport": ["boxing"]})
    state["whoop_profile"] = {"first_name": "Previous connection"}
    state["whoop_body"] = {"max_heart_rate": 190}
    state["whoop_sync_warnings"] = ["Previous warning"]


def _result():
    return WhoopSyncResult(
        daily=pd.DataFrame({"Date": [pd.Timestamp.now().normalize()], "Strain": [11.]}),
        workouts=pd.DataFrame(), profile={"first_name": "Restored account"},
    )


def test_stored_token_adopts_database_connection_in_a_new_session(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    token = whoop_session.stored_token()
    assert token.to_dict() == store.connection.token.to_dict()
    assert isolated_session.session_state["whoop_token"] == token.to_dict()
    assert isolated_session.session_state["whoop_connection_restored"]
    assert not isolated_session.session_state["whoop_oauth_state"]


def test_stored_token_reloads_and_purges_data_when_another_session_disconnects(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    _seed_data(isolated_session.session_state)
    store.delete()
    assert whoop_session.stored_token() is None
    assert not isolated_session.session_state["whoop_token"]
    assert isolated_session.session_state["whoop_daily"].empty
    assert isolated_session.session_state["whoop_workouts"].empty
    assert not isolated_session.session_state["whoop_profile"]
    assert not isolated_session.session_state["whoop_sync_warnings"]


def test_a_new_connection_generation_cannot_inherit_old_account_data(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    _seed_data(isolated_session.session_state)
    replacement = _token("replacement-access", "replacement-refresh")
    store.save_new(replacement)
    assert whoop_session.stored_token().access_token == "replacement-access"
    assert isolated_session.session_state["whoop_daily"].empty
    assert not isolated_session.session_state["whoop_profile"]


def test_database_failure_never_falls_back_to_a_stale_session_token(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    store.failure = TokenStoreError("Le coffre WHOOP est indisponible.")
    store.failure.__cause__ = RuntimeError("postgresql://owner:secret@private.example/db")
    assert whoop_session.stored_token() is None
    whoop_session.render_connection_status()
    rendered = " ".join(isolated_session.messages)
    assert "private.example" not in rendered
    assert "owner:secret" not in rendered


def test_failed_save_does_not_claim_a_connected_session(monkeypatch, isolated_session):
    store = MemoryStore()
    store.failure = TokenStoreError("Le coffre WHOOP est indisponible.")
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    assert not whoop_session.store_token(_token())
    assert not isolated_session.session_state["whoop_token"]
    assert store.connection is None


def test_disconnect_only_succeeds_after_shared_deletion(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    _seed_data(isolated_session.session_state)
    assert whoop_session.disconnect_whoop()
    assert store.connection is None
    assert not isolated_session.session_state["whoop_token"]
    assert isolated_session.session_state["whoop_daily"].empty


def test_disconnect_failure_is_not_reported_as_success(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    store.failure = TokenStoreError("Le coffre WHOOP est indisponible.")
    assert not whoop_session.disconnect_whoop()
    assert store.connection is not None


def test_sync_uses_database_token_and_persists_refresh_before_fetch(monkeypatch, isolated_session):
    store = MemoryStore(_token(expired=True))
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    refreshed = _token("stored-access", "rotated-refresh")  # Access unchanged, refresh rotated.
    received = []

    def refresh(credentials, old):
        assert old.refresh_token == "stored-refresh"
        assert credentials == CREDENTIALS
        return refreshed

    def fetch(token, **kwargs):
        assert store.events[-1] == "refresh-saved"
        assert store.connection.token.refresh_token == "rotated-refresh"
        received.append(token)
        return _result()

    monkeypatch.setattr(whoop_session, "ensure_fresh_token", refresh)
    monkeypatch.setattr(whoop_session, "fetch_whoop_data", fetch)
    assert whoop_session.run_whoop_sync(CREDENTIALS, _token("stale-tab-token", "stale-tab-refresh"), 30)
    assert received == [refreshed]
    assert isolated_session.session_state["whoop_token"]["refresh_token"] == "rotated-refresh"
    assert isolated_session.session_state["whoop_profile"]["first_name"] == "Restored account"


@pytest.mark.parametrize("change", ["disconnect", "replace"])
def test_inflight_sync_cannot_restore_data_after_connection_changes(monkeypatch, isolated_session, change):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()

    def fetch(token, **kwargs):
        if change == "disconnect":
            store.delete()
        else:
            store.save_new(_token("new-account", "new-account-refresh"))
        return _result()

    monkeypatch.setattr(whoop_session, "fetch_whoop_data", fetch)
    assert not whoop_session.run_whoop_sync(CREDENTIALS, _token(), 30)
    assert isolated_session.session_state["whoop_daily"].empty
    assert not isolated_session.session_state["whoop_profile"]


def test_network_failure_keeps_refreshed_connection_and_previous_measurements(monkeypatch, isolated_session):
    store = MemoryStore(_token(expired=True))
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    _seed_data(isolated_session.session_state)
    old_data = isolated_session.session_state["whoop_daily"].copy()
    refreshed = _token("new-access", "new-refresh")
    monkeypatch.setattr(whoop_session, "ensure_fresh_token", lambda *args: refreshed)

    def unavailable(*args, **kwargs):
        raise WhoopError("Quota WHOOP atteint (429).")

    monkeypatch.setattr(whoop_session, "fetch_whoop_data", unavailable)
    assert not whoop_session.run_whoop_sync(CREDENTIALS, _token(), 30)
    assert store.connection.token == refreshed
    pd.testing.assert_frame_equal(isolated_session.session_state["whoop_daily"], old_data)


def test_auto_sync_is_attempted_only_once_after_restoration_even_on_failure(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    calls = []
    monkeypatch.setattr(whoop_session, "run_whoop_sync", lambda *args: calls.append(args) or False)
    whoop_session.stored_token()
    whoop_session.sync_restored_connection(CREDENTIALS)
    whoop_session.stored_token()
    whoop_session.sync_restored_connection(CREDENTIALS)
    assert len(calls) == 1


@pytest.mark.parametrize("page", ["app/pages/Whoop.py", "app/pages/Boxe.py"])
def test_independent_page_sessions_restore_one_shared_connection_and_observe_disconnect(monkeypatch, page):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    monkeypatch.setattr(whoop_session, "fetch_whoop_data", lambda *args, **kwargs: _result())
    first = AppTest.from_file("app/pages/Whoop.py")
    second = AppTest.from_file(page)
    for app in (first, second):
        app.session_state["password_correct"] = True
        app.run(timeout=30)
        assert not app.exception
        assert app.session_state["whoop_token"]["access_token"] == "stored-access"
    next(button for button in first.button if button.label == "Déconnecter WHOOP").click().run(timeout=30)
    assert not first.exception
    assert store.connection is None
    second.run(timeout=30)
    assert not second.exception
    assert not second.session_state["whoop_token"]
    assert second.session_state["whoop_daily"].empty
    assert not any(button.label == "Synchroniser maintenant" for button in second.button)


def _config():
    return {
        "auth": {"required": True, "password": "synthetic-app-password"},
        "whoop": {"client_id": "server-id", "client_secret": "server-secret", "redirect_uri": "https://app.example/"},
        "whoop_persistence": {
            "enabled": True, "mode": "personal", "database_url": "postgresql://unused.invalid/test",
            "encryption_key": "synthetic-test-key", "namespace": "test-personal",
        },
    }


@pytest.mark.parametrize("case", ["unauthenticated", "demo", "no_password", "wrong_mode", "truthy_enabled"])
def test_persistence_configuration_refuses_unauthorized_access_before_store_construction(monkeypatch, isolated_session, case):
    config = _config()
    isolated_session.session_state["password_correct"] = True
    if case == "unauthenticated":
        isolated_session.session_state["password_correct"] = False
    elif case == "demo":
        config["auth"]["required"] = False
    elif case == "no_password":
        config["auth"]["password"] = ""
    elif case == "wrong_mode":
        config["whoop_persistence"]["mode"] = "shared"
    elif case == "truthy_enabled":
        config["whoop_persistence"]["enabled"] = "true"
    calls = []
    monkeypatch.setattr(whoop_session, "secrets_mapping", lambda: config)
    monkeypatch.setattr(whoop_session, "PostgresTokenStore", lambda **kwargs: calls.append(kwargs))
    isolated_session.session_state["whoop_token"] = _token("old-session-access").to_dict()
    assert whoop_session.stored_token() is None
    assert not calls
    assert not isolated_session.session_state["whoop_token"]


def test_personal_store_uses_server_credentials_and_is_bound_to_client_id(monkeypatch, isolated_session):
    config = _config()
    isolated_session.session_state["password_correct"] = True
    isolated_session.session_state["whoop_request_offline"] = False
    isolated_session.session_state["whoop_manual_credentials"] = {
        "client_id": "untrusted-tab-id", "client_secret": "untrusted-tab-secret", "redirect_uri": "https://another.example/",
    }
    calls = []
    fake_store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "secrets_mapping", lambda: config)

    def construct(**kwargs):
        calls.append(kwargs)
        return fake_store

    monkeypatch.setattr(whoop_session, "PostgresTokenStore", construct)
    assert whoop_session._persistent_store() is fake_store
    credentials = whoop_session.resolve_credentials()
    assert credentials.client_id == "server-id"
    assert credentials.client_secret == "server-secret"
    assert credentials.redirect_uri == "https://app.example/"
    assert "offline" in credentials.scopes
    config["whoop"]["client_id"] = "second-application-id"
    whoop_session._persistent_store()
    assert calls[0]["namespace"] != calls[1]["namespace"]


def test_legacy_mode_does_not_open_storage_or_change_manual_credentials(monkeypatch, isolated_session):
    monkeypatch.setattr(whoop_session, "secrets_mapping", lambda: {})
    monkeypatch.setattr(whoop_session, "PostgresTokenStore", lambda **kwargs: pytest.fail("Legacy mode must not open a database"))
    isolated_session.session_state["whoop_manual_credentials"] = {
        "client_id": "manual-id", "client_secret": "manual-secret", "redirect_uri": "https://manual.example/",
    }
    isolated_session.session_state["whoop_request_offline"] = False
    assert whoop_session.store_token(_token())
    assert whoop_session.stored_token().access_token == "stored-access"
    credentials = whoop_session.resolve_credentials()
    assert credentials.client_id == "manual-id"
    assert "offline" not in credentials.scopes


def test_disabling_persistence_does_not_expose_previously_restored_private_token(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    _seed_data(isolated_session.session_state)
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: None)
    assert whoop_session.stored_token() is None
    assert isolated_session.session_state["whoop_daily"].empty


def test_empty_store_preserves_an_oauth_request_still_waiting_for_callback(monkeypatch, isolated_session):
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: MemoryStore())
    isolated_session.session_state["whoop_oauth_state"] = "pending-authorization-nonce"
    assert whoop_session.stored_token() is None
    assert isolated_session.session_state["whoop_oauth_state"] == "pending-authorization-nonce"


def test_authorized_callback_is_persisted_and_restored_by_a_fresh_boxing_session(monkeypatch):
    config = _config()
    store = MemoryStore()
    exchanges = []
    syncs = []
    monkeypatch.setattr(whoop_session, "secrets_mapping", lambda: config)
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)

    def exchange(credentials, code, **kwargs):
        exchanges.append(code)
        return _token("oauth-saved-access", "oauth-saved-refresh")

    def fetch(token, **kwargs):
        syncs.append(token.access_token)
        return _result()

    monkeypatch.setattr("app.core.whoop.exchange_code_for_token", exchange)
    monkeypatch.setattr(whoop_session, "fetch_whoop_data", fetch)
    original = AppTest.from_file("app/pages/Whoop.py")
    original.session_state["password_correct"] = True
    original.session_state["whoop_oauth_state"] = "expected-state"
    original.session_state[whoop_session.PENDING_KEY] = {"code": "valid-code", "state": "expected-state", "error": ""}
    original.run(timeout=30)
    assert not original.exception
    assert exchanges == ["valid-code"]
    assert store.connection.token.access_token == "oauth-saved-access"

    restored = AppTest.from_file("app/pages/Boxe.py")
    restored.session_state["password_correct"] = True
    restored.run(timeout=30)
    assert not restored.exception
    assert restored.session_state["whoop_token"]["access_token"] == "oauth-saved-access"
    count_after_restore = len(syncs)
    restored.run(timeout=30)
    assert len(syncs) == count_after_restore
    assert exchanges == ["valid-code"]


def test_storage_failure_during_autosync_does_not_retry_for_the_same_generation(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    monkeypatch.setattr(whoop_session, "fetch_whoop_data", lambda *args, **kwargs: _result())
    attempts = []
    original_freshen = store.freshen

    def counted_freshen(refresh):
        attempts.append(store.connection.generation if store.connection else None)
        return original_freshen(refresh)

    monkeypatch.setattr(store, "freshen", counted_freshen)
    whoop_session.stored_token()
    generation = store.connection.generation
    store.failure = TokenStoreError("Le coffre WHOOP est indisponible.")
    whoop_session.sync_restored_connection(CREDENTIALS)
    assert attempts == [generation]
    assert isolated_session.session_state["whoop_autosync_attempted_generation"] == generation
    assert not isolated_session.session_state["whoop_token"]

    # La reprise de lecture réadopte la même génération sans relancer la
    # requête qui vient d'échouer. Une action explicite reste possible.
    store.failure = None
    assert whoop_session.stored_token() is not None
    whoop_session.sync_restored_connection(CREDENTIALS)
    assert attempts == [generation]
    assert whoop_session.run_whoop_sync(CREDENTIALS, _token(), 30)
    assert attempts == [generation, generation]

    # Un nouveau consentement possède sa propre tentative de reprise.
    store.save_new(_token("replacement-access", "replacement-refresh"))
    whoop_session.stored_token()
    whoop_session.sync_restored_connection(CREDENTIALS)
    assert attempts[-1] == store.connection.generation
    assert attempts[-1] != generation
    assert len(attempts) == 3


def test_failed_oauth_save_remains_visible_after_empty_store_is_read(monkeypatch):
    store = MemoryStore()
    message = "La connexion durable n’a pas pu être conservée."
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    monkeypatch.setattr(whoop_session, "secrets_mapping", _config)
    monkeypatch.setattr("app.core.whoop.exchange_code_for_token", lambda *args, **kwargs: _token())

    def failed_save(token):
        raise TokenStoreError(message)

    monkeypatch.setattr(store, "save_new", failed_save)
    app = AppTest.from_file("app/pages/Whoop.py")
    app.session_state["password_correct"] = True
    app.session_state["whoop_oauth_state"] = "expected-state"
    app.session_state[whoop_session.PENDING_KEY] = {"code": "valid-code", "state": "expected-state", "error": ""}
    app.run(timeout=30)
    assert not app.exception
    assert store.loads > 0
    assert store.connection is None
    assert not app.session_state["whoop_token"]
    assert app.session_state["whoop_link_error"] == message
    assert any(message in str(error.value) for error in app.error)
    assert not any("Compte WHOOP connecté" in str(success.value) for success in app.success)
    app.run(timeout=30)
    assert not app.exception
    assert any(message in str(error.value) for error in app.error)


@pytest.mark.parametrize("page", ["app/pages/Whoop.py", "app/pages/Boxe.py"])
@pytest.mark.parametrize("change", ["disconnect", "replace"])
def test_page_discards_old_account_render_after_connection_changes_during_manual_sync(monkeypatch, page, change):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    calls = []

    def fetch(token, **kwargs):
        calls.append(token.access_token)
        if change == "disconnect":
            store.delete()
        else:
            store.save_new(_token("replacement-access", "replacement-refresh"))
        return _result()

    monkeypatch.setattr(whoop_session, "fetch_whoop_data", fetch)
    app = AppTest.from_file(page)
    app.session_state["password_correct"] = True
    app.session_state["whoop_token"] = store.connection.token.to_dict()
    app.session_state["whoop_persistence_generation"] = store.connection.generation
    app.session_state["whoop_autosync_attempted_generation"] = store.connection.generation
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=14)
    app.session_state["whoop_daily"] = pd.DataFrame({
        "Date": dates, "Récupération (%)": [60 + day % 7 for day in range(14)],
        "HRV (ms)": 45., "FC repos (bpm)": 55., "Sommeil (heures)": 7.5,
        "Efficacité sommeil (%)": 92., "Heure de coucher": -1.,
        "Calories (kcal)": 2700., "Strain": 11.,
    })
    app.session_state["whoop_workouts"] = pd.DataFrame({
        "Date": dates[::3], "Début": dates[::3] + pd.Timedelta(hours=18),
        "Sport": "boxing", "Durée (min)": 55., "Strain séance": 12.,
        "Calories séance (kcal)": 500., "FC moyenne (bpm)": 140., "FC max (bpm)": 180.,
    })
    app.session_state["whoop_body"] = {"max_heart_rate": 190}
    app.run(timeout=30)
    assert not app.exception
    assert app.tabs
    assert app.metric
    assert not calls

    next(button for button in app.button if button.label == "Synchroniser maintenant").click().run(timeout=30)
    assert not app.exception
    assert calls == ["stored-access"]
    assert app.session_state["whoop_daily"].empty
    assert app.session_state["whoop_workouts"].empty
    # The session purge alone is insufficient: local DataFrame snapshots used
    # later in this same run must not render the old account's analyses.
    assert not app.tabs
    assert not [metric for metric in app.metric if metric.label != "Dernière synchro"]
    assert not app.get("plotly_chart")


def test_legacy_connection_can_resume_after_disabling_an_unavailable_vault(monkeypatch, isolated_session):
    store = MemoryStore(_token())
    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: store)
    whoop_session.stored_token()
    _seed_data(isolated_session.session_state)
    store.failure = TokenStoreError("Le coffre WHOOP est indisponible.")
    assert not whoop_session.store_token(_token())
    assert isolated_session.session_state["whoop_link_error"]
    assert isolated_session.session_state["whoop_persistence_error"]

    monkeypatch.setattr(whoop_session, "_persistent_store", lambda: None)
    assert whoop_session.stored_token() is None
    assert isolated_session.session_state["whoop_daily"].empty
    assert not isolated_session.session_state.get("whoop_link_error")
    assert not isolated_session.session_state.get("whoop_persistence_error")
    assert whoop_session.store_token(_token("legacy-access", "legacy-refresh"))
    assert whoop_session.stored_token().access_token == "legacy-access"
    assert store.connection.token.access_token == "stored-access"
