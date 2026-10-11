"""Coffre chiffré : tests hors ligne et concurrence sur PostgreSQL optionnel."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

from cryptography.fernet import Fernet
import psycopg
import pytest

from app.core.whoop import WhoopError, WhoopToken
from app.core.whoop_token_store import PostgresTokenStore, TokenStoreError


TOKEN = WhoopToken(
    access_token="secret-access-token",
    refresh_token="secret-refresh-token",
    expires_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    scopes=("offline", "read:recovery"),
)
LOCAL_URL = "postgresql://test:database-password@127.0.0.1/test"


class MemoryConnection:
    """Transport SQL minimal avec rollback, réservé aux tests sans réseau."""

    def __init__(self, backend):
        self.backend = backend
        self.before = dict(backend.rows)
        self.row = None

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is not None or self.backend.fail_commit:
            self.backend.rows = self.before
        if self.backend.fail_commit:
            raise RuntimeError("database-password secret-access-token")

    def cursor(self):
        return self

    def execute(self, query, params=()):
        self.backend.queries.append(query)
        if query.startswith(("SET LOCAL", "SELECT pg_advisory")):
            return
        if query.startswith("SELECT payload"):
            value = self.backend.rows.get(params[0])
            self.row = (value,) if value is not None else None
        elif query.startswith("INSERT INTO"):
            self.backend.rows[params[0]] = params[1]
        elif query.startswith("UPDATE"):
            self.backend.rows[params[1]] = params[0]
        elif query.startswith("DELETE FROM"):
            self.backend.rows.pop(params[0], None)
        else:
            raise AssertionError(query)

    def fetchone(self):
        return self.row


class MemoryBackend:
    def __init__(self):
        self.rows = {}
        self.queries = []
        self.options = {}
        self.fail_commit = False

    def connect(self, url, **kwargs):
        self.options = kwargs
        return MemoryConnection(self)


@pytest.fixture
def memory_store():
    backend = MemoryBackend()
    key = Fernet.generate_key().decode()
    store = PostgresTokenStore(LOCAL_URL, key, "personal:client-a", connect=backend.connect)
    return store, backend, key


def test_encrypted_round_trip_and_new_generation(memory_store):
    store, backend, key = memory_store
    saved = store.save_new(TOKEN)
    assert store.read() == saved
    ciphertext = backend.rows["personal:client-a"]
    assert TOKEN.access_token not in ciphertext
    assert TOKEN.refresh_token not in ciphertext
    decoded = json.loads(Fernet(key).decrypt(ciphertext))
    assert decoded["namespace"] == "personal:client-a"
    assert decoded["version"] == 1
    assert decoded["token"]["refresh_token"] == TOKEN.refresh_token
    assert store.save_new(TOKEN).generation != saved.generation
    assert "secret" not in repr(saved)
    assert "database-password" not in repr(store)
    assert key not in repr(store)
    assert backend.options["connect_timeout"] == 5
    assert backend.options["autocommit"] is False
    assert not any("CREATE " in query for query in backend.queries)


def test_refresh_persists_rotated_refresh_even_with_same_access_token(memory_store):
    store, _, _ = memory_store
    saved = store.save_new(TOKEN)
    rotated = replace(TOKEN, refresh_token="new-refresh", expires_at=TOKEN.expires_at + timedelta(hours=1))
    renewed = store.freshen(lambda current: rotated if current == TOKEN else None)
    assert renewed.token == rotated
    assert renewed.generation == saved.generation
    assert store.read() == renewed


def test_unchanged_refresh_does_not_rewrite_ciphertext(memory_store):
    store, backend, _ = memory_store
    saved = store.save_new(TOKEN)
    before = dict(backend.rows)
    assert store.freshen(lambda current: current) == saved
    assert backend.rows == before


def test_delete_is_idempotent_and_freshen_never_recreates(memory_store):
    store, _, _ = memory_store
    store.save_new(TOKEN)
    store.delete()
    store.delete()
    assert store.read() is None
    assert store.freshen(lambda _: pytest.fail("Aucun refresh après suppression")) is None


@pytest.mark.parametrize("operation", ["read", "save_new", "freshen"])
def test_wrong_key_never_overwrites_existing_connection(memory_store, operation):
    store, backend, _ = memory_store
    store.save_new(TOKEN)
    before = dict(backend.rows)
    wrong = PostgresTokenStore(LOCAL_URL, Fernet.generate_key().decode(), "personal:client-a", connect=backend.connect)
    with pytest.raises(TokenStoreError, match="illisible"):
        if operation == "save_new":
            wrong.save_new(TOKEN)
        elif operation == "freshen":
            wrong.freshen(lambda _: pytest.fail("Le déchiffrement doit précéder WHOOP"))
        else:
            wrong.read()
    assert backend.rows == before
    # Une suppression explicitement demandée reste possible, même si la clé est perdue.
    wrong.delete()
    assert store.read() is None


def test_ciphertext_cannot_be_moved_to_another_namespace(memory_store):
    store, backend, key = memory_store
    store.save_new(TOKEN)
    backend.rows["personal:client-b"] = backend.rows["personal:client-a"]
    other = PostgresTokenStore(LOCAL_URL, key, "personal:client-b", connect=backend.connect)
    with pytest.raises(TokenStoreError, match="illisible"):
        other.read()
    assert store.read().token == TOKEN


@pytest.mark.parametrize("mutation", ["ciphertext", "version", "generation", "naive_expiry", "token_type", "refresh_type"])
def test_corrupt_or_unsupported_payload_is_rejected_without_overwrite(memory_store, mutation):
    store, backend, key = memory_store
    store.save_new(TOKEN)
    raw = json.loads(Fernet(key).decrypt(backend.rows["personal:client-a"]))
    if mutation == "version":
        raw["version"] = True
    elif mutation == "generation":
        raw["generation"] = "not-a-generation"
    elif mutation == "naive_expiry":
        raw["token"]["expires_at"] = "2026-01-01T00:00:00"
    elif mutation == "token_type":
        raw["token"]["token_type"] = None
    elif mutation == "refresh_type":
        raw["token"]["refresh_token"] = ["secret"]
    ciphertext = "broken!" if mutation == "ciphertext" else Fernet(key).encrypt(json.dumps(raw).encode()).decode()
    backend.rows["personal:client-a"] = ciphertext
    with pytest.raises(TokenStoreError, match="illisible"):
        store.save_new(TOKEN)
    assert backend.rows["personal:client-a"] == ciphertext


@pytest.mark.parametrize("invalid", [
    replace(TOKEN, refresh_token=None), replace(TOKEN, refresh_token=" "),
    replace(TOKEN, access_token=""), replace(TOKEN, expires_at=None),
    replace(TOKEN, expires_at=datetime(2026, 1, 1)),
])
def test_invalid_token_cannot_replace_valid_connection(memory_store, invalid):
    store, _, _ = memory_store
    original = store.save_new(TOKEN)
    with pytest.raises(TokenStoreError, match="renouvelable"):
        store.save_new(invalid)
    with pytest.raises(TokenStoreError, match="renouvelable"):
        store.freshen(lambda _: invalid)
    assert store.read() == original


def test_refresh_failure_does_not_retry_or_change_pair(memory_store):
    store, _, _ = memory_store
    original = store.save_new(TOKEN)
    calls = []

    def fail(token):
        calls.append(token)
        raise RuntimeError("secret-refresh-token")

    with pytest.raises(TokenStoreError) as error:
        store.freshen(fail)
    assert "secret-refresh-token" not in str(error.value)
    assert len(calls) == 1
    assert store.read() == original


def test_commit_failure_is_not_reported_as_saved(memory_store):
    store, backend, _ = memory_store
    original = store.save_new(TOKEN)
    backend.fail_commit = True
    with pytest.raises(TokenStoreError) as error:
        store.freshen(lambda token: replace(token, refresh_token="rotated"))
    assert "database-password" not in str(error.value)
    backend.fail_commit = False
    assert store.read() == original


def test_whoop_functional_error_is_preserved_and_connection_unchanged(memory_store):
    store, _, _ = memory_store
    original = store.save_new(TOKEN)
    error = WhoopError("Autorisation WHOOP expirée : reconnectez votre compte.")

    def renew(_):
        raise error

    with pytest.raises(WhoopError) as caught:
        store.freshen(renew)
    assert caught.value is error
    assert store.read() == original


@pytest.mark.parametrize("url", [
    "sqlite:///local.db", "postgresql:///db", "postgresql://remote.invalid/db?sslmode=disable",
    "postgresql://remote.invalid/db?sslmode=prefer",
    "postgresql://localhost/db?hostaddr=192.0.2.1&sslmode=disable",
])
def test_unsafe_database_configuration_is_rejected(url):
    with pytest.raises(TokenStoreError, match="Configuration"):
        PostgresTokenStore(url, Fernet.generate_key().decode(), "personal:client-a")


def test_remote_connections_require_tls_and_connection_errors_hide_credentials():
    options = {}

    def failing_connect(url, **kwargs):
        options.update(kwargs)
        raise RuntimeError(url)

    url = "postgresql://admin:never-disclose@remote.invalid/db"
    store = PostgresTokenStore(url, Fernet.generate_key().decode(), "personal:client-a", connect=failing_connect)
    with pytest.raises(TokenStoreError) as error:
        store.read()
    assert options["sslmode"] == "require"
    assert "never-disclose" not in str(error.value)
    assert "remote.invalid" not in str(error.value)


@pytest.fixture
def postgres_store():
    """Base de TEST dédiée seulement : le fixture installe le schéma privé."""
    url = os.environ.get("WHOOP_TEST_DATABASE_URL")
    if not url:
        pytest.skip("WHOOP_TEST_DATABASE_URL absent : concurrence PostgreSQL non exécutée")
    sql = Path(__file__).parents[1] / "docs/sql/whoop_token_store.sql"
    with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
        conn.execute(sql.read_text())
    key = Fernet.generate_key().decode()
    namespace = f"test:{uuid4()}"
    store = PostgresTokenStore(url, key, namespace)
    try:
        yield store, url, key, namespace
    finally:
        store.delete()


def test_postgres_round_trip_survives_new_store_and_encrypts_both_tokens(postgres_store):
    store, url, key, namespace = postgres_store
    saved = store.save_new(TOKEN)
    restored = PostgresTokenStore(url, key, namespace)
    assert restored.read() == saved
    with psycopg.connect(url) as conn:
        ciphertext = conn.execute(
            "SELECT payload FROM whoop_private.oauth_connections WHERE namespace = %s", (namespace,)
        ).fetchone()[0]
    assert TOKEN.access_token not in ciphertext
    assert TOKEN.refresh_token not in ciphertext
    restored.delete()
    assert store.read() is None


def _wait_for_database_lock(url, application_name):
    """Observer le blocage réel : une simple barrière de threads ne suffit pas."""
    deadline = time.monotonic() + 5
    with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
        while time.monotonic() < deadline:
            waiting = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                "WHERE application_name = %s AND wait_event = 'advisory')",
                (application_name,),
            ).fetchone()[0]
            if waiting:
                return
            time.sleep(0.01)
    pytest.fail("La seconde transaction n'a pas attendu le verrou PostgreSQL")


def _named_store(url, key, namespace):
    application_name = f"whoop-test-{uuid4()}"

    def connect(dsn, **kwargs):
        return psycopg.connect(dsn, application_name=application_name, **kwargs)

    return PostgresTokenStore(url, key, namespace, connect=connect), application_name


def test_postgres_concurrent_sessions_refresh_only_latest_pair(postgres_store):
    store, url, key, namespace = postgres_store
    saved = store.save_new(TOKEN)
    second, application_name = _named_store(url, key, namespace)
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    seen_refreshes = []
    rotated = replace(TOKEN, refresh_token="rotated-once", expires_at=datetime.now(timezone.utc) + timedelta(hours=1))

    def renew(current):
        if not current.is_expired():
            return current
        seen_refreshes.append(current.refresh_token)
        entered.set()
        assert release.wait(5), "Le test n'a pas libéré le refresh"
        return rotated

    def second_sync():
        second_started.set()
        return second.freshen(renew)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_job = pool.submit(store.freshen, renew)
        try:
            assert entered.wait(5)
            second_job = pool.submit(second_sync)
            assert second_started.wait(5)
            _wait_for_database_lock(url, application_name)
        finally:
            release.set()
        first_result = first_job.result(timeout=10)
        second_result = second_job.result(timeout=10)
    assert seen_refreshes == [TOKEN.refresh_token]
    assert first_result == second_result == store.read()
    assert first_result.token == rotated
    assert first_result.generation == saved.generation


def test_postgres_disconnect_during_refresh_cannot_resurrect_connection(postgres_store):
    store, url, key, namespace = postgres_store
    store.save_new(TOKEN)
    other, application_name = _named_store(url, key, namespace)
    entered, release, deletion_started = threading.Event(), threading.Event(), threading.Event()

    def renew(token):
        entered.set()
        assert release.wait(5)
        return replace(token, refresh_token="rotated-before-deletion")

    def disconnect():
        deletion_started.set()
        other.delete()

    with ThreadPoolExecutor(max_workers=2) as pool:
        refresh_job = pool.submit(store.freshen, renew)
        try:
            assert entered.wait(5)
            delete_job = pool.submit(disconnect)
            assert deletion_started.wait(5)
            _wait_for_database_lock(url, application_name)
        finally:
            release.set()
        refresh_job.result(timeout=10)
        delete_job.result(timeout=10)
    assert store.read() is None
    assert other.freshen(lambda _: pytest.fail("La suppression est durable")) is None
