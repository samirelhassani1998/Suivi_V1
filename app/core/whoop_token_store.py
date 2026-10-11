"""Coffre WHOOP personnel : jetons chiffrés et opérations PostgreSQL sérialisées.

Le schéma est installé séparément par un administrateur. Aucun jeton ni connexion
SQL ne doit être mis en cache globalement. Le verrou couvre aussi une ligne absente.
Un renouvellement WHOOP et le commit SQL ne constituent pas une transaction
distribuée : une interruption entre eux peut exceptionnellement imposer OAuth.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import ipaddress
import json
from typing import Callable, Iterator
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from cryptography.fernet import Fernet
import psycopg
from psycopg.conninfo import conninfo_to_dict

from app.core.whoop import WhoopError, WhoopToken


class TokenStoreError(RuntimeError):
    """Erreur affichable sans adresse SQL, identifiant, clé ou jeton."""


@dataclass(frozen=True)
class SavedConnection:
    token: WhoopToken = field(repr=False)
    generation: str


_CONFIG_ERROR = "Configuration du coffre WHOOP invalide."
_STORAGE_ERROR = "Le coffre WHOOP est indisponible. Réessayez ou vérifiez sa configuration."
_DATA_ERROR = "La connexion WHOOP enregistrée est illisible. Vérifiez la clé du coffre."
_TOKEN_ERROR = "La connexion durable nécessite un jeton WHOOP renouvelable avec une expiration valide."


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _ssl_mode(database_url: str) -> str:
    """Refuse un DSN non URI et le transport non chiffré hors boucle locale."""
    if urlsplit(database_url).scheme not in {"postgres", "postgresql"}:
        raise ValueError("URI PostgreSQL attendue")
    params = conninfo_to_dict(database_url)
    hosts = params.get("host", "").split(",")
    addresses = params.get("hostaddr", "").split(",") if params.get("hostaddr") else []
    # Une cible explicite évite une résolution implicite via PGHOST/service.
    if not all(hosts) or params.get("service"):
        raise ValueError("Hôte PostgreSQL explicite attendu")
    local = all(_is_loopback(host) for host in hosts + addresses)
    mode = params.get("sslmode", "prefer" if local else "require")
    allowed = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
    if mode not in allowed or (not local and mode not in {"require", "verify-ca", "verify-full"}):
        raise ValueError("TLS PostgreSQL requis")
    return mode


def _validate_token(token: WhoopToken) -> None:
    if (
        not isinstance(token, WhoopToken)
        or not isinstance(token.access_token, str)
        or not token.access_token.strip()
        or not isinstance(token.refresh_token, str)
        or not token.refresh_token.strip()
        or not isinstance(token.expires_at, datetime)
        or token.expires_at.utcoffset() is None
        or not isinstance(token.token_type, str)
        or not token.token_type.strip()
        or not isinstance(token.scopes, tuple)
        or any(not isinstance(scope, str) or not scope.strip() for scope in token.scopes)
    ):
        raise TokenStoreError(_TOKEN_ERROR)


class PostgresTokenStore:
    """Un namespace personnel par application OAuth, derrière l'auth de l'app.

    ``connect`` permet les tests ; en production chaque opération possède sa
    connexion et sa transaction, sans partage d'une transaction entre onglets.
    """

    def __init__(
        self,
        database_url: str,
        encryption_key: str,
        namespace: str,
        *,
        connect: Callable | None = None,
    ) -> None:
        try:
            if not isinstance(namespace, str) or not namespace.strip() or len(namespace) > 512:
                raise ValueError("Namespace invalide")
            self._sslmode = _ssl_mode(database_url)
            self._fernet = Fernet(encryption_key)
            self._lock_id = int.from_bytes(
                hashlib.sha256(("whoop_private.oauth_connections\0" + namespace).encode()).digest()[:8],
                "big",
                signed=True,
            )
        except Exception:
            raise TokenStoreError(_CONFIG_ERROR) from None
        self._database_url = database_url
        self._namespace = namespace
        self._connect = connect or psycopg.connect

    def __repr__(self) -> str:
        return "PostgresTokenStore()"

    @contextmanager
    def _transaction(self) -> Iterator:
        try:
            with self._connect(
                self._database_url, connect_timeout=5, sslmode=self._sslmode, autocommit=False
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL lock_timeout = '8s'")
                    cursor.execute("SET LOCAL statement_timeout = '15s'")
                    # Le HTTP de refresh est borné à 30 s ; éviter une transaction
                    # abandonnée, tout en lui laissant le temps de sauvegarder.
                    cursor.execute("SET LOCAL idle_in_transaction_session_timeout = '60s'")
                    cursor.execute("SELECT pg_advisory_xact_lock(%s::bigint)", (self._lock_id,))
                    yield cursor
        except (TokenStoreError, WhoopError):
            raise
        except Exception:
            raise TokenStoreError(_STORAGE_ERROR) from None

    def _encode(self, saved: SavedConnection) -> str:
        payload = {
            "version": 1,
            "namespace": self._namespace,
            "generation": saved.generation,
            "token": saved.token.to_dict(),
        }
        return self._fernet.encrypt(json.dumps(payload, separators=(",", ":")).encode()).decode("ascii")

    def _decode(self, encrypted: str) -> SavedConnection:
        try:
            payload = json.loads(self._fernet.decrypt(encrypted.encode("ascii")))
            if (
                not isinstance(payload, dict)
                or type(payload.get("version")) is not int
                or payload["version"] != 1
                or payload.get("namespace") != self._namespace
                or not isinstance(payload.get("generation"), str)
                or str(UUID(payload["generation"])) != payload["generation"]
                or not isinstance(payload.get("token"), dict)
            ):
                raise ValueError("Format invalide")
            raw = payload["token"]
            if (
                not isinstance(raw.get("access_token"), str)
                or not isinstance(raw.get("refresh_token"), str)
                or not isinstance(raw.get("expires_at"), str)
                or not isinstance(raw.get("scopes"), list)
                or not isinstance(raw.get("token_type"), str)
            ):
                raise ValueError("Jeton invalide")
            token = WhoopToken(
                access_token=raw["access_token"],
                refresh_token=raw["refresh_token"],
                expires_at=datetime.fromisoformat(raw["expires_at"]),
                scopes=tuple(raw["scopes"]),
                token_type=raw["token_type"],
            )
            _validate_token(token)
            return SavedConnection(token=token, generation=payload["generation"])
        except Exception:
            raise TokenStoreError(_DATA_ERROR) from None

    def _read(self, cursor) -> SavedConnection | None:
        cursor.execute(
            "SELECT payload FROM whoop_private.oauth_connections WHERE namespace = %s FOR UPDATE",
            (self._namespace,),
        )
        row = cursor.fetchone()
        return None if row is None else self._decode(row[0])

    def read(self) -> SavedConnection | None:
        with self._transaction() as cursor:
            saved = self._read(cursor)
        return saved

    def save_new(self, token: WhoopToken) -> SavedConnection:
        """Enregistre un nouveau consentement explicite, jamais un cache de session."""
        _validate_token(token)
        with self._transaction() as cursor:
            self._read(cursor)  # Ne jamais écraser un coffre illisible/mauvaise clé.
            saved = SavedConnection(token=token, generation=str(uuid4()))
            cursor.execute(
                "INSERT INTO whoop_private.oauth_connections (namespace, payload) VALUES (%s, %s) "
                "ON CONFLICT (namespace) DO UPDATE SET payload = EXCLUDED.payload",
                (self._namespace, self._encode(saved)),
            )
        return saved

    def freshen(self, renew: Callable[[WhoopToken], WhoopToken]) -> SavedConnection | None:
        """Recharge la dernière paire sous verrou ; aucune recréation si supprimée.

        Le callback doit borner son appel HTTP à 30 secondes et ne pas réessayer
        aveuglément un refresh token consommé. Le commit précède tout retour.
        """
        with self._transaction() as cursor:
            saved = self._read(cursor)
            if saved is not None:
                try:
                    token = renew(saved.token)
                except WhoopError:
                    # Erreur fonctionnelle déjà expurgée par le client WHOOP ;
                    # conserver notamment la demande explicite de réautorisation.
                    raise
                except Exception:
                    raise TokenStoreError(
                        "Le renouvellement WHOOP a échoué. Réessayez ou reconnectez votre compte."
                    ) from None
                _validate_token(token)
                if token != saved.token:
                    saved = SavedConnection(token=token, generation=saved.generation)
                    cursor.execute(
                        "UPDATE whoop_private.oauth_connections SET payload = %s WHERE namespace = %s",
                        (self._encode(saved), self._namespace),
                    )
        return saved

    def delete(self) -> None:
        """Supprime durablement la paire ; les autres sessions doivent relire."""
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM whoop_private.oauth_connections WHERE namespace = %s", (self._namespace,)
            )
