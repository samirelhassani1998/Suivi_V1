"""Intégration WHOOP : OAuth2, appels API v2 et normalisation en DataFrames.

Ce module ne dépend pas de Streamlit et n'effectue aucun appel réseau implicite :
le transport HTTP est injectable, ce qui rend toute la logique testable hors ligne.

Références API (v2) :
- Autorisation : ``https://api.prod.whoop.com/oauth/oauth2/auth``
- Jeton        : ``https://api.prod.whoop.com/oauth/oauth2/token``
- Données      : ``https://api.prod.whoop.com/developer/v2/...``
Les collections sont paginées via ``limit`` (25 max), ``start``, ``end`` et
``nextToken``; la réponse expose ``records`` et ``next_token``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlencode
import re
import secrets as _secrets

import numpy as np
import pandas as pd

AUTHORIZE_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
API_BASE_URL = "https://api.prod.whoop.com/developer"

READ_SCOPES: tuple[str, ...] = (
    "read:profile",
    "read:body_measurement",
    "read:cycles",
    "read:recovery",
    "read:sleep",
    "read:workout",
)
# ``offline`` est la seule condition pour obtenir un refresh token.
OFFLINE_SCOPE = "offline"
DEFAULT_SCOPES: tuple[str, ...] = (OFFLINE_SCOPE,) + READ_SCOPES


def build_scopes(*, offline: bool = True) -> tuple[str, ...]:
    """Scopes demandés ; sans ``offline``, la session expire sans renouvellement."""
    return (OFFLINE_SCOPE,) + READ_SCOPES if offline else READ_SCOPES

COLLECTION_PATHS: dict[str, str] = {
    "recovery": "v2/recovery",
    "sleep": "v2/activity/sleep",
    "cycle": "v2/cycle",
    "workout": "v2/activity/workout",
}

PROFILE_PATH = "v2/user/profile/basic"
BODY_MEASUREMENT_PATH = "v2/user/measurement/body"

# WHOOP plafonne la pagination à 25 éléments par page.
MAX_PAGE_LIMIT = 25
# Garde-fou pour éviter une boucle infinie si l'API renvoie toujours un token.
MAX_PAGES = 200
# Marge de sécurité avant expiration du jeton d'accès.
TOKEN_EXPIRY_MARGIN_SECONDS = 60

KJ_TO_KCAL = 0.2390057361
MILLI_TO_HOURS = 1.0 / 3_600_000.0

WHOOP_DAILY_METRICS: tuple[str, ...] = (
    "Récupération (%)",
    "HRV (ms)",
    "FC repos (bpm)",
    "Température peau (°C)",
    "SpO2 (%)",
    "Sommeil (heures)",
    "Besoin de sommeil (heures)",
    "Dette de sommeil (heures)",
    "Performance sommeil (%)",
    "Efficacité sommeil (%)",
    "Régularité sommeil (%)",
    "Sommeil profond (heures)",
    "Sommeil REM (heures)",
    "Perturbations sommeil",
    "Fréquence respiratoire (resp/min)",
    "Heure de coucher",
    "Strain",
    "Calories (kcal)",
    "FC moyenne (bpm)",
    "FC max (bpm)",
)


class WhoopError(RuntimeError):
    """Erreur fonctionnelle remontée à l'UI sans exposer de secret."""


@dataclass(frozen=True)
class HttpResponse:
    """Réponse HTTP minimale, indépendante de la librairie utilisée."""

    status_code: int
    payload: Any = None
    text: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


# Signature du transport : (method, url, params, data, headers) -> HttpResponse
Transport = Callable[..., HttpResponse]


@dataclass(frozen=True)
class WhoopCredentials:
    """Identifiants applicatifs WHOOP (jamais journalisés)."""

    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: tuple[str, ...] = DEFAULT_SCOPES

    @property
    def is_complete(self) -> bool:
        return bool(self.client_id and self.client_secret and self.redirect_uri)

    def scope_string(self) -> str:
        return " ".join(self.scopes)


@dataclass(frozen=True)
class WhoopToken:
    """Jeton OAuth2 et sa date d'expiration absolue."""

    access_token: str
    refresh_token: str | None = None
    expires_at: datetime | None = None
    scopes: tuple[str, ...] = ()
    token_type: str = "Bearer"

    def is_expired(self, *, now: datetime | None = None, margin_seconds: int = TOKEN_EXPIRY_MARGIN_SECONDS) -> bool:
        if self.expires_at is None:
            return False
        current = now or datetime.now(timezone.utc)
        return current >= self.expires_at - timedelta(seconds=margin_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": list(self.scopes),
            "token_type": self.token_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WhoopToken":
        expires_at = data.get("expires_at")
        parsed_expiry: datetime | None = None
        if isinstance(expires_at, datetime):
            parsed_expiry = expires_at
        elif expires_at:
            parsed = pd.to_datetime(expires_at, errors="coerce", utc=True)
            parsed_expiry = None if pd.isna(parsed) else parsed.to_pydatetime()
        return cls(
            access_token=str(data.get("access_token", "")),
            refresh_token=data.get("refresh_token") or None,
            expires_at=parsed_expiry,
            scopes=tuple(data.get("scopes") or ()),
            token_type=str(data.get("token_type", "Bearer")),
        )


def _requests_transport(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 30,
) -> HttpResponse:
    """Transport par défaut basé sur ``requests`` (import paresseux)."""
    import requests  # import local : le module reste utilisable sans réseau

    response = requests.request(
        method=method,
        url=url,
        params=dict(params or {}),
        data=dict(data or {}),
        headers=dict(headers or {}),
        timeout=timeout,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return HttpResponse(status_code=response.status_code, payload=payload, text=response.text[:500])


def default_transport() -> Transport:
    return _requests_transport


DEFAULT_REDIRECT_URI = "http://localhost:8501"
ENV_CLIENT_ID = "WHOOP_CLIENT_ID"
ENV_CLIENT_SECRET = "WHOOP_CLIENT_SECRET"
ENV_REDIRECT_URI = "WHOOP_REDIRECT_URI"


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def credentials_from_sources(
    secrets: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    default_redirect_uri: str = DEFAULT_REDIRECT_URI,
    scopes: Sequence[str] = DEFAULT_SCOPES,
) -> WhoopCredentials:
    """Résout les identifiants par priorité : saisie UI > secrets > variables d'env.

    ``secrets`` accepte soit un bloc ``[whoop]``, soit des clés à plat
    (``whoop_client_id``...), ce qui évite d'imposer un format unique.
    """
    secrets_map: Mapping[str, Any] = secrets or {}
    block = secrets_map.get("whoop") if isinstance(secrets_map.get("whoop"), Mapping) else {}
    env_map: Mapping[str, str] = env or {}
    overrides_map: Mapping[str, Any] = overrides or {}

    client_id = _first_non_empty(
        overrides_map.get("client_id"),
        block.get("client_id"),
        secrets_map.get("whoop_client_id"),
        env_map.get(ENV_CLIENT_ID),
    )
    client_secret = _first_non_empty(
        overrides_map.get("client_secret"),
        block.get("client_secret"),
        secrets_map.get("whoop_client_secret"),
        env_map.get(ENV_CLIENT_SECRET),
    )
    redirect_uri = _first_non_empty(
        overrides_map.get("redirect_uri"),
        block.get("redirect_uri"),
        secrets_map.get("whoop_redirect_uri"),
        env_map.get(ENV_REDIRECT_URI),
        default_redirect_uri,
    )
    return WhoopCredentials(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=tuple(scopes),
    )


def generate_state(nbytes: int = 24) -> str:
    """État anti-CSRF ; WHOOP impose au moins 8 caractères."""
    return _secrets.token_urlsafe(nbytes)


def build_authorization_url(credentials: WhoopCredentials, state: str) -> str:
    """Construit l'URL de consentement WHOOP."""
    if not credentials.client_id:
        raise WhoopError("Client ID WHOOP manquant.")
    if not credentials.redirect_uri:
        raise WhoopError("Redirect URI WHOOP manquante.")
    if len(state) < 8:
        raise WhoopError("Le paramètre state doit contenir au moins 8 caractères.")
    query = urlencode(
        {
            "response_type": "code",
            "client_id": credentials.client_id,
            "redirect_uri": credentials.redirect_uri,
            "scope": credentials.scope_string(),
            "state": state,
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def _token_from_payload(payload: Mapping[str, Any], *, now: datetime | None = None, fallback_refresh: str | None = None) -> WhoopToken:
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise WhoopError("Réponse WHOOP sans access_token.")
    current = now or datetime.now(timezone.utc)
    expires_in = payload.get("expires_in")
    expires_at: datetime | None = None
    try:
        if expires_in is not None:
            expires_at = current + timedelta(seconds=int(float(expires_in)))
    except (TypeError, ValueError):
        expires_at = None
    raw_scopes = payload.get("scope") or payload.get("scopes") or ()
    if isinstance(raw_scopes, str):
        scopes = tuple(part for part in raw_scopes.split() if part)
    else:
        scopes = tuple(str(part) for part in raw_scopes)
    return WhoopToken(
        access_token=access_token,
        refresh_token=payload.get("refresh_token") or fallback_refresh,
        expires_at=expires_at,
        scopes=scopes,
        token_type=str(payload.get("token_type", "Bearer")),
    )


def _post_token(data: Mapping[str, Any], transport: Transport, *, now: datetime | None = None, fallback_refresh: str | None = None) -> WhoopToken:
    response = transport("POST", TOKEN_URL, data=dict(data), headers={"Content-Type": "application/x-www-form-urlencoded"})
    if not response.ok:
        raise WhoopError(f"Échec d'authentification WHOOP (HTTP {response.status_code}).")
    if not isinstance(response.payload, Mapping):
        raise WhoopError("Réponse d'authentification WHOOP illisible.")
    return _token_from_payload(response.payload, now=now, fallback_refresh=fallback_refresh)


def exchange_code_for_token(
    credentials: WhoopCredentials,
    code: str,
    *,
    transport: Transport | None = None,
    now: datetime | None = None,
) -> WhoopToken:
    """Échange le code d'autorisation contre un jeton d'accès."""
    if not credentials.is_complete:
        raise WhoopError("Identifiants WHOOP incomplets (client id, secret ou redirect uri).")
    if not code:
        raise WhoopError("Code d'autorisation WHOOP manquant.")
    return _post_token(
        {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "redirect_uri": credentials.redirect_uri,
        },
        transport or default_transport(),
        now=now,
    )


def refresh_access_token(
    credentials: WhoopCredentials,
    token: WhoopToken,
    *,
    transport: Transport | None = None,
    now: datetime | None = None,
) -> WhoopToken:
    """Renouvelle le jeton d'accès via le refresh token (scope ``offline``)."""
    if not token.refresh_token:
        raise WhoopError("Aucun refresh token disponible : reconnectez-vous avec le scope offline.")
    return _post_token(
        {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scope": "offline",
        },
        transport or default_transport(),
        now=now,
        fallback_refresh=token.refresh_token,
    )


def ensure_fresh_token(
    credentials: WhoopCredentials,
    token: WhoopToken,
    *,
    transport: Transport | None = None,
    now: datetime | None = None,
) -> WhoopToken:
    """Retourne un jeton valide, rafraîchi seulement si nécessaire."""
    if not token.is_expired(now=now):
        return token
    if not token.refresh_token:
        raise WhoopError("Jeton WHOOP expiré et non renouvelable : reconnectez-vous.")
    return refresh_access_token(credentials, token, transport=transport, now=now)


def _iso_utc(value: Any, *, end_of_day: bool = False) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise WhoopError("Date de synchronisation invalide.")
    if timestamp.tz is None:
        if end_of_day and timestamp.normalize() == timestamp:
            timestamp = timestamp + pd.Timedelta(days=1)
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def fetch_collection(
    resource: str,
    token: WhoopToken,
    *,
    start: Any,
    end: Any,
    transport: Transport | None = None,
    limit: int = MAX_PAGE_LIMIT,
    max_pages: int = MAX_PAGES,
) -> list[dict[str, Any]]:
    """Récupère une collection WHOOP paginée entre deux dates."""
    path = COLLECTION_PATHS.get(resource)
    if path is None:
        raise WhoopError(f"Ressource WHOOP inconnue : {resource}")
    call = transport or default_transport()
    params: dict[str, Any] = {
        "start": _iso_utc(start),
        "end": _iso_utc(end, end_of_day=True),
        "limit": max(1, min(int(limit), MAX_PAGE_LIMIT)),
    }
    headers = {"Authorization": f"{token.token_type} {token.access_token}"}
    records: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()

    for _ in range(max(1, int(max_pages))):
        response = call("GET", f"{API_BASE_URL}/{path}", params=dict(params), headers=headers)
        if response.status_code == 401:
            raise WhoopError("Jeton WHOOP refusé (401) : reconnectez votre compte.")
        if response.status_code == 429:
            raise WhoopError("Quota WHOOP atteint (429) : réessayez dans quelques minutes.")
        if not response.ok:
            raise WhoopError(f"Appel WHOOP {resource} en échec (HTTP {response.status_code}).")
        payload = response.payload if isinstance(response.payload, Mapping) else {}
        page = payload.get("records") or []
        records.extend(item for item in page if isinstance(item, Mapping))
        next_token = payload.get("next_token") or payload.get("nextToken")
        if not next_token or next_token in seen_tokens:
            break
        seen_tokens.add(str(next_token))
        params["nextToken"] = next_token

    return [dict(item) for item in records]


def fetch_profile(token: WhoopToken, *, transport: Transport | None = None) -> dict[str, Any]:
    """Profil de base ; renvoie un dictionnaire vide en cas d'indisponibilité."""
    call = transport or default_transport()
    response = call("GET", f"{API_BASE_URL}/{PROFILE_PATH}", headers={"Authorization": f"{token.token_type} {token.access_token}"})
    if not response.ok or not isinstance(response.payload, Mapping):
        return {}
    return dict(response.payload)


def fetch_body_measurement(token: WhoopToken, *, transport: Transport | None = None) -> dict[str, Any]:
    """Mesures corporelles WHOOP (taille, poids, FC max)."""
    call = transport or default_transport()
    response = call("GET", f"{API_BASE_URL}/{BODY_MEASUREMENT_PATH}", headers={"Authorization": f"{token.token_type} {token.access_token}"})
    if not response.ok or not isinstance(response.payload, Mapping):
        return {}
    return dict(response.payload)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation des enregistrements en DataFrames
# ──────────────────────────────────────────────────────────────────────────────


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})
    for column in ("Date", "Début"):
        if column in columns:
            frame[column] = pd.Series(dtype="datetime64[ns]")
    return frame[list(columns)]


_OFFSET_PATTERN = re.compile(r"^([+-])(\d{1,2}):?(\d{2})$")


def parse_timezone_offset(offset: Any) -> pd.Timedelta:
    """Convertit un décalage WHOOP (``+01:00``, ``-0500``) en Timedelta."""
    if not isinstance(offset, str):
        return pd.Timedelta(0)
    match = _OFFSET_PATTERN.match(offset.strip())
    if not match:
        return pd.Timedelta(0)
    sign, hours, minutes = match.groups()
    delta = pd.Timedelta(hours=int(hours), minutes=int(minutes))
    return -delta if sign == "-" else delta


def _to_local_date(value: Any, offset: Any = None) -> pd.Timestamp:
    """Convertit un instant UTC en date calendaire locale WHOOP."""
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    local = timestamp.tz_localize(None) + parse_timezone_offset(offset)
    return pd.Timestamp(local).normalize()


SCORED_STATE = "SCORED"


def _is_scored(record: Mapping[str, Any]) -> bool:
    """Un enregistrement non noté n'a pas de score exploitable.

    WHOOP publie des cycles et des nuits en ``PENDING_SCORE`` ou ``UNSCORABLE``.
    Les conserver revient à laisser une ligne vide écraser la mesure valide du
    même jour lors de la déduplication.
    """
    state = record.get("score_state")
    if state is None:
        return isinstance(record.get("score"), Mapping)
    return str(state) == SCORED_STATE


def _score(record: Mapping[str, Any]) -> Mapping[str, Any]:
    score = record.get("score")
    return score if isinstance(score, Mapping) else {}


def _number(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


RECOVERY_COLUMNS = ("Date", "Récupération (%)", "HRV (ms)", "FC repos (bpm)", "Température peau (°C)", "SpO2 (%)", "Calibration")


def recoveries_to_frame(
    records: Iterable[Mapping[str, Any]],
    timezone_offsets: Mapping[Any, str] | None = None,
) -> pd.DataFrame:
    """Normalise les récupérations en une ligne par jour.

    ``timezone_offsets`` associe un identifiant de cycle à son décalage horaire :
    l'enregistrement de récupération n'en porte pas, alors qu'une récupération
    créée à 23h30 UTC appartient au lendemain pour un utilisateur en heure d'été
    européenne. Sans cette table, la date reste celle d'UTC.
    """
    offsets = dict(timezone_offsets or {})
    rows = []
    for record in records or []:
        if not _is_scored(record):
            continue
        score = _score(record)
        offset = offsets.get(record.get("cycle_id"))
        date = _to_local_date(record.get("created_at") or record.get("updated_at"), offset)
        if pd.isna(date):
            continue
        rows.append(
            {
                "Date": date,
                "Récupération (%)": _number(score.get("recovery_score")),
                "HRV (ms)": _number(score.get("hrv_rmssd_milli")),
                "FC repos (bpm)": _number(score.get("resting_heart_rate")),
                "Température peau (°C)": _number(score.get("skin_temp_celsius")),
                "SpO2 (%)": _number(score.get("spo2_percentage")),
                "Calibration": bool(score.get("user_calibrating", False)),
            }
        )
    if not rows:
        return _empty_frame(RECOVERY_COLUMNS)
    frame = pd.DataFrame(rows).sort_values("Date", kind="mergesort")
    # Deux récupérations le même jour rendaient la trame non réindexable, ce qui
    # faisait remonter une ValueError jusqu'à l'affichage de la page entière.
    frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return frame[list(RECOVERY_COLUMNS)]


SLEEP_COLUMNS = (
    "Date",
    "Sommeil (heures)",
    "Besoin de sommeil (heures)",
    "Dette de sommeil (heures)",
    "Performance sommeil (%)",
    "Efficacité sommeil (%)",
    "Régularité sommeil (%)",
    "Sommeil profond (heures)",
    "Sommeil REM (heures)",
    "Perturbations sommeil",
    "Fréquence respiratoire (resp/min)",
    "Heure de coucher",
    "Sieste",
)

SLEEP_NEED_PARTS = (
    "baseline_milli",
    "need_from_sleep_debt_milli",
    "need_from_recent_strain_milli",
    # WHOOP renvoie déjà une valeur négative pour la sieste : elle réduit le besoin.
    "need_from_recent_nap_milli",
)


def _total_sleep_need_milli(score: Mapping[str, Any]) -> float:
    """Besoin total de sommeil : somme des composantes renvoyées par WHOOP."""
    needed = score.get("sleep_needed")
    if not isinstance(needed, Mapping):
        return float("nan")
    parts = [_number(needed.get(part)) for part in SLEEP_NEED_PARTS]
    if not any(np.isfinite(part) for part in parts):
        return float("nan")
    return float(np.nansum(parts))


def _decimal_hour(value: Any, offset: Any = None) -> float:
    """Heure locale en décimal (22h30 -> 22.5), recentrée autour de minuit.

    Les couchers après minuit deviennent négatifs (00h30 -> -23.5 serait absurde,
    on renvoie donc 0.5) : l'échelle reste continue entre 18h et 6h du matin en
    ramenant les heures d'après-midi dans le négatif.
    """
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return float("nan")
    local = timestamp.tz_localize(None) + parse_timezone_offset(offset)
    hour = local.hour + local.minute / 60.0
    # 18h..24h -> -6..0 pour rester continu avec 0h..6h du matin.
    return hour - 24.0 if hour >= 12.0 else hour


def sleeps_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records or []:
        if not _is_scored(record):
            continue
        score = _score(record)
        stages = score.get("stage_summary") if isinstance(score.get("stage_summary"), Mapping) else {}
        # La nuit est rattachée au jour du réveil, cohérent avec la pesée du matin.
        date = _to_local_date(record.get("end") or record.get("start"), record.get("timezone_offset"))
        if pd.isna(date):
            continue
        in_bed = _number(stages.get("total_in_bed_time_milli"))
        awake = _number(stages.get("total_awake_time_milli"))
        # Le capteur perd parfois le signal : ce temps n'est ni de l'éveil ni du
        # sommeil, et le compter comme dormi surestime la nuit.
        no_data = _number(stages.get("total_no_data_time_milli"))
        no_data = no_data if np.isfinite(no_data) else 0.0
        asleep = in_bed - awake - no_data if np.isfinite(in_bed) and np.isfinite(awake) else float("nan")
        asleep_hours = asleep * MILLI_TO_HOURS if np.isfinite(asleep) else float("nan")
        needed_milli = _total_sleep_need_milli(score)
        needed_hours = needed_milli * MILLI_TO_HOURS if np.isfinite(needed_milli) else float("nan")
        debt_hours = (
            needed_hours - asleep_hours
            if np.isfinite(needed_hours) and np.isfinite(asleep_hours)
            else float("nan")
        )
        rows.append(
            {
                "Date": date,
                "Sommeil (heures)": asleep_hours,
                "Besoin de sommeil (heures)": needed_hours,
                "Dette de sommeil (heures)": debt_hours,
                "Heure de coucher": _decimal_hour(record.get("start"), record.get("timezone_offset")),
                # Signe vital nocturne renvoyé par WHOOP et jusqu'ici ignoré :
                # une fréquence respiratoire qui s'élève est l'un des signaux
                # les plus précoces d'une infection des voies respiratoires.
                "Fréquence respiratoire (resp/min)": _number(score.get("respiratory_rate")),
                "Performance sommeil (%)": _number(score.get("sleep_performance_percentage")),
                "Efficacité sommeil (%)": _number(score.get("sleep_efficiency_percentage")),
                "Régularité sommeil (%)": _number(score.get("sleep_consistency_percentage")),
                "Sommeil profond (heures)": _number(stages.get("total_slow_wave_sleep_time_milli")) * MILLI_TO_HOURS,
                "Sommeil REM (heures)": _number(stages.get("total_rem_sleep_time_milli")) * MILLI_TO_HOURS,
                "Perturbations sommeil": _number(stages.get("disturbance_count")),
                "Sieste": bool(record.get("nap", False)),
            }
        )
    if not rows:
        return _empty_frame(SLEEP_COLUMNS)
    frame = pd.DataFrame(rows)
    # Les siestes ne remplacent pas la nuit principale : on garde la plus longue par jour.
    frame = frame.sort_values(["Date", "Sieste", "Sommeil (heures)"], ascending=[True, True, False], kind="mergesort")
    frame = frame.drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)
    return frame[list(SLEEP_COLUMNS)]


CYCLE_COLUMNS = ("Date", "Strain", "Calories (kcal)", "FC moyenne (bpm)", "FC max (bpm)")


def cycles_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records or []:
        if not _is_scored(record):
            continue
        score = _score(record)
        date = _to_local_date(record.get("start"), record.get("timezone_offset"))
        if pd.isna(date):
            continue
        kilojoule = _number(score.get("kilojoule"))
        rows.append(
            {
                "Date": date,
                "Strain": _number(score.get("strain")),
                "Calories (kcal)": kilojoule * KJ_TO_KCAL if np.isfinite(kilojoule) else float("nan"),
                "FC moyenne (bpm)": _number(score.get("average_heart_rate")),
                "FC max (bpm)": _number(score.get("max_heart_rate")),
            }
        )
    if not rows:
        return _empty_frame(CYCLE_COLUMNS)
    frame = pd.DataFrame(rows).sort_values("Date", kind="mergesort")
    frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return frame[list(CYCLE_COLUMNS)]


WORKOUT_COLUMNS = ("Date", "Début", "Sport", "Durée (min)", "Strain séance", "Calories séance (kcal)", "FC moyenne (bpm)", "FC max (bpm)", "Distance (km)")


def workouts_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records or []:
        if not _is_scored(record):
            continue
        score = _score(record)
        date = _to_local_date(record.get("start"), record.get("timezone_offset"))
        if pd.isna(date):
            continue
        start = pd.to_datetime(record.get("start"), errors="coerce", utc=True)
        end = pd.to_datetime(record.get("end"), errors="coerce", utc=True)
        duration = (end - start).total_seconds() / 60.0 if not pd.isna(start) and not pd.isna(end) else float("nan")
        # Sans l'heure de début, plusieurs séances du même jour deviennent
        # indiscernables dans le tableau : trois lignes identiques au lecteur.
        local_start = (
            pd.Timestamp(start.tz_localize(None) + parse_timezone_offset(record.get("timezone_offset")))
            if not pd.isna(start)
            else pd.NaT
        )
        kilojoule = _number(score.get("kilojoule"))
        distance = _number(score.get("distance_meter"))
        rows.append(
            {
                "Date": date,
                "Début": local_start,
                "Sport": str(record.get("sport_name") or "Inconnu"),
                "Durée (min)": duration,
                "Strain séance": _number(score.get("strain")),
                "Calories séance (kcal)": kilojoule * KJ_TO_KCAL if np.isfinite(kilojoule) else float("nan"),
                "FC moyenne (bpm)": _number(score.get("average_heart_rate")),
                "FC max (bpm)": _number(score.get("max_heart_rate")),
                "Distance (km)": distance / 1000.0 if np.isfinite(distance) else float("nan"),
            }
        )
    if not rows:
        return _empty_frame(WORKOUT_COLUMNS)
    frame = pd.DataFrame(rows).sort_values(["Date", "Début"], kind="mergesort").reset_index(drop=True)
    return frame[list(WORKOUT_COLUMNS)]


def build_daily_frame(
    recovery: pd.DataFrame | None = None,
    sleep: pd.DataFrame | None = None,
    cycle: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fusionne les trois sources en une ligne par jour calendaire."""
    frames = [frame for frame in (recovery, sleep, cycle) if frame is not None and not frame.empty]
    if not frames:
        return _empty_frame(("Date",) + WHOOP_DAILY_METRICS)

    merged = frames[0].copy(deep=True)
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Date", how="outer", suffixes=("", "_dup"))
        merged = merged.drop(columns=[column for column in merged.columns if column.endswith("_dup")])

    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"]).sort_values("Date", kind="mergesort").reset_index(drop=True)
    merged = merged.drop(columns=[column for column in ("Calibration", "Sieste") if column in merged.columns])
    return merged


def available_metrics(frame: pd.DataFrame) -> list[str]:
    """Colonnes WHOOP réellement exploitables (au moins une valeur)."""
    if frame is None or frame.empty:
        return []
    return [
        metric
        for metric in WHOOP_DAILY_METRICS
        if metric in frame.columns and frame[metric].notna().any()
    ]


def merge_with_weight(weight_df: pd.DataFrame, whoop_daily: pd.DataFrame) -> pd.DataFrame:
    """Associe le poids quotidien aux métriques WHOOP du même jour.

    La variation brute entre deux pesées n'est pas comparable d'une ligne à
    l'autre : dix jours d'écart produisent mécaniquement un chiffre plus grand
    qu'un jour d'écart. La variation quotidienne, elle, se compare — c'est donc
    elle que les corrélations utilisent.
    """
    columns = [
        "Date",
        "Poids (Kgs)",
        "Variation poids (kg)",
        "Jours depuis la pesée précédente",
        "Variation poids (kg/jour)",
    ] + list(WHOOP_DAILY_METRICS)
    if weight_df is None or weight_df.empty or whoop_daily is None or whoop_daily.empty:
        return _empty_frame(columns)

    weights = weight_df[["Date", "Poids (Kgs)"]].copy()
    weights["Date"] = pd.to_datetime(weights["Date"], errors="coerce").dt.normalize()
    weights = weights.dropna(subset=["Date", "Poids (Kgs)"])
    if weights.empty:
        return _empty_frame(columns)
    weights = weights.groupby("Date", as_index=False)["Poids (Kgs)"].mean().sort_values("Date", kind="mergesort")
    weights["Variation poids (kg)"] = weights["Poids (Kgs)"].diff()
    gap_days = weights["Date"].diff().dt.days
    weights["Jours depuis la pesée précédente"] = gap_days
    weights["Variation poids (kg/jour)"] = weights["Variation poids (kg)"] / gap_days.where(gap_days > 0)

    whoop = whoop_daily.copy(deep=True)
    whoop["Date"] = pd.to_datetime(whoop["Date"], errors="coerce").dt.normalize()
    whoop = whoop.dropna(subset=["Date"])

    merged = weights.merge(whoop, on="Date", how="inner").sort_values("Date", kind="mergesort").reset_index(drop=True)
    ordered = [column for column in columns if column in merged.columns]
    return merged[ordered]


def summarise_daily(frame: pd.DataFrame, days: int = 7) -> dict[str, dict[str, float]]:
    """Moyennes récentes et écart par rapport à la période précédente."""
    summary: dict[str, dict[str, float]] = {}
    if frame is None or frame.empty:
        return summary
    data = frame.copy(deep=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date", kind="mergesort")
    if data.empty:
        return summary

    window = max(1, int(days))
    # Fenêtres calendaires : « les 7 derniers jours » doit désigner sept jours,
    # pas les sept dernières lignes, qui peuvent s'étaler sur des semaines.
    reference = data["Date"].max()
    recent_start = reference - pd.Timedelta(days=window - 1)
    previous_end = recent_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=window - 1)
    recent = data[data["Date"] >= recent_start]
    previous = data[(data["Date"] >= previous_start) & (data["Date"] <= previous_end)]
    for metric in available_metrics(data):
        current_mean = float(recent[metric].mean()) if recent[metric].notna().any() else float("nan")
        previous_mean = float(previous[metric].mean()) if not previous.empty and previous[metric].notna().any() else float("nan")
        delta = current_mean - previous_mean if np.isfinite(current_mean) and np.isfinite(previous_mean) else float("nan")
        summary[metric] = {"current": current_mean, "previous": previous_mean, "delta": delta}
    return summary
