"""Synchronisation WHOOP partagée entre les onglets.

L'onglet Whoop et l'onglet Boxe lisent les mêmes données de session : un seul
chemin d'import évite que deux copies du code divergent (une récupération
datée d'UTC d'un côté, du fuseau de son cycle de l'autre).

Ce module ne dépend pas de Streamlit : le transport HTTP est injectable, comme
dans :mod:`app.core.whoop`, ce qui rend la synchronisation testable hors ligne.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.whoop import (
    Transport,
    WhoopToken,
    build_daily_frame,
    cycles_to_frame,
    fetch_body_measurement,
    fetch_collection,
    fetch_profile,
    recoveries_to_frame,
    sleeps_to_frame,
    workouts_to_frame,
)


@dataclass(frozen=True)
class WhoopSyncResult:
    """Ce qu'une synchronisation rapporte, prêt à être rangé en session."""

    daily: pd.DataFrame
    workouts: pd.DataFrame
    profile: dict[str, Any] = field(default_factory=dict)
    # Taille, poids et FC maximale : la FC maximale est celle dont WHOOP tire
    # ses zones, et c'est contre elle qu'une intensité de séance se lit.
    body: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.counts.get('recovery', 0)} récupérations, {self.counts.get('sleep', 0)} nuits, "
            f"{self.counts.get('cycle', 0)} cycles et {self.counts.get('workout', 0)} séances importés."
        )


def sync_window(days: int, *, now: Any = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Bornes de la synchronisation : les *days* jours qui précèdent aujourd'hui (UTC)."""
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    if current.tz is not None:
        current = current.tz_convert("UTC").tz_localize(None)
    end = current.normalize()
    return end - pd.Timedelta(days=int(days)), end


def fetch_whoop_data(
    token: WhoopToken,
    *,
    start: Any,
    end: Any,
    transport: Transport | None = None,
) -> WhoopSyncResult:
    """Importe cycles, récupérations, nuits, séances, profil et mesures corporelles."""
    cycle_records = fetch_collection("cycle", token, start=start, end=end, transport=transport)
    # Une récupération ne porte pas de fuseau : sans celui de son cycle, un
    # score créé à 23 h 30 UTC se retrouvait daté de la veille.
    offsets = {record.get("id"): record.get("timezone_offset") for record in cycle_records}
    recovery = recoveries_to_frame(fetch_collection("recovery", token, start=start, end=end, transport=transport), offsets)
    sleep = sleeps_to_frame(fetch_collection("sleep", token, start=start, end=end, transport=transport))
    cycle = cycles_to_frame(cycle_records)
    workouts = workouts_to_frame(fetch_collection("workout", token, start=start, end=end, transport=transport))
    profile = fetch_profile(token, transport=transport)
    body = fetch_body_measurement(token, transport=transport)
    return WhoopSyncResult(
        daily=build_daily_frame(recovery, sleep, cycle),
        workouts=workouts,
        profile=profile,
        body=body,
        counts={"recovery": len(recovery), "sleep": len(sleep), "cycle": len(cycle), "workout": len(workouts)},
    )
