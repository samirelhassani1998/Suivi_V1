"""Configuration centralisée de l'application Suivi V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

DATA_URL: Final[str] = "https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv"
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("Date", "Poids (Kgs)")
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "Calories consommées",
    "Calories brûlées",
    "Pas",
    "Sommeil (heures)",
    "Hydratation",
    "Tour de taille",
    "Body fat %",
    "Notes",
    "Condition de mesure",
)
ALL_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

DUPLICATE_STRATEGIES: Final[tuple[str, ...]] = (
    "garder_la_derniere",
    "moyenne_journaliere",
    "mediane_journaliere",
)


@dataclass(frozen=True)
class AppDefaults:
    """Valeurs par défaut modifiables via l'UI."""

    height_cm: int = 182
    target_weight: float = 80.0
    confidence_level: float = 0.9
    duplicate_strategy: str = "garder_la_derniere"
    default_model: str = "ridge"
