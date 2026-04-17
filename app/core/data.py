"""Chargement, validation et qualité des données."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.config import OPTIONAL_COLUMNS, REQUIRED_COLUMNS


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    cleaned: pd.DataFrame


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for c in df.columns:
        cleaned = str(c).replace("\ufeff", "").strip()
        low = cleaned.lower()
        if low in {"date", "dates"}:
            mapping[c] = "Date"
        elif low in {"poids", "poids (kg)", "poids(kg)", "poids (kgs)"}:
            mapping[c] = "Poids (Kgs)"
        else:
            mapping[c] = cleaned
    return df.rename(columns=mapping)


def _parse_dates(values: pd.Series) -> pd.Series:
    txt = values.astype(str).str.strip().str.replace("'", "", regex=False)
    dt = pd.to_datetime(txt, errors="coerce", dayfirst=True)
    miss = dt.isna()
    if miss.any():
        dt.loc[miss] = pd.to_datetime(txt.loc[miss], errors="coerce", dayfirst=False)
    return dt


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    mandatory = [c for c in REQUIRED_COLUMNS if c in df.columns]
    known_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns and c not in mandatory]
    extras = [c for c in df.columns if c not in mandatory and c not in known_optional]
    return mandatory + known_optional + extras


def clean_weight_dataframe(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """Nettoie un dataframe sans perdre les colonnes additionnelles utilisateur."""
    out = _normalize_columns(df.copy())

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["Poids (Kgs)"] = pd.to_numeric(
        out["Poids (Kgs)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    out["Date"] = _parse_dates(out["Date"])

    if drop_invalid:
        out = out.dropna(subset=["Date", "Poids (Kgs)"])

    out = out.sort_values("Date").reset_index(drop=True)
    return out[_ordered_columns(out)]


def validate_journal(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    normalized = _normalize_columns(df.copy())
    for col in REQUIRED_COLUMNS:
        if col not in normalized.columns:
            errors.append(f"Colonne obligatoire absente: {col}")

    if errors:
        return ValidationResult(errors=errors, warnings=warnings, cleaned=normalized)

    preview = clean_weight_dataframe(normalized, drop_invalid=False)
    invalid_mask = preview["Date"].isna() | preview["Poids (Kgs)"].isna()
    if invalid_mask.any():
        idx = ", ".join(str(i + 1) for i in preview.index[invalid_mask][:10])
        warnings.append(f"Lignes ignorées (Date/Poids invalide): {idx}")

    cleaned = clean_weight_dataframe(normalized, drop_invalid=True)
    if not cleaned.empty:
        invalid_weight = int((cleaned["Poids (Kgs)"] <= 0).sum())
        if invalid_weight:
            errors.append(f"{invalid_weight} ligne(s) avec poids <= 0")

        duplicate_count = int(cleaned.duplicated(subset=["Date"]).sum())
        if duplicate_count:
            warnings.append(f"{duplicate_count} doublon(s) de date détecté(s)")

        q1, q3 = cleaned["Poids (Kgs)"].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_count = int(((cleaned["Poids (Kgs)"] < low) | (cleaned["Poids (Kgs)"] > high)).sum())
            if outlier_count:
                warnings.append(f"{outlier_count} valeur(s) aberrante(s) potentielle(s)")

    return ValidationResult(errors=errors, warnings=warnings, cleaned=cleaned)


def resolve_duplicates(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    data = df.copy()
    if data.empty:
        return data

    if strategy == "garder_la_derniere":
        return data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    numeric_cols = [c for c in data.select_dtypes(include=["number"]).columns if c != "Date"]
    agg = "mean" if strategy == "moyenne_journaliere" else "median"

    grouped_num = data.groupby("Date", as_index=False)[numeric_cols].agg(agg) if numeric_cols else pd.DataFrame({"Date": data["Date"].drop_duplicates()})
    others = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    merged = others.drop(columns=numeric_cols, errors="ignore").merge(grouped_num, on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged[_ordered_columns(merged)]


def data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "score": 0,
            "duplicates": 0,
            "missing_days": 0,
            "weekly_measurements": 0.0,
            "irregularity": 1.0,
            "last_entry": None,
            "anomalies": 0,
        }
    series = df.sort_values("Date")
    duplicates = int(series.duplicated(subset=["Date"]).sum())
    span = max((series["Date"].max() - series["Date"].min()).days, 1)
    expected_days = span + 1
    unique_days = int(series["Date"].nunique())
    missing_days = max(expected_days - unique_days, 0)
    weekly = unique_days / max(span / 7, 1)
    deltas = series["Date"].diff().dt.days.dropna()
    irregularity = float(deltas.std() / (deltas.mean() + 1e-9)) if not deltas.empty else 1.0
    mad = np.median(np.abs(series["Poids (Kgs)"] - series["Poids (Kgs)"].median())) + 1e-9
    robust_z = np.abs((series["Poids (Kgs)"] - series["Poids (Kgs)"].median()) / (1.4826 * mad))
    anomalies = int((robust_z > 3.5).sum())

    penalties = duplicates * 2 + missing_days * 0.1 + anomalies * 2 + irregularity * 10
    score = int(max(0, min(100, 100 - penalties)))
    return {
        "score": score,
        "duplicates": duplicates,
        "missing_days": missing_days,
        "weekly_measurements": round(float(weekly), 2),
        "irregularity": round(irregularity, 3),
        "last_entry": series["Date"].max(),
        "anomalies": anomalies,
    }
