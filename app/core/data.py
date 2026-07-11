"""Chargement, validation et qualité des données."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.config import OPTIONAL_COLUMNS, REQUIRED_COLUMNS


@dataclass
class RejectedRow:
    index: Any
    reasons: list[str]


@dataclass
class CleaningReport:
    raw_rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_dates: int
    columns_kept: int
    extra_columns: list[str]
    source: str = "unknown"
    rejected_rows: list[RejectedRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_rows": self.raw_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "duplicate_dates": self.duplicate_dates,
            "columns_kept": self.columns_kept,
            "extra_columns": list(self.extra_columns),
            "source": self.source,
            "rejected_rows": [ {"index": r.index, "reasons": list(r.reasons)} for r in self.rejected_rows],
        }


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    cleaned: pd.DataFrame
    quality: CleaningReport | None = None


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
    numeric = pd.to_numeric(txt, errors="coerce")
    serial = dt.isna() & numeric.notna() & numeric.between(20000, 80000)
    if serial.any():
        dt.loc[serial] = pd.to_datetime(numeric.loc[serial], unit="D", origin="1899-12-30", errors="coerce")
    return dt


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    mandatory = [c for c in REQUIRED_COLUMNS if c in df.columns]
    known_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns and c not in mandatory]
    extras = [c for c in df.columns if c not in mandatory and c not in known_optional]
    return mandatory + known_optional + extras


def _row_reasons(preview: pd.DataFrame) -> pd.Series:
    reasons = pd.Series([[] for _ in range(len(preview))], index=preview.index, dtype=object)
    if "Date" in preview:
        for idx in preview.index[preview["Date"].isna()]:
            reasons.at[idx] = reasons.at[idx] + ["date invalide ou vide"]
    if "Poids (Kgs)" in preview:
        for idx in preview.index[preview["Poids (Kgs)"].isna()]:
            reasons.at[idx] = reasons.at[idx] + ["poids non numérique ou vide"]
        for idx in preview.index[preview["Poids (Kgs)"].notna() & (preview["Poids (Kgs)"] <= 0)]:
            reasons.at[idx] = reasons.at[idx] + ["poids nul ou négatif"]
    return reasons


def clean_weight_dataframe_with_report(df: pd.DataFrame, drop_invalid: bool = True, source: str = "unknown") -> tuple[pd.DataFrame, CleaningReport]:
    """Nettoie un dataframe sans dédupliquer et avec rapport des rejets explicites."""
    out = _normalize_columns(df.copy())
    raw_rows = len(out)

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["Poids (Kgs)"] = pd.to_numeric(out["Poids (Kgs)"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out["Date"] = _parse_dates(out["Date"])

    reasons = _row_reasons(out)
    invalid_mask = reasons.map(bool)
    rejected = [RejectedRow(index=i, reasons=list(reasons.at[i])) for i in out.index[invalid_mask]]

    cleaned = out.loc[~invalid_mask].copy() if drop_invalid else out.copy()
    cleaned = cleaned.sort_values("Date", kind="mergesort").reset_index(drop=True)
    cleaned = cleaned[_ordered_columns(cleaned)]
    duplicate_dates = int(cleaned.duplicated(subset=["Date"]).sum()) if "Date" in cleaned else 0
    extras = [c for c in cleaned.columns if c not in REQUIRED_COLUMNS and c not in OPTIONAL_COLUMNS]
    report = CleaningReport(raw_rows, len(cleaned) if drop_invalid else int((~invalid_mask).sum()), len(rejected), duplicate_dates, len(cleaned.columns), extras, source, rejected)
    return cleaned, report


def clean_weight_dataframe(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    cleaned, _ = clean_weight_dataframe_with_report(df, drop_invalid=drop_invalid)
    return cleaned


def validate_journal(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    normalized = _normalize_columns(df.copy())
    for col in REQUIRED_COLUMNS:
        if col not in normalized.columns:
            errors.append(f"Colonne obligatoire absente: {col}")
    if errors:
        return ValidationResult(errors=errors, warnings=warnings, cleaned=normalized)
    cleaned, quality = clean_weight_dataframe_with_report(normalized, drop_invalid=True, source="journal")
    if quality.invalid_rows:
        rows = ", ".join(str(r.index) for r in quality.rejected_rows[:10])
        warnings.append(f"{quality.invalid_rows} ligne(s) rejetée(s) pour données invalides (indices: {rows})")
    if quality.duplicate_dates:
        warnings.append(f"{quality.duplicate_dates} mesure(s) avec date déjà présente conservée(s)")
    return ValidationResult(errors=errors, warnings=warnings, cleaned=cleaned, quality=quality)


def _moment_order_key(values: pd.Series) -> pd.Series:
    """Return sortable moment keys; unknown values keep stable source order.

    Text order: nuit < matin < midi < après-midi < soir. Clock-like values
    (08:00, 20:00) are sorted by minutes since midnight.
    """
    text_order = {
        "nuit": 0.0,
        "matin": 8 * 60.0,
        "midi": 12 * 60.0,
        "apres-midi": 15 * 60.0,
        "après-midi": 15 * 60.0,
        "aprem": 15 * 60.0,
        "soir": 20 * 60.0,
    }
    raw = values.astype(str).str.strip().str.lower()
    mapped = raw.map(text_order)
    clock = pd.to_datetime(raw, errors="coerce", format="%H:%M")
    fallback_clock = pd.to_datetime(raw, errors="coerce")
    minutes = clock.dt.hour * 60 + clock.dt.minute
    minutes = minutes.where(clock.notna(), fallback_clock.dt.hour * 60 + fallback_clock.dt.minute)
    return mapped.fillna(minutes)


def _timestamp_sort_columns(data: pd.DataFrame) -> list[str]:
    for col in ["Timestamp", "Horodatage", "DateTime", "Datetime"]:
        if col in data.columns:
            data["__suivi_time_key"] = pd.to_datetime(data[col], errors="coerce", dayfirst=True)
            return ["Date", "__suivi_time_key", "__suivi_order"]
    for col in ["Moment", "Heure", "Time"]:
        if col in data.columns:
            data["__suivi_time_key"] = _moment_order_key(data[col])
            return ["Date", "__suivi_time_key", "__suivi_order"]
    return ["Date", "__suivi_order"]


def resolve_duplicates(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    data = df.copy(deep=True)
    if data.empty:
        return data
    if "Date" in data.columns:
        data["Date"] = _parse_dates(data["Date"])
        data = data.dropna(subset=["Date"]).copy()
    if data.empty:
        return data.drop(columns=["__suivi_order", "__suivi_time_key"], errors="ignore")
    data["__suivi_order"] = np.arange(len(data))
    if strategy == "garder_la_derniere":
        sorted_data = data.sort_values(_timestamp_sort_columns(data), kind="mergesort")
        out = sorted_data.drop_duplicates(subset=["Date"], keep="last")
        return out.drop(columns=["__suivi_order", "__suivi_time_key"], errors="ignore").sort_values("Date", kind="mergesort").reset_index(drop=True)[_ordered_columns(out.drop(columns=["__suivi_order", "__suivi_time_key"], errors="ignore"))]
    numeric_cols = [c for c in data.select_dtypes(include=["number"]).columns if c not in {"Date", "__suivi_order"}]
    agg = "mean" if strategy == "moyenne_journaliere" else "median"
    grouped_num = data.groupby("Date", as_index=False)[numeric_cols].agg(agg) if numeric_cols else pd.DataFrame({"Date": data["Date"].drop_duplicates()})
    sorted_data = data.sort_values(_timestamp_sort_columns(data), kind="mergesort")
    others = sorted_data.drop_duplicates(subset=["Date"], keep="last")
    merged = others.drop(columns=numeric_cols + ["__suivi_order", "__suivi_time_key"], errors="ignore").merge(grouped_num, on="Date", how="left")
    merged = merged.sort_values("Date", kind="mergesort").reset_index(drop=True)
    return merged[_ordered_columns(merged)]


def prepare_analysis_data(df: pd.DataFrame, duplicate_strategy: str = "garder_la_derniere") -> pd.DataFrame:
    """Crée une copie analytique indépendante; seule cette copie peut agréger les mêmes dates."""
    return resolve_duplicates(df.copy(deep=True), duplicate_strategy)


def data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"score": 0, "duplicates": 0, "missing_days": 0, "coverage_pct": 0.0, "weekly_measurements": 0.0, "irregularity": 1.0, "last_entry": None, "anomalies": 0}
    series = df.sort_values("Date", kind="mergesort")
    duplicates = int(series.duplicated(subset=["Date"]).sum())
    span = max((series["Date"].max() - series["Date"].min()).days, 1)
    expected_days = span + 1
    unique_days = int(series["Date"].nunique())
    missing_days = max(expected_days - unique_days, 0)
    coverage_pct = unique_days / expected_days * 100
    weekly = unique_days / max(span / 7, 1)
    deltas = series["Date"].diff().dt.days.dropna()
    irregularity = float(deltas.std() / (deltas.mean() + 1e-9)) if not deltas.empty else 1.0
    mad = np.median(np.abs(series["Poids (Kgs)"] - series["Poids (Kgs)"].median())) + 1e-9
    robust_z = np.abs((series["Poids (Kgs)"] - series["Poids (Kgs)"].median()) / (1.4826 * mad))
    anomalies = int((robust_z > 3.5).sum())
    score = int(max(0, min(100, min(50, coverage_pct * 0.5) + max(0, 25 - irregularity * 8) + 25 - min(15, anomalies * 3) - min(10, duplicates * 2))))
    return {"score": score, "duplicates": duplicates, "missing_days": missing_days, "coverage_pct": round(coverage_pct, 1), "weekly_measurements": round(float(weekly), 2), "irregularity": round(irregularity, 3), "last_entry": series["Date"].max(), "anomalies": anomalies}
