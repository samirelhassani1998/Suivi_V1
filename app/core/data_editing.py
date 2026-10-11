from __future__ import annotations

import pandas as pd

from app.core.time_utils import normalize_calendar_dates


def apply_editor_changes(base: pd.DataFrame, changes: dict) -> pd.DataFrame:
    """Materialize Streamlit's row deltas before navigation can remove its widget.

    Deltas always refer to the unchanged editor input, so this operation is
    idempotent even when callbacks or page renders run repeatedly.
    """
    edited = base.copy(deep=True)

    def typed_value(column, value):
        if pd.api.types.is_datetime64_any_dtype(base[column].dtype):
            return pd.to_datetime(value, errors="coerce") if value is not None else pd.NaT
        return value

    for position, values in changes.get("edited_rows", {}).items():
        for column, value in values.items():
            if column in edited.columns:
                edited.iat[int(position), edited.columns.get_loc(column)] = typed_value(column, value)
    deleted = set(changes.get("deleted_rows", []))
    edited = edited.iloc[[index for index in range(len(edited)) if index not in deleted]]
    added = changes.get("added_rows", [])
    if added:
        rows = [{column: typed_value(column, row.get(column)) for column in base.columns} for row in added]
        edited = pd.concat([edited, pd.DataFrame(rows, columns=base.columns)], ignore_index=True)
    return edited.reset_index(drop=True)


def _canonical_for_change_detection(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = normalize_calendar_dates(out["Date"])
    if "Poids (Kgs)" in out.columns:
        out["Poids (Kgs)"] = (
            pd.to_numeric(
                out["Poids (Kgs)"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            ).astype("float64")
        )
    return out.reset_index(drop=True)


def has_unsaved_changes(edited: pd.DataFrame, saved: pd.DataFrame) -> bool:
    """Return whether edited journal rows differ semantically from saved rows."""
    return not _canonical_for_change_detection(edited).equals(_canonical_for_change_detection(saved))
