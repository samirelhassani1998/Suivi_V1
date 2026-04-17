"""Utility helpers shared across Streamlit pages."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import plotly.graph_objects as go
import streamlit as st

from scipy import stats
from sklearn.ensemble import IsolationForest 

DATA_URL = "https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv"

PLOTLY_TEMPLATES = {
    "Dark": "plotly_dark",
    "Light": "plotly_white",
    "Solar": "plotly",
    "Seaborn": "seaborn",
}


def _resolve_data_url(default_url: str) -> str:
    """Resolve data URL from secrets when available, fallback otherwise."""
    try:
        return st.secrets.get("data_url", default_url)
    except Exception:
        return default_url


def _read_csv_with_fallbacks(url: str) -> pd.DataFrame:
    """Read CSV using resilient separator/encoding fallbacks."""
    read_attempts = [
        {"sep": None, "engine": "python", "decimal": ","},
        {"sep": ",", "decimal": ","},
        {"sep": ";", "decimal": ","},
        {"sep": "\t", "decimal": ","},
    ]
    errors: list[str] = []
    for options in read_attempts:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(url, encoding=encoding, **options)
            except Exception as error:  # pragma: no cover - diagnostic fallback
                errors.append(f"{encoding}/{options}: {type(error).__name__}")
    raise RuntimeError(
        "Impossible de parser le CSV (séparateur/encodage). "
        f"Tentatives: {', '.join(errors[:5])}"
    )


def _normalise_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise common CSV header variants for required columns."""
    normalised_cols = {}
    for col in df.columns:
        clean = str(col).replace("\ufeff", "").strip()
        lowered = clean.lower()
        if lowered in {"date", "dates"}:
            normalised_cols[col] = "Date"
        elif lowered in {"poids (kgs)", "poids", "poids(kg)", "poids (kg)"}:
            normalised_cols[col] = "Poids (Kgs)"
        else:
            normalised_cols[col] = clean
    return df.rename(columns=normalised_cols)


@st.cache_data(ttl=300, show_spinner=False)
def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load and clean the dataset from the provided URL.

    The Google Sheets export occasionally returns transient HTTP/empty payload
    errors.  We normalise the weight column, convert the dates and keep only
    valid rows.  Any failure is wrapped into a ``RuntimeError`` so that the
    caller can display a helpful message instead of crashing the Streamlit app.
    """

    url = _resolve_data_url(url)

    try:
        df = _read_csv_with_fallbacks(url)
    except (EmptyDataError, ParserError, ValueError) as error:
        raise RuntimeError("Le fichier de données est vide ou invalide.") from error
    except Exception as error:  # pragma: no cover
        raise RuntimeError("Impossible de télécharger ou parser les données distantes.") from error

    if df.empty:
        raise RuntimeError("Aucune donnée disponible dans la source distante.")

    df = _normalise_expected_columns(df)

    # Robust column cleaning
    if "Poids (Kgs)" not in df.columns or "Date" not in df.columns:
         raise RuntimeError("Le fichier ne contient pas les colonnes requises ('Poids (Kgs)', 'Date').")

    df["Poids (Kgs)"] = (
        df["Poids (Kgs)"]
        .astype(str)
        .str.replace(",", ".")
        .str.strip()
    )
    df["Poids (Kgs)"] = pd.to_numeric(df["Poids (Kgs)"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    df = (
        df.dropna(subset=["Poids (Kgs)", "Date"])
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError("Les données récupérées ne contiennent aucune date/valeur valide.")

    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_data_diagnostics(url: str = DATA_URL) -> Dict[str, Any]:
    """Return row counts through the CSV -> DataFrame preparation pipeline."""
    url = _resolve_data_url(url)
    df_raw = _normalise_expected_columns(_read_csv_with_fallbacks(url))
    raw_rows = int(df_raw.shape[0])

    if "Poids (Kgs)" not in df_raw.columns or "Date" not in df_raw.columns:
        return {
            "raw_rows": raw_rows,
            "valid_rows": 0,
            "final_rows": 0,
            "dropped_invalid_rows": raw_rows,
            "duplicate_date_rows": 0,
        }

    df_work = df_raw.copy()
    df_work["Poids (Kgs)"] = (
        df_work["Poids (Kgs)"].astype(str).str.replace(",", ".").str.strip()
    )
    df_work["Poids (Kgs)"] = pd.to_numeric(df_work["Poids (Kgs)"], errors="coerce")
    df_work["Date"] = pd.to_datetime(df_work["Date"], dayfirst=True, errors="coerce")
    valid_df = df_work.dropna(subset=["Poids (Kgs)", "Date"])
    final_df = valid_df.sort_values("Date", ascending=True).reset_index(drop=True)
    return {
        "raw_rows": raw_rows,
        "valid_rows": int(valid_df.shape[0]),
        "final_rows": int(final_df.shape[0]),
        "dropped_invalid_rows": int(raw_rows - valid_df.shape[0]),
        "duplicate_date_rows": int(valid_df.duplicated(subset=["Date"]).sum()),
    }


def apply_theme(fig: go.Figure, theme_name: str) -> go.Figure:
    """Apply a plotly theme to the provided figure."""
    template = PLOTLY_TEMPLATES.get(theme_name)
    if template:
        fig.update_layout(template=template)
    fig.update_layout(
        title=dict(font=dict(size=22)),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )
    return fig


@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Return a CSV export encoded as UTF-8."""
    return df.to_csv(index=False).encode("utf-8")


def detect_anomalies(
    df: pd.DataFrame,
    method: str,
    z_threshold: float = 2.0,
    contamination: float = 0.1,
) -> pd.DataFrame:
    """Detect weight anomalies using the requested method."""
    df = df.copy()
    if df.empty:
        df["Anomalies"] = []
        return df

    if method == "IsolationForest":
        model = IsolationForest(contamination=contamination, random_state=42)
        df["Anomalies"] = model.fit_predict(df[["Poids (Kgs)"]]) == -1
    else:
        df["Z_score"] = np.abs(stats.zscore(df["Poids (Kgs)"]))
        df["Anomalies"] = df["Z_score"] > z_threshold
    return df


def get_date_range(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return the minimum and maximum available dates."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    return df["Date"].min(), df["Date"].max()


def filter_by_dates(
    df: pd.DataFrame,
    date_range: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """Filter the dataset based on a date range coming from the date_input widget."""
    filtered = df
    if len(date_range) == 2:
        start_date, end_date = map(pd.to_datetime, date_range)
        filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    return filtered


def calculate_moving_average(
    df: pd.DataFrame,
    column: str,
    window_size: int,
    method: str = "Simple",
) -> pd.Series:
    """Calculate moving average for a specific column."""
    if method == "Exponentielle":
        return df[column].ewm(span=window_size, adjust=False).mean()
    return df[column].rolling(window=window_size, min_periods=1).mean()


def train_linear_regression(
    df: pd.DataFrame, target_col: str, date_col: str = "Date"
) -> Tuple[object, float]:
    """Train a simple linear regression model on date (numeric)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    df = df.copy()
    df["Date_numeric"] = (df[date_col] - df[date_col].min()) / np.timedelta64(1, "D")
    X = df[["Date_numeric"]]
    y = df[target_col]

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    
    return model, mse


def predict_future_linear(
    model: object,
    df: pd.DataFrame,
    days: int,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Generate future predictions using a trained linear model."""
    last_date = df[date_col].max()
    start_date = df[date_col].min()
    
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    future_numeric = (future_dates - start_date) / np.timedelta64(1, "D")
    
    predictions = model.predict(future_numeric.values.reshape(-1, 1))
    return pd.DataFrame({date_col: future_dates, "Prediction": predictions})
