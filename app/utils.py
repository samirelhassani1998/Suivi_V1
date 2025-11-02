"""Utility helpers shared across Streamlit pages."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import IsolationForest

DATA_URL = "https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv"


@st.cache_data(ttl=300)
def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load and clean the dataset from the provided URL."""
    df = pd.read_csv(url, decimal=",")
    df["Poids (Kgs)"] = (
        df["Poids (Kgs)"]
        .astype(str)
        .str.replace(",", ".")
        .str.strip()
    )
    df["Poids (Kgs)"] = pd.to_numeric(df["Poids (Kgs)"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Poids (Kgs)", "Date"]).sort_values("Date", ascending=True)
    return df


def apply_theme(fig: go.Figure, theme_name: str) -> go.Figure:
    """Apply a plotly theme to the provided figure."""
    theme_templates = {
        "Dark": "plotly_dark",
        "Light": "plotly_white",
        "Solar": "plotly_solar",
        "Seaborn": "seaborn",
    }
    if theme_name in theme_templates:
        fig.update_layout(template=theme_templates[theme_name])
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
