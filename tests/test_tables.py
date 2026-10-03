"""Tests des aides partagées par les onglets Whoop et Boxe : tableaux et filtre de période."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.whoop_analytics import filter_period
from app.ui.tables import format_table


def test_format_table_renders_dates_durations_clocks_booleans_and_decimals():
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-09-08")],
            "Durée (min)": [90.4],
            "Coucher suivant": [-1.5],
            "Heure de coucher": [0.5],
            "Calibration": [True],
            "Strain séance": [13.9323],
            "Récupération (%)": [np.nan],
        }
    )

    display = format_table(frame, {"Strain séance": 1}, default_decimals={"Récupération (%)": 0})

    row = display.iloc[0]
    assert row["Date"] == "mar. 8 sept."
    assert row["Durée (min)"] == "1 h 30"
    assert row["Coucher suivant"] == "22:30"
    assert row["Heure de coucher"] == "00:30"
    assert row["Calibration"] == "oui"
    assert row["Strain séance"] == "13,9"
    assert row["Récupération (%)"] == "—"


def test_format_table_tolerates_empty_frames():
    assert format_table(None).empty
    assert format_table(pd.DataFrame()).empty


def test_filter_period_counts_from_today_not_from_the_last_measurement():
    frame = pd.DataFrame({"Date": pd.date_range("2026-09-01", "2026-09-20", freq="D")})

    kept = filter_period(frame, 7, today="2026-09-24")

    # Du 18 au 24 septembre : seules trois mesures existent dans cette fenêtre.
    assert kept["Date"].tolist() == list(pd.date_range("2026-09-18", "2026-09-20", freq="D"))
    assert len(filter_period(frame, None)) == 20
