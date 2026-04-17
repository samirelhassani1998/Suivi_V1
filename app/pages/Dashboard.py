from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.core.data import data_quality_report
from app.core.insights import detect_plateau
from app.ui.components import alert_banner, confidence_badge, empty_state, help_box, kpi_card


def _df() -> pd.DataFrame:
    return st.session_state.get("filtered_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"]))


def main() -> None:
    st.title("Dashboard")
    df = _df()
    if df.empty:
        empty_state("Aucune donnée disponible. Utilisez Journal ou import CSV.")
        return

    df = df.sort_values("Date")
    current = df["Poids (Kgs)"].iloc[-1]
    prev = df["Poids (Kgs)"].iloc[-2] if len(df) > 1 else current
    short = df["Poids (Kgs)"].tail(7).mean()
    long = df["Poids (Kgs)"].tail(30).mean() if len(df) >= 30 else df["Poids (Kgs)"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Poids actuel", f"{current:.2f} kg", f"{current-prev:+.2f}")
    with c2:
        kpi_card("Tendance 7j", f"{short:.2f} kg")
    with c3:
        kpi_card("Tendance longue", f"{long:.2f} kg")
    with c4:
        quality = data_quality_report(df)
        kpi_card("Qualité des données", f"{quality['score']}/100")

    plateau = detect_plateau(df, window=14)
    if plateau["status"] == "plateau probable":
        alert_banner("Plateau probable détecté sur 14 jours.", "warning")

    confidence = "élevée" if quality["score"] > 80 else "moyenne" if quality["score"] > 60 else "faible"
    confidence_badge("Confiance signal", confidence)

    fig = px.line(df, x="Date", y="Poids (Kgs)", title="Évolution du poids")
    st.plotly_chart(fig, use_container_width=True)

    help_box("Résumé hebdo", "Cette vue est analytique et informative; elle ne remplace pas un avis médical.")


main()
