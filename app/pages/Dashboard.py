from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.data import data_quality_report
from app.core.insights import detect_plateau
from app.core.session_state import get_filtered_or_working_data
from app.core.targets import get_target_weights, normalise_target_weights
from app.ui.components import alert_banner, confidence_badge, empty_state, help_box, kpi_card


def _df() -> pd.DataFrame:
    return get_filtered_or_working_data()


def _target_annotation_shift(targets: tuple[float, ...], index: int) -> int:
    target = round(float(targets[index]), 3)
    duplicate_position = sum(1 for previous in targets[:index] if round(float(previous), 3) == target)
    return duplicate_position * 18


def _add_target_lines(fig: go.Figure, targets: tuple[float, ...]) -> None:
    """Draw the five configured target lines on the weight evolution graph."""
    targets = normalise_target_weights(targets)
    colors = ("#2E7BCF", "#27AE60", "#F39C12", "#8E44AD", "#E74C3C")
    for idx, target in enumerate(targets, start=1):
        fig.add_hline(
            y=float(target),
            line_dash="dash",
            line_color=colors[(idx - 1) % len(colors)],
            annotation_text=f"Objectif {idx}: {float(target):.1f} kg",
            annotation_position="top right",
            annotation_yshift=_target_annotation_shift(targets, idx - 1),
        )


def _targets_caption(targets: tuple[float, ...]) -> str:
    goals = " · ".join(f"Objectif {idx}: {target:.1f} kg" for idx, target in enumerate(targets, start=1))
    return f"🎯 Objectifs affichés : {goals}."


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
    targets = get_target_weights(st.session_state)

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
    _add_target_lines(fig, targets)
    st.caption(_targets_caption(targets))
    st.plotly_chart(fig, use_container_width=True)

    help_box("Résumé hebdo", "Cette vue est analytique et informative; elle ne remplace pas un avis médical.")


main()
