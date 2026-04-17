from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.data import data_quality_report
from app.core.insights import detect_plateau
from app.ui.components import alert_banner, confidence_badge, empty_state, help_box, kpi_card


def _df() -> pd.DataFrame:
    return st.session_state.get("filtered_data", st.session_state.get("working_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"])))


def _moving_average(series: pd.Series, window: int, ma_type: str) -> pd.Series:
    if ma_type == "Exponentielle":
        return series.ewm(span=max(window, 2), adjust=False).mean()
    return series.rolling(window=window, min_periods=1).mean()


def main() -> None:
    st.title("Dashboard")
    df = _df()
    if df.empty:
        empty_state("Aucune donnée disponible. Utilisez Journal ou import CSV.")
        return

    df = df.sort_values("Date").copy()
    current = df["Poids (Kgs)"].iloc[-1]
    prev = df["Poids (Kgs)"].iloc[-2] if len(df) > 1 else current
    short = df["Poids (Kgs)"].tail(7).mean()
    long = df["Poids (Kgs)"].tail(30).mean() if len(df) >= 30 else df["Poids (Kgs)"].mean()

    height_m = st.session_state.get("height_m", 1.82)
    imc = current / (height_m**2)
    targets = st.session_state.get("target_weights", (95.0, 90.0, 85.0, 80.0))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Poids actuel", f"{current:.2f} kg", f"{current-prev:+.2f}")
    with c2:
        kpi_card("Tendance 7j", f"{short:.2f} kg")
    with c3:
        kpi_card("Tendance 30j", f"{long:.2f} kg")
    with c4:
        kpi_card("IMC", f"{imc:.2f}")
    with c5:
        quality = data_quality_report(df)
        kpi_card("Qualité des données", f"{quality['score']}/100")

    initial = df["Poids (Kgs)"].iloc[0]
    final_target = targets[-1]
    total = initial - final_target
    progress = ((initial - current) / total * 100) if total > 0 else 0.0
    progress = max(0.0, min(100.0, progress))
    st.progress(progress / 100)
    st.caption(f"Progression vers l'objectif final ({final_target:.1f} kg): {progress:.1f}%")

    plateau = detect_plateau(df, window=14)
    if plateau["status"] == "plateau probable":
        alert_banner("Plateau probable détecté sur 14 jours.", "warning")

    confidence = "élevée" if quality["score"] > 80 else "moyenne" if quality["score"] > 60 else "faible"
    confidence_badge("Confiance signal", confidence)

    ma_type = st.session_state.get("ma_type", "Simple")
    window_size = int(st.session_state.get("window_size", 7))
    df["Poids_MA"] = _moving_average(df["Poids (Kgs)"], window_size, ma_type)

    fig = px.line(df, x="Date", y="Poids (Kgs)", title="Évolution du poids")
    fig.add_scatter(x=df["Date"], y=df["Poids_MA"], mode="lines", name=f"MM {ma_type} ({window_size}j)")
    fig.add_hline(y=float(df["Poids (Kgs)"].mean()), line_dash="dot", annotation_text="Moyenne globale")

    for idx, target in enumerate(targets, start=1):
        fig.add_hline(y=float(target), line_dash="dash", annotation_text=f"Objectif {idx}: {target:.1f} kg")
    st.plotly_chart(fig, use_container_width=True)

    wk = df["Poids (Kgs)"].tail(7).mean()
    prev_wk = df["Poids (Kgs)"].iloc[-14:-7].mean() if len(df) >= 14 else wk
    delta_wk = wk - prev_wk
    st.metric("Comparaison hebdo", f"{wk:.2f} kg", f"{delta_wk:+.2f} kg", delta_color="inverse")

    with st.expander("Distribution et évolution IMC"):
        hist = px.histogram(df, x="Poids (Kgs)", nbins=25, title="Distribution du poids")
        st.plotly_chart(hist, use_container_width=True)
        df["IMC"] = df["Poids (Kgs)"] / (height_m**2)
        bmi_line = px.line(df, x="Date", y="IMC", title="Évolution de l'IMC")
        st.plotly_chart(bmi_line, use_container_width=True)

    help_box("Résumé hebdo", "Cette vue est analytique et informative; elle ne remplace pas un avis médical.")


main()
