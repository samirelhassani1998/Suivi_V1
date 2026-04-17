from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.evaluation import evaluate_baselines
from app.core.forecasting import forecast_with_ml, forecast_with_sarimax
from app.core.insights import estimate_target_eta
from app.ui.components import alert_banner, empty_state


def _df() -> pd.DataFrame:
    return st.session_state.get("filtered_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"]))


@st.fragment
def _heavy_predictions(df: pd.DataFrame, horizon: int, height_m: float) -> None:
    sarimax = forecast_with_sarimax(df, horizon=horizon)
    quantile = forecast_with_ml(df, horizon=horizon, height_m=height_m)

    if sarimax.empty and quantile.empty:
        alert_banner("Données insuffisantes pour calculer des prévisions fiables.", "warning")
        return

    fig = go.Figure()
    fig.add_scatter(x=df["Date"], y=df["Poids (Kgs)"], name="Historique")
    for name, pred in [("SARIMAX", sarimax), ("Quantile", quantile)]:
        if pred.empty:
            continue
        fig.add_scatter(x=pred["Date"], y=pred["prevision"], name=f"Prévision {name}")
        fig.add_scatter(x=pred["Date"], y=pred["borne_basse"], name=f"Borne basse {name}", line=dict(dash="dot"))
        fig.add_scatter(x=pred["Date"], y=pred["borne_haute"], name=f"Borne haute {name}", line=dict(dash="dot"))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Prévisions")
    df = _df()
    if df.empty:
        empty_state("Ajoutez des mesures dans Journal pour lancer les prévisions.")
        return

    horizon = st.slider("Horizon (jours)", 7, 90, 30)
    baseline_df = evaluate_baselines(df["Poids (Kgs)"])
    st.subheader("Leaderboard baselines (backtest walk-forward)")
    st.dataframe(baseline_df.sort_values("mae"), use_container_width=True)

    _heavy_predictions(df, horizon=horizon, height_m=st.session_state.get("height_m", 1.82))

    eta = estimate_target_eta(df, st.session_state.get("target_weight", 80.0))
    st.subheader("Estimation de date objectif")
    if eta.get("credible"):
        st.write(f"Date cible estimée : **{eta['eta'].date()}**")
        st.caption(f"Plage plausible: {eta['eta_min'].date()} → {eta['eta_max'].date()} | Confiance {eta['confidence']:.0%}")
    else:
        alert_banner(str(eta.get("message", "Estimation indisponible")), "warning")


main()
