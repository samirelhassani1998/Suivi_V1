from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from app.core.evaluation import evaluate_baselines
from app.core.features import build_features
from app.core.forecasting import forecast_with_ml, forecast_with_sarimax
from app.core.insights import estimate_target_eta
from app.ui.components import alert_banner, empty_state

warnings.filterwarnings("ignore")


def _df() -> pd.DataFrame:
    return st.session_state.get("filtered_data", st.session_state.get("working_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"])))


def _model_comparison(df: pd.DataFrame) -> None:
    st.subheader("Comparaison Régression Linéaire vs Random Forest")
    if len(df) < 10:
        st.warning("Données insuffisantes pour comparer les modèles.")
        return

    feat = build_features(df, height_m=st.session_state.get("height_m", 1.82)).dropna()
    X = feat.drop(columns=["Date", "Poids (Kgs)", "Notes", "Condition de mesure"], errors="ignore").select_dtypes(include=["number"]).fillna(0.0)
    y = feat["Poids (Kgs)"]

    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    if len(X_test) < 2:
        st.warning("Pas assez de données de test.")
        return

    models = {
        "Régression linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "modèle": name,
            "MSE": mean_squared_error(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "R²": r2_score(y_test, pred),
        })
    scores = pd.DataFrame(rows).sort_values("MAE")
    st.dataframe(scores, use_container_width=True)


def _sarima_block(df: pd.DataFrame, horizon: int) -> None:
    st.subheader("SARIMA / SARIMAX")
    try:
        pred = forecast_with_sarimax(df, horizon=horizon)
        if pred.empty:
            st.warning("SARIMAX indisponible: données insuffisantes.")
            return
        fig = go.Figure()
        fig.add_scatter(x=df["Date"], y=df["Poids (Kgs)"], name="Historique")
        fig.add_scatter(x=pred["Date"], y=pred["prevision"], name="Prévision SARIMAX")
        fig.add_scatter(x=pred["Date"], y=pred["borne_basse"], name="Borne basse", line=dict(dash="dot"))
        fig.add_scatter(x=pred["Date"], y=pred["borne_haute"], name="Borne haute", line=dict(dash="dot"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        alert_banner(f"Bloc SARIMAX en erreur: {exc}", "warning")


def _auto_arima_block(df: pd.DataFrame, horizon: int) -> None:
    st.subheader("Auto-ARIMA")
    try:
        from pmdarima import auto_arima

        model = auto_arima(df["Poids (Kgs)"], seasonal=False, error_action="ignore", suppress_warnings=True)
        fc, conf = model.predict(n_periods=horizon, return_conf_int=True)
        dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        out = pd.DataFrame({"Date": dates, "prevision": fc, "borne_basse": conf[:, 0], "borne_haute": conf[:, 1]})
        fig = go.Figure()
        fig.add_scatter(x=df["Date"], y=df["Poids (Kgs)"], name="Historique")
        fig.add_scatter(x=out["Date"], y=out["prevision"], name="Auto-ARIMA")
        fig.add_scatter(x=out["Date"], y=out["borne_basse"], name="Basse", line=dict(dash="dot"))
        fig.add_scatter(x=out["Date"], y=out["borne_haute"], name="Haute", line=dict(dash="dot"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        alert_banner(f"Auto-ARIMA indisponible: {exc}", "warning")


def _ml_quantile_block(df: pd.DataFrame, horizon: int) -> None:
    st.subheader("Prévision ML quantile")
    try:
        pred = forecast_with_ml(df, horizon=horizon, height_m=st.session_state.get("height_m", 1.82))
        if pred.empty:
            st.warning("Prévision ML indisponible: données insuffisantes.")
            return
        fig = go.Figure()
        fig.add_scatter(x=df["Date"], y=df["Poids (Kgs)"], name="Historique")
        fig.add_scatter(x=pred["Date"], y=pred["prevision"], name="Prévision ML")
        fig.add_scatter(x=pred["Date"], y=pred["borne_basse"], name="Basse", line=dict(dash="dot"))
        fig.add_scatter(x=pred["Date"], y=pred["borne_haute"], name="Haute", line=dict(dash="dot"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        alert_banner(f"Bloc ML en erreur: {exc}", "warning")


def _stl_acf_pacf_block(df: pd.DataFrame) -> None:
    st.subheader("STL & ACF/PACF")
    try:
        series = df.set_index("Date")["Poids (Kgs)"].asfreq("D").interpolate()
        if len(series) < 20:
            st.warning("Données insuffisantes pour STL/ACF/PACF")
            return
        stl = STL(series, period=7, robust=True).fit()
        comp = pd.DataFrame({"Date": series.index, "Tendance": stl.trend, "Saisonnalité": stl.seasonal, "Résidu": stl.resid})
        st.line_chart(comp.set_index("Date"))

        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots()
        plot_acf(series, ax=ax1, lags=min(30, len(series) // 2))
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        plot_pacf(series, ax=ax2, lags=min(30, len(series) // 2), method="ywm")
        st.pyplot(fig2)
    except Exception as exc:
        alert_banner(f"Bloc STL/ACF/PACF en erreur: {exc}", "warning")


def main() -> None:
    st.title("Prévisions")
    df = _df()
    if df.empty:
        empty_state("Ajoutez des mesures dans Journal pour lancer les prévisions.")
        return

    df = df.sort_values("Date")
    horizon = st.slider("Horizon (jours)", 7, 90, 30)

    st.subheader("Leaderboard baselines (backtest walk-forward)")
    baseline_df = evaluate_baselines(df["Poids (Kgs)"])
    st.dataframe(baseline_df.sort_values("mae"), use_container_width=True)

    t1, t2, t3, t4 = st.tabs(["Comparaison modèles", "SARIMA", "Auto-ARIMA", "STL / ACF-PACF"])
    fast_mode = st.session_state.get("fast_mode", False)
    with t1:
        _model_comparison(df)
        _ml_quantile_block(df, horizon)
    with t2:
        _sarima_block(df, horizon)
    with t3:
        if fast_mode:
            st.warning("Auto-ARIMA sauté en mode rapide (tests).")
        else:
            _auto_arima_block(df, horizon)
    with t4:
        if fast_mode:
            st.warning("STL/ACF/PACF sauté en mode rapide (tests).")
        else:
            _stl_acf_pacf_block(df)

    eta = estimate_target_eta(df, st.session_state.get("target_weight", 80.0))
    st.subheader("Estimation de date objectif")
    if eta.get("credible"):
        st.write(f"Date cible estimée : **{eta['eta'].date()}**")
        st.caption(f"Plage plausible: {eta['eta_min'].date()} → {eta['eta_max'].date()} | Confiance {eta['confidence']:.0%}")
    else:
        alert_banner(str(eta.get("message", "Estimation indisponible")), "warning")


main()
