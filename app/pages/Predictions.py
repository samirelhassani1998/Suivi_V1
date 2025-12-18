
"""Prédictions page focusing on forecasting and time-series analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.utils import resample
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf



from app.utils import apply_theme, load_data
from app.deploy import show_deployment_info

# Debug: show page path only if debug_mode is enabled
if st.secrets.get("debug_mode", False):
    st.caption(f"PAGE={__file__}")
def _get_data():
    df = st.session_state.get("filtered_data")
    if df is None:
        df = load_data()
        st.session_state["filtered_data"] = df
        st.session_state["raw_data"] = df
    return df.copy()


def render_linear_regression(df):
    st.header("Régression Linéaire et Intervalles de Confiance")
    if df.empty or len(df) < 3:
        st.warning("Données insuffisantes pour faire des prévisions (minimum 3 entrées).")
        return

    df = df.copy()
    df["Date_numeric"] = (df["Date"] - df["Date"].min()) / np.timedelta64(1, "D")
    X = df[["Date_numeric"]]
    y = df["Poids (Kgs)"]

    n_splits = min(5, len(df) - 1)
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        lin_scores = cross_val_score(
            LinearRegression(),
            X,
            y,
            scoring="neg_mean_squared_error",
            cv=tscv,
        )
        st.write(f"MSE moyen (Régression Linéaire) : **{-lin_scores.mean():.2f}**")
    except ValueError as error:
        st.warning(f"Validation croisée impossible avec les données actuelles : {error}")
        lin_scores = None

    reg = LinearRegression().fit(X, y)
    predictions = reg.predict(X)

    n_bootstraps = 500
    boot_preds = np.zeros((n_bootstraps, len(X)))
    for i in range(n_bootstraps):
        X_boot, y_boot = resample(X, y)
        reg_boot = LinearRegression().fit(X_boot, y_boot)
        boot_preds[i] = reg_boot.predict(X)
    pred_lower = np.percentile(boot_preds, 2.5, axis=0)
    pred_upper = np.percentile(boot_preds, 97.5, axis=0)

    theme = st.session_state.get("theme", "Default")
    fig_reg = px.scatter(
        df,
        x="Date",
        y="Poids (Kgs)",
        title="Régression Linéaire avec IC",
        labels={"Poids (Kgs)": "Poids (en Kgs)"},
    )
    fig_reg.add_trace(
        go.Scatter(
            x=df["Date"],
            y=predictions,
            mode="lines",
            name="Prévisions",
            line=dict(color="blue"),
        )
    )
    fig_reg.add_trace(
        go.Scatter(
            x=df["Date"],
            y=pred_lower,
            mode="lines",
            line_color="lightblue",
            name="IC Inférieur",
        )
    )
    fig_reg.add_trace(
        go.Scatter(
            x=df["Date"],
            y=pred_upper,
            mode="lines",
            line_color="lightblue",
            fill="tonexty",
            name="IC Supérieur",
        )
    )
    fig_reg = apply_theme(fig_reg, theme)
    st.plotly_chart(fig_reg, use_container_width=True)
    
    # MID-AI-1: Model Confidence Metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    st.subheader("📊 Métriques de Confiance du Modèle")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("R²", f"{r2:.3f}", help="Coefficient de détermination (1.0 = parfait)")
    with mcol2:
        st.metric("MAE", f"{mae:.2f} kg", help="Erreur absolue moyenne")
    with mcol3:
        st.metric("RMSE", f"{rmse:.2f} kg", help="Racine de l'erreur quadratique moyenne")
    with mcol4:
        # Confidence level based on R²
        if r2 >= 0.8:
            confidence = "🟢 Haute"
            conf_help = "R² ≥ 0.8 : Le modèle explique bien les données"
        elif r2 >= 0.5:
            confidence = "🟡 Moyenne"
            conf_help = "0.5 ≤ R² < 0.8 : Précision modérée"
        else:
            confidence = "🔴 Faible"
            conf_help = "R² < 0.5 : Prédictions peu fiables"
        st.metric("Confiance", confidence, help=conf_help)
    
    # Width of confidence interval as percentage
    ic_width = (pred_upper - pred_lower).mean()
    st.caption(f"Largeur moyenne de l'intervalle de confiance : ±{ic_width/2:.2f} kg")

    target_weight = st.session_state.get("target_weights", (95.0, 90.0, 85.0, 80.0))[-1]
    try:
        if reg.coef_[0] >= 0:
            st.error("Le modèle n’indique pas une perte de poids. Impossible d’estimer la date d’atteinte de l’objectif.")
        else:
            days_to_target = (target_weight - reg.intercept_) / reg.coef_[0]
            target_date = df["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
            if target_date < df["Date"].max():
                st.warning("L'objectif a déjà été atteint selon les prévisions.")
            else:
                st.write(f"**Date estimée** pour atteindre {target_weight} Kgs : {target_date.date()}")
    except Exception as error:
        st.error(f"Erreur dans le calcul de la date estimée : {error}")

    future_days = st.slider("Nombre de jours à prédire", 1, 365, 30)
    future_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=future_days)
    future_numeric = (future_dates - df["Date"].min()) / np.timedelta64(1, "D")
    future_predictions = reg.predict(future_numeric.values.reshape(-1, 1))
    future_df = pd.DataFrame({"Date": future_dates, "Prévisions": future_predictions})
    fig_future = px.line(future_df, x="Date", y="Prévisions", title="Prévisions Futures")
    fig_future = apply_theme(fig_future, theme)
    st.plotly_chart(fig_future, use_container_width=True)

    df["Poids_diff"] = df["Poids (Kgs)"].diff()
    mean_change_rate = df["Poids_diff"].mean()
    st.write(f"Taux de changement moyen du poids : **{mean_change_rate:.2f} Kgs/jour**")
    coef_var = df["Poids (Kgs)"].std() / df["Poids (Kgs)"].mean()
    st.write(f"Coefficient de variation du poids : **{coef_var * 100:.2f}%**")


def render_stl_and_sarima(df):
    st.header("Analyse STL et SARIMA")
    if df.empty:
        st.warning("Aucune donnée pour l’analyse.")
        return

    period = 7
    if len(df) < period * 2:
        st.warning("Au moins 14 enregistrements sont nécessaires pour l'analyse STL/SARIMA.")
        return

    theme = st.session_state.get("theme", "Default")
    df = df.copy()

    st.subheader("Analyse STL (Tendance et Saisonnalité)")
    try:
        stl = STL(df.set_index("Date")["Poids (Kgs)"], period=period)
        res = stl.fit()
    except ValueError as error:
        st.warning(f"Impossible de réaliser la décomposition STL : {error}")
        return
    df["Trend"] = res.trend.values
    df["Seasonal"] = res.seasonal.values
    df["Resid"] = res.resid.values

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_trend = px.line(df, x="Date", y="Trend", title="Tendance")
        fig_trend = apply_theme(fig_trend, theme)
        st.plotly_chart(fig_trend, use_container_width=True)
    with col2:
        fig_seasonal = px.line(df, x="Date", y="Seasonal", title="Saisonnalité")
        fig_seasonal = apply_theme(fig_seasonal, theme)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    with col3:
        fig_resid = px.line(df, x="Date", y="Resid", title="Résidus")
        fig_resid = apply_theme(fig_resid, theme)
        st.plotly_chart(fig_resid, use_container_width=True)

    st.subheader("Prédictions SARIMA")
    try:
        with st.spinner("🔄 Calcul du modèle SARIMA en cours..."):
            sarima_model = SARIMAX(df["Poids (Kgs)"], order=(1, 1, 1), seasonal_order=(1, 1, 1, period))
            sarima_results = sarima_model.fit(disp=False)
            df["SARIMA_Predictions"] = sarima_results.predict(start=0, end=len(df) - 1, dynamic=False)
        fig_sarima = px.scatter(df, x="Date", y="Poids (Kgs)", title="SARIMA")
        fig_sarima.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["SARIMA_Predictions"],
                mode="lines",
                name="Prévisions SARIMA",
            )
        )
        fig_sarima = apply_theme(fig_sarima, theme)
        st.plotly_chart(fig_sarima, use_container_width=True)
    except Exception as error:  # statsmodels peut lever plusieurs types d'exceptions
        st.warning(f"Le modèle SARIMA n'a pas pu converger : {error}")

    st.subheader("Variabilité et Corrélations Temporelles")
    max_window = min(30, len(df))
    if max_window < 3:
        st.warning("Données insuffisantes pour calculer un écart type mobile.")
        return
    std_window = st.slider(
        "Fenêtre pour l'écart type mobile",
        min_value=3,
        max_value=max_window,
        value=min(7, max_window),
        key="std_window",
    )
    df["Rolling_STD"] = df["Poids (Kgs)"].rolling(window=std_window).std()
    fig_std = px.line(df, x="Date", y="Rolling_STD", title="Écart Type Mobile")
    fig_std = apply_theme(fig_std, theme)
    st.plotly_chart(fig_std, use_container_width=True)

    max_lag = min(30, len(df) - 1)
    if max_lag < 1:
        st.warning("Pas assez d'observations pour calculer l'ACF/PACF.")
        return
    lag = st.slider(
        "Nombre de décalages pour l'ACF/PACF",
        min_value=1,
        max_value=max_lag,
        value=min(7, max_lag),
        key="acf_lag",
    )
    try:
        acf_vals = acf(df["Poids (Kgs)"], nlags=lag, missing="drop")
        pacf_vals = pacf(df["Poids (Kgs)"], nlags=lag, method="ywm")
    except ValueError as error:
        st.warning(f"Impossible de calculer l'ACF/PACF : {error}")
        return
    fig_acf = px.bar(x=list(range(len(acf_vals))), y=acf_vals, title="Autocorrélation (ACF)")
    fig_pacf = px.bar(x=list(range(len(pacf_vals))), y=pacf_vals, title="Autocorrélation Partielle (PACF)")
    fig_acf = apply_theme(fig_acf, theme)
    fig_pacf = apply_theme(fig_pacf, theme)
    st.plotly_chart(fig_acf, use_container_width=True)
    st.plotly_chart(fig_pacf, use_container_width=True)


def render_auto_arima(df):
    st.header("Prévisions Auto-ARIMA")
    if df.empty:
        st.warning("Aucune donnée pour générer des prévisions Auto-ARIMA.")
        return

    if len(df) < 10:
        st.warning("Au moins 10 enregistrements sont nécessaires pour entraîner Auto-ARIMA.")
        return

    theme = st.session_state.get("theme", "Default")
    future_arima_days = st.slider(
        "Jours à prédire",
        min_value=1,
        max_value=60,
        value=30,
        key="arima_days",
    )
    try:
        with st.spinner("🔄 Calcul Auto-ARIMA en cours (peut prendre quelques secondes)..."):
            arima_model = auto_arima(
                df["Poids (Kgs)"],
                seasonal=True,
                m=7,
                suppress_warnings=True,
            )
            arima_forecast = arima_model.predict(n_periods=future_arima_days)
    except Exception as error:  # pmdarima renvoie divers types d'erreurs
        st.warning(f"Le modèle Auto-ARIMA n'a pas pu être ajusté : {error}")
        return

    arima_dates = pd.date_range(
        start=df["Date"].max() + pd.Timedelta(days=1),
        periods=future_arima_days,
    )
    fig_arima = px.line(
        x=arima_dates,
        y=arima_forecast,
        labels={"x": "Date", "y": "Prévision (Kgs)"},
        title="Prévisions Auto-ARIMA",
    )
    fig_arima.add_scatter(
        x=df["Date"],
        y=df["Poids (Kgs)"],
        mode="lines",
        name="Historique",
    )
    fig_arima = apply_theme(fig_arima, theme)
    st.plotly_chart(fig_arima, use_container_width=True)


def main():
    df = _get_data()
    render_linear_regression(df)
    render_stl_and_sarima(df)
    render_auto_arima(df)


if __name__ == "__main__":
    main()
main()
show_deployment_info()
