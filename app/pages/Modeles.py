
"""Modèles page regrouping model comparison and ML analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score



from app.utils import apply_theme, load_data

# ALWAYS show page proof (non-conditional)
st.caption(f"PAGE={__file__}")
def _get_data():
    df = st.session_state.get("filtered_data")
    if df is None:
        df = load_data()
        st.session_state["filtered_data"] = df
        st.session_state["raw_data"] = df
    return df.copy()


def render_model_comparison(df):
    st.header("Comparaison des Modèles")
    if df.empty or len(df) < 3:
        st.warning("Pas assez de données pour comparer les modèles (minimum 3 entrées).")
        return

    df = df.copy()
    df["Date_numeric"] = (df["Date"] - df["Date"].min()) / np.timedelta64(1, "D")
    X = df[["Date_numeric"]]
    y = df["Poids (Kgs)"]
    n_splits = min(5, len(df) - 1)
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits)
    except ValueError as error:
        st.warning(f"Impossible de configurer la validation croisée : {error}")
        return

    models = {
        "Régression Linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    model_scores = {}
    for name, model in models.items():
        try:
            scores = cross_val_score(
                model,
                X,
                y,
                scoring="neg_mean_squared_error",
                cv=tscv,
            )
        except ValueError as error:
            st.warning(f"Échec de la validation croisée pour {name} : {error}")
            continue
        model_scores[name] = -scores.mean()

    if not model_scores:
        st.warning("Impossible de calculer les scores des modèles avec les données actuelles.")
        return

    scores_df = pd.DataFrame.from_dict(model_scores, orient="index", columns=["MSE"])
    st.write("**Comparaison des MSE moyens :**")
    st.dataframe(scores_df)

    for name, model in models.items():
        try:
            model.fit(X, y)
        except ValueError as error:
            st.warning(f"Impossible d'entraîner le modèle {name} : {error}")
            continue
        df[name + "_Predictions"] = model.predict(X)

    theme = st.session_state.get("theme", "Default")
    fig_compare = px.scatter(df, x="Date", y="Poids (Kgs)", title="Comparaison des Modèles")
    for name in models.keys():
        fig_compare.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[name + "_Predictions"],
                mode="lines",
                name=name,
            )
        )
    fig_compare = apply_theme(fig_compare, theme)
    st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("Métriques de Performance")
    for name in models.keys():
        mse = mean_squared_error(y, df[name + "_Predictions"])
        mae = mean_absolute_error(y, df[name + "_Predictions"])
        r2 = r2_score(y, df[name + "_Predictions"])
        st.write(f"**{name}:** MSE = {mse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")


def render_personalisation(df):
    st.header("Personnalisation et Indicateurs")
    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    target_weights = st.session_state.get(
        "target_weights",
        (95.0, 90.0, 85.0, 80.0),
    )
    theme = st.session_state.get("theme", "Default")

    current_weight = df["Poids (Kgs)"].iloc[-1]
    initial_weight = df["Poids (Kgs)"].iloc[0]

    if current_weight <= target_weights[-1]:
        st.balloons()
        st.success("Félicitations ! Objectif ultime atteint.")
    elif current_weight <= target_weights[2]:
        st.success("Bravo ! Vous avez atteint l'objectif 3.")
    elif current_weight <= target_weights[1]:
        st.info("Bien joué ! Objectif 2 atteint.")
    elif current_weight <= target_weights[0]:
        st.info("Bon travail ! Objectif 1 atteint.")
    else:
        st.warning("Continuez vos efforts pour atteindre vos objectifs.")

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_weight,
            delta={"reference": target_weights[-1]},
            gauge={
                "axis": {"range": [target_weights[-1] - 10, initial_weight + 10]},
                "steps": [
                    {"range": [target_weights[-1], target_weights[2]], "color": "darkgreen"},
                    {"range": [target_weights[2], target_weights[1]], "color": "lightgreen"},
                    {"range": [target_weights[1], target_weights[0]], "color": "yellow"},
                    {"range": [target_weights[0], initial_weight], "color": "orange"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": current_weight,
                },
            },
        )
    )
    fig_gauge.update_layout(title="Progression vers l'Objectif Final")
    fig_gauge = apply_theme(fig_gauge, theme)
    st.plotly_chart(fig_gauge, use_container_width=True)


def render_ml_insights(df):
    st.header("Analyses IA/ML")
    if df.empty or len(df) < 2:
        st.warning("Données insuffisantes pour l'analyse.")
        return

    theme = st.session_state.get("theme", "Default")

    iso = IsolationForest(contamination=0.1, random_state=42)
    df_iso = df.copy()
    df_iso["IF_Anomaly"] = iso.fit_predict(df_iso[["Poids (Kgs)"]]) == -1
    fig_if = px.scatter(
        df_iso,
        x="Date",
        y="Poids (Kgs)",
        color="IF_Anomaly",
        color_discrete_map={False: "blue", True: "red"},
        title="Anomalies détectées",
    )
    fig_if = apply_theme(fig_if, theme)
    st.plotly_chart(fig_if, use_container_width=True)
    st.write("Anomalies détectées :")
    st.dataframe(df_iso[df_iso["IF_Anomaly"]])

    st.subheader("Clustering K-Means")
    max_clusters = min(5, len(df))
    if max_clusters < 2:
        st.warning("Données insuffisantes pour réaliser un clustering.")
        return
    clusters = st.slider(
        "Nombre de clusters",
        min_value=2,
        max_value=max_clusters,
        value=min(3, max_clusters),
        key="kmeans_clusters",
    )
    df_cluster = df.copy()
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    df_cluster["Cluster"] = kmeans.fit_predict(df_cluster[["Poids (Kgs)"]])
    fig_cluster = px.scatter(
        df_cluster,
        x="Date",
        y="Poids (Kgs)",
        color="Cluster",
        title="Clusters de poids (K-Means)",
    )
    fig_cluster = apply_theme(fig_cluster, theme)
    st.plotly_chart(fig_cluster, use_container_width=True)


def main():
    df = _get_data()
    render_model_comparison(df)
    render_personalisation(df)
    render_ml_insights(df)


if __name__ == "__main__":
    main()
