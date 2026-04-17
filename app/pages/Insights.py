from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans

from app.core.data import data_quality_report
from app.core.insights import detect_anomalies_robust, detect_plateau
from app.ui.components import empty_state


def _df() -> pd.DataFrame:
    return st.session_state.get("filtered_data", st.session_state.get("working_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"])))


def main() -> None:
    st.title("Insights / Analyse")
    df = _df()
    if df.empty:
        empty_state("Pas encore de données à analyser.")
        return

    df = df.sort_values("Date").copy()
    quality = data_quality_report(df)
    st.subheader("Qualité et plateau")
    st.json(quality)

    plateau14 = detect_plateau(df, 14)
    plateau30 = detect_plateau(df, 30)
    st.write(f"14j: **{plateau14['status']}** | pente={plateau14['slope']:.3f}")
    st.write(f"30j: **{plateau30['status']}** | pente={plateau30['slope']:.3f}")

    t1, t2 = st.tabs(["Anomalies", "Clustering"])
    with t1:
        anomalies = detect_anomalies_robust(df, use_iforest=st.toggle("Activer IsolationForest", value=True))
        st.dataframe(anomalies[["Date", "Poids (Kgs)", "anomalie", "raison", "decision"]], use_container_width=True)
        fig_anom = px.scatter(anomalies, x="Date", y="Poids (Kgs)", color="anomalie", title="Anomalies détectées")
        st.plotly_chart(fig_anom, use_container_width=True)

    with t2:
        max_clusters = min(6, len(df))
        if max_clusters < 2:
            st.warning("Données insuffisantes pour KMeans.")
        else:
            k = st.slider("Nombre de clusters", 2, max_clusters, 3)
            clustered = df.copy()
            clustered["cluster"] = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(clustered[["Poids (Kgs)"]])
            fig_cluster = px.scatter(clustered, x="Date", y="Poids (Kgs)", color="cluster", title="Clusters KMeans")
            st.plotly_chart(fig_cluster, use_container_width=True)


main()
