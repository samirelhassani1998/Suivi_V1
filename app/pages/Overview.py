"""Analyse page for the multipage Streamlit application."""


import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st



from app.utils import (
    apply_theme,
    calculate_moving_average,
    convert_df_to_csv,
    detect_anomalies,
    DATA_URL,
    load_data,
)


def _reset_data_cache() -> None:
    """Clear cached data and session copies before retrying a load."""

    load_data.clear()
    st.session_state.pop("filtered_data", None)
    st.session_state.pop("raw_data", None)


def _render_empty_state(title: str, details: str, show_retry: bool = False) -> None:
    """Display a guided empty/error state with optional retry button."""

    st.warning(title)
    st.info(details)
    if show_retry and st.button("↻ Réessayer le chargement", type="primary"):
        _reset_data_cache()
        st.rerun()


def _get_data():
    """Fetch dataset from session or remote source with resilience."""

    cached_df = st.session_state.get("filtered_data")
    if cached_df is not None:
        return cached_df.copy()

    data_url = st.session_state.get("data_url") or st.secrets.get("data_url", DATA_URL)
    st.session_state["data_url"] = data_url

    with st.spinner("Chargement des données de poids..."):
        try:
            df = load_data(data_url)
        except RuntimeError as error:
            _render_empty_state(
                "Impossible de charger les données.",
                "Vérifiez la connectivité, la permission Google Sheets et la clé `data_url` "
                "dans `st.secrets`. Configurez `data_url` ou placez votre CSV accessible "
                "publiquement.",
                show_retry=True,
            )
            st.caption(str(error))
            return None

    st.session_state["raw_data"] = df
    st.session_state["filtered_data"] = df
    return df.copy()


def render_summary(df):
    st.header("Résumé")
    if df.empty:
        _render_empty_state(
            "Aucune donnée disponible.",
            "Ajoutez des lignes avec les colonnes `Date` (jj/mm/aaaa) et `Poids (Kgs)`.",
        )
        return

    height_m = st.session_state.get("height_m", 1.82)
    target_weights = st.session_state.get(
        "target_weights",
        (95.0, 90.0, 85.0, 80.0),
    )

    initial_weight = df["Poids (Kgs)"].iloc[0]
    current_weight = df["Poids (Kgs)"].iloc[-1]
    weight_lost = initial_weight - current_weight
    total_weight_to_lose = (
        initial_weight - target_weights[-1]
        if initial_weight > target_weights[-1]
        else 1.0
    )
    progress_percent = (weight_lost / total_weight_to_lose) * 100
    current_bmi = current_weight / (height_m ** 2)

    # Calcul des variations
    def get_variation(days_ago: int) -> float | None:
        target_date = df["Date"].iloc[-1] - pd.Timedelta(days=days_ago)
        # Trouver la date la plus proche
        closest_idx = (df["Date"] - target_date).abs().idxmin()
        closest_date = df["Date"].iloc[closest_idx]
        
        # Si la date est trop éloignée (> 2 jours d'écart), on ignore
        if abs((closest_date - target_date).days) > 2:
            return None
            
        old_weight = df["Poids (Kgs)"].iloc[closest_idx]
        return current_weight - old_weight

    var_7d = get_variation(7)
    var_30d = get_variation(30)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Poids Actuel", f"{current_weight:.2f} kg", f"{current_weight - df['Poids (Kgs)'].iloc[-2]:.2f} kg" if len(df) > 1 else None, delta_color="inverse")
    col2.metric("Objectif", f"{target_weights[-1]:.2f} kg")
    col3.metric("7 Jours", f"{var_7d:+.2f} kg" if var_7d is not None else "N/A", delta_color="inverse")
    col4.metric("30 Jours", f"{var_30d:+.2f} kg" if var_30d is not None else "N/A", delta_color="inverse")
    col5.metric("IMC", f"{current_bmi:.2f}")

    st.progress(min(max(progress_percent / 100, 0.0), 1.0))
    st.caption(f"Avancement vers l'objectif final : {progress_percent:.1f}%")

    with st.expander("Afficher les Statistiques Descriptives"):
        st.dataframe(df["Poids (Kgs)"].describe())

    st.subheader("Interprétation de l'IMC")
    if current_bmi < 18.5:
        st.info("**Sous-poids** (IMC < 18.5). [OMS](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")
    elif 18.5 <= current_bmi < 25:
        st.success("**Poids Normal** (18.5 ≤ IMC < 25). [OMS](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")
    elif 25 <= current_bmi < 30:
        st.warning("**Surpoids** (25 ≤ IMC < 30). [OMS](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")
    else:
        st.error("**Obésité** (IMC ≥ 30). [OMS](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")


def render_graphs(df):
    st.header("Graphiques")
    if df.empty:
        st.warning("Pas de données à afficher.")
        return

    theme = st.session_state.get("theme", "Default")
    ma_type = st.session_state.get("ma_type", "Simple")
    window_size = st.session_state.get("window_size", 7)
    height_m = st.session_state.get("height_m", 1.82)
    target_weights = st.session_state.get(
        "target_weights",
        (95.0, 90.0, 85.0, 80.0),
    )

    df = df.copy()
    df["Poids_MA"] = calculate_moving_average(df, "Poids (Kgs)", window_size, ma_type)

    fig = px.line(
        df,
        x="Date",
        y="Poids (Kgs)",
        markers=True,
        title="Évolution du Poids dans le Temps",
        labels={"Poids (Kgs)": "Poids (en Kgs)"},
    )
    mean_weight = df["Poids (Kgs)"].mean()
    fig.add_hline(
        y=mean_weight,
        line_dash="dot",
        annotation_text="Moyenne Globale",
        annotation_position="bottom right",
    )
    fig.add_scatter(
        x=df["Date"],
        y=df["Poids_MA"],
        mode="lines",
        name=f"Moyenne Mobile ({ma_type}) {window_size} jours",
    )

    objective_colors = [None, "red", "green", "purple"]
    for idx, value in enumerate(target_weights):
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=objective_colors[idx] if idx < len(objective_colors) else None,
            annotation_text=f"Objectif {idx + 1}",
            annotation_position="bottom right",
        )

    fig = apply_theme(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    fig_hist = px.histogram(df, x="Poids (Kgs)", nbins=30, title="Distribution des Poids")
    fig_hist = apply_theme(fig_hist, theme)
    st.plotly_chart(fig_hist, use_container_width=True)

    df["IMC"] = df["Poids (Kgs)"] / (height_m ** 2)
    fig_bmi = px.line(
        df,
        x="Date",
        y="IMC",
        markers=True,
        title="Évolution de l'IMC",
        labels={"IMC": "Indice de Masse Corporelle"},
    )
    fig_bmi = apply_theme(fig_bmi, theme)
    st.plotly_chart(fig_bmi, use_container_width=True)

    fig_bmi_hist = px.histogram(df, x="IMC", nbins=30, title="Distribution de l'IMC")
    fig_bmi_hist = apply_theme(fig_bmi_hist, theme)
    st.plotly_chart(fig_bmi_hist, use_container_width=True)


def render_anomalies(df):
    st.header("Anomalies")
    if df.empty:
        st.warning("Aucune donnée pour détecter des anomalies.")
        return

    anomaly_method = st.session_state.get("anomaly_method", "Z-score")
    z_threshold = st.session_state.get("z_threshold", 2.0)
    theme = st.session_state.get("theme", "Default")

    df_anom = detect_anomalies(df, anomaly_method, z_threshold)
    title = (
        f"Détection des Anomalies (Z-score > {z_threshold})"
        if anomaly_method == "Z-score"
        else "Détection des Anomalies (IsolationForest)"
    )
    fig_anomaly = px.scatter(
        df_anom,
        x="Date",
        y="Poids (Kgs)",
        color="Anomalies",
        color_discrete_map={False: "blue", True: "red"},
        title=title,
    )
    fig_anomaly = apply_theme(fig_anomaly, theme)
    st.plotly_chart(fig_anomaly, use_container_width=True)
    st.write("Points de données considérés comme anomalies :")
    st.dataframe(df_anom[df_anom["Anomalies"]])


def render_download(df):
    st.header("Téléchargement des données")
    csv_data = convert_df_to_csv(df)
    st.download_button(
        label="Télécharger",
        data=csv_data,
        file_name="donnees_poids.csv",
        mime="text/csv",
    )


def render_correlation(df):
    st.header("Analyse de Corrélation")
    if df.empty:
        st.warning("Aucune donnée disponible pour la corrélation.")
        return

    if "Calories" not in df.columns:
        df["Calories"] = np.nan

    last_date = df["Date"].max()
    st.write(f"Dernier enregistrement le : {last_date.date()}")
    user_cal = st.number_input(
        f"Calories consommées pour le {last_date.date()}",
        min_value=0,
        value=st.session_state.get("calories", 2000),
        key="calories_input",
    )
    if st.button("Mettre à jour les calories"):
        df.loc[df["Date"] == last_date, "Calories"] = user_cal
        st.success(f"Calories du {last_date.date()} mises à jour !")

    corr_cols = st.multiselect(
        "Variables à corréler avec le Poids (Kgs) :",
        ["Calories", "IMC"],
        default=["Calories"],
    )
    theme = st.session_state.get("theme", "Default")
    if corr_cols:
        for col in corr_cols:
            valid_df = df.dropna(subset=[col, "Poids (Kgs)"])
            if len(valid_df) > 1:
                correlation = valid_df["Poids (Kgs)"].corr(valid_df[col])
                st.write(f"**Corrélation (Poids vs {col})** : {correlation:.2f}")
                fig_corr = px.scatter(
                    valid_df,
                    x=col,
                    y="Poids (Kgs)",
                    trendline="ols",
                    title=f"Poids vs {col}",
                )
                fig_corr = apply_theme(fig_corr, theme)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning(f"Pas assez de données pour {col}.")


def main():
    df = _get_data()
    if df is None:
        return

    expected_columns = {"Date", "Poids (Kgs)"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        _render_empty_state(
            "Colonnes manquantes pour afficher les analyses.",
            "Le fichier doit contenir les colonnes `Date` et `Poids (Kgs)` (séparateur point ou virgule).",
            show_retry=True,
        )
        return

    if df.empty:
        _render_empty_state(
            "Aucune donnée de poids disponible.",
            "Ajoutez des mesures de poids à la source ou chargez un CSV avec les colonnes attendues.",
            show_retry=True,
        )
        return

    st.write(f"**Nombre total de lignes chargées :** {df.shape[0]}")
    st.write("Aperçu des dernières lignes :", df.tail())

    render_summary(df)
    render_graphs(df)
    render_anomalies(df)
    render_correlation(df)
    render_download(df)


if __name__ == "__main__":
    main()
