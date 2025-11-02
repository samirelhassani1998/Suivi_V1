import streamlit as st

# IMPORTANT : appeler st.set_page_config en premier !
st.set_page_config(page_title="Suivi du Poids Amélioré", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from typing import Tuple

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

from pmdarima import auto_arima

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

# --- Injection de CSS personnalisé pour améliorer l'apparence globale ---
st.markdown(
    """
    <style>
    /* Style global */
    .reportview-container {
        font-family: 'Segoe UI', sans-serif;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf, #2e7bcf);
        color: white;
    }
    .stButton>button {
        background-color: #2e7bcf;
        color: white;
        border: none;
    }
    .stMetric {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#############
# TITRE ET DESCRIPTION
#############
st.title("Suivi de l'Évolution du Poids")
st.markdown(
    """
    Suivez votre poids, calculez votre IMC, visualisez des tendances, réalisez des prévisions et détectez des anomalies.
    Utilisez les onglets pour naviguer entre les fonctionnalités.
    """
)

#############
# FONCTIONS
#############
@st.cache_data(ttl=300)
def load_data(url: str) -> pd.DataFrame:
    """
    Charge et nettoie le jeu de données depuis un fichier CSV.
    """
    df = pd.read_csv(url, decimal=",")
    df['Poids (Kgs)'] = (
        df['Poids (Kgs)']
        .astype(str)
        .str.replace(',', '.')
        .str.strip()
    )
    df['Poids (Kgs)'] = pd.to_numeric(df['Poids (Kgs)'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Poids (Kgs)', 'Date']).sort_values('Date', ascending=True)
    return df

def apply_theme(fig: go.Figure, theme_name: str) -> go.Figure:
    """
    Applique un thème Plotly à une figure et ajuste la mise en page.
    """
    theme_templates = {
        "Dark": "plotly_dark",
        "Light": "plotly_white",
        "Solar": "plotly_solar",
        "Seaborn": "seaborn",
    }
    if theme_name in theme_templates:
        fig.update_layout(template=theme_templates[theme_name])
    # Améliorations supplémentaires : marges et titres
    fig.update_layout(
        title=dict(font=dict(size=22)),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    return fig

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convertit un DataFrame en CSV encodé en UTF-8.
    """
    return df.to_csv(index=False).encode('utf-8')

def detect_anomalies(df: pd.DataFrame, method: str, z_threshold: float = 2.0) -> pd.DataFrame:
    """Détecte les anomalies dans les poids selon la méthode choisie.

    Parameters
    ----------
    df : pd.DataFrame
        Jeu de données contenant la colonne ``Poids (Kgs)``.
    method : str
        "Z-score" ou "IsolationForest" pour sélectionner la technique.
    z_threshold : float, optional
        Seuil au-delà duquel un point est considéré comme anomalie pour la
        méthode Z-score.

    Returns
    -------
    pd.DataFrame
        DataFrame enrichi avec une colonne booléenne ``Anomalies``.
    """
    df = df.copy()
    if df.empty:
        df["Anomalies"] = []
        return df

    if method == "IsolationForest":
        model = IsolationForest(contamination=0.1, random_state=42)
        df["Anomalies"] = model.fit_predict(df[["Poids (Kgs)"]]) == -1
    else:
        df["Z_score"] = np.abs(stats.zscore(df["Poids (Kgs)"]))
        df["Anomalies"] = df["Z_score"] > z_threshold
    return df


@st.cache_data
def filter_data_by_period(
    data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Retourne les données filtrées entre deux dates incluses."""

    if data.empty:
        return data
    mask = (data["Date"] >= start) & (data["Date"] <= end)
    return data.loc[mask].copy()


def _fit_forecast_model(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray
):
    """Entraîne un modèle de prévision selon le nom choisi."""

    if model_name == "Régression Linéaire":
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        return model
    if model_name == "Auto-ARIMA":
        # Auto-ARIMA s'entraîne uniquement sur la série temporelle.
        return auto_arima(
            y_train,
            seasonal=True,
            m=7,
            suppress_warnings=True,
            error_action="ignore",
        )
    raise ValueError(f"Modèle non supporté: {model_name}")


@st.cache_resource
def get_cached_forecast_model(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray
):
    """Renvoie un modèle de prévision entraîné et mis en cache."""

    return _fit_forecast_model(model_name, X_train, y_train)


def compute_bootstrap_interval(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    future_X: np.ndarray,
    n_bootstraps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule un intervalle de confiance via bootstrap."""

    if future_X.size == 0:
        return np.array([]), np.array([])

    preds = np.zeros((n_bootstraps, future_X.shape[0]))
    rng = np.random.default_rng(42)
    for i in range(n_bootstraps):
        indices = rng.integers(0, len(X_train), len(X_train))
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        model = _fit_forecast_model(model_name, X_boot, y_boot)
        preds[i] = model.predict(future_X)

    return (
        np.percentile(preds, 2.5, axis=0),
        np.percentile(preds, 97.5, axis=0),
    )

#############
# CHARGEMENT DES DONNÉES
#############
url = 'https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv'
df = load_data(url)

# Bouton pour recharger les données
if st.button("Recharger les données"):
    load_data.clear()
    df = load_data(url)

st.write(f"**Nombre total de lignes chargées :** {df.shape[0]}")
st.write("Aperçu des dernières lignes :", df.tail())

#############
# SIDEBAR AVANCÉE
#############
st.sidebar.header("Paramètres Généraux")

# Thème et type de moyenne mobile dans des expanders
with st.sidebar.expander("Personnalisation du Thème"):
    theme = st.selectbox("Choisir un thème", ["Default", "Dark", "Light", "Solar", "Seaborn"])

with st.sidebar.expander("Paramètres de la Moyenne Mobile"):
    ma_type = st.selectbox("Type de moyenne mobile", ["Simple", "Exponentielle"])
    window_size = st.slider("Taille de la moyenne mobile (jours)", 1, 30, 7)

# Filtre de dates
with st.sidebar.expander("Filtre de Dates"):
    if not df.empty:
        date_min, date_max = df['Date'].min(), df['Date'].max()
        date_range = st.date_input("Sélectionnez une plage de dates", [date_min, date_max])
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Objectifs de poids et informations personnelles
with st.sidebar.expander("Objectifs et Infos Personnelles"):
    target_weight = st.number_input("Objectif 1 (Kgs)", value=95.0)
    target_weight_2 = st.number_input("Objectif 2 (Kgs)", value=90.0)
    target_weight_3 = st.number_input("Objectif 3 (Kgs)", value=85.0)
    target_weight_4 = st.number_input("Objectif 4 (Kgs)", value=80.0)
    st.markdown("---")
    height_cm = st.number_input("Votre taille (cm)", value=182)
    height_m = height_cm / 100.0

# Paramètres anomalies et calories/activité
with st.sidebar.expander("Paramètres d'Anomalies & Activité"):
    anomaly_method = st.selectbox("Méthode de détection", ["Z-score", "IsolationForest"])
    z_score_threshold = st.slider("Seuil Z-score", 1.0, 5.0, 2.0, step=0.5)
    st.markdown("**Santé et Activité**")
    calories = st.number_input("Calories consommées aujourd'hui", min_value=0, value=2000)
    calories_brul = st.number_input("Calories brûlées (approximatif)", min_value=0, value=500)
    st.write("Bilan calorique estimé :", calories - calories_brul, "kcal")

#############
# CRÉATION DES ONGLETS
#############
tabs = st.tabs([
    "Résumé", "Graphiques", "Prévisions", "Analyse des Données",
    "Comparaison des Modèles", "Corrélation", "Personnalisation",
    "Téléchargement", "Perte de Poids Hebdo", "Conseils", "Analyses IA/ML"
])

#################################
# 1. Onglet: RÉSUMÉ
#################################
with tabs[0]:
    st.header("Résumé")
    if df.empty:
        st.warning("Aucune donnée disponible.")
    else:
        initial_weight = df['Poids (Kgs)'].iloc[0]
        current_weight = df['Poids (Kgs)'].iloc[-1]
        weight_lost = initial_weight - current_weight
        total_weight_to_lose = initial_weight - target_weight_4 if initial_weight > target_weight_4 else 1
        progress_percent = (weight_lost / total_weight_to_lose) * 100
        current_bmi = current_weight / (height_m ** 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Poids Actuel", f"{current_weight:.2f} Kgs")
        col2.metric("Objectif Final", f"{target_weight_4:.2f} Kgs")
        col3.metric("Progrès (%)", f"{progress_percent:.2f} %")
        col4.metric("IMC Actuel", f"{current_bmi:.2f}")

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

#################################
# 2. Onglet: GRAPHIQUES
#################################
with tabs[1]:
    st.header("Graphiques")
    if df.empty:
        st.warning("Pas de données à afficher.")
    else:
        # Calcul de la moyenne mobile
        if ma_type == "Simple":
            df["Poids_MA"] = df["Poids (Kgs)"].rolling(window=window_size, min_periods=1).mean()
        else:
            df["Poids_MA"] = df["Poids (Kgs)"].ewm(span=window_size, adjust=False).mean()

        fig = px.line(
            df, x="Date", y="Poids (Kgs)", markers=True,
            title="Évolution du Poids dans le Temps",
            labels={"Poids (Kgs)": "Poids (en Kgs)"}
        )
        # Ligne moyenne globale
        mean_weight = df["Poids (Kgs)"].mean()
        fig.add_hline(y=mean_weight, line_dash="dot",
                      annotation_text="Moyenne Globale", annotation_position="bottom right")
        # Courbe de moyenne mobile
        fig.add_scatter(x=df["Date"], y=df["Poids_MA"], mode="lines",
                        name=f"Moyenne Mobile ({ma_type}) {window_size} jours")
        # Lignes d'objectifs
        fig.add_hline(y=target_weight, line_dash="dash",
                      annotation_text="Objectif 1", annotation_position="bottom right")
        fig.add_hline(y=target_weight_2, line_dash="dash", line_color="red",
                      annotation_text="Objectif 2", annotation_position="bottom right")
        fig.add_hline(y=target_weight_3, line_dash="dash", line_color="green",
                      annotation_text="Objectif 3", annotation_position="bottom right")
        fig.add_hline(y=target_weight_4, line_dash="dash", line_color="purple",
                      annotation_text="Objectif 4", annotation_position="bottom right")

        fig = apply_theme(fig, theme)
        st.plotly_chart(fig, use_container_width=True)

        # Histogramme de distribution du poids
        fig_hist = px.histogram(df, x="Poids (Kgs)", nbins=30,
                                title="Distribution des Poids")
        fig_hist = apply_theme(fig_hist, theme)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Évolution et distribution de l'IMC
        df["IMC"] = df["Poids (Kgs)"] / (height_m ** 2)
        fig_bmi = px.line(
            df, x="Date", y="IMC", markers=True,
            title="Évolution de l'IMC", labels={"IMC": "Indice de Masse Corporelle"}
        )
        fig_bmi = apply_theme(fig_bmi, theme)
        st.plotly_chart(fig_bmi, use_container_width=True)

        fig_bmi_hist = px.histogram(df, x="IMC", nbins=30,
                                    title="Distribution de l'IMC")
        fig_bmi_hist = apply_theme(fig_bmi_hist, theme)
        st.plotly_chart(fig_bmi_hist, use_container_width=True)

        # Détection d'anomalies
        df_anom = detect_anomalies(df, anomaly_method, z_score_threshold)
        title = (
            f"Détection des Anomalies (Z-score > {z_score_threshold})"
            if anomaly_method == "Z-score"
            else "Détection des Anomalies (IsolationForest)"
        )
        fig_anomaly = px.scatter(
            df_anom, x="Date", y="Poids (Kgs)",
            color="Anomalies",
            color_discrete_map={False: 'blue', True: 'red'},
            title=title
        )
        fig_anomaly = apply_theme(fig_anomaly, theme)
        st.plotly_chart(fig_anomaly, use_container_width=True)

        st.write("Points de données considérés comme anomalies :")
        st.dataframe(df_anom[df_anom["Anomalies"]])

#################################
# 3. Onglet: PRÉVISIONS
#################################
with tabs[2]:
    st.header("Prévisions")
    if df.empty or len(df) < 2:
        st.warning("Données insuffisantes pour faire des prévisions.")
    else:
        df["Date_numeric"] = (df["Date"] - df["Date"].min()) / np.timedelta64(1, "D")

        st.subheader("Paramètres de projection")
        col_left, col_right = st.columns(2)
        with col_left:
            default_period = (
                df["Date"].min().date(),
                df["Date"].max().date(),
            )
            training_period = st.date_input(
                "Période d'entraînement",
                value=default_period,
                min_value=df["Date"].min().date(),
                max_value=df["Date"].max().date(),
            )
        with col_right:
            model_name = st.selectbox(
                "Modèle de prévision",
                ["Régression Linéaire", "Random Forest", "Auto-ARIMA"],
            )

        if not isinstance(training_period, (list, tuple)) or len(training_period) != 2:
            st.warning("Veuillez sélectionner une plage de dates valide pour l'entraînement.")
        else:
            start_train = pd.to_datetime(training_period[0])
            end_train = pd.to_datetime(training_period[1])
            train_df = filter_data_by_period(df, start_train, end_train)
            train_df = train_df.sort_values("Date").reset_index(drop=True)

            if train_df.empty or len(train_df) < 2:
                st.warning("La période d'entraînement sélectionnée ne contient pas assez de données.")
            else:
                projection_start_default = (
                    train_df["Date"].max() + pd.Timedelta(days=1)
                ).date()
                col_params1, col_params2 = st.columns(2)
                with col_params1:
                    projection_start = st.date_input(
                        "Date de début de projection",
                        value=projection_start_default,
                        min_value=train_df["Date"].max().date(),
                    )
                with col_params2:
                    horizon_days = st.slider(
                        "Horizon (jours)",
                        min_value=7,
                        max_value=180,
                        value=30,
                    )

                projection_start_ts = pd.to_datetime(projection_start)
                future_dates = pd.date_range(
                    start=projection_start_ts,
                    periods=horizon_days,
                    freq="D",
                )

                train_df["Date_numeric_model"] = (
                    (train_df["Date"] - train_df["Date"].min())
                    / np.timedelta64(1, "D")
                )
                X_train = train_df[["Date_numeric_model"]].values
                y_train = train_df["Poids (Kgs)"].values

                try:
                    model = get_cached_forecast_model(model_name, X_train, y_train)
                except Exception as exc:
                    st.error(f"Erreur lors de l'entraînement du modèle : {exc}")
                    model = None

                if model is not None:
                    if model_name == "Auto-ARIMA":
                        in_sample_pred = model.predict_in_sample()
                        forecast_values, conf_int = model.predict(
                            n_periods=horizon_days, return_conf_int=True
                        )
                        lower_ci = conf_int[:, 0]
                        upper_ci = conf_int[:, 1]
                    else:
                        in_sample_pred = model.predict(X_train)
                        future_numeric = (
                            (future_dates - train_df["Date"].min())
                            / np.timedelta64(1, "D")
                        ).values.reshape(-1, 1)
                        forecast_values = model.predict(future_numeric)
                        lower_ci, upper_ci = compute_bootstrap_interval(
                            model_name, X_train, y_train, future_numeric
                        )

                    forecast_values = np.asarray(forecast_values)
                    lower_ci = np.asarray(lower_ci)
                    upper_ci = np.asarray(upper_ci)

                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=train_df["Date"],
                            y=train_df["Poids (Kgs)"],
                            mode="markers+lines",
                            name="Historique",
                        )
                    )
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=train_df["Date"],
                            y=in_sample_pred,
                            mode="lines",
                            name="Ajustement du modèle",
                            line=dict(dash="dot"),
                        )
                    )
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=forecast_values,
                            mode="lines+markers",
                            name="Prévision",
                            line=dict(color="#1E88E5", width=3),
                        )
                    )

                    if lower_ci.size and upper_ci.size:
                        ci_x = list(future_dates) + list(future_dates[::-1])
                        ci_y = list(upper_ci) + list(lower_ci[::-1])
                        fig_forecast.add_trace(
                            go.Scatter(
                                x=ci_x,
                                y=ci_y,
                                fill="toself",
                                fillcolor="rgba(30, 136, 229, 0.15)",
                                line=dict(color="rgba(30, 136, 229, 0.1)"),
                                hoverinfo="skip",
                                name="IC 95%",
                                showlegend=True,
                            )
                        )

                    anomalies_df = detect_anomalies(
                        train_df[["Date", "Poids (Kgs)"]].copy(),
                        anomaly_method,
                        z_score_threshold,
                    )
                    anomalies_points = anomalies_df[anomalies_df["Anomalies"]]
                    if not anomalies_points.empty:
                        fig_forecast.add_trace(
                            go.Scatter(
                                x=anomalies_points["Date"],
                                y=anomalies_points["Poids (Kgs)"],
                                mode="markers",
                                marker=dict(color="red", size=10, symbol="x"),
                                name="Anomalies",
                            )
                        )
                        for _, row in anomalies_points.head(3).iterrows():
                            fig_forecast.add_annotation(
                                x=row["Date"],
                                y=row["Poids (Kgs)"],
                                text=f"Anomalie ({row['Poids (Kgs)']:.1f} kg)",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="red",
                                ay=-40,
                            )

                    if forecast_values.size:
                        if target_weight_4 and np.any(forecast_values <= target_weight_4):
                            target_idx = int(np.where(forecast_values <= target_weight_4)[0][0])
                            fig_forecast.add_annotation(
                                x=future_dates[target_idx],
                                y=forecast_values[target_idx],
                                text=f"Objectif {target_weight_4:.1f} kg atteint",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="green",
                                ay=-60,
                                bgcolor="rgba(76, 175, 80, 0.15)",
                            )

                        if forecast_values.size > 1:
                            diffs = np.diff(forecast_values)
                            jump_idx = int(np.argmax(np.abs(diffs)))
                            if np.abs(diffs[jump_idx]) > 0.3:
                                fig_forecast.add_annotation(
                                    x=future_dates[jump_idx + 1],
                                    y=forecast_values[jump_idx + 1],
                                    text="Variation marquée",
                                    showarrow=True,
                                    arrowcolor="#FB8C00",
                                    ay=-50,
                                    bgcolor="rgba(251, 140, 0, 0.15)",
                                )

                    fig_forecast.update_layout(
                        title="Prévisions personnalisées",
                        yaxis_title="Poids (Kgs)",
                        xaxis_title="Date",
                    )
                    fig_forecast = apply_theme(fig_forecast, theme)
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    latest_forecast = forecast_values[-1] if forecast_values.size else np.nan
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    col_metrics1.metric(
                        "Poids prévu en fin d'horizon",
                        f"{latest_forecast:.2f} Kgs" if not np.isnan(latest_forecast) else "-",
                    )
                    delta_value = (
                        latest_forecast - y_train[-1]
                        if forecast_values.size and y_train.size
                        else np.nan
                    )
                    col_metrics2.metric(
                        "Variation prévue",
                        f"{delta_value:+.2f} Kgs" if not np.isnan(delta_value) else "-",
                    )
                    if lower_ci.size and upper_ci.size:
                        col_metrics3.metric(
                            "IC 95% (fin d'horizon)",
                            f"[{lower_ci[-1]:.2f}; {upper_ci[-1]:.2f}] Kgs",
                        )
                    else:
                        col_metrics3.metric("IC 95% (fin d'horizon)", "Non disponible")

                    forecast_df = pd.DataFrame({
                        "Date": future_dates,
                        "Prévision (Kgs)": forecast_values,
                    })
                    if lower_ci.size and upper_ci.size:
                        forecast_df["IC Inférieur (Kgs)"] = lower_ci
                        forecast_df["IC Supérieur (Kgs)"] = upper_ci

                    st.write("Tableau des projections")
                    st.dataframe(forecast_df)

                    mean_change_rate = train_df["Poids (Kgs)"].diff().mean()
                    st.write(
                        f"Taux de changement moyen (ensemble d'entraînement) : **{mean_change_rate:.2f} Kgs/jour**"
                    )
                    coef_var = train_df["Poids (Kgs)"].std() / train_df["Poids (Kgs)"].mean()
                    st.write(
                        f"Coefficient de variation (ensemble d'entraînement) : **{coef_var * 100:.2f}%**"
                    )

#################################
# 4. Onglet: ANALYSE DES DONNÉES
#################################
with tabs[3]:
    st.header("Analyse des Données")
    if df.empty:
        st.warning("Aucune donnée pour l’analyse.")
    else:
        st.subheader("Analyse STL (Tendance et Saisonnalité)")
        stl = STL(df.set_index('Date')['Poids (Kgs)'], period=7)
        res = stl.fit()
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
        sarima_model = SARIMAX(df["Poids (Kgs)"], order=(1,1,1), seasonal_order=(1,1,1,7))
        sarima_results = sarima_model.fit(disp=False)
        df["SARIMA_Predictions"] = sarima_results.predict(start=0, end=len(df)-1, dynamic=False)
        fig_sarima = px.scatter(df, x="Date", y="Poids (Kgs)", title="SARIMA")
        fig_sarima.add_trace(go.Scatter(x=df["Date"], y=df["SARIMA_Predictions"],
                                        mode='lines', name='Prévisions SARIMA'))
        fig_sarima = apply_theme(fig_sarima, theme)
        st.plotly_chart(fig_sarima, use_container_width=True)

        st.subheader("Variabilité et Corrélations Temporelles")
        std_window = st.slider("Fenêtre pour l'écart type mobile", 3, 30, 7, key="std_window")
        df["Rolling_STD"] = df["Poids (Kgs)"].rolling(window=std_window).std()
        fig_std = px.line(df, x="Date", y="Rolling_STD", title="Écart Type Mobile")
        fig_std = apply_theme(fig_std, theme)
        st.plotly_chart(fig_std, use_container_width=True)

        lag = st.slider("Nombre de décalages pour l'ACF/PACF", 1, 30, 7, key="acf_lag")
        acf_vals = acf(df["Poids (Kgs)"], nlags=lag, missing='drop')
        pacf_vals = pacf(df["Poids (Kgs)"], nlags=lag, method='ywm')
        fig_acf = px.bar(x=list(range(len(acf_vals))), y=acf_vals, title="Autocorrélation (ACF)")
        fig_pacf = px.bar(x=list(range(len(pacf_vals))), y=pacf_vals, title="Autocorrélation Partielle (PACF)")
        fig_acf = apply_theme(fig_acf, theme)
        fig_pacf = apply_theme(fig_pacf, theme)
        st.plotly_chart(fig_acf, use_container_width=True)
        st.plotly_chart(fig_pacf, use_container_width=True)

#################################
# 5. Onglet: COMPARAISON DES MODÈLES
#################################
with tabs[4]:
    st.header("Comparaison des Modèles")
    if df.empty or len(df) < 2:
        st.warning("Pas assez de données pour comparer les modèles.")
    else:
        X = df[["Date_numeric"]]
        y = df["Poids (Kgs)"]
        tscv = TimeSeriesSplit(n_splits=5)
        models = {
            "Régression Linéaire": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        model_scores = {}
        for name, model in models.items():
            scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=tscv)
            model_scores[name] = -scores.mean()
        scores_df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['MSE'])
        st.write("**Comparaison des MSE moyens :**")
        st.dataframe(scores_df)

        for name, model in models.items():
            model.fit(X, y)
            df[name + "_Predictions"] = model.predict(X)

        fig_compare = px.scatter(df, x="Date", y="Poids (Kgs)", title="Comparaison des Modèles")
        for name in models.keys():
            fig_compare.add_trace(go.Scatter(x=df["Date"], y=df[name + "_Predictions"],
                                             mode='lines', name=name))
        fig_compare = apply_theme(fig_compare, theme)
        st.plotly_chart(fig_compare, use_container_width=True)

        st.subheader("Métriques de Performance")
        for name in models.keys():
            mse = mean_squared_error(y, df[name + "_Predictions"])
            mae = mean_absolute_error(y, df[name + "_Predictions"])
            r2 = r2_score(y, df[name + "_Predictions"])
            st.write(f"**{name}:** MSE = {mse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")

#################################
# 6. Onglet: ANALYSE DE CORRÉLATION
#################################
with tabs[5]:
    st.header("Analyse de Corrélation")
    st.markdown(
    """
    Analysez la relation entre votre poids et d'autres variables (ex. calories, IMC).
    """
    )
    if "Calories" not in df.columns:
        df["Calories"] = np.nan

    if not df.empty:
        last_date = df["Date"].max()
        st.write(f"Dernier enregistrement le : {last_date.date()}")
        user_cal = st.number_input(f"Calories consommées pour le {last_date.date()}",
                                   min_value=0, value=2000)
        if st.button("Mettre à jour les calories"):
            df.loc[df["Date"] == last_date, "Calories"] = user_cal
            st.success(f"Calories du {last_date.date()} mises à jour !")

        corr_cols = st.multiselect("Variables à corréler avec le Poids (Kgs) :", 
                                   ["Calories", "IMC"], default=["Calories"])
        if corr_cols:
            for col in corr_cols:
                valid_df = df.dropna(subset=[col, "Poids (Kgs)"])
                if len(valid_df) > 1:
                    correlation = valid_df["Poids (Kgs)"].corr(valid_df[col])
                    st.write(f"**Corrélation (Poids vs {col})** : {correlation:.2f}")
                    fig_corr = px.scatter(valid_df, x=col, y="Poids (Kgs)",
                                          trendline="ols",
                                          title=f"Poids vs {col}")
                    fig_corr = apply_theme(fig_corr, theme)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning(f"Pas assez de données pour {col}.")

#################################
# 7. Onglet: PERSONNALISATION
#################################
with tabs[6]:
    st.header("Personnalisation et Indicateurs")
    if df.empty:
        st.warning("Aucune donnée disponible.")
    else:
        current_weight = df['Poids (Kgs)'].iloc[-1]
        initial_weight = df['Poids (Kgs)'].iloc[0]

        if current_weight <= target_weight_4:
            st.balloons()
            st.success("Félicitations ! Objectif ultime atteint.")
        elif current_weight <= target_weight_3:
            st.success("Bravo ! Vous avez atteint l'objectif 3.")
        elif current_weight <= target_weight_2:
            st.info("Bien joué ! Objectif 2 atteint.")
        elif current_weight <= target_weight:
            st.info("Bon travail ! Objectif 1 atteint.")
        else:
            st.warning("Continuez vos efforts pour atteindre vos objectifs.")

        # Indicateur de progression avec gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_weight,
            delta={'reference': target_weight_4},
            gauge={
                'axis': {'range': [target_weight_4 - 10, initial_weight + 10]},
                'steps': [
                    {'range': [target_weight_4, target_weight_3], 'color': "darkgreen"},
                    {'range': [target_weight_3, target_weight_2], 'color': "lightgreen"},
                    {'range': [target_weight_2, target_weight], 'color': "yellow"},
                    {'range': [target_weight, initial_weight], 'color': "orange"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': current_weight
                }
            }
        ))
        fig_gauge.update_layout(title="Progression vers l'Objectif Final")
        fig_gauge = apply_theme(fig_gauge, theme)
        st.plotly_chart(fig_gauge, use_container_width=True)

#################################
# 8. Onglet: TÉLÉCHARGEMENT & GESTION
#################################
with tabs[7]:
    st.header("Téléchargement et Gestion des Données")
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
    if uploaded_file:
        df_user = pd.read_csv(uploaded_file)
        st.write("Aperçu :", df_user.head())
        if st.button("Fusionner avec les données existantes"):
            df_user['Poids (Kgs)'] = (
                df_user['Poids (Kgs)']
                .astype(str)
                .str.replace(',', '.')
                .str.strip()
            )
            df_user['Poids (Kgs)'] = pd.to_numeric(df_user['Poids (Kgs)'], errors='coerce')
            df_user['Date'] = pd.to_datetime(df_user['Date'], dayfirst=True, errors='coerce')
            df_user = df_user.dropna(subset=['Poids (Kgs)', 'Date']).sort_values('Date', ascending=True)
            df = pd.concat([df, df_user]).drop_duplicates(subset=['Date']).sort_values('Date')
            st.success("Fusion réussie. Données mises à jour.")

    st.subheader("Télécharger vos données en CSV")
    csv_data = convert_df_to_csv(df)
    st.download_button(
        label="Télécharger",
        data=csv_data,
        file_name='donnees_poids.csv',
        mime='text/csv'
    )

    if st.button("Réinitialiser les données"):
        df = pd.DataFrame(columns=['Date', 'Poids (Kgs)'])
        st.success("Données réinitialisées. Rechargez ou importez un nouveau fichier.")

#################################
# 9. Onglet: PERTE DE POIDS HEBDO
#################################
with tabs[8]:
    st.header("Perte de Poids Hebdomadaire")
    if df.empty:
        st.warning("Aucune donnée pour calculer la perte hebdomadaire.")
    else:
        df_weekly = df.set_index('Date').resample('W').mean().reset_index()
        df_weekly['Perte_Poids'] = -df_weekly['Poids (Kgs)'].diff()
        df_weekly = df_weekly.dropna()
        st.write("Perte de poids par semaine :")
        st.dataframe(df_weekly[['Date', 'Poids (Kgs)', 'Perte_Poids']])
        fig_weekly = px.bar(
            df_weekly, x='Date', y='Perte_Poids',
            title='Perte de Poids Hebdomadaire',
            labels={"Perte_Poids": "Perte (Kgs)"}
        )
        fig_weekly = apply_theme(fig_weekly, theme)
        st.plotly_chart(fig_weekly, use_container_width=True)

#################################
# 10. Onglet: CONSEILS
#################################
with tabs[9]:
    st.header("Conseils et Recommandations")
    st.markdown(
    """
    **Recommandations générales :**  
    - Adoptez une alimentation équilibrée (plus de détails sur [le guide OMS](https://www.who.int/publications/m/item/healthy-diet-factsheet)).  
    - Pratiquez une activité physique régulière (cf. [recommandations OMS](https://www.who.int/news-room/fact-sheets/detail/physical-activity)).  
    - Surveillez régulièrement votre poids et votre IMC ([Calculateur d'IMC OMS](https://www.who.int/tools/body-mass-index-bmi)).  
    - En cas de doute, consultez un professionnel de santé.
    """
    )

#################################
# 11. Onglet: ANALYSES IA/ML
#################################
with tabs[10]:
    st.header("Analyses IA/ML")
    if df.empty or len(df) < 2:
        st.warning("Données insuffisantes pour l'analyse.")
    else:
        st.subheader("Détection d'anomalies (IsolationForest)")
        iso = IsolationForest(contamination=0.1, random_state=42)
        df['IF_Anomaly'] = iso.fit_predict(df[["Poids (Kgs)"]]) == -1
        fig_if = px.scatter(
            df, x="Date", y="Poids (Kgs)",
            color="IF_Anomaly",
            color_discrete_map={False: 'blue', True: 'red'},
            title="Anomalies détectées"
        )
        fig_if = apply_theme(fig_if, theme)
        st.plotly_chart(fig_if, use_container_width=True)
        st.write("Anomalies détectées :")
        st.dataframe(df[df["IF_Anomaly"]])

        st.subheader("Prévisions Auto-ARIMA")
        future_arima_days = st.slider(
            "Jours à prédire", 1, 60, 30, key="arima_days"
        )
        arima_model = auto_arima(
            df["Poids (Kgs)"], seasonal=True, m=7, suppress_warnings=True
        )
        arima_forecast = arima_model.predict(n_periods=future_arima_days)
        arima_dates = pd.date_range(
            start=df["Date"].max() + pd.Timedelta(days=1),
            periods=future_arima_days
        )
        fig_arima = px.line(
            x=arima_dates, y=arima_forecast,
            labels={"x": "Date", "y": "Prévision (Kgs)"},
            title="Prévisions Auto-ARIMA"
        )
        fig_arima.add_scatter(
            x=df["Date"], y=df["Poids (Kgs)"], mode='lines', name='Historique'
        )
        fig_arima = apply_theme(fig_arima, theme)
        st.plotly_chart(fig_arima, use_container_width=True)

        st.subheader("Clustering K-Means")
        clusters = st.slider(
            "Nombre de clusters", 2, 5, 3, key="kmeans_clusters"
        )
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[["Poids (Kgs)"]])
        fig_cluster = px.scatter(
            df, x="Date", y="Poids (Kgs)", color="Cluster",
            title="Clusters de poids (K-Means)"
        )
        fig_cluster = apply_theme(fig_cluster, theme)
        st.plotly_chart(fig_cluster, use_container_width=True)

#############
# SOURCES & RÉFÉRENCES
#############
st.markdown("---")
st.markdown("**Sources et Références :**")
st.markdown("- [Streamlit Documentation](https://docs.streamlit.io/)")
st.markdown("- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)")
st.markdown("- [Plotly Graph Objects](https://plotly.com/python/)")
st.markdown("- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)")
st.markdown("- [Statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)")
st.markdown("- [Pandas Documentation](https://pandas.pydata.org/docs/)")
st.markdown("- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)")
st.markdown("- [OMS - Obésité et Surpoids](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")
