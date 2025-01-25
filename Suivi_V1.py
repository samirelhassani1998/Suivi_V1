import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
from scipy import stats

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

#############
# PARAMÈTRES
#############
st.set_page_config(page_title="Suivi du Poids Amélioré", layout="wide")
st.title("Suivi de l'Évolution du Poids")
st.markdown("""
Cette application permet de suivre l'évolution de votre poids, de calculer votre IMC, 
d'identifier des tendances, de faire des prévisions et de détecter des anomalies. 
Utilisez les différents onglets pour naviguer entre les fonctionnalités.
""")

#############
# FONCTIONS
#############

@st.cache_data(ttl=300)
def load_data(url: str) -> pd.DataFrame:
    """
    Charge et nettoie le jeu de données depuis un fichier CSV.
    Paramètres :
        url: Lien vers le fichier CSV
    Retour :
        Un DataFrame nettoyé et trié
    """
    df = pd.read_csv(url, decimal=",")

    # Nettoyage de la colonne 'Poids (Kgs)'
    df['Poids (Kgs)'] = df['Poids (Kgs)'].astype(str).str.replace(',', '.').str.strip()
    df['Poids (Kgs)'] = pd.to_numeric(df['Poids (Kgs)'], errors='coerce')

    # Conversion de la colonne 'Date'
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Suppression des lignes où 'Poids (Kgs)' ou 'Date' est manquant
    df = df.dropna(subset=['Poids (Kgs)', 'Date']).sort_values('Date', ascending=True)
    return df

def apply_theme(fig, theme_name: str) -> go.Figure:
    """
    Applique un thème Plotly à une figure.
    Paramètres :
        fig : Figure plotly
        theme_name : Nom du thème choisi par l’utilisateur
    Retour :
        Figure mise à jour
    """
    if theme_name == "Dark":
        fig.update_layout(template="plotly_dark")
    elif theme_name == "Light":
        fig.update_layout(template="plotly_white")
    elif theme_name == "Solar":
        fig.update_layout(template="plotly_solar")
    elif theme_name == "Seaborn":
        fig.update_layout(template="seaborn")
    return fig

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convertit un DataFrame en CSV encodé en UTF-8.
    Paramètres :
        df : DataFrame
    Retour :
        Bytes CSV
    """
    return df.to_csv(index=False).encode('utf-8')

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
# SIDEBAR
#############
st.sidebar.header("Paramètres Généraux")

# Choix du thème
theme = st.sidebar.selectbox("Choisir un thème", ["Default", "Dark", "Light", "Solar", "Seaborn"])

# Choix du type de moyenne mobile
ma_type = st.sidebar.selectbox("Type de moyenne mobile", ["Simple", "Exponentielle"])

# Taille de la fenêtre pour la moyenne mobile
window_size = st.sidebar.slider("Taille de la moyenne mobile (jours)", 1, 30, 7)

# Filtre de dates
st.sidebar.subheader("Filtre de dates")
if not df.empty:
    date_min, date_max = df['Date'].min(), df['Date'].max()
    date_range = st.sidebar.date_input("Plage de dates", [date_min, date_max])
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Objectifs de poids
st.sidebar.subheader("Objectifs de poids")
target_weight = st.sidebar.number_input("Objectif 1 (Kgs)", value=95.0)
target_weight_2 = st.sidebar.number_input("Objectif 2 (Kgs)", value=90.0)
target_weight_3 = st.sidebar.number_input("Objectif 3 (Kgs)", value=85.0)
target_weight_4 = st.sidebar.number_input("Objectif 4 (Kgs)", value=80.0)

# Informations personnelles
st.sidebar.subheader("Informations Personnelles")
height_cm = st.sidebar.number_input("Votre taille (cm)", value=182)
height_m = height_cm / 100.0

# Paramètres anomalie
st.sidebar.subheader("Paramètres Anomalies")
z_score_threshold = st.sidebar.slider("Seuil Z-score", 1.0, 5.0, 2.0, step=0.5)

# Suivi des calories et activités
st.sidebar.subheader("Santé et Activité")
calories = st.sidebar.number_input("Calories consommées aujourd'hui", min_value=0, value=2000)
calories_brul = st.sidebar.number_input("Calories brûlées (approximatif)", min_value=0, value=500)
st.sidebar.write("Bilan calorique estimé :", calories - calories_brul, "kcal")

#############
# CRÉATION DES ONGLETS
#############
tabs = st.tabs([
    "Résumé", 
    "Graphiques", 
    "Prévisions", 
    "Analyse des Données", 
    "Comparaison des Modèles", 
    "Analyse de Corrélation",
    "Personnalisation", 
    "Téléchargement & Gestion", 
    "Perte de Poids Hebdo", 
    "Conseils"
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
        col1.metric("Poids actuel", f"{current_weight:.2f} Kgs")
        col2.metric("Objectif final", f"{target_weight_4:.2f} Kgs")
        col3.metric("Progrès (%)", f"{progress_percent:.2f} %")
        col4.metric("IMC actuel", f"{current_bmi:.2f}")

        with st.expander("Statistiques descriptives du poids"):
            st.write(df["Poids (Kgs)"].describe())

        st.subheader("Interprétation de l'IMC")
        if current_bmi < 18.5:
            st.info("Vous êtes en **sous-poids** (IMC < 18.5). [Source: OMS](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")
        elif 18.5 <= current_bmi < 25:
            st.success("Vous avez un **poids normal** (18.5 <= IMC < 25). [Source: OMS](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")
        elif 25 <= current_bmi < 30:
            st.warning("Vous êtes en **surpoids** (25 <= IMC < 30). [Source: OMS](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")
        else:
            st.error("Vous êtes en **obésité** (IMC >= 30). [Source: OMS](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")

#################################
# 2. Onglet: GRAPHIQUES
#################################
with tabs[1]:
    st.header("Graphiques")
    if df.empty:
        st.warning("Pas de données à afficher.")
    else:
        # Graphique d'évolution du poids
        fig = px.line(df, x="Date", y="Poids (Kgs)", markers=True, title="Évolution du poids dans le temps")

        # Moyenne globale
        mean_weight = df["Poids (Kgs)"].mean()
        fig.add_hline(y=mean_weight, line_dash="dot", 
                      annotation_text="Moyenne Globale", annotation_position="bottom right")

        # Ajout moyenne mobile
        if ma_type == "Simple":
            df["Poids_MA"] = df["Poids (Kgs)"].rolling(window=window_size, min_periods=1).mean()
        else:
            df["Poids_MA"] = df["Poids (Kgs)"].ewm(span=window_size, adjust=False).mean()

        fig.add_scatter(x=df["Date"], y=df["Poids_MA"], mode="lines", 
                        name=f"Moyenne mobile ({ma_type}) {window_size} jours")

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
        st.plotly_chart(fig)

        # Histogramme du poids
        fig2 = px.histogram(df, x="Poids (Kgs)", nbins=30, title="Distribution des poids")
        fig2 = apply_theme(fig2, theme)
        st.plotly_chart(fig2)

        # Évolution de l'IMC
        df["IMC"] = df["Poids (Kgs)"] / (height_m ** 2)
        fig_bmi = px.line(df, x="Date", y="IMC", title="Évolution de l'IMC", markers=True)
        fig_bmi = apply_theme(fig_bmi, theme)
        st.plotly_chart(fig_bmi)

        # Histogramme de l'IMC
        fig_bmi_hist = px.histogram(df, x="IMC", nbins=30, title="Distribution de l'IMC")
        fig_bmi_hist = apply_theme(fig_bmi_hist, theme)
        st.plotly_chart(fig_bmi_hist)

        # Détection d'anomalies par Z-score
        df['Z_score'] = np.abs(stats.zscore(df['Poids (Kgs)']))
        df['Anomalies_Z'] = df['Z_score'] > z_score_threshold

        fig_anomaly = px.scatter(
            df, x="Date", y="Poids (Kgs)", 
            color="Anomalies_Z", 
            color_discrete_map={False: 'blue', True: 'red'},
            title=f"Détection des anomalies (Z-score > {z_score_threshold})"
        )
        fig_anomaly = apply_theme(fig_anomaly, theme)
        st.plotly_chart(fig_anomaly)

        st.write("Points de données inhabituels détectés :")
        st.dataframe(df[df["Anomalies_Z"]])

#################################
# 3. Onglet: PRÉVISIONS
#################################
with tabs[2]:
    st.header("Prévisions")
    if df.empty or len(df) < 2:
        st.warning("Données insuffisantes pour faire des prévisions.")
    else:
        # Création d'une variable numérique pour la régression
        df["Date_numeric"] = (df["Date"] - df["Date"].min()) / np.timedelta64(1, "D")
        X = df[["Date_numeric"]]
        y = df["Poids (Kgs)"]

        # Validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=5)
        lin_scores = cross_val_score(LinearRegression(), X, y, scoring='neg_mean_squared_error', cv=tscv)
        st.write(f"MSE moyen (Régression Linéaire) : **{-lin_scores.mean():.2f}**")

        # Modèle linéaire
        reg = LinearRegression().fit(X, y)
        predictions = reg.predict(X)

        # Intervalles de confiance par bootstrap
        n_bootstraps = 500
        boot_preds = np.zeros((n_bootstraps, len(X)))
        for i in range(n_bootstraps):
            X_boot, y_boot = resample(X, y)
            reg_boot = LinearRegression().fit(X_boot, y_boot)
            boot_preds[i] = reg_boot.predict(X)

        pred_lower = np.percentile(boot_preds, 2.5, axis=0)
        pred_upper = np.percentile(boot_preds, 97.5, axis=0)

        fig4 = px.scatter(df, x="Date", y="Poids (Kgs)", title="Régression Linéaire avec Intervalles de Confiance")
        fig4.add_trace(go.Scatter(x=df["Date"], y=predictions, 
                                  mode='lines', name='Prévisions', line=dict(color='blue')))
        fig4.add_trace(go.Scatter(x=df["Date"], y=pred_lower, fill=None, 
                                  mode='lines', line_color='lightblue', name='IC Inférieur'))
        fig4.add_trace(go.Scatter(x=df["Date"], y=pred_upper, fill='tonexty', 
                                  mode='lines', line_color='lightblue', name='IC Supérieur'))
        fig4 = apply_theme(fig4, theme)
        st.plotly_chart(fig4)

        # Date d'atteinte de l'objectif (ex : objectif 4)
        try:
            if reg.coef_[0] >= 0:
                st.error("Le modèle n’indique pas une perte de poids, impossible d’estimer la date d’atteinte de l’objectif.")
            else:
                days_to_target = (target_weight_4 - reg.intercept_) / reg.coef_[0]
                target_date = df["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
                if target_date < df["Date"].max():
                    st.warning("L'objectif de poids a déjà été atteint selon les prévisions.")
                else:
                    st.write(f"**Date estimée** pour atteindre l'objectif final ({target_weight_4} Kgs) : {target_date.date()}")
        except Exception as e:
            st.error(f"Erreur dans le calcul de la date estimée : {e}")

        # Prévisions futures
        st.subheader("Prédictions futures")
        future_days = st.slider("Nombre de jours à prédire", 1, 365, 30)
        future_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=future_days)
        future_dates_numeric = (future_dates - df["Date"].min()) / np.timedelta64(1, "D")
        future_predictions = reg.predict(future_dates_numeric.values.reshape(-1, 1))

        future_df = pd.DataFrame({"Date": future_dates, "Prévisions": future_predictions})
        fig_future = px.line(future_df, x="Date", y="Prévisions", title="Prévisions Futures de Poids")
        fig_future = apply_theme(fig_future, theme)
        st.plotly_chart(fig_future)

        # Taux de changement moyen
        df["Poids_diff"] = df["Poids (Kgs)"].diff()
        mean_change_rate = df["Poids_diff"].mean()
        st.write(f"Taux de changement moyen du poids : **{mean_change_rate:.2f} Kgs/jour**")

#################################
# 4. Onglet: ANALYSE DES DONNÉES
#################################
with tabs[3]:
    st.header("Analyse des Données")
    if df.empty:
        st.warning("Aucune donnée pour l’analyse.")
    else:
        st.subheader("Analyse de Tendance et Saisonnalité (STL)")
        stl = STL(df.set_index('Date')['Poids (Kgs)'], period=7)
        res = stl.fit()
        df["Trend"] = res.trend.values
        df["Seasonal"] = res.seasonal.values
        df["Resid"] = res.resid.values

        col1, col2, col3 = st.columns(3)
        with col1:
            fig_trend = px.line(df, x="Date", y="Trend", title="Tendance (Trend)")
            fig_trend = apply_theme(fig_trend, theme)
            st.plotly_chart(fig_trend)
        with col2:
            fig_seasonal = px.line(df, x="Date", y="Seasonal", title="Saisonnalité (Seasonal)")
            fig_seasonal = apply_theme(fig_seasonal, theme)
            st.plotly_chart(fig_seasonal)
        with col3:
            fig_resid = px.line(df, x="Date", y="Resid", title="Résidus (Resid)")
            fig_resid = apply_theme(fig_resid, theme)
            st.plotly_chart(fig_resid)

        st.subheader("Prédictions avec SARIMA")
        sarima_model = SARIMAX(df["Poids (Kgs)"], order=(1,1,1), seasonal_order=(1,1,1,7))
        sarima_results = sarima_model.fit(disp=False)
        df["SARIMA_Predictions"] = sarima_results.predict(start=0, end=len(df)-1, dynamic=False)

        fig_sarima = px.scatter(df, x="Date", y="Poids (Kgs)", title="Prédictions SARIMA")
        fig_sarima.add_trace(go.Scatter(x=df["Date"], y=df["SARIMA_Predictions"], 
                                        mode='lines', name='Prévisions SARIMA'))
        fig_sarima = apply_theme(fig_sarima, theme)
        st.plotly_chart(fig_sarima)

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
        st.write("Comparaison des MSE moyens :")
        st.dataframe(scores_df)

        # Entraînement final des modèles
        for name, model in models.items():
            model.fit(X, y)
            df[name + "_Predictions"] = model.predict(X)

        # Graphique de comparaison
        fig_compare = px.scatter(df, x="Date", y="Poids (Kgs)", title="Comparaison des Modèles")
        for name in models.keys():
            fig_compare.add_trace(go.Scatter(x=df["Date"], y=df[name + "_Predictions"], 
                                             mode='lines', name=name))
        fig_compare = apply_theme(fig_compare, theme)
        st.plotly_chart(fig_compare)

        st.subheader("Métriques de Performance (Ensemble complet)")
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
    st.markdown("""
    Ici, vous pouvez analyser la relation entre votre poids et d'autres variables 
    (par exemple : calories consommées, calories brûlées, etc.).
    """)

    if "Calories" not in df.columns:
        # On suppose qu'on veut intégrer la variable de calories suivie quotidiennement
        # S'il n'y a pas de données, on propose une colonne "Calories" en plus
        df["Calories"] = np.nan

    # Possibilité de saisir une valeur de calories pour la dernière date
    if not df.empty:
        last_date = df["Date"].max()
        st.write(f"Dernier enregistrement le : {last_date.date()}")
        
        user_cal = st.number_input(f"Calories consommées pour le {last_date.date()}", 
                                   min_value=0, value=2000)
        if st.button("Mettre à jour les calories pour la dernière date"):
            df.loc[df["Date"] == last_date, "Calories"] = user_cal
            st.success(f"Calories du {last_date.date()} mises à jour avec succès !")

        # Visualisation de la corrélation
        corr_cols = st.multiselect(
            "Choisissez les variables à corréler avec le Poids (Kgs) :", 
            ["Calories", "IMC"],
            default=["Calories"]
        )
        if corr_cols:
            for col in corr_cols:
                if col in df.columns:
                    valid_df = df.dropna(subset=[col, "Poids (Kgs)"])
                    if len(valid_df) > 1:
                        correlation = valid_df["Poids (Kgs)"].corr(valid_df[col])
                        st.write(f"**Corrélation entre Poids (Kgs) et {col}** : {correlation:.2f}")
                        fig_corr = px.scatter(valid_df, x=col, y="Poids (Kgs)", 
                                              trendline="ols",
                                              title=f"Corrélation Poids vs {col}")
                        fig_corr = apply_theme(fig_corr, theme)
                        st.plotly_chart(fig_corr)
                    else:
                        st.warning(f"Pas assez de données non-nulles pour {col}.")

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

        # Affichage des statuts d'objectifs
        if current_weight <= target_weight_4:
            st.balloons()
            st.success("Félicitations ! Vous avez atteint votre objectif le plus ambitieux !")
        elif current_weight <= target_weight_3:
            st.success("Bravo ! Vous avez atteint le troisième objectif de poids.")
        elif current_weight <= target_weight_2:
            st.info("Bien joué ! Vous avez atteint le deuxième objectif de poids.")
        elif current_weight <= target_weight:
            st.info("Bon travail ! Vous avez atteint le premier objectif de poids.")
        else:
            st.warning("Continuez vos efforts pour atteindre vos objectifs.")

        # Indicateur de progression
        fig_objectifs = go.Figure(go.Indicator(
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
        fig_objectifs.update_layout(title="Progression vers les Objectifs de Poids")
        fig_objectifs = apply_theme(fig_objectifs, theme)
        st.plotly_chart(fig_objectifs)

#################################
# 8. Onglet: TÉLÉCHARGEMENT & GESTION
#################################
with tabs[7]:
    st.header("Téléchargement et Gestion des Données")

    # Téléchargement d'un nouveau CSV
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
    if uploaded_file:
        df_user = pd.read_csv(uploaded_file)
        st.write("Aperçu des données téléchargées :", df_user.head())
        if st.button("Fusionner avec les données existantes"):
            df_user['Poids (Kgs)'] = df_user['Poids (Kgs)'].astype(str).str.replace(',', '.').str.strip()
            df_user['Poids (Kgs)'] = pd.to_numeric(df_user['Poids (Kgs)'], errors='coerce')
            df_user['Date'] = pd.to_datetime(df_user['Date'], dayfirst=True, errors='coerce')
            df_user = df_user.dropna(subset=['Poids (Kgs)', 'Date'])
            df_user = df_user.sort_values('Date', ascending=True)

            # Fusion
            df = pd.concat([df, df_user])
            df = df.drop_duplicates(subset=['Date']).sort_values('Date')
            st.success("Fusion réussie. Les données sont mises à jour.")

    # Téléchargement des données
    st.subheader("Téléchargez vos données au format CSV")
    csv_data = convert_df_to_csv(df)
    st.download_button(
        label="Télécharger les données",
        data=csv_data,
        file_name='donnees_poids.csv',
        mime='text/csv',
    )

    # Réinitialiser les données
    if st.button("Réinitialiser toutes les données"):
        df = pd.DataFrame(columns=['Date', 'Poids (Kgs)'])
        st.success("Données réinitialisées. Veuillez recharger ou importer un nouveau fichier.")

#################################
# 9. Onglet: PERTE DE POIDS HEBDO
#################################
with tabs[8]:
    st.header("Perte de Poids Hebdomadaire")
    if df.empty:
        st.warning("Aucune donnée pour calculer la perte de poids hebdomadaire.")
    else:
        df_weekly = df.set_index('Date').resample('W').mean().reset_index()
        df_weekly['Perte_Poids'] = df_weekly['Poids (Kgs)'].diff() * -1
        df_weekly = df_weekly.dropna()

        st.write("Perte de poids par semaine :")
        st.dataframe(df_weekly[['Date', 'Poids (Kgs)', 'Perte_Poids']])

        fig_weekly = px.bar(df_weekly, x='Date', y='Perte_Poids', title='Perte de Poids Hebdomadaire')
        fig_weekly.update_layout(yaxis_title="Perte (Kgs)")
        fig_weekly = apply_theme(fig_weekly, theme)
        st.plotly_chart(fig_weekly)

#################################
# 10. Onglet: CONSEILS
#################################
with tabs[9]:
    st.header("Conseils et Recommandations")
    st.markdown("""
    Ces recommandations sont de nature générale et ne remplacent pas un avis médical.
    - Maintenez une alimentation équilibrée, riche en fruits, légumes et protéines maigres.  
      [Guide de l'Alimentation Saine (OMS)](https://www.who.int/publications/m/item/healthy-diet-factsheet)
    - Pratiquez une activité physique régulière : au moins 30 minutes par jour.  
      [Recommandations OMS](https://www.who.int/news-room/fact-sheets/detail/physical-activity)
    - Surveillez régulièrement votre poids et votre IMC.  
      [Calculateur d'IMC - OMS](https://www.who.int/tools/body-mass-index-bmi)
    - Consultez un professionnel de santé si vous avez des préoccupations particulières.
    """)

#############
# SOURCES
#############
st.write("---")
st.markdown("**Sources et Références :**")
st.markdown("- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)")
st.markdown("- [Plotly Graph Objects](https://plotly.com/python/)")
st.markdown("- [Streamlit Documentation](https://docs.streamlit.io/)")
st.markdown("- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)")
st.markdown("- [Statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)")
st.markdown("- [Pandas Documentation](https://pandas.pydata.org/docs/)")
st.markdown("- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)")
st.markdown("- [OMS - Obésité et Surpoids](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")
