import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# Titre de l'application Streamlit
st.set_page_config(page_title="Suivi du poids", layout="wide")
st.title("Suivi de l'évolution du poids")

# Fonction pour charger et traiter les données
@st.cache_data(ttl=300)  # Expiration de cache toutes les 5 minutes
def load_data(url):
    df = pd.read_csv(url, decimal=",")

    # Nettoyage de la colonne 'Poids (Kgs)'
    df['Poids (Kgs)'] = df['Poids (Kgs)'].astype(str).str.replace(',', '.').str.strip()
    df['Poids (Kgs)'] = pd.to_numeric(df['Poids (Kgs)'], errors='coerce')

    # Conversion de la colonne 'Date'
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Supprimer uniquement les lignes où 'Poids (Kgs)' ou 'Date' est manquant
    df = df.dropna(subset=['Poids (Kgs)', 'Date'])

    df = df.sort_values('Date', ascending=True)
    return df

# URL du fichier CSV
url = 'https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv'
df = load_data(url)

# Ajouter un bouton pour recharger les données
if st.button("Recharger les données"):
    load_data.clear()  # Efface le cache
    df = load_data(url)  # Recharge les données

# Vérification des données chargées
st.write(f"Nombre total de lignes chargées : {df.shape[0]}")
st.write(df.tail())  # Afficher les dernières lignes pour vérifier que toutes les données sont là

# Interface utilisateur améliorée
st.sidebar.header("Paramètres")

# Thème
theme = st.sidebar.selectbox("Choisir un thème", ["Default", "Dark", "Light"])

# Taille de la moyenne mobile
window_size = st.sidebar.slider("Taille de la moyenne mobile (jours)", 1, 30, 7)

# Filtre de dates
st.sidebar.header("Filtre de dates")
date_min = df['Date'].min()
date_max = df['Date'].max()
date_range = st.sidebar.date_input("Sélectionnez une plage de dates", [date_min, date_max])

# Appliquer le filtre de dates
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Objectifs de poids
st.sidebar.header("Objectifs de poids")
target_weight = st.sidebar.number_input("Objectif de poids 1 (Kgs)", value=95.0)
target_weight_2 = st.sidebar.number_input("Objectif de poids 2 (Kgs)", value=90.0)
target_weight_3 = st.sidebar.number_input("Objectif de poids 3 (Kgs)", value=85.0)

# Ajout de la taille pour le calcul de l'IMC
st.sidebar.header("Informations personnelles")
height_cm = st.sidebar.number_input("Votre taille (cm)", value=170)

# Fonction pour appliquer le thème
def apply_theme(fig):
    if theme == "Dark":
        fig.update_layout(template="plotly_dark")
    elif theme == "Light":
        fig.update_layout(template="plotly_white")
    return fig

# Diviser l'application en onglets
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Résumé", "Graphiques", "Prévisions", "Analyse des Données", "Comparaison des Modèles", "Personnalisation", "Téléchargement"])

with tab1:
    st.header("Résumé")
    st.write("Voici un résumé de vos progrès vers les objectifs de poids.")

    if df.empty:
        st.warning("Aucune donnée disponible.")
    else:
        # Calcul du progrès réel
        initial_weight = df['Poids (Kgs)'].iloc[0]
        current_weight = df['Poids (Kgs)'].iloc[-1]
        weight_lost = initial_weight - current_weight
        total_weight_to_lose = initial_weight - target_weight_3
        progress_percent = (weight_lost / total_weight_to_lose) * 100 if total_weight_to_lose != 0 else 0

        # Calcul de l'IMC actuel
        height_m = height_cm / 100
        current_bmi = current_weight / (height_m ** 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Poids actuel", f"{current_weight:.2f} Kgs")
        col2.metric("Objectif de poids", f"{target_weight_3} Kgs")
        col3.metric("Progrès vers l'objectif", f"{progress_percent:.2f} %")
        col4.metric("IMC actuel", f"{current_bmi:.2f}")

        st.write("Statistiques des poids :")
        st.write(df["Poids (Kgs)"].describe())

        # Ajout d'un indicateur de classification de l'IMC
        st.subheader("Interprétation de l'IMC")
        if current_bmi < 18.5:
            st.info("Vous êtes en sous-poids.")
        elif 18.5 <= current_bmi < 25:
            st.success("Vous avez un poids normal.")
        elif 25 <= current_bmi < 30:
            st.warning("Vous êtes en surpoids.")
        else:
            st.error("Vous êtes en obésité.")

with tab2:
    st.header("Graphiques")
    # Graphique de l'évolution du poids avec la moyenne globale
    fig = px.line(df, x="Date", y="Poids (Kgs)", markers=True)

    # Calcul de la moyenne globale des poids
    mean_weight = df["Poids (Kgs)"].mean()

    # Ajout de la moyenne globale au graphique
    fig.add_hline(y=mean_weight, line_dash="dot", annotation_text="Moyenne Globale", annotation_position="bottom right")

    # Calcul de la moyenne mobile
    df["Poids_rolling_mean"] = df["Poids (Kgs)"].rolling(window=window_size, min_periods=1).mean()

    # Ajout de la courbe de la moyenne mobile au graphique
    fig.add_scatter(x=df["Date"], y=df["Poids_rolling_mean"], mode="lines", name=f"Moyenne mobile {window_size} jours")

    fig.update_layout(title="Évolution du poids")

    # Ajout des lignes des objectifs de poids
    fig.add_hline(y=target_weight, line_dash="dash", annotation_text="Objectif 1", annotation_position="bottom right")
    fig.add_hline(y=target_weight_2, line_dash="dash", line_color="red", annotation_text="Objectif 2", annotation_position="bottom right")
    fig.add_hline(y=target_weight_3, line_dash="dash", line_color="green", annotation_text="Objectif 3", annotation_position="bottom right")

    # Appliquer le thème
    fig = apply_theme(fig)
    st.plotly_chart(fig)

    # Histogramme de la distribution des poids
    fig2 = px.histogram(df, x="Poids (Kgs)", nbins=30, title="Distribution des poids")
    fig2 = apply_theme(fig2)
    st.plotly_chart(fig2)

    # Graphique de l'évolution de l'IMC
    df["IMC"] = df["Poids (Kgs)"] / (height_m ** 2)
    fig_bmi = px.line(df, x="Date", y="IMC", title="Évolution de l'IMC", markers=True)
    fig_bmi = apply_theme(fig_bmi)
    st.plotly_chart(fig_bmi)

    # Graphique des anomalies détectées
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomalies'] = iso_forest.fit_predict(df[['Poids (Kgs)']])
    fig3 = px.scatter(df, x="Date", y="Poids (Kgs)", color="Anomalies", color_discrete_sequence=["blue", "red"])
    fig3.update_layout(title="Évolution du poids avec détection des anomalies")
    fig3 = apply_theme(fig3)
    st.plotly_chart(fig3)
    st.write("Points de données inhabituels détectés :")
    st.write(df[df["Anomalies"] == -1])

with tab3:
    st.header("Prévisions")
    # Prédiction de la date d'atteinte de l'objectif de poids avec régression linéaire
    df["Date_numeric"] = (df["Date"] - df["Date"].min()) / np.timedelta64(1, "D")
    X = df[["Date_numeric"]]
    y = df["Poids (Kgs)"]

    # Utilisation de TimeSeriesSplit pour la validation croisée
    tscv = TimeSeriesSplit(n_splits=5)
    lin_scores = cross_val_score(LinearRegression(), X, y, scoring='neg_mean_squared_error', cv=tscv)
    st.write(f"Score MSE moyen pour la régression linéaire avec TimeSeriesSplit : {-lin_scores.mean():.2f}")

    # Entraînement du modèle sur l'ensemble des données
    reg = LinearRegression().fit(X, y)
    predictions = reg.predict(X)

    fig4 = px.scatter(df, x="Date", y="Poids (Kgs)", title="Régression linéaire de l'évolution du poids")
    fig4.add_trace(go.Scatter(x=df["Date"], y=predictions, mode='lines', name='Prévisions'))
    fig4 = apply_theme(fig4)
    st.plotly_chart(fig4)

    # Calculer correctement la date d'atteinte de l'objectif de poids
    try:
        if reg.coef_[0] == 0:
            st.error("Impossible de prédire la date d'atteinte de l'objectif car le coefficient de régression est nul.")
        else:
            days_to_target = (target_weight_3 - reg.intercept_) / reg.coef_[0]
            target_date = df["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
            st.write(f"Date estimée pour atteindre l'objectif de poids : {target_date.date()}")
    except Exception as e:
        st.error(f"Erreur dans le calcul de la date estimée : {e}")

    # Prédictions futures
    st.subheader("Prédictions futures")
    future_days = st.slider("Nombre de jours à prédire", 1, 365, 30)
    future_dates = pd.date_range(start=df["Date"].max(), periods=future_days)
    future_dates_numeric = (future_dates - df["Date"].min()) / np.timedelta64(1, "D")
    future_predictions = reg.predict(future_dates_numeric.values.reshape(-1, 1))

    future_df = pd.DataFrame({"Date": future_dates, "Prévisions": future_predictions})
    fig_future = px.line(future_df, x="Date", y="Prévisions", title="Prévisions futures de poids")
    fig_future = apply_theme(fig_future)
    st.plotly_chart(fig_future)

    # Calculer le taux de changement moyen du poids
    df["Poids_diff"] = df["Poids (Kgs)"].diff()
    mean_change_rate = df["Poids_diff"].mean()
    st.write(f"Taux de changement moyen du poids : {mean_change_rate:.2f} Kgs par jour")

with tab4:
    st.header("Analyse des Données")
    st.subheader("Analyse de tendance saisonnière")
    stl = STL(df.set_index('Date')['Poids (Kgs)'], period=7)
    res = stl.fit()
    df["Trend"] = res.trend.values
    df["Seasonal"] = res.seasonal.values
    df["Resid"] = res.resid.values

    # Graphique de la tendance
    fig_trend = px.line(df, x="Date", y="Trend", title="Tendance de l'évolution du poids")
    fig_trend = apply_theme(fig_trend)
    st.plotly_chart(fig_trend)

    # Graphique de la saisonnalité
    fig_seasonal = px.line(df, x="Date", y="Seasonal", title="Saisonnalité de l'évolution du poids")
    fig_seasonal = apply_theme(fig_seasonal)
    st.plotly_chart(fig_seasonal)

    # Graphique des résidus
    fig_resid = px.line(df, x="Date", y="Resid", title="Résidus de l'évolution du poids")
    fig_resid = apply_theme(fig_resid)
    st.plotly_chart(fig_resid)

    st.subheader("Prédictions avec SARIMA")
    sarima_model = SARIMAX(df["Poids (Kgs)"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_results = sarima_model.fit(disp=False)
    df["SARIMA_Predictions"] = sarima_results.predict(start=0, end=len(df) - 1, dynamic=False)

    fig8 = px.scatter(df, x="Date", y="Poids (Kgs)", title="Prédictions avec le modèle SARIMA")
    fig8.add_trace(go.Scatter(x=df["Date"], y=df["SARIMA_Predictions"], mode='lines', name='Prédictions SARIMA'))
    fig8 = apply_theme(fig8)
    st.plotly_chart(fig8)

with tab5:
    st.header("Comparaison des Modèles")
    st.subheader("Évaluation des performances des modèles")
    models = {
        "Régression Linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    model_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=tscv)
        model_scores[name] = -scores.mean()

    scores_df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['MSE'])
    st.write(scores_df)

    # Entraînement des modèles sur l'ensemble des données
    for name, model in models.items():
        model.fit(X, y)
        df[name + "_Predictions"] = model.predict(X)

    # Graphique de comparaison
    fig_compare = px.scatter(df, x="Date", y="Poids (Kgs)", title="Comparaison des Modèles")
    for name in models.keys():
        fig_compare.add_trace(go.Scatter(x=df["Date"], y=df[name + "_Predictions"], mode='lines', name=name))
    fig_compare = apply_theme(fig_compare)
    st.plotly_chart(fig_compare)

    # Affichage des métriques de performance
    st.subheader("Métriques de performance")
    for name in models.keys():
        mse = mean_squared_error(y, df[name + "_Predictions"])
        r2 = r2_score(y, df[name + "_Predictions"])
        st.write(f"**{name}:** MSE = {mse:.2f}, R² = {r2:.2f}")

with tab6:
    st.header("Personnalisation")
    st.write("Sélectionnez un thème pour les graphiques dans la barre latérale.")

    if current_weight <= target_weight_3:
        st.balloons()
        st.success("Félicitations ! Vous avez atteint votre objectif de poids le plus ambitieux !")
    elif current_weight <= target_weight_2:
        st.success("Bravo ! Vous avez atteint le deuxième objectif de poids.")
    elif current_weight <= target_weight:
        st.info("Bien joué ! Vous avez atteint le premier objectif de poids.")
    else:
        st.warning("Continuez vos efforts pour atteindre vos objectifs de poids.")

    # Graphique de progression vers les objectifs
    objectifs = {
        "Poids actuel": current_weight,
        "Objectif 1": target_weight,
        "Objectif 2": target_weight_2,
        "Objectif 3": target_weight_3
    }
    fig_objectifs = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_weight,
        delta={'reference': target_weight_3},
        gauge={
            'axis': {'range': [target_weight_3 - 20, initial_weight + 10]},
            'steps': [
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
    fig_objectifs.update_layout(title="Progression vers les objectifs de poids")
    fig_objectifs = apply_theme(fig_objectifs)
    st.plotly_chart(fig_objectifs)

with tab7:
    st.header("Téléchargement et Gestion des Données")
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV avec vos données de poids", type=["csv"])
    if uploaded_file:
        df_user = pd.read_csv(uploaded_file)
        st.write("Aperçu de vos données téléchargées :")
        st.write(df_user.head())
        # Option pour fusionner avec les données existantes
        if st.button("Fusionner avec les données existantes"):
            df_user['Poids (Kgs)'] = df_user['Poids (Kgs)'].astype(str).str.replace(',', '.').str.strip()
            df_user['Poids (Kgs)'] = pd.to_numeric(df_user['Poids (Kgs)'], errors='coerce')
            df_user['Date'] = pd.to_datetime(df_user['Date'], dayfirst=True, errors='coerce')
            df_user = df_user.dropna(subset=['Poids (Kgs)', 'Date'])
            df_user = df_user.sort_values('Date', ascending=True)
            df = pd.concat([df, df_user])
            df = df.drop_duplicates(subset=['Date']).sort_values('Date')
            st.success("Vos données ont été fusionnées avec succès.")

    # Option pour télécharger les données actuelles
    st.subheader("Téléchargez vos données")
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    st.download_button(
        label="Télécharger les données au format CSV",
        data=csv,
        file_name='donnees_poids.csv',
        mime='text/csv',
    )

    # Option pour réinitialiser les données
    if st.button("Réinitialiser les données"):
        df = pd.DataFrame(columns=['Date', 'Poids (Kgs)'])
        st.success("Les données ont été réinitialisées.")
