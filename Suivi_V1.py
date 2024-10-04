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

# Configuration de l'application Streamlit
st.set_page_config(page_title="Suivi du Poids - Tableau de Bord Interactif", layout="wide")
st.title("Suivi de l'Évolution du Poids")
st.markdown("Cette application vous aide à suivre votre progression de poids, prédire des tendances et atteindre vos objectifs de façon efficace.")

# Fonction pour charger et traiter les données
@st.cache_data(ttl=300)  # Expiration du cache toutes les 5 minutes
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

# Affichage des données chargées
st.write(f"**Nombre total de lignes chargées :** {df.shape[0]}")
st.write(df.tail())  # Afficher les dernières lignes pour vérifier que toutes les données sont présentes

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
target_weight_1 = st.sidebar.number_input("Objectif de poids 1 (Kgs)", value=95.0)
target_weight_2 = st.sidebar.number_input("Objectif de poids 2 (Kgs)", value=90.0)
target_weight_3 = st.sidebar.number_input("Objectif de poids 3 (Kgs)", value=85.0)

# Fonction pour appliquer le thème
def apply_theme(fig):
    if theme == "Dark":
        fig.update_layout(template="plotly_dark")
    elif theme == "Light":
        fig.update_layout(template="plotly_white")
    return fig

# Diviser l'application en onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Résumé", "Graphiques", "Prévisions", "Analyse des Données", "Téléchargement"])

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
        progress_percent = (weight_lost / total_weight_to_lose) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Poids actuel", f"{current_weight:.2f} Kgs")
        col2.metric("Objectif de poids", f"{target_weight_3} Kgs")
        col3.metric("Progrès vers l'objectif", f"{progress_percent:.2f} %")

        st.write("Statistiques des poids :")
        st.write(df["Poids (Kgs)"].describe())

with tab2:
    st.header("Graphiques")
    # Graphique de l'évolution du poids avec la moyenne mobile
    df["Poids_rolling_mean"] = df["Poids (Kgs)"].rolling(window=window_size, min_periods=1).mean()
    fig = px.line(df, x="Date", y=["Poids (Kgs)", "Poids_rolling_mean"], markers=True, labels={"value": "Poids (Kgs)", "variable": "Type"})
    fig.update_layout(title="Évolution du poids")

    # Ajout des lignes des objectifs de poids
    fig.add_hline(y=target_weight_1, line_dash="dash", annotation_text="Objectif 1", annotation_position="bottom right")
    fig.add_hline(y=target_weight_2, line_dash="dash", line_color="red", annotation_text="Objectif 2", annotation_position="bottom right")
    fig.add_hline(y=target_weight_3, line_dash="dash", line_color="green", annotation_text="Objectif 3", annotation_position="bottom right")

    # Appliquer le thème
    fig = apply_theme(fig)
    st.plotly_chart(fig)

    # Histogramme de la distribution des poids
    fig2 = px.histogram(df, x="Poids (Kgs)", nbins=30, title="Distribution des poids")
    fig2 = apply_theme(fig2)
    st.plotly_chart(fig2)

    # Graphique des anomalies détectées
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomalies'] = iso_forest.fit_predict(df[['Poids (Kgs)']])
    fig3 = px.scatter(df, x="Date", y="Poids (Kgs)", color="Anomalies", color_discrete_sequence=["blue", "red"], title="Évolution du poids avec détection des anomalies")
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

    fig4 = px.line(df, x="Date", y=["Poids (Kgs)", predictions], labels={"y": "Poids (Kgs)"}, title="Régression linéaire de l'évolution du poids")
    fig4 = apply_theme(fig4)
    st.plotly_chart(fig4)

    # Calculer correctement la date d'atteinte de l'objectif de poids
    if reg.coef_[0] != 0:
        days_to_target = (target_weight_3 - reg.intercept_) / reg.coef_[0]
        target_date = df["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
        st.write(f"Date estimée pour atteindre l'objectif de poids : {target_date.date()}")
    else:
        st.error("Impossible de prédire la date d'atteinte de l'objectif car le coefficient de régression est nul.")

with tab4:
    st.header("Analyse des Données")
    st.subheader("Analyse de tendance saisonnière")
    stl = STL(df['Poids (Kgs)'], period=7)
    res = stl.fit()
    fig5 = res.plot()
    st.pyplot(fig5)

    st.subheader("Clustering des données")
    try:
        kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['Poids (Kgs)']])
        df['Cluster'] = kmeans.labels_

        fig_cluster = px.scatter(df, x='Date', y='Poids (Kgs)', color='Cluster', title='Clustering des données de poids')
        fig_cluster = apply_theme(fig_cluster)
        st.plotly_chart(fig_cluster)
    except ValueError as e:
        st.error(f"Erreur lors du clustering : {e}")

with tab5:
    st.header("Téléchargement de vos données")
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
