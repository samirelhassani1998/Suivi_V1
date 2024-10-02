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

# Fonction pour appliquer le thème
def apply_theme(fig):
    if theme == "Dark":
        fig.update_layout(template="plotly_dark")
    elif theme == "Light":
        fig.update_layout(template="plotly_white")
    return fig

# Diviser l'application en onglets
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Résumé", "Graphiques", "Prévisions", "Analyse des Données", "Personnalisation", "Téléchargement"])

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

    fig4 = px.scatter(df, x="Date", y="Poids (Kgs)")
    fig4.add_trace(px.line(df, x="Date", y=predictions, labels={"y": "Prévisions"}).data[0])
    fig4.update_layout(title="Régression linéaire de l'évolution du poids")
    fig4 = apply_theme(fig4)
    st.plotly_chart(fig4)

    # Calculer correctement la date d'atteinte de l'objectif de poids
    try:
        if reg.coef_[0] == 0:
            st.error("Impossible de prédire la date d'atteinte de l'objectif car le coefficient de régression est nul.")
        else:
            days_to_target = (target_weight - reg.intercept_) / reg.coef_[0]
            target_date = df["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
            st.write(f"Date estimée pour atteindre l'objectif de poids : {target_date.date()}")
    except Exception as e:
        st.error(f"Erreur dans le calcul de la date estimée : {e}")

    # Calculer le taux de changement moyen du poids
    df["Poids_diff"] = df["Poids (Kgs)"].diff()
    mean_change_rate = df["Poids_diff"].mean()
    st.write(f"Taux de changement moyen du poids : {mean_change_rate:.2f} Kgs par jour")

with tab4:
    st.header("Analyse des Données")
    st.subheader("Analyse de tendance saisonnière")
    stl = STL(df['Poids (Kgs)'], period=7)
    res = stl.fit()
    fig5 = res.plot()
    st.pyplot(fig5)

    df["Trend"] = res.trend
    df["Seasonal"] = res.seasonal

    fig6 = px.line(df, x="Date", y="Trend", title="Tendance de l'évolution du poids")
    fig6 = apply_theme(fig6)
    st.plotly_chart(fig6)

    fig7 = px.line(df, x="Date", y="Seasonal", title="Saisonnalité de l'évolution du poids")
    fig7 = apply_theme(fig7)
    st.plotly_chart(fig7)

    st.subheader("Prédictions avec SARIMA")
    sarima_model = SARIMAX(df["Poids (Kgs)"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_results = sarima_model.fit(disp=False)
    df["SARIMA_Predictions"] = sarima_results.predict(start=0, end=len(df) - 1)

    fig8 = px.scatter(df, x="Date", y="Poids (Kgs)")
    fig8.add_trace(px.line(df, x="Date", y=df["SARIMA_Predictions"], labels={"y": "Prédictions SARIMA"}).data[0])
    fig8.update_layout(title="Prédictions avec le modèle SARIMA")
    fig8 = apply_theme(fig8)
    st.plotly_chart(fig8)

    st.subheader("Comparaison des modèles de régression")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf_reg, X, y, scoring='neg_mean_squared_error', cv=tscv)
    st.write(f"Score MSE moyen pour le modèle Random Forest : {-rf_scores.mean():.2f}")

    # Entraînement du modèle Random Forest sur l'ensemble des données
    rf_reg.fit(X, y)
    rf_predictions = rf_reg.predict(X)

    fig9 = px.scatter(df, x="Date", y="Poids (Kgs)", title="Comparaison des modèles")
    fig9.add_trace(px.line(df, x="Date", y=predictions, labels={"y": "Régression linéaire"}).data[0])
    fig9.add_trace(px.line(df, x="Date", y=rf_predictions, labels={"y": "Random Forest"}).data[0])
    fig9 = apply_theme(fig9)
    st.plotly_chart(fig9)

    # Clustering des données
    if not df.empty:
        try:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['Poids (Kgs)']])
            df['Cluster'] = kmeans.labels_

            fig_cluster = px.scatter(df, x='Date', y='Poids (Kgs)', color='Cluster', title='Clustering des données de poids')
            fig_cluster = apply_theme(fig_cluster)
            st.plotly_chart(fig_cluster)
        except ValueError as e:
            st.error(f"Erreur lors du clustering : {e}")
    else:
        st.warning("Pas de données suffisantes pour le clustering.")

with tab5:
    st.header("Personnalisation")
    st.write("Sélectionnez un thème pour les graphiques dans la barre latérale.")

    if current_weight < target_weight:
        st.success("Félicitations ! Vous avez atteint votre objectif de poids !")
    elif current_weight > target_weight_3:
        st.warning("Attention, votre poids est au-dessus de l'objectif 3.")

    fig_objectifs = px.bar(
        x=["Poids actuel", "Objectif 1", "Objectif 2", "Objectif 3"],
        y=[current_weight, target_weight, target_weight_2, target_weight_3],
        title="Progression vers les objectifs de poids",
        color=["Poids actuel", "Objectif 1", "Objectif 2", "Objectif 3"]
    )
    fig_objectifs = apply_theme(fig_objectifs)
    st.plotly_chart(fig_objectifs)

with tab6:
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
