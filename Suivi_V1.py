import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans

# Titre de l'application Streamlit
st.set_page_config(page_title="Suivi du poids", layout="wide")
st.title("Suivi de l'évolution du poids")

# Fonction pour charger et traiter les données
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, decimal=",")
    df['Poids (Kgs)'] = pd.to_numeric(df['Poids (Kgs)'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.dropna()
    df = df.sort_values('Date', ascending=True)
    st.write("Vérification des données après traitement :")
    st.write(df.tail(10))  # Afficher les 10 dernières lignes pour confirmer
    return df

# URL du fichier CSV
url = 'https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv'
df = load_data(url)

# Vérification des données chargées
st.write(f"Nombre total de lignes chargées : {df.shape[0]}")
st.write(df.tail())  # Afficher les dernières lignes pour vérifier que toutes les données sont là

# Interface utilisateur pour les paramètres
st.sidebar.header("Paramètres de la moyenne mobile")
window_size = st.sidebar.slider("Taille de la fenêtre pour la moyenne mobile (jours)", 1, 30, 7)
df["Poids_rolling_mean"] = df["Poids (Kgs)"].rolling(window=window_size).mean()

st.sidebar.header("Objectifs de poids")
target_weight = st.sidebar.number_input("Objectif de poids 1 (Kgs)", value=90.0)
target_weight_2 = st.sidebar.number_input("Objectif de poids 2 (Kgs)", value=85.0)
target_weight_3 = st.sidebar.number_input("Objectif de poids 3 (Kgs)", value=80.0)

# Interface utilisateur pour le thème
st.sidebar.header("Thème")
theme = st.sidebar.selectbox("Choisir un thème", ["Default", "Dark", "Light"])

# Diviser l'application en onglets
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Résumé", "Graphiques", "Prévisions", "Analyse des Données", "Personnalisation", "Téléchargement"])

with tab1:
    st.header("Résumé")
    st.write("Voici un résumé de vos progrès vers les objectifs de poids.")
    if df.empty:
        st.warning("Aucune donnée disponible.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Poids actuel", f"{df['Poids (Kgs)'].iloc[-1]:.2f} Kgs")
        col2.metric("Objectif de poids", f"{target_weight} Kgs")
        col3.metric("Progrès vers l'objectif", f"{100 * (df['Poids (Kgs)'].iloc[-1] / target_weight):.2f} %")
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

    # Calcul de la moyenne mobile des 7 derniers jours
    df["Last_7_rolling_mean"] = df["Poids (Kgs)"].rolling(window=7, min_periods=1).mean()

    # Ajout de la courbe de la moyenne mobile des 7 derniers jours au graphique
    fig.add_scatter(x=df["Date"], y=df["Last_7_rolling_mean"], mode="lines", name="Moyenne mobile 7 jours")

    fig.update_layout(title="Evolution du poids")
    fig.add_hline(y=target_weight, line_dash="dash", annotation_text="Objectif 1", annotation_position="bottom right")
    fig.add_hline(y=target_weight_2, line_dash="dash", line_color="red", annotation_text="Objectif 2", annotation_position="bottom right")
    fig.add_hline(y=target_weight_3, line_dash="dash", line_color="green", annotation_text="Objectif 3", annotation_position="bottom right")

    # Appliquer le thème sélectionné
    if theme == "Dark":
        fig.update_layout(template="plotly_dark")
    elif theme == "Light":
        fig.update_layout(template="plotly_white")

    st.plotly_chart(fig)

    # Histogramme de la distribution des poids
    fig2 = px.histogram(df, x="Poids (Kgs)", nbins=30, title="Distribution des poids")
    st.plotly_chart(fig2)

    # Graphique des anomalies détectées
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomalies'] = iso_forest.fit_predict(df[['Poids (Kgs)']])
    fig3 = px.scatter(df, x="Date", y="Poids (Kgs)", color="Anomalies", color_discrete_sequence=["blue", "red"])
    fig3.update_layout(title="Evolution du poids avec détection des anomalies")
    st.plotly_chart(fig3)
    st.write("Points de données inhabituels détectés :")
    st.write(df[df["Anomalies"] == -1])

with tab3:
    st.header("Prévisions")
    # Prédiction de la date d'atteinte de l'objectif de poids avec régression linéaire
    df["Date_numeric"] = (df["Date"] - df["Date"].min()) / np.timedelta64(1, "D")
    X = df[["Date_numeric"]]
    y = df["Poids (Kgs)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    predictions = reg.predict(X)
    y_test_pred = reg.predict(X_test)

    fig4 = px.scatter(df, x="Date", y="Poids (Kgs)")
    fig4.add_trace(px.line(df, x="Date", y=predictions, labels={"y": "Prévisions"}).data[0])
    fig4.update_layout(title="Régression linéaire de l'évolution du poids")
    st.plotly_chart(fig4)

    # Calculer correctement la date d'atteinte de l'objectif de poids
    try:
        days_to_target = int((target_weight - reg.intercept_) / reg.coef_[0])
        if days_to_target < 0:
            st.warning("La date d'atteinte de l'objectif de poids est passée.")
            target_date = df["Date"].max()  # Si la date est passée, utilisez la date actuelle
        else:
            target_date = df["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
        st.write(f"Date estimée pour atteindre l'objectif de poids : {target_date.date()}")
    except Exception as e:
        st.error(f"Erreur dans le calcul de la date estimée : {e}")

    # Calculer le taux de changement moyen du poids
    df["Poids_diff"] = df["Poids (Kgs)"].diff()
    mean_change_rate = df["Poids_diff"].mean()
    st.write(f"Taux de changement moyen du poids : {mean_change_rate:.2f} Kgs par jour")

    st.subheader("Validation des modèles")
    lin_scores = cross_val_score(reg, X, y, scoring='neg_mean_squared_error', cv=5)
    st.write(f"Score MSE moyen pour la régression linéaire : {-lin_scores.mean():.2f}")

with tab4:
    st.header("Analyse des Données")
    st.subheader("Analyse de tendance saisonnière")
    stl = STL(df['Poids (Kgs)'], period=7)
    res = stl.fit()
    fig5 = res.plot()
    st.pyplot(fig5)

    df["Trend"] = res.trend
    df["Seasonal"] = res.seasonal

    fig6 = px.line(df, x="Date", y="Trend")
    fig6.update_layout(title="Tendance de l'évolution du poids")
    st.plotly_chart(fig6)

    fig7 = px.line(df, x="Date", y="Seasonal")
    fig7.update_layout(title="Saisonnalité de l'évolution du poids")
    st.plotly_chart(fig7)

    st.subheader("Prédictions avec SARIMA")
    sarima_model = SARIMAX(df["Poids (Kgs)"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_results = sarima_model.fit(disp=False)
    df["SARIMA_Predictions"] = sarima_results.predict(start=0, end=len(df) - 1)

    fig8 = px.scatter(df, x="Date", y="Poids (Kgs)")
    fig8.add_trace(px.line(df, x="Date", y=df["SARIMA_Predictions"], labels={"y": "Prédictions SARIMA"}).data[0])
    fig8.update_layout(title="Prédictions avec le modèle SARIMA")
    st.plotly_chart(fig8)

    st.subheader("Comparaison des modèles de régression")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    rf_scores = cross_val_score(rf_reg, X, y, scoring='neg_mean_squared_error', cv=5)
    st.write(f"Score MSE moyen pour le modèle Random Forest : {-rf_scores.mean():.2f}")

    # Importance des caractéristiques pour le modèle Random Forest
    st.write("Importance des caractéristiques pour le modèle Random Forest :")
    st.write(rf_reg.feature_importances_)

    fig9 = px.scatter(df, x="Date", y="Poids (Kgs)")
    fig9.add_trace(px.line(df, x="Date", y=predictions, labels={"y": "Régression linéaire"}).data[0])
    fig9.add_scatter(x=df.iloc[X_test.index]["Date"], y=y_test_pred, mode="markers", name="Prédictions sur ensemble de test")
    fig9.update_layout(title="Régression linéaire avec prédictions sur ensemble de test")
    st.plotly_chart(fig9)

    # Clustering des données
    if not df.empty:
        try:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['Poids (Kgs)']])
            df['Cluster'] = kmeans.labels_

            fig_cluster = px.scatter(df, x='Date', y='Poids (Kgs)', color='Cluster', title='Clustering des données de poids')
            st.plotly_chart(fig_cluster)
        except ValueError as e:
            st.error(f"Erreur lors du clustering : {e}")
    else:
        st.warning("Pas de données suffisantes pour le clustering.")

with tab5:
    st.header("Personnalisation")
    st.write("Sélectionnez un thème pour les graphiques.")
    if theme == "Dark":
        fig.update_layout(template="plotly_dark")
    elif theme == "Light":
        fig.update_layout(template="plotly_white")

    if df['Poids (Kgs)'].iloc[-1] < target_weight:
        st.success("Félicitations ! Vous avez atteint votre objectif de poids !")
    elif df['Poids (Kgs)'].iloc[-1] > target_weight_3:
        st.warning("Attention, votre poids est au-dessus de l'objectif 3.")

    fig_objectifs = px.bar(
        x=["Poids actuel", "Objectif 1", "Objectif 2", "Objectif 3"],
        y=[df['Poids (Kgs)'].iloc[-1], target_weight, target_weight_2, target_weight_3],
        title="Progression vers les objectifs de poids",
        color=["Poids actuel", "Objectif 1", "Objectif 2", "Objectif 3"]
    )
    st.plotly_chart(fig_objectifs)

with tab6:
    st.header("Téléchargement")
    uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])
    if uploaded_file:
        df_user = pd.read_csv(uploaded_file)
        st.write("Aperçu des données téléchargées :")
        st.write(df_user.head())
