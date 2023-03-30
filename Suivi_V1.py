import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Titre de l'application
st.title("Evolution du poids")

# Charger le fichier CSV à partir du lien
url = 'https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv'
data_frame = pd.read_csv(url, decimal=",")

# Vérifier si les données sont chargées avec succès
if data_frame.empty:
    st.error("Le chargement des données a échoué.")
else:
    # Convertir la colonne "Poids (Kgs)" en nombres décimaux et remplacer les valeurs non numériques par NaN
    data_frame['poids'] = pd.to_numeric(data_frame['Poids (Kgs)'], errors='coerce')
    # Supprimer les lignes contenant des NaN
    data_frame = data_frame.dropna()
    # Modifier le format de la chaîne de date dans la colonne "Date"
    data_frame['date'] = pd.to_datetime(data_frame['Date'], format='%d/%m/%Y')
    # Trier le DataFrame par ordre croissant en fonction de la colonne "Date"
    data_frame = data_frame.sort_values('date')
    # Utiliser le DataFrame complet sans filtrer par plage de dates
    data_frame_filtered = data_frame.copy()
    # Mettre à jour l'index du DataFrame
    data_frame_filtered = data_frame_filtered.reset_index(drop=True)

    # Calculer la moyenne mobile
    window_size = st.slider("Taille de la fenêtre pour la moyenne mobile (jours)", 1, 30, 7)
    data_frame_filtered["poids_rolling_mean"] = data_frame_filtered["poids"].rolling(window=window_size).mean()

    # Entrer l'objectif de poids
    target_weight = st.number_input("Objectif de poids (Kgs)", value=85.0)

    # Afficher les statistiques des poids
    poids_stats = data_frame_filtered["poids"].describe()
    st.write("Statistiques des poids :", poids_stats)

    # Créer un graphique interactif avec Plotly
    fig = px.line(data_frame_filtered, x="date", y="poids", markers=True, labels={"poids": "Poids (Kgs)", "date": "
# Afficher la courbe de la moyenne mobile et l'objectif de poids
    fig.add_scatter(x=data_frame_filtered["date"], y=data_frame_filtered["poids_rolling_mean"], mode="lines", name="Moyenne mobile")
    fig.update_layout(title="Evolution du poids")
    fig.add_hline(y=target_weight, line_dash="dash", annotation_text="Objectif", annotation_position="bottom right")
    st.plotly_chart(fig)

    # Histogramme de la distribution des poids
    fig2 = px.histogram(data_frame_filtered, x="poids", nbins=30, title="Distribution des poids")
    st.plotly_chart(fig2)

    # Prédiction de la date d'atteinte de l'objectif de poids
    data_frame_filtered["date_numeric"] = (data_frame_filtered["date"] - data_frame_filtered["date"].min()) / np.timedelta64(1, "D")
    X = data_frame_filtered[["date_numeric"]]
    y = data_frame_filtered["poids"]
    reg = LinearRegression().fit(X, y)
    # Ajouter un graphique de régression linéaire
    predictions = reg.predict(X)
    fig3 = px.scatter(data_frame_filtered, x="date", y="poids", labels={"poids": "Poids (Kgs)", "date": "Date"})
    fig3.add_traces(px.line(data_frame_filtered, x="date", y=predictions, labels={"y": "Régression linéaire"}).data[0])
    fig3.update_layout(title="Régression linéaire de l'évolution du poids")
    st.plotly_chart(fig3)
    days_to_target = int((target_weight- reg.intercept_) / reg.coef_[0])
    target_date = data_frame_filtered["date"].min() + pd.to_timedelta(days_to_target, unit="D")
    st.write(f"Date estimée pour atteindre l'objectif de poids : {target_date.date()}")

    # Calculer le taux de changement moyen du poids
    data_frame_filtered["poids_diff"] = data_frame_filtered["poids"].diff()
    mean_change_rate = data_frame_filtered["poids_diff"].mean()
    st.write(f"Taux de changement moyen du poids : {mean_change_rate:.2f} Kgs par jour")

    # Demander l'âge, la taille et le sexe pour calculer les calories nécessaires
    age = st.number_input("Âge (années)", value=24)
    height = st.number_input("Taille (cm)", value=182)
    sex = st.selectbox("Sexe", options=["Homme", "Femme"])

    # Calculer le taux métabolique de base (BMR)
    if sex == "Homme":
        bmr = 10 * target_weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * target_weight + 6.25 * height - 5 * age - 161

    # Demander le niveau d'activité et calculer les calories nécessaires pour atteindre l'objectif de poids
    activity_levels = {
        "Sédentaire": 1.2,
        "Légèrement actif": 1.375,
        "Modérément actif": 1.55,
        "Très actif": 1.725,
        "Extrêmement actif": 1.9,
    }
    activity_level = st.select
    box2 = st.sidebar.box(title='Activité physique')
    box2.write('**Niveau d\'activité**')
    selected_activity_level = box2.radio("", list(activity_levels.keys()))
    selected_activity_level_value = activity_levels[selected_activity_level]

    # Calculer les calories nécessaires pour atteindre l'objectif de poids
    current_weight = data_frame_filtered["poids"].iloc[-1]
    weight_difference = target_weight - current_weight
    days_to_target = st.slider("Nombre de jours pour atteindre l'objectif de poids", 30, 365, 90)
    calories_needed = bmr * selected_activity_level_value
    calories_needed_per_day = calories_needed + (weight_difference * 7700) / days_to_target

    # Afficher les résultats
    st.markdown(f'<p style="color: red; font-weight: bold; font-style: italic;">Calories nécessaires à consommer par jour pour atteindre l\'objectif de poids en {days_to_target} jours : {calories_needed_per_day:.0f} kcal</p>', unsafe_allow_html=True)

    # Analyse de la série chronologique
    data_frame_filtered["residuals"] = data_frame_filtered["poids"] - predictions
    fig4 = px.scatter(data_frame_filtered, x="date", y="residuals", labels={"residuals": "Résidus", "date": "Date"})
    fig4.update_layout(title="Analyse des résidus")
    st.plotly_chart(fig4)

    # Prédictions avec le modèle SARIMA
    sarima_model = auto_arima(data_frame_filtered["poids"], seasonal=True, m=7, suppress_warnings=True, stepwise=True)
    st.write(f"Meilleur modèle SARIMA : {sarima_model.summary()}")
    sarima_predictions = sarima_model.predict_in_sample()
    data_frame_filtered["sarima_predictions"] = sarima_predictions
    fig5 = px.scatter(data_frame_filtered, x="date", y="poids", labels={"poids": "Poids (Kgs)", "date": "Date"})
    fig5.add_traces(px.line(data_frame_filtered, x="date", y=sarima_predictions, labels={"y": "Prédictions SARIMA"}).data[0])
    fig5.update_layout(title="Prédictions avec le modèle SARIMA")
    st.plotly_chart(fig5)

    # Comparaison des modèles de régression
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X, y)
    lin_reg_r2 = r2_score(y, predictions)
    rf_reg_r2 = r2_score(y, rf_reg.predict(X))
    st.write(f"Score R2 pour la régression linéaire : {lin_reg_r2:.2f}")
    st.write(f"Score R2 pour la régression Random Forest : {rf_reg_r2:.2f}")

    # Importance des caractéristiques pour le modèle Random Forest
    st.write("Importance des caractéristiques pour le modèle Random Forest :", rf_reg.feature_importances_)
