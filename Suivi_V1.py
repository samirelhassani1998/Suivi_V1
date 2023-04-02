import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


st.title("Evolution du poids")
# Charger le fichier CSV à partir du lien
url = 'https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv'
df = pd.read_csv(url, decimal=",")
# Convertir la colonne "Poids (Kgs)" en nombres décimaux et remplacer les valeurs non numériques par NaN
df['Poids (Kgs)'] = pd.to_numeric(df['Poids (Kgs)'], errors='coerce')
# Supprimer les lignes contenant des NaN
df = df.dropna()
# Modifier le format de la chaîne de date dans la colonne "Date"
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# Trier le DataFrame par ordre croissant en fonction de la colonne "Date"
df = df.sort_values('Date')
# Utiliser le DataFrame complet sans filtrer par plage de dates
df_filtered = df.copy()
# Mettre à jour l'index du DataFrame
df_filtered = df_filtered.reset_index(drop=True)
# Calculer la moyenne mobile
window_size = st.slider("Taille de la fenêtre pour la moyenne mobile (jours)", 1, 30, 7)
df_filtered["Poids_rolling_mean"] = df_filtered["Poids (Kgs)"].rolling(window=window_size).mean()
# Entrer l'objectif de poids
target_weight = st.number_input("Objectif de poids (Kgs)", value=85.0)
poids_stats = df_filtered["Poids (Kgs)"].describe()
st.write("Statistiques des poids :", poids_stats)
# Créer un graphique interactif avec Plotly
fig = px.line(df_filtered, x="Date", y="Poids (Kgs)", markers=True, labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig.add_scatter(x=df_filtered["Date"], y=df_filtered["Poids_rolling_mean"], mode="lines", name="Moyenne mobile")
fig.update_layout(title="Evolution du poids")
fig.add_hline(y=target_weight, line_dash="dash", annotation_text="Objectif", annotation_position="bottom right")
st.plotly_chart(fig)
# Histogramme de la distribution des poids
fig2 = px.histogram(df_filtered, x="Poids (Kgs)", nbins=30, title="Distribution des poids")
st.plotly_chart(fig2)
# Prédiction de la date d'atteinte de l'objectif de poids
df_filtered["Date_numeric"] = (df_filtered["Date"] - df_filtered["Date"].min()) / np.timedelta64(1, "D")
X = df_filtered[["Date_numeric"]]
y = df_filtered["Poids (Kgs)"]
reg = LinearRegression().fit(X, y)
# Ajouter un graphique de régression linéaire
predictions = reg.predict(X)
fig3 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig3.add_traces(px.line(df_filtered, x="Date", y=predictions, labels={"y": "Régression linéaire"}).data[0])
fig3.update_layout(title="Régression linéaire de l'évolution du poids")
st.plotly_chart(fig3)
days_to_target = int((target_weight- reg.intercept_) / reg.coef_[0])
target_date = df_filtered["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
st.write(f"Date estimée pour atteindre l'objectif de poids : {target_date.date()}")
# Calculer le taux de changement moyen du poids
df_filtered["Poids_diff"] = df_filtered["Poids (Kgs)"].diff()
mean_change_rate = df_filtered["Poids_diff"].mean()
st.write(f"Taux de changement moyen du poids : {mean_change_rate:.2f} Kgs par jour")
age = st.number_input("Âge (années)", value=24)
height = st.number_input("Taille (cm)", value=182)
sex = st.selectbox("Sexe", options=["Homme", "Femme"])
if sex == "Homme":
    bmr = 10 * target_weight + 6.25 * height - 5 * age + 5
else:
    bmr = 10 * target_weight + 6.25 * height - 5 * age - 161
activity_levels = {
    "Sédentaire": 1.2,
    "Légèrement actif": 1.375,
    "Modérément actif": 1.55,
    "Très actif": 1.725,
    "Extrêmement actif": 1.9,
}
activity_level = st.selectbox("Niveau d'activité", options=list(activity_levels.keys()))
current_weight = df_filtered["Poids (Kgs)"].iloc[-1]
weight_difference = target_weight - current_weight
# Durée en jours pour atteindre l'objectif de poids
days_to_target = 90
# Calculer les calories nécessaires pour atteindre l'objectif de poids en 90 jours
calories_needed = bmr * activity_levels[activity_level]
calories_needed_per_day = calories_needed + (weight_difference * 7700) / days_to_target
st.markdown(f'<p style="color: red; font-weight: bold; font-style: italic;">Calories nécessaires à consommer par jour pour atteindre l\'objectif de poids en {days_to_target} jours : {calories_needed_per_day:.0f} kcal</p>', unsafe_allow_html=True)
#NEWS
decomposition = seasonal_decompose(df_filtered['Poids (Kgs)'], period=7, model='additive')
fig = decomposition.plot()
st.pyplot(fig)
df_filtered["Residuals"] = df_filtered["Poids (Kgs)"] - predictions
fig4 = px.scatter(df_filtered, x="Date", y="Residuals", labels={"Residuals": "Résidus", "Date": "Date"})
fig4.update_layout(title="Analyse des résidus")
st.plotly_chart(fig4)
sarima_model = auto_arima(df_filtered["Poids (Kgs)"], seasonal=True, m=7, suppress_warnings=True, stepwise=True)
st.write(f"Meilleur modèle SARIMA : {sarima_model.summary()}")
sarima_predictions = sarima_model.predict_in_sample()
df_filtered["SARIMA_Predictions"] = sarima_predictions
fig5 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig5.add_traces(px.line(df_filtered, x="Date", y=sarima_predictions, labels={"y": "Prédictions SARIMA"}).data[0])
fig5.update_layout(title="Prédictions avec le modèle SARIMA")
st.plotly_chart(fig5)

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X, y)

# Comparaison des modèles de régression
lin_reg_r2 = r2_score(y, predictions)
rf_reg_r2 = r2_score(y, rf_reg.predict(X))

st.write(f"Score R2 pour la régression linéaire : {lin_reg_r2:.2f}")
st.write(f"Score R2 pour la régression Random Forest : {rf_reg_r2:.2f}")

# Importance des caractéristiques pour le modèle Random Forest
st.write("Importance des caractéristiques pour le modèle Random Forest :", rf_reg.feature_importances_)

# Diviser les données en ensembles d'entraînement et de test
train_size = int(len(df_filtered) * 0.67)
train, test = df_filtered.iloc[:train_size], df_filtered.iloc[train_size:]

# Entraîner un modèle de régression
X_train, y_train = train["Date_numeric"].values.reshape(-1, 1), train["Poids (Kgs)"].values
X_test, y_test = test["Date_numeric"].values.reshape(-1, 1), test["Poids (Kgs)"].values
reg = LinearRegression().fit(X_train, y_train)

# Faites des prédictions sur l'ensemble de test
y_pred = reg.predict(X_test)

# Comparez les prédictions du modèle avec les vraies valeurs
r2 = r2_score(y_test, y_pred)
st.write(f"Score R2 pour la régression linéaire : {r2:.2f}")

# Tracez le modèle de régression linéaire en superposant les données d'entraînement et de test
X = df_filtered[["Date_numeric"]]
predictions = reg.predict(X)
fig6 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig6.add_traces(px.line(df_filtered, x="Date", y=predictions, labels={"y": "Régression linéaire"}).data[0])
fig6.add_scatter(x=test["Date"], y=y_pred, mode="markers", name="Prédictions sur ensemble de test")
fig6.update_layout(title="Régression linéaire avec prédictions sur ensemble de test")
st.plotly_chart(fig6)

# Calculer le coefficient de corrélation de Pearson
correlation = df_filtered['Date_numeric'].corr(df_filtered['Poids (Kgs)'])

# Afficher le coefficient de corrélation de Pearson
st.write(f"Coefficient de corrélation de Pearson entre le temps et le poids : {correlation:.2f}")
