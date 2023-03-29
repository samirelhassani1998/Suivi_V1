import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
import streamlit as st

# Ajouter ces lignes pour installer fbprophet si nécessaire
# import sys
# !{sys.executable} -m pip install fbprophet
from fbprophet import Prophet

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

# Modèle STL
stl = STL(df_filtered["Poids (Kgs)"], period=7, seasonal=7)
stl_result = stl.fit()
st.write("Décomposition STL :")
stl_fig = stl_result.plot()
st.pyplot(stl_fig)

# Régression polynomiale
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X, y)
poly_predictions = poly_model.predict(X)

fig6 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig6.add_traces(px.line(df_filtered, x="Date", y=poly_predictions, labels={"y": "Régression polynomiale"}).data[0])
fig6.update_layout(title="Régression polynomiale de l'évolution du poids")
st.plotly_chart(fig6)

# Régression Ridge
ridge_model = Ridge(alpha=0.5)
ridge_model.fit(X, y)
ridge_predictions = ridge_model.predict(X)

fig7 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig7.add_traces(px.line(df_filtered, x="Date", y=ridge_predictions, labels={"y": "Régression Ridge"}).data[0])
fig7.update_layout(title="Régression Ridge de l'évolution du poids")
st.plotly_chart(fig7)

# Régression Lasso
lasso_model = Lasso(alpha=0.5)
lasso_model.fit(X, y)
lasso_predictions = lasso_model.predict(X)

fig8 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig8.add_traces(px.line(df_filtered, x="Date", y=lasso_predictions, labels={"y": "Régression Lasso"}).data[0])
fig8.update_layout(title="Régression Lasso de l'évolution du poids")
st.plotly_chart(fig8)

# Analyse des anomalies
fig9 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig9.add_traces(px.scatter(df_filtered, x=anomalies["Date"], y=anomalies["Poids (Kgs)"], labels={"y": "Anomalies"}, marker=dict(color="red", symbol="x", size=10)).data[0])
fig9.update_layout(title="Analyse des anomalies")
st.plotly_chart(fig9)

# Prédictions avec le modèle Facebook Prophet
# Préparer les données pour le modèle Prophet
prophet_df = df_filtered[["Date", "Poids (Kgs)"]].rename(columns={"Date": "ds", "Poids (Kgs)": "y"})

prophet_model = Prophet()
prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=30)  # Prédire 30 jours dans le futur
forecast = prophet_model.predict(future)

fig10 = prophet_model.plot(forecast)
st.write("Prédictions avec le modèle Facebook Prophet :")
st.pyplot(fig10)

fig11 = prophet_model.plot_components(forecast)
st.write("Composantes du modèle Facebook Prophet :")
st.pyplot(fig11)
