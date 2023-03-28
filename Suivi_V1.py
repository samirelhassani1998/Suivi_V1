import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st

st.title("Evolution du poids")

# Charger le fichier CSV à partir du lien
url = 'https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv'
df = pd.read_csv(url)

# Convertir la colonne "Poids (Kgs)" en nombres décimaux et remplacer les valeurs non numériques par NaN
df['Poids (Kgs)'] = pd.to_numeric(df['Poids (Kgs)'], errors='coerce')

# Supprimer les lignes contenant des NaN
df = df.dropna()

# Modifier le format de la chaîne de date dans la colonne "Date"
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Trier le DataFrame par ordre croissant en fonction de la colonne "Date"
df = df.sort_values('Date')

# Sélectionner la plage de dates
start_date = st.date_input("Date de début", df["Date"].min().date())
end_date = st.date_input("Date de fin", df["Date"].max().date())

# Filtrer le DataFrame en fonction de la plage de dates
df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

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

days_to_target = int((target_weight- reg.intercept_) / reg.coef_[0])

target_date = df_filtered["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
st.write(f"Date estimée pour atteindre l'objectif de poids : {target_date.date()}")

# Calculer le taux de changement moyen du poids
df_filtered["Poids_diff"] = df_filtered["Poids (Kgs)"].diff()
mean_change_rate = df_filtered["Poids_diff"].mean()
st.write(f"Taux de changement moyen du poids : {mean_change_rate:.2f} Kgs par jour")

age = st.number_input("Âge (années)", value=25)
height = st.number_input("Taille (cm)", value=175)
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
