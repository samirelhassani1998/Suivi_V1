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
