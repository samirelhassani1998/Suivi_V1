import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st

st.title("Evolution du poids")

# Charger le fichier CSV à partir du lie
url = 'https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv'
df = pd.read_csv(url)

# Convertir la colonne "Poids (Kgs)" en nombres décimaux et remplacer les valeurs non numériques par NaN
df['Poids (Kgs)'] = pd.to_numeric(df['Poids (Kgs)'], errors='coerce')

# Supprimer les lignes contenant des NaN
df = df.dropna()

# Modifier le format de la chaîne de date dans la colonne "Date"
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y') # à adapter selon le format de date utilisé

# Trier le DataFrame par ordre croissant en fonction de la colonne "Date"
df = df.sort_values('Date')

# Sélectionner la plage de dates
start_date = st.date_input("Date de début", df["Date"].min().date())
end_date = st.date_input("Date de fin", df["Date"].max().date())

# Filtrer le DataFrame en fonction de la plage de dates
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

poids_stats = df_filtered["Poids (Kgs)"].describe()
st.write("Statistiques des poids :", poids_stats)

# Créer un graphique interactif avec Plotly
fig = px.line(df_filtered, x="Date", y="Poids (Kgs)", markers=True, labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
fig.update_layout(title="Evolution du poids")
fig.add_hline(y=85, line_dash="dash", annotation_text="Objectif", annotation_position="bottom right")
st.plotly_chart(fig)

# Convertir les dates en nombres pour la régression linéaire
df_filtered["Date_numeric"] = (df_filtered["Date"] - df_filtered["Date"].min()) / np.timedelta64(1, "D")

# Entraîner le modèle de régression linéaire
X = df_filtered[["Date_numeric"]]
y = df_filtered["Poids (Kgs)"]
reg = LinearRegression().fit(X, y)

# Calculer le coefficient de détermination (R²)
r_squared = reg.score(X, y)
st.write(f"R²: {r_squared}")
