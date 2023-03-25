import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Charger le fichier CSV à partir du lien
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

# Afficher le graphique de l'évolution du poids par rapport à la date
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Poids (Kgs)'], marker='o')
ax.set_xlabel('Date')
ax.set_ylabel('Poids (Kgs)')
ax.set_title('Evolution du poids')

# Corriger l'axe des poids et des dates
ax.set_ylim(bottom=80, top=100)  # La plage minimale est 80 et la plage maximale est 100

# Afficher uniquement les dates pour lesquelles il y a des données de poids
ax.set_xticks(df['Date'])
ax.set_xticklabels(df['Date'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
st.pyplot(fig)
