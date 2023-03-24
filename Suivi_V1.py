import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Charger le fichier CSV à partir du lien
url = 'https://docs.google.com/spreadsheets/d/1qPhLKvm4BREErQrm0L38DcZFG4a-K0msSzARVIG_T_U/export?format=csv'
df = pd.read_csv(url)

# Convertir la colonne "Poids (Kgs)" en nombres décimaux
df['Poids (Kgs)'] = pd.to_numeric(df['Poids (Kgs)'])

# Modifier le format de la chaîne de date dans la colonne "Date"
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y') # à adapter selon le format de date utilisé

# Trier le DataFrame par ordre croissant en fonction de la colonne "Date"
df = df.sort_values('Date')

# Afficher le graphique de l'évolution du poids par rapport à la date
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Poids (Kgs)'])
ax.set_xlabel('Date')
ax.set_ylabel('Poids (Kgs)')
ax.set_title('Evolution du poids')

# Corriger l'axe des poids et des dates
ax.set_ylim(bottom=0)  # La plage minimale est 0
st.pyplot(fig)
