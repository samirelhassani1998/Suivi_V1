import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

poids_stats = df["Poids (Kgs)"].describe()
print(poids_stats)

plt.plot(df["Date"], df["Poids (Kgs)"], marker="o")
plt.xlabel("Date")
plt.ylabel("Poids (Kgs)")
plt.title("Evolution du poids")
plt.show()

# Convertir les dates en nombres pour la régression linéaire
df["Date_numeric"] = (df["Date"] - df["Date"].min()) / np.timedelta64(1, "D")

# Entraîner le modèle de régression linéair
X = df[["Date_numeric"]]
y = df["Poids (Kgs)"]
reg = LinearRegression().fit(X, y)

# Tracer la ligne de régression
plt.scatter(df["Date"], df["Poids (Kgs)"], label="Données")
plt.plot(df["Date"], reg.predict(X), color="red", label="Régression linéaire")
plt.xlabel("Date")
plt.ylabel("Poids (Kgs)")
plt.title("Evolution du poids avec régression linéaire")
plt.legend()
plt.show()

# Calculer le coefficient de détermination (R²)
r_squared = reg.score(X, y)
print(f"R²: {r_squared}")
