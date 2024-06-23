import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Titre de l'application Streamlit
st.set_page_config(page_title="Suivi du poids", layout="wide")
st.title("Suivi de l'évolution du poids")

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
st.sidebar.header("Paramètres de la moyenne mobile")
window_size = st.sidebar.slider("Taille de la fenêtre pour la moyenne mobile (jours)", 1, 30, 7)
df_filtered["Poids_rolling_mean"] = df_filtered["Poids (Kgs)"].rolling(window=window_size).mean()

# Entrer les objectifs de poids
st.sidebar.header("Objectifs de poids")
target_weight = st.sidebar.number_input("Objectif de poids 1 (Kgs)", value=90.0)
target_weight_2 = st.sidebar.number_input("Objectif de poids 2 (Kgs)", value=87.5)
target_weight_3 = st.sidebar.number_input("Objectif de poids 3 (Kgs)", value=85.0)

# Diviser l'application en onglets
tab1, tab2, tab3, tab4 = st.tabs(["Résumé", "Graphiques", "Prévisions", "Analyse des Données"])

with tab1:
    st.header("Résumé")
    st.write("Voici un résumé de vos progrès vers les objectifs de poids.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Poids actuel", f"{df_filtered['Poids (Kgs)'].iloc[-1]:.2f} Kgs")
    col2.metric("Objectif de poids", f"{target_weight} Kgs")
    col3.metric("Progrès vers l'objectif", f"{100 * (df_filtered['Poids (Kgs)'].iloc[-1] / target_weight):.2f} %")
    st.write("Statistiques des poids :")
    st.write(df_filtered["Poids (Kgs)"].describe())

with tab2:
    st.header("Graphiques")
    # Graphique de l'évolution du poids avec la moyenne mobile
    fig = px.line(df_filtered, x="Date", y="Poids (Kgs)", markers=True, labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
    fig.add_scatter(x=df_filtered["Date"], y=df_filtered["Poids_rolling_mean"], mode="lines", name="Moyenne mobile")
    fig.update_layout(title="Evolution du poids")
    fig.add_hline(y=target_weight, line_dash="dash", annotation_text="Objectif 1", annotation_position="bottom right")
    fig.add_hline(y=target_weight_2, line_dash="dash", line_color="red", annotation_text="Objectif 2", annotation_position="bottom right")
    fig.add_hline(y=target_weight_3, line_dash="dash", line_color="green", annotation_text="Objectif 3", annotation_position="bottom right")
    st.plotly_chart(fig)

    # Histogramme de la distribution des poids
    fig2 = px.histogram(df_filtered, x="Poids (Kgs)", nbins=30, title="Distribution des poids")
    st.plotly_chart(fig2)

    # Graphique des anomalies détectées
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(df_filtered[["Poids (Kgs)"]])
    df_filtered["Anomalies"] = anomalies
    fig3 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", color="Anomalies", color_discrete_sequence=["blue", "red"], labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date", "Anomalies": "Anomalies"})
    fig3.update_layout(title="Evolution du poids avec détection des anomalies")
    st.plotly_chart(fig3)
    st.write("Points de données inhabituels détectés :")
    st.write(df_filtered[df_filtered["Anomalies"] == -1])

with tab3:
    st.header("Prévisions")
    # Prédiction de la date d'atteinte de l'objectif de poids avec régression linéaire
    df_filtered["Date_numeric"] = (df_filtered["Date"] - df_filtered["Date"].min()) / np.timedelta64(1, "D")
    X = df_filtered[["Date_numeric"]]
    y = df_filtered["Poids (Kgs)"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    predictions = reg.predict(X)
    y_test_pred = reg.predict(X_test)

    fig4 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
    fig4.add_traces(px.line(df_filtered, x="Date", y=predictions, labels={"y": "Régression linéaire"}).data[0])
    fig4.update_layout(title="Régression linéaire de l'évolution du poids")
    st.plotly_chart(fig4)

    days_to_target = int((target_weight - reg.intercept_) / reg.coef_[0])
    target_date = df_filtered["Date"].min() + pd.to_timedelta(days_to_target, unit="D")
    st.write(f"Date estimée pour atteindre l'objectif de poids : {target_date.date()}")

    # Calculer le taux de changement moyen du poids
    df_filtered["Poids_diff"] = df_filtered["Poids (Kgs)"].diff()
    mean_change_rate = df_filtered["Poids_diff"].mean()
    st.write(f"Taux de changement moyen du poids : {mean_change_rate:.2f} Kgs par jour")

    # Calcul des calories nécessaires
    st.subheader("Calcul des calories")
    age = st.number_input("Âge (années)", value=26)
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
    days_to_target = 90

    calories_needed = bmr * activity_levels[activity_level]
    calories_needed_per_day = calories_needed + (weight_difference * 7700) / days_to_target

    st.markdown(
        f'<p style="color: red; font-weight: bold; font-style: italic;">'
        f'Calories nécessaires à consommer par jour pour atteindre l\'objectif de poids en {days_to_target} jours : {calories_needed_per_day:.0f} kcal</p>',
        unsafe_allow_html=True
    )

with tab4:
    st.header("Analyse des Données")
    st.subheader("Analyse de tendance saisonnière")
    stl = STL(df_filtered['Poids (Kgs)'], period=7)
    res = stl.fit()
    fig5 = res.plot()
    st.pyplot(fig5)

    df_filtered["Trend"] = res.trend
    df_filtered["Seasonal"] = res.seasonal

    # Visualisation de la tendance et de la saisonnalité
    fig6 = px.line(df_filtered, x="Date", y="Trend", labels={"Trend": "Tendance", "Date": "Date"})
    fig6.update_layout(title="Tendance de l'évolution du poids")
    st.plotly_chart(fig6)

    fig7 = px.line(df_filtered, x="Date", y="Seasonal", labels={"Seasonal": "Saisonnalité", "Date": "Date"})
    fig7.update_layout(title="Saisonnalité de l'évolution du poids")
    st.plotly_chart(fig7)

    # Prédictions avec SARIMA
    st.subheader("Prédictions avec SARIMA")
    sarima_model = SARIMAX(df_filtered["Poids (Kgs)"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_results = sarima_model.fit(disp=False)
    sarima_predictions = sarima_results.predict(start=0, end=len(df_filtered)-1)
    df_filtered["SARIMA_Predictions"] = sarima_predictions

    fig8 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
    fig8.add_traces(px.line(df_filtered, x="Date", y=sarima_predictions, labels={"y": "Prédictions SARIMA"}).data[0])
    fig8.update_layout(title="Prédictions avec le modèle SARIMA")
    st.plotly_chart(fig8)

    # Comparaison des modèles de régression
    st.subheader("Comparaison des modèles de régression")
    from sklearn.ensemble import RandomForestRegressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)

    lin_reg_r2 = r2_score(y, predictions)
    rf_reg_r2 = r2_score(y_test, rf_reg.predict(X_test))

    st.write(f"Score R2 pour la régression linéaire : {lin_reg_r2:.2f}")
    st.write(f"Score R2 pour la régression Random Forest : {rf_reg_r2:.2f}")

    # Importance des caractéristiques pour le modèle Random Forest
    st.write("Importance des caractéristiques pour le modèle Random Forest :")
    st.write(rf_reg.feature_importances_)

    # Tracez le modèle de régression linéaire en superposant les données d'entraînement et de test
    fig9 = px.scatter(df_filtered, x="Date", y="Poids (Kgs)", labels={"Poids (Kgs)": "Poids (Kgs)", "Date": "Date"})
    fig9.add_traces(px.line(df_filtered, x="Date", y=predictions, labels={"y": "Régression linéaire"}).data[0])
    fig9.add_scatter(x=df_filtered.iloc[X_test.index]["Date"], y=y_test_pred, mode="markers", name="Prédictions sur ensemble de test")
    fig9.update_layout(title="Régression linéaire avec prédictions sur ensemble de test")
    st.plotly_chart(fig9)

    # Calculer le coefficient de corrélation de Pearson
    correlation = df_filtered['Date_numeric'].corr(df_filtered['Poids (Kgs)'])
    st.write(f"Coefficient de corrélation de Pearson entre le temps et le poids : {correlation:.2f}")

    # Analyse de sensibilité avec permutation_importance
    from sklearn.inspection import permutation_importance

    # Mapper le sexe en valeur numérique
    def map_sex(sex):
        return 0 if sex == "Homme" else 1

    # Ajouter des variables d'entrée au DataFrame
    df_filtered["Age"] = age
    df_filtered["Height"] = height
    df_filtered["Sex"] = map_sex(sex)
    df_filtered["Activity_Level"] = activity_levels[activity_level]

    # Entraîner un modèle de régression avec les nouvelles variables
    X_with_features = df_filtered[["Date_numeric", "Age", "Height", "Sex", "Activity_Level"]]
    y_with_features = df_filtered["Poids (Kgs)"]
    reg_with_features = LinearRegression().fit(X_with_features, y_with_features)

    # Analyse de sensibilité avec permutation_importance
    result = permutation_importance(reg_with_features, X_with_features, y_with_features, n_repeats=10, random_state=42)

    # Afficher l'importance des différentes variables
    st.write("Importance des différentes variables d'entrée :")
    for i in result.importances_mean.argsort()[::-1]:
        if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
            st.write(f"{X_with_features.columns[i]}: {result.importances_mean[i]:.3f} +/- {result.importances_std[i]:.3f}")
