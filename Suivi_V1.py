import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

# Configuration de la page Streamlit
st.set_page_config(page_title="Suivi de Poids", layout="wide")

# CSS personnalisé pour un léger ajustement du style (par exemple couleur de fond plus claire)
st.markdown("""
<style>
.stApp {
    background-color: #F8F9FA;
}
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("📊 Suivi de Poids Intelligent")

# Sidebar - Profil de l'utilisateur
st.sidebar.header("Profil de l'utilisateur")
height = st.sidebar.number_input("Taille (cm)", min_value=50, max_value=250, value=170)
age = st.sidebar.number_input("Âge", min_value=0, max_value=120, value=30)
sex = st.sidebar.selectbox("Sexe", options=["Homme", "Femme"])

# Fonction pour générer des données d'exemple si l'utilisateur n'en a pas fourni
def generate_sample_data():
    start_date = pd.to_datetime("2023-01-01")
    days = pd.date_range(start_date, periods=60)
    weights = []
    weight = 80.0  # poids initial en kg
    # Simulation d'un cycle de perte/prise/perte de poids
    for i, day in enumerate(days):
        if i < 20:
            # Perte de poids progressive
            weight += np.random.normal(-0.05, 0.1)
        elif i < 40:
            # Prise de poids
            weight += np.random.normal(0.1, 0.1)
        else:
            # Nouvelle perte de poids
            weight += np.random.normal(-0.1, 0.1)
        weights.append(round(weight, 1))
    # Générer des données de calories et de pas cohérentes avec le cycle de poids
    calories = []
    steps = []
    for i, w in enumerate(weights):
        if i < 20:
            cal = int(np.random.normal(1800, 100))  # en déficit
            step = int(np.random.normal(10000, 1000))
        elif i < 40:
            cal = int(np.random.normal(2500, 150))  # surplus calorique
            step = int(np.random.normal(5000, 500))
        else:
            cal = int(np.random.normal(2000, 150))  # maintenance modérée
            step = int(np.random.normal(8000, 800))
        calories.append(max(cal, 1200))
        steps.append(max(step, 0))
    df_sample = pd.DataFrame({
        "Date": days, 
        "Poids": weights, 
        "Calories": calories, 
        "Steps": steps
    })
    return df_sample

# Initialisation ou chargement des données
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_current' not in st.session_state:
    st.session_state['df_current'] = None

# Chargement d'un fichier CSV si l'utilisateur en importe un
uploaded_file = st.file_uploader("Importer vos données de poids (CSV)", type="csv")
if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    # On s'assure que la colonne Date est bien de type date
    if 'Date' in new_df.columns:
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        new_df.sort_values('Date', inplace=True)
    st.session_state['df_original'] = new_df.copy()
    st.session_state['df_current'] = new_df.copy()
elif st.session_state['df_current'] is None:
    # Pas de fichier importé : on utilise des données d'exemple par défaut
    sample_df = generate_sample_data()
    st.session_state['df_original'] = sample_df.copy()
    st.session_state['df_current'] = sample_df.copy()

# Récupération du DataFrame actuel
df = st.session_state['df_current']

# Création des onglets de l'application
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Données", "Analyse Statistique", "Prévisions", 
    "Modèles ML", "Clustering", "Aide"
])

# --- Section 1: Données (saisie et édition) ---
with tab1:
    st.subheader("📂 Données de poids")
    st.write("Vous pouvez ajouter, modifier ou supprimer des entrées dans le tableau ci-dessous:")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="data_editor")
    # Mise à jour des données courantes avec les modifications
    st.session_state['df_current'] = edited_df.copy()
    df = st.session_state['df_current']
    # Boutons d'action pour annuler ou comparer les modifications
    col1, col2, col3 = st.columns(3)
    if col1.button("💾 Télécharger CSV mis à jour"):
        # Export des données actuelles en CSV
        csv = df.to_csv(index=False).encode('utf-8')
        col1.download_button("Télécharger", data=csv, file_name="suivi_poids.csv", mime="text/csv")
    if col2.button("↩️ Annuler les modifications"):
        # Restaurer les données originales
        st.session_state['df_current'] = st.session_state['df_original'].copy()
        st.experimental_rerun()
    if col3.button("🔍 Comparer avant/après"):
        original = st.session_state['df_original'].reset_index(drop=True)
        current = st.session_state['df_current'].reset_index(drop=True)
        diff = current.compare(original)
        if diff.empty:
            st.info("Aucune modification par rapport aux données initiales.")
        else:
            st.write("Différences entre les données initiales et les données actuelles :")
            st.dataframe(diff)
    # Affichage d'un graphique de la série de poids
    st.line_chart(df.set_index('Date')['Poids'], use_container_width=True)

# --- Section 2: Analyse Statistique ---
with tab2:
    st.subheader("📊 Analyse Statistique")
    # Statistiques descriptives de base
    st.markdown("**Statistiques descriptives (Poids)**")
    st.write(df['Poids'].describe())
    # Histogramme de la distribution du poids
    st.markdown("**Distribution du poids (Histogramme)**")
    hist_chart = alt.Chart(df).mark_bar(color='#4E79A7').encode(
        alt.X('Poids', bin=alt.Bin(maxbins=20), title='Poids (kg)'),
        alt.Y('count()', title='Fréquence')
    )
    st.altair_chart(hist_chart, use_container_width=True)
    # Boîte à moustaches (box plot) pour visualiser les quartiles et outliers
    st.markdown("**Boîte à moustaches (quartiles et outliers)**")
    box_chart = alt.Chart(df).mark_boxplot(color='#59A14F').encode(y=alt.Y('Poids', title='Poids (kg)'))
    st.altair_chart(box_chart, use_container_width=True)
    # Détection des outliers via la méthode IQR
    Q1, Q3 = df['Poids'].quantile(0.25), df['Poids'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers_df = df[(df['Poids'] < lower_bound) | (df['Poids'] > upper_bound)]
    if not outliers_df.empty:
        st.warning(f"Outliers détectés (en dehors de [{lower_bound:.1f} kg, {upper_bound:.1f} kg]) :")
        st.dataframe(outliers_df)
    else:
        st.info("Aucun outlier détecté selon la méthode IQR.")
    # Matrice de corrélation si au moins une autre variable numérique est présente en plus du poids
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        st.markdown("**Matrice de corrélation**")
        corr_matrix = df[numeric_cols].corr()
        # Affichage de la matrice de corrélation sous forme de heatmap
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    # Indicateurs IMC et TMB
    st.markdown("**Indicateurs de santé**")
    if height is not None and height > 0:
        # Calcul de l'IMC avec la dernière valeur de poids connue
        latest_weight = df['Poids'].iloc[-1]
        height_m = height / 100.0
        bmi = latest_weight / (height_m ** 2)
        st.write(f"**IMC actuel :** {bmi:.1f}")
        # Classification OMS de l'IMC
        if bmi < 18.5:
            st.write("Catégorie IMC : **Insuffisance pondérale** (maigreur)")
        elif bmi < 25:
            st.write("Catégorie IMC : **Corpulence normale**")
        elif bmi < 30:
            st.write("Catégorie IMC : **Surpoids**")
        elif bmi < 35:
            st.write("Catégorie IMC : **Obésité I**")
        elif bmi < 40:
            st.write("Catégorie IMC : **Obésité II**")
        else:
            st.write("Catégorie IMC : **Obésité III**")
        # Calcul du TMB (métabolisme basal) via formule de Mifflin-St Jeor
        if sex == "Homme":
            bmr = 10 * latest_weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * latest_weight + 6.25 * height - 5 * age - 161
        st.write(f"**Taux Métabolique de Base estimé :** {bmr:.0f} kcal/jour")
        # Calcul de la balance énergétique du dernier jour si les calories sont renseignées
        if 'Calories' in df.columns:
            last_cal = df['Calories'].iloc[-1]
            # Estimation simplifiée des calories dépensées (BMR * 1.2 + calories brûlées par les pas)
            cal_burn = bmr * 1.2
            if 'Steps' in df.columns:
                cal_burn += df['Steps'].iloc[-1] * 0.04  # approx 0.04 kcal par pas
            balance = last_cal - cal_burn
            st.write(f"**Balance énergétique du dernier jour :** {balance:+.0f} kcal")
    else:
        st.info("Veuillez renseigner votre taille, âge et sexe dans la barre latérale pour calculer l'IMC et le TMB.")

# --- Section 3: Prévisions de poids ---
with tab3:
    st.subheader("🔮 Prévisions de poids")
    st.write("Choisissez un modèle de prévision et l'horizon désiré :")
    # Sélection du modèle de prévision et de l'horizon
    model_choice = st.selectbox("Modèle", ["ARIMA", "Prophet", "LSTM"])
    horizon = st.slider("Horizon (jours)", min_value=1, max_value=30, value=7)
    future_dates = pd.date_range(df['Date'].max() + timedelta(days=1), periods=horizon, freq='D')
    # Prévision avec ARIMA
    if model_choice == "ARIMA":
        try:
            from pmdarima import auto_arima
            model_arima = auto_arima(df['Poids'], seasonal=False)
            forecast = model_arima.predict(n_periods=horizon)
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement ARIMA: {e}")
            forecast = [None] * horizon
        pred_df = pd.DataFrame({"Date": future_dates, "Prévision": forecast})
        st.line_chart(pred_df.set_index('Date')['Prévision'], use_container_width=True)
        st.dataframe(pred_df.reset_index(drop=True))
    # Prévision avec Prophet
    elif model_choice == "Prophet":
        try:
            from prophet import Prophet
            prophet_df = df[['Date', 'Poids']].rename(columns={'Date': 'ds', 'Poids': 'y'})
            model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
            model_prophet.fit(prophet_df)
            future = model_prophet.make_future_dataframe(periods=horizon, freq='D')
            forecast = model_prophet.predict(future)
            forecast_future = forecast[['ds', 'yhat']].tail(horizon).rename(columns={'ds': 'Date', 'yhat': 'Prévision'})
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement Prophet: {e}")
            forecast_future = pd.DataFrame({"Date": future_dates, "Prévision": [None]*horizon})
        st.line_chart(forecast_future.set_index('Date')['Prévision'], use_container_width=True)
        st.dataframe(forecast_future.reset_index(drop=True))
    # Prévision avec LSTM
    else:
        import tensorflow as tf
        from tensorflow import keras
        # Préparation des données pour LSTM (série normalisée entre 0 et 1)
        data = df['Poids'].values.astype(np.float32)
        # Normalisation min-max
        min_val, max_val = data.min(), data.max()
        data_norm = (data - min_val) / (max_val - min_val) if max_val > min_val else data
        seq_len = 5  # utilise les 5 derniers jours pour prédire le suivant
        X_train, y_train = [], []
        for i in range(len(data_norm) - seq_len):
            X_train.append(data_norm[i:i+seq_len])
            y_train.append(data_norm[i+seq_len])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        # Construction du modèle LSTM simple
        model_lstm = keras.Sequential([
            keras.layers.LSTM(50, input_shape=(seq_len, 1)),
            keras.layers.Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        # Entraînement du modèle (epochs réduit si peu de données)
        model_lstm.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        # Préparation de la séquence initiale pour prédire en boucle
        last_seq = data_norm[-seq_len:]
        preds = []
        seq = last_seq.copy()
        for i in range(horizon):
            x_input = seq.reshape(1, seq_len, 1)
            pred_norm = model_lstm.predict(x_input, verbose=0)
            # Ajouter la prédiction et faire glisser la séquence
            seq = np.append(seq[1:], pred_norm)
            preds.append(float(pred_norm))
        # Reconversion des prédictions à l'échelle originale
        preds = [(p * (max_val - min_val) + min_val) for p in preds]
        pred_df = pd.DataFrame({"Date": future_dates, "Prévision": np.round(preds, 2)})
        st.line_chart(pred_df.set_index('Date')['Prévision'], use_container_width=True)
        st.dataframe(pred_df.reset_index(drop=True))

# --- Section 4: Modèles supervisés (XGBoost + SHAP) ---
with tab4:
    st.subheader("🤖 Modèles ML supervisés & Interprétabilité")
    st.write("Entraînement d’un modèle XGBoost pour prédire le poids et explication des facteurs (SHAP) :")
    # Construction d'un jeu de données supervisé à partir de la série (features = poids des jours précédents + autres variables)
    window = 3  # nombre de jours précédents utilisés comme features
    features_list = []
    target_list = []
    df_ml = df.reset_index(drop=True)
    for i in range(window, len(df_ml)):
        # Features: poids des 3 jours précédents
        feat = {f'Poids_j-{j}': df_ml.loc[i-j, 'Poids'] for j in range(1, window+1)}
        # On peut ajouter d'autres variables des jours précédents comme les calories ou pas
        if 'Calories' in df_ml.columns:
            feat['Calories_j-1'] = df_ml.loc[i-1, 'Calories']
        if 'Steps' in df_ml.columns:
            feat['Steps_j-1'] = df_ml.loc[i-1, 'Steps']
        features_list.append(feat)
        target_list.append(df_ml.loc[i, 'Poids'])
    if len(features_list) < 5:
        st.info("Pas assez de données pour entraîner un modèle supervisé.")
    else:
        X = pd.DataFrame(features_list)
        y = np.array(target_list)
        # Séparation entraînement/test (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        # Entraînement du modèle XGBoost
        import xgboost as xgb
        model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model_xgb.fit(X_train, y_train)
        # Prédiction sur le test et évaluation de l'erreur
        preds = model_xgb.predict(X_test)
        rmse = np.sqrt(np.mean((preds - y_test) ** 2))
        st.write(f"**Erreur RMSE sur un échantillon de test :** {rmse:.2f} kg")
        # Importance des caractéristiques via SHAP
        import shap
        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(X_train)
        st.write("**Importance des facteurs selon SHAP :**")
        # Graphique de synthèse SHAP (barres)
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        # (En alternative, on affiche un tableau des importances moyennes)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        imp_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance_SHAP': feature_importance
        }).sort_values('Importance_SHAP', ascending=False)
        st.dataframe(imp_df.reset_index(drop=True))

# --- Section 5: Clustering des cycles de poids ---
with tab5:
    st.subheader("📈 Clustering des cycles de poids")
    st.write("Les données sont regroupées par semaine pour identifier des cycles de perte/stabilité/prise de poids :")
    # Préparation des données hebdomadaires
    df_week = df.copy()
    df_week['Semaine'] = df_week['Date'].dt.isocalendar().week
    df_week['Année'] = df_week['Date'].dt.year
    weekly = df_week.groupby(['Année', 'Semaine'], as_index=False).agg(
        poids_moy=('Poids', 'mean'),
        variation=('Poids', lambda x: x.iloc[-1] - x.iloc[0]),
        ecart_type=('Poids', 'std')
    )
    weekly = weekly.dropna()  # supprime les semaines incomplètes (écart-type NaN si une seule mesure)
    if weekly.empty:
        st.info("Pas assez de données pour effectuer un clustering hebdomadaire.")
    else:
        # Sélection du nombre de clusters k
        k = st.slider("Nombre de clusters", min_value=2, max_value=5, value=3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
        weekly['cluster'] = kmeans.fit_predict(weekly[['variation', 'ecart_type']])
        # Analyse des clusters formés
        cluster_info = weekly.groupby('cluster').agg(
            variation_moy=('variation', 'mean'),
            ecart_type_moy=('ecart_type', 'mean'),
            effectif=('cluster', 'count')
        ).reset_index()
        # Attribution d'un label descriptif à chaque cluster en fonction de la variation moyenne
        labels = {}
        for _, row in cluster_info.iterrows():
            cid = int(row['cluster'])
            if row['variation_moy'] > 0.2:
                labels[cid] = "Prise de poids"
            elif row['variation_moy'] < -0.2:
                labels[cid] = "Perte de poids"
            else:
                labels[cid] = "Stable"
        weekly['Type_semaine'] = weekly['cluster'].map(labels)
        st.write("**Résumé des clusters :**")
        st.dataframe(cluster_info.style.format({"variation_moy": "{:.2f}", "ecart_type_moy": "{:.2f}"}))
        # Graphique de dispersion variation vs écart-type, coloré par cluster
        scatter = alt.Chart(weekly).mark_circle(size=60).encode(
            x=alt.X('variation', title='Variation hebdomadaire (kg)'),
            y=alt.Y('ecart_type', title='Écart-type (kg)'),
            color=alt.Color('Type_semaine', title='Cluster'),
            tooltip=['Année', 'Semaine', 'variation', 'ecart_type', 'Type_semaine']
        )
        st.altair_chart(scatter, use_container_width=True)
        # Description textuelle de chaque cluster
        for cid, group in cluster_info.iterrows():
            st.write(f"Cluster {group['cluster']} – {labels[int(group['cluster'])]} : variation moyenne = {group['variation_moy']:.2f} kg, écart-type moyen = {group['ecart_type_moy']:.2f} kg.")

# --- Section 6: Aide et Documentation ---
with tab6:
    st.subheader("❓ Aide & Documentation")
    st.markdown(
        "**Bienvenue dans l'application de suivi de poids avancée !** Voici un guide rapide :\n"
        "- **Données :** Importez un fichier CSV ou modifiez le tableau pour renseigner vos pesées (une date et un poids, éventuellement des colonnes Calories et Steps). Les modifications sont prises en compte immédiatement. Vous pouvez annuler des changements ou exporter le CSV mis à jour.\n"
        "- **Analyse Statistique :** Visualisez la distribution de votre poids et repérez des anomalies. Les indicateurs de santé comme l'IMC sont calculés en temps réel. *(L'IMC = poids(kg)/taille(m)^2)*:contentReference[oaicite:13]{index=13}. Un IMC normal se situe entre 18,5 et 24,9 selon l'OMS:contentReference[oaicite:14]{index=14}.\n"
        "- **Prévisions :** Trois modèles sont proposés pour prédire l'évolution de votre poids : ARIMA (modèle autorégressif classique):contentReference[oaicite:15]{index=15}, Prophet (outil de Facebook adapté aux tendances saisonnières):contentReference[oaicite:16]{index=16}, et LSTM (réseau de neurones récurrent apprenant les séquences):contentReference[oaicite:17]{index=17}. Sélectionnez un modèle et un horizon pour voir les projections futures.\n"
        "- **Modèles ML :** Entraînez un modèle XGBoost (gradient boosting) sur vos données. XGBoost est une technique avancée de forêts d'arbres décisionnels boostés:contentReference[oaicite:18]{index=18} intégrant de la régularisation pour améliorer la précision des prédictions tout en évitant l'overfitting:contentReference[oaicite:19]{index=19}. L’application explique ensuite l'importance de chaque facteur sur le poids via les valeurs SHAP (issues de la théorie des jeux de Shapley):contentReference[oaicite:20]{index=20}, pour vous aider à comprendre **pourquoi** le modèle prédit tel ou tel poids.\n"
        "- **Clustering :** Cette section regroupe vos semaines de suivi par similarité. Par exemple, vous pourrez distinguer vos semaines de **perte de poids** de celles de **prise de poids** grâce au clustering KMeans.\n\n"
        "*(Sources : [OMS – Obésité et surpoids](https://www.who.int/fr/news-room/fact-sheets/detail/obesity-and-overweight), [WHO Healthy Lifestyle](https://www.who.int/europe/news-room/fact-sheets/item/a-healthy-lifestyle---who-recommendations).)*",
        unsafe_allow_html=True
    )
