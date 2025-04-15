import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configurer la page Streamlit
st.set_page_config(page_title="Suivi de Poids Intelligent", page_icon="⚖️", layout="wide")

# Titre de l'application
st.title("⚖️ Tableau de bord de Suivi du Poids")

# Bouton d'information (ouvre un modal)
if st.button("ℹ️ À propos de l'application"):
    with st.modal("À propos de l'application"):
        st.write("""Cette application de suivi du poids offre des analyses avancées pour vous aider 
        à comprendre votre progression:
- **Indicateurs dynamiques**: variations de poids sur 7 et 30 jours, perte totale, IMC, etc.
- **Visualisations interactives**: courbe de poids avec tendance lissée, annotations (événements, plateaux), heatmap hebdomadaire.
- **Analyses avancées**: détection automatique des **plateaux** (stagnations >10 jours), segmentation des **phases** de perte/reprise de poids avec vitesse par segment.
- **Prédictions IA**: projection de poids future avec intervalle de confiance.
- **Interface améliorée**: indicateurs visuels, jauges de progression vers l'objectif, notifications dynamiques (plateau détecté, objectif atteint, etc.).""")

# Paramètres utilisateur
with st.expander("⚙️ Paramètres utilisateur"):
    # Entrée de la taille pour le calcul de l'IMC
    height_cm = st.number_input("Taille (cm)", min_value=50, max_value=250, value=170)
    height_m = height_cm / 100.0
    # Entrée du poids cible (objectif)
    target_weight = st.number_input("Objectif de poids (kg)", min_value=1.0, max_value=500.0, value=70.0)

# Chargement des données de poids
# L'utilisateur peut fournir un fichier CSV avec des colonnes "Date" et "Poids"
data_file = st.file_uploader("📄 Importer des données (CSV avec Date et Poids)", type=["csv"])
if data_file:
    df = pd.read_csv(data_file)
    # S'assurer que les colonnes sont bien nommées
    if 'Date' in df.columns and 'Poids' in df.columns:
        # Convertir la colonne Date en type datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        st.error("Le fichier CSV doit contenir des colonnes 'Date' et 'Poids'. Chargement des données par défaut...")
        data_file = None

# Si pas de fichier fourni, on génère des données d'exemple (autonome)
if not data_file:
    # Générer une série de poids simulée (exemple) sur ~6 mois
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=180)
    # Simulation d'un scénario avec phases: perte rapide initiale, plateau, perte lente, reprise, etc.
    weight_values = []
    w = 100.0  # poids initial fictif
    np.random.seed(1)
    for i, date in enumerate(dates):
        if i < 60:       # 60 jours de perte rapide
            trend = -0.1  # -100 g/jour
        elif i < 80:     # 20 jours de plateau
            trend = 0.0
        elif i < 130:    # 50 jours de perte lente
            trend = -0.05
        elif i < 150:    # 20 jours de reprise de poids
            trend = 0.1
        else:            # 30 jours de reprise de la perte
            trend = -0.1
        # Ajouter une fluctuation aléatoire autour de la tendance
        w += trend + np.random.normal(scale=0.1)
        weight_values.append(w)
    df = pd.DataFrame({"Date": dates, "Poids": weight_values})
    # Définir le poids initial et cible fictifs pour correspondre aux données simulées
    if 'Poids' in df.columns:
        # Poids initial = premier poids, ajuster objectif si pas défini
        start_weight = df['Poids'].iloc[0]
        if target_weight == 70.0:  # si valeur par défaut non modifiée
            target_weight = round(start_weight - 15.0, 1)  # objectif = 15kg de moins, par ex.
    st.info("Aucune donnée fournie, utilisation d'un jeu de données simulé pour démonstration.")

# S'assurer que les données sont présentes
if df.empty:
    st.error("Pas de données de poids disponibles.")
    st.stop()

# Calculer des statistiques globales
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)
dates = df['Date']
weights = df['Poids']

current_weight = weights.iloc[-1]
initial_weight = weights.iloc[0]
# Calcul de l'IMC actuel
bmi = None
if height_m > 0:
    bmi = current_weight / (height_m**2)
# Variations sur 7 et 30 jours
delta_7 = None
delta_30 = None
if len(weights) >= 2:
    # Variation sur 7 jours (si données disponibles)
    date_7 = dates.iloc[-1] - pd.Timedelta(days=7)
    past_7 = df[df['Date'] <= date_7]
    if not past_7.empty:
        weight_7 = past_7.iloc[-1]['Poids']
        delta_7 = current_weight - weight_7
    # Variation sur 30 jours
    date_30 = dates.iloc[-1] - pd.Timedelta(days=30)
    past_30 = df[df['Date'] <= date_30]
    if not past_30.empty:
        weight_30 = past_30.iloc[-1]['Poids']
        delta_30 = current_weight - weight_30
# Perte totale depuis le début
total_change = current_weight - initial_weight  # négatif si perte de poids

# Préparer les indicateurs pour affichage
col1, col2, col3, col4 = st.columns(4)
# Poids actuel
col1.metric("Poids actuel", f"{current_weight:.1f} kg")
# IMC actuel
if bmi:
    col2.metric("IMC", f"{bmi:.1f}", help="Indice de Masse Corporelle actuel")
else:
    col2.metric("IMC", "N/A")
# Variation 7 derniers jours
if delta_7 is not None:
    delta_str = f"{delta_7:+.1f} kg"
    col3.metric("Sur 7 jours", f"{delta_7:.1f} kg", delta=delta_str, delta_color="inverse")
else:
    col3.metric("Sur 7 jours", "N/A")
# Variation 30 derniers jours
if delta_30 is not None:
    delta_str = f"{delta_30:+.1f} kg"
    col4.metric("Sur 30 jours", f"{delta_30:.1f} kg", delta=delta_str, delta_color="inverse")
else:
    col4.metric("Sur 30 jours", "N/A")

col5, col6, col7, col8 = st.columns(4)
# Perte de poids totale depuis le début
if total_change < 0:
    col5.metric("Perte totale", f"{-total_change:.1f} kg")  # afficher en positif la perte
else:
    col5.metric("Gain total", f"{total_change:.1f} kg")
# Progression vers l'objectif
if target_weight:
    if initial_weight == target_weight:
        progress_pct = 100
    elif initial_weight > target_weight:
        # objectif de perte de poids
        total_to_lose = initial_weight - target_weight
        already_lost = initial_weight - current_weight
        progress_pct = min(max(already_lost / total_to_lose * 100, 0), 100)
    else:
        # objectif de prise de poids
        total_to_gain = target_weight - initial_weight
        already_gained = current_weight - initial_weight
        progress_pct = min(max(already_gained / total_to_gain * 100, 0), 100)
    remaining = abs(current_weight - target_weight)
    col6.metric("Objectif atteint", f"{progress_pct:.0f}%", f"Reste {remaining:.1f} kg", delta_color="off")
else:
    col6.metric("Objectif", "N/A")
# Vitesse de tendance sur 30 jours (kg/semaine)
if delta_30 is not None:
    # pente glissante (approximation sur 30 derniers jours)
    weekly_rate = (delta_30 / 30.0) * 7.0
    col7.metric("Tendance 30j", f"{weekly_rate:+.2f} kg/sem", delta_color="inverse")
else:
    col7.metric("Tendance 30j", "N/A")

# Afficher une jauge de progression vers l'objectif
if target_weight:
    progress_value = progress_pct if 'progress_pct' in locals() else 0
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=progress_value,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4caf50"},  # barre de progression en vert
            'steps': [
                {'range': [0, progress_value], 'color': "#4caf50"},
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.8, 'value': 100}
        },
        title={'text': "Progression objectif"}
    ))
    gauge_fig.update_layout(height=220, width=220, margin=dict(t=40, b=30, l=20, r=20))
    col8.plotly_chart(gauge_fig, use_container_width=True)

# Séparer l'interface en onglets
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Vue d'ensemble", "📈 Évolution", "🔍 Analyse", "🔮 Prévisions"])

# **Tab 1: Vue d'ensemble (Dashboard)**
with tab1:
    st.subheader("Résumé")
    st.write(f"**Poids actuel** : {current_weight:.1f} kg")
    if bmi:
        st.write(f"**IMC actuel** : {bmi:.1f}")
    if delta_7 is not None:
        st.write(f"**Variation sur 7 jours** : {delta_7:+.1f} kg")
    if delta_30 is not None:
        st.write(f"**Variation sur 30 jours** : {delta_30:+.1f} kg")
    if total_change < 0:
        st.write(f"**Perte totale depuis le début** : {-total_change:.1f} kg")
    else:
        st.write(f"**Gain total depuis le début** : {total_change:.1f} kg")
    if target_weight:
        st.write(f"**Objectif** : {target_weight:.1f} kg &nbsp; (atteint à {progress_pct:.0f}%, reste {remaining:.1f} kg)")
    if delta_30 is not None:
        weekly_rate = (delta_30 / 30.0) * 7.0
        trend_text = "perte" if weekly_rate < 0 else "prise"
        st.write(f"**Tendance 30 jours** : {weekly_rate:+.2f} kg/semaine ({trend_text} de poids)")
    st.write("")

    # Afficher un message de félicitations si l'objectif est atteint
    if target_weight and ((initial_weight > target_weight and current_weight <= target_weight) or (initial_weight < target_weight and current_weight >= target_weight)):
        st.success("🎉 Félicitations, vous avez atteint votre **objectif de poids**!")
        st.balloons()
    # Afficher un message si un palier (milestone) intermédiaire est franchi
    # Définir des paliers tous les 5 kg entre départ et objectif
    milestones = []
    if target_weight:
        if initial_weight > target_weight:
            # perte: paliers de perte de 5kg
            step = 5.0
            weight_val = initial_weight - step
            while weight_val > target_weight:
                milestones.append(weight_val)
                weight_val -= step
        else:
            # gain: paliers de gain de 5kg
            step = 5.0
            weight_val = initial_weight + step
            while weight_val < target_weight:
                milestones.append(weight_val)
                weight_val += step
    # Vérifier si un palier vient d'être atteint (le dernier poids a franchi un palier)
    for m in milestones:
        # perte: franchi si poids actuel <= palier < poids précédent
        # gain: franchi si poids actuel >= palier > poids précédent
        if len(weights) >= 2:
            prev_w = weights.iloc[-2]
            if initial_weight > target_weight and current_weight <= m < prev_w:
                st.success(f"🏅 Palier de **{m} kg** atteint ! Bravo, continuez comme ça!")
                st.balloons()
            if initial_weight < target_weight and current_weight >= m > prev_w:
                st.success(f"🏅 Palier de **{m} kg** atteint ! Bravo, continuez vos efforts!")
                st.balloons()

# **Tab 2: Évolution (courbes de poids)**
with tab2:
    st.subheader("Évolution du poids dans le temps")
    # Courbe de poids réelle et tendance lissée
    fig = go.Figure()
    # Trace poids réel
    fig.add_trace(go.Scatter(x=dates, y=weights, mode='lines+markers', name='Poids mesuré',
                             marker=dict(size=4, color='#1f77b4'), line=dict(color='#1f77b4', width=2),
                             hovertemplate='%{x|%d %b %Y}: %{y:.1f} kg'))
    # Calcul de la tendance lissée (moyenne mobile sur 7 jours)
    window = 7
    smooth_weights = weights.rolling(window, center=False).mean()
    fig.add_trace(go.Scatter(x=dates, y=smooth_weights, mode='lines', name='Tendance lissée (7j)',
                             line=dict(color='#ff7f0e', width=3, dash='dash'),
                             hovertemplate='%{x|%d %b %Y}: %{y:.1f} kg (tendance)'))

    # Détection des plateaux pour annotation visuelle
    plateau_intervals = []
    vals = weights.values
    n = len(vals)
    i = 0
    while i < n - 10:
        # Fenêtre glissante de 10 jours
        if vals[i:i+11].max() - vals[i:i+11].min() < 0.5:
            j = i + 10
            # Étendre le plateau tant que la variation reste faible
            while j < n and vals[i:j+1].max() - vals[i:j+1].min() < 0.5:
                j += 1
            plateau_intervals.append((i, j-1))
            i = j
        else:
            i += 1
    # Fusionner plateaux proches (séparés par <3 jours)
    merged_plateaus = []
    for (a, b) in plateau_intervals:
        if merged_plateaus and a <= merged_plateaus[-1][1] + 3:
            prev_a, prev_b = merged_plateaus[-1]
            merged_plateaus[-1] = (prev_a, b)
        else:
            merged_plateaus.append((a, b))
    plateau_intervals = merged_plateaus
    # Ajouter des zones ombrées pour indiquer les plateaux
    for (start_idx, end_idx) in plateau_intervals:
        start_date = dates.iloc[start_idx]
        end_date = dates.iloc[end_idx]
        fig.add_vrect(x0=start_date, x1=end_date, fillcolor="orange", opacity=0.2, line_width=0,
                      annotation_text="Plateau", annotation_position="top left",
                      annotation=dict(font_size=10, font_color="orange"))
    # Détection des milestones (paliers intermédiaires) atteints pour annotations
    events = []
    # Milestones calculés plus haut
    for m in milestones:
        # trouver la première date où on passe sous (ou au-dessus) du palier
        if initial_weight > target_weight:
            crossed = df[df['Poids'] <= m]
        else:
            crossed = df[df['Poids'] >= m]
        if not crossed.empty:
            date_reached = crossed.iloc[0]['Date']
            events.append((date_reached, f"Palier {m} kg"))
    # Ajouter des annotations pour les events
    for (event_date, text) in events:
        event_weight = float(df[df['Date'] == event_date]['Poids'])
        fig.add_trace(go.Scatter(x=[event_date], y=[event_weight], mode="markers+text", name=text,
                                 marker=dict(symbol="star-diamond", size=12, color="#FFD700"),
                                 text=[text], textposition="top center",
                                 hovertemplate='%{text} le %{x|%d %b %Y}'))

    # Configurer les sélecteurs de période et le slider
    fig.update_layout(
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7j", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all", label="Tout")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(title="Poids (kg)"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap calendaire hebdomadaire
    st.subheader("Heatmap hebdomadaire des variations quotidiennes")
    # Préparer une matrice [Jour de semaine x Semaine] avec la variation de poids jour par jour
    df_diff = df.copy()
    df_diff['Diff'] = df_diff['Poids'].diff()
    df_diff.dropna(inplace=True)
    if not df_diff.empty:
        # Calculer semaine ISO et jour de semaine
        df_diff['Week'] = df_diff['Date'].dt.isocalendar().week
        df_diff['Weekday'] = df_diff['Date'].dt.weekday  # Lundi=0, Dimanche=6
        # Pour distinguer les années, combiner année et numéro de semaine
        df_diff['YearWeek'] = df_diff['Date'].dt.strftime('%Y-%W')
        # Tableau croisé: lignes = jour de semaine, colonnes = semaine
        pivot = df_diff.pivot(index='Weekday', columns='YearWeek', values='Diff')
        # Ordonner les jours de semaine de Lundi(0) à Dimanche(6)
        pivot = pivot.reindex(index=range(0, 7))
        # Libellés pour jours et semaines
        days_labels = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        week_labels = pivot.columns  # YearWeek labels
        # Créer la heatmap
        heatmap = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=week_labels,
            y=days_labels,
            colorscale="RdYlGn_r",
            zmid=0,
            colorbar=dict(title="Variation (kg)")
        ))
        heatmap.update_layout(height=300, margin=dict(t=30, b=0))
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.write("Pas assez de données pour générer la heatmap des variations quotidiennes.")

# **Tab 3: Analyse avancée**
with tab3:
    st.subheader("Analyse avancée des tendances")
    # Détection automatique des plateaux (stagnations)
    plateau_periods = []
    for (start_idx, end_idx) in plateau_intervals:
        start_date = dates.iloc[start_idx].strftime("%d %b %Y")
        end_date = dates.iloc[end_idx].strftime("%d %b %Y")
        duration = end_idx - start_idx + 1
        plateau_periods.append((start_date, end_date, duration))
    if plateau_periods:
        plateau_texts = [f"{sd} au {ed} ({dur} jours)" for sd, ed, dur in plateau_periods]
        st.info(f"**Plateaux détectés** (≈stagnation sur ≥10 jours) : " + ", ".join(plateau_texts))
        # Si le dernier plateau inclut la date actuelle
        last_plateau = plateau_periods[-1]
        last_plateau_end = pd.to_datetime(last_plateau[1], format="%d %b %Y")
        if dates.iloc[-1].strftime("%d %b %Y") == last_plateau[1]:
            st.warning(f"⚠️ Vous êtes actuellement en **plateau** depuis {last_plateau[-1]} jours.")
    else:
        st.write("Aucun plateau prolongé détecté récemment.")
    # Segmentation des phases de poids avec KMeans
    # Construire les segments en combinant les plateaux et changements de tendance
    segments = []
    seg_start = 0
    i = 0
    sign_flips = []
    # Re-calculer les flips de tendance (hors plateaux) sur la base du signe des différences
    weight_diff = weights.diff().fillna(0.0).values
    current_sign = 0
    while i < len(weight_diff):
        # Passer les périodes de plateau (on les traitera comme segments à part)
        in_plateau = False
        for (ps, pe) in plateau_intervals:
            if i == ps:
                # Ajouter segment précédent (non-plateau) si existant
                if seg_start < ps:
                    segments.append((seg_start, ps-1))
                # Ajouter le segment plateau
                segments.append((ps, pe))
                i = pe + 1
                seg_start = i
                in_plateau = True
                break
        if in_plateau:
            continue
        # Détection de changement de signe hors plateau
        s = 0
        if abs(weight_diff[i]) < 0.01:
            s = 0
        else:
            s = 1 if weight_diff[i] > 0 else -1
        if current_sign == 0:
            current_sign = s if s != 0 else current_sign
        elif s != 0 and s != current_sign:
            # changement de tendance à i-1 (fin du segment précédent)
            segments.append((seg_start, i-1))
            seg_start = i
            current_sign = s
            sign_flips.append(i-1)
        i += 1
    # Ajouter le dernier segment non-plateau restant
    if seg_start < len(weights):
        segments.append((seg_start, len(weights)-1))

    # Enlever d'éventuels segments de longueur nulle ou négative
    segments = [seg for seg in segments if seg[0] <= seg[1]]
    # Calculer métriques par segment
    seg_data = []
    for (a, b) in segments:
        start_date = dates.iloc[a].strftime("%d %b %Y")
        end_date = dates.iloc[b].strftime("%d %b %Y")
        duration = b - a + 1
        change = weights.iloc[b] - weights.iloc[a]
        # kg par semaine
        rate_per_week = (change / duration) * 7.0
        seg_data.append({
            "Début": start_date,
            "Fin": end_date,
            "Durée (jours)": duration,
            "Variation (kg)": f"{change:+.1f}",
            "Vitesse (kg/sem)": f"{rate_per_week:+.2f}",
        })
    # Clustering KMeans des segments selon la vitesse (slope moyenne)
    from sklearn.cluster import KMeans
    # Préparer les features (slope par jour)
    slopes = np.array([ (weights.iloc[b] - weights.iloc[a]) / (b - a + 1) if b > a else 0.0 for (a, b) in segments ])
    n_clusters = min(3, len(slopes)) if len(slopes) > 1 else 1
    if n_clusters >= 1:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(slopes.reshape(-1, 1))
    else:
        labels = [0] * len(slopes)
    # Définir les étiquettes lisibles pour chaque cluster
    cluster_names = {}
    if n_clusters == 1:
        cluster_names[0] = "Stable"
    elif n_clusters == 2:
        # Dans le cas de 2 clusters, on différencie généralement perte vs gain ou perte vs plateau
        center0, center1 = kmeans.cluster_centers_.flatten()
        for idx, center in enumerate([center0, center1]):
            if center > 0.02:
                cluster_names[idx] = "Reprise de poids"
            elif center < -0.02:
                cluster_names[idx] = "Perte de poids"
            else:
                cluster_names[idx] = "Stagnation"
    else:  # 3 clusters
        centers = kmeans.cluster_centers_.flatten()
        # Trier les clusters par valeur de slope
        order = np.argsort(centers)
        # plus négatif -> perte rapide, milieu -> plateau ou faible variation, plus positif -> reprise
        for idx, center in enumerate(centers):
            # Trouver son rang dans l'ordre trié
            rank = np.where(order == idx)[0][0]
            if rank == 0:
                cluster_names[idx] = "Perte rapide"
            elif rank == 1:
                cluster_names[idx] = "Stagnation" if -0.02 <= center <= 0.02 else "Perte modérée"
            elif rank == 2:
                cluster_names[idx] = "Reprise de poids"

    # Ajouter la classification au tableau des segments
    for i, seg in enumerate(seg_data):
        label = labels[i] if len(labels) > i else 0
        phase_name = cluster_names.get(label, "")
        seg_data[i]["Phase"] = phase_name

    # Afficher le tableau des segments détectés
    st.write("**Phases détectées dans la période:**")
    seg_df = pd.DataFrame(seg_data)
    st.dataframe(seg_df, use_container_width=True)

    # Messages dynamiques en fonction de la phase actuelle
    if seg_data:
        last_phase = seg_data.iloc[-1] if isinstance(seg_data, pd.DataFrame) else seg_data[-1]
        last_label = labels[-1] if isinstance(labels, np.ndarray) else labels[-1]
        last_phase_name = cluster_names.get(last_label, "")
        if "Plateau" in last_phase_name or last_phase_name == "Stagnation":
            st.warning("⚠️ Votre poids est actuellement en **plateau**. Patience, persévérez !")
        elif "Reprise" in last_phase_name:
            st.warning("⚠️ Attention, tendance à la **reprise de poids** sur la période récente.")
        elif "Perte rapide" in last_phase_name:
            st.info("🏃 Perte de poids rapide en cours. Continuez vos efforts, bravo!")
        elif "Perte" in last_phase_name:
            st.success("📉 Vous êtes en phase de **perte de poids**. Bonne progression récente!")
        else:
            st.info(f"Phase actuelle: {last_phase_name}")

# **Tab 4: Prévisions IA (Prophet)**
with tab4:
    st.subheader("Prévision de poids future")
    try:
        from prophet import Prophet
    except ImportError:
        Prophet = None
    if Prophet is None:
        st.error("Le module Prophet n'est pas installé. Veuillez exécuter `pip install prophet` pour les prévisions.")
    else:
        # Préparer les données pour Prophet
        df_prophet = df[['Date', 'Poids']].rename(columns={'Date': 'ds', 'Poids': 'y'})
        # Entraîner le modèle Prophet
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
        model.fit(df_prophet)
        # Choix de l'horizon de prévision
        horizon_days = st.slider("Horizontale de prévision (jours)", min_value=7, max_value=180, value=30)
        future = model.make_future_dataframe(periods=horizon_days, freq='D', include_history=False)
        forecast = model.predict(future)
        # Tracer la prévision
        fig_pred = go.Figure()
        # Trace historique
        fig_pred.add_trace(go.Scatter(x=dates, y=weights, mode='lines', name="Historique", line=dict(color='#1f77b4')))
        # Trace prévision (moyenne)
        fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Prévision", line=dict(color='#2ca02c', dash='dash')))
        # Intervalle de confiance
        fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'))
        fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', fillcolor='rgba(46, 204, 113, 0.2)',
                                      line=dict(color='rgba(46, 204, 113, 0)'), showlegend=False, hoverinfo='skip', name="Confiance 95%"))
        # Ligne verticale "Aujourd'hui"
        today = dates.iloc[-1]
        fig_pred.add_vline(x=today, line_width=2, line_dash="dot", line_color="grey", annotation_text="Aujourd'hui", annotation_position="top right")
        fig_pred.update_layout(yaxis_title="Poids (kg)", legend_title_text="", margin=dict(t=10, b=0))
        st.plotly_chart(fig_pred, use_container_width=True)
        # Message sur la date potentielle d'atteinte de l'objectif
        if target_weight:
            # Chercher la première date forecast où l'objectif est atteint
            if initial_weight > target_weight:
                achieved = forecast[forecast['yhat'] <= target_weight]
            else:
                achieved = forecast[forecast['yhat'] >= target_weight]
            if not achieved.empty:
                goal_date = achieved.iloc[0]['ds']
                st.success(f"🎯 D'après la prévision, l'objectif de **{target_weight:.1f} kg** pourrait être atteint vers le **{goal_date.strftime('%d %b %Y')}**.")
            else:
                st.info("Selon la tendance actuelle, l'objectif ne serait pas atteint sur l'horizon de prévision choisi.")
