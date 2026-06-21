# Suivi V1 — Application Streamlit de suivi de poids

## Présentation

Suivi V1 est une application personnelle de suivi de poids développée avec Streamlit. Elle permet de consulter l’évolution des mesures, de suivre une trajectoire cible, d’éditer un journal de données et d’explorer des analyses descriptives ainsi que des projections indicatives.

- Application déployée : <https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/>
- Repository : <https://github.com/samirelhassani1998/Suivi_V1>

## Aperçu fonctionnel

L’application fournit notamment :

- le suivi du poids dans le temps avec poids réel, variations récentes et tendance ;
- des moyennes mobiles calendaires, dont des vues sur plusieurs fenêtres ;
- une progression vers l’objectif final et vers des paliers intermédiaires ;
- une trajectoire cible officielle planifiée et plafonnée à l’objectif final ;
- un journal des mesures avec édition en session, validation et export CSV ;
- un import CSV local en complément de la source Google Sheets configurée ;
- des indicateurs de qualité des données, de régularité, de volatilité et d’anomalies ;
- une détection de plateau ou de stagnation sur des fenêtres calendaires ;
- des projections basées sur les mesures réellement observées ;
- des modèles statistiques avancés expérimentaux : SARIMAX, Auto-ARIMA et ML quantile ;
- une interface Streamlit configurée en mise en page large, utilisable sur desktop et mobile.

## Trajectoire cible officielle

La trajectoire cible officielle est une règle métier fixe, indépendante de la dernière mesure saisie :

- date de départ : **26/05/2026** ;
- poids de départ : **106,2 kg** ;
- rythme cible : **baisse de 1 kg tous les 7 jours** ;
- objectif final : **80 kg** ;
- date cible planifiée : **25/11/2026** ;
- aucune valeur de trajectoire n’existe avant le 26/05/2026 ;
- la trajectoire ne descend jamais sous 80 kg ;
- la série graphique s’arrête au point d’atteinte de 80 kg.

Formule simplifiée :

```text
poids cible = 106,2 − (jours écoulés / 7)
```

avec un plancher à **80 kg**.

## Clarification des projections

Le projet distingue trois notions qui ne doivent pas être confondues :

1. **Date cible planifiée**  
   Date issue de la trajectoire officielle fixe décrite ci-dessus. Elle ne dépend pas de la tendance observée.

2. **Projection selon les mesures**  
   Estimation calculée à partir de la tendance réellement observée dans les données disponibles. Elle peut être bloquée si le signal est trop fragile ou si la tendance ne va pas clairement vers l’objectif.

3. **Projection statistique avancée**  
   Résultat expérimental produit par des modèles statistiques ou de machine learning, par exemple SARIMAX, Auto-ARIMA ou ML quantile.

Les projections restent indicatives : elles ne constituent pas une garantie. Les projections visibles sont contraintes pour s’arrêter lorsqu’elles atteignent **80 kg** et aucune courbe visible ne doit afficher de valeur sous **80 kg**. Les sorties brutes des modèles peuvent toutefois rester utilisées en interne pour l’évaluation ou la construction des intervalles.

## Pages de l’application

La navigation actuelle expose uniquement les pages suivantes : **Dashboard**, **Journal**, **Prévisions**, **Insights** et **Paramètres**.

### Tableau de bord

Le tableau de bord présente les informations principales de suivi :

- KPI de poids actuel, variations récentes, tendance et écart à la trajectoire ou à l’objectif ;
- résumé du parcours et lecture rapide ;
- insights automatiques non alarmistes ;
- graphique d’évolution du poids réel ;
- moyennes mobiles calendaires ;
- objectifs et paliers affichés ;
- trajectoire cible officielle avec date cible planifiée ;
- analyses secondaires dans des sections, onglets ou expanders : effort actuel, vitesse, discipline, progression, volatilité, vue hebdomadaire, distribution et comparaisons.

### Journal

La page Journal permet de gérer les mesures chargées en session :

- visualisation et édition des lignes via l’éditeur Streamlit ;
- ajout ou suppression de lignes ;
- validation des colonnes obligatoires, dates, poids invalides, doublons et valeurs potentiellement aberrantes ;
- filtre de date pour aperçu ;
- export CSV ;
- conservation des colonnes personnalisées du CSV.

Les dates sont normalisées en dates pandas et les poids sont convertis en valeurs numériques. Les valeurs de poids doivent être positives pour être enregistrées sans erreur bloquante.

### Prévisions

La page Prévisions regroupe les estimations prospectives :

- projection selon les mesures et scénarios optimiste, réaliste ou pessimiste ;
- comparaison de modèles sur découpage chronologique ;
- SARIMAX ;
- Auto-ARIMA ;
- ML quantile avec intervalle P10/P90 ;
- analyses STL, ACF et PACF ;
- leaderboard de baselines en backtest walk-forward ;
- vitesses de variation sur fenêtres calendaires.

Les modèles avancés sont expérimentaux et leurs résultats doivent être interprétés avec prudence. Les courbes visibles de projection sont contraintes pour ne pas descendre sous 80 kg et pour s’arrêter à l’atteinte de cet objectif.

### Insights

La page Insights expose des analyses descriptives plus détaillées :

- qualité des données ;
- plateau et stagnation sur fenêtres calendaires ;
- scores de discipline, cohérence et volatilité ;
- segmentation en phases de perte, plateau ou reprise ;
- ruptures de tendance ;
- meilleures et pires semaines ;
- comparaison semaine/mois ;
- patterns par jour de semaine ;
- streaks ;
- anomalies robustes et clustering.

Lorsque l’application détecte une période d’effort récente après une interruption, l’analyse peut être limitée à l’effort actuel ou appliquée à l’historique complet.

### Paramètres

La page Paramètres permet de modifier les préférences stockées dans la session Streamlit :

- taille en centimètres, utilisée notamment pour l’IMC ;
- cinq paliers ou objectifs affichés ;
- type de moyenne mobile : simple ou exponentielle ;
- fenêtre de moyenne mobile, exprimée en nombre de mesures pour les fonctions de rolling classiques, tandis que le tableau de bord utilise aussi des moyennes calendaires dédiées ;
- stratégie de gestion des doublons journaliers : garder la dernière mesure, moyenne journalière ou médiane journalière ;
- modèle par défaut ;
- thème Plotly ;
- diagnostic de session.

Le champ “Poids objectif final” affiché en haut du formulaire est désactivé. En pratique, l’objectif final de session est synchronisé avec l’Objectif 5 ; l’objectif officiel de la trajectoire métier reste **80 kg**.

## Fiabilité des calculs

Plusieurs garde-fous limitent les interprétations trop optimistes :

- les pentes et vitesses sont calculées sur les jours calendaires réels, pas uniquement sur le nombre de lignes ;
- les mesures irrégulières sont prises en compte via les dates réelles ;
- certaines analyses exigent un minimum de mesures avant d’afficher un résultat ;
- l’application distingue les fenêtres en jours calendaires des fenêtres en nombre de mesures ;
- les dates sont préparées pour éviter les problèmes de mélange timezone-aware et timezone-naive ;
- la détection de plateau/stagnation passe par un moteur commun ;
- l’alignement à la trajectoire cible utilise une tolérance de **0,5 kg** ;
- les projections visibles sont arrêtées à **80 kg** ;
- les lignes manquantes, les dates invalides, les poids invalides et les valeurs aberrantes potentielles sont signalés ou ignorés selon le contexte.

Ces analyses ne sont pas des validations médicales. Elles servent à résumer les données personnelles de suivi et à produire des repères indicatifs.

## Sources et préparation des données

### Sources

L’application peut charger :

- une source Google Sheets exportée en CSV, configurée par `st.secrets["data_url"]` ou par une valeur par défaut du code ;
- un CSV local importé depuis la sidebar, qui remplace les données de la session courante.

Aucune URL privée ou secret ne doit être publié dans ce README.

### Colonnes attendues

Les colonnes obligatoires sont :

- `Date` ;
- `Poids (Kgs)`.

Des colonnes optionnelles comme notes, calories, sommeil, hydratation ou condition de mesure peuvent être conservées si elles existent dans le CSV.

### Nettoyage

Le chargement et la validation appliquent les traitements suivants :

- normalisation des noms de colonnes usuels vers `Date` et `Poids (Kgs)` ;
- conversion des dates avec priorité au format jour/mois/année ;
- conversion des poids avec virgule ou point décimal ;
- suppression des lignes sans date ou poids valide lors du nettoyage strict ;
- tri chronologique ;
- conservation des colonnes additionnelles ;
- détection des poids inférieurs ou égaux à zéro comme erreurs dans le journal ;
- détection des doublons de date et valeurs potentiellement aberrantes.

Au chargement depuis Google Sheets, les doublons de date sont résolus automatiquement en gardant la dernière mesure du jour. Dans les paramètres, l’utilisateur peut choisir une stratégie de doublons en session : garder la dernière mesure, moyenne journalière ou médiane journalière.

## Architecture du projet

```text
Suivi_V1/
├── Suivi_V1.py
├── requirements.txt
├── runtime.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.example.toml
├── app/
│   ├── auth.py
│   ├── config.py
│   ├── deploy.py
│   ├── utils.py
│   ├── core/
│   │   ├── analytics.py
│   │   ├── business.py
│   │   ├── data.py
│   │   ├── evaluation.py
│   │   ├── features.py
│   │   ├── forecasting.py
│   │   ├── formatting.py
│   │   ├── insights.py
│   │   ├── models.py
│   │   ├── plateau.py
│   │   ├── projection_constraints.py
│   │   ├── session_state.py
│   │   ├── target_trajectory.py
│   │   ├── targets.py
│   │   ├── time_utils.py
│   │   └── weight_summary.py
│   ├── pages/
│   │   ├── Dashboard.py
│   │   ├── Insights.py
│   │   ├── Journal.py
│   │   ├── Predictions.py
│   │   └── Settings.py
│   └── ui/
│       ├── components.py
│       └── theme.py
└── tests/
    ├── conftest.py
    ├── test_analytics.py
    ├── test_core_v2.py
    ├── test_phase2_reliability.py
    ├── test_streamlit_smoke.py
    ├── test_target_trajectory.py
    ├── test_utils.py
    ├── test_v3_guardrails.py
    └── test_weight_summary.py
```

### Fichiers principaux

- `Suivi_V1.py` : point d’entrée Streamlit, authentification, chargement des données, sidebar et navigation.
- `app/config.py` : configuration applicative, colonnes attendues, valeurs par défaut et stratégies de doublons.
- `app/auth.py` : protection optionnelle par mot de passe via secrets Streamlit.
- `app/core/business.py` : constantes métier officielles, dont trajectoire cible, objectif final et règles de stagnation.
- `app/core/target_trajectory.py` : construction et comparaison de la trajectoire cible officielle.
- `app/core/projection_constraints.py` : arrêt des projections visibles au plancher de 80 kg.
- `app/core/weight_summary.py` : synthèse du parcours, moyennes calendaires, variations et insights quotidiens.
- `app/core/plateau.py` : moteur commun de détection de plateau/stagnation.
- `app/core/time_utils.py` : normalisation défensive des dates.
- `app/core/formatting.py` : formatage français des dates, poids et nombres.
- `app/core/analytics.py` : fonctions analytiques descriptives, scores, phases, scénarios et tendances.
- `app/core/insights.py` : plateau, anomalies robustes et estimation d’ETA selon les mesures.
- `app/core/forecasting.py` : prévisions SARIMAX et ML quantile.
- `app/core/data.py` : nettoyage, validation, rapport qualité et résolution des doublons.
- `app/core/session_state.py` : cycle de vie des données en session Streamlit.
- `app/pages/` : pages visibles dans la navigation.
- `app/ui/` : composants visuels et thème global.
- `tests/` : tests automatisés couvrant le cœur analytique, les garde-fous et les pages Streamlit.

## Installation locale

Commandes macOS/Linux :

```bash
git clone https://github.com/samirelhassani1998/Suivi_V1.git
cd Suivi_V1
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run Suivi_V1.py
```

Activation Windows :

```powershell
.venv\Scripts\activate
```

## Configuration des secrets

Créer un fichier local non versionné :

```text
.streamlit/secrets.toml
```

à partir du template :

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Exemple avec placeholders uniquement :

```toml
data_url = "URL_D_EXPORT_CSV"

[auth]
required = true
password = "VOTRE_MOT_DE_PASSE"
```

Ne jamais committer `.streamlit/secrets.toml` ni y placer de vraie valeur dans la documentation.

## Tests et vérifications

Les tests automatisés sont basés sur `pytest` :

```bash
pytest
```

N’annoncez un résultat de test comme réussi qu’après l’avoir réellement exécuté dans l’environnement courant.

## Déploiement

Le projet est conçu pour être exécuté avec Streamlit. Le runtime Python attendu par Streamlit Cloud est indiqué dans `runtime.txt`, et les dépendances Python sont listées dans `requirements.txt`.
