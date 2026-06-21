# Suivi V1 — Base Streamlit générique de suivi de poids

## Présentation

Suivi V1 est une application Streamlit générique destinée au suivi de mesures de poids. Elle fournit une base réutilisable pour charger des données, visualiser leur évolution, comparer les mesures à une trajectoire cible configurable, gérer un journal en session et produire des analyses descriptives ainsi que des projections indicatives.

Ce repository est conçu pour être adapté par chaque utilisateur ou équipe en remplaçant la configuration, les secrets et la source de données par ses propres valeurs locales ou privées.

L’application peut être déployée sur Streamlit Community Cloud.

## Principes de confidentialité

Le README public ne doit contenir aucune donnée personnelle, aucune mesure réelle, aucune URL privée de données et aucun secret. Les données de suivi doivent rester dans une source privée ou dans des fichiers locaux non publiés.

À ne jamais documenter publiquement :

- mesures de poids réelles ;
- objectifs personnels réels ;
- dates réelles de suivi ;
- projections personnelles ;
- données de santé ;
- habitudes ou historiques individuels ;
- URL privée d’export CSV ;
- mot de passe ou secret Streamlit.

## Aperçu fonctionnel

L’application fournit notamment :

- le suivi des mesures de poids dans le temps ;
- l’affichage de variations récentes, de tendances et de moyennes mobiles ;
- une progression vers un objectif final configurable ;
- une trajectoire cible configurable et bornée par l’objectif final ;
- un journal des mesures avec édition en session, validation et export CSV ;
- un import CSV local en complément d’une source CSV distante optionnelle ;
- des indicateurs de qualité des données, de régularité, de volatilité et d’anomalies ;
- une détection de plateau ou de stagnation sur des fenêtres calendaires ;
- des projections indicatives basées sur les mesures disponibles ;
- des modèles statistiques avancés expérimentaux ;
- une interface Streamlit configurée pour une consultation desktop ou mobile.

## Trajectoire cible configurable

La trajectoire cible repose sur des paramètres métier centralisés. Elle est indépendante de la dernière mesure saisie et sert de repère de comparaison.

Les paramètres principaux sont :

- date de départ ;
- valeur initiale ;
- rythme hebdomadaire cible ;
- objectif final.

Formule abstraite :

```text
target_value = start_value - elapsed_weeks × weekly_rate
```

La trajectoire est bornée par l’objectif final et s’arrête lorsque cet objectif est atteint. Aucune valeur concrète propre à une utilisation personnelle ne doit être publiée dans cette documentation.

Exemple de configuration avec placeholders fictifs uniquement :

```python
TARGET_START_DATE = "YYYY-MM-DD"
TARGET_START_WEIGHT = 0.0
TARGET_WEEKLY_RATE = 0.0
TARGET_FINAL_WEIGHT = 0.0
```

## Clarification des projections

Le projet distingue trois notions :

1. **Trajectoire cible configurable**
   Repère déterministe issu des paramètres métier centralisés. Elle ne dépend pas de la tendance observée.

2. **Projection selon les mesures**
   Estimation calculée à partir de la tendance observée dans les données chargées. Elle peut être bloquée si le signal est trop fragile ou si la tendance ne va pas clairement vers l’objectif configuré.

3. **Projection statistique avancée**
   Résultat expérimental produit par des modèles statistiques ou de machine learning, par exemple SARIMAX, Auto-ARIMA ou ML quantile.

Les projections restent indicatives et ne constituent pas une garantie. Les courbes visibles peuvent être contraintes pour s’arrêter lorsqu’elles atteignent l’objectif final configuré.

## Pages de l’application

La navigation expose les pages suivantes : **Dashboard**, **Journal**, **Prévisions**, **Insights** et **Paramètres**.

### Dashboard

Le tableau de bord présente les informations principales de suivi :

- KPI de mesure actuelle, variations récentes, tendance et écart à la trajectoire ou à l’objectif ;
- résumé du parcours chargé en session ;
- insights automatiques non alarmistes ;
- graphique d’évolution des mesures ;
- moyennes mobiles calendaires ;
- objectifs et paliers affichés ;
- trajectoire cible configurable ;
- analyses secondaires : effort actuel, vitesse, discipline, progression, volatilité, vue hebdomadaire, distribution et comparaisons.

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

Les modèles avancés sont expérimentaux et leurs résultats doivent être interprétés avec prudence. Les courbes visibles de projection peuvent être contraintes par l’objectif final configuré.

### Insights

La page Insights expose des analyses descriptives plus détaillées :

- qualité des données ;
- plateau et stagnation sur fenêtres calendaires ;
- scores de discipline, cohérence et volatilité ;
- segmentation en phases de perte, plateau ou reprise ;
- ruptures de tendance ;
- comparaisons par semaine ou par mois ;
- patterns par jour de semaine ;
- streaks ;
- anomalies robustes et clustering.

Selon les données chargées, certaines analyses peuvent être limitées à une période récente ou appliquées à l’ensemble de l’historique disponible en session.

### Paramètres

La page Paramètres permet de modifier des préférences stockées dans la session Streamlit :

- paliers ou objectifs affichés ;
- type de moyenne mobile ;
- fenêtre de moyenne mobile ;
- stratégie de gestion des doublons journaliers ;
- modèle par défaut ;
- thème Plotly ;
- diagnostic de session.

Les valeurs affichées dans l’application dépendent de la configuration et des données chargées par l’utilisateur.

## Fiabilité des calculs

Plusieurs garde-fous limitent les interprétations trop optimistes :

- les pentes et vitesses sont calculées sur les jours calendaires réels, pas uniquement sur le nombre de lignes ;
- les mesures irrégulières sont prises en compte via les dates ;
- certaines analyses exigent un minimum de mesures avant d’afficher un résultat ;
- l’application distingue les fenêtres en jours calendaires des fenêtres en nombre de mesures ;
- les dates sont préparées pour éviter les problèmes de mélange timezone-aware et timezone-naive ;
- la détection de plateau ou de stagnation passe par un moteur commun ;
- l’alignement à la trajectoire cible utilise une tolérance configurable dans le code ;
- les projections visibles peuvent être arrêtées à l’objectif final configuré ;
- les lignes manquantes, les dates invalides, les poids invalides et les valeurs aberrantes potentielles sont signalés ou ignorés selon le contexte.

Ces analyses ne sont pas des validations médicales. Elles servent à résumer des données de suivi et à produire des repères indicatifs.

## Sources et préparation des données

### Sources

L’application peut charger :

- une source CSV distante configurée via les secrets Streamlit ou une valeur applicative ;
- un CSV local importé depuis la sidebar, qui remplace les données de la session courante.

Aucune URL privée ou valeur de secret ne doit être publiée dans ce README.

### Colonnes attendues

Les colonnes obligatoires sont :

- `Date` ;
- `Poids (Kgs)`.

Des colonnes optionnelles comme notes, calories, sommeil, hydratation ou condition de mesure peuvent être conservées si elles existent dans le CSV.

### Nettoyage

Le chargement et la validation appliquent les traitements suivants :

- normalisation des noms de colonnes usuels vers `Date` et `Poids (Kgs)` ;
- conversion des dates ;
- conversion des poids avec virgule ou point décimal ;
- suppression des lignes sans date ou poids valide lors du nettoyage strict ;
- tri chronologique ;
- conservation des colonnes additionnelles ;
- détection des poids inférieurs ou égaux à zéro comme erreurs dans le journal ;
- détection des doublons de date et valeurs potentiellement aberrantes.

Selon la stratégie configurée, les doublons de date peuvent être résolus en gardant la dernière mesure, une moyenne journalière ou une médiane journalière.

## Architecture du projet

```text
Suivi_V1/
├── Suivi_V1.py
├── README.md
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
- `app/core/business.py` : paramètres métier centralisés et règles de trajectoire.
- `app/core/target_trajectory.py` : construction et comparaison de la trajectoire cible.
- `app/core/projection_constraints.py` : contraintes appliquées aux projections visibles.
- `app/core/weight_summary.py` : synthèse du parcours, moyennes calendaires, variations et insights quotidiens.
- `app/core/plateau.py` : moteur commun de détection de plateau ou stagnation.
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

Ne jamais committer `.streamlit/secrets.toml` ni placer de vraie valeur dans la documentation.

## Tests et vérifications

Les tests automatisés sont basés sur `pytest` :

```bash
pytest
```

N’annoncez un résultat de test comme réussi qu’après l’avoir réellement exécuté dans l’environnement courant.

## Déploiement

Le projet est conçu pour être exécuté avec Streamlit. Le runtime Python attendu par Streamlit Community Cloud est indiqué dans `runtime.txt`, et les dépendances Python sont listées dans `requirements.txt`.

Pour un déploiement public ou semi-public :

- configurer les secrets dans l’interface de la plateforme de déploiement ;
- ne pas versionner de fichier contenant des secrets ;
- protéger l’accès si les données chargées sont personnelles ou sensibles ;
- vérifier que la documentation publique ne contient aucune donnée réelle.
