# Suivi V1 — Application Streamlit de suivi de mesures

## 1. Présentation

Suivi V1 est une application Streamlit générique permettant de suivre des mesures datées et d'analyser leur évolution dans le temps. Elle fournit un socle réutilisable pour charger des données, visualiser les tendances, comparer les mesures à des objectifs configurables et explorer des projections indicatives.

L'application permet notamment de :

- charger des mesures datées depuis une source CSV distante ou un fichier local ;
- visualiser l'évolution des mesures sous forme de tableaux, graphiques et indicateurs ;
- calculer des tendances, variations calendaires et moyennes mobiles ;
- suivre des objectifs configurables et une trajectoire cible ;
- produire des analyses descriptives sur la qualité, la régularité et la dynamique des données ;
- explorer des projections indicatives à partir des mesures disponibles.

## 2. Fonctionnalités

### Tableau de bord

Le tableau de bord synthétise l'état courant des mesures chargées en session :

- indicateurs principaux ;
- variation récente ;
- tendance globale ;
- écart à l'objectif ou à la trajectoire cible ;
- graphiques d'évolution ;
- moyennes mobiles ;
- objectifs et paliers ;
- analyses secondaires de progression, volatilité et distribution.

### Journal des mesures

Le journal permet de gérer les données chargées dans la session Streamlit :

- affichage tabulaire ;
- édition des lignes ;
- ajout ou suppression de mesures ;
- validation des dates, valeurs, doublons et valeurs potentiellement aberrantes ;
- filtrage par période ;
- import CSV local ;
- export CSV ;
- conservation des colonnes personnalisées.

### Variations calendaires

Les variations sont calculées sur des fenêtres exprimées en jours calendaires. Cette approche tient compte de l'espacement réel entre les mesures et évite de confondre une fenêtre temporelle avec un simple nombre de lignes.

### Moyennes mobiles

L'application prend en charge des moyennes mobiles configurables afin de lisser les mesures et de rendre les tendances plus lisibles. Les paramètres de moyenne mobile peuvent être ajustés depuis les préférences de session.

### Objectifs et paliers

Les objectifs configurables servent de repères visuels et analytiques. Des paliers intermédiaires peuvent être affichés pour suivre la progression vers une valeur finale.

### Trajectoire cible configurable

La trajectoire cible fournit un repère déterministe basé sur des paramètres métier centralisés. Elle est utilisée pour comparer les mesures observées à une progression attendue.

### Qualité des données

Le chargement et le journal signalent les problèmes de qualité courants :

- colonnes obligatoires manquantes ;
- dates invalides ;
- valeurs non numériques ;
- valeurs non positives ;
- doublons de date ;
- valeurs potentiellement aberrantes ;
- irrégularité des mesures.

### Plateau et stagnation

Un moteur commun détecte les périodes de plateau ou de stagnation sur des fenêtres calendaires. Les résultats sont utilisés dans les pages d'analyse pour qualifier les phases d'évolution.

### Projections simples

Les projections simples extrapolent une tendance observée à partir des mesures disponibles. Elles restent indicatives et peuvent être limitées lorsque les données sont insuffisantes, irrégulières ou incohérentes avec l'objectif configuré.

### Modèles avancés

La page de prévisions inclut des modèles statistiques ou expérimentaux, notamment SARIMAX, Auto-ARIMA, ML quantile, analyses STL, ACF/PACF et comparaisons de baselines en backtest chronologique.

### Import et export CSV

Les données peuvent être chargées depuis une source CSV distante configurée ou importées localement depuis l'interface. Le journal permet également d'exporter les données de session au format CSV.

### Authentification optionnelle

Une protection par mot de passe peut être activée via les secrets Streamlit. Lorsqu'elle est désactivée, l'application démarre sans étape d'authentification.

## 3. Architecture

```text
Suivi_V1/
├── Suivi_V1.py
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

### Rôle des principaux modules

- `Suivi_V1.py` : point d'entrée Streamlit, configuration de page, authentification, chargement des données, sidebar et navigation.
- `app/core/business.py` : paramètres métier centralisés, objectif final et constantes de trajectoire.
- `app/core/target_trajectory.py` : calcul de la trajectoire cible, alignement des mesures et comparaisons à la cible.
- `app/core/projection_constraints.py` : contraintes d'affichage appliquées aux projections et arrêt à l'objectif configuré.
- `app/core/plateau.py` : moteur de détection de plateau et de stagnation.
- `app/core/time_utils.py` : normalisation défensive des dates et conversion des séries temporelles.
- `app/core/formatting.py` : formatage des dates, valeurs et nombres pour l'interface.
- `app/core/weight_summary.py` : synthèse des mesures, variations calendaires, moyennes mobiles et indicateurs de suivi.
- `app/core/analytics.py` : fonctions descriptives, tendances, phases, scénarios, scores et comparaisons temporelles.
- `app/core/insights.py` : analyses de plateau, anomalies robustes, ETA et synthèses analytiques.
- `app/core/forecasting.py` : prévisions statistiques, modèles SARIMAX et modèles ML quantile.
- `app/core/data.py` : chargement, nettoyage, validation, rapport qualité et résolution des doublons.
- `app/core/session_state.py` : initialisation, lecture, écriture et réinitialisation des données en session Streamlit.
- `app/pages/` : pages visibles de l'application : Dashboard, Journal, Prévisions, Insights et Paramètres.
- `app/ui/` : composants d'interface, cartes, graphiques et thème visuel.
- `tests/` : tests automatisés couvrant les calculs, garde-fous, composants Streamlit et comportements métier.

## 4. Flux de données

Le flux de données suit une chaîne simple :

```text
source CSV distante ou fichier local
→ validation
→ nettoyage
→ normalisation
→ données de session
→ calculs analytiques
→ visualisations
```

Étapes principales :

1. La source CSV est chargée depuis une configuration distante ou un import local.
2. Les colonnes attendues sont validées.
3. Les dates et valeurs numériques sont converties dans des formats exploitables.
4. Les lignes invalides sont signalées ou exclues selon le contexte d'utilisation.
5. Les données sont triées chronologiquement et stockées dans la session Streamlit.
6. Les modules analytiques calculent variations, tendances, moyennes, scores, projections et indicateurs.
7. Les pages Streamlit affichent les résultats sous forme de KPI, tableaux, graphiques et messages d'analyse.

## 5. Configuration métier

La trajectoire cible utilise quatre paramètres :

- une date de départ ;
- une valeur initiale ;
- un rythme hebdomadaire ;
- un objectif final.

Formule générique :

```text
target_value = start_value - elapsed_weeks × weekly_rate
```

La valeur cible est calculée à partir du nombre de semaines écoulées depuis la date de départ. La trajectoire est bornée par l'objectif final et s'arrête lorsque cet objectif est atteint.

Exemple de paramètres génériques :

```python
TARGET_START_DATE = "YYYY-MM-DD"
TARGET_START_WEIGHT = START_VALUE
TARGET_WEEKLY_RATE = WEEKLY_RATE
TARGET_FINAL_WEIGHT = FINAL_VALUE
```

## 6. Configuration technique

### Secrets Streamlit

Créer un fichier local de secrets :

```text
.streamlit/secrets.toml
```

Exemple de configuration :

```toml
data_url = "CSV_EXPORT_URL"

[auth]
required = true
password = "APPLICATION_PASSWORD"
```

Le template `.streamlit/secrets.example.toml` peut servir de point de départ.

### Configuration applicative

Les paramètres applicatifs sont centralisés dans `app/config.py` et les constantes métier dans `app/core/business.py`. Les préférences modifiables depuis l'interface sont conservées dans la session Streamlit.

### Colonnes CSV attendues

Les colonnes obligatoires sont :

- `Date` ;
- `Poids (Kgs)`.

Des colonnes additionnelles peuvent être présentes et conservées, par exemple des notes ou des attributs contextuels.

## 7. Installation

### Prérequis

- Python compatible avec la version indiquée dans `runtime.txt` ;
- `pip` ;
- un environnement virtuel Python recommandé.

### Installation locale

```bash
git clone REPOSITORY_URL
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

## 8. Tests

Les tests automatisés utilisent `pytest` :

```bash
pytest
```

Les tests couvrent notamment :

- les fonctions analytiques ;
- la trajectoire cible ;
- les garde-fous de fiabilité ;
- la synthèse des mesures ;
- les utilitaires ;
- le chargement des pages Streamlit.

## 9. Déploiement

L'application peut être déployée sur une plateforme compatible Streamlit.

Étapes générales :

1. publier le code applicatif ;
2. définir les dépendances Python via `requirements.txt` ;
3. définir la version Python via `runtime.txt` lorsque la plateforme le prend en charge ;
4. configurer les secrets dans l'interface de déploiement ;
5. définir la commande de lancement Streamlit :

```bash
streamlit run Suivi_V1.py
```

## 10. Limites d'interprétation

Les analyses et projections fournies par l'application sont descriptives et indicatives. Elles dépendent de la qualité, de la régularité et du volume des mesures disponibles. Les modèles avancés doivent être interprétés comme des outils exploratoires plutôt que comme des garanties de résultat.
