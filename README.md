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
- score de fiabilité des tendances fondé sur la couverture et la régularité des mesures ;
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

La trajectoire cible fournit un repère métier fixe entre le 03/09/2026 et le 16/12/2026, avec un point quotidien inclusif (105 points) et un objectif final exact de 80,0 kg. Son poids de départ est toujours 106,1 kg le 03/09/2026 : les données CSV servent uniquement à comparer le poids observé, jamais à modifier l’ancrage ou la pente. La durée exacte est de 104 jours, soit une perte quotidienne théorique de 0,250962 kg/jour et un rythme moyen requis de 1,76 kg/semaine.

### Qualité des données

Le chargement et le journal signalent les problèmes de qualité courants :

- colonnes obligatoires manquantes ;
- dates invalides ;
- valeurs non numériques ;
- valeurs non positives ;
- doublons de date ;
- valeurs potentiellement aberrantes ;
- irrégularité des mesures.

### Fiabilité des tendances

Le Dashboard affiche un score de fiabilité de **0 à 100** qui aide à savoir si les tendances courtes peuvent être interprétées sereinement. Il combine quatre dimensions :

- volume total de mesures (30 points) ;
- durée de l'historique, plafonnée à 30 jours (25 points) ;
- couverture des 30 derniers jours de données, plafonnée à 8 mesures (30 points) ;
- régularité de mesure selon l'intervalle médian entre deux dates (15 points).

Les niveaux sont `faible` (< 55), `moyenne` (55–79) et `élevée` (≥ 80). Ce KPI évalue uniquement la **couverture des données** : ce n'est ni un score de santé, ni un jugement sur la progression. La fenêtre est ancrée sur la dernière mesure disponible afin que les imports historiques restent reproductibles.

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

### Intégration WHOOP

L'onglet `Whoop` connecte un bracelet WHOOP en lecture seule via OAuth 2.0 et importe, pour la période choisie, les récupérations, les nuits de sommeil, les cycles physiologiques et les séances. Les métriques sont agrégées par jour calendaire puis croisées avec la courbe de poids.

Au-delà de ce que restitue l'application WHOOP, l'onglet exploite la seule donnée qu'elle ne possède pas — le poids :

- **Bilan énergétique estimé** : la dépense quotidienne mesurée par WHOOP est combinée à la pente du poids (convertie à raison de 7 700 kcal par kilogramme) pour en déduire l'apport calorique moyen. Ni la balance ni le bracelet ne peuvent produire ce chiffre seuls.
- **Corrélations décalées** : chaque métrique est confrontée à la variation de poids avec 0, 1 et 2 jours de décalage, un effort pesant rarement sur la balance le jour même.
- **Charge d'entraînement aigüe/chronique** : la charge des 7 derniers jours rapportée à celle des 28 derniers, indicateur classique de gestion de charge que WHOOP n'affiche pas.
- **Moteurs de la récupération** : régression du score de récupération sur le sommeil et la charge de la veille, qui chiffre ce que rapporte une heure de sommeil supplémentaire.
- **Dette de sommeil** : écart entre le besoin estimé par WHOOP et le sommeil obtenu, cumulé sur la semaine.
- **Zones de récupération**, **synthèse hebdomadaire**, **profil par jour de semaine** et **régularité du coucher**.

- **Lecture narrative** : les constats sont rédigés en français, chiffres à l'appui, et classés par importance (montée de charge brutale, dette de sommeil, apport calorique estimé, creux récurrent le tel jour…). Chaque règle reste muette tant que son effectif minimal n'est pas atteint.
- **Repère personnel** : la HRV et la fréquence au repos sont comparées à la médiane de vos 30 derniers jours plutôt qu'à une norme générale.
- **Calendrier de récupération** : une grille semaine × jour qui donne un mois de lecture d'un seul coup d'œil.

Choix de visualisation appliqués : palette catégorielle validée pour la vision des couleurs, couleurs de statut réservées aux significations bon/mauvais, grille en filet, légende absente pour une série unique, jours sans mesure laissés vides plutôt que reliés, et **aucun graphique à double axe vertical** — le poids et une métrique WHOOP sont ramenés à une base 100 commune, car caler deux échelles verticales l'une sur l'autre fabrique une corrélation visuelle arbitraire. Chaque graphique coloré possède sa vue tableau. Un filtre de période unique (7 / 30 / 90 jours / tout) s'applique à tous les onglets.

Chaque analyse annonce son effectif minimal et affiche le nombre de jours restants tant qu'il n'est pas atteint : sur un historique trop court, une corrélation ou une pente reflète le bruit de mesure plutôt qu'une tendance. Les graphiques laissent visibles les jours sans mesure au lieu de les relier par une droite, et les tableaux sont mis en forme (décimales maîtrisées, dates courtes, valeurs manquantes explicites).

Les données WHOOP vivent uniquement dans la session Streamlit : elles ne sont jamais écrites dans la source de poids, ni dans `working_data`. Sans compte connecté, l'onglet affiche l'écran de connexion et le reste de l'application fonctionne à l'identique.

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
│   │   ├── weight_summary.py
│   │   ├── whoop.py
│   │   ├── whoop_analytics.py
│   │   └── whoop_session.py
│   ├── pages/
│   │   ├── Dashboard.py
│   │   ├── Insights.py
│   │   ├── Journal.py
│   │   ├── Predictions.py
│   │   ├── Settings.py
│   │   └── Whoop.py
│   └── ui/
│       ├── components.py
│       ├── theme.py
│       └── whoop_visuals.py
└── tests/
    ├── conftest.py
    ├── test_analytics.py
    ├── test_core_v2.py
    ├── test_phase2_reliability.py
    ├── test_streamlit_smoke.py
    ├── test_target_trajectory.py
    ├── test_utils.py
    ├── test_v3_guardrails.py
    ├── test_weight_summary.py
    ├── test_whoop.py
    ├── test_whoop_analytics.py
    ├── test_whoop_oauth_callback.py
    └── test_whoop_visuals.py
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
- `app/core/whoop.py` : client WHOOP (OAuth 2.0, pagination API v2), normalisation des enregistrements en DataFrames, agrégat journalier et croisement avec le poids. Le transport HTTP est injectable, ce qui rend le module testable sans réseau.
- `app/core/whoop_analytics.py` : analyses croisées WHOOP × poids : grille calendaire continue, zones de récupération, charge aigüe/chronique, dette de sommeil, bilan énergétique, corrélations décalées, synthèses hebdomadaire et par jour de semaine. Chaque fonction refuse de conclure sous son effectif minimal.
- `app/core/whoop_session.py` : glue Streamlit du flux OAuth : détection de l'URL publique de l'application, capture du retour de redirection sur n'importe quelle page et bascule vers l'onglet Whoop.
- `app/pages/` : pages visibles de l'application : Dashboard, Journal, Prévisions, Insights, Whoop et Paramètres.
- `app/ui/` : composants d'interface, cartes, graphiques et thème visuel.
- `app/ui/whoop_visuals.py` : construction des figures Plotly de l'onglet WHOOP, sans dépendance à Streamlit, ce qui rend les règles de lisibilité vérifiables par des tests plutôt que par relecture visuelle.
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

### Principes de calcul

- Les variations 7/30/90 jours utilisent des jours calendaires et retournent `N/A` lorsque le recul est insuffisant.
- Les données source ne sont jamais dédupliquées silencieusement ; les vues analytiques appliquent leur propre règle documentée.
- Les tendances et projections sont indicatives. Les modèles expérimentaux sont séparés des baselines et doivent être lus avec leur niveau de confiance.
- Les moyennes mobiles nommées « N mesures » ne doivent pas être confondues avec les fenêtres calendaires « N jours ».

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

[whoop]
client_id = "WHOOP_CLIENT_ID"
client_secret = "WHOOP_CLIENT_SECRET"
redirect_uri = "https://votre-app.streamlit.app"
```

Le template `.streamlit/secrets.example.toml` peut servir de point de départ. Le fichier `.streamlit/secrets.toml` est ignoré par Git : il ne doit jamais être committé.

### Configuration WHOOP

1. Créer une application sur le tableau de bord développeur WHOOP et relever le `client_id` et le `client_secret`.
2. Déclarer côté WHOOP, dans le champ *Redirect URLs*, l'URL exacte de l'application (par exemple `https://votre-app.streamlit.app/` en ligne, ou `http://localhost:8501` en local). **WHOOP compare caractère pour caractère** : une barre oblique finale en trop ou en moins suffit à provoquer l'erreur `invalid_request` avec le message *« The "redirect_uri" parameter does not match any of the OAuth 2.0 Client's pre-registered redirect urls »*. L'onglet affiche l'URL qu'il détecte et celle qu'il envoie, à recopier telle quelle dans le tableau de bord WHOOP.
3. Renseigner le bloc `[whoop]` dans les secrets Streamlit. Trois autres sources sont acceptées, par ordre de priorité décroissante : saisie manuelle dans l'onglet (valable le temps de la session), clés à plat `whoop_client_id` / `whoop_client_secret` / `whoop_redirect_uri`, variables d'environnement `WHOOP_CLIENT_ID` / `WHOOP_CLIENT_SECRET` / `WHOOP_REDIRECT_URI`.
4. Ouvrir l'onglet `Whoop`, cliquer sur « Autoriser l'accès WHOOP », accepter côté WHOOP, puis lancer une synchronisation.

Les scopes demandés sont en lecture seule : `read:profile`, `read:body_measurement`, `read:cycles`, `read:recovery`, `read:sleep`, `read:workout`, plus `offline` qui permet de renouveler le jeton sans redemander l'autorisation. Le jeton d'accès est conservé en session et rafraîchi automatiquement à l'approche de son expiration.

Points d'attention :

- la redirection peut pointer vers la racine de l'application : le code d'autorisation est alors capté par le point d'entrée, avant la porte d'authentification et avant tout rendu de page, puis l'onglet Whoop est activé automatiquement. Les trois formes gérées sont la racine avec barre oblique finale, sans barre oblique, et le chemin `/Whoop` ;
- le scope `offline` peut être retiré depuis l'onglet si l'application WHOOP ne l'autorise pas : la connexion fonctionne alors sans renouvellement automatique du jeton ;
- l'API WHOOP plafonne la pagination à 25 éléments par page ; le client suit les pages via `nextToken` jusqu'à épuisement ;
- une nuit de sommeil est rattachée au jour du réveil, ce qui la rend comparable à la pesée du matin ;
- les siestes ne remplacent pas la nuit principale : la plus longue période de sommeil du jour est conservée ;
- les dépenses énergétiques renvoyées en kilojoules sont converties en kilocalories ;
- les corrélations affichées sont des associations linéaires sur de courtes séries : elles n'établissent aucune causalité.

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
