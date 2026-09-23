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

- **Vue rapide** : poids du jour, **poids de tendance** (régression locale robuste LOWESS sur 14 jours), **rythme sur 14 jours avec son intervalle de confiance à 95 %**, variation 30 jours et écart à la trajectoire cible ;
- **Lecture rapide** : la pesée du jour est située par rapport au bruit quotidien habituel (± 1,96 σ des écarts à la tendance) : une pesée dans la bande n'est pas un changement de poids ;
- score de fiabilité des tendances fondé sur la couverture et la régularité des mesures ;
- graphique d'évolution avec la tendance, sa **bande de bruit habituel**, les objectifs et la trajectoire cible, puis un zoom sur la période cible ;
- indicateurs avancés : rythme de fond sur 28 jours, IMC avec le repère OMS, bruit quotidien, discipline, série en cours ;
- onglet **Objectifs & paliers** : reste à perdre depuis la tendance et date indicative pour chaque palier quand la pente est établie ;
- historique : vue hebdomadaire, moyennes mobiles, distribution des écarts à la tendance, IMC.

Les deltas des indicateurs utilisent le signe ASCII attendu par Streamlit : une perte s'affiche désormais avec une flèche descendante verte, et non plus avec une flèche montante.

### Journal des mesures

Le journal permet de gérer les données chargées dans la session Streamlit :

- **ajout rapide** d'une pesée en trois champs (date, poids, note) sans faire défiler l'éditeur ;
- affichage tabulaire avec dates en jour/mois/année et poids à une décimale ;
- édition des lignes, ajout ou suppression de mesures ;
- validation des dates, valeurs, doublons et valeurs potentiellement aberrantes ;
- filtrage par période ;
- import CSV local ;
- export CSV ;
- conservation des colonnes personnalisées.

### Variations calendaires

Les variations sont calculées sur des fenêtres exprimées en jours calendaires. Cette approche tient compte de l'espacement réel entre les mesures et évite de confondre une fenêtre temporelle avec un simple nombre de lignes.

### Poids de tendance, bruit et rythme

Une pesée quotidienne varie de plusieurs centaines de grammes sous l'effet de l'eau, du glycogène et du contenu digestif. Le module `app/core/trend.py` fournit trois lectures qui résistent à ce bruit :

- un **poids de tendance** par régression locale robuste (LOWESS, [Cleveland 1979](https://doi.org/10.1080/01621459.1979.10481038)) sur une fenêtre de 14 jours calendaires, insensible à une pesée isolée ;
- le **bruit quotidien** : écart-type robuste (MAD × 1,4826) des écarts à la tendance, et sa demi-largeur à 95 % ;
- le **rythme en kg/semaine** par moindres carrés sur les jours calendaires, avec intervalle de confiance de Student et valeur p : une pente dont l'intervalle contient zéro est dite « que le hasard suffit à produire », jamais « en baisse ».

Les moyennes mobiles restent disponibles en option d'affichage.

### Objectifs et paliers

Les objectifs configurables servent de repères visuels et analytiques. Des paliers intermédiaires peuvent être affichés pour suivre la progression vers une valeur finale.

### Trajectoire cible configurable

La trajectoire cible fournit un repère métier fixe entre le 20/09/2026 et le 16/12/2026, avec un point quotidien inclusif (88 points) et un objectif final exact de 80,0 kg. Son poids de départ est toujours 106,1 kg le 20/09/2026 : les données CSV servent uniquement à comparer le poids observé, jamais à modifier l’ancrage ou la pente. La durée exacte est de 87 jours, soit une perte quotidienne théorique de 0,3 kg/jour et un rythme moyen requis de 2,10 kg/semaine.

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

La page Prévisions ouvre sur la **projection selon vos mesures** : le poids de tendance prolongé au rythme des 28 derniers jours, avec un cône d'incertitude à 95 % qui combine l'erreur sur la pente (croissante avec l'horizon), l'erreur sur le niveau de la tendance et le bruit d'une pesée. La date d'arrivée à l'objectif est encadrée par les deux bornes de la pente et n'est projetée que si la pente descend de façon établie. La projection est bornée à l'objectif final.

### Modèles avancés et leaderboard

Un **leaderboard walk-forward** réajuste chaque modèle sur le passé puis le juge sur la semaine de mesures suivante, sur plusieurs blocs chronologiques partagés (au moins 20 mesures d'apprentissage au premier bloc). Il compare la dernière valeur répétée, les moyennes mobiles, la tendance linéaire, la tendance robuste avec pente, SARIMAX et Auto-ARIMA. Chaque ligne porte la MAE, le **gain par rapport à la dernière valeur**, la **couverture empirique de l'intervalle à 95 %** et un verdict : un modèle qui ne bat pas « la dernière pesée répétée » est dit tel quel.

Les modèles expérimentaux (régression sur variables dérivées, ML quantile, SARIMAX, Auto-ARIMA, STL, ACF/PACF, scénarios) restent disponibles ; les variables dérivées sont toutes **décalées d'au moins une mesure** afin qu'aucune ne contienne la cible du jour. Les ajustements coûteux sont mis en cache.

### Insights

- effet du **jour de la semaine** sur les écarts à la tendance, testé par un test de Welch par jour avec correction de Bonferroni sur les sept jours candidats ;
- **pesées atypiques** par z-score robuste ([Iglewicz & Hoaglin 1993, via le NIST e-Handbook](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)) des écarts à la tendance, seuil 3,5 : comparer à la médiane globale signalait à tort les extrêmes d'une perte régulière ;
- distribution des fluctuations d'un jour à l'autre face au bruit habituel ;
- phases du parcours, ruptures de tendance, meilleures et pires semaines, comparaisons périodiques, séries consécutives.

### Import et export CSV

Les données peuvent être chargées depuis une source CSV distante configurée ou importées localement depuis l'interface. Le journal permet également d'exporter les données de session au format CSV.

### Authentification optionnelle

Une protection par mot de passe peut être activée via les secrets Streamlit. Lorsqu'elle est désactivée, l'application démarre sans étape d'authentification.

### Intégration WHOOP

L'onglet `Whoop` connecte un bracelet WHOOP en lecture seule via OAuth 2.0 et importe, pour la période choisie, les récupérations, les nuits de sommeil, les cycles physiologiques et les séances. Les métriques sont agrégées par jour calendaire puis croisées avec la courbe de poids.

Au-delà de ce que restitue l'application WHOOP, l'onglet exploite la seule donnée qu'elle ne possède pas — le poids :

- **Bilan énergétique estimé** : la dépense quotidienne mesurée par WHOOP est combinée à la pente du poids (convertie à raison de 7 700 kcal par kilogramme) pour en déduire l'apport calorique moyen. Ni la balance ni le bracelet ne peuvent produire ce chiffre seuls.
- **Corrélations décalées** : chaque métrique est confrontée à la variation de poids avec 0, 1 et 2 jours de décalage, un effort pesant rarement sur la balance le jour même.
- **Charge d'entraînement aigüe/chronique** : la charge des 7 derniers jours rapportée à celle des 21 jours qui les précèdent — fenêtres découplées, sans quoi le rapport est borné par construction ([Windt & Gabbett, 2019](https://bjsm.bmj.com/content/53/16/988)) —, indicateur classique de gestion de charge que WHOOP n'affiche pas. Il décrit la charge relative, il ne prédit pas la blessure.
- **Lectures par date** : chaque matin noté (score, zone, HRV, FC repos, nuit précédente, strain de la veille) et chaque séance (sport, heure, durée, strain, zones de FC, **récupération du lendemain matin en face**), du plus récent au plus ancien, jour de semaine en tête.
- **HRV et FC repos lissées** : la moyenne mobile de sept jours face à la plage habituelle des trente jours précédents, méthode recommandée pour la variabilité cardiaque ([Buchheit, 2014](https://doi.org/10.3389/fphys.2014.00073)) ; une nuit isolée ne fait pas une tendance.
- **Zones de fréquence cardiaque** : part du temps facile / modéré / dur par sport, additionnée sur les séances de la période, à partir des `zone_durations` de l'API.
- **Moteurs de la récupération** : régression du score de récupération sur le sommeil et la charge de la veille, qui chiffre ce que rapporte une heure de sommeil supplémentaire.
- **Dette de sommeil** : écart entre le besoin estimé par WHOOP et le sommeil obtenu, cumulé sur la semaine.
- **Zones de récupération**, **synthèse hebdomadaire**, **profil par jour de semaine** et **régularité du coucher**.

- **Où va votre poids, et quand la cible tomberait** : la tendance mesurée en kg/semaine, la position face à la trajectoire cible, et la date d'arrivée projetée **confrontée à l'échéance visée** — être en avance aujourd'hui ne dit rien de la date d'arrivée.
- **Séries de journées** : le nombre de jours consécutifs dans la même zone de récupération, qu'aucune moyenne hebdomadaire ne fait ressortir.
- **Ce que pèsent vraiment vos séances** : part de la dépense quotidienne attribuable aux entraînements. Les calories s'additionnent, contrairement au strain qui est une échelle logarithmique.
- **Lecture narrative** : les constats sont rédigés en français, chiffres à l'appui, et classés par importance (montée de charge brutale, dette de sommeil, apport calorique estimé, creux récurrent le tel jour…). Chaque règle reste muette tant que son effectif minimal n'est pas atteint.
- **Repère personnel** : la HRV et la fréquence au repos sont comparées à la médiane de vos 30 derniers jours plutôt qu'à une norme générale.
- **Journal jour par jour** : une carte par journée, du plus récent au plus ancien, réunissant récupération et sa zone, sommeil et dette, charge, poids et **les séances du jour situées par leur heure de début**. Les courbes disent comment les choses évoluent ; elles ne disent jamais ce qui s'est passé mardi.
- **Veille physiologique** : les cinq signes vitaux nocturnes (fréquence cardiaque de repos, variabilité, **fréquence respiratoire**, température cutanée, saturation) comparés à un repère personnel robuste. Le seuil de signalement a été choisi par mesure du taux de fausse alerte sur séries sans anomalie. Ces mesures ne constituent pas un diagnostic.
- **Architecture du sommeil** : la part de sommeil profond et de sommeil REM dans la nuit, rapportée aux plages usuellement citées chez l'adulte — une nuit courte réduisant mécaniquement les heures de chaque stade.
- **Calendrier de récupération** : une grille semaine × jour qui donne un mois de lecture d'un seul coup d'œil.
- **Vos bons jours contre vos mauvais** : comparaison du tiers de vos meilleures journées de récupération au tiers des pires, pour identifier le facteur qui les sépare le plus (sommeil, charge, heure de coucher, perturbations).
- **Votre objectif traduit en calories** : le rythme visé par la trajectoire cible, converti en déficit puis confronté à la dépense mesurée par WHOOP, donne l'apport quotidien que l'objectif suppose. C'est le croisement le plus structurant, et ni la balance ni le bracelet ne le produisent seuls.
- **Coût de chaque sport** : récupération du lendemain sport par sport — le score du jour même précède la séance, seul le lendemain en porte la trace.

Lisibilité des dates : mois et jours écrits en français (`3 sept.`, `jeudi 3 septembre 2026`), ancienneté exprimée en clair (`hier`, `il y a 5 jours`), durées en heures et minutes (`1 h 30` plutôt que `90,0`), heures de coucher rendues en horloge, et semaines nommées (`Semaine du 3 août`). Les noms sont codés dans l'application plutôt que tirés de la locale système, qui ferait dépendre l'affichage des paquets installés sur la machine de déploiement. Un bandeau signale les données qui s'arrêtent il y a plus de deux jours, et le filtre de période compte à partir d'aujourd'hui — pas de la dernière mesure, ce qui donnerait à « 7 jours » un sens flottant.

Choix de visualisation appliqués : palette catégorielle validée pour la vision des couleurs, couleurs de statut réservées aux significations bon/mauvais, grille en filet, légende absente pour une série unique, jours sans mesure laissés vides plutôt que reliés, et **aucun graphique à double axe vertical** — le poids et une métrique WHOOP sont ramenés à une base 100 commune, car caler deux échelles verticales l'une sur l'autre fabrique une corrélation visuelle arbitraire. Chaque graphique coloré possède sa vue tableau. Un filtre de période unique (7 / 30 / 90 jours / tout) s'applique à tous les onglets.

Garde-fous statistiques : les corrélations sont corrigées pour tests multiples (une vingtaine de métriques testées à trois décalages produit sinon une corrélation « forte » par pur hasard), la régression de récupération publie un R² **ajusté** au nombre de variables, le bilan énergétique affiche son intervalle à 95 %, le contraste entre bons et mauvais jours classe les facteurs par taille d'effet et non par écart brut, et le rapport de charge aigu/chronique se tait tant que les deux fenêtres portent sur les mêmes jours.

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
    ├── test_date_labels.py
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
- `app/core/date_labels.py` : libellés de dates en français (mois, jours, ancienneté relative, durées, plages, heures de coucher), indépendants de la locale système.
- `app/core/weight_summary.py` : synthèse des mesures, variations calendaires, moyennes mobiles et indicateurs de suivi.
- `app/core/trend.py` : poids de tendance LOWESS, bruit quotidien robuste, rythme avec intervalle de confiance, cône de projection, date d'arrivée encadrée, classes d'IMC de l'OMS.
- `app/core/analytics.py` : fonctions descriptives, tendances, phases, scénarios, scores, effet du jour de la semaine testé et comparaisons temporelles.
- `app/core/insights.py` : analyses de plateau, pesées atypiques sur les écarts à la tendance, ETA et synthèses analytiques.
- `app/core/evaluation.py` : métriques, découpages walk-forward partagés et leaderboard des modèles avec gain face à la dernière valeur et couverture des intervalles.
- `app/core/forecasting.py` : prévisions statistiques, modèles SARIMAX et modèles ML quantile.
- `app/core/data.py` : chargement, nettoyage, validation, rapport qualité et résolution des doublons.
- `app/core/session_state.py` : initialisation, lecture, écriture et réinitialisation des données en session Streamlit.
- `app/core/whoop.py` : client WHOOP (OAuth 2.0, pagination API v2), normalisation des enregistrements en DataFrames, agrégat journalier et croisement avec le poids. Le transport HTTP est injectable, ce qui rend le module testable sans réseau.
- `app/core/whoop_analytics.py` : analyses croisées WHOOP × poids : grille calendaire continue, zones de récupération, charge aigüe/chronique, dette de sommeil, bilan énergétique, corrélations décalées, synthèses hebdomadaire et par jour de semaine. Chaque fonction refuse de conclure sous son effectif minimal.
- `app/core/whoop_session.py` : glue Streamlit du flux OAuth : détection de l'URL publique de l'application, capture du retour de redirection sur n'importe quelle page et bascule vers l'onglet Whoop.
- `app/pages/` : pages visibles de l'application : Dashboard, Journal, Prévisions, Insights, Whoop et Paramètres.
- `app/ui/` : composants d'interface, cartes, graphiques et thème visuel.
- `app/ui/charts.py` : style Plotly partagé par les pages poids (rôles de couleur fixes, axes de dates sans mois anglais, légende sous le graphique), aligné sur celui de l'onglet WHOOP.
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
- Une pente n'est annoncée « en baisse » ou « en hausse » que si son intervalle de confiance à 95 % exclut zéro ; sinon elle est dite stable.
- Les variables dérivées des modèles ML sont décalées d'au moins une mesure : aucune ne contient la cible du jour.

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
- la tendance robuste, le bruit et le rythme avec intervalle (`tests/test_trend.py`) ;
- le leaderboard walk-forward (`tests/test_evaluation_forecasters.py`) ;
- l'effet du jour de la semaine et les pesées atypiques (`tests/test_calendar_effects.py`) ;
- le chargement des pages Streamlit, y compris l'ajout rapide du Journal (`tests/test_pages_v4.py`).

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
