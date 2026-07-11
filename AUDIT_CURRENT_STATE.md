# Audit de l'état actuel — Suivi_V1

Date de l'audit : 2026-07-11. Périmètre : dépôt local `/workspace/Suivi_V1`, code Python Streamlit, tests, documentation, configuration et historique Git récent.

## 1. Executive summary

L'application est déjà plus avancée qu'un simple tableau de suivi : elle possède une architecture multipage, une couche `app/core` assez riche, des objectifs multiples, une trajectoire cible, des analyses de plateau, des prévisions, des tests unitaires et des smoke tests Streamlit. Le niveau de maturité fonctionnelle est **intermédiaire à avancé**, mais la maturité de fiabilité analytique reste **intermédiaire** car plusieurs calculs ou modèles peuvent être interprétés trop fortement par l'utilisateur.

### Forces principales

- Architecture lisible : point d'entrée `Suivi_V1.py`, pages dans `app/pages`, logique métier dans `app/core`, composants UI dans `app/ui`.
- État de session centralisé dans `app/core/session_state.py`, avec séparation `source_data`, `working_data`, `filtered_data` et `raw_data`.
- Nettoyage des données qui conserve les colonnes supplémentaires via `clean_weight_dataframe`.
- Tests existants sur le nettoyage, la trajectoire cible, les garde-fous, les prévisions et quelques pages Streamlit.
- Plusieurs garde-fous récents : ETA bloqué sur signaux fragiles, plateau basé sur jours calendaires, backtesting de baselines chronologique.

### Faiblesses principales

- Une anomalie critique de conservation existait au chargement distant : `load_remote_csv()` dédupliquait automatiquement les dates et supprimait les mesures multiples d'un même jour. Cette anomalie est corrigée dans cette intervention.
- La page Prévisions expose des modèles coûteux ou expérimentaux sans indicateur de fiabilité suffisamment unifié.
- Plusieurs analyses avancées peuvent être statistiquement fragiles sur petits volumes : STL hebdomadaire, ACF/PACF, KMeans, IsolationForest, SARIMAX saisonnier.
- Certaines fonctions préparatoires dédupliquent les dates pour les calculs ; c'est acceptable analytiquement si documenté, mais ne doit jamais modifier la source.
- `app/utils.py` contient encore une ancienne pile utilitaire partiellement redondante avec `app/core`, ce qui augmente le risque de divergence.

### Niveau de maturité

- **Produit utilisateur** : 6,5/10. L'application est utile au quotidien, mais l'organisation Dashboard / Journal / Prévisions / Insights / Paramètres pourrait mieux distinguer données, analyses, qualité et modèles.
- **Fiabilité données-calculs** : 6/10 après correction critique ; avant correction, la conservation distante des mesures multiples était un risque majeur.
- **Industrialisation** : 5,5/10. La suite de tests est réelle, mais ne couvre pas encore assez la persistance, les graphiques essentiels et les cas limites bout en bout.

### Principaux risques

1. Perte ou transformation silencieuse des données source si une logique analytique est appliquée trop tôt dans le pipeline.
2. Surinterprétation de modèles ML/forecasting entraînés sur peu de données individuelles.
3. Recalculs lourds au rerun Streamlit, surtout Auto-ARIMA, SARIMAX et STL/ACF/PACF.
4. Dérive entre l'ancien module `app/utils.py` et les modules modernes `app/core/*`.
5. Confusion utilisateur entre prévisions expérimentales, trajectoire cible déterministe et tendances observées.

## 2. Cartographie de l'application

### Entrée et navigation accessible

- `Suivi_V1.py` configure Streamlit, applique le thème, vérifie le mot de passe, initialise les données et déclare la navigation.
- Pages accessibles via `st.navigation` :
  - `app/pages/Dashboard.py` — Dashboard.
  - `app/pages/Journal.py` — Journal.
  - `app/pages/Predictions.py` — Prévisions.
  - `app/pages/Insights.py` — Insights.
  - `app/pages/Settings.py` — Paramètres.

### Pages ou fonctionnalités présentes mais non accessibles directement

- Il n'existe pas de page dédiée « Qualité des données » : la qualité est dans `Insights.py` et partiellement dans `Journal.py`.
- Il n'existe pas de page dédiée « Modèles ML / statistiques » : les modèles sont dans `Predictions.py` et les clusters/anomalies dans `Insights.py`.
- `app/utils.py` contient d'anciennes fonctions de chargement, filtrage, anomalies et régression linéaire qui ne semblent plus être le chemin principal de l'application multipage moderne, mais qui restent testées.
- La fonction `_render_smart_alerts()` dans `Dashboard.py` est définie mais n'est pas appelée dans le flux principal observé.

### Sources de données et flux

1. Source distante : `DATA_URL` dans `app/config.py`, éventuellement remplacée par `st.secrets["data_url"]`.
2. Chargement distant : `load_remote_csv(url)` lit un CSV avec `pd.read_csv`, puis appelle `clean_weight_dataframe`.
3. Import local : `_import_local_csv(uploaded_file)` lit un CSV local et appelle `clean_weight_dataframe`.
4. Stockage : `set_source_data(df, source_name)` remplit `source_data`, `working_data`, `filtered_data`, `raw_data`, `data_source`.
5. Journal : `st.data_editor` édite `working_data`; `validate_journal` nettoie et `set_working_data` persiste en session.
6. Pages analytiques : `get_filtered_or_working_data()` renvoie `filtered_data` si disponible, sinon `working_data`.

### Variables `st.session_state` identifiées

- Données : `source_data`, `working_data`, `filtered_data`, `raw_data`, `data_source`, `data_url`.
- Objectifs et préférences : `target_weights`, `target_weight`, `height_cm`, `height_m`, `duplicate_strategy`, `default_model`, `ma_type`, `window_size`, `theme`.
- Authentification : `password_correct`, `password` temporaire.
- Tests/performance : `fast_mode`.
- Widgets : `sidebar_csv_import`, `journal_editor`, clés implicites des sliders/selectbox/toggles.

### Fonctions utilisant le cache

- `Suivi_V1.py::load_remote_csv` : `@st.cache_data(ttl=300)`.
- `app/utils.py::load_data` : `@st.cache_data(ttl=300, show_spinner=False)`.
- `app/utils.py::get_data_diagnostics` : `@st.cache_data(ttl=300, show_spinner=False)`.
- `app/utils.py::convert_df_to_csv` : `@st.cache_data`.

### Modèles statistiques et ML présents

- Régression linéaire : `LinearRegression` dans `Predictions.py`, `app/core/models.py`, `app/utils.py`.
- Ridge, ElasticNet, RandomForest, GradientBoosting : définis dans `app/core/models.py`.
- QuantileRegressor q10/q50/q90 : utilisé par `forecast_with_ml`.
- RandomForestRegressor : comparaison modèles dans `Predictions.py`.
- SARIMAX : `forecast_with_sarimax`.
- Auto-ARIMA : import dynamique `pmdarima.auto_arima` dans `Predictions.py`.
- STL, ACF, PACF : bloc `STL & ACF/PACF` dans `Predictions.py`.
- KMeans : clustering dans `Insights.py`.
- IsolationForest : anomalies dans `app/core/insights.py` et ancien `app/utils.py`.
- Baselines : dernière valeur, moyennes mobiles 7/14, drift dans `app/core/evaluation.py`.

### Graphiques disponibles

- Dashboard : courbe principale poids + objectifs + trajectoire, écart à l'objectif, vue hebdomadaire, moyennes mobiles, distribution du poids, IMC.
- Prévisions : SARIMAX + intervalle, Auto-ARIMA + intervalle, ML quantile + intervalle, scénarios 90 jours, STL/ACF/PACF.
- Insights : phases du parcours, jour de semaine, anomalies, KMeans.
- Journal : tableaux et aperçu filtré, pas de graphique majeur.

## 3. Audit des données

### Constat global

La séparation source/travail/session est saine, mais il faut distinguer strictement :

- **Données source** : doivent conserver toutes les lignes et colonnes valides.
- **Données analytiques préparées** : peuvent dédupliquer ou agréger si la méthode l'explique.
- **Données filtrées** : ne doivent jamais remplacer la source sans action explicite.

### Points vérifiés

- Chargement Google Sheets/CSV : présent via `pd.read_csv`.
- Gestion erreurs réseau : `init_data_once()` capture toute exception et affiche un warning générique ; le bouton de reload affiche l'exception.
- Conservation des colonnes : `clean_weight_dataframe` conserve les colonnes extras.
- Parsing dates : `dayfirst=True`, puis fallback `dayfirst=False`.
- Parsing poids : virgules remplacées par points, conversion numérique.
- Lignes invalides : supprimées par défaut au chargement ; signalées dans `validate_journal`.
- Doublons : signalés dans le journal ; les fonctions analytiques peuvent les résoudre.
- Mesures multiples même jour : doivent être conservées en source ; corrigé pour le chargement distant.
- Export CSV : exporte le DataFrame édité, donc peut inclure des lignes invalides présentes dans l'éditeur avant sauvegarde.
- Mutations cache : les fonctions de session copient les DataFrames ; les fonctions cache renvoient des DataFrames nettoyés. Aucun cas évident de mutation directe du cache n'a été constaté.

## 4. Audit des calculs métier

| Calcul | Formule actuelle observée | Évaluation | Recommandation |
|---|---|---|---|
| Poids actuel | Dernière ligne après tri par date | Correct si dates valides ; ambigu si plusieurs mesures même jour | Afficher la dernière mesure horodatée si une colonne heure existe ; sinon documenter la règle |
| Poids moyen | Moyenne simple des mesures | Correct descriptivement, mais dépend de la fréquence de mesure | Ajouter moyenne journalière optionnelle si mesures multiples fréquentes |
| Min/max | Min/max des mesures | Correct | Conserver |
| Variation 7j/30j | `delta_since_days` sur fenêtre calendaire avec recul minimal | Bonne amélioration récente | Conserver les messages N/A quand recul insuffisant |
| Variation hebdomadaire | Agrégation par semaine, variation de moyenne hebdo | Correct pour lecture agrégée | Indiquer le nombre de mesures par semaine |
| Vitesse kg/semaine | Différence poids courant vs référence / jours réels × 7 | Correct mais sensible aux fluctuations d'eau | Afficher confiance selon nb mesures et span |
| Moyenne mobile simple | Rolling N mesures dans `multi_rolling_averages`; rolling calendaire dans `moving_average_by_days` | Les deux sont utiles mais doivent être libellées clairement | Garder les deux avec labels « mesures » vs « jours » |
| EMA | `compute_trend_ema` sur nombre de mesures | Utile visuellement, pas prédictif | Label explicite |
| IMC | poids / taille² | Correct si taille configurée | Ajouter avertissement si taille par défaut non confirmée |
| Progression objectifs | Écart courant et scores dans analytics/target trajectory | Globalement correct | Vérifier cas objectif supérieur au poids initial |
| ETA objectif | Régression linéaire sur fenêtres récentes avec garde-fous | Correctement prudent, mais encore sensible aux petits volumes | Refuser si span réel < 14 jours pour ETA forte confiance |
| Plateau | Moteur calendaire `evaluate_plateau_window` | Bonne direction | Conserver seuils centralisés et expliquer limites |
| Reprise poids | CUSUM simplifié / phase / streak | Indicatif uniquement | Ne pas alarmer sans confirmation multi-jours |
| Bilan calorique | calories consommées - brûlées dans features | Présent comme feature, peu exploité en UI | Ne pas conclure causalement sans qualité nutritionnelle |

## 5. Audit statistique

- Les moyennes mobiles sont pertinentes si leur unité est claire. L'application mélange encore des moyennes en nombre de mesures et des fenêtres calendaires ; c'est acceptable si explicitement indiqué.
- La volatilité par écart-type/CV est descriptive, mais sensible aux petits volumes. L'UI affiche parfois des scores même avec peu de points.
- Les Z-scores robustes via MAD sont préférables aux Z-scores classiques ; l'ancienne fonction `app/utils.py::detect_anomalies` garde un Z-score classique.
- ACF/PACF/STL sur série interpolée quotidiennement peuvent créer une illusion de fréquence régulière. Il faut refuser ou dégrader la confiance si la couverture journalière est trop basse.
- Le CUSUM simplifié dans `detect_trend_breaks` est intéressant, mais ne doit pas être présenté comme détection statistique formelle de rupture.
- Les intervalles SARIMAX sont des intervalles modèle, pas des intervalles de prédiction intégrant toutes les incertitudes utilisateur.
- Les corrélations et effets jour de semaine ne doivent pas être interprétés causalement.

## 6. Audit ML et forecasting

### Inventaire et pertinence

| Modèle | Usage | Risques | Pertinence |
|---|---|---|---|
| Baseline dernière valeur | Backtest | Simple, robuste | Très pertinente |
| Baseline MA7/MA14 | Backtest | Rolling en mesures, pas jours | Pertinente |
| Drift | Backtest | Sensible début/fin | Pertinente comme baseline |
| LinearRegression | Comparaison modèle | Features auto-corrélées, peu de données | Pertinente si prudente |
| Ridge/ElasticNet | Définis dans modèles | Pas exposés clairement | Pertinents si intégrés au backtest |
| RandomForest | Comparaison | Overfitting probable sur petit historique | Faible valeur actuelle |
| GradientBoosting | Défini non utilisé directement | Overfitting | Expérimental |
| QuantileRegressor | Prévision ML | Entraînement in-sample ; intervalles conditionnels fragiles | À conserver en expérimental |
| SARIMAX | Prévision | Fréquence irrégulière, saisonnalité hebdo imposée | À conditionner à données suffisantes |
| Auto-ARIMA | Prévision | Coût élevé, peut échouer, dépendance optionnelle | Optionnel/expérimental |
| STL | Décomposition | Interpolation quotidienne artificielle | Seulement si couverture suffisante |
| ACF/PACF | Diagnostic | Peu interprétable pour suivi individuel irrégulier | Avancé, à masquer par défaut |
| KMeans | Clustering poids | Clusters par niveau de poids peu actionnables | Faible valeur |
| IsolationForest | Anomalies | Fluctuations normales signalées | Utile seulement en aide à revue |

### Stratégie recommandée de forecasting

1. Refuser les prévisions si moins de 14 mesures ou moins de 21 jours de span pour les modèles simples ; seuil plus élevé pour SARIMAX/STL.
2. Toujours afficher les baselines : dernière valeur, moyenne mobile calendaire, drift.
3. Walk-forward chronologique avec horizon fixe et métriques hors échantillon uniquement.
4. Régression linéaire prudente sur temps réel + lags seulement si données suffisantes.
5. SARIMAX uniquement si série rééchantillonnée raisonnablement régulière, couverture suffisante et diagnostics OK.
6. Intervalles de prédiction calibrés à partir des erreurs walk-forward, pas uniquement des intervalles internes du modèle.
7. Indicateur de fiabilité : faible / moyenne / élevée selon nb mesures, span, régularité, erreur baseline et stabilité des résidus.
8. Masquer ou marquer expérimentalement RandomForest/KMeans/Auto-ARIMA tant qu'ils n'apportent pas un gain clair vs baseline.

## 7. Audit des graphiques

### Dashboard

À conserver : courbe principale, trajectoire cible, objectifs, vue rapide, distribution, IMC, hebdomadaire. À corriger/améliorer : indiquer explicitement quand les moyennes mobiles sont en jours vs mesures ; afficher le nombre de mesures dans les agrégats hebdomadaires ; appeler ou supprimer proprement `_render_smart_alerts` après décision produit.

### Journal

À conserver : éditeur, aperçu filtré, export. À améliorer : aperçu de validation avant sauvegarde plus visible, compteur lignes invalides, avertissement si export avant sauvegarde contient des lignes invalides.

### Prévisions

À conserver : leaderboard baselines, scénarios, ETA avec garde-fous. À rendre optionnel/expérimental : SARIMAX, Auto-ARIMA, STL/ACF/PACF, ML quantile. À corriger : harmoniser les intervalles comme intervalles de prédiction empiriques et afficher fiabilité.

### Insights

À conserver : qualité, plateau, phases, meilleures/pires semaines, jour de semaine, anomalies. À rendre optionnel : KMeans. À corriger : préciser que les anomalies sont des points à revoir et pas des erreurs certaines.

### Settings

À conserver : objectifs, taille, moyenne mobile, doublons, modèle par défaut. À corriger : la stratégie de doublons doit être appliquée explicitement aux analyses ou à une vue agrégée, pas au dataset source.

## 8. Audit UI/UX

### Navigation

La navigation actuelle est claire, mais l'architecture cible devrait distinguer plus explicitement : Dashboard, Journal, Analyses/Tendances, Prévisions, Modèles/Stats, Qualité données, Paramètres. Il n'est pas nécessaire de multiplier les pages immédiatement ; une première étape peut déplacer des blocs vers des tabs mieux nommés.

### Page par page

- Dashboard : objectif clair, bonne synthèse. Confusion possible entre objectif final, trajectoire cible et prévision. Améliorer les labels de confiance.
- Journal : fonctionnel et important. Ajouter feedback plus fort sur lignes ignorées et conservation des colonnes.
- Prévisions : riche mais dense. Séparer clairement « baselines fiables » et « modèles expérimentaux ».
- Insights : riche, mais mélange qualité, statistiques, clustering et anomalies. Ajouter hiérarchie.
- Paramètres : simple. Le champ objectif final désactivé synchronisé avec Objectif 5 est cohérent mais peut surprendre.

## 9. Architecture et qualité du code

### Points positifs

- Séparation UI / core réelle.
- Tests existants et nombreux modules purs.
- Configuration métier centralisée.
- Composants UI réutilisables.

### Points à corriger

- `app/utils.py` duplique des responsabilités modernes de `app/core/data.py`, `app/core/forecasting.py` et `app/core/insights.py`.
- Plusieurs pages appellent `main()` au niveau module. C'est courant pour Streamlit multipage mais complique certains imports de test.
- Quelques blocs `try/except Exception` larges masquent les causes exactes côté utilisateur.
- Les modèles sont entraînés au rerun sans cache ressource ni contrôle fin.
- Certaines fonctions analytiques mutent des copies, ce qui est correct, mais la règle doit rester stricte.

## 10. Performances

- `load_remote_csv` est caché 5 minutes : bon pour réseau.
- Auto-ARIMA, SARIMAX, STL/ACF/PACF sont recalculés au rerun si l'onglet/bloc s'exécute : risque de lenteur.
- `st.form` est bien utilisé dans Paramètres, mais pas dans Journal pour encadrer toutes les éditions.
- `st.session_state` est utilisé pour l'état éditable : bonne pratique.
- Recommandation : utiliser `st.cache_data` pour features/diagnostics purs dépendants d'un hash de données, `st.cache_resource` seulement pour ressources lourdes stables, ne jamais mettre `working_data` éditable en cache global.

## 11. Audit des tests

### Couverture actuelle

- `tests/test_core_v2.py` : nettoyage, colonnes, doublons, baselines, ETA, plateau, prévisions, objectifs.
- `tests/test_weight_summary.py` : synthèse, deltas, projection, insights, tri, virgules.
- `tests/test_phase2_reliability.py` : trajectoire cible, contraintes de projection, plateau, formatting.
- `tests/test_streamlit_smoke.py` : rendu Dashboard/Journal/Predictions/Settings et navigation simulée.
- `tests/test_utils.py` : ancienne couche utils.

### Faiblesses des tests

- Peu de tests vérifient le contenu exact des graphiques Plotly.
- Peu de tests vérifient la persistance après édition réelle dans `st.data_editor`.
- Les tests de modèles vérifient surtout la forme des DataFrames, pas la qualité prédictive.
- Pas assez de datasets synthétiques irréguliers avec mesures multiples même jour, longues pauses, reprise, plateau et outliers combinés.

### Plan de tests recommandé

Créer des fixtures synthétiques : vide, une mesure, 5 mesures, 30 jours réguliers, irrégulier, multiples mesures/jour, colonnes extras, calories, outliers, longue pause. Couvrir : import CSV, conservation lignes/colonnes, parsing, doublons, calculs 7/30j, IMC, progression, ETA, plateau, anomalies, backtesting, prévisions, navigation, session_state, graphiques essentiels, lignes d'objectifs, petits datasets.

## 12. Score par domaine

| Domaine | Score | Justification |
|---|---:|---|
| Données | 7/10 | Pipeline clair et colonnes conservées ; correction apportée à la perte des mesures multiples au chargement distant. |
| Calculs métier | 7/10 | Variations calendaires et garde-fous solides ; quelques ambiguïtés sur mesures multiples et objectifs. |
| Statistiques | 5,5/10 | Analyses riches, mais certains diagnostics avancés fragiles sur séries irrégulières ou petites. |
| ML / forecasting | 5/10 | Baselines utiles ; modèles avancés expérimentaux, coûteux et parfois surdimensionnés. |
| UI/UX | 6,5/10 | Interface moderne et riche ; densité forte dans Prévisions/Insights. |
| Architecture | 6,5/10 | Bonne modularité core/pages ; dette `app/utils.py` et imports exécutants. |
| Performances | 5,5/10 | Cache réseau présent ; modèles lourds recalculés. |
| Tests | 6/10 | Suite existante utile ; manque tests graphiques/persistance/qualité prédictive. |
| Documentation | 7/10 | README, audit précédent, model card, changelog ; manque doc des limites statistiques dans l'UI. |
| Sécurité/confidentialité | 6,5/10 | Auth optionnelle, secrets exemple ; URL Google Sheet par défaut dans config et détails erreurs activés. |

## 13. Anomalies détectées

| ID | Criticité | Fichier | Fonction/zone | Description | Impact | Cause probable | Correction proposée |
|---|---|---|---|---|---|---|---|
| A-001 | Critique | `Suivi_V1.py` | `load_remote_csv` | Déduplication automatique des dates avec conservation de la dernière ligne. | Mesures multiples le même jour supprimées au chargement distant. | Confusion entre préparation analytique et conservation source. | Corrigé : ne plus appeler `resolve_duplicates` dans le chargement source. |
| A-002 | Majeur | `app/core/weight_summary.py` | `prepare_weight_series` | Déduplication des dates pour calculs. | Les métriques peuvent ignorer une mesure intra-journalière. | Règle analytique implicite. | Documenter et éventuellement agréger selon stratégie utilisateur. |
| A-003 | Majeur | `app/pages/Predictions.py` | Auto-ARIMA/STL/SARIMAX | Modèles avancés exécutés sans seuils de couverture temporelle assez stricts. | Lenteur et sorties fragiles. | Blocs expérimentaux exposés directement. | Ajouter seuils nb points/span/couverture et cache. |
| A-004 | Majeur | `app/core/forecasting.py` | `forecast_with_ml` | Modèles quantiles entraînés sur tout l'historique, intervalles non calibrés sur backtest. | Intervalles trop optimistes. | Intervalle conditionnel présenté comme incertitude générale. | Calibrer avec erreurs walk-forward. |
| A-005 | Moyen | `app/pages/Insights.py` | KMeans | Clustering uniquement sur poids. | Information peu interprétable. | Modèle non supervisé sans features métier. | Marquer expérimental ou déplacer dans modèles avancés. |
| A-006 | Moyen | `app/core/insights.py` | `detect_anomalies_robust` | IsolationForest contamination fixe 10 %. | Peut signaler des fluctuations normales. | Hypothèse fixe d'anomalies. | Contamination adaptative et libellé « à vérifier ». |
| A-007 | Moyen | `app/utils.py` | Module entier | Ancienne pile redondante. | Divergence possible entre comportements testés et UI actuelle. | Migration progressive. | Déprécier ou réaligner avec `app/core`. |
| A-008 | Moyen | `Suivi_V1.py` | `init_data_once` | `except Exception` sans détail visible. | Diagnostic utilisateur limité. | Gestion réseau défensive. | Journaliser détail et afficher message actionnable. |
| A-009 | Mineur | `Dashboard.py` | `_render_smart_alerts` | Fonction définie mais non appelée. | Fonctionnalité invisible. | Régression ou fonctionnalité abandonnée. | Décider de restaurer ou supprimer après validation. |
| A-010 | Mineur | `.streamlit/config.toml` | `showErrorDetails=true` | Détails d'erreurs activés. | Peut exposer détails techniques en production. | Configuration dev. | Désactiver en production. |

## 14. Régressions fonctionnelles / fonctionnalités invisibles

- Fonctionnalité invisible : alertes intelligentes Dashboard définies mais non appelées.
- Fonctionnalité partiellement fonctionnelle : stratégie de doublons dans Paramètres n'est pas appliquée de manière visible au pipeline source ; elle existe plutôt comme choix potentiel.
- Fonctionnalité présente mais séparée : qualité des données intégrée dans Insights, pas accessible comme page dédiée.
- Fonctionnalité potentiellement abandonnée : ancienne couche `app/utils.py`, encore testée mais non centrale dans l'UI multipage.
- Fonctionnalité expérimentale exposée : Auto-ARIMA, STL/ACF/PACF, KMeans sans assez de pédagogie sur leurs limites.

## 15. Améliorations recommandées

### Corrections urgentes

- Corriger toute perte de lignes source lors du chargement distant. Fait.
- Ajouter test de non-régression pour conservation des mesures multiples même jour. Fait.
- Empêcher tout remplacement de source par données filtrées sans action explicite.

### Améliorations à forte valeur

- Ajouter une page ou tab « Qualité des données » avec diagnostics actionnables.
- Harmoniser les règles de mesures multiples : conserver source, choisir vue analytique explicite.
- Ajouter indicateur de fiabilité commun à chaque prévision.
- Calibrer les intervalles de prévision sur backtesting walk-forward.
- Clarifier dans l'UI : trajectoire cible vs prévision observée vs modèle expérimental.

### Améliorations secondaires

- Déprécier `app/utils.py` ou le transformer en wrappers vers `app/core`.
- Ajouter tests Plotly sur présence lignes objectifs et traces essentielles.
- Mieux organiser Insights en sections repliables.

### Améliorations expérimentales

- Modèle statistique simple sélectionné automatiquement après comparaison aux baselines.
- Détection de ruptures plus robuste si volume suffisant.
- Quantification empirique de l'incertitude avec erreurs historiques.

## 16. Roadmap priorisée

### Lot 0 — Sécurisation

- Fichiers : `Suivi_V1.py`, `app/core/data.py`, `app/core/session_state.py`, tests core.
- Modifications : garantir conservation lignes/colonnes, messages invalides, aucune mutation destructive implicite.
- Risques : changer des hypothèses analytiques existantes.
- Dépendances : tests fixtures multi-mesures.
- Critères : import distant/local conserve toutes les lignes valides ; aucune page ne perd les données en navigation.

### Lot 1 — Fiabilité analytique

- Fichiers : `weight_summary.py`, `analytics.py`, `insights.py`, `target_trajectory.py`.
- Modifications : libellés jours/mesures, seuils petits volumes, objectifs prise/perte.
- Risques : résultats changent légèrement.
- Dépendances : Lot 0.
- Critères : calculs 7j/30j/ETA/plateau documentés et testés sur cas limites.

### Lot 2 — Restauration fonctionnelle

- Fichiers : `Dashboard.py`, `Insights.py`, `Settings.py`.
- Modifications : décider sur alertes intelligentes, stratégie doublons visible, qualité données plus accessible.
- Risques : densité UI.
- Dépendances : Lot 1.
- Critères : fonctionnalités présentes dans le code visibles ou explicitement supprimées après décision.

### Lot 3 — UI/UX

- Fichiers : pages Streamlit, `app/ui/*`.
- Modifications : hiérarchie Dashboard / Journal / Analyses / Prévisions / Qualité / Paramètres sans refonte globale.
- Risques : perturbation habitudes utilisateur.
- Dépendances : Lots 0-2.
- Critères : moins de confusion, états vides et erreurs clairs, mobile lisible.

### Lot 4 — ML et forecasting

- Fichiers : `forecasting.py`, `evaluation.py`, `Predictions.py`, `models.py`.
- Modifications : baselines obligatoires, walk-forward réel, intervalles empiriques, refus si insuffisant, modèles complexes optionnels.
- Risques : suppression apparente de résultats si seuils stricts.
- Dépendances : Lots 0-1.
- Critères : aucun modèle ne présente une métrique in-sample comme prédictive.

### Lot 5 — Industrialisation

- Fichiers : tests, CI, README, docs.
- Modifications : tests graphiques, persistance session, datasets synthétiques, documentation limites.
- Risques : temps de maintenance.
- Dépendances : tous lots précédents.
- Critères : suite stable, rapide, sans réseau obligatoire.

## 17. Quick wins

- Ajouter un compteur de lignes avant/après import.
- Afficher clairement les lignes invalides supprimées au chargement.
- Ajouter `fast_mode` ou toggle utilisateur pour ne pas lancer Auto-ARIMA/STL par défaut.
- Ajouter des captions « expérimental » sur KMeans, IsolationForest et Auto-ARIMA.
- Ajouter tests sur traces Plotly : historique, objectif, intervalle.
- Désactiver `showErrorDetails` en production.

## 18. Fonctionnalités à ne pas modifier inutilement

- Centralisation `session_state` actuelle : bonne base.
- Conservation des colonnes supplémentaires par `clean_weight_dataframe`.
- Variations calendaires 7j/30j prudentes.
- Moteur plateau centralisé.
- Trajectoire cible avec paramètres métier centralisés.
- Backtesting des baselines chronologique.
- Import/export CSV du Journal.
- Authentification optionnelle par secrets.

## 19. Corrections appliquées dans cette intervention

- Correction A-001 : suppression de la déduplication destructive dans `load_remote_csv`.
- Ajout d'un test de non-régression : `test_clean_weight_dataframe_preserves_multiple_measurements_same_day`.

Les anomalies majeures, moyennes et mineures listées ci-dessus sont documentées mais non corrigées automatiquement, conformément à la consigne de limiter les changements aux anomalies critiques/bloquantes.
