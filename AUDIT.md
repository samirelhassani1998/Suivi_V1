# Audit applicatif : Suivi_V1 (Streamlit)

## 1) Architecture actuelle

- **Entrée Streamlit** : `Suivi_V1.py` configure l'application en mode wide, applique le thème global, vérifie l'authentification, charge la source Google Sheets/CSV et déclare la navigation multipage.
- **Pages principales** :
  - `app/pages/Dashboard.py` : tableau de bord de suivi du poids, KPIs, tendances, objectifs, graphiques et résumé actionnable.
  - `app/pages/Journal.py` : édition/import des mesures en session.
  - `app/pages/Predictions.py` : prévisions et estimation d'atteinte d'objectif.
  - `app/pages/Insights.py` : analyses complémentaires.
  - `app/pages/Settings.py` : paramètres utilisateur, taille, objectifs et options.
- **Couche données** : `app/core/data.py` normalise les colonnes, convertit dates/poids, conserve les colonnes utilisateur, supprime les lignes invalides et résout les doublons.
- **État applicatif** : `app/core/session_state.py` centralise les DataFrames source/travail/filtrés et évite de perdre les données lors de la navigation.
- **Analytique** : `app/core/analytics.py`, `app/core/insights.py`, `app/core/forecasting.py`, `app/core/evaluation.py` fournissent tendances, plateaux, scores, prévisions et backtests.
- **Objectifs** : `app/core/targets.py` maintient cinq objectifs configurables et migre les anciennes sessions à quatre objectifs.
- **UI** : `app/ui/components.py` contient les composants simples réutilisables ; `app/ui/theme.py` applique le style global.

## 2) Source et cycle de vie des données

1. `Suivi_V1.py` lit `st.secrets["data_url"]` ou `DATA_URL` depuis `app/config.py`.
2. Le CSV est chargé par `load_remote_csv()`, mis en cache cinq minutes, nettoyé par `clean_weight_dataframe()` puis dédoublonné avec la stratégie `garder_la_derniere`.
3. Les données nettoyées sont stockées dans `st.session_state` via `set_source_data()`.
4. L'utilisateur peut recharger la source, réinitialiser les modifications locales ou importer un CSV depuis la sidebar.
5. Les pages consomment `get_filtered_or_working_data()` pour afficher soit la vue filtrée, soit les données de travail.

## 3) Points forts constatés

- Structure déjà modulaire : données, analytics, prévisions, cibles, UI et pages sont séparées.
- Les colonnes supplémentaires de l'utilisateur sont préservées lors du nettoyage.
- Les doublons de dates sont gérés explicitement.
- Les objectifs multiples sont centralisés et compatibles avec les anciennes sessions.
- Des garde-fous existent déjà pour les prévisions, les plateaux, l'ETA et les données insuffisantes.
- La suite de tests couvre plusieurs modules critiques et des smoke tests Streamlit.

## 4) Points faibles avant correction

- Le haut du dashboard mélangeait KPIs existants, scores avancés et résumés sans vue quotidienne très lisible.
- Les variations demandées explicitement sur 7 jours, 30 jours, depuis le début, depuis la mesure précédente, et l'écart objectif n'étaient pas regroupées dans une zone unique.
- Les moyennes mobiles du graphique principal dépendaient surtout d'une fenêtre en nombre de mesures ; une moyenne mobile calendaire 7j/30j était utile pour des mesures irrégulières.
- Les projections existaient ailleurs, mais le dashboard principal ne présentait pas une projection courte et prudente directement liée à l'objectif final.
- Les insights automatiques n'étaient pas suffisamment visibles pour une lecture quotidienne rapide.
- Le thème global était minimal : cartes KPI peu hiérarchisées, peu de contraste visuel et peu d'optimisation mobile.
- Une couche défensive supplémentaire était souhaitable pour éviter qu'une donnée de session mal formatée produise un calcul trompeur sur le dashboard.

## 5) Améliorations implémentées

- Ajout d'un module de synthèse robuste `app/core/weight_summary.py` pour préparer les séries de poids, calculer les deltas calendaires, les moyennes mobiles 7j/30j, la tendance, le rythme moyen, les stagnations, l'écart objectif et une projection prudente.
- Ajout d'une **Vue rapide** en haut du dashboard : poids actuel, variation depuis le début, variation 7j, variation 30j, moyenne mobile 7j, tendance, poids min/max, rythme moyen et écart à l'objectif.
- Ajout d'**insights automatiques** simples et visibles avant le résumé actionnable existant.
- Amélioration du graphique principal avec poids mesuré + moyenne mobile 7j + moyenne mobile 30j + EMA + lignes d'objectifs et trajectoire existante.
- Ajout d'un graphique optionnel d'écart à l'objectif final.
- Renforcement du thème : hero discret, cartes KPI plus lisibles, bordures légères, espacements et règles mobiles.
- Ajout de tests unitaires ciblant les nouveaux calculs de synthèse.

## 6) Robustesse ajoutée

- Les métriques quotidiennes passent par une préparation défensive : dates invalides supprimées, poids non numériques/vides ignorés, virgules décimales acceptées, poids <= 0 exclus et doublons de date dédupliqués.
- Les deltas 7j/30j affichent `N/A` et une explication si l'historique est insuffisant.
- La projection vers l'objectif n'est affichée que si la tendance récente est suffisamment claire, avec une formulation prudente.
- Les insights signalent explicitement quand les calculs restent à confirmer faute de données.

## 7) Risques et points à surveiller

- Les objectifs restent configurés via `Settings.py` et la session Streamlit ; il faudra vérifier que les utilisateurs comprennent quel objectif est l'objectif final.
- Les projections linéaires restent indicatives : elles ne doivent pas être interprétées comme une promesse médicale ou sportive.
- L'environnement local de validation peut manquer de dépendances lourdes (`numpy`, `pandas`, `streamlit`) si `pip install -r requirements.txt` est bloqué par le réseau.
- L'application déployée dépend toujours de l'accessibilité du Google Sheet configuré.

## 8) Validation recommandée après déploiement

- Ouvrir le dashboard sur desktop et mobile pour vérifier que la nouvelle Vue rapide reste lisible.
- Vérifier le comportement avec un historique court (1 à 4 mesures) : pas d'exception et messages prudents.
- Vérifier un CSV avec dates mal formatées, poids vides ou virgules décimales.
- Vérifier que les pages Journal, Prévisions, Insights et Paramètres restent accessibles et que les données de session ne sont pas perdues.
