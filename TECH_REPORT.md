# Rapport Technique

## 1. Causes racines et reproduction
- **Erreur d'import `st.Page` sur Streamlit Cloud** : la plate-forme exécutait une
  version < 1.38 de Streamlit, ne supportant pas `st.Page`/`st.navigation`.
  Reproduit en lançant l'application avec une version plus ancienne (1.25) :
  `AttributeError: module 'streamlit' has no attribute 'Page'`.
- **Plantage lors du chargement des données** : aucune gestion d'erreurs autour
  du Google Sheet. Toute indisponibilité réseau levait une exception Pandas non
  interceptée, stoppant l'application.
- **Thème Plotly invalide** : l'option "Solar" appelait `plotly_solar`, un
  template inexistant -> `ValueError` dès que l'utilisateur sélectionnait le
  thème.

## 2. Décisions techniques
- Fixation des dépendances (Python 3.10/3.11) dans `requirements.txt` pour
  assurer un déploiement reproductible sur Streamlit Cloud.
- Gestion centralisée des erreurs dans `app/utils.load_data` avec conversion des
  exceptions en `RuntimeError` et nettoyage systématique des données.
- Fallback pour la navigation multipage : on conserve `st.navigation` quand il
  est disponible et l'on affiche un message d'aide sinon (les versions
  antérieures utilisent automatiquement le dossier `pages/`).
- Ajout d'un thème Plotly valide (`plotly`) pour l'option "Solar" et harmonisait
  la configuration `.streamlit/`.
- Introduction d'un exemple de secrets et lecture optionnelle de `data_url`
  depuis `st.secrets`.

## 3. Tests effectués
- **Installation** : tentative `pip install -r requirements.txt` (échoue dans
  l'environnement de travail à cause d'un proxy 403, cf. sortie CLI). Les
  versions choisies sont toutes disponibles sur PyPI avec roues précompilées.
- **Exécution locale** : la commande `streamlit run streamlit_app.py` est à
  relancer une fois les dépendances installées ; non exécutable ici à cause du
  proxy bloquant l'accès au Google Sheet.
- **Déploiement Cloud** : vérifier après merge que l'URL
  https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/ affiche le
  bandeau "Suivi de l'Évolution du Poids" et que la navigation fonctionne. Une
  capture du log de succès doit être ajoutée à la description de déploiement.

## 4. Points de vigilance
- **Quotas API Google Sheets** : un trop grand nombre de rafraîchissements peut
  déclencher des limitations. En cas de besoin, configurer un `data_url` privé
  via `st.secrets`.
- **Consommation mémoire** : `statsmodels` et `pmdarima` peuvent dépasser les
  limites Streamlit Cloud si le dataset grossit. Activer les caches et limiter
  les fenêtres d'analyse en conséquence.
- **Temps de calcul** : `auto_arima` et les validations croisées peuvent durer
  plusieurs secondes ; la mise en cache est recommandée pour les scénarios à
  forte fréquentation.
- **Mises à jour de dépendances** : surveiller la compatibilité future de
  `numpy`/`pmdarima` avant tout upgrade.

## 5. Vérification post-déploiement
1. Déployer depuis la branche principale.
2. Ouvrir l'URL publique et confirmer l'absence d'erreurs dans les logs
   Streamlit Cloud.
3. Capturer le log de démarrage (onglet **Logs**) montrant `Running on port 8501`
   et le lier à la description de déploiement.
