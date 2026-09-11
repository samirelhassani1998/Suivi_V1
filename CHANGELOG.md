# Changelog

## 2026-09-11 — Intégration WHOOP

- Ajout d'un onglet `Whoop` (`app/pages/Whoop.py`) connecté à l'API WHOOP v2 en lecture seule via OAuth 2.0 : autorisation, échange de code, rafraîchissement automatique du jeton et déconnexion.
- Ajout de `app/core/whoop.py` : construction de l'URL d'autorisation, appels paginés (`limit`, `start`, `end`, `nextToken`), normalisation des récupérations, nuits, cycles et séances en DataFrames, agrégat journalier, croisement avec le poids et corrélations. Le transport HTTP est injectable, donc testable hors ligne.
- Onglets internes : vue d'ensemble, récupération et HRV, sommeil, effort et séances, croisement « Poids × WHOOP » avec export CSV.
- Isolation stricte : les données WHOOP vivent dans des clés de session dédiées (`whoop_*`) et ne modifient jamais `source_data`, `working_data` ni `filtered_data`.
- Identifiants lus depuis les secrets Streamlit (`[whoop]` ou clés à plat), les variables d'environnement ou une saisie de session ; `.streamlit/secrets.toml` est désormais ignoré par Git.
- Ajout de `requests` aux dépendances, de `tests/test_whoop.py` (32 tests hors ligne) et de 5 smoke tests Streamlit sur le nouvel onglet.


## 2026-04-17
- Refonte V2: nouvelle navigation 5 pages (Dashboard/Journal/Prévisions/Insights/Paramètres).
- Refactor modulaire `app/core/*` et `app/ui/*`.
- Ajout fallback upload CSV local si Google Sheets indisponible.
- Ajout couche qualité des données + score + diagnostics.
- Ajout backtesting walk-forward et leaderboard des baselines.
- Ajout intervalles de prédiction (SARIMAX + quantile regression), ETA prudente, plateau 14j/30j.
- Ajout tests unitaires core + smoke tests Streamlit AppTest.

## Lot 0 — Sécurisation du pipeline de données

- Clarification du cycle `source_data` / `working_data` / `filtered_data` / `analysis_data` : les sources validées, les éditions Journal, les vues filtrées et les copies analytiques sont séparées par copies profondes.
- Conservation explicite des mesures multiples le même jour et des colonnes additionnelles à l'import distant, à l'import CSV local, dans le Journal et à l'export.
- Ajout d'un rapport qualité indiquant lignes lues, valides, invalides, dates dupliquées, colonnes conservées, colonnes additionnelles et raisons de rejet.
- Limitation des stratégies de doublons (`garder_la_derniere`, moyenne, médiane) à la préparation d'une copie `analysis_data`.
- Ajout de tests de non-régression sur `load_remote_csv`, l'import local, la persistance session, les colonnes additionnelles, les lignes invalides et les copies analytiques.
- Ajout d'un workflow GitHub Actions minimal exécutant `pytest -q` après installation de `requirements.txt`.

Limites restantes : l'environnement local du conteneur bloque l'accès au registre Python, donc la suite `pytest` doit être validée par la CI ou un environnement disposant des dépendances.
