# Changelog

## 2026-09-11 — Lecture narrative et refonte visuelle de l'onglet WHOOP

Nouvelles analyses (`app/core/whoop_analytics.py`) :
- **Moteur de constats rédigés** (`generate_insights`) : dix règles produisent des phrases françaises chiffrées, classées par importance — montée de charge brutale, dette de sommeil accumulée, apport calorique estimé, ce qu'une heure de sommeil rapporte, creux récurrent par jour de semaine, écart au repère personnel, port irrégulier du bracelet. Chaque règle reste muette sous son effectif minimal.
- **Repère personnel** (`personal_baseline`) : comparaison à la médiane des 30 derniers jours, la dernière mesure étant exclue de sa propre référence.
- **Lissage** (`rolling_trend`) : moyenne glissante 7 jours superposée aux séries bruitées.
- **Base commune** (`indexed_series`) et **calendrier** (`calendar_matrix`).

Refonte visuelle (`app/ui/whoop_visuals.py`, nouveau module sans dépendance à Streamlit) :
- **Suppression du graphique à double axe vertical** poids / métrique WHOOP. Caler deux échelles l'une sur l'autre fabrique une corrélation visuelle arbitraire ; les deux séries sont désormais ramenées à une base 100 commune sur un axe unique.
- Palette catégorielle **validée pour la vision des couleurs** (séparation ΔE contrôlée), couleurs de statut réservées aux significations bon/mauvais.
- Légende supprimée pour les séries uniques, grille en filet continu, marqueurs cerclés de surface, hauteurs incluant la bande d'axe.
- La courbe de tendance lissée déclare elle aussi `connectgaps=False` : elle enjambait les jours manquants.
- Ajout d'une **jauge de récupération** avec les seuils WHOOP, d'un **calendrier semaine × jour**, de **sparklines** sous les indicateurs, et d'un **profil par jour de semaine** en série unique.

Ergonomie :
- **Filtre de période unique** (7 / 30 / 90 jours / tout) placé au-dessus des onglets et appliqué à tous.
- Vue tableau disponible sous chaque graphique coloré.
- Barres de progression indiquant ce qui se débloquera et quand, à la place des seuls messages textuels.

Tests : `tests/test_whoop_visuals.py` (20 tests) vérifie l'encodage des figures — absence de double axe, rupture des courbes sur les trous, ordre fixe des couleurs, légende conditionnelle, axe calendaire, seuils de la jauge. `tests/test_whoop_analytics.py` passe à 45 tests. Quatre tests de page supplémentaires couvrent les constats, la jauge, le filtre de période et la vue tableau.


## 2026-09-11 — Analyses WHOOP approfondies

Ajouts (`app/core/whoop_analytics.py`) :
- **Bilan énergétique estimé** : apport calorique moyen déduit de la dépense WHOOP et de la pente du poids (7 700 kcal/kg).
- **Corrélations décalées** (0, 1, 2 jours) entre chaque métrique et la variation de poids, remplaçant la corrélation simple à décalage nul.
- **Charge d'entraînement** aigüe (7 j) rapportée à la charge chronique (28 j), avec lecture du rapport.
- **Moteurs de la récupération** : régression du score sur le sommeil et la charge de la veille, avec pouvoir explicatif affiché et avertissement lorsqu'il est faible.
- **Dette de sommeil** à partir du besoin renvoyé par WHOOP, **zones de récupération** (seuils rouge/jaune/vert), **synthèse hebdomadaire**, **profil par jour de semaine**, **régularité du coucher** et **jours à surveiller** (charge élevée sur récupération basse).
- Panneau de fiabilité indiquant la couverture réelle et le nombre de jours restants avant que chaque analyse ne se débloque.

Corrections d'affichage relevées en conditions réelles :
- Les courbes reliaient deux mesures espacées de plusieurs jours par une droite, donnant l'illusion d'une tendance mesurée. Les séries sont désormais posées sur un calendrier continu avec `connectgaps=False`.
- Avec deux ou trois points, Plotly graduait l'axe en heures (`03:00`, `06:00`) sur des mesures quotidiennes ; l'axe est forcé au format calendaire.
- Les tableaux affichaient les décimales brutes de l'API (`43.4038` min, `505.512` kcal), des horodatages (`2026-09-10 00:00:00`) et `None` pour les valeurs absentes. Tout est désormais mis en forme.
- Le graphique de synthèse superposait récupération (0-100), sommeil (heures) et strain sur un axe commun, écrasant les deux derniers. Chaque indicateur a son propre graphique.
- Les métriques encore nulles chez WHOOP (régularité du sommeil sur historique court) sont masquées avec une mention explicite au lieu d'afficher une ligne plate à zéro.
- Enrichissement de `sleeps_to_frame` : besoin de sommeil, dette et heure de coucher.
- `correlation_table` est retirée au profit de `lagged_correlations`, qui la couvre entièrement au décalage 0.

Tests : `tests/test_whoop_analytics.py` (28 tests), dont plusieurs vérifient que les régressions retrouvent des coefficients injectés volontairement et qu'elles n'ajustent pas du bruit sur un historique court. Quatre tests de page supplémentaires couvrent le déblocage des analyses, le cas d'un historique de quatre jours, la non-interpolation des trous et la mise en forme des tableaux.


## 2026-09-11 — Correctif du retour d'autorisation WHOOP

- Correction d'un défaut bloquant : lorsque la *Redirect URI* déclarée chez WHOOP pointait vers la racine de l'application, le retour d'autorisation affichait le Dashboard et le code était perdu, rendant la connexion impossible. Le code est désormais capté par `Suivi_V1.py` avant la porte d'authentification et avant tout rendu de page, puis l'onglet Whoop est activé automatiquement (`app/core/whoop_session.py`).
- La *Redirect URI* est détectée depuis les en-têtes de la requête au lieu d'être saisie à l'aveugle, et l'onglet affiche l'URL exacte à déclarer côté WHOOP.
- Les erreurs renvoyées par WHOOP (`invalid_request`, `access_denied`) sont interprétées et accompagnées de la marche à suivre, au lieu d'aboutir à une page d'erreur du fournisseur sans explication.
- Les paramètres OAuth sont retirés de l'URL après lecture, ce qui évite de rejouer un code déjà consommé au rafraîchissement.
- Ajout d'un réglage permettant de retirer le scope `offline` si l'application WHOOP ne l'autorise pas.
- Ajout de `tests/test_whoop_oauth_callback.py` (15 tests), dont deux vérifient que le code survit à la porte d'authentification — ils échouent bien avec l'ordre d'exécution fautif.


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
