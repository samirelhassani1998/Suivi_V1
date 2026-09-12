# Changelog

## 2026-09-12 — Révision de la logique des calculs

Audit de l'onglet WHOOP sur sept angles (calculs, statistiques, normalisation, UI, dates, robustesse, constats). Les défauts ci-dessous ont été reproduits avant correction.

Plantage et données fausses :
- **La page entière plantait** (`ValueError: cannot reindex on an axis with duplicate labels`) dès que WHOOP renvoyait deux récupérations le même jour : `recoveries_to_frame` était le seul convertisseur sans déduplication. La source est corrigée et `daily_grid` tolère désormais les doublons plutôt que de lever une exception, étant trop en aval pour emporter l'affichage.
- **Le sommeil était surestimé** : `total_in_bed − total_awake` comptait le temps où le capteur perd le signal comme du sommeil (8 h affichées pour 7 h réelles).
- **Les enregistrements non notés** (`PENDING_SCORE`, `UNSCORABLE`) étaient conservés et écrasaient la mesure valide du même jour à la déduplication.
- **La date des récupérations ignorait le fuseau horaire**, contrairement au sommeil et aux cycles : une récupération créée à 23h30 UTC était datée de la veille.
- **« Les 7 derniers jours » désignait les 7 dernières lignes** dans quatre fonctions. Sur un bracelet porté par à-coups, la « dette des 7 dernières nuits » incluait une nuit vieille de six semaines. Introduction de `last_days` et `previous_days`, qui filtrent sur le calendrier.
- **La variation de poids entre deux pesées espacées** était traitée comme une variation journalière dans les corrélations : trois kilos sur dix jours pesaient comme trois kilos en un jour. La cible devient le rythme en kg/jour, et la durée écoulée est exposée.

Honnêteté statistique :
- **Les corrélations décalées testaient une cinquantaine d'hypothèses** et affichaient la plus forte : le constat se déclenchait sur presque toutes les séries de bruit. Après correction pour tests multiples, le taux de faux positifs mesuré tombe de la quasi-totalité à 5 % sur 40 simulations, l'effet réel injecté restant détecté.
- **Le rapport aigu/chronique valait exactement 1,00** sous 14 jours d'historique, les deux fenêtres portant sur les mêmes jours : « charge maîtrisée » était garanti. Il se tait désormais, et refuse aussi de conclure sur une semaine où le bracelet n'a été porté que deux jours.
- **Le R² de la régression de récupération n'était pas ajusté** : il dépassait 0,25 sur du bruit pur à douze observations, déclenchant à tort le constat sur le sommeil.
- **Le bilan énergétique publiait un apport au kcal près** sans incertitude, et estimait la pente sur les seuls jours croisés avec WHOOP en ignorant les pesées intermédiaires. Il expose désormais son intervalle à 95 % et utilise toutes les pesées de la période.
- **Le contraste bons/mauvais jours classait par écart brut**, donc par unité de mesure : un écart de 0,6 point de strain devançait 0,5 h de sommeil. Le classement se fait sur la taille d'effet.
- **`indexed_series` inversait le sens** des métriques négatives ou traversant zéro : se coucher plus tôt (−2 h) donnait un indice de 200. Ces métriques sont écartées du sélecteur.
- **La couverture annonçait 100 %** dès qu'une seule des trois sources répondait ; les jours complets sont désormais comptés séparément.

Nouveau croisement :
- **Votre objectif traduit en calories** (`target_pace_feasibility`) : le rythme de la trajectoire cible est converti en déficit, puis confronté à la dépense mesurée, pour donner l'apport quotidien que l'objectif suppose. Le résultat est qualifié, et un apport sous les repères usuels renvoie vers un professionnel de santé plutôt que vers un tableau de bord.

Lisibilité :
- Les heures de coucher s'affichaient en décimal signé (« −1,5 ») dans les graphiques : elles se lisent en horloge sur l'axe comme au survol.
- Deux unités partageaient un axe nommé « Valeur », écrasant la courbe la plus basse : HRV, fréquence au repos, température et SpO2 ont chacune leur graphique.
- La jauge de récupération n'avait pas de titre et affichait 0 % (zone rouge) pour une mesure absente.
- La colonne « Semaine » affichait une date nue, la branche de formatage étant inatteignable.
- Le calendrier dépassait 2 000 pixels de haut sur un an d'historique.
- Le croisement « Poids × WHOOP » passe du cinquième au deuxième onglet, et la plomberie de connexion est repliée derrière les données.
- Les « jours à surveiller » mêlaient alertes et occasions manquées : les deux sont séparés.
- Le strain était coloré comme une mauvaise nouvelle à la hausse ; il est désormais neutre.
- Accords grammaticaux dans les constats, à la place des « (s) ».
- Retrait du sélecteur de thème Plotly, qui était du code mort : l'onglet utilise une palette fixe validée pour la vision des couleurs.

Tests : 341 passés, 4 échecs préexistants inchangés. Les correctifs statistiques sont validés par simulation (taux de faux positifs mesuré, effets injectés retrouvés), et non par relecture.


## 2026-09-11 — Dates lisibles et analyse des bons jours

Lisibilité des dates (`app/core/date_labels.py`, nouveau module) :
- Mois et jours en français partout : `3 sept.` sur les axes au lieu de `Sep 3`, `jeudi 3 septembre 2026` dans les phrases, `Semaine du 3 août` en tête de ligne. Les noms sont codés dans l'application plutôt que tirés de la locale système, dont l'absence ferait silencieusement réapparaître l'anglais.
- Ancienneté en clair : `aujourd'hui`, `hier`, `il y a 5 jours`.
- Durées en heures et minutes : `1 h 30` au lieu de `90,0`, `43 min` au lieu de `43,4038`.
- Heures de coucher rendues en horloge (`22:30`) plutôt qu'en décimal signé.
- Dernière synchronisation affichée avec son ancienneté et son heure exacte.

Corrections relevées à la relecture :
- **Les séances perdaient leur heure de début** : trois entraînements le même jour donnaient trois lignes identiques au lecteur. `workouts_to_frame` expose désormais une colonne `Début` en heure locale, et trie les séances chronologiquement.
- **Le filtre de période comptait depuis la dernière mesure**, pas depuis aujourd'hui : après une semaine sans porter le bracelet, « 7 jours » affichait silencieusement une autre tranche. Il compte désormais à partir du jour courant.
- **Rien ne signalait des données anciennes** : un bandeau indique depuis quand les mesures s'arrêtent et avertit au-delà de deux jours.
- Un test dépendait d'un échec réseau réel pour vérifier une gestion d'erreur : sur une machine connectée, il envoyait un code d'autorisation bidon aux serveurs WHOOP. L'échange est désormais simulé.

Nouvelles analyses :
- **Vos bons jours contre vos mauvais** (`contrast_best_worst_days`) : comparaison du tiers supérieur au tiers inférieur de récupération sur le sommeil, la dette, l'heure de coucher, la charge et les perturbations, classée par écart décroissant.
- **Coût de chaque sport** (`sport_recovery_impact`) : récupération du lendemain sport par sport, rapportée à votre moyenne.
- Les deux alimentent le moteur de constats rédigés.

Tests : `tests/test_date_labels.py` (29 tests) couvre les libellés et leurs cas dégradés ; les analyses ajoutées sont validées sur des effets injectés volontairement (le facteur planté ressort bien en tête, le sport pénalisant bien en premier). Total : 287 passés, 4 échecs préexistants inchangés.


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
