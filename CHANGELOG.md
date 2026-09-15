# Changelog

## 2026-09-15 — Lectures par date, données API inexploitées et constats testés

Passe menée en trois temps : audit multi-lentilles du code (calculs, cohérence, UI/UX et dates, données API, physiologie), mesure du taux de fausse alerte de chaque constat sur 300 séries sans aucun lien réel, puis correction.

Lectures par date (demande explicite : les récupérations et les séances de boxe à leur date) :
- **Onglet Récupération** : une barre par matin noté, colorée de sa zone WHOOP avec les seuils 34 / 67 en fond, puis un tableau daté du plus récent au plus ancien — score, zone en toutes lettres, HRV, FC repos, nuit précédente, strain de la veille et du jour. Quatorze lignes visibles, l'historique complet derrière un expander.
- **Onglet Effort** : chaque séance à sa date, heure de début, sport en français, durée, strain, FC, calories, minutes en zones 4–5 et **la récupération du lendemain matin en face** (`session_log`) ; chronologie des séances colorée par sport ; répartition de l'intensité facile / modérée / dure par sport (`hr_zone_profile`). Une séance enregistrée à moins de 80 % est signalée.
- Les dates de tableau portent leur jour de semaine (`format_short_date` : « mar. 8 sept. ») ; les booléens se lisent « oui / non ».
- Bandes de référence WHOOP dessinées en fond des courbes de récupération (zones) et de strain (léger / modéré / élevé / maximal) ; l'empilement des stades de sommeil totalise désormais la nuit (léger et éveil ajoutés).

Données de l'API v2 récupérées mais jetées, désormais lues (`app/core/whoop.py`) : `zone_durations` (six zones de FC en minutes), `percent_recorded`, `altitude_gain_meter`, sommeil léger, éveil, cycles de sommeil, besoin de base, drapeaux « Calibration » et « Cycle en cours ». Les siestes ne tiennent plus lieu de nuit ; les récupérations reçoivent le fuseau de leur cycle.

Logique des calculs :
- **Rapport aigu/chronique découplé** : la fenêtre chronique couvre les 21 jours qui précèdent la semaine aigüe au lieu des 28 jours qui la contiennent — couplées, les fenêtres bornent le rapport par construction (Windt & Gabbett, Br J Sports Med 2019). Les bornes calendaires des deux fenêtres sont affichées. Le strain du cycle en cours, provisoire, est écarté de la charge et des « occasions manquées ».
- **Strain hebdomadaire** : moyenne et non somme — le strain est une échelle logarithmique, la page le disait elle-même deux sections plus loin.
- **Dette de sommeil cumulée** calculée contre le besoin de base : le besoin affiché par WHOOP chaque soir contient déjà le rattrapage des nuits précédentes.
- **Séries de journées rouges** comptées en jours calendaires consécutifs, plus en lignes mesurées.
- Incohérence entre 1,3 et 1,5 : le panneau disait « montée en charge soutenue » pendant que la carte disait « charge maîtrisée, dans la plage 0,8 à 1,3 ».

Constats testés plutôt que déclenchés sur un seuil brut — taux de fausse alerte mesurés sur des séries sans lien (avant → après) :
- « Vos meilleurs jours : … » 70 % → 4 % (test de Welch par facteur, seuil divisé par le nombre de facteurs ; colonne « Écart établi »).
- « Récupération en hausse / en baisse » 67 % → 6 % (Welch entre les deux semaines, seuil à 1 % pour compenser l'autocorrélation des nuits).
- « HRV sous votre repère » 55 % → 7 % : remplacé par la **moyenne mobile de sept jours face à la plage habituelle** des trente jours précédents (`smoothed_baseline`, méthode de Buchheit 2014 ; plage de ± 0,75 écart-type, calibrée sur simulation), avec le nombre de jours consécutifs hors plage. Idem pour la FC repos.
- « Creux récurrent le lundi » 31 % → 2 % (Welch contre les autres jours, Bonferroni sur les sept jours candidats).
- « La boxe pèse sur votre lendemain » 10 % → 3 % ; deux séances du même jour ne comptent plus qu'un lendemain.
- Puissance conservée : boxe à −15 points détectée dans 90 % des cas, sommeil explicatif dans 99 %.

Cohérence et lisibilité : même pente de poids pour la carte « apport estimé » et le panneau ; bandeau 7 j / 7 j indépendant de la période choisie ; coefficients non établis signalés dans « Moteurs de la récupération » ; heure de coucher rendue en horloge dans le contraste ; verdict « tenable » pour un objectif qui laisse plus de 2 000 kcal ; année affichée sur une date d'arrivée de l'année suivante ; sigles HRV / FC conservés ; accords (« zone verte », pluriels) ; journal replié au-delà de trois semaines ; sports nommés en français.

Tests : 487 passés, 0 échec (+52).

## 2026-09-13 — Cohérence des constats et nouveaux indicateurs

Passe de cohérence menée en générant les constats sur cinq profils d'utilisateur (perte régulière, surcharge, manque de sommeil, prise de poids, stagnation) puis en relisant les listes produites.

Le défaut principal : **l'application ne disait pas où allait le poids.**
- Sur un profil de prise de poids (+2,3 kg en 40 jours, 2,75 kg de retard sur la trajectoire), aucun des six constats affichés ne le mentionnait. La liste parlait de dette de sommeil, du creux du lundi et de l'apport estimé.
- Deux règles sont ajoutées : `_weight_direction_insight` énonce la tendance en kg/semaine et qualifie sa fiabilité, `_target_progress_insight` situe le poids face à la trajectoire cible.
- Nouvelle fonction `projected_goal_date` : date d'arrivée si le rythme actuel se maintenait, **confrontée à l'échéance visée**. Être en avance aujourd'hui ne dit rien de la date d'arrivée — dans un cas mesuré, « en avance » de 3,5 kg correspondait à une arrivée 109 jours après la date visée.
- La carte « votre objectif suppose X kcal/jour » passe de la priorité 95 à 74 : ce chiffre découle des paramètres de la cible et bouge à peine d'un jour sur l'autre, il monopolisait la première place tous les jours.

Autres correctifs de cohérence :
- **Deux familles de constats partageaient les mêmes icônes** : la flèche descendante désignait à la fois une baisse de poids et une baisse de récupération. Le poids prend la balance, la charge d'entraînement l'haltère, et un test vérifie désormais que chaque famille possède une icône unique.
- **Un avertissement de santé pouvait être masqué** par la limite d'affichage si six autres cartes se déclenchaient. Les constats peuvent désormais être épinglés ; seule la veille physiologique à plusieurs signaux concordants l'est, et elle survit à une limite d'un seul constat.

Nouveaux indicateurs :
- **Séries de journées** (`recovery_streaks`) : jours consécutifs dans la même zone, plus longue série verte et rouge. Une moyenne hebdomadaire lisse une succession de journées rouges ; sa durée dit si un état s'installe. Un constat se déclenche à partir de trois journées rouges consécutives.
- **Part de l'entraînement dans la dépense** (`training_energy_share`) : les calories s'additionnent, contrairement au strain qui est une échelle logarithmique. Répond à une question que WHOOP ne pose pas — le sport pèse-t-il dans la dépense, ou est-ce le quotidien qui la porte ?

Tests : 393 passés, 4 échecs préexistants inchangés. Relecture automatisée de 22 phrases générées sur trois profils : aucun gabarit de pluriel, aucune valeur non finie, aucun signe moins mal formé.


## 2026-09-12 — Lecture jour par jour et veille physiologique

Lecture chronologique (demande explicite : voir les récupérations et les séances par date) :
- **Nouvel onglet « Jour par jour »**, placé en deuxième position. Une carte par journée, du plus récent au plus ancien, avec la récupération et sa zone en liseré coloré, le sommeil et sa dette, l'heure de coucher, la charge, la HRV, la fréquence au repos, le poids et sa variation — puis **les séances du jour, chacune située par son heure de début** avec durée, charge, calories et fréquence maximale. Trois séances de boxe le même jour se lisent enfin comme trois séances distinctes.
- Les journées sans aucune mesure sont masquées par défaut et affichables au besoin, plutôt que de disparaître silencieusement.
- Export CSV du journal et vue tableau équivalente.

Veille physiologique (`vital_deviations`, `physiological_watch`) :
- **La fréquence respiratoire était renvoyée par WHOOP et jamais extraite.** C'est l'un des cinq signes vitaux nocturnes, et l'un des signaux les plus précoces d'une atteinte respiratoire. Elle est désormais captée.
- Les cinq signes vitaux sont comparés non à une norme de population mais à un **repère personnel robuste** : médiane et écart absolu médian des trente derniers jours. Sur dix à trente nuits, une seule nuit aberrante gonfle un écart-type classique et masque ensuite tout écart réel.
- Seule la direction cliniquement pertinente est signalée : une variabilité cardiaque haute n'est pas un signal à surveiller, une variabilité basse l'est.
- Le seuil a été **choisi par mesure** sur 200 séries sans anomalie : à 2,0 unités un signal apparaît une nuit sur cinq sans raison, à 2,5 une nuit sur treize, et deux signaux simultanés n'apparaissent jamais. Un indicateur de santé qui crie au loup finit ignoré.
- Cadrage strictement informatif : aucune pathologie n'est nommée, l'avertissement « ces mesures ne constituent pas un diagnostic » est affiché, et plusieurs signaux concordants renvoient vers un professionnel de santé. Les cartes de repère individuelles sont supprimées quand la veille les nomme déjà, pour ne pas dire trois fois la même chose.

Architecture du sommeil (`sleep_architecture`) :
- Part de sommeil profond et de sommeil REM dans la nuit, rapportée aux plages usuellement citées chez l'adulte. WHOOP affiche des heures ; c'est la proportion qui se compare, une nuit courte réduisant mécaniquement les heures de chaque stade.
- La moyenne porte sur les parts de chaque nuit et non sur la part du total, pour qu'une nuit très longue ne pèse pas davantage qu'une nuit courte.
- Les plages sont présentées comme des repères de population, jamais comme des objectifs personnels.

Tests : 368 passés, 4 échecs préexistants inchangés. Les seuils de la veille physiologique sont validés par simulation, la robustesse de la dispersion par comparaison directe à l'écart-type, et la moyenne des parts de sommeil par un cas où les deux méthodes divergent.


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
