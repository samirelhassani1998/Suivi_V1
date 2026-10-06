# Audit complet et améliorations — Suivi V1

**6 octobre 2026.** État initial : `35909e5`, après intégration de la PR #99. Cet audit remplace l’état des lieux du 7 août. Il couvre les sept pages, leurs sous-onglets, les composants communs, le cycle des données, WHOOP, les statistiques, les prévisions et la validation. Les observations portent sur le code et des données synthétiques ; aucune donnée personnelle ni connexion WHOOP réelle n’a été utilisée.

## 1. Conclusion et ordre des priorités

Le projet possède déjà un socle utile : journal éditable, données source conservées, trajectoire distincte des mesures, tendance robuste, baselines, validation chronologique, corrections des comparaisons multiples, analyses WHOOP et Boxe, tests de pages. L’enjeu principal est de rendre les résultats aussi fiables que leur présentation le laisse penser.

L’audit a reproduit des pertes possibles à l’enregistrement, des incohérences de périodes, des prévisions numériquement instables et des conclusions issues de données manquantes. Les corrections de cette PR ciblent ces problèmes et améliorent les interactions communes. Les changements de méthode qui exigent une vraie calibration, ainsi que le stockage durable, restent explicitement dans la feuille de route.

| Priorité | Résultat recherché | Traitement dans cette PR |
|---|---|---|
| P1 — données et accès | Une erreur de saisie/import ne détruit pas le travail ; retour OAuth vérifié | Validation bloquante, remplacement protégé, configuration atomique, état OAuth à usage unique, authentification robuste |
| P1 — exactitude analytique | Les périodes, effectifs, unités et jours comparés correspondent aux libellés | Filtres cohérents, interruptions préservées, plateaux avec recul suffisant, qualité WHOOP conservée, charge Boxe cohérente |
| P1 — prévisions | Aucun objectif artificiellement atteint par divergence ou fuite de cible | Variables autorisées, cadence explicite, quantiles régularisés, refus des sorties instables |
| P2 — compréhension | Distinguer mesure, estimation, résultat incertain et calcul impossible | Textes descriptifs, couverture réelle, raisons d’indisponibilité, interactions graphiques et accessibilité améliorées |
| P2/P3 — évolution | Mesurer les gains avant d’ajouter des modèles ou de l’IA | Feuille de route avec critères d’acceptation ci-dessous |

## 2. Améliorations appliquées et preuves

### Données, Journal et Paramètres

Références : [`Suivi_V1.py`](Suivi_V1.py), [`data.py`](app/core/data.py), [`Journal.py`](app/pages/Journal.py), [`Settings.py`](app/pages/Settings.py), [`session_state.py`](app/core/session_state.py).

- **Édition sans suppression silencieuse.** Effacer la date d’une ligne produisait un simple avertissement, puis le nettoyage supprimait la mesure à l’enregistrement. Les lignes invalides bloquent désormais l’enregistrement. L’état modifié compare l’éditeur brut aux données enregistrées, y compris les nouvelles lignes incomplètes. L’ajout rapide demande de terminer les modifications de l’éditeur avant d’ajouter une pesée.
- **Import contrôlé.** Prise en charge des CSV à virgule, point-virgule ou tabulation, BOM UTF-8 et encodage Windows courant. Les en-têtes ambigus sont refusés avant leur renommage automatique par pandas. Un fichier vide, sans colonnes requises ou sans mesure valide ne remplace pas la session. Le téléchargement distant utilise le même parseur, vérifie la réponse HTTP et limite les délais de connexion/lecture à 5/20 secondes.
- **Valeurs et dates.** Les poids infinis sont rejetés. Une valeur finie hors des bornes du formulaire reste corrigeable dans le journal, sans faire planter l’ajout rapide. Le numéro de jour Excel `46000` donne le même jour sous forme numérique ou textuelle, au lieu de devenir une date de 1970.
- **Protection du travail local.** Importer, recharger ou réinitialiser exige une confirmation dans l’application lorsque des modifications locales existent. Chaque remplacement renouvelle cette confirmation et réinitialise l’éditeur pour empêcher la réapplication d’anciennes modifications. Le chargement initial ne remplace plus automatiquement des données de travail déjà présentes. Cette protection ne constitue pas une sauvegarde durable.
- **Paramètres atomiques.** Objectifs positifs et finis, ordre des paliers et dates du zoom sont validés avant toute écriture. Une erreur de dates ne sauvegarde plus simultanément une nouvelle taille et une cible invalide.

### Dashboard et Insights

Références : [`Dashboard.py`](app/pages/Dashboard.py), [`Insights.py`](app/pages/Insights.py), [`analytics.py`](app/core/analytics.py), [`plateau.py`](app/core/plateau.py), [`trend.py`](app/core/trend.py).

- **Périmètre commun.** « Effort actuel » s’applique aussi aux anomalies et aux fluctuations. Les dates et l’effectif du périmètre sont affichés.
- **Interruptions réelles.** Deux phases de perte séparées de plusieurs mois ne fusionnent plus en une phase continue. Le cas reproduit de deux blocs de 14 jours devenait auparavant une phase de 104 jours.
- **Plateau avec recul suffisant.** Quatre pesées constantes sur quatre jours ne suffisent plus à déclarer un plateau de 14 ou 30 jours. Les fenêtres demandent respectivement au moins 12 ou 28 jours de recul, en plus du nombre minimal de mesures ; une indisponibilité expose sa raison.
- **Couverture bornée.** La régularité des 30 derniers jours compte 30 dates civiles distinctes, sans gonflement par doublons horaires. Le résultat ne peut plus afficher 31/30 ou 103,3 %.
- **Date d’objectif bornée.** Le sous-onglet Prévisions utilise le calcul d’arrivée partagé, limité à trois ans. Une pente très faible ne provoque plus de débordement de date et n’affiche pas une échéance démesurée.
- **Dispersion cohérente.** L’histogramme des fluctuations et sa bande portent tous deux sur les écarts à la tendance. Les différences entre deux pesées n’étaient pas comparables à une bande calculée sur les résidus.
- **Lecture plus précise.** Dernière pesée datée, fenêtres glissantes nommées comme telles, qualité présentée en tableau lisible. Une pente non significative devient une « direction non établie ». Les bandes et intervalles sont indiqués comme approximatifs ; l’association à un jour de semaine ne permet pas de conclure à une cause hydrique ou à une variation de graisse.

Le zoom conserve la simplification de la PR #99 : poids mesuré, trajectoire cible et objectifs.

### Prévisions, statistiques et ML

Références : [`Predictions.py`](app/pages/Predictions.py), [`evaluation.py`](app/core/evaluation.py), [`features.py`](app/core/features.py), [`forecasting.py`](app/core/forecasting.py), [`models.py`](app/core/models.py), [fiche des modèles](docs/MODEL_CARD.md).

- **Mesure ≠ jour.** Les modèles récursifs quotidiens refusent les jours manquants et les mesures multiples par jour au lieu de convertir implicitement une ligne en journée. La tendance qui utilise les dates réelles reste disponible. Le drift du benchmark utilise lui aussi le temps réel. Dans le cas reproduit de 40 pesées hebdomadaires à −0,2 kg par pesée, SARIMAX présentait auparavant sept semaines d’évolution comme J+7.
- **Prévention des fuites.** Les variables ML proviennent d’une liste explicite de dérivées du poids passé et du calendrier. Ajouter un IMC du jour ou une copie numérique de la cible à l’import ne peut plus produire artificiellement un score parfait. Des notes facultatives vides ne suppriment plus l’ensemble des lignes d’apprentissage.
- **Quantiles stabilisés.** Standardisation, régularisation, seuil effectif annoncé, ordre des quantiles et contrôle de finitude/stabilité avant application de la borne d’affichage. Le cas stationnaire `100 ± 0,5 kg`, 40 jours, graine 42, produisait auparavant une médiane brute divergente et un objectif de 80 kg affiché atteint en environ une semaine.
- **Évaluation honnête.** Les modèles incompatibles restent visibles comme « non évalués » avec une raison. La précision directionnelle ne compte plus les sauts entre blocs de réajustement comme des prédictions réussies. Les bandes expérimentales ne sont pas présentées comme calibrées.
- **Coût des modèles.** Les calculs expérimentaux coûteux sont déclenchés explicitement ; la synthèse et les références restent disponibles. Les corps des onglets Streamlit masqués ne doivent pas lancer implicitement tous les modèles.

La stabilité numérique, le tri des quantiles et un backtest chronologique ne prouvent ni un gain prédictif durable ni la calibration à 30 jours. Cette distinction est conservée dans la fiche des modèles.

### WHOOP

Références : [`Whoop.py`](app/pages/Whoop.py), [`whoop.py`](app/core/whoop.py), [`whoop_analytics.py`](app/core/whoop_analytics.py), [`whoop_session.py`](app/core/whoop_session.py).

- **Qualité propagée au croisement.** Les drapeaux de cycle ouvert et de calibration ne disparaissent plus lors de la jointure avec le poids. Un cycle encore partiel ne redevient pas une journée complète dans le bilan énergétique. Le cas de 13 cycles clos à 2 500 kcal et un cycle ouvert à 100 kcal faisait auparavant passer le seuil de disponibilité et abaissait artificiellement la moyenne.
- **Signaux comparables.** La veille physiologique rapproche des mesures de la même nuit, affiche leur date et le nombre réellement évalué. Une FC anormale ancienne ne s’additionne plus à une HRV anormale récente pour former une alerte « concordante ».
- **Couverture explicite.** « Journée complète » demande les sources nécessaires ; une récupération seule ne vaut plus une journée complète à 100 %. Les états de disponibilité suivent les critères des moteurs, notamment les deux fenêtres de charge.
- **Synchronisation complète ou erreur.** Une pagination malformée, répétée ou tronquée échoue explicitement au lieu de remplacer l’historique par un import présenté comme complet.
- **OAuth vérifié.** Le retour exige un état attendu non vide, concordant et consommé une seule fois. Lors d’un retour dans un nouvel onglet, un parcours permet de recopier l’URL de retour dans le champ masqué de l’onglet d’origine, qui détient encore l’état attendu. Si cet onglet a expiré, la connexion doit être relancée ; l’absence d’état n’est plus acceptée comme une autorisation. Les diagnostics de configuration évitent les chemins internes et utilisent une explication française.
- **Interprétation.** Les associations exploratoires et les marges énergétiques sont accompagnées de leurs limites ; les incertitudes du bracelet ne sont pas confondues avec la seule erreur de pente.

### Boxe

Références : [`Boxe.py`](app/pages/Boxe.py), [`boxing_analytics.py`](app/core/boxing_analytics.py), [`boxing_visuals.py`](app/ui/boxing_visuals.py).

- **Fenêtre de comparaison du poids.** Les journées exposées à la boxe sont comparées dans la période sélectionnée, avec leur résultat à J+1. L’historique extérieur ne peut plus être reclassé implicitement en journées sans boxe. La sonde initiale de 60 pesées, filtrées sur 30 jours de séances, comptait 49 journées « autres » au lieu de 19.
- **Charge et données manquantes.** L’unité est choisie de façon cohérente pour les fenêtres aiguë et chronique. Une séance sans TRIMP ne devient plus une charge nulle ; le repli en durée est annoncé, ou le résultat reste indisponible si la couverture ne suffit pas.
- **Une nuit, une observation.** Plusieurs séances le même jour ne multiplient plus artificiellement les nuits du test sur l’horaire d’entraînement.
- **Inférence indisponible visible.** Le statut du test circule jusqu’à l’interface. Des groupes constants ou trop petits ne produisent plus un message rassurant affirmant que les séances tardives n’affectent pas le sommeil. Absence d’effet détecté et absence d’effet démontrée restent distinctes.
- **Langage et énergie.** Les comparaisons sont décrites comme des associations. Une dépense brute ne remplace plus silencieusement une estimation nette pour annoncer une part du déficit couverte. Les équivalences énergétiques restent des scénarios, pas des prévisions de perte de poids.

### UI commune, accessibilité et accès

Références : [`theme.py`](app/ui/theme.py), [`charts.py`](app/ui/charts.py), [`components.py`](app/ui/components.py), [`whoop_visuals.py`](app/ui/whoop_visuals.py), [`auth.py`](app/auth.py).

- En-têtes plus compacts pour remonter les indicateurs utiles.
- Clic sur une légende pour masquer/afficher une série ; double-clic pour l’isoler. Un clic ne fait plus disparaître les autres séries.
- Légendes du poids agrandies et contraste renforcé des libellés d’axes communs.
- Focus clavier visible, prise en compte de la préférence de réduction des animations, rôle et valeurs accessibles pour les barres de progression personnalisées.
- Mot de passe avec caractères accentués accepté par la comparaison sécurisée ; configuration absente traitée explicitement en conservant l’accès fermé. L’exemple de configuration place `data_url` à la racine TOML pour qu’il soit réellement utilisé.

Ces corrections ne constituent pas une certification d’accessibilité : lecteurs d’écran, zoom navigateur et interactions complexes des graphiques demandent des essais complémentaires.

## 3. Revue de chaque onglet et de ses sous-onglets

| Surface examinée | État utile / amélioration livrée | Prochaine amélioration proposée |
|---|---|---|
| Dashboard — vue rapide, courbe et zoom | Mesure datée, objectifs distincts, légende compréhensible, intervalles indicatifs | Limiter la synthèse initiale à 3–4 indicateurs et une action « Ajouter une pesée » ; conserver les détails accessibles |
| Dashboard — Analyse détaillée | Couverture et recul explicités, statistiques préservées | Réduire les constats répétés et relier chaque conclusion à sa preuve |
| Dashboard — Prévisions | ETA partagée et bornée | Un seul contrat de projection commun aux pages, avec même période et même cible |
| Dashboard — Historique | Résidus et bande cohérents | Points/moyennes hebdomadaires avec effectif, durée réellement couverte et semaines manquantes |
| Dashboard — Objectifs & paliers | Trajectoire fixe et paliers conservés | Clarifier « atteint par pesée » / « atteint par tendance » ; éviter une date trop précise pour chaque palier |
| Journal | Ajout rapide, validation bloquante, modifications protégées, export conservé | Édition adaptée au téléphone, export des lignes rejetées avec motifs et restauration d’une version |
| Prévisions — synthèse et leaderboard | Baselines, dates réelles, modèle incompatible explicitement indisponible | Benchmark par J+1/J+7/J+14/J+30 du pipeline réellement publié, largeur et couverture des bandes |
| Prévisions — Régression ML | Variables autorisées et horizon décrit correctement | Validation chronologique répétée, baseline de prochaine mesure, coût du calcul mesuré |
| Prévisions — SARIMA / Auto-ARIMA | Cadence et disponibilité contrôlées | Modèle d’état acceptant les jours manquants, diagnostics de convergence et budget de recherche |
| Prévisions — STL / ACF-PACF | Exploration séparée des projections | Afficher fraction observée/imputée, refuser les longs trous et expliquer la différence entre saisonnalité descriptive et effet établi |
| Prévisions — Scénarios | Trajectoires conditionnelles conservées | Exposer clairement les hypothèses et dates ; aucune probabilité sans modèle calibré |
| Insights — qualité, régularité, phases | Tableau lisible, 30 jours bornés, interruptions préservées, plateau avec recul | Comparaisons mois-à-date appariées ; détecteur de changements normalisé par les jours et réinitialisé aux interruptions |
| Insights — jour de semaine, séries | Association descriptive et correction multiple conservées | Effets avec intervalle temporel validé ; noms neutres « baisses/hausses observées » |
| Insights — Anomalies / Fluctuations | Même périmètre, résidus correctement comparés | Confirmer une anomalie et annoter ses conditions sans supprimer automatiquement la mesure |
| WHOOP — Vue d’ensemble | Couverture et disponibilité plus fidèles | Trois signaux prioritaires, fenêtre de référence toujours visible et preuve accessible |
| WHOOP — Jour par jour | Calendrier et statut provisoire conservés | Provenance et couverture par métrique ; identifiant source pour les mises à jour incrémentales |
| WHOOP — Poids × WHOOP | Qualité conservée à la jointure | Corrélations avec inférence temporelle validée ; bilan sur tous les cycles complets de la fenêtre |
| WHOOP — Récupération | Signaux d’une même nuit, date et nombre évalué | Références personnelles calculées sur le passé, pas sur toute la période lors d’un futur usage prédictif |
| WHOOP — Sommeil | Sous-composantes, dette et heures conservées | Comparaisons avec covariables et sensibilité aux nuits manquantes |
| WHOOP — Effort | Charge actuelle avec critères de disponibilité réels | Historique de synchronisation, couverture des fenêtres et attribution des séances multi-jours |
| Boxe — Vue d’ensemble | Interprétations prudentes et données insuffisantes visibles | Repère du jour distinct du filtre historique, nombre réduit de cartes |
| Boxe — Séances | Journal, durée, intensité et export | Détail d’une séance, types de travail et RPE facultatifs ; séances multiples d’un jour distinctes sur les graphiques |
| Boxe — Récupération | Statut des comparaisons conservé | Effets et intervalles, variables d’ajustement et observation par jour |
| Boxe — Sommeil | Une nuit par observation, aucune équivalence déduite d’un test indisponible | Relation continue entre horaire, intensité et sommeil ; seuil de 4 h présenté comme repère |
| Boxe — Charge & progression | Charge manquante distincte de zéro | Qualité capteur et unités visibles, séances comparables ; maxima de charge distincts de performance technique |
| Boxe — Poids & énergie | Période commune, brut/net explicites | Distributions des variations avec effectifs ; scénario énergétique avec compensation non observée explicitée |
| Paramètres | Validation globale avant sauvegarde | Diagnostics techniques repliés, choix de stockage, confirmation de taille et articulation cible fixe/paliers personnels |

## 4. Feuille de route priorisée

Les estimations suivantes représentent des lots de travail, pas des dates de livraison garanties. S = changement local ; M = plusieurs modules et validation ; L = évolution d’architecture ou expérimentation.

| Priorité / effort | Proposition | Critère d’acceptation |
|---|---|---|
| P1 / L | Sauvegarde durable, versionnée, avec restauration et identité utilisateur | Fermer puis reprendre une session sans perdre les pesées ; restaurer une version sans écraser des changements concurrents ; politique de suppression explicite |
| P1 / M | Dates civiles de pesées distinctes des instants UTC WHOOP | Une pesée à minuit en Europe/Paris conserve le jour choisi à l’import/export et au croisement ; tests minuit et changements d’heure |
| P1 / L | Calibration temporelle des intervalles et des tests | Mesurer couverture, largeur, faux positifs et puissance sur séries indépendantes, autocorrélées, trouées et à changement de régime ; publier les limites par effectif |
| P2 / M | Évaluation du pipeline complet par horizon | Même origine, mêmes dates et mêmes transformations qu’en production ; score, biais, couverture et nombre d’origines face à la baseline pour chaque horizon |
| P2 / M | Disponibilité et couverture par métrique | Une valeur partielle/manquante ne devient jamais zéro ; afficher nombre utilisé/éligible et raison des exclusions |
| P2 / M | Contrat de période partagé par toutes les pages | Début, fin, granularité, jours réellement observés et résultat J+1 explicités ; modifier des données hors fenêtre n’altère pas la comparaison |
| P2 / M | Tableau de données conservant les types au tri | Tri numérique 1, 2, 10 et chronologique correct, tout en conservant unités et format français ; export complet inchangé |
| P2 / M | Synchronisation WHOOP incrémentale et dédupliquée | IDs/updated_at conservés ; rejeu idempotent ; reprise bornée sur 429/5xx ; extraction correcte aux frontières de fuseaux |
| P2 / M | Performance mesurée et calculs à la demande | Temps froid/chaud mesuré pour 100/365/1 000 jours ; changement d’un contrôle simple sans réajuster tous les modèles |
| P2 / M | Tests d’accessibilité et parcours mobile | Clavier, zoom 200 %, lecteur d’écran et 320/390/768 px ; tout résultat essentiel accessible sans couleur seule ni infobulle obligatoire |
| P3 / M | Réduire les doublons de maintenance | Une seule définition des objectifs et du nettoyage ; déprécier progressivement `app/utils.py` derrière des adaptateurs testés |
| P3 / S–M | Configuration de déploiement indépendante d’une source personnelle | Source définie par configuration ; jeu de démonstration synthétique explicite ; aucun secret ou export individuel dans les fixtures/documentations |

### Direction ML / IA / apprentissage actif

1. **Comparer avant de complexifier.** Garder dernière valeur et tendance locale comme références. Tester ensuite un modèle d’état local linéaire acceptant les jours manquants, puis un petit modèle régularisé. Les réseaux profonds ne sont pas justifiés par le volume actuel d’une seule personne.
2. **Introduire WHOOP sans fuite.** Les variables doivent exister à l’origine de prévision ; références et normalisation apprises uniquement sur le passé. Évaluer plusieurs blocs chronologiques et s’abstenir si le gain sur la baseline n’est pas stable.
3. **Produire une synthèse fondée sur des preuves.** Un assistant peut reformuler les résultats calculés : période, effectif, valeur, intervalle et lien au graphique. Il ne doit pas calculer librement les statistiques, inventer une cause ou transformer une extrapolation en promesse. Un objet de faits vérifiable précède toute génération de texte.
4. **Collecter l’information utile.** Si « AL » désigne l’apprentissage actif, commencer par des demandes ciblées et facultatives : confirmer une pesée atypique, préciser les conditions de mesure ou le type/RPE d’une séance. Mesurer la pertinence des demandes et leur coût pour l’utilisateur ; ne pas supprimer automatiquement les observations rejetées par un algorithme.
5. **Séparer validité et utilité.** La MAE, la couverture et la calibration évaluent le modèle ; la compréhension, le taux de correction d’erreurs et l’utilité déclarée évaluent le produit. Un chatbot supplémentaire n’améliore pas ces mesures par lui-même.

## 5. Vérification et limites de l’audit

- **Référence avant modifications :** 594 tests passés sous Python 3.12. Les nouveaux cas reproduits montrent pourquoi une suite verte ne suffit pas à valider toutes les hypothèses statistiques.
- **Validation finale :** `pytest -q` sous Python 3.11.16, version majeure/mineure ciblée par `runtime.txt` : **698 tests réussis**, 9 avertissements, en 139 secondes. Les régressions ajoutées couvrent les pertes de données, les périodes, les prévisions, WHOOP, Boxe et l’authentification.
- **UI :** les sept pages sont chargées avec des données synthétiques dans Chromium aux largeurs 1 440 et 390 px, sans exception ni débordement horizontal de la page. Les 23 sous-onglets sont ouverts dans les deux formats ; l’isolation d’une bande par la légende est également vérifiée. Cette vérification ne remplace pas une étude utilisateur ni une certification WCAG.
- **Statistiques :** sondes synthétiques, pas d’évaluation de santé ni d’estimation de performance sur les données individuelles réelles. Les comparaisons répétées d’une même personne restent exposées à l’autocorrélation et aux facteurs de confusion.
- **Intégrations :** callbacks et réponses WHOOP simulés dans les tests ; pas de connexion OAuth réelle ni de synchronisation avec un compte personnel. Le retour OAuth après expiration complète de la session reste une limite UX documentée.
- **Livraison :** cette PR corrige les défauts décrits comme appliqués. Les propositions de la feuille de route restent à réaliser ; aucun nouveau stockage externe, modèle distant ou envoi de données de santé n’est introduit.
