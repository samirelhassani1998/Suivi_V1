# Modèles et limites — Suivi V1

## Lecture de référence

- **Poids de tendance** : LOWESS robuste visant environ 14 jours, avec une fenêtre effective dépendant de la densité des pesées ; repli sur une moyenne mobile calendaire quand l'effectif est insuffisant.
- **Bruit quotidien** : MAD des résidus × 1,4826 ; bande indicative de largeur ±1,96 σ.
- **Rythme** : régression sur les jours réellement écoulés, fenêtres de 14 et 28 jours. L'intervalle de Student et la valeur p supposent des erreurs indépendantes ; la corrélation sérielle peut rendre ces résultats trop affirmatifs.
- **Projection conditionnelle** : prolongement de la tendance selon le rythme récent ; combinaison de l'incertitude sur la pente, le niveau et le bruit. Son niveau de 95 % est nominal et approximatif, pas une couverture garantie sur de nouvelles pesées.
- **Objectif final** : arrêt graphique à 80 kg selon la règle métier. Ce bornage d'affichage ne corrige ni une prévision instable ni un intervalle mal calibré.

## Cadence et variables

- La tendance et le drift utilisent les dates réelles. Les moyennes de référence portent sur les 7 ou 14 **mesures** précédentes.
- SARIMAX, Auto-ARIMA et la prévision quantile quotidienne exigent une seule pesée par jour, sans jour manquant. Les historiques irréguliers ou avec doublons rendent ces modèles indisponibles avec une explication ; ils restent exploitables par la tendance datée. Aucune interpolation de cible n'est effectuée pour rendre un modèle éligible.
- Les features ML sont limitées au calendrier et aux poids antérieurs : retards de 1, 3, 7, 14 et 30 mesures, moyennes/écarts-types glissants décalés, variation précédente et IMC précédent. Les colonnes importées supplémentaires, y compris IMC courant et calories, sont exclues pour empêcher une fuite de cible ou de disponibilité temporelle.
- Les notes vides et autres métadonnées facultatives ne suppriment pas les lignes d'apprentissage.

## Modèles expérimentaux

- **SARIMAX** : (1,1,1)(1,0,1)₇, au moins 14 mesures pour une projection. Son backtest se lance explicitement avec « Évaluer SARIMAX dans le classement », à partir de 30 mesures. La saison de 7 jours n'est interprétable que sur la grille quotidienne requise. Intervalle nominal à 95 % dans la projection comme dans le backtest.
- **Auto-ARIMA** : recherche non saisonnière pmdarima, intervalle nominal à 95 %. Calcul et ajout au classement à la demande ; désactivé en mode rapide.
- **Comparaison ML** : régression linéaire et Random Forest, séparation chronologique 80/20. Elle prédit la prochaine **mesure** à partir des mesures antérieures réellement disponibles, pas un chemin récursif à 30 jours. Le R² ne démontre pas un gain par rapport à la dernière pesée.
- **Quantiles P10/P50/P90** : standardisation ajustée sur l'apprentissage puis régression quantile avec régularisation L1 (`alpha=0.1`). Minimum de 40 mesures quotidiennes : 30 pour les retards et 10 lignes d'apprentissage effectives. Quantiles réordonnés pour éviter leur croisement ; une récursion non finie ou sortant fortement de l'échelle observée est refusée avant bornage graphique.
- La bande quantile a un niveau **nominal de 80 % à un pas**. La réinjection de la médiane ne propage pas toute l'incertitude future ; sa couverture à plusieurs jours n'est pas calibrée. La régularisation et les gardes numériques ne constituent pas une validation statistique.
- Le classement initial calcule uniquement les références et la tendance. Les modèles avancés et leurs backtests se calculent à la demande. STL regroupe explicitement les pesées par moyenne journalière et indique le nombre de jours interpolés ; cette décomposition est descriptive et demande au moins 20 jours réellement observés.

## Évaluation et portée du classement

- Walk-forward partagé entre références et modèles : jusqu'à 8 blocs de 7 mesures lorsque l'historique permet au moins 20 mesures d'apprentissage et deux blocs ; découpage chronologique proportionnel pour les historiques plus courts (minimum 8 mesures). Les modèles inéligibles restent signalés comme non évalués.
- MAE, RMSE, biais, gain de MAE contre la dernière valeur et couverture empirique de l'intervalle nominal à 95 %. La précision directionnelle est calculée à l'intérieur de chaque bloc, sans compter les sauts entre réajustements. Les intervalles non finis, incomplets ou inversés sont refusés.
- Les seuils ±5 % du verdict sont des repères de gain pratique, pas un test de supériorité statistique. « Plus faible erreur sur ces tests » ne garantit pas le meilleur modèle futur.
- Le classement repose sur les blocs de **mesures** du backtest, indépendamment du curseur d'horizon de 7 à 90 jours. Il n'évalue pas encore le modèle quantile récursif et ne doit pas être étendu à un horizon différent sans vérification.

## Travaux restant à valider

Évaluer tous les pipelines publiés aux horizons calendaires J+1/J+7/J+14/J+30, afficher l'effectif et la largeur des intervalles, puis calibrer leur couverture hors apprentissage avec une méthode adaptée à la dépendance temporelle. Étudier un modèle d'état acceptant les jours manquants et la sensibilité des intervalles de pente à l'autocorrélation avant d'ajouter des modèles plus complexes. Les prévisions restent indicatives ; elles ne fournissent ni date certaine d'objectif ni diagnostic.
