# MODEL_CARD — Suivi V1

## Lectures de référence (sans apprentissage)
- **Poids de tendance** : LOWESS robuste (Cleveland 1979), fenêtre de 14 jours calendaires, deux itérations de repondération ; repli sur une moyenne mobile calendaire sous 5 mesures.
- **Bruit quotidien** : MAD × 1,4826 des écarts à la tendance ; bande à 95 % = 1,96 σ.
- **Rythme** : moindres carrés sur les jours calendaires (fenêtres 14 et 28 jours), intervalle de confiance de Student et valeur p ; direction annoncée seulement si l'intervalle exclut zéro.
- **Cône de projection** : incertitude sur la pente (croissante avec l'horizon), sur le niveau (σ/√n local) et bruit d'une pesée, combinés en quadrature ; borné à l'objectif final.

## Modèles évalués
- Références : dernière valeur, moyennes mobiles 7 et 14 mesures, tendance linéaire (drift).
- Tendance robuste + pente 28 jours (cône ci-dessus).
- SARIMAX (1,1,1)(1,0,1)₇ ; Auto-ARIMA (pmdarima, non saisonnier).
- Expérimental : régression linéaire et Random Forest sur variables dérivées décalées, régression quantile récursive (P10/P50/P90).

## Évaluation
- Walk-forward : blocs de test de 7 mesures, au moins 20 mesures d'apprentissage au premier bloc, jusqu'à 8 blocs, mêmes découpages pour tous les modèles.
- Métriques : MAE, RMSE, biais, précision directionnelle, **gain de MAE face à la dernière valeur**, **couverture empirique de l'intervalle à 95 %**.
- Verdict : « bat la dernière valeur » au-delà de 5 % de gain, « moins bon » en deçà de −5 %, « équivalent » entre les deux, « non évalué » avec la raison sinon.

## Fuites de cible évitées
- Toutes les variables dérivées du poids sont décalées d'au moins une mesure (lags, moyennes et écarts-types glissants sur les pesées précédentes, IMC de la pesée précédente). L'IMC du jour et la variation du jour, présents auparavant, donnaient un R² de 1,000 et une prévision plate.

## Fiabilité / limites
- Aucune conclusion sous l'effectif minimal : chaque résultat porte son effectif et la raison d'une indisponibilité.
- Les prévisions sont des extrapolations (« et si cela continuait ») et ne sont ni des diagnostics ni des promesses ; le leaderboard dit quand aucun modèle ne fait mieux que la dernière pesée.
- Une couverture empirique nettement sous 95 % signale un intervalle trop étroit ; elle est affichée telle quelle.

## Hypothèses
- Série quotidienne ou quasi quotidienne ; les fenêtres sont calendaires, pas en nombre de lignes.
- Les colonnes facultatives (calories) ne sont utilisées que si elles sont renseignées.
