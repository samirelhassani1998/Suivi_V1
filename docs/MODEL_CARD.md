# MODEL_CARD — Suivi V2

## Modèles utilisés
- Baselines: dernière valeur, MA7, MA14, drift.
- Régression: LinearRegression, Ridge, ElasticNet.
- Non-linéaires: RandomForestRegressor, GradientBoostingRegressor.
- Série temporelle: SARIMAX.
- Intervalles: QuantileRegressor (q10/q50/q90) et IC SARIMAX.

## Évaluation
- Validation: walk-forward / expanding window (`TimeSeriesSplit`).
- Métriques: MAE, RMSE, MAPE, sMAPE, biais moyen, directional accuracy.
- Affichage distinct backtest vs métriques d'ajustement local.

## Fiabilité / limites
- Les prévisions ne sont pas fiables avec peu de données; l'application l'indique.
- Les sorties sont analytiques, non diagnostiques et non médicales.
- En présence d'anomalies ou forte irrégularité, le score de confiance est abaissé.

## Hypothèses
- Série temporelle quotidienne avec bruit modéré.
- Les patterns comportementaux ne sont disponibles que si colonnes optionnelles renseignées.
