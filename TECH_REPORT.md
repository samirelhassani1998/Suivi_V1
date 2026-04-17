# TECH_REPORT — Suivi V2

## Architecture
- `app/core/data.py`: nettoyage, validation, qualité des données, stratégie de doublons.
- `app/core/features.py`: lags, rolling stats, variation, IMC, bilan calorique.
- `app/core/evaluation.py`: métriques (MAE/RMSE/MAPE/sMAPE/biais/directional accuracy), walk-forward.
- `app/core/models.py`: modèles linéaires, arbres, boosting, quantiles.
- `app/core/forecasting.py`: prévisions SARIMAX + quantile regression (intervalles).
- `app/core/insights.py`: anomalies, plateau, ETA objectif.
- `app/ui/components.py`: KPI, badges confiance, alertes, aides.

## Fiabilité statistique
- Séparation explicite backtest vs in-sample.
- Baselines obligatoires pour éviter les métriques optimistes.
- Intervalles non-i.i.d. naïfs: SARIMAX CI et quantiles conditionnels.

## Résilience
- Fallback CSV local si Google Sheets indisponible.
- Gestion dataset vide/petit/mal formaté sans crash.
- Persistance session + export CSV.

## Compatibilité
- Python 3.10/3.11
- Streamlit 1.38+
