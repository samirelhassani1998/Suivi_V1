# Changelog

## 2026-04-17
- Refonte V2: nouvelle navigation 5 pages (Dashboard/Journal/Prévisions/Insights/Paramètres).
- Refactor modulaire `app/core/*` et `app/ui/*`.
- Ajout fallback upload CSV local si Google Sheets indisponible.
- Ajout couche qualité des données + score + diagnostics.
- Ajout backtesting walk-forward et leaderboard des baselines.
- Ajout intervalles de prédiction (SARIMAX + quantile regression), ETA prudente, plateau 14j/30j.
- Ajout tests unitaires core + smoke tests Streamlit AppTest.
