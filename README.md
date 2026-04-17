# Suivi_V1 → V2 (Streamlit)

Application Streamlit de suivi du poids (français), orientée analyse statistique robuste et prévisions avec incertitude.

## Nouveautés V2
- Navigation pro en 5 espaces: **Dashboard, Journal, Prévisions, Insights, Paramètres**.
- Journal éditable avec `st.data_editor`, validation (date/poids/doublons/aberrants) et export CSV.
- Source par défaut Google Sheets (`st.secrets.data_url`) + fallback upload CSV local.
- Backtesting walk-forward + leaderboard des baselines (last, MA7, MA14, drift).
- Prévisions avec intervalles de prédiction (SARIMAX + régression quantile).
- Détection de plateau (14j/30j), anomalies robustes (z-score robuste, option IsolationForest), ETA objectif prudente.
- Architecture modulaire (`app/core`, `app/ui`) et tests unitaires + smoke tests Streamlit.

## Exécution
```bash
pip install -r requirements.txt
streamlit run Suivi_V1.py
pytest -q
```

## Secrets (exemple)
```toml
[auth]
required = true
password = "mot_de_passe"

data_url = "https://.../export?format=csv"
```

## Limites importantes
- Les prévisions sont **informatives et non médicales**.
- Si peu de données (< 14/20 points selon modèle), l'app affiche des avertissements et limite les sorties.
