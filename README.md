# 📊 Suivi V2 — Application de Suivi de Poids Intelligent

Application Streamlit premium de suivi de poids avec analyses avancées, prévisions multi-modèles, détection d'anomalies, scores comportementaux et insights automatiques.

🔗 **[Application déployée](https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/)**

---

## 🏗️ Architecture du projet

```
Suivi_V1/
├── Suivi_V1.py                    # Point d'entrée Streamlit (auth, data, navigation)
├── requirements.txt               # Dépendances Python
├── runtime.txt                    # Version Python pour Streamlit Cloud
│
├── .streamlit/
│   ├── config.toml                # Configuration serveur + thème visuel
│   ├── secrets.example.toml       # Template secrets (data_url, auth)
│   └── secrets.toml.example       # Template alternatif
│
├── app/
│   ├── __init__.py
│   ├── auth.py                    # Authentification par mot de passe (HMAC)
│   ├── config.py                  # Configuration centralisée (colonnes, defaults)
│   ├── deploy.py                  # Info déploiement (commit SHA)
│   ├── utils.py                   # Utilitaires legacy (chargement CSV, thèmes Plotly)
│   │
│   ├── core/                      # ── Moteur analytique ──
│   │   ├── analytics.py           # 17+ métriques avancées (vitesse, scores, phases, streaks...)
│   │   ├── data.py                # Nettoyage, validation, qualité des données
│   │   ├── evaluation.py          # Walk-forward backtest, métriques (MAE, RMSE, SMAPE, DA)
│   │   ├── features.py            # Feature engineering (lags, rolling stats, IMC)
│   │   ├── forecasting.py         # Prévisions SARIMAX + ML quantile (P10/P50/P90)
│   │   ├── insights.py            # Plateau, anomalies robustes, ETA multi-scénario
│   │   ├── models.py              # Définition modèles sklearn (régression, quantile)
│   │   └── session_state.py       # Gestion cycle de vie données en session
│   │
│   ├── pages/                     # ── Pages Streamlit ──
│   │   ├── Dashboard.py           # Dashboard principal (KPIs, charts, insights auto)
│   │   ├── Journal.py             # Édition des données + validation
│   │   ├── Predictions.py         # Prévisions multi-modèles + scénarios prospectifs
│   │   ├── Insights.py            # Analyses avancées (phases, scores, patterns)
│   │   ├── Settings.py            # Paramètres utilisateur (objectifs, MA, thème)
│   │   ├── Overview.py            # [Legacy] Page d'analyse originale
│   │   └── Modeles.py             # [Legacy] Page de comparaison modèles originale
│   │
│   └── ui/                        # ── Composants UI ──
│       ├── components.py          # KPI cards, badges, alertes, empty states
│       └── theme.py               # CSS global (métriques stylisées)
│
├── tests/                         # ── Tests automatisés ──
│   ├── conftest.py                # Fixtures partagées (synthetic_df)
│   ├── test_core_v2.py            # Tests core (data, forecasting, insights, evaluation)
│   ├── test_analytics.py          # Tests analytics (17 fonctions, 24 tests)
│   ├── test_streamlit_smoke.py    # Tests smoke des pages Streamlit
│   └── test_utils.py              # Tests utilitaires (MA, anomalies, filters)
│
└── docs/                          # Documentation additionnelle
```

---

## 🔄 Flux de données

```
Google Sheets CSV ──► load_remote_csv() ──► clean_weight_dataframe()
                          │                        │
                     cache 5min              Normalisation colonnes
                                             Parse dates (multi-format)
                                             Conversion poids (virgule → point)
                                                    │
                                                    ▼
                              ┌─────── source_data (immutable) ───────┐
                              │                                        │
                              ▼                                        ▼
                        working_data                            filtered_data
                     (éditable via Journal)                  (filtrage date actif)
                              │                                        │
                              └────────────────┬───────────────────────┘
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                         Dashboard        Predictions       Insights
                        (analytics)      (forecasting)     (analytics)
```

### Sources de données
- **Primaire** : Google Sheets exporté en CSV (URL configurable via `st.secrets["data_url"]`)
- **Alternative** : Import CSV local via la sidebar
- **Cache** : `st.cache_data(ttl=300)` — rechargement automatique toutes les 5 minutes

### Nettoyage (`app/core/data.py`)
1. **Normalisation colonnes** : variantes (`dates`, `poids`, `poids(kg)`) → `Date`, `Poids (Kgs)`
2. **Parse dates** : `dayfirst=True` en priorité, fallback `dayfirst=False`, fallback serial Excel
3. **Conversion poids** : `"80,2"` → `80.2` (virgule → point)
4. **Validation** : suppression lignes sans date/poids, détection outliers (z-robuste > 3.5)
5. **Colonnes additionnelles** : préservées (Calories, Notes, Condition de mesure, etc.)

---

## 📊 Pages de l'application

### 1. Dashboard (`app/pages/Dashboard.py`)

**KPIs principaux (9 métriques)** :
| KPI | Source | Logique |
|-----|--------|---------|
| Poids actuel | Dernière entrée | `df["Poids (Kgs)"].iloc[-1]` |
| Moy. 7 derniers jours | Dates calendaires | Filtre `Date >= J-7`, puis moyenne |
| Moy. 30 derniers jours | Dates calendaires | Filtre `Date >= J-30`, puis moyenne |
| IMC | Calcul | `poids / taille_m²` |
| Qualité des données | `data_quality_report()` | Score 0-100 rate-based |
| Vitesse 7j | `weight_velocity()` | kg/semaine basé sur dates réelles |
| Discipline | `discipline_score()` | % jours mesurés sur 30j |
| Score global | `progression_score()` | Composite progression+vitesse+discipline+cohérence |
| Série en cours | `streak_analysis()` | N mesures consécutives en baisse/hausse |

**Sections** :
- Barre de progression vers l'objectif final
- Détection de plateau (14 jours)
- Badge de confiance signal
- **Insights automatiques** : textes générés en français (vitesse, volatilité, discipline, objectif...)
- Indicateur d'accélération (perte accélère/ralentit)
- **Graphique principal** : courbe poids + moyenne mobile + 4 lignes d'objectifs + moyenne globale
- **Multi-MA chart** : MA 7/14/30 mesures superposées
- Comparaison hebdo (7 jours calendaires vs 7 jours précédents)
- Volatilité 14 mesures
- Distribution poids + évolution IMC
- Détail score de progression (4 composantes)

### 2. Journal (`app/pages/Journal.py`)

Édition interactive des données avec :
- **Data editor** Streamlit (`st.data_editor`, lignes ajoutables/supprimables)
- **Validation en temps réel** : erreurs bloquantes (colonnes manquantes, poids <= 0) + warnings (doublons, outliers)
- **Filtre date** en aperçu
- Export CSV
- Sauvegarde en session

### 3. Prévisions (`app/pages/Predictions.py`)

**5 onglets** :

| Onglet | Modèle | Détail |
|--------|--------|--------|
| Comparaison modèles | LinearRegression vs RandomForest | Split 80/20, métriques MSE/MAE/R² |
| SARIMA | SARIMAX(1,1,1)(1,0,1,7) | IC 90% avec fill area |
| Auto-ARIMA | pmdarima `auto_arima()` | Sélection automatique ordre, IC avec fill area |
| STL / ACF-PACF | Décomposition STL (période=7) | Tendance + saisonnalité + résidu + ACF/PACF |
| 🔮 Scénarios | 3 projections | Optimiste/réaliste/pessimiste basé sur vitesses 7/14/30j |

**Autres sections** :
- **Leaderboard baselines** : walk-forward backtest (dernière valeur, MA7, MA14, tendance linéaire)
- **ML Quantile** : Quantile Regression P10/P50/P90 avec features engineerées
- **ETA objectif multi-scénario** : date estimée par fenêtre (7j/30j/90j)
- **Vitesses de variation** : kg/semaine sur 4 fenêtres (7/14/30/90j)
- **Graphique de projection** : courbes à 90 jours pour 3 scénarios

### 4. Insights (`app/pages/Insights.py`)

**9 sections analytiques** :

| Section | Description |
|---------|-------------|
| Qualité & Plateau | Score qualité en cards visuelles + plateau 14j/30j |
| Scores & Discipline | Discipline (0-100), cohérence (0-100), volatilité — avec progress bars |
| Phases du parcours | Segmentation en phases (perte/plateau/reprise) + timeline visuelle |
| Ruptures de tendance | Détection CUSUM des changements structurels |
| Meilleures/Pires semaines | Top 5 meilleures et pires semaines par variation |
| Comparaison périodique | Semaine vs semaine précédente, mois vs mois précédent |
| Patterns jour de semaine | Poids moyen par jour + bar chart |
| Streaks | Séries consécutives de perte/gain + records |
| Anomalies & Clustering | Z-robuste + IsolationForest, KMeans clustering |

### 5. Paramètres (`app/pages/Settings.py`)

- Poids objectif final + 4 objectifs intermédiaires
- Taille (cm) pour calcul IMC
- Type de moyenne mobile (Simple / Exponentielle)
- Fenêtre moyenne mobile (3-60)
- Gestion doublons (garder dernière, moyenne, médiane)
- Modèle par défaut
- Thème Plotly
- Diagnostic système

---

## 🧠 Moteur analytique (`app/core/analytics.py`)

17 fonctions pures, sans dépendance Streamlit :

| Fonction | Input | Output | Description |
|----------|-------|--------|-------------|
| `weight_velocity()` | df, fenêtres | `{7: -0.5, 14: -0.3, ...}` | kg/semaine basé sur dates réelles |
| `multi_rolling_averages()` | df, fenêtres | DataFrame + colonnes MA | MA glissantes sur N mesures |
| `weight_volatility()` | df, window | `{std, cv, range, interpretation}` | Stabilité du poids |
| `discipline_score()` | df, window_days | `{score, rate, interpretation}` | Régularité de saisie 0-100 |
| `consistency_score()` | df, n_weeks | `{score, avg_weekly_std}` | Cohérence intra-semaine |
| `detect_trend_breaks()` | df, threshold | `[{date, type, description}]` | CUSUM simplifié |
| `best_worst_weeks()` | df, n | `{best: DataFrame, worst: DataFrame}` | Top N par variation |
| `segment_phases()` | df, min_days | `[Phase(start, end, type, slope)]` | Découpage perte/plateau/reprise |
| `period_comparison()` | df | `{week: {delta, ...}, month: {delta, ...}}` | Comparaison calendaire |
| `prospective_scenarios()` | df, target | `{optimiste, réaliste, pessimiste}` | Projections à 30/60/90j |
| `streak_analysis()` | df | `{current_streak, longest_loss/gain}` | Séries consécutives |
| `day_of_week_analysis()` | df | DataFrame par jour | Patterns hebdomadaires |
| `progression_score()` | df, target | `{score, grade, components}` | Score composite A+ à D |
| `weight_acceleration()` | df | `{acceleration, interpretation}` | Perte accélère/ralentit |
| `generate_insights_text()` | df, target | `[str, ...]` | Insights textuels FR |

### Distinction mesures vs jours calendaires

> **Important** : l'application distingue clairement :
> - **Jours calendaires** : utilisés pour les KPIs "Moy. 7 derniers jours" et "Moy. 30 derniers jours" (filtre par `Date >= J-N`)
> - **Mesures consécutives** : utilisées pour les rolling averages et les streaks (`.rolling(N)` et comptage d'entrées)
> - **Dates réelles** : utilisées pour `weight_velocity()` qui calcule la vraie durée entre deux mesures

---

## 🤖 Composants ML/Stats

### Feature engineering (`app/core/features.py`)
```
Lags         : lag_1, lag_3, lag_7, lag_14, lag_30
Rolling      : roll_mean_7/14/30, roll_std_7/14/30
Temporel     : jour_semaine, jours_depuis_derniere_mesure
Dérivé       : variation_journaliere, IMC (si taille fournie)
Optionnel    : bilan_calorique (si colonnes Calories présentes)
```

### Modèles disponibles (`app/core/models.py`)
- **Régression** : LinearRegression, Ridge, ElasticNet, RandomForest, GradientBoosting
- **Quantile** : QuantileRegressor P10, P50, P90 (intervalles de confiance)

### Prévisions (`app/core/forecasting.py`)
- **SARIMAX** : `(1,1,1)(1,0,1,7)` — modèle ARIMA saisonnier (période 7j)
- **Auto-ARIMA** : sélection automatique des paramètres via `pmdarima`
- **ML Quantile** : prévision jour par jour avec rétro-alimentation des features

### Évaluation (`app/core/evaluation.py`)
- **Walk-forward backtest** : TimeSeriesSplit avec N splits
- **Métriques** : MAE, RMSE, MAPE, SMAPE, biais, directional accuracy
- **Baselines** : dernière valeur, MA7, MA14, tendance linéaire

---

## 🔐 Authentification

Module `app/auth.py` — protection par mot de passe HMAC :
- Configurable via `st.secrets["auth"]["password"]`
- Mode démo : `st.secrets["auth"]["required"] = false`
- Fallback : `st.secrets["password"]` (compatibilité)

---

## ⚙️ Installation locale

```bash
git clone https://github.com/samirelhassani1998/Suivi_V1.git
cd Suivi_V1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configuration
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# Éditer secrets.toml avec votre mot de passe et data_url

# Lancer
streamlit run Suivi_V1.py
```

### Tests
```bash
python3 -m pytest tests/ -v
# 53 tests couvrant : core, analytics, smoke pages, utilitaires
```

---

## 📦 Dépendances

| Package | Version | Rôle |
|---------|---------|------|
| streamlit | 1.38.0 | Framework web (st.navigation, st.Page) |
| pandas | 2.1.4 | Manipulation données |
| numpy | 1.26.4 | Calculs numériques |
| plotly | 5.24.1 | Graphiques interactifs |
| scikit-learn | 1.4.2 | ML (RandomForest, IsolationForest, KMeans, Quantile) |
| statsmodels | 0.14.1 | SARIMAX, STL, ACF/PACF |
| pmdarima | 2.0.4 | Auto-ARIMA |
| scipy | 1.11.4 | Statistiques |
| matplotlib | 3.8.4 | ACF/PACF plots |
| pyarrow | 15.0.2 | Performance pandas |
| pytest | 8.3.4 | Tests |

---

## 🚀 Déploiement Streamlit Cloud

L'application est déployée automatiquement depuis `main` :
- **URL** : https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/
- **Secrets** : configurés dans Settings → Secrets sur Streamlit Cloud
- **Entrée** : `Suivi_V1.py`
- **Redéploiement** : automatique à chaque push sur `main`

---

## 📄 Licence

MIT License — voir [LICENSE](LICENSE)
