# 📊 Suivi V2 — Application de Suivi de Poids Intelligent

Application Streamlit de suivi de poids avec analyses comportementales avancées, garde-fous anti-extrapolation, détection de patterns, résumé actionnable et prévisions multi-modèles.

🔗 **[Application déployée](https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/)**

---

## ✨ Fonctionnalités clés

| Feature | Description |
|---|---|
| **Résumé actionnable** | Bloc Situation / Interprétation / Action adapté au contexte (phase de démarrage, effort établi) |
| **Garde-fous V3** | Score plafonné à 60 si < 7 mesures, ETA refusé si données fragiles, vitesse plafonnée si > 2 kg/sem |
| **Tendance EMA** | Poids lissé comme KPI principal (filtre les fluctuations quotidiennes) |
| **Détection yo-yo** | Analyse factuelle des rebonds après chaque pause de suivi |
| **Détection d'effort** | Identification automatique de la période d'effort actuelle (gap > 21j = nouveau départ) |
| **Trajectoire cible** | Corridor ±1 kg autour d'une pente saine de -0.5 kg/sem |
| **Phase de démarrage** | Les 7 premiers jours = "données fragiles", pas d'extrapolation |
| **Milestone intelligent** | Prochain palier avec ETA réaliste (gardes-fous physiologiques) |
| **Vue hebdomadaire** | Bar chart consolidé avec coloration baisse/hausse |
| **Toggle effort/historique** | Analyse ciblée sur l'effort actuel ou l'historique complet |
| **Prévisions multi-modèles** | SARIMAX, Auto-ARIMA, ML Quantile, scénarios prospectifs |
| **59 tests unitaires** | Couverture core, analytics, garde-fous V3, smoke pages |

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
│   │   ├── analytics.py           # 20+ fonctions pures (scores, phases, yo-yo, milestones...)
│   │   ├── data.py                # Nettoyage, validation, qualité des données
│   │   ├── evaluation.py          # Walk-forward backtest, métriques (MAE, RMSE, SMAPE, DA)
│   │   ├── features.py            # Feature engineering (lags, rolling stats, IMC)
│   │   ├── forecasting.py         # Prévisions SARIMAX + ML quantile (P10/P50/P90)
│   │   ├── insights.py            # Plateau, anomalies robustes, ETA avec garde-fous
│   │   ├── models.py              # Définition modèles sklearn (régression, quantile)
│   │   └── session_state.py       # Gestion cycle de vie données en session
│   │
│   ├── pages/                     # ── Pages Streamlit ──
│   │   ├── Dashboard.py           # Dashboard (KPIs, résumé actionnable, trajectoire, yo-yo)
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
├── tests/                         # ── Tests automatisés (59 tests) ──
│   ├── conftest.py                # Fixtures partagées (synthetic_df)
│   ├── test_analytics.py          # Tests analytics (24 tests)
│   ├── test_core_v2.py            # Tests core (data, forecasting, insights, evaluation)
│   ├── test_v3_guardrails.py      # Tests garde-fous V3 (score, ETA, milestone, résumé)
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
| **Tendance (EMA)** | `compute_trend_ema()` | Poids lissé — filtre les fluctuations quotidiennes |
| Moy. 30 derniers jours | Dates calendaires | Filtre `Date >= J-30`, puis moyenne |
| IMC | Calcul | `poids / taille_m²` |
| Qualité des données | `data_quality_report()` | Score 0-100 rate-based |
| Vitesse 7j | `weight_velocity()` | kg/semaine basé sur dates réelles (⚠️ si < 7j) |
| Discipline | `discipline_score()` | % jours mesurés sur 30j |
| Score global | `progression_score()` | Composite progression+vitesse+discipline+cohérence (⚠️ si fragile) |
| Série en cours | `streak_analysis()` | N mesures consécutives en baisse/hausse |

**Sections** :
- Bannière période d'effort actuelle (détection automatique)
- Barre de progression vers l'objectif (recalibrée sur l'effort)
- **Prochain palier** avec ETA intelligent (garde-fous : pas d'ETA si < 7 mesures)
- Signal de tendance + badge de confiance
- Alertes intelligentes (pas de mesure récente, série en cours, nouveau plus bas)
- **Pattern yo-yo** : alerte factuelle si rebond détecté + meilleur effort passé
- **🧭 Résumé actionnable** : Situation / Interprétation / Action (contextualisé)
- 💡 Insights détaillés (expander)
- **Graphique principal** : courbe poids + EMA + trajectoire cible (-0.5 kg/sem) + corridor ±1 kg
- Vue hebdomadaire consolidée (expander)
- Multi-MA chart (expander)
- Comparaison hebdo + volatilité
- Distribution + IMC (expander)
- Score détaillé (expander)

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
- **ETA objectif** : avec garde-fous V3 (refusé si < 7 mesures, plafonné si > 2 kg/sem)
- **Vitesses de variation** : kg/semaine sur 4 fenêtres calendaires (7/14/30/90 jours)

### 4. Insights (`app/pages/Insights.py`)

**9 sections analytiques** avec toggle effort actuel / historique complet :

| Section | Description |
|---------|-------------|
| Qualité & Plateau | Score qualité en cards visuelles + plateau 14/30 derniers jours calendaires |
| Scores & Discipline | Discipline (0-100), cohérence (0-100), volatilité (14 derniers jours) — avec progress bars |
| Phases du parcours | Segmentation gap-aware en phases (perte/plateau/reprise) + timeline visuelle |
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

20+ fonctions pures, sans dépendance Streamlit :

| Fonction | Description |
|----------|-------------|
| `weight_velocity()` | Vitesse kg/semaine sur plusieurs fenêtres calendaires |
| `multi_rolling_averages()` | MA glissantes sur N mesures consécutives |
| `weight_volatility()` | Stabilité (std, CV, amplitude) sur N jours calendaires |
| `discipline_score()` | Régularité de saisie 0-100 |
| `consistency_score()` | Cohérence intra-semaine |
| `detect_trend_breaks()` | Détection CUSUM des ruptures |
| `best_worst_weeks()` | Top N meilleures/pires semaines |
| `segment_phases()` | Segmentation gap-aware perte/plateau/reprise |
| `period_comparison()` | Comparaison semaine/mois vs précédent |
| `prospective_scenarios()` | Projections 30/60/90j (3 scénarios) |
| `streak_analysis()` | Séries consécutives de perte/gain |
| `day_of_week_analysis()` | Patterns par jour de la semaine |
| `progression_score()` | Score composite 0-100 avec confidence (fragile/solide) |
| `weight_acceleration()` | La perte accélère/ralentit |
| `detect_current_effort()` | Période d'effort actuelle (gap > 21j) |
| `compute_trend_ema()` | Tendance EMA robuste |
| `pace_comparison()` | Rythme actuel vs rythme nécessaire |
| `generate_insights_text()` | Insights textuels en français |
| `generate_action_summary()` | Résumé Situation/Interprétation/Action |
| `analyze_effort_history()` | Détection yo-yo (rebond moyen après pauses) |
| `next_milestone()` | Prochain palier avec ETA et garde-fous |

### Garde-fous V3 (honnêteté des sorties)

| Garde-fou | Règle | Pourquoi |
|---|---|---|
| Score plafonné | ≤ 60/100 si < 7 mesures | Empêche les faux 100/100 avec 3 jours de données |
| Champ `confidence` | `"fragile"` ou `"solide"` dans chaque score | L'utilisateur sait quand un chiffre est fiable |
| ETA refusé | Pas d'ETA si < 7 mesures dans l'effort | Empêche "objectif dans 5 jours" après 3 mesures |
| Vitesse plafonnée | ETA recalculé à 0.75 kg/sem si > 2 kg/sem | Empêche les projections basées sur la perte hydrique |
| Phase de démarrage | Pas d'extrapolation les 7 premiers jours | "Cette perte est probablement de l'eau" |

### Distinction mesures vs jours calendaires

> **Important** : l'application distingue rigoureusement deux types de fenêtres temporelles.
> Chaque métrique affiche le nombre de mesures réelles dans la fenêtre.

| Type de fenêtre | Logique | Fonctions concernées |
|---|---|---|
| **Jours calendaires** (`Date >= J-N`) | Filtre par vraies dates | `weight_velocity()`, `weight_volatility()`, `detect_plateau()`, `estimate_target_eta()`, `discipline_score()`, `period_comparison()`, KPIs Dashboard |
| **Mesures consécutives** (`.rolling(N)`) | Compte les N dernières entrées | `multi_rolling_averages()`, `streak_analysis()`, baselines backtest |
| **Dates réelles inter-mesures** | Vraie durée entre deux points | `weight_velocity()` (kg/semaine), `detect_current_effort()` (gap > 21j) |

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
# 59 tests couvrant : core, analytics, garde-fous V3, smoke pages, utilitaires
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
