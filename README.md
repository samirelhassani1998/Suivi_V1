# Suivi_V1 🏋️

Application Streamlit interactive pour le **suivi, l'analyse et la prédiction de l'évolution du poids**, intégrant des modèles de machine learning et des techniques de séries temporelles.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/)

---

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Stack Technique](#-stack-technique)
- [Installation](#-installation)
- [Configuration](#%EF%B8%8F-configuration)
- [Utilisation](#-utilisation)
- [Déploiement](#-déploiement)
- [Troubleshooting](#-troubleshooting)
- [Ressources](#-ressources)

---

## ✨ Fonctionnalités

### 📊 Vue d'ensemble (Overview)
- **Métriques clés** : Poids actuel, poids moyen, IMC, variation 7j/30j avec calcul intelligent de la date la plus proche
- **Progression vers l'objectif** : Visualisation de l'avancement vers les objectifs personnalisés
- **Graphiques interactifs** : Évolution du poids avec moyennes mobiles (simple ou exponentielle)
- **Détection d'anomalies** : Identification des valeurs atypiques via Z-score ou IsolationForest
- **Export CSV** : Téléchargement des données filtrées
- **📊 Tendances** *(NEW)* : Indicateurs visuels de tendance (📉/📈/➡️) basés sur la moyenne mobile 7j/30j
- **📅 Comparaison Hebdomadaire** *(NEW)* : Moyenne cette semaine vs semaine précédente avec évolution en %

### 🤖 Modèles (Modeles)
- **Comparaison de modèles ML** : Régression Linéaire vs Random Forest avec validation croisée temporelle
- **Métriques de performance** : MSE, MAE, R² pour chaque modèle
- **Jauge de progression** : Indicateur visuel vers les objectifs de poids
- **Clustering K-Means** : Segmentation des données de poids en clusters
- **Détection d'anomalies ML** : Identification via IsolationForest

### 📈 Prédictions
- **Régression linéaire** : Prévision avec intervalle de confiance configurable
- **📊 Métriques de Confiance** *(NEW)* : R², MAE, RMSE avec indicateur de confiance (🟢 Haute / 🟡 Moyenne / 🔴 Faible)
- **Décomposition STL** : Séparation tendance, saisonnalité et résidus
- **SARIMA** : Modèle de prévision avec composantes saisonnières (avec spinner de chargement)
- **Auto-ARIMA** : Sélection automatique des meilleurs paramètres avec `pmdarima` (avec spinner de chargement)
- **Autocorrélation** : Visualisation ACF/PACF pour l'analyse des séries temporelles

---

## 🏗 Architecture

```
Suivi_V1/
├── Suivi_V1.py              # Point d'entrée principal
├── app/
│   ├── __init__.py
│   ├── auth.py              # Module d'authentification par mot de passe
│   ├── deploy.py            # Infos de déploiement
│   ├── utils.py             # Fonctions utilitaires partagées
│   └── pages/
│       ├── Overview.py      # Page Vue d'ensemble
│       ├── Modeles.py       # Page Comparaison des modèles
│       └── Predictions.py   # Page Prédictions
├── .streamlit/
│   ├── config.toml          # Configuration Streamlit (thème, etc.)
│   └── secrets.toml.example # Exemple de configuration des secrets
├── tests/                   # Tests unitaires
│   └── test_utils.py        # Tests de non-régression
├── docs/                    # Documentation
│   └── MODEL_CARD.md        # Carte de modèle ML (gouvernance)
├── requirements.txt         # Dépendances Python
├── runtime.txt              # Version Python pour le déploiement
├── TECH_REPORT.md           # Rapport technique détaillé
├── AUDIT.md                 # Notes d'audit
└── CHANGELOG.md             # Historique des modifications
```

### Flux de données

1. **Chargement** : Les données sont récupérées depuis Google Sheets via export CSV
2. **Cache** : Mise en cache intelligente avec `st.cache_data` (TTL 5 min)
3. **Filtrage** : Filtrage par plage de dates dans la sidebar
4. **Traitement** : Calcul des moyennes mobiles, détection d'anomalies
5. **Visualisation** : Graphiques Plotly avec thèmes personnalisables

---

## 🔧 Stack Technique

| Catégorie | Technologies |
|-----------|-------------|
| **Framework UI** | Streamlit 1.38+ |
| **Visualisation** | Plotly Express, Plotly Graph Objects |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (LinearRegression, RandomForest, KMeans, IsolationForest) |
| **Time Series** | Statsmodels (STL, ARIMA, ACF/PACF), pmdarima (Auto-ARIMA) |
| **Statistiques** | SciPy (Z-score) |
| **Stockage** | Google Sheets (source), PyArrow (traitement) |

---

## 💻 Installation

### Prérequis

- **Python** : 3.10 ou 3.11
- **Accès réseau** : Connectivité vers Google Sheets

### Installation locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/samirelhassani1998/Suivi_V1.git
cd Suivi_V1

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
.\venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Fichier de secrets

Créez le fichier `.streamlit/secrets.toml` basé sur l'exemple fourni :

```toml
[auth]
required = true
password = "votre_mot_de_passe"

# Optionnel : URL personnalisée pour les données
# data_url = "https://..."

# Optionnel : Mode debug (affiche les chemins de page)
# debug_mode = false
```

### Modes d'accès

| Mode | Configuration | Description |
|------|--------------|-------------|
| **Protégé** | `required = true` + `password = "..."` | Mot de passe requis |
| **Démo** | `required = false` | Accès libre sans authentification |

---

## 🚀 Utilisation

### Lancement local

```bash
streamlit run Suivi_V1.py
```

L'application sera accessible sur [http://localhost:8501](http://localhost:8501)

### Paramètres de la sidebar

- **Données** : Rechargement, filtrage par dates
- **Affichage** : Thème Plotly (Dark, Light, Solar, Seaborn)
- **Moyennes mobiles** : Type (Simple/Exponentielle) et fenêtre (1-30 jours)
- **Objectifs** : 4 niveaux d'objectifs personnalisables
- **Anomalies** : Méthode de détection et seuil Z-score
- **Activité** : Suivi calorique (calories consommées/brûlées)

---

## ☁️ Déploiement

### Streamlit Community Cloud

1. Poussez votre code sur GitHub
2. Connectez-vous à [share.streamlit.io](https://share.streamlit.io)
3. Déployez en pointant sur `Suivi_V1.py`
4. Configurez les secrets dans **Settings → Secrets** :
   ```toml
   [auth]
   required = true
   password = "votre_mot_de_passe_secret"
   ```

### URL de l'application

🔗 https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/

---

## 🔍 Troubleshooting

| Problème | Cause possible | Solution |
|----------|---------------|----------|
| **Erreur réseau** | Connectivité Google Sheets | Vérifier firewall, configurer `data_url` dans secrets |
| **`st.Page` manquant** | Streamlit < 1.31 | Mettre à jour : `pip install streamlit>=1.38` |
| **Dépendances manquantes** | Environnement incomplet | Recréer venv et réinstaller requirements |
| **Quota API dépassé** | Trop de requêtes Google | Utiliser source de données alternative |
| **Calculs lents** | Gros dataset + Auto-ARIMA | Réduire la plage de dates, activer le cache |
| **Erreur d'authentification** | Secrets non configurés | Configurer `.streamlit/secrets.toml` |

---

## 📚 Ressources

### Documentation officielle
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Statsmodels](https://www.statsmodels.org/)
- [pmdarima (Auto-ARIMA)](https://alkaline-ml.com/pmdarima/)

### Fichiers du projet
- [TECH_REPORT.md](./TECH_REPORT.md) - Rapport technique détaillé
- [CHANGELOG.md](./CHANGELOG.md) - Historique des modifications
- [AUDIT.md](./AUDIT.md) - Notes d'audit du code

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](./LICENSE) pour plus de détails.
