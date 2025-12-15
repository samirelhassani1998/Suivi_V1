# Audit applicatif : Suivi_V1 (Streamlit)

## 1) Cartographie du dépôt
- **Entrée Streamlit** : `Suivi_V1.py` – configure la page, charge les données, construit la barre latérale et enregistre manuellement trois pages via `st.navigation` (Analyses, Modèles, Prédictions). Il ne contient aucun graphique principal.
- **Pages multipages** :
  - `pages/1_Analyse.py` – page d'analyse du poids (résumé, graphes, anomalies, corrélation, export).
  - `pages/2_Modeles.py` – comparaisons de modèles et indicateurs d'objectif.
  - `pages/3_Predictions.py` – prévisions (régression linéaire, STL/SARIMA, Auto-ARIMA).
- **Modules** :
  - `app/utils.py` – chargement/nettoyage des données Google Sheets, outils graphiques, calculs (moyennes mobiles, anomalies, régression linéaire, prédiction future).
  - `app/auth.py` – protection par mot de passe (mot de passe codé en dur « 1234567890 »).
- **Configuration** : `.streamlit/config.toml` (thème/serveur), `.streamlit/secrets.example.toml` (gabarit des secrets, dont `data_url`).
- **Dépendances** : `requirements.txt` (Streamlit 1.38.0, Plotly, scikit-learn, pmdarima...).
- **Documentation** : README/CHANGELOG/TECH_REPORT (instructions d'installation générale).

### Page censée afficher les analyses de poids
`pages/1_Analyse.py` contient l'intégralité des graphiques et statistiques de suivi (résumé, courbes, histogrammes, anomalies, corrélations). Elle est atteinte via la navigation `st.navigation` (icône 📊) ou via le menu multipage natif si `st.navigation` n'est pas disponible.

## 2) Reproduction et diagnostic
### Exécution locale
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Suivi_V1.py
```
Mot de passe requis : `1234567890` (non documenté dans l'UI).

### Symptômes observables
- Page « Analyses » non évidente : la page d'accueil n'affiche aucun graphique et les intitulés de navigation ne mentionnent pas « Suivi/Poids ».
- Navigation potentiellement absente si `st.navigation` n'est pas supporté (versions < 1.38) : seul un message d'info apparaît dans la sidebar.
- Données absentes ou erreurs silencieuses : `load_data` lève une `RuntimeError` non gérée dans `pages/1_Analyse.py`, donc la page peut crasher sans message utilisateur si le Google Sheet n'est pas accessible.
- Données vides : si le Google Sheet est vide/inaccessible, les graphes ne s'affichent pas (DataFrame vide) et aucun état vide guidé n'est proposé.
- Authentification bloquante : mot de passe nécessaire mais non annoncé sur l'écran d'accueil ni dans le README de l'app déployée.

### Points de contrôle Streamlit Cloud
- **Logs** : vérifier les traces d'échec de `load_data` (erreurs réseau Google Sheets) et d'import pmdarima/statsmodels.
- **requirements** : Streamlit 1.38.0 requis pour `st.navigation`; si la version cloud est plus ancienne, la navigation custom ne fonctionne pas.
- **secrets** : `data_url` optionnel pour pointer vers un CSV accessible publiquement; absence/faille réseau → échec de chargement.
- **config** : vérifier que l'entrée principale est bien `Suivi_V1.py`.

## 3) Audit technique (anomalies priorisées)
1) **Navigation / routing**
   - Utilisation de `st.navigation`/`st.Page` uniquement disponible à partir de Streamlit 1.38. Si le runtime cloud exécute une version antérieure, la navigation custom disparaît et seule une info dans la sidebar reste, sans liens directs vers « Analyses » → fonctionnalité principale introuvable.
   - Les titres de pages sont « Analyse/Modèles/Prédictions » sans mention « Suivi du poids »/« Analyses du poids », ce qui rend la fonctionnalité principale peu détectable.
   - La page d'accueil ne contient aucun bouton CTA ni lien explicite vers la page d'analyses.

2) **Couche données**
   - Source unique Google Sheets (`DATA_URL`) sans fallback local, ni validation de disponibilité. En cas d'échec réseau/permissions, la page d'analyse n'affiche rien.
   - `pages/1_Analyse.py` charge les données via `load_data()` sans gestion des `RuntimeError` encapsulant les problèmes réseau/parsing → plantage ou écran vide sans explication.
   - Aucune validation de schéma : si les colonnes « Date » ou « Poids (Kgs) » sont absentes/renommées, les pages échouent silencieusement ou via exceptions non interceptées.
   - Les calories saisies dans `render_correlation` ne sont pas persistées (en mémoire uniquement, aucune sauvegarde ni fusion avec la source).

3) **Transformations / calculs**
   - Les moyennes mobiles utilisent `rolling`/`ewm` sans garde sur la longueur du jeu de données; avec < window_size, les graphes peuvent afficher des NaN partiels sans message.
   - Progression vers l'objectif peut diviser par zéro lorsque `initial_weight <= target_weights[-1]` (code force 1.0 mais rend le pourcentage incohérent pour un utilisateur ayant déjà atteint l'objectif à la première mesure).

4) **Visualisations**
   - Aucun état vide guidé : les sections affichent seulement des warnings génériques, sans CTA (« Importer des données », « Rafraîchir »). Graphes Plotly restent vides lorsque `filtered_data` est vide.
   - Les annotations d'objectifs utilisent une palette partielle (`objective_colors` taille 4 pour 4 objectifs mais premier `None`), ce qui peut masquer des lignes sans couleur explicite.

5) **State & UX**
   - La barre latérale regroupe de nombreux contrôles sans regroupement clair; aucune section dédiée « Suivi/Analyses du poids » pour guider l'utilisateur.
   - Le mot de passe n'est pas communiqué; un échec affiche seulement « Mot de passe incorrect » sans contexte.
   - `st.session_state` dépend d'un passage sur la page d'accueil pour initialiser `raw_data`/`filtered_data`/filtres; accès direct à `pages/1_Analyse.py` (via URL directe) contourne `_configure_sidebar` et laisse les filtres non initialisés.

6) **Performance / cache**
   - `st.cache_data` utilisé mais `ttl=300` uniquement dans `load_data` depuis l'entrée principale. Les pages appellent `load_data` sans `ttl` ni spinner, pouvant recharger inutilement le Google Sheet.

7) **Packaging / deps**
   - `pmdarima` + `statsmodels` sont coûteux et peuvent rallonger les temps de build sur Streamlit Cloud; aucune mention de timeout ou fallback en cas d'échec d'installation.

8) **Déploiement**
   - L'application repose sur `st.navigation`; si l'instance Streamlit Cloud ne met pas à jour vers 1.38, les pages restent accessibles uniquement via le système natif (numérotation `1_`, `2_`, `3_`) mais la page d'accueil ne fournit pas de lien explicite.
   - Aucun `secrets.toml` n'est fourni côté cloud; si le Google Sheet est privé, la récupération échouera.

## 4) Audit UX
- **Découvrabilité** : aucun bouton « Voir le suivi du poids » sur l'accueil, ni libellé « Suivi/Analyses du poids » dans la navigation. Un utilisateur peut croire que la fonctionnalité n'existe pas.
- **Navigation** : la sidebar est chargée de paramètres avant d'avoir vu un graphique; l'ordre des pages (Analyse en premier) n'est pas mis en avant visuellement.
- **Empty states** : absence de messages guidés (« Aucune donnée — vérifiez la connexion ou configurez `data_url` dans les Secrets »).
- **Guidage data** : unités et prérequis de données (format date, séparateur décimal) ne sont pas rappelés; pas d'indication sur la période filtrée actuelle.
- **Feedback d’erreur** : les exceptions de chargement dans `pages/1_Analyse.py` ne sont pas affichées à l’utilisateur.
- **Cohérence visuelle** : mix de headers/markdown sans sections claires; manque d’un tableau de bord synthétique (KPIs + graphes clés) sur l’accueil.

## 5) Plan de correction
### Quick wins (0–2h)
- Documenter et afficher le mot de passe/CTA d’accès sur la page d’accueil.
- Ajouter un bouton « Accéder aux analyses du poids » qui utilise `st.page_link("pages/1_Analyse.py", label="Suivi / Analyses du poids")`.
- Renommer les titres de pages pour inclure « Suivi du poids » (ex. « Suivi & Analyses du poids », « Modèles de prévision », « Prédictions avancées »).
- Gérer explicitement les erreurs de `load_data` dans `pages/1_Analyse.py` avec un état vide guidé (message + bouton « Réessayer » / lien vers configuration `data_url`).
- Ajouter un dataset de secours (CSV local) si le Google Sheet est inaccessible.
- Regrouper la sidebar en sections (Chargement/Filtres/Anomalies/Objectifs) et afficher la période filtrée active.

### Refactor robuste (1–2 jours)
- Extraire un module `app/data.py` centralisant chargement, validation de schéma, fallback local, et gestion des messages d’erreur utilisateur.
- Remplacer `st.navigation` par le système multipage natif + `st.page_link` dans l’accueil pour compatibilité rétro-versions, ou fixer explicitement la version Streamlit >=1.38 dans Cloud.
- Créer un composant de dashboard d’accueil (KPI + mini-graphes) pour que la fonctionnalité clé soit visible dès l’ouverture.
- Implémenter des tests unitaires pour `load_data`, `filter_by_dates`, détection d’anomalies, et validation de schéma.
- Documenter clairement les prérequis de données (colonnes attendues, format de date/decimal) dans README et dans l’UI (popover d’aide).

## 6) Corrections code proposées (extraits)
- **CTA et navigation explicite** (`Suivi_V1.py`)
```python
st.title("Suivi de l'Évolution du Poids")
st.link_button("Accéder aux analyses du poids", "pages/1_Analyse.py")
```
Remplace `link_button` par `st.page_link` si disponible (>=1.30) pour rester dans la même appli.

- **Gestion d'erreur de données dans la page d'analyse** (`pages/1_Analyse.py`)
```python
def _get_data():
    df = st.session_state.get("filtered_data")
    if df is None:
        try:
            df = load_data()
        except RuntimeError as error:
            st.error("Impossible de charger les données. Vérifiez la connexion ou configurez `data_url` dans les Secrets.")
            st.caption(str(error))
            return pd.DataFrame(columns=["Date", "Poids (Kgs)"])
        st.session_state["filtered_data"] = df
        st.session_state["raw_data"] = df
    return df.copy()
```

- **Fallback local** (`app/utils.py`)
```python
def load_data(url: str = DATA_URL) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, decimal=",")
    except Exception:
        local_path = Path(__file__).resolve().parent.parent / "data" / "poids.csv"
        if local_path.exists():
            df = pd.read_csv(local_path)
        else:
            raise RuntimeError("Impossible de télécharger les données distantes et aucun fichier local n'est disponible.")
```

- **Renommage des pages** (`Suivi_V1.py`)
```python
pages = [
    st.Page("pages/1_Analyse.py", title="Suivi & Analyses du poids", icon="📊"),
    st.Page("pages/2_Modeles.py", title="Modèles de prévision", icon="🤖"),
    st.Page("pages/3_Predictions.py", title="Prédictions avancées", icon="📈"),
]
```

- **Empty state guidé** (`pages/1_Analyse.py`)
```python
if df.empty:
    st.warning("Aucune donnée disponible.")
    st.info("Configurez une URL publique dans st.secrets['data_url'] ou réessayez plus tard.")
    if st.button("Réessayer"):
        st.session_state.pop("filtered_data", None)
        st.rerun()
    return
```

## 7) Critères d'acceptation (DoD)
- La page « Suivi & Analyses du poids » est visible dans la navigation (icône 📊) et un CTA est présent sur l’accueil.
- Après connexion, les graphiques s’affichent avec la plage de dates par défaut et les objectifs visibles.
- En cas d’absence de données ou d’erreur réseau, un message explicite + bouton de réessai apparaît (pas d’exception non gérée).
- Possibilité de configurer une `data_url` alternative via les secrets Streamlit Cloud; fallback local opérationnel.
- Temps de chargement acceptable (<5 s sur dataset courant) grâce au cache et à la réduction des appels répétés.
- Aucun crash lors des prévisions/analyses avec <10 lignes : messages d’avertissement clairs.
