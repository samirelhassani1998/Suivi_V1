# Suivi_V1

Application Streamlit pour analyser l'évolution du poids, détecter les anomalies et générer des prévisions à l'aide de modèles statistiques et de machine learning.

- **Application déployée :** https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/
- **Entrée principale :** `Suivi_V1.py`
- **Pages** : `pages/1_Analyse.py`, `pages/2_Modeles.py`, `pages/3_Predictions.py`

## Accès Sécurisé

L'application est protégée par un mot de passe pour restreindre l'accès aux données.
Veuillez contacter l'administrateur pour obtenir les identifiants.

## Prérequis

- Python 3.10 ou 3.11
- Accès réseau sortant vers Google Sheets (pour le jeu de données distant)
- `git` pour récupérer le dépôt

## Installation locale

```bash
git clone https://github.com/samirelhassani1998/Suivi_V1.git
cd Suivi_V1
python -m venv .venv
# Activer l'environnement virtuel :
# Windows :
.venv\Scripts\activate
# Mac/Linux :
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Secrets (optionnel)

L'application peut lire une URL de données alternative ou des clés API via `.streamlit/secrets.toml`. Inspirez-vous de [`.streamlit/secrets.example.toml`](.streamlit/secrets.example.toml).

## Lancement de l'application

```bash
streamlit run Suivi_V1.py
```

La commande démarre un serveur local sur http://localhost:8501.

## Déploiement sur Streamlit Cloud

1. Pousser la branche sur GitHub/GitLab.
2. Créer une application Streamlit Cloud en pointant vers `Suivi_V1.py`.
3. Définir les secrets éventuels directement dans l'interface Streamlit Cloud (onglet **Secrets**).
4. Déployer. Les dépendances sont contrôlées par `requirements.txt` (versions épinglées pour la stabilité).

## Tests et qualité

- **Authentification** : Un module `app/auth.py` gère la sécurité.
- **Données** : Le chargement est mis en cache (`st.cache_data`) et gère les erreurs réseau.
- **Dépendances** : `requirements.txt` contient des versions strictes pour éviter les conflits.

## Troubleshooting

| Problème | Solution |
| --- | --- |
| **Erreur réseau** | Vérifier la connectivité sortante. Configurer `data_url` dans `st.secrets` si besoin. |
| **`st.Page` manquant** | Mettre à jour Streamlit (>=1.38) ou utiliser le menu multipage natif. |
| **Dépendances manquantes** | Recréer l'environnement virtuel et réinstaller les paquets depuis `requirements.txt`. |
| **Quota API** | Ajouter un mécanisme de repli et afficher un message utilisateur. |

## Ressources complémentaires

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Statsmodels](https://www.statsmodels.org/)
