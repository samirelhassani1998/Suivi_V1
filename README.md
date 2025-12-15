# Suivi_V1

Application Streamlit pour analyser l'évolution du poids, détecter les anomalies et générer des prévisions à l'aide de modèles statistiques et de machine learning.

- **Application déployée :** https://samirelhassani1998-suivi-v1-suivi-v1-knzeqy.streamlit.app/
- **Entrée principale :** `Suivi_V1.py`
- **Architecture** : `app/pages/` contient les modules de page (`Analyse`, `Modeles`, `Predictions`).
- **Navigation** : Gérée via `st.navigation` dans `Suivi_V1.py`.

## Accès Sécurisé

L'application est protégée par un mot de passe (défini dans `secrets.toml` ou par défaut).

## Prérequis

- Python 3.10 ou 3.11
- Accès réseau sortant vers Google Sheets.

## Installation locale

1.  **Cloner le dépôt** :
    ```bash
    git clone https://github.com/samirelhassani1998/Suivi_V1.git
    cd Suivi_V1
    ```

2.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurer les secrets** :
    - Copiez `.streamlit/secrets.toml.example` vers `.streamlit/secrets.toml`.
    - Modifiez-le avec votre URL et mot de passe.

## Lancement

```bash
streamlit run Suivi_V1.py
```

## Déploiement

1. Poussez sur GitHub.
2. Déployez sur Streamlit Cloud en pointant sur `Suivi_V1.py`.
3. Ajoutez vos secrets dans la configuration de l'application Cloud.

## Qualité & Robustesse

- **Navigation** : Centralisée et robuste (pas de conflits `pages/`).
- **Données** : Gestion des erreurs et chargement résilient.
- **Sécurité** : Pas de secrets en dur (utilisation de `st.secrets`).

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
