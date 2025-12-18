# Model Card – Suivi_V1 Predictions

## Modèles Utilisés

### 1. Régression Linéaire
- **Type** : Régression simple sur données temporelles
- **Features** : Date numérique (jours depuis début)
- **Cible** : Poids (Kgs)
- **Usage** : Estimation de la tendance générale et date d'atteinte objectif

### 2. Random Forest
- **Type** : Ensemble de 100 arbres de décision
- **Features** : Date numérique
- **Cible** : Poids (Kgs)
- **Usage** : Comparaison de modèle, robustesse aux outliers

### 3. SARIMA / Auto-ARIMA
- **Type** : Modèle de séries temporelles avec saisonnalité
- **Paramètres SARIMA** : order=(1,1,1), seasonal_order=(1,1,1,7)
- **Auto-ARIMA** : Sélection automatique via pmdarima, m=7 (saisonnalité hebdomadaire)
- **Usage** : Prévisions à court/moyen terme

---

## Métriques de Performance

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| **R²** | Coefficient de détermination | 0.8+ = Haute confiance, 0.5-0.8 = Moyenne, <0.5 = Faible |
| **MAE** | Erreur absolue moyenne | Erreur moyenne en kg |
| **RMSE** | Racine erreur quadratique moyenne | Pénalise les grandes erreurs |
| **MSE** | Erreur quadratique moyenne | Utilisé pour validation croisée |

### Validation
- **TimeSeriesSplit** : 5-fold temporel (respecte l'ordre chronologique)
- **Bootstrap** : 500 itérations pour intervalles de confiance (95%)

---

## Hypothèses et Limitations

### Hypothèses
1. **Linéarité** : La tendance du poids est approximativement linéaire
2. **Saisonnalité** : Cycles hebdomadaires (7 jours)
3. **Stationnarité** : Après différenciation, la série est stationnaire

### Limitations
| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Peu de données (<30 points) | Prédictions peu fiables | Indicateur confiance visuel |
| Outliers non traités | Biais des modèles | Détection anomalies séparée |
| Pas de features externes | Manque contexte (régime, sport) | Documentation |
| Drift non détecté automatiquement | Modèle devient obsolète | Ré-entraînement manuel |

### Biais Potentiels
- **Biais de sélection** : Données saisies manuellement (peut omettre mauvais jours)
- **Biais temporel** : Plus de données récentes → modèle biaisé vers période récente

---

## Guide de Ré-entraînement

### Quand ré-entraîner ?
1. Après ajout de >20% nouvelles données
2. Si R² chute sous 0.5
3. Après changement important de comportement (nouveau régime, etc.)

### Comment ré-entraîner ?
Les modèles sont entraînés **automatiquement** à chaque chargement de page.
Aucune action manuelle requise.

Pour forcer un ré-entraînement :
1. Cliquer sur "Recharger les données" dans la sidebar
2. La page recalculera tous les modèles

### Monitoring Minimal
```python
# Vérifier les résidus
residuals = y_true - y_pred
if np.abs(residuals).mean() > seuil_acceptable:
    print("⚠️ Modèle potentiellement obsolète")
```

---

## Confidentialité et Éthique

- **Données personnelles** : Poids uniquement, pas d'identifiants
- **Stockage** : Google Sheets (responsabilité utilisateur)
- **Pas de partage** : Modèles locaux, pas d'envoi de données

---

## Changelog
| Version | Date | Modifications |
|---------|------|---------------|
| 1.0 | 2024-12 | Création initiale |
| 1.1 | 2024-12 | Ajout métriques confiance (R², MAE, RMSE) |
