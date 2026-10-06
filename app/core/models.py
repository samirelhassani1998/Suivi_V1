"""Définition des modèles sklearn utilisés en prévision."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def get_regression_models() -> dict[str, object]:
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "boosting": GradientBoostingRegressor(random_state=42),
    }


def get_quantile_models() -> dict[str, Pipeline]:
    """Régressions régularisées, avec échelle apprise sur le passé uniquement.

    Sans pénalité, les nombreux retards colinéaires interpolent les petits
    historiques et leur réinjection récursive peut exploser en quelques jours.
    """
    return {
        f"q{int(quantile * 100)}": make_pipeline(
            StandardScaler(), QuantileRegressor(quantile=quantile, alpha=0.1)
        )
        for quantile in (0.1, 0.5, 0.9)
    }
