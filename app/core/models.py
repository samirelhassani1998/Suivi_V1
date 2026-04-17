"""Définition des modèles sklearn utilisés en prévision."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, QuantileRegressor, Ridge


def get_regression_models() -> dict[str, object]:
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "boosting": GradientBoostingRegressor(random_state=42),
    }


def get_quantile_models() -> dict[str, QuantileRegressor]:
    return {
        "q10": QuantileRegressor(quantile=0.1, alpha=0),
        "q50": QuantileRegressor(quantile=0.5, alpha=0),
        "q90": QuantileRegressor(quantile=0.9, alpha=0),
    }
