"""Évaluation robuste des prévisions (walk-forward).

Deux niveaux :

- ``evaluate_baselines`` compare les références naïves sur une simple série ;
- ``evaluate_forecasters`` met les modèles de l'application (tendance robuste,
  SARIMAX, Auto-ARIMA) face à ces références sur les mêmes découpages
  chronologiques, calcule un **gain par rapport à la dernière valeur** et la
  **couverture empirique** des intervalles annoncés. Un modèle qui ne bat pas
  « la dernière pesée répétée » n'apporte rien : le tableau le dit.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from app.core.trend import LONG_RATE_WINDOW_DAYS, trend_cone, trend_weight

NAIVE_MODEL = "Dernière valeur"
TREND_MODEL = "Tendance robuste + pente 28 j"
SARIMAX_MODEL = "SARIMAX (1,1,1)(1,0,1)₇"
AUTO_ARIMA_MODEL = "Auto-ARIMA"
# Au-delà de ce gain, un modèle est jugé réellement meilleur que la référence ;
# en deçà de son opposé, réellement moins bon. Entre les deux, indistinguable.
SKILL_MARGIN = 0.05
MIN_SERIES_LENGTH = 8
# Découpage walk-forward : des blocs de test d'une semaine de mesures, en
# gardant au moins vingt mesures d'apprentissage au premier bloc — l'effectif
# minimal des modèles ARIMA. Sous cet effectif, on retombe sur le découpage
# proportionnel de scikit-learn.
TEST_FOLD_SIZE = 7
MIN_TRAIN_SIZE = 20
MAX_SPLITS = 8


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(denom == 0, 0, 2 * np.abs(y_pred - y_true) / denom)) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    return float((np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).mean() * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "mape": mape,
        "smape": smape(y_true, y_pred),
        "biais": float(np.mean(y_pred - y_true)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def baseline_predictions(train: pd.Series, test_len: int, kind: str) -> np.ndarray:
    if kind == "last":
        return np.repeat(train.iloc[-1], test_len)
    if kind == "ma7":
        return np.repeat(train.tail(7).mean(), test_len)
    if kind == "ma14":
        return np.repeat(train.tail(14).mean(), test_len)
    slope = (train.iloc[-1] - train.iloc[0]) / max(len(train) - 1, 1)
    return np.array([train.iloc[-1] + slope * (i + 1) for i in range(test_len)])


def walk_forward_splits(n_rows: int, *, n_splits: int = MAX_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Découpages chronologiques partagés par tous les modèles d'un même tableau."""
    if n_rows < MIN_SERIES_LENGTH:
        return []
    available = (n_rows - MIN_TRAIN_SIZE) // TEST_FOLD_SIZE
    if available >= 2:
        splitter = TimeSeriesSplit(n_splits=int(min(n_splits, available)), test_size=TEST_FOLD_SIZE)
    else:
        splitter = TimeSeriesSplit(n_splits=min(n_splits, max(2, n_rows // 5)))
    return [(np.asarray(train), np.asarray(test)) for train, test in splitter.split(np.arange(n_rows))]


def walk_forward_backtest(series: pd.Series, model_fn, n_splits: int = 5) -> dict[str, float]:
    if len(series) < MIN_SERIES_LENGTH:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "smape": np.nan, "biais": np.nan, "directional_accuracy": np.nan}

    splitter = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(series) // 5)))
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in splitter.split(series):
        train = series.iloc[train_idx]
        test = series.iloc[test_idx]
        pred = model_fn(train, len(test))
        y_true_all.extend(test.values)
        y_pred_all.extend(pred)
    return compute_metrics(np.asarray(y_true_all), np.asarray(y_pred_all))


def evaluate_baselines(series: pd.Series) -> pd.DataFrame:
    rows = []
    mapping = {"last": "Dernière valeur", "ma7": "Moyenne mobile 7j", "ma14": "Moyenne mobile 14j", "drift": "Tendance linéaire"}
    for key, label in mapping.items():
        metrics = walk_forward_backtest(series, lambda tr, h, k=key: baseline_predictions(tr, h, k))
        rows.append({"modèle": label, **metrics})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Modèles datés : chaque prédicteur reçoit l'historique et les dates à prédire
# ──────────────────────────────────────────────────────────────────────────────

Prediction = tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
Forecaster = Callable[[pd.DataFrame, pd.DatetimeIndex], Prediction]


def trend_forecast_for_dates(train: pd.DataFrame, dates: pd.DatetimeIndex) -> Prediction:
    """Prolongement du poids de tendance au rythme des 28 derniers jours, avec son cône.

    Sans pente estimable (historique trop court), la tendance est simplement
    répétée : c'est le comportement le plus prudent et il reste évaluable.
    """
    frame = trend_weight(train)
    if frame.empty:
        last = float(train["Poids (Kgs)"].iloc[-1])
        return np.repeat(last, len(dates)), None, None
    last_date = frame["Date"].iloc[-1]
    days = np.array([(pd.Timestamp(d) - last_date).total_seconds() / 86400 for d in dates], dtype=float)
    cone = trend_cone(train, days, rate_window_days=LONG_RATE_WINDOW_DAYS)
    if not cone["ready"]:
        return np.repeat(float(frame["Tendance"].iloc[-1]), len(dates)), None, None
    if cone["has_bounds"]:
        return cone["central"], cone["low"], cone["high"]
    return cone["central"], None, None


def _sarimax_forecaster(train: pd.DataFrame, dates: pd.DatetimeIndex) -> Prediction:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if len(train) < 14:
        raise ValueError("historique trop court pour SARIMAX")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train["Poids (Kgs)"].to_numpy(dtype=float),
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=len(dates))
        interval = forecast.conf_int(alpha=0.05)
    interval = np.asarray(interval)
    return np.asarray(forecast.predicted_mean, dtype=float), interval[:, 0], interval[:, 1]


def _auto_arima_forecaster(train: pd.DataFrame, dates: pd.DatetimeIndex) -> Prediction:
    from pmdarima import auto_arima

    if len(train) < 20:
        raise ValueError("historique trop court pour Auto-ARIMA")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = auto_arima(train["Poids (Kgs)"].to_numpy(dtype=float), seasonal=False, error_action="ignore", suppress_warnings=True)
        forecast, interval = model.predict(n_periods=len(dates), return_conf_int=True, alpha=0.05)
    interval = np.asarray(interval)
    return np.asarray(forecast, dtype=float), interval[:, 0], interval[:, 1]


def _series_forecaster(kind: str) -> Forecaster:
    def _predict(train: pd.DataFrame, dates: pd.DatetimeIndex) -> Prediction:
        return baseline_predictions(train["Poids (Kgs)"].reset_index(drop=True), len(dates), kind), None, None

    return _predict


def default_forecasters(*, include_sarimax: bool = True, include_auto_arima: bool = False) -> dict[str, Forecaster]:
    models: dict[str, Forecaster] = {
        NAIVE_MODEL: _series_forecaster("last"),
        "Moyenne mobile 7 mesures": _series_forecaster("ma7"),
        "Moyenne mobile 14 mesures": _series_forecaster("ma14"),
        "Tendance linéaire (drift)": _series_forecaster("drift"),
        TREND_MODEL: trend_forecast_for_dates,
    }
    if include_sarimax:
        models[SARIMAX_MODEL] = _sarimax_forecaster
    if include_auto_arima:
        models[AUTO_ARIMA_MODEL] = _auto_arima_forecaster
    return models


def _verdict(skill: float, available: bool) -> str:
    if not available or not np.isfinite(skill):
        return "non évalué"
    if skill > SKILL_MARGIN:
        return "bat la dernière valeur"
    if skill < -SKILL_MARGIN:
        return "moins bon que la dernière valeur"
    return "équivalent à la dernière valeur"


def evaluate_forecasters(
    df: pd.DataFrame,
    *,
    include_sarimax: bool = True,
    include_auto_arima: bool = False,
    n_splits: int = MAX_SPLITS,
    forecasters: dict[str, Forecaster] | None = None,
) -> pd.DataFrame:
    """Backtest walk-forward de tous les modèles sur les mêmes découpages.

    Colonnes : ``Modèle``, ``MAE``, ``RMSE``, ``Biais``, ``Précision
    directionnelle (%)``, ``Gain vs dernière valeur (%)``, ``Couverture IC 95 %
    (%)`` (modèles à intervalle seulement), ``Verdict``, ``Disponible`` et
    ``Détail`` (raison d'une indisponibilité).
    """
    columns = [
        "Modèle",
        "MAE",
        "RMSE",
        "Biais",
        "Précision directionnelle (%)",
        "Gain vs dernière valeur (%)",
        "Couverture IC 95 % (%)",
        "Verdict",
        "Disponible",
        "Détail",
    ]
    if df is None or df.empty or "Date" not in df.columns or "Poids (Kgs)" not in df.columns:
        return pd.DataFrame(columns=columns)
    data = df[["Date", "Poids (Kgs)"]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Poids (Kgs)"] = pd.to_numeric(data["Poids (Kgs)"], errors="coerce")
    data = data.dropna().sort_values("Date", kind="mergesort").reset_index(drop=True)
    if len(data) < MIN_SERIES_LENGTH:
        return pd.DataFrame(columns=columns)

    models = forecasters or default_forecasters(include_sarimax=include_sarimax, include_auto_arima=include_auto_arima)
    splits = walk_forward_splits(len(data), n_splits=n_splits)

    collected: dict[str, dict[str, Any]] = {}
    for name, forecaster in models.items():
        y_true: list[float] = []
        y_pred: list[float] = []
        covered: list[bool] = []
        has_bounds = True
        failure: str | None = None
        for train_idx, test_idx in splits:
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            try:
                central, low, high = forecaster(train, pd.DatetimeIndex(test["Date"]))
            except Exception as exc:  # un modèle qui échoue reste dans le tableau, marqué indisponible
                failure = str(exc) or exc.__class__.__name__
                break
            central = np.asarray(central, dtype=float)
            if len(central) != len(test) or not np.all(np.isfinite(central)):
                failure = "prédiction incomplète"
                break
            truth = test["Poids (Kgs)"].to_numpy(dtype=float)
            y_true.extend(truth.tolist())
            y_pred.extend(central.tolist())
            if low is None or high is None:
                has_bounds = False
            else:
                low_arr = np.asarray(low, dtype=float)
                high_arr = np.asarray(high, dtype=float)
                covered.extend(((truth >= low_arr) & (truth <= high_arr)).tolist())
        collected[name] = {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "coverage": float(np.mean(covered) * 100) if has_bounds and covered else float("nan"),
            "failure": failure,
        }

    naive = collected.get(NAIVE_MODEL)
    naive_mae = float(mean_absolute_error(naive["y_true"], naive["y_pred"])) if naive and naive["failure"] is None and len(naive["y_true"]) else float("nan")

    rows = []
    for name, result in collected.items():
        available = result["failure"] is None and len(result["y_true"]) > 0
        if available:
            metrics = compute_metrics(result["y_true"], result["y_pred"])
            skill = (1.0 - metrics["mae"] / naive_mae) * 100 if np.isfinite(naive_mae) and naive_mae > 0 else float("nan")
        else:
            metrics = {"mae": np.nan, "rmse": np.nan, "biais": np.nan, "directional_accuracy": np.nan}
            skill = float("nan")
        rows.append(
            {
                "Modèle": name,
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "Biais": metrics["biais"],
                "Précision directionnelle (%)": metrics["directional_accuracy"],
                "Gain vs dernière valeur (%)": skill,
                "Couverture IC 95 % (%)": result["coverage"],
                "Verdict": _verdict(skill / 100 if np.isfinite(skill) else float("nan"), available),
                "Disponible": available,
                "Détail": result["failure"] or "",
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values(["Disponible", "MAE"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def best_model(table: pd.DataFrame) -> dict[str, Any] | None:
    """Ligne du meilleur modèle disponible, ou ``None`` si le tableau est vide."""
    if table is None or table.empty:
        return None
    usable = table[table["Disponible"].astype(bool)]
    if usable.empty:
        return None
    return usable.iloc[0].to_dict()
