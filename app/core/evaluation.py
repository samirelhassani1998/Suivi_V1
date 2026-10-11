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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from app.core.trend import LONG_RATE_WINDOW_DAYS, trend_cone, trend_weight
from app.core.features import LAGS, build_features
from app.core.forecasting import require_converged_fit, require_daily_sampling, validate_prediction
from app.core.models import get_regression_models
from app.core.time_utils import normalize_calendar_dates

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
    ratio = np.divide(2 * np.abs(y_pred - y_true), denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
    return float(np.mean(ratio) * 100)


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


def walk_forward_splits(
    n_rows: int, *, n_splits: int = MAX_SPLITS, dates=None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Découpages communs, en gardant toutes les pesées d'un jour du même côté."""
    if dates is not None:
        days = pd.DatetimeIndex(dates).tz_localize(None).normalize()
        if len(days) != n_rows or days.hasnans or not days.is_monotonic_increasing:
            raise ValueError("Les dates du backtest doivent être valides et triées.")
        groups, unique_days = pd.factorize(days, sort=False)
        day_splits = walk_forward_splits(len(unique_days), n_splits=n_splits)
        return [(np.flatnonzero(np.isin(groups, train)), np.flatnonzero(np.isin(groups, test))) for train, test in day_splits]
    if n_rows < MIN_SERIES_LENGTH:
        return []
    n_splits = max(2, n_splits)
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
    direction_correct, direction_total = 0, 0
    for train_idx, test_idx in splitter.split(series):
        train = series.iloc[train_idx]
        test = series.iloc[test_idx]
        pred = model_fn(train, len(test))
        y_true_all.extend(test.values)
        y_pred_all.extend(pred)
        direction_correct += int(np.sum(np.sign(np.diff(test.values)) == np.sign(np.diff(pred))))
        direction_total += max(0, len(test) - 1)
    metrics = compute_metrics(np.asarray(y_true_all), np.asarray(y_pred_all))
    metrics["directional_accuracy"] = 100.0 * direction_correct / direction_total if direction_total else float("nan")
    return metrics


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
    last_date = pd.Timestamp(frame["Date"].iloc[-1]).tz_localize(None).normalize()
    days = (pd.DatetimeIndex(dates).tz_localize(None).normalize() - last_date).days.to_numpy(dtype=float)
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
    require_daily_sampling(pd.DataFrame({"Date": list(train["Date"]) + list(dates)}))
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
        require_converged_fit(fitted, "SARIMAX")
        forecast = fitted.get_forecast(steps=len(dates))
        interval = forecast.conf_int(alpha=0.05)
    interval = np.asarray(interval)
    return np.asarray(forecast.predicted_mean, dtype=float), interval[:, 0], interval[:, 1]


def _auto_arima_forecaster(train: pd.DataFrame, dates: pd.DatetimeIndex) -> Prediction:
    if len(train) < 20:
        raise ValueError("historique trop court pour Auto-ARIMA")
    require_daily_sampling(pd.DataFrame({"Date": list(train["Date"]) + list(dates)}))
    from pmdarima import auto_arima

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = auto_arima(train["Poids (Kgs)"].to_numpy(dtype=float), seasonal=False, error_action="ignore", suppress_warnings=True)
        require_converged_fit(model.arima_res_, "Auto-ARIMA")
        forecast, interval = model.predict(n_periods=len(dates), return_conf_int=True, alpha=0.05)
    interval = np.asarray(interval)
    return np.asarray(forecast, dtype=float), interval[:, 0], interval[:, 1]


def _series_forecaster(kind: str) -> Forecaster:
    def _predict(train: pd.DataFrame, dates: pd.DatetimeIndex) -> Prediction:
        if kind == "drift":
            train_days = pd.DatetimeIndex(train["Date"]).tz_localize(None).normalize()
            elapsed = (train_days[-1] - train_days[0]).days
            last = float(train["Poids (Kgs)"].iloc[-1])
            slope = (last - float(train["Poids (Kgs)"].iloc[0])) / elapsed if elapsed > 0 else 0.0
            offsets = (dates.tz_localize(None).normalize() - train_days[-1]).days.to_numpy()
            return last + slope * offsets, None, None
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


def _verdict(skill: float, available: bool, *, mae: float = np.nan, reference_mae: float = np.nan) -> str:
    if available and reference_mae == 0:
        return "équivalent à la dernière valeur" if mae == 0 else "moins bon que la dernière valeur"
    if not available or not np.isfinite(skill):
        return "non évalué"
    if skill > SKILL_MARGIN:
        return "bat la dernière valeur"
    if skill < -SKILL_MARGIN:
        return "moins bon que la dernière valeur"
    return "équivalent à la dernière valeur"


def _relative_gain(mae: float, reference_mae: float) -> float:
    if reference_mae == 0 and mae == 0:
        return 0.0
    return (1.0 - mae / reference_mae) * 100 if np.isfinite(reference_mae) and reference_mae > 0 else float("nan")


def _clean_evaluation_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df[["Date", "Poids (Kgs)"]].copy()
    data["Date"] = normalize_calendar_dates(data["Date"])
    data["Poids (Kgs)"] = pd.to_numeric(data["Poids (Kgs)"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data.dropna().sort_values("Date", kind="mergesort").reset_index(drop=True)


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
        "Largeur moyenne IC (kg)",
        "Pesées testées",
        "Blocs évalués",
        "Horizon min (j)",
        "Horizon max (j)",
        "Verdict",
        "Disponible",
        "Détail",
    ]
    if df is None or df.empty or "Date" not in df.columns or "Poids (Kgs)" not in df.columns:
        return pd.DataFrame(columns=columns)
    data = _clean_evaluation_data(df)
    if len(data) < MIN_SERIES_LENGTH:
        return pd.DataFrame(columns=columns)

    models = forecasters or default_forecasters(include_sarimax=include_sarimax, include_auto_arima=include_auto_arima)
    splits = walk_forward_splits(len(data), n_splits=n_splits, dates=data["Date"])
    if not splits:
        return pd.DataFrame(columns=columns)
    calendar_days = pd.DatetimeIndex(data["Date"]).tz_localize(None).normalize()
    offsets = np.concatenate([(calendar_days[test] - calendar_days[train[-1]]).days for train, test in splits])

    collected: dict[str, dict[str, Any]] = {}
    for name, forecaster in models.items():
        y_true: list[float] = []
        y_pred: list[float] = []
        covered: list[bool] = []
        interval_widths: list[float] = []
        direction_correct, direction_total = 0, 0
        has_bounds = True
        failure: str | None = None
        for train_idx, test_idx in splits:
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            try:
                central, low, high = forecaster(train, pd.DatetimeIndex(test["Date"]))
                central, low, high = validate_prediction(central, low, high, len(test))
            except Exception as exc:  # un modèle qui échoue reste dans le tableau, marqué indisponible
                failure = str(exc) or exc.__class__.__name__
                break
            truth = test["Poids (Kgs)"].to_numpy(dtype=float)
            y_true.extend(truth.tolist())
            y_pred.extend(central.tolist())
            direction_correct += int(np.sum(np.sign(np.diff(truth)) == np.sign(np.diff(central))))
            direction_total += max(0, len(truth) - 1)
            if low is None or high is None:
                has_bounds = False
            else:
                covered.extend(((truth >= low) & (truth <= high)).tolist())
                interval_widths.extend((high - low).tolist())
        collected[name] = {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "coverage": float(np.mean(covered) * 100) if failure is None and has_bounds and covered else float("nan"),
            "interval_width": float(np.mean(interval_widths)) if failure is None and has_bounds and interval_widths else float("nan"),
            "failure": failure,
            "directional_accuracy": 100.0 * direction_correct / direction_total if direction_total else float("nan"),
        }

    naive = collected.get(NAIVE_MODEL)
    naive_mae = float(mean_absolute_error(naive["y_true"], naive["y_pred"])) if naive and naive["failure"] is None and len(naive["y_true"]) else float("nan")

    rows = []
    for name, result in collected.items():
        available = result["failure"] is None and len(result["y_true"]) > 0
        if available:
            metrics = compute_metrics(result["y_true"], result["y_pred"])
            metrics["directional_accuracy"] = result["directional_accuracy"]
            skill = _relative_gain(metrics["mae"], naive_mae)
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
                "Largeur moyenne IC (kg)": result["interval_width"],
                "Pesées testées": len(result["y_true"]) if available else 0,
                "Blocs évalués": len(splits) if available else 0,
                "Horizon min (j)": int(offsets.min()) if available else np.nan,
                "Horizon max (j)": int(offsets.max()) if available else np.nan,
                "Verdict": _verdict(skill / 100, available, mae=metrics["mae"], reference_mae=naive_mae),
                "Disponible": available,
                "Détail": result["failure"] or ("Gain relatif indéfini : la référence a une erreur nulle." if available and naive_mae == 0 and metrics["mae"] > 0 else ""),
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


def evaluate_regressors(df: pd.DataFrame, height_m: float = 1.82, *, models: dict | None = None) -> pd.DataFrame:
    """Test final 20 % : prochaine mesure, avec poids précédents déjà observés.

    La frontière respecte les journées. Au moins 20 journées d'apprentissage et
    5 de test sont requises après les 30 mesures de chauffe des retards.
    Ce test n'est pas une simulation récursive de plusieurs jours.
    """
    data = _clean_evaluation_data(df)
    feat = build_features(data, height_m).dropna(subset=["Date", "Poids (Kgs)", *[f"lag_{lag}" for lag in LAGS]])
    days = feat["Date"].dt.tz_localize(None).dt.normalize()
    unique_days = days.drop_duplicates()
    split = int(len(unique_days) * 0.8)
    table = pd.DataFrame()
    if split < MIN_TRAIN_SIZE or len(unique_days) - split < 5:
        table.attrs["reason"] = (
            "Comparaison indisponible : au moins 20 jours d'apprentissage et 5 jours de test "
            "après les 30 mesures nécessaires aux retards (55 mesures avec une pesée par jour)."
        )
        return table
    test_start = unique_days.iloc[split]
    is_train = days < test_start
    X = feat.drop(columns=["Date", "Poids (Kgs)"]).select_dtypes(include=["number"]).fillna(0.0)
    y = feat["Poids (Kgs)"]
    X_train, X_test = X.loc[is_train], X.loc[~is_train]
    y_train, y_test = y.loc[is_train], y.loc[~is_train]
    naive = X_test["lag_1"].to_numpy(dtype=float)
    naive_mae = float(mean_absolute_error(y_test, naive))
    if models is None:
        candidates = get_regression_models()
        models = {"Régression linéaire": candidates["linear"], "Random Forest": candidates["random_forest"]}
    rows = []
    for name, model in {NAIVE_MODEL: None, **models}.items():
        try:
            if model is None:
                pred = naive
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            pred, _, _ = validate_prediction(pred, None, None, len(y_test))
            metrics = compute_metrics(y_test.to_numpy(dtype=float), pred)
            gain = _relative_gain(metrics["mae"], naive_mae)
            rows.append({
                "Modèle": name, "MAE": metrics["mae"], "RMSE": metrics["rmse"],
                "R²": float(r2_score(y_test, pred)) if y_test.nunique() > 1 else np.nan,
                "Gain vs dernière valeur (%)": gain,
                "Verdict": _verdict(gain / 100, True, mae=metrics["mae"], reference_mae=naive_mae),
                "Disponible": True, "Détail": "",
            })
        except Exception as exc:
            rows.append({"Modèle": name, "MAE": np.nan, "RMSE": np.nan, "R²": np.nan,
                         "Gain vs dernière valeur (%)": np.nan, "Verdict": "non évalué", "Disponible": False, "Détail": str(exc)})
    table = pd.DataFrame(rows).sort_values(["Disponible", "MAE"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    table.attrs.update({"train_rows": len(y_train), "test_rows": len(y_test),
                        "train_days": split, "test_days": len(unique_days) - split,
                        "train_end": feat.loc[is_train, "Date"].max(),
                        "test_start": feat.loc[~is_train, "Date"].min(),
                        "test_end": feat.loc[~is_train, "Date"].max()})
    return table
