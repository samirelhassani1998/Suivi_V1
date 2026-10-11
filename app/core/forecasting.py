"""Prévisions multi-modèles avec intervalles de confiance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.core.features import LAGS, build_features
from app.core.models import get_quantile_models
from app.core.projection_constraints import constrain_interval_dataframe
from app.core.time_utils import normalize_calendar_dates

MIN_ML_TRAIN_ROWS = 10
MIN_ML_MEASUREMENTS = max(LAGS) + MIN_ML_TRAIN_ROWS


def future_daily_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Jours locaux suivants, y compris quand une journée dure 23 ou 25 h."""
    return pd.date_range(pd.Timestamp(last_date).normalize(), periods=horizon + 1, freq="D")[1:]


def require_converged_fit(fitted, model_name: str) -> None:
    """Une sortie numérique ne suffit pas si l'optimiseur signale un échec."""
    result = getattr(fitted, "mle_retvals", {})
    if result and result.get("converged") is not None and not result["converged"]:
        raise ValueError(f"{model_name} : ajustement non convergé ; prévision indisponible.")


def validate_prediction(central, low, high, size: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Vérifie les sorties brutes avant tout bornage ou calcul de métrique."""
    try:
        central = np.asarray(central, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("prédiction invalide") from exc
    if central.shape != (size,) or not np.isfinite(central).all():
        raise ValueError("prédiction incomplète ou non finie")
    if low is None and high is None:
        return central, None, None
    try:
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("intervalle incomplet") from exc
    if low.shape != (size,) or high.shape != (size,) or not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("intervalle incomplet")
    if np.any(low > high):
        raise ValueError("bornes d'intervalle inversées")
    return central, low, high


def _model_history(df: pd.DataFrame) -> pd.DataFrame:
    data = df[["Date", "Poids (Kgs)"]].copy()
    data["Date"] = normalize_calendar_dates(data["Date"])
    data["Poids (Kgs)"] = pd.to_numeric(data["Poids (Kgs)"], errors="coerce")
    require_daily_sampling(data)
    if not np.isfinite(data["Poids (Kgs)"]).all():
        raise ValueError("Ce modèle exige des poids numériques finis pour chaque jour.")
    return data.sort_values("Date", kind="mergesort").reset_index(drop=True)


def require_daily_sampling(df: pd.DataFrame) -> None:
    """Refuse de convertir implicitement un pas de mesure en jour calendaire.

    Les séries irrégulières restent prises en charge par la tendance datée.
    Les modèles récursifs à pas journalier demandent une grille observée,
    sans interpolation des poids manquants ni choix implicite parmi les doublons.
    """
    # Une journée locale peut durer 23 ou 25 heures au changement d'heure.
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).dt.normalize().sort_values()
    if dates.isna().any() or dates.duplicated().any() or not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "Ce modèle exige une seule pesée par jour, sans jour manquant. "
            "La projection selon vos mesures reste disponible pour cet historique irrégulier."
        )


def _training_matrix(df: pd.DataFrame, height_m: float) -> tuple[pd.DataFrame, pd.Series]:
    feat = build_features(df, height_m=height_m)
    feat = feat.dropna(subset=["Poids (Kgs)", *[f"lag_{lag}" for lag in LAGS]]).copy()
    X = feat.drop(columns=["Date", "Poids (Kgs)", "Notes", "Condition de mesure"], errors="ignore")
    X = X.select_dtypes(include=["number"]).fillna(0.0)
    y = feat["Poids (Kgs)"]
    return X, y


def forecast_with_ml(df: pd.DataFrame, horizon: int, height_m: float) -> pd.DataFrame:
    if len(df) < MIN_ML_MEASUREMENTS or horizon <= 0:
        return pd.DataFrame()
    df = _model_history(df)
    X, y = _training_matrix(df, height_m)
    if len(X) < MIN_ML_TRAIN_ROWS:
        return pd.DataFrame()

    models = get_quantile_models()
    for m in models.values():
        m.fit(X, y)

    future_dates = future_daily_dates(df["Date"].max(), horizon)
    base = df[["Date", "Poids (Kgs)"]].copy()
    rows: list[dict[str, object]] = []
    # Garde numérique, pas limite physiologique ni garantie de calibration.
    # Une récursion qui multiplie les variations observées doit être refusée
    # avant que le plafond d'affichage ne la transforme en objectif « atteint ».
    observed_steps = base["Poids (Kgs)"].diff().abs().dropna()
    max_step = max(1.0, 3.0 * float(observed_steps.quantile(0.95)))
    previous = float(base["Poids (Kgs)"].iloc[-1])

    for d in future_dates:
        history_aug = pd.concat([base, pd.DataFrame(rows)[["Date", "Poids (Kgs)"]]], ignore_index=True) if rows else base.copy()
        last_row = history_aug.iloc[-1:].copy()
        last_row.loc[last_row.index[0], "Date"] = d
        history_aug = pd.concat([history_aug, last_row], ignore_index=True)

        feat = build_features(history_aug, height_m=height_m).tail(1)
        Xf = feat.drop(columns=["Date", "Poids (Kgs)", "Notes", "Condition de mesure"], errors="ignore")
        Xf = Xf.select_dtypes(include=["number"]).fillna(0.0)
        Xf = Xf.reindex(columns=X.columns, fill_value=0.0)

        quantiles = np.array([float(models[key].predict(Xf)[0]) for key in ("q10", "q50", "q90")])
        if not np.isfinite(quantiles).all():
            raise ValueError("Prévision ML instable : valeurs non finies ; utilisez la projection selon vos mesures.")
        # Réarrangement monotone des quantiles estimés indépendamment.
        p10, p50, p90 = np.sort(quantiles)
        if abs(p50 - previous) > max_step or np.max(np.abs(quantiles - previous)) > max_step * (3.0 + np.sqrt(len(rows))):
            raise ValueError("Prévision ML instable : variation hors de l'échelle observée ; utilisez la projection selon vos mesures.")
        rows.append({"Date": d, "Poids (Kgs)": p50, "prevision": p50, "borne_basse": p10, "borne_haute": p90, "confiance": 0.8})
        previous = p50

    raw = pd.DataFrame(rows)[["Date", "prevision", "borne_basse", "borne_haute", "confiance"]]
    return constrain_interval_dataframe(raw)


def forecast_with_sarimax(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if len(df) < 14 or horizon <= 0:
        return pd.DataFrame()
    df = _model_history(df)
    model = SARIMAX(df["Poids (Kgs)"], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    require_converged_fit(fitted, "SARIMAX")
    pred = fitted.get_forecast(steps=horizon)
    ci = pred.conf_int(alpha=0.05)
    central, low, high = validate_prediction(pred.predicted_mean, ci.iloc[:, 0], ci.iloc[:, 1], horizon)
    dates = future_daily_dates(df["Date"].max(), horizon)
    raw = pd.DataFrame({
        "Date": dates,
        "prevision": central,
        "borne_basse": low,
        "borne_haute": high,
        "confiance": 0.95,
    })
    return constrain_interval_dataframe(raw)


def forecast_with_auto_arima(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if len(df) < 20 or horizon <= 0:
        return pd.DataFrame()
    from pmdarima import auto_arima

    data = _model_history(df)
    model = auto_arima(data["Poids (Kgs)"].to_numpy(dtype=float), seasonal=False, error_action="ignore", suppress_warnings=True)
    require_converged_fit(model.arima_res_, "Auto-ARIMA")
    forecast, interval = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
    central, low, high = validate_prediction(forecast, interval[:, 0], interval[:, 1], horizon)
    raw = pd.DataFrame({
        "Date": future_daily_dates(data["Date"].max(), horizon),
        "prevision": central, "borne_basse": low, "borne_haute": high, "confiance": 0.95,
    })
    out = constrain_interval_dataframe(raw)
    out.attrs["order"] = str(model.order)
    return out
