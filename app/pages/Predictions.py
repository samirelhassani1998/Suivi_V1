"""Page Prévisions : où mène le rythme actuel, et quels modèles méritent confiance."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from app.core.analytics import detect_current_effort, prospective_scenarios, weight_velocity
from app.core.evaluation import NAIVE_MODEL, TREND_MODEL, best_model, evaluate_forecasters, walk_forward_splits
from app.core.features import build_features
from app.core.forecasting import forecast_with_ml, forecast_with_sarimax
from app.core.formatting import format_fr_date, format_fr_kg, format_fr_kg_per_week, format_fr_number
from app.core.insights import estimate_target_eta
from app.core.projection_constraints import constrain_interval_dataframe
from app.core.session_state import get_filtered_or_working_data
from app.core.target_trajectory import build_target_trajectory
from app.core.targets import get_target_weights
from app.core.trend import LONG_RATE_WINDOW_DAYS, TREND_WINDOW_DAYS, eta_to_target, noise_level, project_trend, rate_of_change, trend_weight
from app.ui.charts import (
    ACCENT_COLOR,
    BAND_FILL,
    MEASURE_COLOR,
    NEUTRAL_LINE,
    STATUS_CRITICAL,
    STATUS_GOOD,
    TARGET_COLOR,
    TRAJECTORY_COLOR,
    TREND_BAND_FILL,
    TREND_COLOR,
    add_band,
    add_horizontal_reference,
    add_line_trace,
    add_measure_trace,
    apply_layout,
    bar_figure,
)
from app.ui.components import alert_banner, empty_state, insight_card, kpi_card, page_hero, section_header

warnings.filterwarnings("ignore")

# Historique affiché à côté d'une prévision : au-delà, l'horizon devient un
# trait illisible au bord droit du graphique.
HISTORY_DAYS_SHOWN = 120
MIN_ROWS_SARIMAX = 30
MIN_ROWS_AUTO_ARIMA = 40


def _df() -> pd.DataFrame:
    return get_filtered_or_working_data()


def _fr(value, decimals: int = 2, *, sign: bool = False) -> str:
    return format_fr_number(value, decimals=decimals, sign=sign)


# ──────────────────────────────────────────────────────────────────────────────
# Calculs mis en cache : un modèle ne se réajuste pas à chaque clic
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_leaderboard(df: pd.DataFrame, include_sarimax: bool, include_auto_arima: bool) -> pd.DataFrame:
    return evaluate_forecasters(df, include_sarimax=include_sarimax, include_auto_arima=include_auto_arima)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_sarimax(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return forecast_with_sarimax(df, horizon=horizon)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_auto_arima(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    from pmdarima import auto_arima

    model = auto_arima(df["Poids (Kgs)"].to_numpy(dtype=float), seasonal=False, error_action="ignore", suppress_warnings=True)
    forecast, interval = model.predict(n_periods=horizon, return_conf_int=True)
    dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    raw = pd.DataFrame({"Date": dates, "prevision": np.asarray(forecast), "borne_basse": interval[:, 0], "borne_haute": interval[:, 1]})
    out = constrain_interval_dataframe(raw)
    out.attrs["order"] = str(model.order)
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ml(df: pd.DataFrame, horizon: int, height_m: float) -> pd.DataFrame:
    return forecast_with_ml(df, horizon=horizon, height_m=height_m)


# ──────────────────────────────────────────────────────────────────────────────
# Graphiques
# ──────────────────────────────────────────────────────────────────────────────


def _recent_history(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = df["Date"].max() - pd.Timedelta(days=HISTORY_DAYS_SHOWN)
    return df[df["Date"] >= cutoff]


def _forecast_figure(
    df: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    name: str,
    color: str,
    fill: str,
    title: str,
    target_weight: float,
    band_label: str = "Intervalle à 95 %",
) -> go.Figure:
    history = _recent_history(df)
    fig = go.Figure()
    if {"borne_basse", "borne_haute"}.issubset(pred.columns) and not pred.empty:
        add_band(fig, pred["Date"], pred["borne_basse"], pred["borne_haute"], name=band_label, fill=fill)
    add_measure_trace(fig, history["Date"], history["Poids (Kgs)"], name="Historique", marker_size=5, width=1.6)
    if not pred.empty:
        add_line_trace(fig, pred["Date"], pred["prevision"], name=name, color=color, dash="dash", width=2.4)
    dates = list(history["Date"]) + (list(pred["Date"]) if not pred.empty else [])
    add_horizontal_reference(fig, dates, target_weight, name=f"Objectif : {format_fr_kg(target_weight, decimals=1)}", color=TARGET_COLOR)
    return apply_layout(fig, title, y_title="Poids (kg)", height=440)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Projection principale : le rythme mesuré, prolongé avec son incertitude
# ──────────────────────────────────────────────────────────────────────────────


def _primary_projection(df: pd.DataFrame, horizon: int, target_weight: float) -> None:
    section_header(
        "Projection selon vos mesures",
        f"Le poids de tendance prolongé au rythme des {LONG_RATE_WINDOW_DAYS} derniers jours, avec un cône d'incertitude à 95 %.",
        "🧭",
    )
    frame = trend_weight(df)
    rate = rate_of_change(df, LONG_RATE_WINDOW_DAYS)
    noise = noise_level(df)
    eta = eta_to_target(df, target_weight)
    projection = project_trend(df, horizon)
    if not projection.empty:
        projection = constrain_interval_dataframe(projection)

    trend_now = float(frame["Tendance"].iloc[-1]) if not frame.empty else float("nan")
    cols = st.columns(4)
    with cols[0]:
        kpi_card(
            f"Rythme {LONG_RATE_WINDOW_DAYS} jours",
            f"{_fr(rate['slope_kg_week'], sign=True)} kg/sem" if rate["ready"] else "—",
            help_text=(
                f"IC 95 % : {_fr(rate['ci_low'], sign=True)} à {_fr(rate['ci_high'], sign=True)} kg/sem, "
                f"sur {rate['n']} pesées et {rate['span_days']} jours (p = {_fr(rate['p_value'], 3)})."
                if rate["ready"]
                else f"Indisponible : {rate['reason']}."
            ),
        )
    with cols[1]:
        kpi_card(
            "Poids de tendance",
            format_fr_kg(trend_now, decimals=1) if np.isfinite(trend_now) else "—",
            help_text=f"Régression locale robuste sur {TREND_WINDOW_DAYS} jours : point de départ de la projection.",
        )
    with cols[2]:
        if not projection.empty:
            last = projection.iloc[-1]
            kpi_card(
                f"Projeté à J+{int((last['Date'] - df['Date'].max()).days)}",
                format_fr_kg(last["prevision"], decimals=1),
                help_text=f"Entre {format_fr_kg(last['borne_basse'], decimals=1)} et {format_fr_kg(last['borne_haute'], decimals=1)} à 95 %, si le rythme se maintenait.",
            )
        else:
            kpi_card(f"Projeté à J+{horizon}", "—", help_text="Projection indisponible tant que la pente n'est pas estimable.")
    with cols[3]:
        if eta["ready"] and eta["reached"]:
            kpi_card("Arrivée à l'objectif", "atteint", help_text="Le poids de tendance est déjà sous l'objectif.")
        elif eta["ready"]:
            plage = ""
            if eta["eta_early"] is not None:
                plage = f"Plage plausible : {format_fr_date(eta['eta_early'])} → " + (format_fr_date(eta["eta_late"]) if eta["eta_late"] is not None else "au-delà de trois ans")
            kpi_card("Arrivée à l'objectif", format_fr_date(eta["eta"]), help_text=f"Dans {eta['days']} jours au rythme central. {plage}.")
        else:
            kpi_card("Arrivée à l'objectif", "—", help_text=f"Non projetée : {eta['reason']}.")

    fig = go.Figure()
    history = _recent_history(df)
    if not projection.empty:
        add_band(fig, projection["Date"], projection["borne_basse"], projection["borne_haute"], name="Cône d'incertitude à 95 %", fill=TREND_BAND_FILL)
    add_measure_trace(fig, history["Date"], history["Poids (Kgs)"], name="Pesées", marker_size=5, width=1.4, opacity=0.85)
    trend_recent = frame[frame["Date"] >= history["Date"].min()] if not frame.empty else frame
    if not trend_recent.empty:
        add_line_trace(fig, trend_recent["Date"], trend_recent["Tendance"], name="Poids de tendance", color=TREND_COLOR, width=2.8)
    if not projection.empty:
        add_line_trace(fig, projection["Date"], projection["prevision"], name="Prolongement au rythme actuel", color=TREND_COLOR, dash="dash", width=2.2)
    dates = list(history["Date"]) + (list(projection["Date"]) if not projection.empty else [])
    add_horizontal_reference(fig, dates, target_weight, name=f"Objectif : {format_fr_kg(target_weight, decimals=1)}", color=TARGET_COLOR)
    trajectory = build_target_trajectory(df)
    if trajectory.get("available") and dates:
        traj = trajectory["trajectory"]
        traj = traj[(traj["Date"] >= min(dates)) & (traj["Date"] <= max(dates))]
        if not traj.empty:
            add_line_trace(fig, traj["Date"], traj["Poids cible (kg)"], name="Trajectoire cible", color=TRAJECTORY_COLOR, dash="dashdot", width=1.8, decimals=1)
    apply_layout(fig, f"Rythme actuel prolongé sur {horizon} jours", y_title="Poids (kg)", height=460)
    st.plotly_chart(fig, use_container_width=True, key="predictions-primary")

    if eta["ready"] and not eta["reached"]:
        plage = ""
        if eta["eta_early"] is not None:
            plage = f" Plage plausible : du {format_fr_date(eta['eta_early'])} " + (
                f"au {format_fr_date(eta['eta_late'])}." if eta["eta_late"] is not None else "à au-delà de trois ans."
            )
        insight_card(
            f"Objectif atteint vers le {format_fr_date(eta['eta'])} si le rythme se maintient",
            f"Rythme {_fr(rate['slope_kg_week'], sign=True)} kg/semaine (IC 95 % {_fr(rate['ci_low'], sign=True)} à {_fr(rate['ci_high'], sign=True)}).{plage} "
            "Extrapolation linéaire : elle répond à « et si cela continuait ainsi », pas à « que va-t-il se passer ».",
            tone="success",
            icon="🧭",
        )
    elif eta["ready"] and eta["reached"]:
        insight_card("Objectif atteint", "Le poids de tendance est sous l'objectif principal : l'enjeu devient le maintien.", tone="success", icon="🎯")
    else:
        insight_card(
            "Pas de date projetée",
            f"{str(eta['reason']).capitalize()}. Le cône reste affiché quand la pente est estimable ; une date ne l'est que lorsque la pente descend de façon établie.",
            tone="info",
            icon="🧭",
        )
    st.caption(
        "Cône : incertitude sur la pente (croît avec l'horizon), sur le niveau de la tendance et bruit d'une pesée, combinés en quadrature ; "
        f"bruit quotidien actuel ± {format_fr_kg(noise['band'], decimals=1) if noise['ready'] else '—'}. "
        "La projection est bornée à l'objectif final : elle ne descend pas sous 80 kg."
    )

    with st.expander("Détail par scénario (fenêtres 7 / 30 / 90 jours)", expanded=False):
        effort = detect_current_effort(df)
        effort_df = effort["effort_df"] if effort["is_subset"] and len(effort["effort_df"]) >= 3 else None
        legacy = estimate_target_eta(df, target_weight, effort_df=effort_df)
        if legacy.get("credible") and legacy.get("eta"):
            st.write(f"Estimation par pente simple : objectif autour du **{format_fr_date(legacy['eta'])}**.")
            if legacy.get("eta_min") and legacy.get("eta_max"):
                st.caption(f"Plage plausible : {format_fr_date(legacy['eta_min'])} → {format_fr_date(legacy['eta_max'])} | Confiance {format_fr_number(legacy.get('confidence', 0) * 100, 0)} %")
        else:
            alert_banner(str(legacy.get("message", "Estimation indisponible")), "warning")
        for name, sc in legacy.get("scenarios", {}).items():
            icon = {"optimiste": "🟢", "réaliste": "🟡", "pessimiste": "🔴"}.get(name, "⚪")
            if sc.get("credible"):
                st.caption(f"{icon} **{name.title()}** : autour du {format_fr_date(sc['eta'])} ({sc['days_remaining']} j) — rythme : {format_fr_kg_per_week(sc['kg_per_week'], decimals=3, sign=True)}")
            else:
                st.caption(f"{icon} **{name.title()}** : {sc.get('message', 'N/A')} (pente : {format_fr_kg_per_week(sc.get('slope', 0) * 7, decimals=3, sign=True)})")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Leaderboard : quel modèle bat la dernière valeur ?
# ──────────────────────────────────────────────────────────────────────────────


def _leaderboard_section(df: pd.DataFrame, fast_mode: bool) -> None:
    section_header(
        "Leaderboard des modèles (backtest walk-forward)",
        "Chaque modèle est réajusté sur le passé puis jugé sur la semaine suivante, plusieurs fois. La référence est la dernière pesée répétée.",
        "🏁",
    )
    series = df[["Date", "Poids (Kgs)"]].reset_index(drop=True)
    include_sarimax = len(series) >= MIN_ROWS_SARIMAX
    include_auto_arima = (not fast_mode) and len(series) >= MIN_ROWS_AUTO_ARIMA
    with st.spinner("Backtest des modèles…"):
        table = _cached_leaderboard(series, include_sarimax, include_auto_arima)
    if table.empty:
        st.info("Backtest disponible à partir de 8 mesures.")
        return

    best = best_model(table)
    folds = len(walk_forward_splits(len(series)))
    if best is not None:
        naive_row = table[table["Modèle"] == NAIVE_MODEL]
        naive_mae = float(naive_row["MAE"].iloc[0]) if not naive_row.empty else float("nan")
        gain = best["Gain vs dernière valeur (%)"]
        if best["Modèle"] == NAIVE_MODEL or not np.isfinite(gain) or gain <= 5:
            insight_card(
                "Aucun modèle ne fait mieux que la dernière pesée",
                f"Sur {folds} découpages, répéter la dernière valeur donne une erreur moyenne de {format_fr_kg(naive_mae, decimals=2)} ; "
                "aucun modèle ne la réduit d'au moins 5 %. Les projections ci-dessous sont à lire comme des ordres de grandeur.",
                tone="warning",
                icon="🏁",
            )
        else:
            insight_card(
                f"Modèle le plus fiable : {best['Modèle']}",
                f"Erreur absolue moyenne {format_fr_kg(best['MAE'], decimals=2)} sur {folds} découpages, soit {_fr(gain, 0)} % de mieux que la dernière pesée répétée "
                f"({format_fr_kg(naive_mae, decimals=2)})."
                + (
                    f" Son intervalle à 95 % a couvert {_fr(best['Couverture IC 95 % (%)'], 0)} % des pesées de test."
                    if np.isfinite(best["Couverture IC 95 % (%)"])
                    else ""
                ),
                tone="success",
                icon="🏁",
            )

    display = table.copy()
    verdict_icons = {
        "bat la dernière valeur": "🟢",
        "équivalent à la dernière valeur": "⚪",
        "moins bon que la dernière valeur": "🔴",
        "non évalué": "—",
    }
    display["Verdict"] = [f"{verdict_icons.get(v, '')} {v}" if d else f"— non évalué ({detail})" for v, d, detail in zip(display["Verdict"], display["Disponible"], display["Détail"])]
    for column, decimals in (("MAE", 2), ("RMSE", 2), ("Biais", 2)):
        display[column] = display[column].apply(lambda v, d=decimals: f"{format_fr_number(v, decimals=d, sign=(column == 'Biais'))} kg" if pd.notna(v) else "—")
    for column in ("Précision directionnelle (%)", "Gain vs dernière valeur (%)", "Couverture IC 95 % (%)"):
        display[column] = display[column].apply(lambda v: format_fr_number(v, decimals=0, sign=(column.startswith("Gain"))) if pd.notna(v) else "—")
    display = display.drop(columns=["Disponible", "Détail"])
    display = display[["Modèle", "Verdict", "MAE", "Gain vs dernière valeur (%)", "Couverture IC 95 % (%)", "RMSE", "Biais", "Précision directionnelle (%)"]]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        f"Découpage chronologique en {folds} blocs de test d'une semaine de mesures, jamais de mélange aléatoire. "
        "MAE : erreur absolue moyenne ; biais positif = le modèle surestime le poids ; précision directionnelle : part des variations dont le sens est prédit ; "
        "gain : réduction de la MAE par rapport à la dernière valeur ; couverture : part des pesées réelles tombées dans l'intervalle à 95 % annoncé "
        "(une couverture nettement sous 95 % signale un intervalle trop étroit)."
        + (" Auto-ARIMA est ignoré en mode rapide." if fast_mode else "")
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Modèles avancés (expérimental)
# ──────────────────────────────────────────────────────────────────────────────


def _model_comparison(df: pd.DataFrame) -> None:
    st.markdown("**Régression sur variables dérivées (découpage chronologique 80/20)**")
    if len(df) < 10:
        st.warning("Données insuffisantes pour comparer les modèles.")
        return

    # Une colonne facultative entièrement vide (Notes) ne doit pas vider le jeu :
    # seules les variables dérivées du poids conditionnent les lignes gardées.
    numeric_source = df[["Date", "Poids (Kgs)"] + [c for c in df.columns if c not in ("Date", "Poids (Kgs)") and pd.api.types.is_numeric_dtype(df[c])]]
    feat = build_features(numeric_source, height_m=st.session_state.get("height_m", 1.82))
    lag_columns = [c for c in feat.columns if c.startswith("lag_")]
    feat = feat.dropna(subset=lag_columns)
    X = feat.drop(columns=["Date", "Poids (Kgs)"], errors="ignore").select_dtypes(include=["number"]).fillna(0.0)
    y = feat["Poids (Kgs)"]

    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    if len(X_test) < 2:
        st.warning("Pas assez de données de test.")
        return

    models = {
        "Régression linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "Modèle": name,
                "MAE": f"{format_fr_number(mean_absolute_error(y_test, pred), decimals=3)} kg",
                "RMSE": f"{format_fr_number(np.sqrt(mean_squared_error(y_test, pred)), decimals=3)} kg",
                "R²": format_fr_number(r2_score(y_test, pred), decimals=3),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "ℹ️ Les variables incluent les pesées de la veille (retards, moyennes glissantes) : cette comparaison mesure la prédiction à un jour, "
        "pas à trente. Un R² négatif signifie que le modèle fait moins bien que la moyenne du jeu de test."
    )


def _sarima_block(df: pd.DataFrame, horizon: int, target_weight: float) -> None:
    st.caption("Modèle saisonnier hebdomadaire SARIMAX(1,1,1)(1,0,1)₇ — expérimental, à lire avec le leaderboard.")
    try:
        with st.spinner("Ajustement SARIMAX…"):
            pred = _cached_sarimax(df[["Date", "Poids (Kgs)"]].reset_index(drop=True), horizon)
        if pred.empty:
            st.warning("SARIMAX indisponible : données insuffisantes (14 mesures requises).")
            return
        st.plotly_chart(
            _forecast_figure(df, pred, name="Prévision SARIMAX", color=ACCENT_COLOR, fill="rgba(237,161,0,0.14)", title="Prévision SARIMAX avec intervalle à 90 %", target_weight=target_weight, band_label="Intervalle à 90 %"),
            use_container_width=True,
            key="predictions-sarimax",
        )
    except Exception as exc:
        alert_banner(f"Bloc SARIMAX en erreur : {exc}", "warning")


def _auto_arima_block(df: pd.DataFrame, horizon: int, target_weight: float) -> None:
    st.caption("Sélection automatique de l'ordre ARIMA (pmdarima) — expérimental, à lire avec le leaderboard.")
    try:
        with st.spinner("Recherche du meilleur ARIMA…"):
            out = _cached_auto_arima(df[["Date", "Poids (Kgs)"]].reset_index(drop=True), horizon)
        st.plotly_chart(
            _forecast_figure(df, out, name="Prévision Auto-ARIMA", color="#7c3aed", fill="rgba(124,58,237,0.14)", title=f"Prévision Auto-ARIMA {out.attrs.get('order', '')} avec intervalle à 95 %", target_weight=target_weight),
            use_container_width=True,
            key="predictions-auto-arima",
        )
    except Exception as exc:
        alert_banner(f"Auto-ARIMA indisponible : {exc}", "warning")


def _ml_quantile_block(df: pd.DataFrame, horizon: int, target_weight: float) -> None:
    st.markdown("**Prévision ML quantile (P10 / P50 / P90)**")
    st.caption("Régression quantile récursive sur variables dérivées — expérimentale.")
    try:
        with st.spinner("Ajustement des régressions quantiles…"):
            pred = _cached_ml(df[["Date", "Poids (Kgs)"]].reset_index(drop=True), horizon, float(st.session_state.get("height_m", 1.82)))
        if pred.empty:
            st.warning("Prévision ML indisponible : données insuffisantes (20 mesures requises).")
            return
        st.plotly_chart(
            _forecast_figure(df, pred, name="Prévision ML (P50)", color=MEASURE_COLOR, fill=BAND_FILL, title="Prévision ML quantile", target_weight=target_weight, band_label="Intervalle P10 – P90"),
            use_container_width=True,
            key="predictions-ml",
        )
    except Exception as exc:
        alert_banner(f"Bloc ML en erreur : {exc}", "warning")


def _correlogram_figure(values: np.ndarray, *, kind: str, title: str) -> go.Figure | None:
    n = len(values)
    nlags = int(min(30, n // 2 - 1))
    if nlags < 2:
        return None
    if kind == "acf":
        coefficients, interval = acf(values, nlags=nlags, alpha=0.05, fft=True)
    else:
        coefficients, interval = pacf(values, nlags=nlags, alpha=0.05, method="ywm")
    lags = np.arange(1, len(coefficients))
    band = np.asarray(interval)[1:, 1] - coefficients[1:]
    colors = [STATUS_CRITICAL if abs(c) > b else MEASURE_COLOR for c, b in zip(coefficients[1:], band)]
    fig = bar_figure(lags, coefficients[1:], title, y_title="Corrélation", colors=colors, hover=[f"Décalage {lag} j : {c:.2f}" for lag, c in zip(lags, coefficients[1:])], height=300)
    fig.add_scatter(x=list(lags), y=list(band), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=list(lags), y=list(-band), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(139,138,131,0.15)", name="Seuil de signification (95 %)", showlegend=True, hoverinfo="skip")
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title="Décalage (jours)", dtick=1 if nlags <= 15 else 5)
    return fig


def _stl_acf_pacf_block(df: pd.DataFrame) -> None:
    st.caption("Décomposition STL (période 7 jours) sur la série interpolée au jour, puis autocorrélations.")
    try:
        series = df.set_index("Date")["Poids (Kgs)"].asfreq("D").interpolate()
        if len(series) < 20:
            st.warning("Données insuffisantes pour STL/ACF/PACF (20 jours requis).")
            return
        stl = STL(series, period=7, robust=True).fit()
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Tendance", "Composante hebdomadaire", "Résidu"))
        fig.add_scatter(x=series.index, y=stl.trend, mode="lines", name="Tendance", line=dict(color=TREND_COLOR, width=2.2), row=1, col=1)
        fig.add_scatter(x=series.index, y=stl.seasonal, mode="lines", name="Hebdomadaire", line=dict(color=MEASURE_COLOR, width=1.6), row=2, col=1)
        fig.add_bar(x=series.index, y=stl.resid, name="Résidu", marker=dict(color=NEUTRAL_LINE), row=3, col=1)
        apply_layout(fig, "Décomposition STL", height=560, show_legend=False)
        fig.update_yaxes(title="kg")
        st.plotly_chart(fig, use_container_width=True, key="predictions-stl")
        amplitude = float(stl.seasonal.max() - stl.seasonal.min())
        st.caption(
            f"Amplitude de la composante hebdomadaire : {format_fr_kg(amplitude, decimals=2)}. "
            "Une amplitude inférieure au bruit quotidien ne justifie pas de lire un cycle de semaine."
        )
        values = series.to_numpy(dtype=float)
        left, right = st.columns(2)
        with left:
            figure = _correlogram_figure(values, kind="acf", title="Autocorrélation (ACF)")
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True, key="predictions-acf")
        with right:
            figure = _correlogram_figure(values, kind="pacf", title="Autocorrélation partielle (PACF)")
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True, key="predictions-pacf")
        st.caption(
            "Une série de poids est fortement autocorrélée par construction (le poids d'aujourd'hui ressemble à celui d'hier) : "
            "l'ACF décroît lentement. Un pic isolé au décalage 7 sur la PACF signalerait un effet de semaine."
        )
    except Exception as exc:
        alert_banner(f"Bloc STL/ACF/PACF en erreur : {exc}", "warning")


def _scenarios_block(df: pd.DataFrame, target_weight: float) -> None:
    st.caption("Trois rythmes observés (fenêtres 7 / 14 / 30 jours) prolongés sur 90 jours ; guidance, pas certitude.")
    scenarios = prospective_scenarios(df, target_weight)
    if not scenarios:
        st.info("Scénarios disponibles à partir de 14 mesures.")
        return

    scenario_icons = {"optimiste": "🟢", "réaliste": "🟡", "pessimiste": "🔴"}
    scenario_cols = st.columns(len(scenarios))
    for column, (name, data) in zip(scenario_cols, scenarios.items()):
        with column:
            st.markdown(f"**{scenario_icons.get(name, '⚪')} {name.title()}**")
            kpi_card("Vitesse", format_fr_kg_per_week(data["velocity_kg_week"], decimals=2, sign=True))
            for label, key in [("30 j", "proj_30j"), ("60 j", "proj_60j"), ("90 j", "proj_90j")]:
                if data.get(key) is not None:
                    st.caption(f"Dans {label} : **{format_fr_kg(data[key], decimals=1)}**")
            if data.get("eta_date"):
                st.caption(f"🎯 Objectif estimé autour du **{format_fr_date(data['eta_date'])}** ({data['eta_days']} j)")
            elif data.get("eta_days") == 0:
                st.caption("🎉 Objectif déjà atteint")
            else:
                st.caption("Objectif non atteint à ce rythme sur 90 jours")

    fig = go.Figure()
    history = _recent_history(df)
    add_measure_trace(fig, history["Date"], history["Poids (Kgs)"], name="Historique", marker_size=5, width=1.6)
    colors = {"optimiste": STATUS_GOOD, "réaliste": ACCENT_COLOR, "pessimiste": STATUS_CRITICAL}
    last_date = df["Date"].max()
    current = float(df["Poids (Kgs)"].iloc[-1])
    all_dates = list(history["Date"])
    for name, data in scenarios.items():
        dates = data.get("projection_dates", [last_date])
        values = data.get("projection_values", [current])
        all_dates.extend(dates)
        add_line_trace(fig, dates, values, name=name.title(), color=colors.get(name, NEUTRAL_LINE), dash="dash", width=2.0)
    add_horizontal_reference(fig, all_dates, target_weight, name=f"Objectif : {format_fr_kg(target_weight, decimals=1)}", color=TARGET_COLOR)
    apply_layout(fig, "Projections à 90 jours", y_title="Poids (kg)", height=420)
    st.plotly_chart(fig, use_container_width=True, key="predictions-scenarios")


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    df = _df()
    page_hero(
        "Analyse prédictive",
        "Prévisions",
        "Où mène le rythme actuel, avec quelle incertitude, et quels modèles méritent confiance sur vos propres données.",
        meta="Projections indicatives — aucune n'est une promesse",
    )
    if df.empty:
        empty_state("Ajoutez des mesures dans Journal pour lancer les prévisions.")
        return

    df = df.sort_values("Date").reset_index(drop=True)
    target_weight = float(get_target_weights(st.session_state)[-1])
    fast_mode = bool(st.session_state.get("fast_mode", False))

    control_cols = st.columns([2, 3])
    with control_cols[0]:
        horizon = st.slider("Horizon (jours)", 7, 90, 30, help="S'applique à la projection principale et aux modèles avancés.")
    with control_cols[1]:
        st.caption(
            f"{len(df)} mesures du {format_fr_date(df['Date'].min())} au {format_fr_date(df['Date'].max())} · objectif principal {format_fr_kg(target_weight, decimals=1)}. "
            "Les modèles lourds sont mis en cache : la première ouverture est la plus lente."
        )

    _primary_projection(df, horizon, target_weight)
    _leaderboard_section(df, fast_mode)

    section_header("Modèles avancés (expérimental)", "Des approches classiques de séries temporelles, à confronter au leaderboard avant d'y croire.", "🧪")
    t1, t2, t3, t4, t5 = st.tabs(["Régression ML", "SARIMA", "Auto-ARIMA", "STL / ACF-PACF", "Scénarios"])
    with t1:
        _model_comparison(df)
        _ml_quantile_block(df, horizon, target_weight)
    with t2:
        _sarima_block(df, horizon, target_weight)
    with t3:
        if fast_mode:
            st.warning("Auto-ARIMA sauté en mode rapide (tests).")
        else:
            _auto_arima_block(df, horizon, target_weight)
    with t4:
        if fast_mode:
            st.warning("STL/ACF/PACF sauté en mode rapide (tests).")
        else:
            _stl_acf_pacf_block(df)
    with t5:
        _scenarios_block(df, target_weight)

    section_header("Vitesses de variation", "Écart entre la dernière pesée et la pesée la plus proche de chaque recul, ramené à la semaine.", "📊")
    vel = weight_velocity(df, windows=(7, 14, 30, 90))
    vel_cols = st.columns(4)
    for column, (window, value) in zip(vel_cols, vel.items()):
        with column:
            kpi_card(f"Vitesse {window} j", format_fr_kg_per_week(value, decimals=2, sign=True) if value is not None else "—")
    st.caption("Convention : valeur négative = perte de poids. Deux pesées seulement par fenêtre : ces vitesses portent le bruit quotidien, contrairement au rythme avec intervalle ci-dessus.")


main()
