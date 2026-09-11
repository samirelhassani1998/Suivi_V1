"""Onglet WHOOP : connexion OAuth2, synchronisation et croisement avec le poids."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.formatting import MISSING_VALUE as MISSING_TEXT, format_fr_date, format_fr_number
from app.core.whoop_analytics import (
    KCAL_PER_KG,
    MIN_DAYS_CORRELATION,
    MIN_DAYS_TRAINING_LOAD,
    analysis_availability,
    coverage_report,
    daily_grid,
    energy_balance,
    lagged_correlations,
    recovery_drivers,
    recovery_zones,
    sleep_debt_summary,
    strain_recovery_balance,
    training_load,
    weekday_profile,
    weekly_rollup,
)
from app.core.session_state import (
    DEFAULT_WHOOP_SYNC_DAYS,
    clear_whoop_session,
    ensure_session_defaults,
    get_filtered_or_working_data,
)
from app.core.whoop_session import (
    clear_oauth_params,
    clear_pending_callback,
    detect_base_url,
    pending_callback,
    redirect_uri_candidates,
)
from app.core.whoop import (
    DEFAULT_REDIRECT_URI,
    build_scopes,
    WhoopError,
    WhoopToken,
    available_metrics,
    build_daily_frame,
    credentials_from_sources,
    cycles_to_frame,
    ensure_fresh_token,
    exchange_code_for_token,
    fetch_collection,
    fetch_profile,
    generate_state,
    merge_with_weight,
    recoveries_to_frame,
    sleeps_to_frame,
    summarise_daily,
    workouts_to_frame,
    build_authorization_url,
)
from app.ui.components import empty_state, insight_card, kpi_card, page_hero, section_header

METRIC_DECIMALS = {
    "Récupération (%)": 0,
    "HRV (ms)": 0,
    "FC repos (bpm)": 0,
    "Température peau (°C)": 1,
    "SpO2 (%)": 1,
    "Sommeil (heures)": 1,
    "Besoin de sommeil (heures)": 1,
    "Dette de sommeil (heures)": 1,
    "Heure de coucher": 1,
    "Performance sommeil (%)": 0,
    "Efficacité sommeil (%)": 0,
    "Régularité sommeil (%)": 0,
    "Sommeil profond (heures)": 1,
    "Sommeil REM (heures)": 1,
    "Perturbations sommeil": 0,
    "Strain": 1,
    "Calories (kcal)": 0,
    "FC moyenne (bpm)": 0,
    "FC max (bpm)": 0,
}

# Une hausse est favorable pour ces métriques, défavorable pour les autres.
HIGHER_IS_BETTER = {
    "Récupération (%)",
    "HRV (ms)",
    "Sommeil (heures)",
    "Performance sommeil (%)",
    "Efficacité sommeil (%)",
    "Régularité sommeil (%)",
    "Sommeil profond (heures)",
    "Sommeil REM (heures)",
    "SpO2 (%)",
    "Calories (kcal)",
}


def _secrets_mapping() -> dict:
    """Lecture défensive des secrets : absents en local, présents sur Streamlit Cloud."""
    try:
        return {key: st.secrets[key] for key in st.secrets}
    except Exception:
        return {}


def _plot_template() -> str:
    return str(st.session_state.get("theme", "plotly"))


def _default_redirect_uri() -> str:
    """URL de redirection proposée par défaut : celle de l'application elle-même."""
    detected = detect_base_url(default="")
    return detected or DEFAULT_REDIRECT_URI


def _resolve_credentials():
    overrides = dict(st.session_state.get("whoop_manual_credentials", {}) or {})
    return credentials_from_sources(
        _secrets_mapping(),
        os.environ,
        overrides,
        default_redirect_uri=_default_redirect_uri(),
        scopes=build_scopes(offline=bool(st.session_state.get("whoop_request_offline", True))),
    )


def _stored_token() -> WhoopToken | None:
    raw = st.session_state.get("whoop_token")
    if not raw:
        return None
    try:
        token = WhoopToken.from_dict(raw)
    except Exception:
        return None
    return token if token.access_token else None


def _store_token(token: WhoopToken) -> None:
    st.session_state["whoop_token"] = token.to_dict()


def _consume_oauth_callback(credentials) -> None:
    """Traite le retour de redirection WHOOP capté par le point d'entrée."""
    # Le retour peut aussi arriver directement sur cette page : on relit l'URL.
    pending = pending_callback()
    if pending is None:
        try:
            code = st.query_params.get("code")
            error = st.query_params.get("error")
        except Exception:
            return
        if not code and not error:
            return
        pending = {
            "code": str(code or ""),
            "state": str(st.query_params.get("state") or ""),
            "error": str(error or ""),
            "error_description": str(st.query_params.get("error_description") or ""),
        }
        clear_oauth_params()

    clear_pending_callback()

    if pending.get("error"):
        _render_oauth_error(pending, credentials)
        return

    expected_state = st.session_state.get("whoop_oauth_state")
    state = pending.get("state") or ""
    if expected_state and state and state != expected_state:
        st.error(
            "État OAuth inattendu : la réponse ne correspond pas à la demande émise. "
            "Relancez la connexion depuis cette page."
        )
        return

    try:
        token = exchange_code_for_token(credentials, str(pending.get("code", "")))
    except WhoopError as exc:
        st.error(f"Connexion WHOOP impossible : {exc}")
        return
    except Exception:
        st.error("Connexion WHOOP impossible : réponse inattendue du service.")
        return

    _store_token(token)
    st.session_state["whoop_oauth_state"] = None
    st.success("Compte WHOOP connecté.")


def _render_oauth_error(pending: dict, credentials) -> None:
    """Traduit l'erreur renvoyée par WHOOP en action concrète."""
    code = str(pending.get("error", ""))
    description = str(pending.get("error_description", "")).strip()
    st.error(f"WHOOP a refusé l'autorisation ({code}).{f' {description}' if description else ''}")
    if "redirect" in description.lower() or code == "invalid_request":
        _render_redirect_uri_help(credentials)
    elif code == "access_denied":
        st.info("L'autorisation a été refusée côté WHOOP. Relancez la connexion pour réessayer.")


def _render_redirect_uri_help(credentials) -> None:
    """Explique la correspondance exacte exigée par WHOOP sur la Redirect URI."""
    st.warning(
        "WHOOP compare la Redirect URI **caractère pour caractère** avec celles "
        "déclarées dans votre application du tableau de bord développeur. "
        "Déclarez-y exactement l'URL ci-dessous, puis relancez la connexion."
    )
    st.caption("URL envoyée par cette application :")
    st.code(credentials.redirect_uri, language="text")
    others = [uri for uri in redirect_uri_candidates() if uri != credentials.redirect_uri]
    if others:
        st.caption(
            "Ces variantes sont également gérées par l'application : déclarer l'une "
            "d'elles côté WHOOP fonctionne aussi, à condition de la recopier ici à l'identique."
        )
        st.code("\n".join(others), language="text")
    st.caption(
        "Pièges fréquents : une barre oblique finale en trop ou en moins, `http` au lieu "
        "de `https`, ou une URL d'application Streamlit qui a changé depuis la déclaration."
    )


def _scope_controls() -> None:
    """Permet de retirer ``offline`` si l'application WHOOP ne l'a pas activé."""
    st.checkbox(
        "Demander le renouvellement automatique du jeton (scope `offline`)",
        key="whoop_request_offline",
        help=(
            "Laissez coché dans le cas général. Décochez uniquement si WHOOP refuse "
            "l'autorisation pour cause de scope invalide : la connexion fonctionnera "
            "alors, mais expirera au bout de quelques heures sans renouvellement."
        ),
    )


def _connection_panel(credentials) -> None:
    section_header(
        "Connexion WHOOP",
        "Autorisez l'application à lire vos données WHOOP en lecture seule.",
        "🔗",
    )

    with st.expander("Identifiants de l'application WHOOP", expanded=not credentials.is_complete):
        st.caption(
            "Renseignez de préférence ces valeurs dans les secrets Streamlit "
            "(`[whoop] client_id / client_secret / redirect_uri`). La saisie ci-dessous "
            "ne vit que dans votre session et n'est jamais enregistrée dans le dépôt."
        )
        overrides = dict(st.session_state.get("whoop_manual_credentials", {}) or {})
        with st.form("whoop_credentials_form"):
            client_id = st.text_input("Client ID", value=str(overrides.get("client_id", "")), type="password")
            client_secret = st.text_input("Client Secret", value=str(overrides.get("client_secret", "")), type="password")
            redirect_uri = st.text_input(
                "Redirect URI",
                value=str(overrides.get("redirect_uri", credentials.redirect_uri)),
                help="Doit être identique, caractère pour caractère, à l'URL déclarée dans le tableau de bord développeur WHOOP.",
            )
            st.caption("URL détectée pour cette application : " + f"`{_default_redirect_uri()}`")
            if st.form_submit_button("Utiliser ces identifiants pour la session"):
                st.session_state["whoop_manual_credentials"] = {
                    "client_id": client_id.strip(),
                    "client_secret": client_secret.strip(),
                    "redirect_uri": redirect_uri.strip(),
                }
                st.success("Identifiants enregistrés pour cette session.")

    credentials = _resolve_credentials()
    if not credentials.is_complete:
        st.info(
            "Client ID, Client Secret et Redirect URI sont nécessaires pour lancer l'autorisation WHOOP."
        )
        return

    if st.session_state.get("whoop_oauth_state") is None:
        st.session_state["whoop_oauth_state"] = generate_state()

    try:
        auth_url = build_authorization_url(credentials, str(st.session_state["whoop_oauth_state"]))
    except WhoopError as exc:
        st.error(str(exc))
        return

    cols = st.columns([2, 1])
    with cols[0]:
        st.link_button("Autoriser l'accès WHOOP", auth_url, use_container_width=True)
    with cols[1]:
        if st.button("Réinitialiser la connexion", use_container_width=True):
            clear_whoop_session()
            st.info("Session WHOOP réinitialisée.")

    with st.expander("La redirection est refusée par WHOOP ?", expanded=False):
        _render_redirect_uri_help(credentials)
        st.divider()
        _scope_controls()


def _sync_panel(credentials, token: WhoopToken) -> None:
    section_header("Synchronisation", "Importez vos cycles, récupérations, nuits et séances.", "🔄")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        days = st.number_input(
            "Profondeur d'historique (jours)",
            min_value=7,
            max_value=365,
            value=int(st.session_state.get("whoop_sync_days", DEFAULT_WHOOP_SYNC_DAYS)),
            step=7,
        )
    with cols[1]:
        st.metric("Dernière synchro", format_fr_date(st.session_state.get("whoop_last_sync")))
    with cols[2]:
        launch = st.button("Synchroniser maintenant", use_container_width=True, type="primary")

    st.session_state["whoop_sync_days"] = int(days)

    if not launch:
        return

    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.Timedelta(days=int(days))

    try:
        fresh = ensure_fresh_token(credentials, token)
        if fresh.access_token != token.access_token:
            _store_token(fresh)
        with st.spinner("Récupération des données WHOOP…"):
            recovery = recoveries_to_frame(fetch_collection("recovery", fresh, start=start, end=end))
            sleep = sleeps_to_frame(fetch_collection("sleep", fresh, start=start, end=end))
            cycle = cycles_to_frame(fetch_collection("cycle", fresh, start=start, end=end))
            workouts = workouts_to_frame(fetch_collection("workout", fresh, start=start, end=end))
            profile = fetch_profile(fresh)
    except WhoopError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("Synchronisation WHOOP interrompue : service indisponible ou réseau bloqué.")
        return

    st.session_state["whoop_daily"] = build_daily_frame(recovery, sleep, cycle)
    st.session_state["whoop_workouts"] = workouts
    st.session_state["whoop_profile"] = profile
    st.session_state["whoop_last_sync"] = pd.Timestamp.utcnow().tz_localize(None)
    st.success(
        f"{len(recovery)} récupérations, {len(sleep)} nuits, {len(cycle)} cycles et "
        f"{len(workouts)} séances importés."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mise en forme
# ──────────────────────────────────────────────────────────────────────────────


def _metric_value(metric: str, value: float) -> str:
    return format_fr_number(value, decimals=METRIC_DECIMALS.get(metric, 1))


def _metric_delta(metric: str, delta: float) -> str:
    """Delta lisible par Streamlit : signe ASCII (flèche et couleur) et décimale française."""
    formatted = format_fr_number(abs(delta), decimals=METRIC_DECIMALS.get(metric, 1))
    if formatted == "—":
        return formatted
    return f"{'-' if delta < 0 else '+'}{formatted}"


def _format_table(frame: pd.DataFrame, decimals: dict[str, int] | None = None) -> pd.DataFrame:
    """Rend un tableau lisible : décimales maîtrisées, dates courtes, vides explicites.

    Les tableaux bruts affichaient « 2026-09-10 00:00:00 » et « 110.7652 » ; le
    bruit numérique masquait l'information utile.
    """
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame()
    rules = decimals or {}
    display = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            display[column] = series.apply(format_fr_date)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            places = rules.get(column, METRIC_DECIMALS.get(column, 1))
            display[column] = series.apply(lambda value, places=places: format_fr_number(value, decimals=places))
        else:
            display[column] = series.fillna(MISSING_TEXT).astype(str)
    return display


def _daily_axis(figure: go.Figure) -> go.Figure:
    """Force un axe calendaire.

    Avec deux ou trois points, Plotly bascule en graduations horaires et affiche
    « 03:00, 06:00 » sur des mesures quotidiennes, ce qui est trompeur.
    """
    figure.update_xaxes(tickformat="%d/%m", dtick="D1", ticklabelmode="period")
    return figure


def _line_chart(
    daily: pd.DataFrame,
    metrics: list[str],
    title: str,
    y_title: str,
    *,
    key: str | None = None,
) -> None:
    """Courbes sur calendrier continu : un jour sans mesure reste un trou visible."""
    grid = daily_grid(daily)
    usable = [metric for metric in metrics if metric in grid.columns and grid[metric].notna().any()]
    if not usable:
        st.info("Métrique indisponible sur la période importée.")
        return

    figure = go.Figure()
    for metric in usable:
        figure.add_scatter(
            x=grid["Date"],
            y=grid[metric],
            mode="lines+markers",
            name=metric,
            connectgaps=False,
            hovertemplate="%{x|%d/%m/%Y}<br>" + metric + " : %{y:.2f}<extra></extra>",
        )
    figure.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
        template=_plot_template(),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(_daily_axis(figure), use_container_width=True, key=key)


def _zone_badge(zone: str | None) -> str:
    return {"Vert": "🟢", "Jaune": "🟡", "Rouge": "🔴"}.get(str(zone), "⚪")


# ──────────────────────────────────────────────────────────────────────────────
# Onglets
# ──────────────────────────────────────────────────────────────────────────────


def _today_panel(daily: pd.DataFrame) -> None:
    """Ce que WHOOP montre sur son écran d'accueil, resitué dans une tendance."""
    zones = recovery_zones(daily)
    summary = summarise_daily(daily, days=7)
    grid = daily_grid(daily)
    # Un subset vide ferait tomber toutes les lignes sans le dire : on vérifie
    # d'abord qu'au moins une colonne clé existe.
    key_columns = [c for c in ("Récupération (%)", "Strain", "Sommeil (heures)") if c in grid.columns]
    latest = grid.dropna(subset=key_columns, how="all") if key_columns else grid.iloc[0:0]
    latest_date = latest["Date"].max() if not latest.empty else None

    section_header(
        "État actuel",
        f"Dernière journée mesurée : {format_fr_date(latest_date)}." if latest_date is not None else "Aucune journée complète mesurée.",
        "🎯",
    )

    headline = [m for m in ("Récupération (%)", "HRV (ms)", "Sommeil (heures)", "Strain") if m in summary]
    if not headline:
        empty_state("Aucune métrique WHOOP exploitable sur la période importée.")
        return

    st.caption("Moyennes des 7 derniers jours disponibles, comparées aux 7 jours précédents.")
    columns = st.columns(len(headline))
    for column, metric in zip(columns, headline):
        stats = summary[metric]
        delta = stats["delta"]
        label = metric
        if metric == "Récupération (%)" and zones["latest_zone"]:
            label = f"{_zone_badge(zones['latest_zone'])} {metric}"
        with column:
            if pd.notna(delta):
                st.metric(
                    label,
                    _metric_value(metric, stats["current"]),
                    _metric_delta(metric, delta),
                    delta_color="normal" if metric in HIGHER_IS_BETTER else "inverse",
                )
            else:
                kpi_card(label, _metric_value(metric, stats["current"]), help_text="Pas encore de période précédente pour comparer.")


def _coverage_panel(daily: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Dit franchement sur quelle matière les analyses reposent."""
    coverage = coverage_report(daily)
    with st.expander("Fiabilité : sur quoi reposent ces chiffres ?", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            kpi_card("Jours mesurés", f"{coverage['days_with_data']}", help_text=f"Sur {coverage['span_days']} jour(s) de période")
        with cols[1]:
            kpi_card("Couverture", f"{format_fr_number(coverage['coverage_pct'], decimals=0)} %", help_text=f"{coverage['gaps']} jour(s) sans mesure")
        with cols[2]:
            kpi_card("Jours croisés avec une pesée", f"{len(merged)}")

        st.caption("Analyses débloquées au fil des mesures :")
        for item in analysis_availability(daily, merged):
            if item.ready:
                st.markdown(f"- ✅ **{item.name}** — active ({item.available} jour(s))")
            else:
                st.markdown(
                    f"- ⏳ **{item.name}** — encore {item.missing} jour(s) de mesure "
                    f"({item.available}/{item.required})"
                )
        st.caption(
            "Un seuil n'est pas une coquetterie : sur trop peu de jours, une corrélation "
            "ou une pente reflète le bruit de mesure plutôt qu'une tendance."
        )


def _overview_tab(daily: pd.DataFrame, merged: pd.DataFrame) -> None:
    _today_panel(daily)
    _coverage_panel(daily, merged)

    section_header("Tendances", "Chaque indicateur sur son échelle, les jours sans mesure restant vides.", "📈")
    # Récupération, sommeil et strain ne partagent pas d'unité : les superposer
    # écrasait le sommeil et le strain sous l'échelle 0-100 de la récupération.
    trend_specs = [
        (["Récupération (%)"], "Récupération", "%"),
        (["Sommeil (heures)", "Besoin de sommeil (heures)"], "Sommeil obtenu et besoin estimé", "heures"),
        (["Strain"], "Charge quotidienne", "strain"),
    ]
    for metrics, title, unit in trend_specs:
        _line_chart(daily, metrics, title, unit, key=f"whoop-trend-{title}")

    with st.expander("Données journalières WHOOP", expanded=False):
        st.dataframe(_format_table(daily), use_container_width=True, hide_index=True)
        st.download_button(
            "Exporter les données WHOOP (CSV)",
            daily.to_csv(index=False).encode("utf-8"),
            file_name="whoop_daily.csv",
            mime="text/csv",
        )


def _recovery_tab(daily: pd.DataFrame) -> None:
    zones = recovery_zones(daily)
    section_header("Zones de récupération", "Répartition selon les seuils WHOOP : rouge sous 34 %, vert à partir de 67 %.", "🚦")
    if zones["days"] == 0:
        st.info("Aucun score de récupération sur la période importée.")
    else:
        cols = st.columns([2, 3])
        with cols[0]:
            st.dataframe(_format_table(zones["counts"], {"Jours": 0, "Part (%)": 1}), use_container_width=True, hide_index=True)
            if zones["latest_zone"]:
                st.caption(
                    f"Dernier score : {_zone_badge(zones['latest_zone'])} "
                    f"{format_fr_number(zones['latest'], decimals=0)} % (zone {zones['latest_zone'].lower()})."
                )
        with cols[1]:
            counts = zones["counts"]
            figure = px.bar(
                counts,
                x="Zone",
                y="Jours",
                color="Zone",
                color_discrete_map={"Rouge": "#dc2626", "Jaune": "#f59e0b", "Vert": "#16a34a"},
                title="Jours par zone de récupération",
                template=_plot_template(),
            )
            figure.update_layout(showlegend=False, margin=dict(t=60, b=40))
            st.plotly_chart(figure, use_container_width=True, key="whoop-recovery-zones")

    _line_chart(daily, ["Récupération (%)"], "Score de récupération quotidien", "%", key="whoop-recovery-series")
    _line_chart(daily, ["HRV (ms)", "FC repos (bpm)"], "Variabilité cardiaque et fréquence au repos", "Valeur", key="whoop-recovery-hrv")
    _line_chart(daily, ["Température peau (°C)", "SpO2 (%)"], "Température cutanée et saturation en oxygène", "Valeur", key="whoop-recovery-temp")

    section_header("Moteurs de la récupération", "Ce qui fait réellement bouger votre score, chiffré plutôt que supposé.", "🔬")
    drivers = recovery_drivers(daily)
    if not drivers["ready"]:
        st.info(
            f"Analyse disponible à partir de {drivers['required_days']} jours complets "
            f"(actuellement {drivers['days']})."
        )
    else:
        driver_cols = st.columns(len(drivers["coefficients"]) + 1)
        for column, (name, coefficient) in zip(driver_cols, drivers["coefficients"].items()):
            unit = "h" if "Sommeil" in name else "point de strain"
            with column:
                kpi_card(
                    name,
                    f"{format_fr_number(coefficient, decimals=1, sign=True)} pt",
                    help_text=f"Variation du score de récupération par {unit} supplémentaire.",
                )
        with driver_cols[-1]:
            kpi_card("Pouvoir explicatif", f"{format_fr_number(drivers['r_squared'] * 100, decimals=0)} %")
        if drivers["r_squared"] < 0.15:
            st.warning(
                "Le modèle explique une part faible des variations : sur vos données actuelles, "
                "sommeil et charge de la veille ne suffisent pas à prédire la récupération."
            )


def _sleep_tab(daily: pd.DataFrame) -> None:
    debt = sleep_debt_summary(daily, days=7)
    section_header("Dette de sommeil", "Écart entre le besoin estimé par WHOOP et le sommeil réellement obtenu.", "🛌")
    if debt["nights"] == 0:
        st.info("Aucune nuit exploitable sur la période importée.")
    else:
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "Dette cumulée (7 nuits)",
                f"{format_fr_number(debt['cumulative_debt'], decimals=1, sign=True)} h",
                help="Positif : vous dormez moins que le besoin estimé.",
            )
        with cols[1]:
            kpi_card("Dette moyenne / nuit", f"{format_fr_number(debt['mean_debt'], decimals=1, sign=True)} h")
        with cols[2]:
            kpi_card("Sommeil moyen", f"{format_fr_number(debt['mean_sleep'], decimals=1)} h")
        with cols[3]:
            kpi_card("Besoin moyen", f"{format_fr_number(debt['mean_need'], decimals=1)} h")

    _line_chart(daily, ["Sommeil (heures)", "Besoin de sommeil (heures)"], "Sommeil obtenu face au besoin", "Heures", key="whoop-sleep-need")
    _line_chart(daily, ["Dette de sommeil (heures)"], "Dette de sommeil nuit par nuit", "Heures", key="whoop-sleep-debt")

    quality_metrics = ["Performance sommeil (%)", "Efficacité sommeil (%)", "Régularité sommeil (%)"]
    # Une métrique constamment nulle n'est pas une information : WHOOP la laisse
    # vide tant que l'historique est trop court pour la calculer.
    informative = [
        metric
        for metric in quality_metrics
        if metric in daily.columns and daily[metric].notna().any() and float(daily[metric].fillna(0).abs().sum()) > 0
    ]
    _line_chart(daily, informative or quality_metrics, "Qualité du sommeil", "%", key="whoop-sleep-quality")
    ignored = [metric for metric in quality_metrics if metric not in informative]
    if informative and ignored:
        st.caption(
            "Métrique(s) masquée(s) car encore nulle(s) chez WHOOP : "
            + ", ".join(ignored)
            + ". Elles apparaîtront une fois l'historique suffisant."
        )

    stages = [m for m in ("Sommeil profond (heures)", "Sommeil REM (heures)") if m in daily.columns and daily[m].notna().any()]
    if stages:
        stacked = daily_grid(daily).melt(id_vars="Date", value_vars=stages, var_name="Stade", value_name="Heures").dropna(subset=["Heures"])
        figure = px.bar(stacked, x="Date", y="Heures", color="Stade", title="Répartition des stades de sommeil", template=_plot_template())
        figure.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="Heures", margin=dict(t=60, b=40), bargap=0.35)
        st.plotly_chart(_daily_axis(figure), use_container_width=True, key="whoop-sleep-stages")

    if "Heure de coucher" in daily.columns and daily["Heure de coucher"].notna().sum() >= 3:
        section_header("Régularité du coucher", "Heure d'endormissement ramenée sur une échelle continue autour de minuit.", "🕰️")
        bedtime = daily_grid(daily)[["Date", "Heure de coucher"]].dropna()
        spread = float(bedtime["Heure de coucher"].std())
        figure = go.Figure()
        figure.add_scatter(
            x=bedtime["Date"],
            y=bedtime["Heure de coucher"],
            mode="lines+markers",
            name="Heure de coucher",
            connectgaps=False,
            hovertemplate="%{x|%d/%m/%Y}<br>Coucher : %{y:.2f} h<extra></extra>",
        )
        figure.update_layout(
            title="Heure de coucher",
            xaxis_title="Date",
            yaxis_title="Heure (négatif = avant minuit)",
            template=_plot_template(),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(_daily_axis(figure), use_container_width=True, key="whoop-sleep-bedtime")
        st.caption(
            f"Dispersion des couchers : {format_fr_number(spread, decimals=1)} h d'écart-type. "
            "Une dispersion faible traduit un rythme régulier, que WHOOP relie à la qualité du sommeil."
        )


def _effort_tab(daily: pd.DataFrame, workouts: pd.DataFrame) -> None:
    load = training_load(daily)
    section_header(
        "Charge d'entraînement",
        "Charge des 7 derniers jours rapportée à celle des 28 derniers : un indicateur de progression que WHOOP n'affiche pas.",
        "⚖️",
    )
    if not np.isfinite(load["ratio"]):
        st.info(
            f"Indicateur disponible à partir de {MIN_DAYS_TRAINING_LOAD} jours de mesure "
            f"(actuellement {load['days']})."
        )
    else:
        cols = st.columns(3)
        with cols[0]:
            kpi_card("Charge aigüe (7 j)", format_fr_number(load["acute"], decimals=1))
        with cols[1]:
            kpi_card("Charge chronique (28 j)", format_fr_number(load["chronic"], decimals=1))
        with cols[2]:
            kpi_card("Rapport aigu / chronique", format_fr_number(load["ratio"], decimals=2), help_text=load["status"])
        tone = "warning" if load["ratio"] > 1.5 or load["ratio"] < 0.8 else "success"
        insight_card(
            f"Charge : {load['status']}",
            "Entre 0,8 et 1,3, la progression est généralement considérée comme soutenable. "
            "Au-delà de 1,5, l'augmentation est brutale par rapport à vos habitudes récentes.",
            tone=tone,
            icon="⚖️",
        )

    _line_chart(daily, ["Strain"], "Charge quotidienne (strain)", "Strain", key="whoop-effort-strain")
    _line_chart(daily, ["Calories (kcal)"], "Dépense énergétique quotidienne", "kcal", key="whoop-effort-calories")

    balance = strain_recovery_balance(daily)
    if not balance.empty:
        flagged = balance[balance["Signal"] != "cohérent"]
        if not flagged.empty:
            section_header("Jours à surveiller", "Charge et récupération qui ne vont pas dans le même sens.", "⚠️")
            st.dataframe(
                _format_table(flagged, {"Récupération (%)": 0, "Strain": 1}),
                use_container_width=True,
                hide_index=True,
            )

    if workouts is None or workouts.empty:
        st.info("Aucune séance enregistrée sur la période importée.")
        return

    section_header("Séances", "Détail des entraînements enregistrés par le bracelet.", "🏃")
    detailed = workouts.copy()
    detailed["Sport"] = detailed["Sport"].astype(str).str.replace("_", " ").str.capitalize()
    by_sport = (
        detailed.groupby("Sport", as_index=False)
        .agg(
            Séances=("Sport", "size"),
            **{
                "Durée totale (min)": ("Durée (min)", "sum"),
                "Strain moyen": ("Strain séance", "mean"),
                "Calories (kcal)": ("Calories séance (kcal)", "sum"),
            },
        )
        .sort_values("Séances", ascending=False)
    )
    st.dataframe(
        _format_table(by_sport, {"Séances": 0, "Durée totale (min)": 0, "Strain moyen": 1, "Calories (kcal)": 0}),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(
        _format_table(
            detailed.sort_values("Date", ascending=False),
            {"Durée (min)": 0, "Strain séance": 1, "Calories séance (kcal)": 0, "FC moyenne (bpm)": 0, "FC max (bpm)": 0, "Distance (km)": 2},
        ),
        use_container_width=True,
        hide_index=True,
    )


def _energy_balance_panel(merged: pd.DataFrame) -> None:
    """Le croisement que ni WHOOP ni une balance ne peuvent produire seuls."""
    section_header(
        "Bilan énergétique estimé",
        "La dépense vient de WHOOP, le déficit de votre courbe de poids : l'apport s'en déduit.",
        "🔥",
    )
    balance = energy_balance(merged)
    if not balance["ready"]:
        st.info(
            f"Estimation disponible à partir de {balance['required_days']} jours croisés "
            f"(actuellement {balance['days']}). Il manque {max(0, balance['required_days'] - balance['days'])} jour(s)."
        )
        if np.isfinite(balance["mean_burn"]):
            st.caption(f"Dépense moyenne déjà mesurée : {format_fr_number(balance['mean_burn'], decimals=0)} kcal/jour.")
        return

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Dépense moyenne", f"{format_fr_number(balance['mean_burn'], decimals=0)} kcal/j")
    with cols[1]:
        kpi_card("Tendance du poids", f"{format_fr_number(balance['slope_kg_per_week'], decimals=2, sign=True)} kg/sem")
    with cols[2]:
        kpi_card(
            "Déséquilibre implicite",
            f"{format_fr_number(balance['imbalance_per_day'], decimals=0, sign=True)} kcal/j",
            help_text="Négatif : déficit. Déduit de la pente du poids.",
        )
    with cols[3]:
        kpi_card("Apport estimé", f"{format_fr_number(balance['estimated_intake'], decimals=0)} kcal/j")

    st.caption(
        "Méthode : la pente du poids est convertie en énergie sur la base de "
        f"{format_fr_number(KCAL_PER_KG, decimals=0)} kcal par kilogramme, puis ajoutée à la dépense "
        "mesurée par WHOOP. C'est une estimation : les variations d'eau et de glycogène, "
        "le bruit de pesée et la précision de la dépense WHOOP s'y répercutent directement."
    )


def _weight_tab(daily: pd.DataFrame, merged: pd.DataFrame) -> None:
    weights = get_filtered_or_working_data()
    if weights.empty:
        empty_state("Aucune mesure de poids chargée : le croisement nécessite les deux sources.")
        return
    if merged.empty:
        st.info("Aucun jour commun entre vos pesées et la période WHOOP importée.")
        return

    st.caption(
        f"{len(merged)} jour(s) couverts à la fois par une pesée et par une mesure WHOOP "
        f"({format_fr_date(merged['Date'].min())} → {format_fr_date(merged['Date'].max())})."
    )

    _energy_balance_panel(merged)

    section_header("Poids et métrique WHOOP", "Deux échelles distinctes, superposées sur la même période.", "⚖️")
    metrics = available_metrics(merged)
    if not metrics:
        st.info("Aucune métrique WHOOP exploitable sur les jours communs.")
        return

    default_index = metrics.index("Récupération (%)") if "Récupération (%)" in metrics else 0
    metric = st.selectbox("Métrique WHOOP à comparer au poids", metrics, index=default_index)

    grid = daily_grid(merged)
    figure = go.Figure()
    figure.add_scatter(
        x=grid["Date"],
        y=grid["Poids (Kgs)"],
        mode="lines+markers",
        name="Poids (kg)",
        connectgaps=False,
        hovertemplate="%{x|%d/%m/%Y}<br>Poids : %{y:.2f} kg<extra></extra>",
    )
    figure.add_scatter(
        x=grid["Date"],
        y=grid[metric],
        mode="lines+markers",
        name=metric,
        yaxis="y2",
        connectgaps=False,
        hovertemplate="%{x|%d/%m/%Y}<br>" + metric + " : %{y:.2f}<extra></extra>",
    )
    figure.update_layout(
        title=f"Poids et {metric}",
        xaxis_title="Date",
        yaxis=dict(title="Poids (kg)"),
        yaxis2=dict(title=metric, overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        template=_plot_template(),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(_daily_axis(figure), use_container_width=True, key="whoop-weight-dual-axis")

    section_header(
        "Corrélations décalées",
        "Un entraînement pèse rarement sur la balance le jour même : chaque métrique est testée avec 0, 1 et 2 jours de décalage.",
        "🧮",
    )
    correlations = lagged_correlations(merged)
    if correlations.empty:
        st.info(
            f"Corrélations calculées à partir de {MIN_DAYS_CORRELATION} jours communs "
            f"(actuellement {len(merged)})."
        )
    else:
        st.dataframe(
            _format_table(correlations, {"Décalage (jours)": 0, "Corrélation": 3, "Observations": 0}),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "⚠️ Une corrélation n'est pas une causalité : sur de courtes séries, ces valeurs restent "
        "indicatives et sensibles au bruit de mesure (hydratation, horaire de pesée)."
    )

    weekly = weekly_rollup(merged)
    if not weekly.empty:
        section_header("Synthèse hebdomadaire", "Une ligne par semaine : récupération, sommeil, charge et variation de poids.", "🗓️")
        st.dataframe(
            _format_table(
                weekly,
                {"Jours": 0, "Récupération (%)": 0, "Sommeil (heures)": 1, "Strain cumulé": 1, "Poids moyen (kg)": 1, "Variation (kg)": 2},
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("La variation compare la première et la dernière pesée de chaque semaine.")

    profile = weekday_profile(merged, "Récupération (%)")
    if not profile.empty and int(profile["Observations"].sum()) >= 7:
        section_header("Profil par jour de la semaine", "Repère les creux récurrents de récupération.", "📆")
        figure_dow = px.bar(
            profile.dropna(subset=["Moyenne"]),
            x="Jour",
            y="Moyenne",
            title="Récupération moyenne par jour de la semaine",
            template=_plot_template(),
        )
        figure_dow.update_layout(yaxis_title="Récupération (%)", margin=dict(t=60, b=40))
        st.plotly_chart(figure_dow, use_container_width=True, key="whoop-weekday-profile")

    with st.expander("Jours communs (poids + WHOOP)", expanded=False):
        st.dataframe(_format_table(merged), use_container_width=True, hide_index=True)


def main() -> None:
    ensure_session_defaults()
    page_hero(
        "Objets connectés",
        "WHOOP",
        "Récupération, sommeil et charge d'entraînement importés depuis votre bracelet, "
        "puis croisés avec votre courbe de poids.",
        meta="Lecture seule — vos données de poids ne sont jamais modifiées",
    )

    credentials = _resolve_credentials()
    _consume_oauth_callback(credentials)

    token = _stored_token()
    if token is None:
        _connection_panel(credentials)
        st.divider()
        st.caption(
            "Aucun compte WHOOP connecté : les autres onglets de l'application restent "
            "pleinement fonctionnels."
        )
        return

    profile = st.session_state.get("whoop_profile", {}) or {}
    owner = " ".join(str(part) for part in (profile.get("first_name"), profile.get("last_name")) if part).strip()
    st.success(f"Compte WHOOP connecté{f' — {owner}' if owner else ''}.")
    if st.button("Déconnecter WHOOP"):
        clear_whoop_session()
        st.info("Compte WHOOP déconnecté.")
        return

    _sync_panel(credentials, token)

    daily = st.session_state.get("whoop_daily", pd.DataFrame())
    workouts = st.session_state.get("whoop_workouts", pd.DataFrame())
    if daily is None or daily.empty:
        st.divider()
        empty_state("Lancez une synchronisation pour afficher vos données WHOOP.")
        return

    merged = merge_with_weight(get_filtered_or_working_data(), daily)

    st.divider()
    tabs = st.tabs(["Vue d'ensemble", "Récupération", "Sommeil", "Effort", "Poids × WHOOP"])
    with tabs[0]:
        _overview_tab(daily, merged)
    with tabs[1]:
        _recovery_tab(daily)
    with tabs[2]:
        _sleep_tab(daily)
    with tabs[3]:
        _effort_tab(daily, workouts)
    with tabs[4]:
        _weight_tab(daily, merged)


main()
