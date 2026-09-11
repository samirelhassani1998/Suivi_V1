"""Onglet WHOOP : connexion OAuth2, synchronisation et croisement avec le poids."""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.formatting import format_fr_date, format_fr_number
from app.core.session_state import (
    DEFAULT_WHOOP_SYNC_DAYS,
    clear_whoop_session,
    ensure_session_defaults,
    get_filtered_or_working_data,
)
from app.core.whoop import (
    DEFAULT_REDIRECT_URI,
    WhoopError,
    WhoopToken,
    available_metrics,
    build_daily_frame,
    correlation_table,
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
from app.ui.components import empty_state, kpi_card, page_hero, section_header

METRIC_DECIMALS = {
    "Récupération (%)": 0,
    "HRV (ms)": 0,
    "FC repos (bpm)": 0,
    "Température peau (°C)": 1,
    "SpO2 (%)": 1,
    "Sommeil (heures)": 1,
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


def _resolve_credentials():
    overrides = dict(st.session_state.get("whoop_manual_credentials", {}) or {})
    return credentials_from_sources(
        _secrets_mapping(),
        os.environ,
        overrides,
        default_redirect_uri=str(overrides.get("redirect_uri") or DEFAULT_REDIRECT_URI),
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
    """Traite le retour de redirection WHOOP (``?code=...&state=...``)."""
    try:
        params = st.query_params
        code = params.get("code")
        state = params.get("state")
    except Exception:
        return
    if not code:
        return

    expected_state = st.session_state.get("whoop_oauth_state")
    if expected_state and state and state != expected_state:
        st.error("État OAuth inattendu : relancez la connexion WHOOP.")
        _clear_oauth_params()
        return

    try:
        token = exchange_code_for_token(credentials, str(code))
    except WhoopError as exc:
        st.error(f"Connexion WHOOP impossible : {exc}")
        _clear_oauth_params()
        return
    except Exception:
        st.error("Connexion WHOOP impossible : réponse inattendue du service.")
        _clear_oauth_params()
        return

    _store_token(token)
    st.session_state["whoop_oauth_state"] = None
    _clear_oauth_params()
    st.success("Compte WHOOP connecté.")


def _clear_oauth_params() -> None:
    try:
        for key in ("code", "state", "scope"):
            if key in st.query_params:
                del st.query_params[key]
    except Exception:
        pass


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
        st.caption(f"Redirection configurée : `{credentials.redirect_uri}`")
    with cols[1]:
        if st.button("Réinitialiser la connexion", use_container_width=True):
            clear_whoop_session()
            st.info("Session WHOOP réinitialisée.")


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


def _metric_value(metric: str, value: float) -> str:
    return format_fr_number(value, decimals=METRIC_DECIMALS.get(metric, 1))


def _metric_delta(metric: str, delta: float) -> str:
    """Delta lisible par Streamlit : signe ASCII (flèche et couleur) et décimale française."""
    formatted = format_fr_number(abs(delta), decimals=METRIC_DECIMALS.get(metric, 1))
    if formatted == "—":
        return formatted
    return f"{'-' if delta < 0 else '+'}{formatted}"


def _overview_tab(daily: pd.DataFrame) -> None:
    summary = summarise_daily(daily, days=7)
    headline = [
        metric
        for metric in ("Récupération (%)", "HRV (ms)", "Sommeil (heures)", "Strain")
        if metric in summary
    ]
    if not headline:
        empty_state("Aucune métrique WHOOP exploitable sur la période importée.")
        return

    st.caption("Moyennes des 7 derniers jours disponibles, comparées aux 7 jours précédents.")
    columns = st.columns(len(headline))
    for column, metric in zip(columns, headline):
        stats = summary[metric]
        delta = stats["delta"]
        if pd.notna(delta):
            direction = "normal" if metric in HIGHER_IS_BETTER else "inverse"
            with column:
                st.metric(metric, _metric_value(metric, stats["current"]), _metric_delta(metric, delta), delta_color=direction)
        else:
            with column:
                kpi_card(metric, _metric_value(metric, stats["current"]))

    trend_metrics = [metric for metric in ("Récupération (%)", "Sommeil (heures)", "Strain") if metric in daily.columns]
    if trend_metrics:
        long_frame = daily.melt(id_vars="Date", value_vars=trend_metrics, var_name="Métrique", value_name="Valeur").dropna(subset=["Valeur"])
        if not long_frame.empty:
            fig = px.line(
                long_frame,
                x="Date",
                y="Valeur",
                color="Métrique",
                markers=True,
                title="Évolution des principaux indicateurs WHOOP",
                template=_plot_template(),
            )
            fig.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="Valeur")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Données journalières WHOOP", expanded=False):
        st.dataframe(daily, use_container_width=True, hide_index=True)
        st.download_button(
            "Exporter les données WHOOP (CSV)",
            daily.to_csv(index=False).encode("utf-8"),
            file_name="whoop_daily.csv",
            mime="text/csv",
        )


def _series_chart(daily: pd.DataFrame, metrics: list[str], title: str, y_title: str) -> None:
    usable = [metric for metric in metrics if metric in daily.columns and daily[metric].notna().any()]
    if not usable:
        st.info("Métriques indisponibles sur la période importée.")
        return
    figure = go.Figure()
    for metric in usable:
        subset = daily[["Date", metric]].dropna()
        figure.add_scatter(
            x=subset["Date"],
            y=subset[metric],
            mode="lines+markers",
            name=metric,
            hovertemplate="Date : %{x|%d/%m/%Y}<br>" + metric + " : %{y:.2f}<extra></extra>",
        )
    figure.update_layout(title=title, xaxis_title="Date", yaxis_title=y_title, hovermode="x unified", template=_plot_template())
    st.plotly_chart(figure, use_container_width=True)


def _recovery_tab(daily: pd.DataFrame) -> None:
    _series_chart(daily, ["Récupération (%)"], "Score de récupération quotidien", "Récupération (%)")
    _series_chart(daily, ["HRV (ms)", "FC repos (bpm)"], "Variabilité cardiaque et fréquence au repos", "Valeur")
    _series_chart(daily, ["Température peau (°C)", "SpO2 (%)"], "Température cutanée et saturation en oxygène", "Valeur")


def _sleep_tab(daily: pd.DataFrame) -> None:
    _series_chart(daily, ["Sommeil (heures)"], "Durée de sommeil par nuit", "Heures")
    _series_chart(
        daily,
        ["Performance sommeil (%)", "Efficacité sommeil (%)", "Régularité sommeil (%)"],
        "Qualité du sommeil",
        "Pourcentage",
    )
    stages = [metric for metric in ("Sommeil profond (heures)", "Sommeil REM (heures)") if metric in daily.columns and daily[metric].notna().any()]
    if stages:
        stacked = daily.melt(id_vars="Date", value_vars=stages, var_name="Stade", value_name="Heures").dropna(subset=["Heures"])
        fig = px.bar(stacked, x="Date", y="Heures", color="Stade", title="Répartition des stades de sommeil", template=_plot_template())
        fig.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="Heures")
        st.plotly_chart(fig, use_container_width=True)


def _effort_tab(daily: pd.DataFrame, workouts: pd.DataFrame) -> None:
    _series_chart(daily, ["Strain"], "Charge quotidienne (strain)", "Strain")
    _series_chart(daily, ["Calories (kcal)"], "Dépense énergétique quotidienne", "kcal")

    if workouts is None or workouts.empty:
        st.info("Aucune séance enregistrée sur la période importée.")
        return

    section_header("Séances", "Détail des entraînements enregistrés par le bracelet.", "🏃")
    by_sport = (
        workouts.groupby("Sport", as_index=False)
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
    st.dataframe(by_sport, use_container_width=True, hide_index=True)
    st.dataframe(workouts.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)


def _weight_tab(daily: pd.DataFrame) -> None:
    weights = get_filtered_or_working_data()
    if weights.empty:
        empty_state("Aucune mesure de poids chargée : le croisement nécessite les deux sources.")
        return

    merged = merge_with_weight(weights, daily)
    if merged.empty:
        st.info("Aucun jour commun entre vos pesées et la période WHOOP importée.")
        return

    st.caption(
        f"{len(merged)} jour(s) couverts à la fois par une pesée et par une mesure WHOOP "
        f"({format_fr_date(merged['Date'].min())} → {format_fr_date(merged['Date'].max())})."
    )

    metrics = available_metrics(merged)
    if not metrics:
        st.info("Aucune métrique WHOOP exploitable sur les jours communs.")
        return

    default_index = metrics.index("Récupération (%)") if "Récupération (%)" in metrics else 0
    metric = st.selectbox("Métrique WHOOP à comparer au poids", metrics, index=default_index)

    figure = go.Figure()
    figure.add_scatter(
        x=merged["Date"],
        y=merged["Poids (Kgs)"],
        mode="lines+markers",
        name="Poids (kg)",
        hovertemplate="Date : %{x|%d/%m/%Y}<br>Poids : %{y:.2f} kg<extra></extra>",
    )
    subset = merged[["Date", metric]].dropna()
    figure.add_scatter(
        x=subset["Date"],
        y=subset[metric],
        mode="lines+markers",
        name=metric,
        yaxis="y2",
        hovertemplate="Date : %{x|%d/%m/%Y}<br>" + metric + " : %{y:.2f}<extra></extra>",
    )
    figure.update_layout(
        title=f"Poids et {metric}",
        xaxis_title="Date",
        yaxis=dict(title="Poids (kg)"),
        yaxis2=dict(title=metric, overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        template=_plot_template(),
    )
    st.plotly_chart(figure, use_container_width=True)

    section_header(
        "Corrélations",
        "Association linéaire entre la variation de poids d'une pesée à la suivante et les métriques WHOOP du jour.",
        "🧮",
    )
    correlations = correlation_table(merged)
    if correlations.empty:
        st.info("Pas encore assez de jours communs pour estimer des corrélations fiables.")
    else:
        st.dataframe(correlations, use_container_width=True, hide_index=True)
    st.caption(
        "⚠️ Une corrélation n'est pas une causalité : sur de courtes séries, ces valeurs sont "
        "surtout indicatives et sensibles au bruit de mesure (hydratation, horaire de pesée)."
    )

    with st.expander("Jours communs (poids + WHOOP)", expanded=False):
        st.dataframe(merged, use_container_width=True, hide_index=True)


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

    st.divider()
    tabs = st.tabs(["Vue d'ensemble", "Récupération", "Sommeil", "Effort", "Poids × WHOOP"])
    with tabs[0]:
        _overview_tab(daily)
    with tabs[1]:
        _recovery_tab(daily)
    with tabs[2]:
        _sleep_tab(daily)
    with tabs[3]:
        _effort_tab(daily, workouts)
    with tabs[4]:
        _weight_tab(daily)


main()
