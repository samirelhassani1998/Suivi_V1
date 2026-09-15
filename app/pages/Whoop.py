"""Onglet WHOOP : connexion OAuth2, synchronisation et croisement avec le poids."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.core.date_labels import (
    describe_freshness,
    format_clock_hour,
    format_date_range,
    format_datetime,
    format_day_month,
    format_duration_minutes,
    format_long_date,
    format_relative_day,
    format_short_date,
    format_week_label,
)
from app.core.formatting import MISSING_VALUE as MISSING_TEXT, format_fr_number
from app.core.business import FINAL_TARGET_WEIGHT_KG, TARGET_TRAJECTORY_END_DATE
from app.core.target_trajectory import compare_to_target_trajectory, required_daily_loss
from app.core.whoop_analytics import (
    KCAL_PER_KG,
    MIN_DAYS_BASELINE,
    MIN_DAYS_CORRELATION,
    MIN_DAYS_PROJECTION,
    MIN_DAYS_TRAINING_LOAD,
    MIN_NIGHTS_VITALS,
    MIN_SESSIONS_PER_SPORT,
    MIN_SMOOTHED_STREAK,
    SWC_FRACTION,
    Insight,
    analysis_availability,
    calendar_matrix,
    contrast_best_worst_days,
    coverage_report,
    daily_log,
    daily_log_table,
    daily_grid,
    energy_balance,
    generate_insights,
    hr_zone_profile,
    indexable_metrics,
    indexed_series,
    lagged_correlations,
    personal_baseline,
    physiological_watch,
    plural,
    projected_goal_date,
    recovery_drivers,
    recovery_log,
    recovery_streaks,
    recovery_zones,
    rolling_trend,
    session_log,
    sleep_architecture,
    sleep_debt_summary,
    smoothed_baseline,
    sport_recovery_impact,
    strain_tolerance,
    target_pace_feasibility,
    training_energy_share,
    strain_recovery_balance,
    training_load,
    vital_name,
    weekday_contrast,
    weekday_profile,
    weekly_rollup,
)
from app.ui.whoop_visuals import (
    RECOVERY_BANDS,
    STRAIN_BANDS,
    bedtime_chart,
    indexed_comparison_chart,
    intensity_profile_chart,
    recovery_bars_chart,
    recovery_calendar,
    recovery_gauge,
    series_chart,
    sessions_timeline_chart,
    sleep_stages_chart,
    smoothed_baseline_chart,
    sparkline,
    weekday_chart,
    zone_distribution_chart,
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
    SLEEP_COLUMNS,
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
    sport_label,
    summarise_daily,
    workouts_to_frame,
    build_authorization_url,
)
from app.ui.components import day_card, empty_state, insight_card, kpi_card, page_hero, section_header

METRIC_DECIMALS = {
    "Variation poids (kg)": 2,
    "Variation poids (kg/jour)": 3,
    "Jours depuis la pesée précédente": 0,
    "Sur (jours)": 0,
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

# Métriques sans direction souhaitable : une hausse n'est ni bonne ni mauvaise.
NEUTRAL_METRICS = {"Strain", "Calories (kcal)", "Heure de coucher"}

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
    """Réglages de synchronisation, repliés pour ne pas précéder les données."""
    profile = st.session_state.get("whoop_profile", {}) or {}
    owner = " ".join(str(part) for part in (profile.get("first_name"), profile.get("last_name")) if part).strip()
    label = f"Compte WHOOP connecté{f' — {owner}' if owner else ''} · synchronisation"
    with st.expander(label, expanded=False):
        _sync_controls(credentials, token)


def _sync_controls(credentials, token: WhoopToken) -> None:
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
        last_sync = st.session_state.get("whoop_last_sync")
        st.metric(
            "Dernière synchro",
            format_relative_day(last_sync) if last_sync is not None else "jamais",
            help=f"Horodatage : {format_datetime(last_sync)}" if last_sync is not None else "Aucune synchronisation dans cette session.",
        )
    with cols[2]:
        launch = st.button("Synchroniser maintenant", use_container_width=True, type="primary")

    st.session_state["whoop_sync_days"] = int(days)

    if st.button("Déconnecter WHOOP"):
        clear_whoop_session()
        st.info("Compte WHOOP déconnecté.")
        return

    if not launch:
        return

    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.Timedelta(days=int(days))

    try:
        fresh = ensure_fresh_token(credentials, token)
        if fresh.access_token != token.access_token:
            _store_token(fresh)
        with st.spinner("Récupération des données WHOOP…"):
            cycle_records = fetch_collection("cycle", fresh, start=start, end=end)
            # Une récupération ne porte pas de fuseau : sans celui de son cycle,
            # un score créé à 23 h 30 UTC se retrouvait daté de la veille.
            offsets = {record.get("id"): record.get("timezone_offset") for record in cycle_records}
            recovery = recoveries_to_frame(fetch_collection("recovery", fresh, start=start, end=end), offsets)
            sleep = sleeps_to_frame(fetch_collection("sleep", fresh, start=start, end=end))
            cycle = cycles_to_frame(cycle_records)
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

SPARKLINE_DAYS = 14

PERIOD_CHOICES: dict[str, int | None] = {
    "7 jours": 7,
    "30 jours": 30,
    "90 jours": 90,
    "Tout": None,
}

def _metric_value(metric: str, value: float) -> str:
    return format_fr_number(value, decimals=METRIC_DECIMALS.get(metric, 1))


def _metric_delta(metric: str, delta: float) -> str:
    """Delta lisible par Streamlit : signe ASCII (flèche et couleur) et décimale française."""
    formatted = format_fr_number(abs(delta), decimals=METRIC_DECIMALS.get(metric, 1))
    if formatted == "—":
        return formatted
    return f"{'-' if delta < 0 else '+'}{formatted}"


def _format_table(frame: pd.DataFrame, decimals: dict[str, int] | None = None) -> pd.DataFrame:
    """Rend un tableau lisible : décimales maîtrisées, dates courtes, vides explicites."""
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame()
    rules = decimals or {}
    display = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if column == "Semaine":
            # Testé avant le cas général : cette colonne est un horodatage, et la
            # branche datetime la réduisait à « 3 août » sans dire que c'est une semaine.
            display[column] = series.apply(format_week_label)
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Une colonne d'horodatages perd tout son sens réduite au seul jour ;
            # et une date de tableau sans son jour de semaine oblige à le
            # retrouver de tête, alors que c'est lui qui explique la mesure.
            formatter = format_datetime if column == "Début" else format_short_date
            display[column] = series.apply(formatter)
        elif column == "Durée (min)" or column == "Durée totale (min)":
            display[column] = series.apply(format_duration_minutes)
        elif column == "Heure de coucher":
            display[column] = series.apply(format_clock_hour)
        elif pd.api.types.is_bool_dtype(series):
            # « True » au milieu d'un tableau français se lit mal : oui / non.
            display[column] = series.map({True: "oui", False: "non"}).fillna(MISSING_TEXT)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            places = rules.get(column, METRIC_DECIMALS.get(column, 1))
            display[column] = series.apply(lambda value, places=places: format_fr_number(value, decimals=places))
        else:
            display[column] = series.fillna(MISSING_TEXT).astype(str)
    return display


def _table_view(frame: pd.DataFrame, label: str = "Voir les valeurs", decimals: dict[str, int] | None = None) -> None:
    """Contrepartie tabulaire d'un graphique.

    Toute valeur portée par une couleur ou une position doit rester lisible
    autrement : c'est la condition d'accessibilité des graphiques colorés.
    """
    if frame is None or frame.empty:
        return
    with st.expander(label, expanded=False):
        st.dataframe(_format_table(frame, decimals), use_container_width=True, hide_index=True)


def _render_chart(figure, key: str, *, fallback: str = "Métrique indisponible sur la période choisie.") -> None:
    if figure is None:
        st.info(fallback)
        return
    st.plotly_chart(figure, use_container_width=True, key=key)


def _apply_period(frame: pd.DataFrame, days: int | None, *, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Restreint une trame aux *days* derniers jours calendaires, à compter d'aujourd'hui.

    Compter depuis la dernière mesure donnerait à « 7 jours » un sens flottant :
    après une semaine sans porter le bracelet, la tranche affichée ne serait plus
    celle que le lecteur a demandée.
    """
    if frame is None or frame.empty or days is None or "Date" not in frame.columns:
        return frame if frame is not None else pd.DataFrame()
    data = frame.copy(deep=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    if data.empty:
        return data
    reference = (today or pd.Timestamp.now()).normalize()
    cutoff = reference - pd.Timedelta(days=int(days) - 1)
    return data[data["Date"] >= cutoff].reset_index(drop=True)


def _freshness_banner(daily: pd.DataFrame) -> None:
    """Dit depuis quand les données s'arrêtent, au lieu de le laisser deviner."""
    if daily is None or daily.empty or "Date" not in daily.columns:
        return
    last = pd.to_datetime(daily["Date"], errors="coerce").max()
    freshness = describe_freshness(last)
    if freshness["days"] is None:
        return
    message = f"Dernière mesure : **{freshness['label']}** — {format_long_date(last)}."
    if freshness["stale"]:
        st.warning(
            message + " Les moyennes récentes portent donc sur des jours déjà anciens — "
            "une synchronisation remettra la page à jour."
        )
    else:
        st.caption(message)


def _zone_badge(zone: str | None) -> str:
    return {"Vert": "🟢", "Jaune": "🟡", "Rouge": "🔴"}.get(str(zone), "⚪")


ZONE_ADJECTIVES = {"Vert": "verte", "Jaune": "jaune", "Rouge": "rouge"}


def _n(count: Any, singular: str, plural_form: str | None = None) -> str:
    """« 3 jours », « 1 jour » : le nombre suivi du nom accordé."""
    return f"{count} {plural(count, singular, plural_form)}"


def _zone_label(zone: Any) -> str:
    """« 🟢 Vert » : la couleur et son nom, pour ne pas reposer sur la seule couleur."""
    if zone in ("Vert", "Jaune", "Rouge"):
        return f"{_zone_badge(zone)} {zone}"
    return MISSING_TEXT


# Lignes affichées d'emblée dans une lecture par date ; le reste s'ouvre à la demande.
DATED_ROWS = 14
JOURNAL_CARDS = 21


def _dated_table(frame: pd.DataFrame, *, decimals: dict[str, int], label: str) -> None:
    """Les lignes les plus récentes visibles d'emblée, l'historique complet à portée de clic."""
    if frame is None or frame.empty:
        return
    st.dataframe(_format_table(frame.head(DATED_ROWS), decimals), use_container_width=True, hide_index=True)
    if len(frame) > DATED_ROWS:
        with st.expander(f"{label} ({len(frame)} lignes)", expanded=False):
            st.dataframe(_format_table(frame, decimals), use_container_width=True, hide_index=True)


def _render_insights(insights: list[Insight]) -> None:
    """Constats rédigés, du plus important au moins important."""
    if not insights:
        st.info(
            "Aucun constat fiable pour l'instant : quelques jours de mesures supplémentaires "
            "suffiront à en faire émerger."
        )
        return
    for insight in insights:
        insight_card(insight.title, insight.body, tone=insight.tone, icon=insight.icon)


def _unlock_progress(daily: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Ce qui est actif et ce qui reste à attendre, sous forme de progression."""
    for item in analysis_availability(daily, merged):
        if item.ready:
            st.markdown(f"✅ **{item.name}** — active sur {_n(item.available, 'jour')}")
            continue
        st.markdown(f"⏳ **{item.name}** — encore {_n(item.missing, 'jour')} de mesure")
        st.progress(min(1.0, item.available / item.required))


# ──────────────────────────────────────────────────────────────────────────────
# Onglets
# ──────────────────────────────────────────────────────────────────────────────


def _headline_panel(daily: pd.DataFrame) -> None:
    """Bandeau d'ouverture : dernier score en jauge, puis moyennes sur 7 jours."""
    zones = recovery_zones(daily)
    summary = summarise_daily(daily, days=7)
    grid = daily_grid(daily)

    gauge_column, metrics_column = st.columns([1, 2])
    with gauge_column:
        if zones["days"]:
            _render_chart(recovery_gauge(zones["latest"], zones["latest_zone"]), "whoop-gauge")
            latest_date = grid.loc[grid["Récupération (%)"].last_valid_index(), "Date"] if grid["Récupération (%)"].notna().any() else None
            when = f" — {format_relative_day(latest_date)}" if latest_date is not None else ""
            st.caption(
                f"Zone {ZONE_ADJECTIVES.get(str(zones['latest_zone']), str(zones['latest_zone']).lower())} "
                f"{_zone_badge(zones['latest_zone'])}{when}. Seuils WHOOP : rouge sous 34 %, vert dès 67 %."
            )
        else:
            empty_state("Aucun score de récupération sur la période choisie.")

    with metrics_column:
        headline = [m for m in ("HRV (ms)", "Sommeil (heures)", "Strain") if m in summary]
        if not headline:
            st.info("Pas encore assez de métriques pour un résumé.")
            return
        st.caption(
            f"Moyennes des 7 derniers jours, comparées aux 7 jours précédents. "
            f"Les micro-courbes montrent les {SPARKLINE_DAYS} derniers jours mesurés."
        )
        columns = st.columns(len(headline))
        for column, metric in zip(columns, headline):
            stats = summary[metric]
            delta = stats["delta"]
            with column:
                if pd.notna(delta):
                    st.metric(
                        metric,
                        _metric_value(metric, stats["current"]),
                        _metric_delta(metric, delta),
                        # Le strain n'est ni bon ni mauvais en soi : le colorer
                        # en rouge à la hausse en ferait une alerte permanente.
                        delta_color="normal" if metric in HIGHER_IS_BETTER else "off" if metric in NEUTRAL_METRICS else "inverse",
                    )
                else:
                    kpi_card(metric, _metric_value(metric, stats["current"]), help_text="Pas encore de période précédente pour comparer.")
                # Quatorze jours donnent une silhouette lisible ; la légende
                # ci-dessus ne parle que des moyennes sur sept jours.
                values = grid[metric].dropna().tail(SPARKLINE_DAYS) if metric in grid.columns else pd.Series(dtype=float)
                if len(values) >= 3:
                    st.plotly_chart(
                        sparkline(values.tolist(), positive=metric in HIGHER_IS_BETTER),
                        use_container_width=True,
                        key=f"whoop-spark-{metric}",
                        config={"displayModeBar": False},
                    )


def _day_entry_stats(entry) -> list[tuple[str, str, str]]:
    """Mesures du jour, dans l'ordre où on les lit : état, sommeil, effort, poids."""
    stats: list[tuple[str, str, str]] = []
    if np.isfinite(entry.recovery):
        stats.append(("récupération", format_fr_number(entry.recovery, decimals=0), "%"))
        # La zone en toutes lettres : le liseré coloré ne suffit pas à qui ne
        # distingue pas les couleurs.
        if entry.zone:
            stats.append(("zone", f"{_zone_badge(entry.zone)} {ZONE_ADJECTIVES.get(entry.zone, '')}", ""))
    if np.isfinite(entry.sleep_hours):
        stats.append(("sommeil", format_fr_number(entry.sleep_hours, decimals=1), "h"))
    if np.isfinite(entry.sleep_debt) and entry.sleep_debt > 0.1:
        stats.append(("dette", format_fr_number(entry.sleep_debt, decimals=1), "h"))
    if np.isfinite(entry.bedtime):
        stats.append(("coucher", format_clock_hour(entry.bedtime), ""))
    if np.isfinite(entry.strain):
        stats.append(("charge", format_fr_number(entry.strain, decimals=1), ""))
    elif np.isfinite(entry.strain_in_progress):
        # La journée n'est pas close : le chiffre est lisible, pas comptable.
        stats.append(("charge en cours", format_fr_number(entry.strain_in_progress, decimals=1), ""))
    if np.isfinite(entry.hrv):
        stats.append(("HRV", format_fr_number(entry.hrv, decimals=0), "ms"))
    if np.isfinite(entry.resting_hr):
        stats.append(("FC repos", format_fr_number(entry.resting_hr, decimals=0), "bpm"))
    if np.isfinite(entry.weight):
        change = (
            f" ({format_fr_number(entry.weight_change, decimals=1, sign=True)})"
            if np.isfinite(entry.weight_change)
            else ""
        )
        stats.append(("poids", f"{format_fr_number(entry.weight, decimals=1)}{change}", "kg"))
    return stats


def _day_entry_sessions(entry) -> list[tuple[str, str, str]]:
    """Séances du jour, situées par leur heure de début."""
    rows = []
    for session in entry.sessions:
        when = (
            pd.Timestamp(session.start).strftime("%H:%M")
            if session.start is not None and pd.notna(session.start)
            else "—"
        )
        details = []
        if np.isfinite(session.duration_min):
            details.append(format_duration_minutes(session.duration_min))
        if np.isfinite(session.strain):
            details.append(f"charge {format_fr_number(session.strain, decimals=1)}")
        if np.isfinite(session.calories):
            details.append(f"{format_fr_number(session.calories, decimals=0)} kcal")
        if np.isfinite(session.max_hr):
            details.append(f"FC max {format_fr_number(session.max_hr, decimals=0)}")
        rows.append((when, sport_label(session.sport), " · ".join(details)))
    return rows


def _day_by_day_tab(daily: pd.DataFrame, workouts: pd.DataFrame) -> None:
    """Lecture chronologique : ce qui s'est passé chaque jour, du plus récent au plus ancien.

    Les courbes disent comment les choses évoluent ; elles ne disent jamais ce
    qui s'est passé mardi. Ce journal comble cet écart.
    """
    weights = get_filtered_or_working_data()
    entries = daily_log(daily, workouts, weights)
    if not entries:
        empty_state("Aucune journée à afficher sur la période choisie.")
        return

    section_header(
        "Journal",
        f"{_n(len(entries), 'jour')} sur la période, du plus récent au plus ancien. "
        "La couleur du liseré reprend la zone de récupération WHOOP.",
        "📔",
    )

    measured = [entry for entry in entries if entry.has_measurement]
    show_gaps = st.toggle(
        "Afficher aussi les jours sans aucune mesure",
        value=False,
        help=f"{_n(len(entries) - len(measured), 'jour')} sans mesure sur la période.",
    )
    visible = entries if show_gaps else measured
    if not visible:
        empty_state("Aucune mesure sur la période choisie.")
        return

    def _card(entry) -> None:
        stats = _day_entry_stats(entry)
        # Au-delà d'une semaine, le badge répétait la date déjà en tête de carte.
        age = (today_reference - pd.Timestamp(entry.date).normalize()).days
        day_card(
            format_long_date(entry.date),
            format_relative_day(entry.date) if 0 <= age <= 7 else "",
            entry.zone,
            stats,
            _day_entry_sessions(entry),
            empty_message=None if stats else "Aucune mesure ce jour-là.",
        )

    today_reference = pd.Timestamp.now().normalize()
    # Trois semaines de cartes suffisent à la lecture courante ; le reste
    # s'ouvre à la demande plutôt que d'allonger la page de plusieurs écrans.
    for entry in visible[:JOURNAL_CARDS]:
        _card(entry)
    if len(visible) > JOURNAL_CARDS:
        with st.expander(f"Journées plus anciennes ({len(visible) - JOURNAL_CARDS})", expanded=False):
            for entry in visible[JOURNAL_CARDS:]:
                _card(entry)

    _table_view(
        daily_log_table(entries),
        "Voir le journal sous forme de tableau",
        {"Récupération (%)": 0, "Sommeil (heures)": 1, "Dette de sommeil (heures)": 1, "Strain": 1, "Poids (Kgs)": 1, "Séances": 0},
    )
    st.download_button(
        "Exporter le journal (CSV)",
        daily_log_table(entries).to_csv(index=False).encode("utf-8"),
        file_name="whoop_journal.csv",
        mime="text/csv",
    )


def _overview_tab(daily: pd.DataFrame, merged: pd.DataFrame, workouts: pd.DataFrame, headline: pd.DataFrame | None = None) -> None:
    # Le bandeau compare les 7 derniers jours aux 7 précédents : sur la période
    # « 7 jours », il n'avait plus de période précédente à comparer et
    # affichait trois KPI orphelins sous une légende qui promettait l'inverse.
    _headline_panel(headline if headline is not None else daily)

    section_header("Ce que disent vos données", "Constats classés par importance, chiffres à l'appui.", "🧠")
    weights = get_filtered_or_working_data()
    target_status = compare_to_target_trajectory(weights) if not weights.empty else None
    _render_insights(
        generate_insights(
            daily,
            merged,
            workouts,
            required_daily_kg=required_daily_loss(),
            target_status=target_status,
            target_weight=FINAL_TARGET_WEIGHT_KG,
            target_date=TARGET_TRAJECTORY_END_DATE,
            weight_history=weights,
        )
    )

    with st.expander("Sur quoi reposent ces chiffres ?", expanded=False):
        coverage = coverage_report(daily)
        cols = st.columns(3)
        with cols[0]:
            kpi_card("Jours mesurés", f"{coverage['days_with_data']}", help_text=f"Sur {_n(coverage['span_days'], 'jour')} de période")
        with cols[1]:
            kpi_card(
                "Jours complets",
                f"{coverage['complete_days']}",
                help_text="Jours où récupération, sommeil et charge sont tous les trois disponibles.",
            )
        with cols[2]:
            kpi_card("Jours croisés avec une pesée", f"{len(merged)}")
        st.divider()
        _unlock_progress(daily, merged)
        st.caption(
            "Un seuil n'est pas une coquetterie : sous cet effectif, une pente ou une "
            "corrélation décrit le bruit de mesure plutôt qu'une tendance."
        )

    section_header("Tendances", "Chaque indicateur sur son échelle, les jours sans mesure restant vides.", "📈")
    grid = daily_grid(daily)
    specs = [
        (["Récupération (%)"], "Récupération", "%", "Récupération (%)", RECOVERY_BANDS),
        (["Sommeil (heures)", "Besoin de sommeil (heures)"], "Sommeil obtenu et besoin estimé", "heures", "Sommeil (heures)", None),
        (["Strain"], "Charge quotidienne", "strain", "Strain", STRAIN_BANDS),
    ]
    for metrics, title, unit, trend_metric, bands in specs:
        _render_chart(
            series_chart(grid, metrics, title, unit, trend=rolling_trend(daily, trend_metric), bands=bands),
            f"whoop-trend-{trend_metric}",
        )

    _table_view(daily.sort_values("Date", ascending=False), "Toutes les données journalières")
    st.download_button(
        "Exporter les données WHOOP (CSV)",
        daily.to_csv(index=False).encode("utf-8"),
        file_name="whoop_daily.csv",
        mime="text/csv",
    )


def _vitals_panel(daily: pd.DataFrame) -> None:
    """Les cinq signes vitaux nocturnes, rapportés au repère personnel.

    Les valeurs absolues de ces grandeurs varient trop d'une personne à l'autre
    pour être interprétées seules : seul l'écart à sa propre habitude se lit.
    """
    watch = physiological_watch(daily)
    section_header(
        "Veille physiologique",
        "Fréquence cardiaque de repos, variabilité, fréquence respiratoire, température cutanée et saturation, comparées à votre habitude.",
        "🩺",
    )
    if not watch["ready"]:
        st.info(
            f"Comparaison disponible à partir de {MIN_NIGHTS_VITALS} nuits mesurées : un repère "
            "construit sur moins que cela décrirait surtout la dernière nuit elle-même."
        )
        return

    badge = {"aucun signal": "🟢", "un signal isolé": "🟡", "plusieurs signaux concordants": "🔴"}
    cols = st.columns([1, 2])
    with cols[0]:
        kpi_card(
            "Signaux inhabituels",
            f"{badge.get(watch['level'], '⚪')} {watch['count']}",
            help_text=f"Nuit du {format_long_date(watch['date'], with_weekday=False)}.",
        )
    with cols[1]:
        if watch["count"] == 0:
            insight_card(
                "Aucun signe vital hors de votre habitude",
                "Les cinq grandeurs mesurées cette nuit se situent dans votre plage usuelle.",
                tone="success",
                icon="🟢",
            )
        else:
            names = ", ".join(f"{vital_name(item['Signe vital'])} {item['Sens']}" for item in watch["flagged"])
            several = watch["count"] > 1
            insight_card(
                f"{watch['count']} {'signes vitaux' if several else 'signe vital'} hors de votre habitude",
                f"Cette nuit : {names}. "
                + (
                    "Plusieurs signes qui dévient ensemble méritent d'être signalés à un professionnel "
                    "de santé s'ils persistent."
                    if several
                    else "Un signe isolé s'explique souvent par une soirée tardive, un repas copieux ou l'alcool."
                ),
                tone="warning" if several else "info",
                icon="🩺",
            )

    st.dataframe(
        _format_table(
            watch["table"].drop(columns=["Inhabituel"]),
            {"Dernière nuit": 1, "Repère habituel": 1, "Écart": 1},
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "L'écart est exprimé en unités de dispersion robuste par rapport à la médiane de vos trente "
        "derniers jours, et non à une norme de population. Un écart est signalé au-delà de 2,5 unités : "
        "sur des séries sans anomalie, ce seuil se déclenche environ une nuit sur treize, contre une "
        "sur cinq à 2,0. **Ces mesures ne constituent pas un diagnostic** : un bracelet ne remplace "
        "pas un examen, et l'interprétation revient à un professionnel de santé."
    )


def _contrast_display(table: pd.DataFrame) -> pd.DataFrame:
    """Tableau des facteurs, l'heure de coucher rendue en horloge et non en heures signées."""
    display = _format_table(table, {"Meilleurs jours": 2, "Pires jours": 2, "Écart": 2, "Écart normalisé": 2})
    mask = table["Facteur"] == "Heure de coucher"
    if mask.any():
        row = table[mask].iloc[0]
        index = table.index[mask][0]
        display.loc[index, "Meilleurs jours"] = format_clock_hour(row["Meilleurs jours"])
        display.loc[index, "Pires jours"] = format_clock_hour(row["Pires jours"])
        display.loc[index, "Écart"] = f"{int(round(abs(float(row['Écart'])) * 60))} min {'plus tard' if float(row['Écart']) > 0 else 'plus tôt'}"
    return display


def _smoothed_panel(
    daily: pd.DataFrame,
    metric: str,
    label: str,
    *,
    unit: str,
    log_scale: bool,
    higher_is_better: bool,
) -> None:
    """Dernière nuit, moyenne de sept jours et plage habituelle, puis la courbe."""
    trend = smoothed_baseline(daily, metric, log_scale=log_scale)
    baseline = personal_baseline(daily, metric)
    cols = st.columns(3)
    with cols[0]:
        if baseline["ready"]:
            st.metric(
                f"{label} — dernière nuit",
                f"{format_fr_number(baseline['latest'], decimals=0)} {unit}",
                f"{'+' if baseline['deviation_pct'] > 0 else '-'}{format_fr_number(abs(baseline['deviation_pct']), decimals=0)} %",
                delta_color="normal" if higher_is_better else "inverse",
                help=f"Écart à la médiane de vos 30 derniers jours ({format_fr_number(baseline['baseline'], decimals=0)} {unit}).",
            )
        else:
            kpi_card(f"{label} — dernière nuit", "—", help_text=f"Repère disponible dès {MIN_DAYS_BASELINE} jours (actuellement {baseline['days']}).")
    with cols[1]:
        if trend["ready"]:
            kpi_card(
                "Moyenne des 7 derniers jours",
                f"{format_fr_number(trend['latest'], decimals=0)} {unit}",
                help_text="Moyenne mobile sur sept jours calendaires, au moins quatre nuits mesurées.",
            )
        else:
            kpi_card("Moyenne des 7 derniers jours", "—", help_text="Disponible après une trentaine de jours de mesure.")
    with cols[2]:
        if trend["ready"]:
            status = trend["status"]
            concerning = (status == "bas") if higher_is_better else (status == "haut")
            badge = "🟢" if status == "dans la plage" else ("🔴" if concerning else "🟢")
            since = f" depuis {trend['streak']} j" if status != "dans la plage" else ""
            kpi_card(
                "Plage habituelle",
                f"{format_fr_number(trend['low'], decimals=0)}–{format_fr_number(trend['high'], decimals=0)} {unit}",
                delta=f"{badge} {status}{since}",
                help_text=f"Autour de {format_fr_number(trend['baseline'], decimals=0)} {unit} : vos trente jours précédant la semaine en cours.",
            )
        else:
            kpi_card("Plage habituelle", "—")
    if trend["ready"]:
        _render_chart(smoothed_baseline_chart(trend["series"], metric, label, unit), f"whoop-smoothed-{metric}")
    else:
        _render_chart(series_chart(daily_grid(daily), [metric], label, unit), f"whoop-smoothed-{metric}")


def _recovery_tab(daily: pd.DataFrame) -> None:
    zones = recovery_zones(daily)
    grid = daily_grid(daily)

    section_header(
        "Récupération, jour par jour",
        "Chaque matin noté, colorié de sa zone WHOOP, du plus récent au plus ancien — "
        "avec la nuit et la charge de la veille qui l'ont précédé.",
        "🗓️",
    )
    if zones["days"] == 0:
        st.info("Aucun score de récupération sur la période choisie.")
    else:
        _render_chart(
            recovery_bars_chart(grid, trend=rolling_trend(daily, "Récupération (%)")),
            "whoop-recovery-bars",
        )
        log = recovery_log(daily)
        log["Zone"] = log["Zone"].map(_zone_label)
        _dated_table(
            log,
            decimals={"Récupération (%)": 0, "HRV (ms)": 0, "FC repos (bpm)": 0, "Sommeil (heures)": 1, "Strain de la veille": 1, "Strain du jour": 1},
            label="Voir toutes les journées notées",
        )
        calibrating = int(daily["Calibration"].fillna(False).astype(bool).sum()) if "Calibration" in daily.columns else 0
        st.caption(
            "Le score est calculé au réveil : la colonne « Sommeil » est la nuit qui le précède, "
            "« Strain de la veille » l'effort de la journée d'avant. « Strain du jour » vient après "
            "le score et ne l'explique pas — il dit ce que vous avez fait de cette récupération."
            + (
                f" {_n(calibrating, 'journée')} en calibration WHOOP sur la période : ces premiers scores "
                "sont provisoires, le bracelet apprend encore votre repère."
                if calibrating
                else ""
            )
        )

    section_header("Zones de récupération", "Répartition des journées sur la période, et calendrier semaine par semaine.", "🚦")
    if zones["days"] == 0:
        st.info("Aucun score de récupération sur la période choisie.")
    else:
        cols = st.columns([3, 2])
        with cols[0]:
            _render_chart(recovery_calendar(calendar_matrix(daily, "Récupération (%)")), "whoop-calendar")
        with cols[1]:
            _render_chart(zone_distribution_chart(zones["counts"]), "whoop-zones")
        _table_view(zones["counts"], "Voir la répartition chiffrée", {"Jours": 0, "Part (%)": 1})

    streaks = recovery_streaks(daily)
    if streaks["ready"]:
        streak_cols = st.columns(3)
        badge = {"Vert": "🟢", "Jaune": "🟡", "Rouge": "🔴"}.get(str(streaks["current_zone"]), "⚪")
        with streak_cols[0]:
            kpi_card(
                "Série en cours",
                f"{badge} {streaks['current_length']} j",
                help_text=f"Journées calendaires consécutives en zone {ZONE_ADJECTIVES.get(str(streaks['current_zone']), '—')}.",
            )
        with streak_cols[1]:
            kpi_card("Plus longue série verte", f"{streaks['longest_green']} j")
        with streak_cols[2]:
            kpi_card("Plus longue série rouge", f"{streaks['longest_red']} j")
        st.caption(
            "Une moyenne hebdomadaire lisse les séries ; leur durée dit si un état s'installe "
            "ou s'il s'agit d'une journée isolée."
        )

    section_header(
        "Variabilité cardiaque et fréquence au repos",
        "La moyenne de sept jours face à votre plage habituelle : une nuit isolée ne fait pas une tendance.",
        "🫀",
    )
    _smoothed_panel(daily, "HRV (ms)", "Variabilité cardiaque", unit="ms", log_scale=True, higher_is_better=True)
    _smoothed_panel(daily, "FC repos (bpm)", "Fréquence au repos", unit="bpm", log_scale=False, higher_is_better=False)
    st.caption(
        "La plage habituelle est centrée sur la moyenne de vos trente jours précédant la semaine en cours, "
        f"large de {format_fr_number(SWC_FRACTION, decimals=2)} écart-type de part et d'autre (la HRV est traitée "
        "sur son logarithme, sa distribution étant asymétrique). Suivre la moyenne mobile de sept jours plutôt "
        "que la valeur d'une nuit est la pratique recommandée pour la variabilité cardiaque "
        "([Buchheit, Front. Physiol. 2014](https://doi.org/10.3389/fphys.2014.00073)). Un signal est retenu "
        f"après {MIN_SMOOTHED_STREAK} jours consécutifs hors plage : sur des séries sans aucun changement, cela "
        "se produit dans 7 % des cas."
    )

    # Deux cadres « métrique indisponible » côte à côte n'apprennent rien :
    # ces courbes ne s'affichent que si le bracelet a renvoyé la mesure.
    body_metrics = [
        (metric, title, unit)
        for metric, title, unit in (("Température peau (°C)", "Température cutanée", "°C"), ("SpO2 (%)", "Saturation en oxygène", "%"))
        if metric in grid.columns and grid[metric].notna().any()
    ]
    if body_metrics:
        body_cols = st.columns(len(body_metrics))
        for column, (metric, title, unit) in zip(body_cols, body_metrics):
            with column:
                _render_chart(series_chart(grid, [metric], title, unit), f"whoop-recovery-{metric}")

    _vitals_panel(daily)

    section_header("Moteurs de la récupération", "Ce qui fait bouger votre score, chiffré plutôt que supposé.", "🔬")
    drivers = recovery_drivers(daily)
    if not drivers["ready"]:
        st.info(f"Analyse disponible à partir de {drivers['required_days']} jours complets (actuellement {drivers['days']}).")
    else:
        driver_cols = st.columns(len(drivers["coefficients"]) + 1)
        threshold = 0.05 / max(1, len(drivers["coefficients"]))
        for column, (name, coefficient) in zip(driver_cols, drivers["coefficients"].items()):
            unit = "h de sommeil" if "Sommeil" in name else "point de strain"
            p_value = drivers.get("p_values", {}).get(name, float("nan"))
            # Un coefficient qui ne se distingue pas de zéro ne peut pas être
            # affiché au même rang qu'un effet établi : la vue d'ensemble le
            # tait, le panneau doit le dire.
            established = np.isfinite(p_value) and p_value <= threshold
            with column:
                kpi_card(
                    name,
                    f"{format_fr_number(coefficient, decimals=1, sign=True)} pt",
                    delta=None if established else "non établi",
                    help_text=(
                        f"Par {unit} supplémentaire. "
                        + (
                            f"Se distingue de zéro (p = {format_fr_number(p_value, decimals=3)})."
                            if established
                            else (
                                f"Ne se distingue pas de zéro sur vos données (p = {format_fr_number(p_value, decimals=2)}) : "
                                "à lire comme un ordre de grandeur, pas comme un effet."
                                if np.isfinite(p_value)
                                else "Test indisponible."
                            )
                        )
                    ),
                )
        with driver_cols[-1]:
            kpi_card(
                "Part des variations expliquée",
                f"{format_fr_number(max(0.0, drivers['r_squared']) * 100, decimals=0)} %",
                help_text="Valeur ajustée au nombre de variables : un modèle sans lien réel retombe vers zéro.",
            )
        if drivers["r_squared"] < 0.15:
            st.warning(
                "Le modèle explique une part faible des variations : sur vos données actuelles, "
                "sommeil et charge de la veille ne suffisent pas à prédire la récupération."
            )

    _render_chart(weekday_chart(weekday_profile(daily, "Récupération (%)"), "Récupération (%)"), "whoop-recovery-weekday", fallback="Profil hebdomadaire disponible après quelques semaines.")
    weekday = weekday_contrast(daily, "Récupération (%)")
    if weekday["ready"]:
        if weekday["significant"]:
            st.caption(
                f"Le {weekday['day']} est votre jour le plus bas ({format_fr_number(weekday['mean'], decimals=0)} % contre "
                f"{format_fr_number(weekday['others'], decimals=0)} % les autres jours, sur {weekday['observations']} "
                "occurrences), et l'écart résiste à un test qui tient compte des sept jours candidats."
            )
        else:
            st.caption(
                f"Le {weekday['day']} est votre jour le plus bas en moyenne, mais l'écart avec les autres jours "
                "ne se distingue pas du hasard : le plus bas de sept jours est toujours sous la moyenne."
            )

    section_header(
        "Vos bons jours contre vos mauvais",
        "Ce qui précédait le tiers des jours où votre récupération était la meilleure.",
        "🔍",
    )
    contrast = contrast_best_worst_days(daily)
    if not contrast["ready"]:
        st.info(f"Comparaison disponible à partir de {contrast['required_days']} jours notés (actuellement {contrast['days']}).")
        st.progress(min(1.0, contrast["days"] / contrast["required_days"]))
    else:
        st.caption(
            f"{contrast['best_days']} meilleurs jours (récupération ≥ "
            f"{format_fr_number(contrast['best_threshold'], decimals=0)} %) comparés aux "
            f"{contrast['worst_days']} pires (≤ {format_fr_number(contrast['worst_threshold'], decimals=0)} %)."
        )
        st.dataframe(
            _contrast_display(contrast["table"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Le facteur en tête est celui qui sépare le plus vos bons et vos mauvais jours. "
            "L'écart normalisé rapporte l'écart à la dispersion : il permet de comparer des heures "
            "de sommeil à des points de strain. « Écart établi » indique que l'écart résiste à un test "
            "statistique tenant compte du nombre de facteurs comparés : sans cela, le premier de la liste "
            "n'est que le plus grand de sept tirages."
        )
        st.caption(
            "⏱️ Votre score de récupération est calculé au réveil, à partir de la nuit écoulée. "
            "Les facteurs comparés ici lui sont donc tous antérieurs — d'où le strain **de la veille** "
            "et non celui du jour même, qui n'a pas encore eu lieu quand le score est établi."
        )


def _sleep_tab(daily: pd.DataFrame) -> None:
    grid = daily_grid(daily)
    debt = sleep_debt_summary(daily, days=7)

    section_header("Dette de sommeil", "Écart entre le besoin estimé par WHOOP et le sommeil obtenu.", "🛌")
    if debt["nights"] == 0:
        st.info("Aucune nuit exploitable sur la période choisie.")
    else:
        cols = st.columns(4)
        with cols[0]:
            nights = debt["nights"]
            st.metric(
                f"Dette cumulée ({_n(nights, 'nuit')})",
                f"{format_fr_number(debt['cumulative_debt'], decimals=1, sign=True)} h",
                help=(
                    "Nuits mesurées au cours des 7 derniers jours. Positif : vous dormez moins que votre besoin. "
                    + (
                        "Cumul calculé contre votre besoin de base : le besoin affiché par WHOOP chaque soir "
                        "contient déjà le rattrapage des nuits précédentes, l'additionner compterait la même dette plusieurs fois."
                        if debt.get("cumulative_basis") == "besoin de base"
                        else "Somme des écarts nuit par nuit au besoin estimé par WHOOP."
                    )
                ),
            )
        with cols[1]:
            kpi_card("Dette moyenne / nuit", f"{format_fr_number(debt['mean_debt'], decimals=1, sign=True)} h")
        with cols[2]:
            kpi_card("Sommeil moyen", f"{format_fr_number(debt['mean_sleep'], decimals=1)} h")
        with cols[3]:
            kpi_card("Besoin moyen", f"{format_fr_number(debt['mean_need'], decimals=1)} h")

    _render_chart(series_chart(grid, ["Sommeil (heures)", "Besoin de sommeil (heures)"], "Sommeil obtenu face au besoin", "Heures"), "whoop-sleep-need")
    _render_chart(
        series_chart(grid, ["Dette de sommeil (heures)"], "Dette de sommeil nuit par nuit", "Heures", trend=rolling_trend(daily, "Dette de sommeil (heures)")),
        "whoop-sleep-debt",
    )

    quality_metrics = ["Performance sommeil (%)", "Efficacité sommeil (%)", "Régularité sommeil (%)"]
    # Une métrique constamment nulle n'est pas une information : WHOOP la laisse
    # à zéro tant que l'historique est trop court pour la calculer.
    informative = [
        metric
        for metric in quality_metrics
        if metric in daily.columns and daily[metric].notna().any() and float(daily[metric].fillna(0).abs().sum()) > 0
    ]
    _render_chart(series_chart(grid, informative or quality_metrics, "Qualité du sommeil", "%"), "whoop-sleep-quality")
    ignored = [metric for metric in quality_metrics if metric not in informative]
    if informative and ignored:
        st.caption("Métrique(s) masquée(s) car encore nulle(s) chez WHOOP : " + ", ".join(ignored) + ".")

    _render_chart(sleep_stages_chart(grid), "whoop-sleep-stages", fallback="Stades de sommeil indisponibles sur la période choisie.")
    if "Sommeil léger (heures)" in grid.columns and grid["Sommeil léger (heures)"].notna().any():
        st.caption(
            "La hauteur de chaque barre est le temps passé au lit : profond et REM en bas, léger au-dessus, "
            "éveil en haut. Une nuit courte réduit tous les stades à la fois — d'où la lecture en parts, ci-dessous."
        )

    architecture = sleep_architecture(daily)
    section_header(
        "Architecture du sommeil",
        "La part de chaque stade dans la nuit, et non ses heures : une nuit courte réduit mécaniquement les deux.",
        "🌙",
    )
    if not architecture["ready"]:
        st.info(
            f"Comparaison disponible à partir de {architecture['required_nights']} nuits mesurées "
            f"(actuellement {architecture['nights']})."
        )
    else:
        st.dataframe(
            _format_table(architecture["table"], {"Votre part (%)": 1}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Les plages indiquées sont des repères de population adulte couramment cités, pas des "
            "objectifs personnels : une nuit hors plage n'a pas de signification isolée, et "
            f"allonger la nuit — {format_fr_number(architecture['mean_sleep'], decimals=1)} h en "
            "moyenne actuellement — augmente généralement les deux stades en valeur absolue."
        )

    if "Heure de coucher" in daily.columns and daily["Heure de coucher"].notna().sum() >= 3:
        section_header("Régularité du coucher", "Heure d'endormissement sur une échelle continue autour de minuit.", "🕰️")
        spread = float(daily["Heure de coucher"].std())
        _render_chart(bedtime_chart(grid), "whoop-sleep-bedtime")
        st.caption(
            f"Dispersion des couchers : {format_fr_number(spread, decimals=1)} h d'écart-type. "
            "Une dispersion faible traduit un rythme régulier, que WHOOP relie à la qualité du sommeil."
        )

    _table_view(
        daily[["Date"] + [column for column in SLEEP_COLUMNS if column in daily.columns and column not in ("Date", "Sieste")]]
        .sort_values("Date", ascending=False),
        "Voir les nuits",
    )


def _sessions_panel(daily: pd.DataFrame, workouts: pd.DataFrame) -> None:
    """Chaque séance à sa date, la récupération du lendemain en face."""
    section_header(
        "Vos séances, par date",
        "Chaque séance avec la récupération du lendemain matin en face : l'effort et ce qu'il a laissé, "
        "sur la même ligne, du plus récent au plus ancien.",
        "🥊",
    )
    sessions = session_log(daily, workouts)
    if sessions.empty:
        st.info("Aucune séance enregistrée sur la période choisie.")
        return

    table = sessions.copy()
    table["Sport"] = table["Sport"].map(sport_label)
    table["Début"] = [
        pd.Timestamp(value).strftime("%H:%M") if pd.notna(value) else MISSING_TEXT for value in table["Début"]
    ]
    table["Zone du lendemain"] = table["Zone du lendemain"].map(_zone_label)
    # Une colonne entièrement vide n'apporte qu'une rangée de tirets.
    for column in ("Distance (km)", "Dénivelé (m)", "Zones 4–5 (min)"):
        if not table[column].notna().any():
            table = table.drop(columns=[column])
    partial = int((table["Part enregistrée (%)"] < 80).sum()) if table["Part enregistrée (%)"].notna().any() else 0
    if not table["Part enregistrée (%)"].notna().any():
        table = table.drop(columns=["Part enregistrée (%)"])
    _dated_table(
        table,
        decimals={
            "Durée (min)": 0,
            "Strain séance": 1,
            "FC moyenne (bpm)": 0,
            "FC max (bpm)": 0,
            "Calories séance (kcal)": 0,
            "Distance (km)": 2,
            "Dénivelé (m)": 0,
            "Zones 4–5 (min)": 0,
            "Part enregistrée (%)": 0,
            "Récupération du lendemain (%)": 0,
        },
        label="Voir toutes les séances",
    )
    note = (
        "« Récupération du lendemain » est le score calculé au réveil suivant : c'est le premier moment "
        "où l'effort de la séance devient mesurable. « Zones 4–5 » compte les minutes au-dessus de 80 % "
        "de votre réserve de fréquence cardiaque."
    )
    if partial:
        note += (
            f" {partial} séance{'s' if partial > 1 else ''} enregistrée{'s' if partial > 1 else ''} à moins de 80 % : "
            "strain et calories y sont sous-estimés."
        )
    st.caption(note)
    _render_chart(sessions_timeline_chart(sessions), "whoop-sessions-timeline")

    profile = hr_zone_profile(workouts)
    if profile.empty:
        return
    section_header(
        "Répartition de l'intensité",
        "Part du temps en zones faciles, modérées et dures, sport par sport, additionnée sur toutes les séances de la période.",
        "🎚️",
    )
    _render_chart(intensity_profile_chart(profile), "whoop-intensity")
    _table_view(
        profile.assign(Sport=[sport_label(sport) if sport != "Toutes séances" else sport for sport in profile["Sport"]]),
        "Voir la répartition chiffrée",
        {"Séances": 0, "Minutes en zones": 0, "Facile (zones 0–2)": 0, "Modéré (zone 3)": 0, "Dur (zones 4–5)": 0},
    )
    st.caption(
        "Les zones WHOOP sont exprimées en pourcentage de la réserve de fréquence cardiaque (zone 3 : 70–80 %, "
        "zones 4–5 : 80 % et plus). Chez les sportifs d'endurance, la répartition la plus souvent observée "
        "est d'environ 80 % de temps facile pour 20 % de temps dur — le modèle « polarisé » "
        "([Seiler, Int J Sports Physiol Perform 2010](https://doi.org/10.1123/ijspp.5.3.276)). C'est un repère "
        "issu d'un autre contexte, pas une cible : la boxe et la musculation ne s'y plient pas de la même façon."
    )


def _effort_tab(daily: pd.DataFrame, workouts: pd.DataFrame) -> None:
    grid = daily_grid(daily)
    load = training_load(daily)

    _sessions_panel(daily, workouts)

    section_header(
        "Charge d'entraînement",
        "Charge des 7 derniers jours rapportée à celle des 21 jours qui les précèdent : une dynamique que WHOOP n'affiche pas.",
        "🏋️",
    )
    if load["status"] == "couverture insuffisante":
        st.info(
            f"Bracelet porté {load['acute_days_measured']} jour(s) sur les 7 derniers : trop peu pour "
            "qualifier votre charge récente. Deux séances isolées ne décrivent pas une semaine d'entraînement."
        )
    elif not np.isfinite(load["ratio"]):
        st.info(f"Indicateur disponible à partir de {MIN_DAYS_TRAINING_LOAD} jours de mesure (actuellement {load['days']}).")
    else:
        cols = st.columns(3)
        with cols[0]:
            kpi_card(
                "Charge aigüe (7 j)",
                format_fr_number(load["acute"], decimals=1),
                help_text=(
                    f"Du {format_short_date(load['acute_start'])} au {format_short_date(load['acute_end'])}, "
                    f"moyenne sur {load['acute_days_measured']} jour(s) réellement mesuré(s)."
                ),
            )
        with cols[1]:
            kpi_card(
                "Charge chronique (21 j précédents)",
                format_fr_number(load["chronic"], decimals=1),
                help_text=(
                    f"Du {format_short_date(load['chronic_start'])} au {format_short_date(load['chronic_end'])}, "
                    f"moyenne sur {load['chronic_days_measured']} jour(s) réellement mesuré(s)."
                ),
            )
        with cols[2]:
            kpi_card("Rapport aigu / chronique", format_fr_number(load["ratio"], decimals=2), help_text=load["status"])
        # Entre 1,3 et 1,5 le statut annonce une montée soutenue : la carte ne
        # peut pas la peindre en vert au motif qu'elle n'est pas brutale.
        tone = {"montée en charge brutale": "warning", "montée en charge soutenue": "info", "charge en retrait": "info"}.get(load["status"], "success")
        insight_card(
            str(load["status"]).capitalize(),
            "Entre 0,8 et 1,3, la progression est généralement considérée comme soutenable. "
            "Au-delà de 1,5, l'augmentation dépasse nettement vos habitudes récentes. La semaine en cours "
            "n'entre pas dans la moyenne de référence : sinon le rapport serait borné par construction "
            "([Windt & Gabbett, Br J Sports Med 2019](https://bjsm.bmj.com/content/53/16/988)). Le strain étant une "
            "échelle logarithmique, ce rapport compare des niveaux d'effort, pas des volumes.",
            tone=tone,
            icon="🏋️",
        )

    _render_chart(
        series_chart(grid, ["Strain"], "Charge quotidienne", "Strain", trend=rolling_trend(daily, "Strain"), bands=STRAIN_BANDS),
        "whoop-effort-strain",
    )
    _render_chart(series_chart(grid, ["Calories (kcal)"], "Dépense énergétique quotidienne", "kcal", trend=rolling_trend(daily, "Calories (kcal)")), "whoop-effort-calories")

    share = training_energy_share(daily, workouts)
    if share["ready"]:
        section_header(
            "Ce que pèsent vraiment vos séances",
            "Part de la dépense quotidienne attribuable aux entraînements — les calories s'additionnent, contrairement au strain.",
            "🍽️",
        )
        share_cols = st.columns(3)
        with share_cols[0]:
            kpi_card("Dépense moyenne", f"{format_fr_number(share['mean_daily_burn'], decimals=0)} kcal/j")
        with share_cols[1]:
            kpi_card("Dont séances", f"{format_fr_number(share['mean_session_burn'], decimals=0)} kcal/j")
        with share_cols[2]:
            kpi_card(
                "Part de l'entraînement",
                f"{format_fr_number(share['share_pct'], decimals=0)} %",
                help_text=f"Sur {share['days']} jour(s), dont {share['session_days']} avec au moins une séance.",
            )
        st.caption(
            "Une part faible n'est pas un reproche : chez la plupart des gens, le métabolisme de "
            "repos et l'activité ordinaire portent l'essentiel de la dépense. Le savoir évite "
            "d'attendre d'une séance qu'elle compense un écart alimentaire."
        )

    balance = strain_recovery_balance(daily)
    if not balance.empty:
        alerts = balance[balance["Type"] == "alerte"]
        opportunities = balance[balance["Type"] == "occasion"]
        if not alerts.empty:
            section_header("Jours à surveiller", "Charge élevée alors que la récupération était basse.", "⚠️")
            st.dataframe(
                _format_table(alerts.drop(columns=["Type"]), {"Récupération (%)": 0, "Strain": 1}),
                use_container_width=True,
                hide_index=True,
            )
        if not opportunities.empty:
            # Une bonne journée rangée parmi les alertes brouillait la lecture.
            section_header("Occasions manquées", "Récupération élevée alors que la charge est restée faible.", "🌤️")
            st.dataframe(
                _format_table(opportunities.drop(columns=["Type"]), {"Récupération (%)": 0, "Strain": 1}),
                use_container_width=True,
                hide_index=True,
            )

    tolerance = strain_tolerance(daily)
    if tolerance["declines"]:
        heading = f"Au-dessus de {format_fr_number(tolerance['high_band_floor'], decimals=1)} de strain, le lendemain se paie"
    elif tolerance["significant"]:
        # Écart établi, mais en sens inverse : vos journées chargées sont
        # suivies d'une MEILLEURE récupération.
        heading = "Vos journées chargées sont suivies d'une meilleure récupération"
    else:
        heading = "Charge de la veille et récupération du lendemain"
    section_header(
        heading,
        "Journées rangées par tiers de charge, jugées sur la récupération du lendemain matin.",
        "🧗",
    )
    if not tolerance["ready"]:
        if tolerance["reason"] == "charge trop uniforme":
            # Réclamer « plus de jours » alors que l'effectif est atteint donne
            # une consigne qu'aucune journée de plus ne peut satisfaire.
            st.info(
                f"Vos {tolerance['pairs']} journées mesurées se ressemblent trop en charge pour être "
                "rangées en trois tiers distincts. Cette comparaison s'ouvrira lorsque vos journées "
                "seront plus contrastées — quelques séances intenses et quelques journées calmes."
            )
        else:
            st.info(
                f"Disponible à partir de {tolerance['required_pairs']} journées suivies d'un lendemain noté "
                f"(actuellement {tolerance['pairs']})."
            )
            st.progress(min(1.0, tolerance["pairs"] / max(1, tolerance["required_pairs"])))
    else:
        cols = st.columns(3)
        with cols[0]:
            kpi_card("Après une journée calme", f"{format_fr_number(tolerance['calm_recovery'], decimals=0)} %")
        with cols[1]:
            kpi_card("Après une journée chargée", f"{format_fr_number(tolerance['heavy_recovery'], decimals=0)} %")
        with cols[2]:
            kpi_card(
                "Écart",
                f"{format_fr_number(tolerance['gap'], decimals=0)} point{'s' if abs(tolerance['gap']) >= 2 else ''}",
                help_text=(
                    f"Intervalle de confiance à 95 % : de {format_fr_number(tolerance['gap_low'], decimals=0)} "
                    f"à {format_fr_number(tolerance['gap_high'], decimals=0)} points."
                    if np.isfinite(tolerance["gap_low"]) and np.isfinite(tolerance["gap_high"])
                    else "Intervalle non calculable sur cet effectif."
                ),
            )
        st.dataframe(
            _format_table(
                tolerance["table"],
                {"Jours": 0, "Strain moyen": 1, "Récupération du lendemain (%)": 0, "Journées rouges (%)": 0},
            ),
            use_container_width=True,
            hide_index=True,
        )
        if tolerance["declines"]:
            st.caption(
                f"Calculé sur {tolerance['pairs']} paires jour chargé → lendemain. L'écart entre vos journées "
                "calmes et vos journées chargées résiste à un test statistique : il ne s'explique pas par le "
                "seul hasard d'échantillonnage."
            )
        elif tolerance["significant"]:
            # Un écart inverse peut être parfaitement établi : le déclarer
            # « indistinguable du hasard » contredirait sa propre p-value.
            st.caption(
                f"Calculé sur {tolerance['pairs']} paires jour chargé → lendemain. L'écart est statistiquement "
                "établi, mais dans l'autre sens : vos journées les plus chargées sont suivies d'une meilleure "
                "récupération. Cela se produit notamment lorsqu'on s'entraîne davantage les jours où l'on se "
                "sent déjà en forme — la charge suit alors la récupération plutôt que l'inverse."
            )
        elif tolerance["inference"] == "indisponible":
            # Conclure « indistinguable du hasard » sur un test qui n'a pas pu
            # tourner ferait dire à l'absence de calcul ce qu'un calcul n'a pas dit.
            st.caption(
                f"Calculé sur {tolerance['pairs']} paires jour chargé → lendemain. Vos mesures ne permettent "
                "pas de tester l'écart statistiquement sur cette période : le tableau est affiché tel quel, "
                "sans conclusion sur sa solidité."
            )
        else:
            st.caption(
                f"Calculé sur {tolerance['pairs']} paires jour chargé → lendemain. L'écart entre les tiers ne se "
                "distingue pas du hasard : à ce stade, vos journées chargées ne se paient pas visiblement le "
                "lendemain. Le tableau reste affiché pour ce qu'il montre, sans en tirer de seuil."
            )
        if tolerance["declines"]:
          st.caption(
            f"⚠️ La valeur de {format_fr_number(tolerance['high_band_floor'], decimals=1)} est le bord du tiers "
            "le plus chargé de **vos** journées, pas un point de rupture physiologique : elle se déplacera si "
            "vous vous mettez à vous entraîner davantage, à réponse identique. WHOOP vous propose un strain "
            "cible établi sur sa population de référence ; ce tableau est établi sur vous. Il décrit une "
            "association : une journée chargée suivie d'une nuit courte pèse deux fois."
        )

    if workouts is None or workouts.empty:
        return
    section_header("Bilan par sport", "Nombre de séances, durée, strain moyen et calories, sport par sport.", "🏃")
    detailed = workouts.copy()
    detailed["Sport"] = detailed["Sport"].map(sport_label)
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
        _format_table(by_sport, {"Séances": 0, "Strain moyen": 1, "Calories (kcal)": 0}),
        use_container_width=True,
        hide_index=True,
    )

    impact = sport_recovery_impact(daily, workouts)
    if not impact.empty:
        section_header(
            "Ce que chaque sport coûte au lendemain",
            "Récupération du jour suivant, sport par sport : le score du jour même précède la séance.",
            "🌡️",
        )
        # Le bilan par sport compte toutes les séances ; ici seules celles
        # suivies d'un matin noté comptent, et le libellé doit le dire.
        impact_display = impact.rename(columns={"Séances": "Lendemains notés"})
        impact_display["Sport"] = impact_display["Sport"].map(sport_label)
        st.dataframe(
            _format_table(impact_display, {"Lendemains notés": 0, "Récupération du lendemain (%)": 0, "Écart à votre moyenne": 1}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Comparaison faite à partir de {MIN_SESSIONS_PER_SPORT} séances par sport. "
            "Un écart négatif indique une récupération plus basse que votre moyenne le lendemain ; "
            "« Écart établi » signifie qu'il résiste à un test tenant compte du nombre de sports comparés."
        )


def _projection_panel(merged: pd.DataFrame) -> None:
    """Où mène le rythme actuel, confronté à l'échéance visée.

    Être en avance aujourd'hui ne dit rien de la date d'arrivée : c'est la
    confrontation des deux qui rend l'information actionnable.
    """
    projection = projected_goal_date(merged, target_weight=FINAL_TARGET_WEIGHT_KG)
    section_header(
        "Au rythme actuel",
        "Projection du rythme mesuré jusqu'à la cible, confrontée à l'échéance.",
        "🧭",
    )
    if not projection["ready"]:
        reasons = {
            "historique trop court": f"Projection disponible à partir de {MIN_DAYS_PROJECTION} pesées croisées.",
            "tendance trop irrégulière": "Les pesées sont trop dispersées autour de la tendance pour projeter une date.",
            "le poids ne va pas vers la cible": "Au rythme actuel, la cible ne serait pas atteinte : le poids ne descend pas.",
            "échéance au-delà de dix ans": "Le rythme actuel place l'arrivée au-delà de dix ans.",
        }
        st.info(reasons.get(projection["reason"], "Projection indisponible sur la période choisie."))
        if np.isfinite(projection["slope_kg_per_week"]):
            st.caption(f"Rythme mesuré : {format_fr_number(projection['slope_kg_per_week'], decimals=2, sign=True)} kg/semaine.")
        return

    deadline = pd.Timestamp(TARGET_TRAJECTORY_END_DATE)
    projected = pd.Timestamp(projection["date"])
    gap_days = int((projected - deadline.normalize()).days)

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Rythme mesuré", f"{format_fr_number(projection['slope_kg_per_week'], decimals=2, sign=True)} kg/sem")
    with cols[1]:
        kpi_card("Poids actuel", f"{format_fr_number(projection['current_weight'], decimals=1)} kg")
    with cols[2]:
        kpi_card(
            f"Arrivée à {format_fr_number(FINAL_TARGET_WEIGHT_KG, decimals=0)} kg",
            # Une date de l'année prochaine sans son année se lit comme une date
            # passée : l'année s'affiche dès qu'elle diffère de celle du jour.
            format_day_month(projected) if projected.year == pd.Timestamp.now().year else format_long_date(projected, with_weekday=False),
            help_text=f"Soit dans {_n(int(round(projection['days'])), 'jour')}, si le rythme se maintenait.",
        )
    with cols[3]:
        st.metric(
            "Face à l'échéance",
            "en retard" if gap_days > 0 else "en avance" if gap_days < 0 else "à l'heure",
            delta=f"{gap_days:+d} j",
            delta_color="inverse",
            help=f"Échéance visée : {format_long_date(deadline, with_weekday=False)}.",
        )

    if gap_days > 7:
        insight_card(
            "Le rythme actuel dépasse l'échéance",
            f"La cible serait atteinte {gap_days} {'jours' if gap_days > 1 else 'jour'} après la date visée. "
            "Allonger l'échéance ou accentuer le rythme sont les deux issues ; la première ne coûte rien à la santé.",
            tone="warning",
            icon="🧭",
        )
    elif gap_days < -7:
        insight_card(
            "Le rythme actuel devance l'échéance",
            f"La cible serait atteinte {abs(gap_days)} {'jours' if abs(gap_days) > 1 else 'jour'} avant la date visée.",
            tone="success",
            icon="🧭",
        )
    st.caption(
        "Extrapolation linéaire du rythme mesuré : elle répond à « et si cela continuait ainsi », "
        "ce qui n'est pas une prédiction. Un palier, un changement d'alimentation ou une variation "
        "d'hydratation la déplacent."
    )


def _energy_balance_panel(merged: pd.DataFrame, weight_history: pd.DataFrame) -> None:
    """Le croisement que ni WHOOP ni une balance ne peuvent produire seuls."""
    section_header("Bilan énergétique estimé", "La dépense vient de WHOOP, le déficit de votre courbe de poids : l'apport s'en déduit.", "🔥")
    balance = energy_balance(merged, weight_history=weight_history)
    if not balance["ready"]:
        missing = max(0, balance["required_days"] - balance["days"])
        st.info(f"Estimation disponible à partir de {balance['required_days']} jours croisés : il en manque {missing}.")
        st.progress(min(1.0, balance["days"] / balance["required_days"]))
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
        margin = balance.get("intake_margin", float("nan"))
        # Publier ce chiffre au kcal près suggérerait une précision que la pente
        # d'une courbe de poids bruitée ne possède pas.
        margin_text = (
            f"± {format_fr_number(margin, decimals=0)} kcal/j" if np.isfinite(margin) else "marge indisponible"
        )
        kpi_card(
            "Apport estimé",
            f"{format_fr_number(balance['estimated_intake'], decimals=0)} kcal/j",
            help_text=f"Intervalle à 95 % : {margin_text}.",
        )

    _target_pace_panel(merged)

    margin = balance.get("intake_margin", float("nan"))
    precision = (
        f" L'incertitude sur la pente donne une marge de ± {format_fr_number(margin, decimals=0)} kcal/jour."
        if np.isfinite(margin)
        else ""
    )
    st.caption(
        "Méthode : la pente du poids est convertie en énergie sur la base de "
        f"{format_fr_number(KCAL_PER_KG, decimals=0)} kcal par kilogramme, puis ajoutée à la dépense "
        f"mesurée par WHOOP.{precision} Sur quelques semaines, une part de la variation de poids est de "
        "l'eau et du glycogène, pour lesquels cette équivalence ne vaut pas : l'estimation situe un "
        "ordre de grandeur, pas une valeur exacte."
    )


def _target_pace_panel(merged: pd.DataFrame) -> None:
    """Ce que l'objectif de poids suppose de manger, au vu de la dépense mesurée."""
    feasibility = target_pace_feasibility(merged, required_daily_kg=required_daily_loss())
    if not feasibility["ready"]:
        return

    st.divider()
    section_header(
        "Votre objectif, traduit en calories",
        "L'objectif est exprimé en kilogrammes, la dépense en kilocalories : les rapprocher dit ce que le rythme visé suppose.",
        "🎯",
    )
    cols = st.columns(4)
    with cols[0]:
        kpi_card("Perte visée", f"{format_fr_number(feasibility['required_weekly_kg'], decimals=2)} kg/sem")
    with cols[1]:
        # Le bilan voisin affiche la tendance signée (−0,42 kg/sem) ; ici la
        # perte est nommée comme telle, pour ne pas montrer le même chiffre
        # avec deux signes.
        kpi_card("Perte actuelle", f"{format_fr_number(feasibility['current_daily_kg'] * 7, decimals=2)} kg/sem")
    with cols[2]:
        kpi_card("Déficit requis", f"{format_fr_number(feasibility['required_deficit'], decimals=0)} kcal/j")
    with cols[3]:
        kpi_card(
            "Apport que cela suppose",
            f"{format_fr_number(feasibility['implied_intake'], decimals=0)} kcal/j",
            help_text=f"Dépense mesurée ({format_fr_number(feasibility['mean_burn'], decimals=0)} kcal/j) moins le déficit requis.",
        )

    if feasibility["verdict"] in ("très exigeant", "sous les repères usuels"):
        insight_card(
            f"Un rythme {feasibility['verdict']}",
            f"L'apport correspondant, environ {format_fr_number(feasibility['implied_intake'], decimals=0)} kcal/jour, "
            "se situe sous les repères couramment cités pour un adulte. Ce n'est pas un avis médical : "
            "c'est l'arithmétique de votre objectif confrontée à votre dépense mesurée. Un tel niveau "
            "se discute avec un professionnel de santé, ou l'échéance peut être allongée.",
            tone="warning",
            icon="⚠️",
        )
    st.caption(
        "Calcul : rythme visé × "
        f"{format_fr_number(KCAL_PER_KG, decimals=0)} kcal/kg = déficit requis ; apport = dépense mesurée − déficit. "
        "Estimation sensible aux mêmes réserves que le bilan énergétique."
    )


def _weight_tab(daily: pd.DataFrame, merged: pd.DataFrame) -> None:
    weights = get_filtered_or_working_data()
    if merged.empty:
        empty_state("Aucun jour commun entre vos pesées et la période WHOOP choisie.")
        return

    st.caption(
        f"{len(merged)} jour(s) couverts à la fois par une pesée et par une mesure WHOOP "
        f"({format_date_range(merged['Date'].min(), merged['Date'].max())})."
    )

    _projection_panel(merged)

    _energy_balance_panel(merged, weights)

    section_header("Poids et métrique WHOOP", "Les deux séries ramenées à une base 100 commune, lisibles sur un axe unique.", "⚖️")
    # Une base 100 sur une grandeur qui passe par zéro ou devient négative
    # inverse le sens de la courbe : ces métriques sont écartées du sélecteur.
    metrics = indexable_metrics(merged, available_metrics(merged))
    if not metrics:
        st.info("Aucune métrique WHOOP comparable au poids sur les jours communs.")
        return

    default_index = metrics.index("Récupération (%)") if "Récupération (%)" in metrics else 0
    metric = st.selectbox("Métrique WHOOP à comparer au poids", metrics, index=default_index)

    indexed = indexed_series(merged, ["Poids (Kgs)", metric])
    _render_chart(indexed_comparison_chart(indexed, ["Poids (Kgs)", metric], f"Poids et {metric}, base 100"), "whoop-weight-indexed")
    st.caption(
        "Chaque série vaut 100 à son premier jour mesuré : les deux courbes deviennent comparables "
        "sans caler arbitrairement deux échelles verticales l'une sur l'autre."
    )
    _table_view(merged[["Date", "Poids (Kgs)", metric]], "Voir les valeurs réelles (non indexées)")

    section_header(
        "Corrélations décalées",
        "Chaque métrique est confrontée à votre rythme de perte en kg/jour, testé à 0, 1 et 2 jours de décalage.",
        "🧮",
    )
    correlations = lagged_correlations(merged)
    if correlations.empty:
        st.info(f"Corrélations calculées à partir de {MIN_DAYS_CORRELATION} jours communs (actuellement {len(merged)}).")
        st.progress(min(1.0, len(merged) / MIN_DAYS_CORRELATION))
    else:
        st.dataframe(
            _format_table(correlations, {"Décalage (jours)": 0, "Corrélation": 3, "Observations": 0}),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "⚠️ Une corrélation n'est pas une causalité. La cible est le rythme quotidien (kg/jour) et non "
        "l'écart brut entre deux pesées : sans cette normalisation, dix jours d'écart pèseraient dix fois "
        "plus lourd qu'un jour. Sur de courtes séries, ces valeurs restent indicatives et sensibles au "
        "bruit de mesure (hydratation, horaire de pesée). Plusieurs métriques et décalages étant testés, "
        "la plus forte corrélation affichée est aussi la plus susceptible d'être un artefact."
    )

    weekly = weekly_rollup(merged)
    if not weekly.empty:
        section_header("Synthèse hebdomadaire", "Une ligne par semaine : récupération, sommeil, charge et variation de poids.", "🗓️")
        st.dataframe(
            _format_table(
                weekly,
                {
                    "Jours": 0,
                    "Récupération (%)": 0,
                    "Sommeil (heures)": 1,
                    "Strain moyen": 1,
                    "Poids moyen (kg)": 1,
                    "Variation (kg)": 2,
                    "Sur (jours)": 0,
                },
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "La variation compare la première et la dernière pesée de la semaine ; la colonne "
            "« Sur (jours) » indique la durée réellement couverte, qui n'est pas toujours sept jours."
        )

    _table_view(merged, "Voir tous les jours communs")


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
        st.caption("Aucun compte WHOOP connecté : les autres onglets restent pleinement fonctionnels.")
        return

    _sync_panel(credentials, token)

    full_daily = st.session_state.get("whoop_daily", pd.DataFrame())
    full_workouts = st.session_state.get("whoop_workouts", pd.DataFrame())
    if full_daily is None or full_daily.empty:
        st.divider()
        empty_state("Lancez une synchronisation pour afficher vos données WHOOP.")
        return

    st.divider()
    _freshness_banner(full_daily)
    # Un filtre unique au-dessus de tout ce qu'il conditionne : chaque onglet
    # s'aligne sur la même tranche, plutôt qu'un réglage par graphique.
    period_label = st.radio(
        "Période analysée",
        list(PERIOD_CHOICES),
        index=len(PERIOD_CHOICES) - 1,
        horizontal=True,
        help="S'applique à tous les onglets ci-dessous.",
    )
    days = PERIOD_CHOICES[period_label]
    today = pd.Timestamp.now().normalize()
    daily = _apply_period(full_daily, days, today=today)
    workouts = _apply_period(full_workouts, days, today=today)
    if daily.empty:
        empty_state("Aucune mesure WHOOP sur cette période.")
        return

    merged = merge_with_weight(get_filtered_or_working_data(), daily)

    tabs = st.tabs(["Vue d'ensemble", "Jour par jour", "Poids × WHOOP", "Récupération", "Sommeil", "Effort"])
    with tabs[0]:
        _overview_tab(daily, merged, workouts, headline=_apply_period(full_daily, 14, today=today))
    with tabs[1]:
        _day_by_day_tab(daily, workouts)
    with tabs[2]:
        _weight_tab(daily, merged)
    with tabs[3]:
        _recovery_tab(daily)
    with tabs[4]:
        _sleep_tab(daily)
    with tabs[5]:
        _effort_tab(daily, workouts)


main()
