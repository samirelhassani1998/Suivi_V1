from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.analytics import (
    analyze_effort_history,
    compute_trend_ema,
    detect_current_effort,
    discipline_score,
    generate_action_summary,
    generate_insights_text,
    multi_rolling_averages,
    next_milestone,
    pace_comparison,
    progression_score,
    streak_analysis,
    weight_acceleration,
    weight_velocity,
    weight_volatility,
)
from app.core.business import TARGET_TRAJECTORY_TOTAL_DURATION_DAYS, STAGNATION_MIN_MEASUREMENTS
from app.core.data import data_quality_report
from app.core.formatting import format_fr_date, format_fr_kg, format_fr_number, format_fr_unit
from app.core.insights import detect_plateau
from app.core.session_state import (
    DEFAULT_ZOOM_TARGET_END_DATE,
    DEFAULT_ZOOM_TARGET_START_DATE,
    ensure_session_defaults,
    get_filtered_or_working_data,
)
from app.core.target_trajectory import (
    DEFAULT_FINAL_TARGET_WEIGHT,
    DEFAULT_TARGET_TRAJECTORY_END_DATE,
    DEFAULT_TARGET_TRAJECTORY_START_DATE,
    DEFAULT_TARGET_TRAJECTORY_START_WEIGHT,
    TargetTrajectoryConfig,
    build_target_trajectory,
    compare_to_target_trajectory,
    required_weekly_loss,
)
from app.core.trend import (
    LONG_RATE_WINDOW_DAYS,
    RATE_WINDOW_DAYS,
    TREND_WINDOW_DAYS,
    bmi_category,
    noise_level,
    rate_of_change,
    reading_vs_trend,
    trend_weight,
    weight_for_bmi,
)
from app.core.weight_summary import moving_average_by_days, summarize_weight_journey
from app.core.targets import get_target_weights
from app.ui.charts import (
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
    histogram_figure,
)
from app.ui.components import (
    alert_banner,
    confidence_badge,
    empty_state,
    help_box,
    insight_card,
    kpi_card,
    page_hero,
    progress_panel,
    section_header,
)


TARGET_TRAJECTORY_CHART_KEY = "dashboard-target-trajectory-to-80kg"
TARGET_TRAJECTORY_ZOOM_CHART_KEY = "dashboard-target-trajectory-zoom"
SECONDARY_TARGET_COLORS = (MEASURE_COLOR, TARGET_COLOR, TREND_COLOR, "#7c3aed", STATUS_CRITICAL)


def _df() -> pd.DataFrame:
    return get_filtered_or_working_data()


# ──────────────────────────────────────────────────────────────────────────────
# Mise en forme
# ──────────────────────────────────────────────────────────────────────────────


def _format_delta(value: float | None, suffix: str = " kg") -> str:
    """Delta typographique (signe « − ») pour le texte courant."""
    return format_fr_unit(value, suffix.strip(), decimals=2, sign=True)


def _format_value(value: float | None, suffix: str = " kg") -> str:
    return format_fr_unit(value, suffix.strip(), decimals=2)


def _metric_delta(value: float | None, unit: str = "kg", decimals: int = 2) -> str | None:
    """Delta pour ``st.metric`` : signe ASCII, seul reconnu pour la flèche et la couleur.

    Le signe « − » typographique n'est pas lu comme négatif par Streamlit :
    une perte de poids s'affichait jusqu'ici avec une flèche montante verte.
    """
    if value is None or not np.isfinite(float(value)):
        return None
    magnitude = format_fr_number(abs(float(value)), decimals=decimals)
    sign = "-" if float(value) < 0 else "+" if float(value) > 0 else ""
    return f"{sign}{magnitude} {unit}".strip()


def _format_fr_number(value, decimals: int = 1, *, sign: bool = False, trim_zeros: bool = True) -> str:
    return format_fr_number(value, decimals=decimals, sign=sign, trim_zeros=trim_zeros)


def _format_fr_kg(value, decimals: int = 1, *, sign: bool = False, trim_zeros: bool = True) -> str:
    return format_fr_kg(value, decimals=decimals, sign=sign, trim_zeros=trim_zeros)


def _rate_text(rate: dict, *, decimals: int = 2) -> str:
    """« −0,45 kg/sem » ou un tiret quand la pente n'est pas disponible."""
    if not rate.get("ready"):
        return "—"
    return f"{format_fr_number(rate['slope_kg_week'], decimals=decimals, sign=True)} kg/sem"


def _rate_interval_text(rate: dict, *, decimals: int = 2) -> str:
    if not rate.get("ready"):
        return "—"
    return (
        f"{format_fr_number(rate['ci_low'], decimals=decimals, sign=True)} à "
        f"{format_fr_number(rate['ci_high'], decimals=decimals, sign=True)} kg/sem"
    )


def _rate_sentence(rate: dict) -> str:
    """Phrase complète sur le rythme, avec son intervalle et son verdict."""
    if not rate.get("ready"):
        reason = rate.get("reason") or "recul insuffisant"
        return f"Rythme sur {rate.get('window_days', RATE_WINDOW_DAYS)} jours indisponible ({reason})."
    verdict = {
        "baisse": "une baisse établie",
        "hausse": "une hausse établie",
        "stable": "une pente que le hasard suffit à produire",
    }.get(rate["direction"], "une pente indéterminée")
    return (
        f"Sur {rate['n']} pesées en {rate['span_days']} jours, le rythme est de {_rate_text(rate)} "
        f"(IC 95 % : {_rate_interval_text(rate)}) : {verdict}."
    )


def _trajectory_gap_label(trajectory_status: dict) -> str:
    gap = float(trajectory_status.get("gap_kg", 0.0))
    status = trajectory_status.get("status")
    gap_text = _format_fr_kg(abs(gap), decimals=1)
    if status == "en avance":
        return f"{gap_text} en avance"
    if status == "aligné":
        return "Aligné"
    return f"{gap_text} en retard"


def _trajectory_position_sentence(trajectory_status: dict) -> str:
    status = trajectory_status.get("status")
    if status == "aligné":
        return "Vous êtes actuellement aligné avec la trajectoire cible."
    return f"Vous êtes actuellement {_trajectory_gap_label(trajectory_status)} sur la trajectoire cible."


def _metric_help_for_period(period) -> str:
    if period.value is None:
        return period.reason or "Données insuffisantes pour cette période."
    date = period.reference_date.strftime("%d/%m/%Y") if period.reference_date is not None else "n/a"
    return f"Comparaison avec la première mesure disponible depuis le {date} ({period.measurements} mesure(s))."


def _targets_caption(targets: tuple[float, ...]) -> str:
    goals = " · ".join(f"Objectif {idx}: {format_fr_kg(target)}" for idx, target in enumerate(targets, start=1))
    return f"🎯 Objectifs affichés : {goals}."


# ──────────────────────────────────────────────────────────────────────────────
# Lecture de tendance partagée par toute la page
# ──────────────────────────────────────────────────────────────────────────────


def _trend_context(df: pd.DataFrame) -> dict:
    """Calcule une seule fois le poids de tendance, le bruit et les rythmes."""
    frame = trend_weight(df, TREND_WINDOW_DAYS)
    noise = noise_level(df, TREND_WINDOW_DAYS)
    reading = reading_vs_trend(df, TREND_WINDOW_DAYS)
    rate_short = rate_of_change(df, RATE_WINDOW_DAYS)
    rate_long = rate_of_change(df, LONG_RATE_WINDOW_DAYS)
    trend_now = float(frame["Tendance"].iloc[-1]) if not frame.empty else float("nan")
    trend_week_delta = float("nan")
    if not frame.empty:
        last_date = frame["Date"].iloc[-1]
        earlier = frame[frame["Date"] <= last_date - pd.Timedelta(days=7)]
        if not earlier.empty:
            trend_week_delta = trend_now - float(earlier["Tendance"].iloc[-1])
    return {
        "frame": frame,
        "noise": noise,
        "reading": reading,
        "rate_short": rate_short,
        "rate_long": rate_long,
        "trend_now": trend_now,
        "trend_week_delta": trend_week_delta,
    }


def _quick_reading_sentence(context: dict, summary: dict, target_weight: float) -> tuple[str, str]:
    reading = context["reading"]
    rate = context["rate_short"]
    parts: list[str] = []
    if reading.get("ready"):
        parts.append(
            f"Pesée du jour : {format_fr_kg(reading['reading'], decimals=1)}, poids de tendance : "
            f"{format_fr_kg(reading['trend'], decimals=1)}. L'écart de {format_fr_kg(reading['deviation'], decimals=1, sign=True)} "
            f"est {reading['verdict']} (± {format_fr_kg(reading['band'], decimals=1)})."
        )
    elif np.isfinite(context["trend_now"]):
        parts.append(f"Poids de tendance : {format_fr_kg(context['trend_now'], decimals=1)}.")
    parts.append(_rate_sentence(rate))
    gap = summary.get("target_gap")
    if gap is not None:
        if gap > 0:
            parts.append(f"Il reste {format_fr_kg(gap, decimals=1)} avant l'objectif principal ({format_fr_kg(target_weight, decimals=1)}).")
        else:
            parts.append("L'objectif principal est atteint.")
    tone = {"baisse": "success", "hausse": "warning"}.get(rate.get("direction"), "info")
    return " ".join(parts), tone


def _render_daily_overview(summary: dict, target_weight: float, trajectory_status: dict | None, context: dict) -> None:
    section_header(
        "Vue rapide",
        "Le poids du jour, le poids de tendance débarrassé du bruit quotidien, le rythme et sa fiabilité.",
        "⚡",
    )

    delta_30 = summary["delta_30"]
    rate = context["rate_short"]
    cols = st.columns(5)
    with cols[0]:
        st.metric(
            "Poids actuel",
            _format_value(summary["current"]),
            _metric_delta(summary["previous_delta"]),
            delta_color="inverse",
            help="Dernière pesée, comparée à la pesée précédente. Une baisse s'affiche en vert.",
        )
    with cols[1]:
        trend_now = context["trend_now"]
        st.metric(
            "Poids de tendance",
            _format_value(trend_now) if np.isfinite(trend_now) else "—",
            _metric_delta(context["trend_week_delta"]) if np.isfinite(context["trend_week_delta"]) else None,
            delta_color="inverse",
            help=(
                f"Régression locale robuste (LOWESS) sur une fenêtre de {TREND_WINDOW_DAYS} jours : "
                "elle suit le poids réel sans être tirée par une pesée isolée. Le delta compare à la tendance sept jours plus tôt."
            ),
        )
    with cols[2]:
        kpi_card(
            f"Rythme {RATE_WINDOW_DAYS} jours",
            _rate_text(rate),
            help_text=(
                f"Pente par moindres carrés sur les {RATE_WINDOW_DAYS} derniers jours calendaires. "
                + (
                    f"Intervalle de confiance à 95 % : {_rate_interval_text(rate)} · p = {format_fr_number(rate['p_value'], decimals=3)}."
                    if rate.get("ready")
                    else f"Indisponible : {rate.get('reason')}."
                )
            ),
        )
        if rate.get("ready"):
            st.caption(("🟢 " if rate["direction"] == "baisse" else "🟠 " if rate["direction"] == "hausse" else "⚪ ") + f"{rate['direction'].capitalize()} · IC {_rate_interval_text(rate)}")
    with cols[3]:
        st.metric(
            "Variation 30 jours",
            _format_delta(delta_30.value) if delta_30.value is not None else "—",
            None,
            help=_metric_help_for_period(delta_30),
        )
    with cols[4]:
        if trajectory_status and trajectory_status.get("available"):
            kpi_card(
                "Écart trajectoire",
                _trajectory_gap_label(trajectory_status),
                help_text=(
                    f"Poids attendu au {trajectory_status['current_date'].strftime('%d/%m/%Y')} "
                    f"dans la trajectoire cible : {_format_fr_kg(trajectory_status['scheduled_weight'])}."
                ),
            )
        else:
            kpi_card("Écart objectif", _format_delta(summary["target_gap"]), help_text=f"Écart entre le poids actuel et l'objectif principal ({target_weight:.1f} kg).")

    sentence, tone = _quick_reading_sentence(context, summary, target_weight)
    insight_card("Lecture rapide", sentence, tone=tone, icon="🔎")

    left, right = st.columns(2)
    with left:
        reliability = summary.get("reliability", {})
        if reliability:
            level = reliability.get("level", "faible")
            reliability_tone = "success" if level == "élevée" else "info" if level == "moyenne" else "warning"
            insight_card(
                "Fiabilité des tendances",
                f"{reliability.get('score', 0)}/100 — confiance {level}. {reliability.get('explanation', '')}",
                tone=reliability_tone,
                icon="🧪",
            )
    with right:
        if trajectory_status and trajectory_status.get("available"):
            insight_card(
                "Trajectoire cible",
                (
                    f"Départ : {_format_fr_kg(trajectory_status['start_weight'])} le {trajectory_status['start_date'].strftime('%d/%m/%Y')}. "
                    f"Objectif : {_format_fr_kg(trajectory_status['final_target_weight'])} le {trajectory_status['end_date'].strftime('%d/%m/%Y')}. "
                    f"Durée : {trajectory_status['total_duration_days']} jours. "
                    f"Rythme moyen requis : {_format_fr_kg(trajectory_status['required_weekly_loss'], decimals=2)} par semaine. "
                    f"Poids cible théorique au {trajectory_status['current_date'].strftime('%d/%m/%Y')} : {_format_fr_kg(trajectory_status['scheduled_weight'])}. "
                    f"Écart : {_format_fr_kg(trajectory_status['gap_kg'], sign=True)} — statut : {trajectory_status['status']}. "
                    f"{_trajectory_position_sentence(trajectory_status)}"
                ),
                tone="success" if trajectory_status.get("status") in {"en avance", "aligné"} else "warning",
                icon="🎯",
            )
        else:
            message = (trajectory_status or {}).get("message") if trajectory_status else None
            insight_card(
                "Trajectoire cible",
                (message or "Aucune trajectoire cible disponible pour la dernière mesure.")
                + f" La trajectoire de référence va de {_format_fr_kg(DEFAULT_TARGET_TRAJECTORY_START_WEIGHT)} le "
                f"{DEFAULT_TARGET_TRAJECTORY_START_DATE.strftime('%d/%m/%Y')} à {_format_fr_kg(DEFAULT_FINAL_TARGET_WEIGHT)} le "
                f"{DEFAULT_TARGET_TRAJECTORY_END_DATE.strftime('%d/%m/%Y')}.",
                tone="info",
                icon="🎯",
            )


# ──────────────────────────────────────────────────────────────────────────────
# Indicateurs avancés
# ──────────────────────────────────────────────────────────────────────────────


def _render_advanced_kpis(
    df: pd.DataFrame,
    effort_df: pd.DataFrame,
    has_effort_period: bool,
    effort_days: int,
    effort_measurements: int,
    current: float,
    target_weight: float,
    targets: tuple[float, ...],
    height_m: float,
    is_startup: bool,
    last_date: pd.Timestamp,
    context: dict,
    trajectory_status: dict | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    analysis_df = effort_df if has_effort_period else df
    quality = data_quality_report(analysis_df)
    vel = weight_velocity(analysis_df, windows=(7, 14, 30))
    disc = discipline_score(analysis_df, window_days=min(effort_days, 30) if effort_days > 0 else 30)
    prog = progression_score(analysis_df, target_weight)
    streaks = streak_analysis(analysis_df)
    rate_long = context["rate_long"]

    imc = current / (height_m**2)
    category, _tone = bmi_category(imc)
    long_data = df[df["Date"] >= last_date - pd.Timedelta(days=30)]["Poids (Kgs)"]
    long = float(long_data.mean()) if not long_data.empty else df["Poids (Kgs)"].mean()
    long_n = len(long_data)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card(
            f"Rythme {LONG_RATE_WINDOW_DAYS} jours",
            _rate_text(rate_long),
            help_text=(
                f"Pente de fond sur {LONG_RATE_WINDOW_DAYS} jours. IC 95 % : {_rate_interval_text(rate_long)}."
                if rate_long.get("ready")
                else f"Indisponible : {rate_long.get('reason')}."
            ),
        )
    with c2:
        kpi_card("Moy. 30 derniers jours", f"{long:.2f} kg", help_text=f"Basé sur {long_n} mesure(s) des 30 derniers jours calendaires")
    with c3:
        st.metric(
            "IMC",
            format_fr_number(imc, decimals=1),
            delta=category,
            delta_color="off",
            help=(
                f"Indice de masse corporelle pour {format_fr_number(height_m, decimals=2)} m. Repère OMS de population, pas un diagnostic : "
                f"un IMC de 25 correspond à {format_fr_kg(weight_for_bmi(25.0, height_m), decimals=1)}, un IMC de 30 à "
                f"{format_fr_kg(weight_for_bmi(30.0, height_m), decimals=1)}."
            ),
        )
    with c4:
        kpi_card("Qualité des données", f"{quality['score']}/100", help_text=f"Couverture {quality['coverage_pct']} %, {quality['anomalies']} valeur(s) atypique(s), {quality['duplicates']} doublon(s).")
    with c5:
        noise = context["noise"]
        kpi_card(
            "Bruit quotidien",
            f"± {format_fr_kg(noise['band'], decimals=1)}" if noise.get("ready") else "—",
            help_text=(
                f"Demi-largeur à 95 % des écarts entre pesées et tendance (σ robuste {format_fr_kg(noise['sigma'], decimals=2)}). "
                "Un écart plus petit que cela d'un jour à l'autre n'est pas un changement de poids."
                if noise.get("ready")
                else "Disponible après cinq mesures."
            ),
        )

    c6, c7, c8, c9 = st.columns(4)
    with c6:
        v7 = vel.get(7)
        v_display = f"{format_fr_number(v7, decimals=2, sign=True)} kg/sem" if v7 is not None else "N/A"
        v_help = "Variation de poids en kg par semaine sur les 7 derniers jours (deux pesées, sans lissage)."
        if is_startup:
            v_display += " ⚠️"
            v_help += " — signal fragile (< 7 jours de données)"
        kpi_card("Vitesse 7j", v_display, help_text=v_help)
    with c7:
        kpi_card("Discipline", f"{disc['score']}/100", help_text=f"{disc['interpretation'].title()} — {disc['measured_days']}/{disc['expected_days']} jours mesurés")
    with c8:
        confidence_label = prog.get("confidence", "solide")
        conf_icon = "⚠️ " if confidence_label == "fragile" else ""
        kpi_card("Score global", f"{conf_icon}{prog['score']}/100 ({prog['grade']})", help_text=f"Score composite. {'Signal fragile (< 7 mesures)' if confidence_label == 'fragile' else 'Signal fiable'}")
    with c9:
        streak_icon = "🔥" if streaks["current_type"] == "perte" else "📈" if streaks["current_type"] == "gain" else "➡️"
        streak_txt = f"{streak_icon} {streaks['current_streak']} mesures en {streaks['current_type']}"
        kpi_card("Série en cours", streak_txt, help_text=f"Record perte : {streaks['longest_loss']} mesures · Record gain : {streaks['longest_gain']} mesures")

    if trajectory_status and trajectory_status.get("available"):
        progress = float(trajectory_status["progress_pct"])
        progress_label = f"Progression vers {_format_fr_kg(trajectory_status['final_target_weight'])}"
        st.caption(
            f"🎯 Trajectoire cible : {_format_fr_kg(trajectory_status['scheduled_weight'])} attendus au "
            f"{trajectory_status['current_date'].strftime('%d/%m/%Y')}. "
            f"{_trajectory_position_sentence(trajectory_status)} "
            f"Date cible : {trajectory_status['end_date'].strftime('%d/%m/%Y')}."
        )
    elif has_effort_period:
        effort_initial_weight = float(effort_df["Poids (Kgs)"].iloc[0])
        if effort_initial_weight > target_weight:
            total = effort_initial_weight - target_weight
            progress = ((effort_initial_weight - current) / total * 100) if total > 0 else 0.0
        else:
            progress = 100.0
        progress_label = f"Progression effort actuel vers {_format_fr_kg(target_weight)}"
    else:
        initial = df["Poids (Kgs)"].iloc[0]
        total = initial - target_weight
        progress = ((initial - current) / total * 100) if total > 0 else 0.0
        progress_label = f"Progression vers l'objectif final ({_format_fr_kg(target_weight)})"

    progress_panel(
        progress_label,
        max(0.0, min(100.0, progress)),
        "Les métriques marquées ⚠️ restent informatives et plus fragiles en phase de démarrage.",
        tone="success" if progress >= 100 else "primary",
    )

    v14 = vel.get(14)
    milestone = next_milestone(current, targets, velocity=v14, measurements=effort_measurements)
    ms_text = f"🎯 **Prochain palier** : {milestone['label']} (reste **{milestone['remaining']:.1f} kg**)"
    if milestone.get("eta_days") is not None:
        conf = milestone.get("eta_confidence", "")
        conf_note = f" (confiance : {conf})" if conf else ""
        ms_text += f" — Date estimée : **~{milestone['eta_days']} jours**{conf_note}"
    elif milestone.get("eta_confidence") == "fragile":
        ms_text += " — Date estimée disponible après 7+ mesures"
    st.caption(ms_text)

    plateau = detect_plateau(analysis_df, window=14)
    nb_mesures_plateau = plateau.get("nb_mesures", 0)
    if nb_mesures_plateau >= STAGNATION_MIN_MEASUREMENTS:
        if plateau["status"] == "plateau probable":
            alert_banner(f"➡️ Plateau probable détecté ({nb_mesures_plateau} mesures, pente = {format_fr_number(plateau['slope'], decimals=3, sign=True)} kg/sem)", "warning")
        elif plateau["status"] == "baisse active":
            alert_banner(f"📉 Tendance à la baisse ({nb_mesures_plateau} mesures, pente = {format_fr_number(plateau['slope'], decimals=3, sign=True)} kg/sem)", "success")
        elif "reprise" in plateau["status"]:
            alert_banner(f"📈 Tendance à la hausse ({nb_mesures_plateau} mesures, pente = {format_fr_number(plateau['slope'], decimals=3, sign=True)} kg/sem)", "warning")
    else:
        st.caption(f"📊 Signal de tendance : données limitées ({nb_mesures_plateau} mesures sur 14 j)")

    confidence = "élevée" if quality["score"] > 80 else "moyenne" if quality["score"] > 60 else "faible"
    confidence_badge("Confiance signal", confidence)
    _render_smart_alerts(df, analysis_df, streaks, last_date)
    return analysis_df, prog, streaks


def _daily_insight_title(body: str, position: int) -> str:
    text = body.lower()
    if "7 jours" in text:
        return "Tendance 7 jours"
    if "30 jours" in text or "reprise" in text:
        return "Évolution récente"
    if "palier" in text or "passer sous" in text:
        return "Prochain palier"
    if "objectif final" in text or "objectif configuré" in text:
        return "Distance à l'objectif"
    if "stabilité" in text or "stable" in text:
        return "Stabilité du poids"
    if "trajectoire" in text:
        return "Écart à la trajectoire"
    fallback_titles = ("Tendance récente", "Distance au prochain palier", "Analyse complémentaire")
    return fallback_titles[min(position, len(fallback_titles) - 1)]


def _daily_insight_tone(body: str, default_tone: str) -> str:
    text = body.lower()
    if "baisse" in text or "perdu" in text:
        return "success"
    if "hausse" in text or "remonte" in text or "attention" in text:
        return "warning"
    if "objectif" in text or "palier" in text:
        return "info"
    return default_tone


def _render_simple_insights(summary: dict) -> None:
    section_header("Insights automatiques", "Les 3 signaux les plus utiles pour suivre la tendance actuelle.", "💡")
    trend_tone = {"Baisse": "success", "Hausse": "warning", "Stable": "info"}.get(summary.get("trend_label"), "neutral")
    insights = [str(item).replace("**", "") for item in summary.get("insights", []) if str(item).strip()]
    primary_insights = insights[:3]
    detailed_insights = insights[3:]

    for idx, insight in enumerate(primary_insights):
        tone = _daily_insight_tone(insight, trend_tone if idx == 0 else "neutral")
        icon = "📉" if tone == "success" else "📈" if tone == "warning" else "🎯" if "objectif" in insight.lower() or "palier" in insight.lower() else "🔎"
        insight_card(_daily_insight_title(insight, idx), insight, tone=tone, icon=icon)

    if detailed_insights:
        with st.expander("Voir les analyses détaillées", expanded=False):
            for idx, insight in enumerate(detailed_insights, start=len(primary_insights)):
                insight_card(_daily_insight_title(insight, idx), insight, tone=_daily_insight_tone(insight, "neutral"), icon="🔎")


def _render_objective_gap_chart(df: pd.DataFrame, target_weight: float, context: dict) -> None:
    if df.empty:
        return
    gap_df = df[["Date", "Poids (Kgs)"]].copy()
    gap_df["Écart"] = gap_df["Poids (Kgs)"] - target_weight
    fig = go.Figure()
    add_measure_trace(fig, gap_df["Date"], gap_df["Écart"], name="Écart à l'objectif (pesées)", color=MEASURE_COLOR, marker_size=5, width=1.6, opacity=0.75)
    frame = context["frame"]
    if not frame.empty:
        add_line_trace(fig, frame["Date"], frame["Tendance"] - target_weight, name="Écart à l'objectif (tendance)", color=TREND_COLOR, width=2.6)
    add_horizontal_reference(fig, gap_df["Date"], 0.0, name="Objectif atteint", color=TARGET_COLOR)
    apply_layout(fig, "Écart par rapport à l'objectif final", y_title="Écart (kg)", height=340)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Graphique principal
# ──────────────────────────────────────────────────────────────────────────────


def filter_weight_period(
    df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return weight rows inside the inclusive date window without mutating the input."""
    filtered = df.copy()
    if start_date is not None:
        filtered = filtered[filtered["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered[filtered["Date"] <= pd.Timestamp(end_date)]
    return filtered.copy()


def _add_secondary_targets(fig: go.Figure, targets: tuple[float, ...], x_values, label_prefix: str = "Objectif secondaire") -> None:
    if x_values is None or len(x_values) == 0:
        return
    for idx, target in enumerate(targets, start=1):
        add_horizontal_reference(
            fig,
            x_values,
            float(target),
            name=f"{label_prefix} {idx} : {format_fr_kg(target)}",
            color=SECONDARY_TARGET_COLORS[(idx - 1) % len(SECONDARY_TARGET_COLORS)],
            dash="dot",
            width=1.0,
        )


def build_weight_chart(
    df: pd.DataFrame,
    *,
    title: str,
    target_weight: float,
    targets: tuple[float, ...],
    trajectory_config: TargetTrajectoryConfig,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    show_secondary_targets: bool = True,
    show_long_term_trend: bool = False,
    show_forecast: bool = True,
    show_moving_average: bool = False,
    ma_window_label: str = "7 jours",
    show_trend: bool = True,
    trend_frame: pd.DataFrame | None = None,
    noise_band: float | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    """Build the dashboard weight chart for either full history or a zoomed period."""
    chart_df = filter_weight_period(df, start_date, end_date)
    fig = go.Figure()

    x_values = chart_df["Date"] if not chart_df.empty else pd.DatetimeIndex(
        [pd.Timestamp(start_date), pd.Timestamp(end_date)]
        if start_date is not None and end_date is not None
        else []
    )

    trend_window = filter_weight_period(trend_frame, start_date, end_date) if (show_trend and trend_frame is not None and not trend_frame.empty) else pd.DataFrame()
    if not trend_window.empty and noise_band is not None and np.isfinite(noise_band) and noise_band > 0:
        add_band(
            fig,
            trend_window["Date"],
            trend_window["Tendance"] - noise_band,
            trend_window["Tendance"] + noise_band,
            name=f"Bruit habituel (± {format_fr_kg(noise_band, decimals=1)})",
            fill=TREND_BAND_FILL,
        )

    if not chart_df.empty:
        # Pesées en retrait (fines, translucides) : la tendance, tracée ensuite,
        # reste lisible par-dessus au lieu de disparaître sous les marqueurs.
        add_measure_trace(fig, chart_df["Date"], chart_df["Poids (Kgs)"], name="Poids mesuré", marker_size=4, width=1.2, opacity=0.75 if not trend_window.empty else 1.0)

    if not trend_window.empty:
        add_line_trace(
            fig,
            trend_window["Date"],
            trend_window["Tendance"],
            name=f"Poids de tendance (LOWESS {TREND_WINDOW_DAYS} j)",
            color=TREND_COLOR,
            width=3.0,
        )

    if show_moving_average:
        ma_col = "MA_7J" if ma_window_label == "7 jours" else "MA_30J"
        full_df = df.copy()
        full_df["MA_7J"] = moving_average_by_days(full_df, 7)
        full_df["MA_30J"] = moving_average_by_days(full_df, 30)
        ma_df = filter_weight_period(full_df, start_date, end_date)
        if not ma_df.empty:
            add_line_trace(fig, ma_df["Date"], ma_df[ma_col], name=f"Moyenne mobile {ma_window_label}", color=NEUTRAL_LINE, dash="dot", width=1.4)

    add_horizontal_reference(fig, x_values, target_weight, name=f"Objectif principal : {format_fr_kg(target_weight)}", color=TARGET_COLOR, dash="dash", width=1.6)

    if show_secondary_targets:
        secondary_targets = tuple(t for t in targets if round(float(t), 3) != round(float(target_weight), 3))
        if secondary_targets:
            _add_secondary_targets(fig, secondary_targets, x_values)

    if show_long_term_trend:
        trend_span = int(st.session_state.get("window_size", 14))
        df_ema = filter_weight_period(compute_trend_ema(df, span=trend_span), start_date, end_date)
        if not df_ema.empty:
            add_line_trace(fig, df_ema["Date"], df_ema["Tendance_EMA"], name=f"Tendance long terme EMA ({trend_span})", color="#7c3aed", dash="dot", width=1.8)

    target_trajectory = build_target_trajectory(df, trajectory_config)
    if show_forecast and target_trajectory.get("available"):
        trajectory_df = filter_weight_period(target_trajectory["trajectory"], start_date, end_date)
        if not trajectory_df.empty:
            rate = target_trajectory["required_weekly_loss"]
            rate_label = format_fr_kg(rate, decimals=2, trim_zeros=False).replace(" kg", " kg/semaine")
            end_date_label = target_trajectory["end_date"].strftime("%d/%m/%Y")
            label = f"Trajectoire cible vers 80 kg au {end_date_label} — {rate_label}"
            fig.add_scatter(
                x=trajectory_df["Date"],
                y=trajectory_df["Poids cible (kg)"],
                mode="lines",
                name=label,
                line=dict(color=TRAJECTORY_COLOR, width=2.2, dash="dashdot"),
                hovertemplate="Poids cible : %{y:.1f} kg<br>Perte moyenne requise : " + rate_label + "<extra></extra>",
            )

    apply_layout(fig, title, y_title="Poids (kg)", height=480)
    if start_date is not None and end_date is not None:
        fig.update_xaxes(range=[pd.Timestamp(start_date), pd.Timestamp(end_date)])
    return fig, chart_df


def _render_main_weight_chart(
    df: pd.DataFrame,
    target_weight: float,
    targets: tuple[float, ...],
    trajectory_config: TargetTrajectoryConfig,
    context: dict,
) -> None:
    section_header(
        "Évolution du poids",
        "Pesées, poids de tendance avec sa zone de bruit habituel, objectifs et trajectoire cible.",
        "📈",
    )

    with st.expander("Options d'affichage du graphique", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            show_trend = st.checkbox(
                "Poids de tendance et bruit",
                value=True,
                help="Régression locale robuste : la courbe orange est le poids débarrassé des fluctuations quotidiennes, la bande la zone où une pesée ordinaire peut tomber.",
            )
            show_moving_average = st.checkbox("Moyenne mobile", value=False, help="Repère secondaire, plus rudimentaire que le poids de tendance.")
            selected_ma = st.radio("Fenêtre", ["7 jours", "30 jours"], index=0, horizontal=True)
        with c2:
            show_secondary_targets = st.checkbox("Objectifs secondaires", value=True)
            show_long_term_trend = st.checkbox("Tendance long terme (EMA)", value=False)
        with c3:
            show_forecast = st.checkbox("Trajectoire cible", value=True)
            st.caption(
                f"Paramètres trajectoire : départ {trajectory_config.start_date.strftime('%d/%m/%Y')} · "
                f"objectif {_format_fr_kg(trajectory_config.final_target_weight)} au {trajectory_config.end_date.strftime('%d/%m/%Y')}"
            )

    noise_band = context["noise"]["band"] if context["noise"].get("ready") else None
    fig, _ = build_weight_chart(
        df,
        title="Évolution du poids",
        target_weight=target_weight,
        targets=targets,
        trajectory_config=trajectory_config,
        show_secondary_targets=show_secondary_targets,
        show_long_term_trend=show_long_term_trend,
        show_forecast=show_forecast,
        show_moving_average=show_moving_average,
        ma_window_label=selected_ma,
        show_trend=show_trend,
        trend_frame=context["frame"],
        noise_band=noise_band,
    )
    st.plotly_chart(fig, use_container_width=True, key=TARGET_TRAJECTORY_CHART_KEY)
    st.caption(
        "Le poids de tendance est une régression locale robuste (LOWESS, Cleveland 1979) sur une fenêtre de "
        f"{TREND_WINDOW_DAYS} jours calendaires ; la bande orange couvre 95 % des écarts habituels entre une pesée et cette tendance. "
        "Une pesée dans la bande n'est pas un changement de poids."
    )

    section_header(
        "Zoom sur la période cible",
        "Même lecture que le graphique principal, limitée à la fenêtre configurable dans les paramètres.",
        "🔎",
    )
    zoom_start = pd.Timestamp(st.session_state.get("zoom_target_start_date", DEFAULT_ZOOM_TARGET_START_DATE))
    zoom_end = pd.Timestamp(st.session_state.get("zoom_target_end_date", DEFAULT_ZOOM_TARGET_END_DATE))
    if zoom_start > zoom_end:
        st.warning("La date de début du zoom doit être antérieure ou égale à la date de fin.")
        return

    zoom_title = f"Évolution du poids — période {zoom_start.strftime('%d/%m/%Y')} au {zoom_end.strftime('%d/%m/%Y')}"
    zoom_fig, zoom_df = build_weight_chart(
        df,
        title=zoom_title,
        target_weight=target_weight,
        targets=targets,
        trajectory_config=trajectory_config,
        start_date=zoom_start,
        end_date=zoom_end,
        show_secondary_targets=show_secondary_targets,
        show_long_term_trend=show_long_term_trend,
        show_forecast=show_forecast,
        show_moving_average=show_moving_average,
        ma_window_label=selected_ma,
        show_trend=show_trend,
        trend_frame=context["frame"],
        noise_band=noise_band,
    )
    if zoom_df.empty:
        st.info("Aucune donnée de poids disponible sur cette période.")
    st.plotly_chart(zoom_fig, use_container_width=True, key=TARGET_TRAJECTORY_ZOOM_CHART_KEY)


def _trajectory_config_controls() -> TargetTrajectoryConfig:
    """Return the fixed business trajectory configuration and document it in the sidebar."""
    with st.sidebar.expander("Trajectoire cible", expanded=False):
        st.caption(
            "La trajectoire cible démarre le "
            f"{DEFAULT_TARGET_TRAJECTORY_START_DATE.strftime('%d/%m/%Y')} et atteint "
            f"{_format_fr_kg(DEFAULT_FINAL_TARGET_WEIGHT)} le {DEFAULT_TARGET_TRAJECTORY_END_DATE.strftime('%d/%m/%Y')}."
        )
        st.caption(f"Départ : {_format_fr_kg(DEFAULT_TARGET_TRAJECTORY_START_WEIGHT)} le {DEFAULT_TARGET_TRAJECTORY_START_DATE.strftime('%d/%m/%Y')}")
        st.caption(
            f"Objectif : {_format_fr_kg(DEFAULT_FINAL_TARGET_WEIGHT, trim_zeros=False)} "
            f"le {DEFAULT_TARGET_TRAJECTORY_END_DATE.strftime('%d/%m/%Y')}"
        )
        st.caption(f"Durée : {TARGET_TRAJECTORY_TOTAL_DURATION_DAYS} jours")
        rate_label = format_fr_kg(required_weekly_loss(), decimals=2, trim_zeros=False).replace(" kg", " kg/semaine")
        st.caption(f"Rythme moyen requis : {rate_label}")
        st.caption(
            "Règle métier fixe : les mesures CSV servent uniquement à comparer le poids réel, "
            "jamais à modifier le départ ou la pente."
        )
    return TargetTrajectoryConfig.from_values(
        DEFAULT_TARGET_TRAJECTORY_START_DATE,
        final_target_weight=DEFAULT_FINAL_TARGET_WEIGHT,
        start_weight=DEFAULT_TARGET_TRAJECTORY_START_WEIGHT,
        duplicate_strategy=st.session_state.get("duplicate_strategy", "garder_la_derniere"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Onglets secondaires
# ──────────────────────────────────────────────────────────────────────────────


def _render_forecast_tab(df: pd.DataFrame, daily_summary: dict, trajectory_status: dict, context: dict, target_weight: float) -> None:
    section_header("Prévisions", "Les projections restent prudentes et sont séparées de la lecture principale.", "🔮")
    rate = context["rate_long"]
    frame = context["frame"]
    if rate.get("ready") and not frame.empty:
        trend_now = float(frame["Tendance"].iloc[-1])
        remaining = trend_now - target_weight
        if remaining > 0 and rate["direction"] == "baisse":
            last_date = frame["Date"].iloc[-1]
            central_days = remaining / abs(rate["slope_kg_week"] / 7.0)
            fast_days = remaining / abs(rate["ci_low"] / 7.0)
            eta_text = f"vers le **{format_fr_date(last_date + pd.Timedelta(days=int(round(central_days))))}**"
            if rate["ci_high"] < 0:
                slow_days = remaining / abs(rate["ci_high"] / 7.0)
                eta_text += (
                    f" (plage plausible : {format_fr_date(last_date + pd.Timedelta(days=int(round(fast_days))))} → "
                    f"{format_fr_date(last_date + pd.Timedelta(days=int(round(slow_days))))})"
                )
            st.success(
                f"📐 Au rythme des {LONG_RATE_WINDOW_DAYS} derniers jours ({_rate_text(rate)}, IC 95 % {_rate_interval_text(rate)}), "
                f"le poids de tendance atteindrait {format_fr_kg(target_weight, decimals=1)} {eta_text}. "
                "Extrapolation linéaire : une réponse à « et si cela continuait », pas une prédiction."
            )
        elif remaining <= 0:
            st.success("📐 Le poids de tendance est déjà sous l'objectif principal.")
        else:
            st.info(
                f"📐 Sur {LONG_RATE_WINDOW_DAYS} jours, la pente ({_rate_text(rate)}, IC 95 % {_rate_interval_text(rate)}) ne se distingue pas "
                "d'une stagnation ou remonte : aucune date d'arrivée n'est projetée."
            )
    else:
        st.caption(f"📐 Projection sur le rythme de fond disponible après cinq pesées réparties sur au moins sept jours ({rate.get('reason', 'recul insuffisant')}).")

    if trajectory_status.get("available"):
        st.info(
            f"🎯 Plan d'atteinte de l'objectif : {_format_fr_kg(trajectory_status['final_target_weight'])} autour du "
            f"**{trajectory_status['eta_date'].strftime('%d/%m/%Y')}**. "
            f"{_trajectory_position_sentence(trajectory_status)}"
        )
    projection = daily_summary.get("projection", {}) if daily_summary.get("valid") else {}
    if projection.get("available") and not projection.get("reached"):
        eta = projection["eta"].strftime("%d/%m/%Y")
        st.info(
            f"📍 Projection prudente : objectif vers le **{eta}** (~{projection['days_needed']} jours) "
            f"si le rythme récent ({projection['pace_kg_week']:+.2f} kg/sem) se maintient. Ce n'est pas une promesse."
        )
    elif projection.get("available") and projection.get("reached"):
        st.success(f"🎯 {projection['message']}")
    else:
        st.caption(f"📍 {projection.get('message', 'Projection non disponible pour le moment.')}")

    history = analyze_effort_history(df)
    if history.get("insight"):
        st.warning(history["insight"])
        if history.get("best_effort"):
            best = history["best_effort"]
            st.caption(
                f"💪 Meilleur effort passé : **-{best['delta']:.1f} kg** en {best['days']} jours "
                f"({best['start_date'].strftime('%d/%m/%Y')} → {best['end_date'].strftime('%d/%m/%Y')}, {best['measurements']} mesures)"
            )
    st.caption("La page Prévisions compare ces projections à des modèles statistiques validés par backtest.")


def _render_history_tab(df: pd.DataFrame, targets: tuple[float, ...], height_m: float, context: dict) -> None:
    section_header("Historique", "Graphiques secondaires utiles, masqués par défaut pour garder la vue principale légère.", "📚")
    with st.expander("📊 Vue hebdomadaire consolidée", expanded=False):
        _render_weekly_view(df, targets)

    with st.expander("📊 Moyennes mobiles multiples (7/14/30 mesures)", expanded=False):
        df_ma = multi_rolling_averages(df, windows=(7, 14, 30))
        fig_ma = go.Figure()
        add_measure_trace(fig_ma, df_ma["Date"], df_ma["Poids (Kgs)"], name="Mesures", mode="markers", opacity=0.45, marker_size=5)
        colors = {"MA_7m": MEASURE_COLOR, "MA_14m": TREND_COLOR, "MA_30m": TARGET_COLOR}
        labels = {"MA_7m": "MA 7 mesures", "MA_14m": "MA 14 mesures", "MA_30m": "MA 30 mesures"}
        for col, color in colors.items():
            if col in df_ma.columns:
                add_line_trace(fig_ma, df_ma["Date"], df_ma[col], name=labels[col], color=color, width=2.0)
        _add_secondary_targets(fig_ma, targets, df_ma["Date"], label_prefix="Obj.")
        apply_layout(fig_ma, "Moyennes mobiles (glissantes sur N mesures consécutives)", y_title="Poids (kg)", height=420)
        st.plotly_chart(fig_ma, use_container_width=True)
        st.caption("ℹ️ Les moyennes mobiles glissent sur N mesures consécutives (et non sur N jours calendaires).")

    with st.expander("Fluctuations quotidiennes", expanded=False):
        frame = context["frame"]
        noise = context["noise"]
        if noise.get("ready") and not frame.empty:
            residuals = frame["Résidu"].dropna()
            fig_res = histogram_figure(
                residuals,
                "Écarts entre chaque pesée et le poids de tendance",
                x_title="Écart (kg)",
                nbins=25,
                reference=0.0,
                reference_label="tendance",
            )
            fig_res.add_vrect(x0=-noise["band"], x1=noise["band"], fillcolor=TREND_BAND_FILL, line_width=0, layer="below", annotation_text="bruit habituel", annotation_position="top left")
            st.plotly_chart(fig_res, use_container_width=True)
            st.caption(
                f"σ robuste : {format_fr_kg(noise['sigma'], decimals=2)} ; 95 % des pesées tombent à moins de "
                f"{format_fr_kg(noise['band'], decimals=1)} de la tendance. Ces écarts tiennent surtout à l'eau, au glycogène et au contenu digestif."
            )
        else:
            st.info("Disponible après cinq pesées.")

    with st.expander("Distribution et évolution IMC", expanded=False):
        hist = histogram_figure(df["Poids (Kgs)"], "Distribution du poids", x_title="Poids (kg)", nbins=25, reference=float(df["Poids (Kgs)"].iloc[-1]), reference_label="dernière pesée")
        st.plotly_chart(hist, use_container_width=True)
        df_bmi = df.copy()
        df_bmi["IMC"] = df_bmi["Poids (Kgs)"] / (height_m**2)
        fig_bmi = go.Figure()
        add_measure_trace(fig_bmi, df_bmi["Date"], df_bmi["IMC"], name="IMC", marker_size=4, width=1.6)
        for threshold, label in ((25.0, "IMC 25 — surpoids"), (30.0, "IMC 30 — obésité")):
            add_horizontal_reference(fig_bmi, df_bmi["Date"], threshold, name=label, color=NEUTRAL_LINE, dash="dot", width=1.0)
        apply_layout(fig_bmi, "Évolution de l'IMC", y_title="IMC", height=360)
        fig_bmi.update_traces(hovertemplate="%{y:.1f}<extra>IMC</extra>", selector=dict(name="IMC"))
        st.plotly_chart(fig_bmi, use_container_width=True)
        st.caption(
            "Seuils de l'OMS pour l'adulte : surpoids à partir de 25, obésité à partir de 30. Repères de population, "
            "sans valeur diagnostique individuelle."
        )


def _render_targets_tab(df: pd.DataFrame, targets: tuple[float, ...], target_weight: float, prog: dict, context: dict) -> None:
    section_header("Objectifs & paliers", "Chaque palier avec ce qu'il reste à perdre et, quand la pente est établie, une date indicative.", "🎯")
    st.caption(_targets_caption(targets))
    st.caption(f"Objectif principal affiché par défaut : {target_weight:.1f} kg.")

    frame = context["frame"]
    rate = context["rate_long"]
    if frame.empty:
        st.info("Aucune mesure exploitable.")
        return
    trend_now = float(frame["Tendance"].iloc[-1])
    last_date = frame["Date"].iloc[-1]
    slope_day = rate["slope_kg_week"] / 7.0 if rate.get("ready") else float("nan")
    rows = []
    for idx, target in enumerate(sorted(set(float(t) for t in targets), reverse=True), start=1):
        remaining = trend_now - target
        if remaining <= 0:
            status, eta = "✅ atteint", "—"
        elif rate.get("ready") and rate["direction"] == "baisse" and slope_day < 0:
            days = remaining / abs(slope_day)
            eta = format_fr_date(last_date + pd.Timedelta(days=int(round(days)))) if days <= 3 * 365 else "> 3 ans"
            status = "⏳ en cours"
        else:
            status, eta = "⏳ en cours", "— (pente non établie)"
        rows.append(
            {
                "Palier": f"{idx}",
                "Poids cible": format_fr_kg(target, decimals=1),
                "Reste (tendance)": format_fr_kg(max(remaining, 0.0), decimals=1),
                "Statut": status,
                f"Date indicative au rythme {LONG_RATE_WINDOW_DAYS} j": eta,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Le reste est calculé depuis le poids de tendance, pas depuis la pesée du jour. "
        + (
            f"Dates fondées sur le rythme de fond {_rate_text(rate)} (IC 95 % {_rate_interval_text(rate)})."
            if rate.get("ready")
            else "Aucune date : la pente de fond n'est pas encore estimable."
        )
    )

    with st.expander("🏆 Détail du score de progression", expanded=False):
        components = prog.get("components", {})
        if components:
            comp_cols = st.columns(len(components))
            for i, (key, val) in enumerate(components.items()):
                with comp_cols[i]:
                    st.metric(key.title(), f"{val:.0f}")
        else:
            st.caption("Score détaillé indisponible pour le moment.")


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ensure_session_defaults()
    df = _df()
    if df.empty:
        empty_state("Aucune donnée disponible. Utilisez Journal ou import CSV.")
        return

    df = df.sort_values("Date").copy()
    current = float(df["Poids (Kgs)"].iloc[-1])
    last_date = df["Date"].max()

    effort = detect_current_effort(df, gap_threshold_days=21)
    effort_df = effort["effort_df"]
    effort_start = effort["start_date"]
    effort_days = effort["days"]
    effort_measurements = effort["measurements"]
    has_effort_period = effort["is_subset"] and len(effort_df) >= 2
    is_startup = effort_days < 7

    height_m = st.session_state.get("height_m", 1.82)
    targets = get_target_weights(st.session_state)
    trajectory_config = _trajectory_config_controls()
    target_weight = float(trajectory_config.final_target_weight)
    trajectory_status = compare_to_target_trajectory(df, trajectory_config)
    daily_summary = summarize_weight_journey(df, target_weight)
    context = _trend_context(df)
    analysis_df = effort_df if has_effort_period else df
    prog = progression_score(analysis_df, target_weight)

    page_hero(
        "Suivi de poids",
        "Dashboard",
        "Le poids du jour, la tendance débarrassée du bruit quotidien, le rythme réel et l'écart à l'objectif.",
        meta=f"Dernière mesure : {last_date.strftime('%d/%m/%Y')} · {len(df)} mesure(s) · objectif principal {_format_fr_kg(target_weight)}",
    )

    if daily_summary.get("valid"):
        _render_daily_overview(daily_summary, target_weight, trajectory_status, context)
    else:
        st.warning(daily_summary.get("message", "Données insuffisantes pour calculer les indicateurs."))

    if has_effort_period:
        effort_initial = float(effort_df["Poids (Kgs)"].iloc[0])
        effort_delta = effort_initial - current
        delta_icon = "📉" if effort_delta > 0 else "📈" if effort_delta < 0 else "➡️"
        if effort_delta > 0:
            delta_text = f"-{_format_fr_kg(effort_delta)}"
        elif effort_delta < 0:
            delta_text = f"+{_format_fr_kg(abs(effort_delta))}"
        else:
            delta_text = _format_fr_kg(0)
        st.info(
            f"📅 **Période d'effort actuelle** : depuis le {effort_start.strftime('%d/%m/%Y')} "
            f"({effort_days} jours, {effort_measurements} mesures) — "
            f"{delta_icon} **{delta_text}** ({_format_fr_kg(effort_initial)} → {_format_fr_kg(current)})"
        )

    _render_main_weight_chart(df, target_weight, targets, trajectory_config, context)

    tab_analysis, tab_forecast, tab_history, tab_targets = st.tabs(
        ["Analyse détaillée", "Prévisions", "Historique", "Objectifs & paliers"]
    )

    with tab_analysis:
        summary = generate_action_summary(df, target_weight)
        section_header("Votre résumé", "Un diagnostic court pour savoir où vous en êtes et quoi regarder ensuite.", "🧭")
        col_s, col_i, col_a = st.columns(3)
        with col_s:
            insight_card("Situation", summary["situation"], tone="info", icon="📍")
        with col_i:
            insight_card("Interprétation", summary["interpretation"], tone="neutral", icon="🔍")
        with col_a:
            insight_card("Action", summary["action"], tone="success", icon="▶️")

        if daily_summary.get("valid"):
            _render_simple_insights(daily_summary)

        with st.expander("📌 Indicateurs avancés", expanded=False):
            analysis_df, prog, _streaks = _render_advanced_kpis(
                df,
                effort_df,
                has_effort_period,
                effort_days,
                effort_measurements,
                current,
                target_weight,
                targets,
                height_m,
                is_startup,
                last_date,
                context,
                trajectory_status,
            )

        with st.expander("💡 Insights détaillés", expanded=False):
            insights = generate_insights_text(df, target_weight)
            for insight in insights:
                st.markdown(f"> {insight}")
            acc = weight_acceleration(analysis_df)
            if acc["interpretation"] != "données insuffisantes":
                st.caption(f"📐 Accélération : {acc['interpretation']}")
            if not is_startup:
                pace = pace_comparison(analysis_df, target_weight)
                if pace.get("current_pace") is not None:
                    st.caption(f"🏎️ {pace['interpretation']}")
            else:
                st.caption("🏎️ Comparaison de rythme : disponible après 7+ jours de suivi.")

        with st.expander("🎯 Écart à l'objectif", expanded=False):
            _render_objective_gap_chart(df, target_weight, context)

        vol = weight_volatility(analysis_df, window=14)
        noise = context["noise"]
        noise_text = (
            f" · bruit quotidien ± {format_fr_kg(noise['band'], decimals=1)} autour de la tendance"
            if noise.get("ready")
            else ""
        )
        st.caption(
            f"📊 Volatilité (14 derniers jours) : {vol['interpretation']} (σ = {format_fr_kg(vol['std'], decimals=2)}, "
            f"amplitude = {format_fr_kg(vol['range'], decimals=1)}, {vol.get('nb_mesures', '?')} mesures){noise_text}."
        )

    with tab_forecast:
        _render_forecast_tab(df, daily_summary, trajectory_status, context, target_weight)

    with tab_history:
        _render_history_tab(df, targets, height_m, context)

    with tab_targets:
        _render_targets_tab(df, targets, target_weight, prog, context)

    help_box("À propos de cette page", "Cette vue est analytique et informative ; elle ne remplace pas un avis médical.")


def _render_smart_alerts(df: pd.DataFrame, analysis_df: pd.DataFrame, streaks: dict, last_date: pd.Timestamp) -> None:
    """Alertes intelligentes contextuelles."""
    days_since_last = (pd.Timestamp.now().normalize() - last_date).days
    if days_since_last >= 3:
        alert_banner(f"📏 Dernière mesure il y a {days_since_last} jours. Pensez à vous peser !", "info")

    if streaks["current_streak"] >= 3 and streaks["current_type"] == "perte":
        alert_banner(f"🔥 Belle série ! {streaks['current_streak']} mesures consécutives en baisse.", "success")

    if len(analysis_df) >= 5:
        recent_min = float(analysis_df["Poids (Kgs)"].tail(3).min())
        older_min = float(analysis_df["Poids (Kgs)"].iloc[:-3].min()) if len(analysis_df) > 3 else recent_min + 1
        if recent_min < older_min:
            alert_banner(f"🎉 Nouveau plus bas atteint : {recent_min:.1f} kg !", "success")

    if streaks["current_streak"] >= 1 and streaks["current_type"] == "gain" and streaks["longest_loss"] >= 3:
        if len(analysis_df) >= 3:
            alert_banner("📊 Petite remontée après une bonne série — c'est normal, les fluctuations quotidiennes sont naturelles.", "info")


def _render_weekly_view(df: pd.DataFrame, targets: tuple) -> None:
    """Vue hebdomadaire consolidée."""
    data = df.copy()
    data["week_start"] = data["Date"].dt.to_period("W").apply(lambda x: x.start_time)
    weekly = data.groupby("week_start").agg(
        poids_moyen=("Poids (Kgs)", "mean"),
        poids_min=("Poids (Kgs)", "min"),
        poids_max=("Poids (Kgs)", "max"),
        nb_mesures=("Poids (Kgs)", "count"),
    ).reset_index()
    weekly = weekly[weekly["nb_mesures"] >= 1]
    weekly["variation"] = weekly["poids_moyen"].diff()

    if weekly.empty:
        st.info("Pas assez de données hebdomadaires.")
        return

    display_weeks = weekly.tail(26)
    colors = [
        STATUS_GOOD if v is not None and np.isfinite(v) and v < -0.1 else STATUS_CRITICAL if v is not None and np.isfinite(v) and v > 0.1 else NEUTRAL_LINE
        for v in display_weeks["variation"]
    ]
    hover = [
        f"Semaine du {row.week_start.strftime('%d/%m/%Y')}<br>Moyenne : {row.poids_moyen:.1f} kg<br>Min : {row.poids_min:.1f} · Max : {row.poids_max:.1f}<br>{row.nb_mesures} mesure(s)"
        + (f"<br>Variation : {row.variation:+.2f} kg" if np.isfinite(row.variation) else "")
        for row in display_weeks.itertuples()
    ]
    fig_wk = bar_figure(
        display_weeks["week_start"],
        display_weeks["poids_moyen"],
        "Poids moyen par semaine (vert = baisse, rouge = hausse par rapport à la semaine précédente)",
        y_title="Poids moyen (kg)",
        colors=colors,
        hover=hover,
        height=400,
        date_axis=True,
    )
    _add_secondary_targets(fig_wk, targets, display_weeks["week_start"], label_prefix="Obj.")
    fig_wk.update_layout(showlegend=False)
    low = float(display_weeks["poids_min"].min())
    high = float(display_weeks["poids_max"].max())
    fig_wk.update_yaxes(range=[low - 1.5, high + 1.5])
    st.plotly_chart(fig_wk, use_container_width=True)
    st.caption("ℹ️ Chaque barre = moyenne des mesures de la semaine ; l'axe est resserré autour des valeurs observées pour rendre les variations lisibles.")


main()
