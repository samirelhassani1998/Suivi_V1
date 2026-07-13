from __future__ import annotations

import pandas as pd
import plotly.express as px
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
from app.core.formatting import format_fr_kg, format_fr_number, format_fr_unit
from app.core.insights import detect_plateau
from app.core.session_state import (
    DEFAULT_ZOOM_TARGET_END_DATE,
    DEFAULT_ZOOM_TARGET_START_DATE,
    ensure_session_defaults,
    get_filtered_or_working_data,
)
from app.core.target_trajectory import (
    DEFAULT_FINAL_TARGET_WEIGHT,
    DEFAULT_TARGET_TRAJECTORY_START_DATE,
    DEFAULT_TARGET_TRAJECTORY_START_WEIGHT,
    TargetTrajectoryConfig,
    build_target_trajectory,
    compare_to_target_trajectory,
)
from app.core.weight_summary import moving_average_by_days, summarize_weight_journey
from app.core.targets import get_target_weights
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


def _df() -> pd.DataFrame:
    return get_filtered_or_working_data()


def _add_target_lines(
    fig: go.Figure,
    targets: tuple[float, ...],
    label_prefix: str = "Objectif",
    x_values: pd.Series | pd.Index | None = None,
    *,
    showlegend: bool = True,
) -> None:
    """Draw target lines without chart annotations to avoid visual overlap."""
    target_values = tuple(float(target) for target in targets)
    colors = ("#2563eb", "#16a34a", "#f97316", "#7c3aed", "#dc2626")
    if x_values is None or len(x_values) == 0 or not target_values:
        return

    x_range = [x_values.min(), x_values.max()]
    for idx, target in enumerate(target_values, start=1):
        color = colors[(idx - 1) % len(colors)]
        label = f"{label_prefix} {idx} : {format_fr_kg(target)}"
        fig.add_scatter(
            x=x_range,
            y=[float(target), float(target)],
            mode="lines",
            name=label,
            showlegend=showlegend,
            line=dict(color=color, dash="dash", width=1.15),
            hovertemplate=f"{label}<extra></extra>",
        )


def _add_single_target_line(fig: go.Figure, target: float, x_values: pd.Series | pd.Index) -> None:
    """Add the single objective used by the default dashboard view."""
    if len(x_values) == 0:
        return
    label = f"Objectif principal : {format_fr_kg(target)}"
    fig.add_scatter(
        x=[x_values.min(), x_values.max()],
        y=[float(target), float(target)],
        mode="lines",
        name=label,
        line=dict(color="#16a34a", width=1.45, dash="dash"),
        hovertemplate=f"{label}<extra></extra>",
    )


def _targets_caption(targets: tuple[float, ...]) -> str:
    goals = " · ".join(f"Objectif {idx}: {format_fr_kg(target)}" for idx, target in enumerate(targets, start=1))
    return f"🎯 Objectifs affichés : {goals}."


def _format_delta(value: float | None, suffix: str = " kg") -> str:
    unit = suffix.strip()
    return format_fr_unit(value, unit, decimals=2, sign=True)


def _format_value(value: float | None, suffix: str = " kg") -> str:
    unit = suffix.strip()
    return format_fr_unit(value, unit, decimals=2)


def _format_fr_number(
    value,
    decimals: int = 1,
    *,
    sign: bool = False,
    trim_zeros: bool = True,
) -> str:
    return format_fr_number(
        value,
        decimals=decimals,
        sign=sign,
        trim_zeros=trim_zeros,
    )


def _format_fr_kg(
    value,
    decimals: int = 1,
    *,
    sign: bool = False,
    trim_zeros: bool = True,
) -> str:
    return format_fr_kg(
        value,
        decimals=decimals,
        sign=sign,
        trim_zeros=trim_zeros,
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



def _apply_modern_chart_layout(fig: go.Figure, title: str, height: int = 520) -> go.Figure:
    """Apply a consistent, polished Plotly style for dashboard charts."""
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=19, color="#0f172a")),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12, color="#475467"),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        margin=dict(l=12, r=12, t=112, b=34),
        font=dict(family="Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#111827"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title_font=dict(color="#667085"), tickfont=dict(color="#667085"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zeroline=False, title_font=dict(color="#667085"), tickfont=dict(color="#667085"))
    return fig


def _trend_sentence(summary: dict, target_weight: float) -> tuple[str, str]:
    trend = summary.get("trend_label", "Stable")
    delta_30 = summary.get("delta_30")
    delta_value = getattr(delta_30, "value", None)
    gap = summary.get("target_gap")

    if trend == "Baisse":
        trend_text = "Votre poids est en baisse sur la période récente."
        tone = "success"
    elif trend == "Hausse":
        trend_text = "Votre poids est en hausse sur la période récente."
        tone = "warning"
    else:
        trend_text = "La tendance récente est stable."
        tone = "info"

    if delta_value is not None:
        trend_text = f"Votre poids a varié de {delta_value:+.2f} kg sur les 30 derniers jours. " + trend_text
    if gap is not None:
        trend_text += f" Vous êtes à {abs(gap):.1f} kg de l’objectif principal ({target_weight:.1f} kg)."
    return trend_text, tone

def _render_daily_overview(summary: dict, target_weight: float, trajectory_status: dict | None = None) -> None:
    section_header(
        "Vue rapide",
        "Les cinq informations clés pour comprendre la situation en quelques secondes.",
        "⚡",
    )

    delta_7 = summary["delta_7"]
    delta_30 = summary["delta_30"]
    cols = st.columns(5)
    with cols[0]:
        kpi_card("Poids actuel", _format_value(summary["current"]), _format_delta(summary["previous_delta"]), "Dernière mesure par rapport à la mesure précédente.")
    with cols[1]:
        kpi_card("Variation 7 jours", _format_delta(delta_7.value), help_text=_metric_help_for_period(delta_7))
    with cols[2]:
        kpi_card("Variation 30 jours", _format_delta(delta_30.value), help_text=_metric_help_for_period(delta_30))
    with cols[3]:
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
            kpi_card("Écart objectif", _format_delta(summary["target_gap"]), help_text=f"Écart entre le poids actuel et l’objectif principal ({target_weight:.1f} kg).")
    with cols[4]:
        trend_icon = {"Baisse": "📉", "Hausse": "📈", "Stable": "➡️"}.get(summary["trend_label"], "🔎")
        kpi_card("Tendance actuelle", f"{trend_icon} {summary['trend_label']}", help_text=summary["trend_explanation"])

    sentence, tone = _trend_sentence(summary, target_weight)
    insight_card("Lecture rapide", sentence, tone=tone, icon="🔎")

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


def _render_advanced_kpis(
    df: pd.DataFrame,
    effort_df: pd.DataFrame,
    has_effort_period: bool,
    effort_days: int,
    effort_measurements: int,
    current: float,
    prev: float,
    target_weight: float,
    targets: tuple[float, ...],
    height_m: float,
    is_startup: bool,
    last_date: pd.Timestamp,
    trajectory_status: dict | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    analysis_df = effort_df if has_effort_period else df
    quality = data_quality_report(analysis_df)
    vel = weight_velocity(analysis_df, windows=(7, 14, 30))
    disc = discipline_score(analysis_df, window_days=min(effort_days, 30) if effort_days > 0 else 30)
    prog = progression_score(analysis_df, target_weight)
    streaks = streak_analysis(analysis_df)

    df_ema_kpi = compute_trend_ema(df, span=7)
    ema_current = float(df_ema_kpi["Tendance_EMA"].iloc[-1])
    ema_prev = float(df_ema_kpi["Tendance_EMA"].iloc[-2]) if len(df_ema_kpi) > 1 else ema_current
    imc = current / (height_m**2)

    long_data = df[df["Date"] >= last_date - pd.Timedelta(days=30)]["Poids (Kgs)"]
    long = float(long_data.mean()) if not long_data.empty else df["Poids (Kgs)"].mean()
    long_n = len(long_data)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Poids actuel", f"{current:.2f} kg", f"{current-prev:+.2f}")
    with c2:
        kpi_card("Tendance EMA", f"{ema_current:.2f} kg", f"{ema_current-ema_prev:+.2f}", help_text="Poids lissé — utile pour filtrer les fluctuations quotidiennes.")
    with c3:
        kpi_card("Moy. 30 derniers jours", f"{long:.2f} kg", help_text=f"Basé sur {long_n} mesure(s) des 30 derniers jours calendaires")
    with c4:
        kpi_card("IMC", f"{imc:.2f}")
    with c5:
        kpi_card("Qualité des données", f"{quality['score']}/100")

    c6, c7, c8, c9 = st.columns(4)
    with c6:
        v7 = vel.get(7)
        v_display = f"{v7:+.2f} kg/sem" if v7 is not None else "N/A"
        v_help = "Variation de poids en kg par semaine sur les 7 derniers jours"
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
        final_target = target_weight
        total = initial - final_target
        progress = ((initial - current) / total * 100) if total > 0 else 0.0
        progress_label = f"Progression vers l’objectif final ({_format_fr_kg(target_weight)})"

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
        conf_note = f" (confiance: {conf})" if conf else ""
        ms_text += f" — Date estimée : **~{milestone['eta_days']} jours**{conf_note}"
    elif milestone.get("eta_confidence") == "fragile":
        ms_text += " — Date estimée disponible après 7+ mesures"
    st.caption(ms_text)

    plateau = detect_plateau(analysis_df, window=14)
    nb_mesures_plateau = plateau.get("nb_mesures", 0)
    if nb_mesures_plateau >= STAGNATION_MIN_MEASUREMENTS:
        if plateau["status"] == "plateau probable":
            alert_banner(f"➡️ Plateau probable détecté ({nb_mesures_plateau} mesures, pente={plateau['slope']:.3f})", "warning")
        elif plateau["status"] == "baisse active":
            alert_banner(f"📉 Tendance à la baisse ({nb_mesures_plateau} mesures, pente={plateau['slope']:.3f})", "success")
        elif "reprise" in plateau["status"]:
            alert_banner(f"📈 Tendance à la hausse ({nb_mesures_plateau} mesures, pente={plateau['slope']:.3f})", "warning")
    else:
        st.caption(f"📊 Signal de tendance : données limitées ({nb_mesures_plateau} mesures sur 14j)")

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
        return "Distance à l’objectif"
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


def _render_objective_gap_chart(df: pd.DataFrame, target_weight: float) -> None:
    if df.empty:
        return
    gap_df = df[["Date", "Poids (Kgs)"]].copy()
    gap_df["Écart objectif (kg)"] = gap_df["Poids (Kgs)"] - target_weight
    fig_gap = go.Figure()
    fig_gap.add_scatter(
        x=gap_df["Date"],
        y=gap_df["Écart objectif (kg)"],
        mode="lines+markers",
        name="Écart à l'objectif",
        line=dict(color="#8E44AD", width=2),
    )
    fig_gap.add_hline(y=0, line_dash="dash", line_color="#16a34a")
    fig_gap.update_layout(xaxis_title="Date", yaxis_title="Écart (kg)")
    _apply_modern_chart_layout(fig_gap, "Écart par rapport à l'objectif final", height=340)
    st.plotly_chart(fig_gap, use_container_width=True)


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
    show_moving_average: bool = True,
    ma_window_label: str = "7 jours",
) -> tuple[go.Figure, pd.DataFrame]:
    """Build the dashboard weight chart for either full history or a zoomed period."""
    chart_df = filter_weight_period(df, start_date, end_date)
    ma_col = "MA_7J" if ma_window_label == "7 jours" else "MA_30J"
    ma_label = f"Moyenne mobile {ma_window_label}"

    full_df = df.copy()
    full_df["MA_7J"] = moving_average_by_days(full_df, 7)
    full_df["MA_30J"] = moving_average_by_days(full_df, 30)
    ma_df = filter_weight_period(full_df, start_date, end_date)

    fig = go.Figure()
    if not chart_df.empty:
        fig.add_scatter(
            x=chart_df["Date"],
            y=chart_df["Poids (Kgs)"],
            mode="lines+markers",
            name="Poids mesuré",
            line=dict(color="#2563eb", width=2.4, shape="spline", smoothing=0.35),
            marker=dict(size=5, color="#2563eb", line=dict(width=1, color="#ffffff")),
            hovertemplate="Date : %{x|%d/%m/%Y}<br>Poids mesuré : %{y:.2f} kg<extra></extra>",
        )
    if show_moving_average and not ma_df.empty:
        fig.add_scatter(
            x=ma_df["Date"],
            y=ma_df[ma_col],
            mode="lines",
            name=ma_label,
            line=dict(color="rgba(100,116,139,0.62)", width=1.35, dash="dot"),
            hovertemplate=f"Date : %{{x|%d/%m/%Y}}<br>{ma_label} : %{{y:.2f}} kg<extra></extra>",
        )

    x_values = chart_df["Date"] if not chart_df.empty else pd.DatetimeIndex(
        [pd.Timestamp(start_date), pd.Timestamp(end_date)]
        if start_date is not None and end_date is not None
        else []
    )
    _add_single_target_line(fig, target_weight, x_values)

    if show_secondary_targets:
        secondary_targets = tuple(t for t in targets if round(float(t), 3) != round(float(target_weight), 3))
        if secondary_targets:
            _add_target_lines(fig, secondary_targets, label_prefix="Objectif secondaire", x_values=x_values)

    if show_long_term_trend:
        trend_span = int(st.session_state.get("window_size", 14))
        df_ema = filter_weight_period(compute_trend_ema(df, span=trend_span), start_date, end_date)
        if not df_ema.empty:
            fig.add_scatter(
                x=df_ema["Date"],
                y=df_ema["Tendance_EMA"],
                mode="lines",
                name=f"Tendance long terme EMA ({trend_span})",
                line=dict(color="#f97316", width=2.2, dash="dot"),
                hovertemplate="Date : %{x|%d/%m/%Y}<br>Tendance long terme : %{y:.2f} kg<extra></extra>",
            )

    target_trajectory = build_target_trajectory(df, trajectory_config)
    if show_forecast and target_trajectory.get("available"):
        trajectory_df = filter_weight_period(target_trajectory["trajectory"], start_date, end_date)
        if not trajectory_df.empty:
            rate = target_trajectory["required_weekly_loss"]
            rate_label = format_fr_kg(rate, decimals=2, trim_zeros=False).replace(" kg", " kg/semaine")
            label = f"Trajectoire cible vers 80 kg au 11/11/2026 — {rate_label}"
            fig.add_scatter(
                x=trajectory_df["Date"],
                y=trajectory_df["Poids cible (kg)"],
                mode="lines",
                name=label,
                line=dict(color="#0f766e", width=2.35, dash="dashdot"),
                hovertemplate="Date : %{x|%d/%m/%Y}<br>Poids cible : %{y:.1f} kg<br>Perte moyenne requise : " + rate_label + "<extra></extra>",
            )

    if start_date is not None and end_date is not None:
        fig.update_xaxes(range=[pd.Timestamp(start_date), pd.Timestamp(end_date)])
    fig.update_layout(xaxis_title="Date", yaxis_title="Poids (kg)")
    _apply_modern_chart_layout(fig, title, height=500)
    return fig, chart_df


def _render_main_weight_chart(
    df: pd.DataFrame,
    target_weight: float,
    targets: tuple[float, ...],
    effort_df: pd.DataFrame,
    effort_start: pd.Timestamp,
    effort_days: int,
    has_effort_period: bool,
    trajectory_config: TargetTrajectoryConfig,
) -> None:
    section_header(
        "Évolution du poids",
        "Lecture principale : poids mesuré, objectifs, trajectoire cible et tendance lissée discrète.",
        "📈",
    )

    with st.expander("Options d’affichage du graphique", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            show_moving_average = st.checkbox(
                "Moyenne mobile",
                value=True,
                help="Affichée comme repère secondaire : fine, neutre et moins dominante que le poids réel.",
            )
            selected_ma = st.radio(
                "Fenêtre",
                ["7 jours", "30 jours"],
                index=0,
                horizontal=True,
                help="Une seule moyenne mobile est affichée à la fois pour préserver la lisibilité.",
            )
        with c2:
            show_secondary_targets = st.checkbox("Objectifs secondaires", value=True)
            show_long_term_trend = st.checkbox("Tendance long terme", value=False)
        with c3:
            show_forecast = st.checkbox("Trajectoire cible", value=True)
            st.caption("Les objectifs et la trajectoire font partie de la lecture par défaut ; désactivez-les ici si besoin.")
            st.caption(
                f"Paramètres trajectoire : départ {trajectory_config.start_date.strftime('%d/%m/%Y')} · "
                f"objectif {_format_fr_kg(trajectory_config.final_target_weight)} au {trajectory_config.end_date.strftime('%d/%m/%Y')}"
            )

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
    )
    st.plotly_chart(fig, use_container_width=True, key=TARGET_TRAJECTORY_CHART_KEY)
    st.caption("Vue par défaut : poids mesuré prioritaire, moyenne mobile discrète, objectifs et trajectoire cible visibles.")

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
            f"{_format_fr_kg(DEFAULT_FINAL_TARGET_WEIGHT)} le 11/11/2026."
        )
        st.caption(f"Départ : {_format_fr_kg(DEFAULT_TARGET_TRAJECTORY_START_WEIGHT)} le {DEFAULT_TARGET_TRAJECTORY_START_DATE.strftime('%d/%m/%Y')}")
        st.caption(f"Objectif : {_format_fr_kg(DEFAULT_FINAL_TARGET_WEIGHT, trim_zeros=False)} le 11/11/2026")
        st.caption(f"Durée : {TARGET_TRAJECTORY_TOTAL_DURATION_DAYS} jours")
        st.caption("Rythme moyen requis : 1,50 kg/semaine")
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

def main() -> None:
    ensure_session_defaults()
    df = _df()
    if df.empty:
        empty_state("Aucune donnée disponible. Utilisez Journal ou import CSV.")
        return

    df = df.sort_values("Date").copy()
    current = float(df["Poids (Kgs)"].iloc[-1])
    prev = float(df["Poids (Kgs)"].iloc[-2]) if len(df) > 1 else current
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
    analysis_df = effort_df if has_effort_period else df
    prog = progression_score(analysis_df, target_weight)

    page_hero(
        "Suivi de poids",
        "Dashboard",
        "Un tableau de bord allégé pour lire immédiatement le poids actuel, la tendance récente et l’écart à l’objectif.",
        meta=f"Dernière mesure : {last_date.strftime('%d/%m/%Y')} · {len(df)} mesure(s) · objectif principal {_format_fr_kg(target_weight)}",
    )

    if daily_summary.get("valid"):
        _render_daily_overview(daily_summary, target_weight, trajectory_status)
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
            f"📅 **Période d’effort actuelle** : depuis le {effort_start.strftime('%d/%m/%Y')} "
            f"({effort_days} jours, {effort_measurements} mesures) — "
            f"{delta_icon} **{delta_text}** ({_format_fr_kg(effort_initial)} → {_format_fr_kg(current)})"
        )

    _render_main_weight_chart(df, target_weight, targets, effort_df, effort_start, effort_days, has_effort_period, trajectory_config)

    tab_analysis, tab_forecast, tab_history, tab_settings = st.tabs(
        ["Analyse détaillée", "Prévisions", "Historique", "Paramètres / objectifs"]
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
                prev,
                target_weight,
                targets,
                height_m,
                is_startup,
                last_date,
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

        with st.expander("🎯 Écart à l’objectif", expanded=False):
            _render_objective_gap_chart(df, target_weight)

        vol = weight_volatility(analysis_df, window=14)
        st.caption(f"📊 Volatilité (14 derniers jours) : {vol['interpretation']} (σ={vol['std']:.2f} kg, amplitude={vol['range']:.1f} kg, {vol.get('nb_mesures', '?')} mesures)")

    with tab_forecast:
        section_header("Prévisions", "Les projections restent prudentes et sont séparées de la lecture principale.", "🔮")
        if trajectory_status.get("available"):
            st.info(
                f"🎯 Plan d’atteinte de l’objectif : {_format_fr_kg(trajectory_status['final_target_weight'])} autour du "
                f"**{trajectory_status['eta_date'].strftime('%d/%m/%Y')}**. "
                f"{_trajectory_position_sentence(trajectory_status)}"
            )
        projection = daily_summary.get("projection", {}) if daily_summary.get("valid") else {}
        if projection.get("available") and not projection.get("reached"):
            eta = projection["eta"].strftime("%d/%m/%Y")
            st.info(
                f"📍 Projection prudente : objectif vers le **{eta}** (~{projection['days_needed']} jours) "
                f"si le rythme récent ({projection['pace_kg_week']:+.2f} kg/sem) se maintient. Ce n’est pas une promesse."
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

    with tab_history:
        section_header("Historique", "Graphiques secondaires utiles, masqués par défaut pour garder la vue principale légère.", "📚")
        with st.expander("📊 Vue hebdomadaire consolidée", expanded=False):
            _render_weekly_view(df, targets)

        with st.expander("📊 Moyennes mobiles multiples (7/14/30 mesures)", expanded=False):
            df_ma = multi_rolling_averages(df, windows=(7, 14, 30))
            fig_ma = go.Figure()
            fig_ma.add_scatter(x=df_ma["Date"], y=df_ma["Poids (Kgs)"], mode="markers", name="Mesures", opacity=0.4, marker=dict(size=4))
            colors = {"MA_7m": "#2563eb", "MA_14m": "#f97316", "MA_30m": "#16a34a"}
            labels = {"MA_7m": "MA 7 mesures", "MA_14m": "MA 14 mesures", "MA_30m": "MA 30 mesures"}
            for col, color in colors.items():
                if col in df_ma.columns:
                    fig_ma.add_scatter(x=df_ma["Date"], y=df_ma[col], mode="lines", name=labels.get(col, col), line=dict(color=color, width=2))
            _add_target_lines(fig_ma, targets, label_prefix="Obj.", x_values=df_ma["Date"])
            _apply_modern_chart_layout(fig_ma, "Moyennes mobiles (glissantes sur N mesures consécutives)", height=420)
            st.plotly_chart(fig_ma, use_container_width=True)
            st.caption("ℹ️ Les moyennes mobiles glissent sur N mesures consécutives (et non sur N jours calendaires).")

        with st.expander("Distribution et évolution IMC", expanded=False):
            hist = px.histogram(df, x="Poids (Kgs)", nbins=25, title="Distribution du poids")
            _apply_modern_chart_layout(hist, "Distribution du poids", height=360)
            st.plotly_chart(hist, use_container_width=True)
            df_bmi = df.copy()
            df_bmi["IMC"] = df_bmi["Poids (Kgs)"] / (height_m**2)
            bmi_line = px.line(df_bmi, x="Date", y="IMC", title="Évolution de l’IMC")
            _apply_modern_chart_layout(bmi_line, "Évolution de l’IMC", height=360)
            st.plotly_chart(bmi_line, use_container_width=True)

    with tab_settings:
        section_header("Paramètres / objectifs", "Les objectifs secondaires existent toujours, mais ne polluent plus la lecture par défaut.", "⚙️")
        st.caption(_targets_caption(targets))
        st.caption(f"Objectif principal affiché par défaut : {target_weight:.1f} kg.")
        with st.expander("🏆 Détail du score de progression", expanded=False):
            components = prog.get("components", {})
            if components:
                comp_cols = st.columns(len(components))
                for i, (key, val) in enumerate(components.items()):
                    with comp_cols[i]:
                        st.metric(key.title(), f"{val:.0f}")
            else:
                st.caption("Score détaillé indisponible pour le moment.")

    help_box("Résumé hebdo", "Cette vue est analytique et informative; elle ne remplace pas un avis médical.")


def _render_smart_alerts(df: pd.DataFrame, analysis_df: pd.DataFrame, streaks: dict, last_date: pd.Timestamp) -> None:
    """Alertes intelligentes contextuelles (A9)."""
    # Alerte : pas de mesure récente
    days_since_last = (pd.Timestamp.now().normalize() - last_date).days
    if days_since_last >= 3:
        alert_banner(f"📏 Dernière mesure il y a {days_since_last} jours. Pensez à vous peser !", "info")

    # Alerte : série de perte en cours
    if streaks["current_streak"] >= 3 and streaks["current_type"] == "perte":
        alert_banner(f"🔥 Belle série ! {streaks['current_streak']} mesures consécutives en baisse.", "success")

    # Alerte : nouveau plus bas récent
    if len(analysis_df) >= 5:
        recent_min = float(analysis_df["Poids (Kgs)"].tail(3).min())
        older_min = float(analysis_df["Poids (Kgs)"].iloc[:-3].min()) if len(analysis_df) > 3 else recent_min + 1
        if recent_min < older_min:
            alert_banner(f"🎉 Nouveau plus bas atteint : {recent_min:.1f} kg !", "success")

    # Alerte : remontée après série de perte (rassurer)
    if streaks["current_streak"] >= 1 and streaks["current_type"] == "gain" and streaks["longest_loss"] >= 3:
        if len(analysis_df) >= 3:
            alert_banner("📊 Petite remontée après une bonne série — c'est normal, les fluctuations quotidiennes sont naturelles.", "info")


def _render_weekly_view(df: pd.DataFrame, targets: tuple) -> None:
    """Vue hebdomadaire consolidée (AN3)."""
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

    # Limiter aux 26 dernières semaines pour la lisibilité
    display_weeks = weekly.tail(26)

    # Bar chart des moyennes hebdo
    colors = ["#27AE60" if v is not None and v < -0.1 else "#E74C3C" if v is not None and v > 0.1 else "#95A5A6"
              for v in display_weeks["variation"]]
    fig_wk = go.Figure()
    fig_wk.add_bar(
        x=display_weeks["week_start"],
        y=display_weeks["poids_moyen"],
        marker_color=colors,
        hovertext=[f"Moy: {row.poids_moyen:.1f} kg | Min: {row.poids_min:.1f} | Max: {row.poids_max:.1f} | {row.nb_mesures} mes."
                   for _, row in display_weeks.iterrows()],
        hoverinfo="text",
    )
    _add_target_lines(fig_wk, targets, label_prefix="Obj.", x_values=display_weeks["week_start"])
    fig_wk.update_layout(yaxis_title="Poids moyen (kg)", showlegend=False)
    _apply_modern_chart_layout(fig_wk, "Poids moyen par semaine (vert = baisse, rouge = hausse)", height=420)
    st.plotly_chart(fig_wk, use_container_width=True)
    st.caption("ℹ️ Chaque barre = moyenne des mesures de la semaine. La couleur indique le sens de la variation par rapport à la semaine précédente.")


main()
