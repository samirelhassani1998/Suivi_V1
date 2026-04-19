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
from app.core.data import data_quality_report
from app.core.insights import detect_plateau
from app.ui.components import alert_banner, confidence_badge, empty_state, help_box, kpi_card


def _df() -> pd.DataFrame:
    return st.session_state.get("filtered_data", st.session_state.get("working_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"])))


def _moving_average(series: pd.Series, window: int, ma_type: str) -> pd.Series:
    if ma_type == "Exponentielle":
        return series.ewm(span=max(window, 2), adjust=False).mean()
    return series.rolling(window=window, min_periods=1).mean()


def main() -> None:
    st.title("Dashboard")
    df = _df()
    if df.empty:
        empty_state("Aucune donnée disponible. Utilisez Journal ou import CSV.")
        return

    df = df.sort_values("Date").copy()
    current = df["Poids (Kgs)"].iloc[-1]
    prev = df["Poids (Kgs)"].iloc[-2] if len(df) > 1 else current
    last_date = df["Date"].max()

    # ── Détection période d'effort actuelle (A1/A3) ────────────────────
    effort = detect_current_effort(df, gap_threshold_days=21)
    effort_df = effort["effort_df"]
    effort_start = effort["start_date"]
    effort_days = effort["days"]
    effort_measurements = effort["measurements"]
    has_effort_period = effort["is_subset"] and len(effort_df) >= 2
    is_startup = effort_days < 7  # C3: phase de démarrage = données fragiles

    # Filtrage par dates calendaires réelles
    short_data = df[df["Date"] >= last_date - pd.Timedelta(days=7)]["Poids (Kgs)"]
    short = float(short_data.mean()) if not short_data.empty else current
    short_n = len(short_data)
    long_data = df[df["Date"] >= last_date - pd.Timedelta(days=30)]["Poids (Kgs)"]
    long = float(long_data.mean()) if not long_data.empty else df["Poids (Kgs)"].mean()
    long_n = len(long_data)

    height_m = st.session_state.get("height_m", 1.82)
    imc = current / (height_m**2)
    targets = st.session_state.get("target_weights", (95.0, 90.0, 85.0, 80.0))
    target_weight = st.session_state.get("target_weight", 80.0)

    # ── Bannière période d'effort (A1) ──────────────────────────────────
    if has_effort_period:
        effort_initial = float(effort_df["Poids (Kgs)"].iloc[0])
        effort_delta = effort_initial - current
        delta_icon = "📉" if effort_delta > 0 else "📈" if effort_delta < 0 else "➡️"
        delta_text = f"-{effort_delta:.1f}" if effort_delta > 0 else f"+{abs(effort_delta):.1f}"
        st.info(
            f"📅 **Période d'effort actuelle** : depuis le {effort_start.strftime('%d/%m/%Y')} "
            f"({effort_days} jours, {effort_measurements} mesures) — "
            f"{delta_icon} **{delta_text} kg** ({effort_initial:.1f} → {current:.1f} kg)"
        )

    # ── KPIs principaux ─────────────────────────────────────────────────
    # C4: Poids EMA lissé comme métrique de référence
    df_ema_kpi = compute_trend_ema(df, span=7)
    ema_current = float(df_ema_kpi["Tendance_EMA"].iloc[-1])
    ema_prev = float(df_ema_kpi["Tendance_EMA"].iloc[-2]) if len(df_ema_kpi) > 1 else ema_current

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Poids actuel", f"{current:.2f} kg", f"{current-prev:+.2f}")
    with c2:
        kpi_card("Tendance (EMA)", f"{ema_current:.2f} kg", f"{ema_current-ema_prev:+.2f}",
                 help_text="Poids lissé — filtre les fluctuations quotidiennes. C'est votre vrai indicateur.")
    with c3:
        kpi_card("Moy. 30 derniers jours", f"{long:.2f} kg", help_text=f"Basé sur {long_n} mesure(s) des 30 derniers jours calendaires")
    with c4:
        kpi_card("IMC", f"{imc:.2f}")
    with c5:
        quality = data_quality_report(effort_df if has_effort_period else df)
        kpi_card("Qualité des données", f"{quality['score']}/100")

    # ── KPIs avancés (calculés sur l'effort, pas l'historique global) ──
    analysis_df = effort_df if has_effort_period else df
    vel = weight_velocity(analysis_df, windows=(7, 14, 30))
    disc = discipline_score(analysis_df, window_days=min(effort_days, 30) if effort_days > 0 else 30)
    prog = progression_score(analysis_df, target_weight)
    streaks = streak_analysis(analysis_df)

    c6, c7, c8, c9 = st.columns(4)
    with c6:
        v7 = vel.get(7)
        v_display = f"{v7:+.2f} kg/sem" if v7 is not None else "N/A"
        # C3: Flag fragile si phase de démarrage
        v_help = "Variation de poids en kg par semaine sur les 7 derniers jours"
        if is_startup:
            v_display += " ⚠️"
            v_help += " — signal fragile (< 7 jours de données)"
        kpi_card("Vitesse 7j", v_display, help_text=v_help)
    with c7:
        kpi_card("Discipline", f"{disc['score']}/100", help_text=f"{disc['interpretation'].title()} — {disc['measured_days']}/{disc['expected_days']} jours mesurés")
    with c8:
        confidence_label = prog.get('confidence', 'solide')
        conf_icon = "⚠️ " if confidence_label == 'fragile' else ""
        kpi_card("Score global", f"{conf_icon}{prog['score']}/100 ({prog['grade']})",
                 help_text=f"Score composite. {'Signal fragile (< 7 mesures)' if confidence_label == 'fragile' else 'Signal fiable'}")
    with c9:
        streak_icon = "🔥" if streaks["current_type"] == "perte" else "📈" if streaks["current_type"] == "gain" else "➡️"
        streak_txt = f"{streak_icon} {streaks['current_streak']} mesures en {streaks['current_type']}"
        kpi_card("Série en cours", streak_txt, help_text=f"Record perte : {streaks['longest_loss']} mesures · Record gain : {streaks['longest_gain']} mesures")

    # ── Barre de progression (recalibrée sur effort) ────────────────────
    if has_effort_period:
        effort_initial_weight = float(effort_df["Poids (Kgs)"].iloc[0])
        if effort_initial_weight > target_weight:
            total = effort_initial_weight - target_weight
            progress = ((effort_initial_weight - current) / total * 100) if total > 0 else 0.0
        else:
            progress = 100.0
        progress_label = f"Progression effort actuel vers {target_weight:.1f} kg"
    else:
        initial = df["Poids (Kgs)"].iloc[0]
        final_target = targets[-1]
        total = initial - final_target
        progress = ((initial - current) / total * 100) if total > 0 else 0.0
        progress_label = f"Progression vers l'objectif final ({targets[-1]:.1f} kg)"

    progress = max(0.0, min(100.0, progress))
    st.progress(progress / 100)
    st.caption(f"{progress_label}: {progress:.1f}%")

    # ── Prochain milestone intelligent (AN4 — avec garde-fous C1/C3) ──
    v14 = vel.get(14)
    milestone = next_milestone(current, targets, velocity=v14, measurements=effort_measurements)
    ms_text = f"🎯 **Prochain palier** : {milestone['label']} (reste **{milestone['remaining']:.1f} kg**)"
    if milestone.get("eta_days") is not None:
        conf = milestone.get('eta_confidence', '')
        conf_note = f" (confiance: {conf})" if conf else ""
        ms_text += f" — ETA: **~{milestone['eta_days']} jours**{conf_note}"
    elif milestone.get("eta_confidence") == "fragile":
        ms_text += " — ETA: disponible après 7+ mesures"
    st.caption(ms_text)

    # ── Signal de tendance cohérent (A5) ────────────────────────────────
    plateau = detect_plateau(analysis_df, window=14)
    nb_mesures_plateau = plateau.get("nb_mesures", 0)

    if nb_mesures_plateau >= 3:
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

    # ── Alertes intelligentes contextuelles (A9) ────────────────────────
    _render_smart_alerts(df, analysis_df, streaks, last_date)

    # ── Insight pattern yo-yo historique (AN2 — NOUVEAU) ────────────────
    history = analyze_effort_history(df)
    if history.get("insight"):
        st.warning(history["insight"])
        if history.get("best_effort"):
            best = history["best_effort"]
            st.caption(
                f"💪 Meilleur effort passé : **-{best['delta']:.1f} kg** en {best['days']} jours "
                f"({best['start_date'].strftime('%d/%m/%Y')} → {best['end_date'].strftime('%d/%m/%Y')}, {best['measurements']} mesures)"
            )

    # ── C2: Résumé actionnable Situation / Interprétation / Action ──────
    st.markdown("---")
    summary = generate_action_summary(df, target_weight)
    st.subheader("🧭 Votre résumé")
    st.markdown(f"📍 **Situation** : {summary['situation']}")
    st.markdown(f"🔍 **Interprétation** : {summary['interpretation']}")
    st.markdown(f"▶️ **Action** : {summary['action']}")

    # ── Insights détaillés (dans un expander) ───────────────────────────
    with st.expander("💡 Insights détaillés", expanded=False):
        insights = generate_insights_text(df, target_weight)
        for insight in insights:
            st.markdown(f"> {insight}")

        # Accélération
        acc = weight_acceleration(analysis_df)
        if acc["interpretation"] != "données insuffisantes":
            st.caption(f"📐 Accélération : {acc['interpretation']}")

        # Comparaison rythme actuel vs nécessaire (A7) — seulement si effort établi
        if not is_startup:
            pace = pace_comparison(analysis_df, target_weight)
            if pace.get("current_pace") is not None:
                st.caption(f"🏎️ {pace['interpretation']}")
        else:
            st.caption("🏎️ Comparaison de rythme : disponible après 7+ jours de suivi.")

    # ── Graphique principal avec MA + EMA + trajectoire cible (AN1 — NOUVEAU) ──
    st.markdown("---")
    ma_type = st.session_state.get("ma_type", "Simple")
    window_size = int(st.session_state.get("window_size", 7))
    df["Poids_MA"] = _moving_average(df["Poids (Kgs)"], window_size, ma_type)

    df_ema = compute_trend_ema(df, span=window_size)

    fig = px.line(df, x="Date", y="Poids (Kgs)", title="Évolution du poids")
    fig.add_scatter(x=df["Date"], y=df["Poids_MA"], mode="lines", name=f"MM {ma_type} ({window_size}j)",
                    line=dict(width=1.5, dash="dot"))
    fig.add_scatter(x=df_ema["Date"], y=df_ema["Tendance_EMA"], mode="lines", name=f"Tendance EMA ({window_size})",
                    line=dict(color="#E74C3C", width=2.5))
    fig.add_hline(y=float(df["Poids (Kgs)"].mean()), line_dash="dot", annotation_text="Moyenne globale")

    for idx, target in enumerate(targets, start=1):
        fig.add_hline(y=float(target), line_dash="dash", annotation_text=f"Objectif {idx}: {target:.1f} kg")

    # AN1: Trajectoire cible depuis début de l'effort (pente saine = -0.5 kg/sem)
    if has_effort_period:
        effort_initial_w = float(effort_df["Poids (Kgs)"].iloc[0])
        healthy_rate = -0.5 / 7  # kg/jour
        # Tracer sur 180 jours max depuis début effort
        traj_dates = pd.date_range(effort_start, periods=min(180, max(effort_days * 3, 90)), freq="D")
        traj_values = [effort_initial_w + healthy_rate * i for i in range(len(traj_dates))]
        # Ne pas descendre en dessous de l'objectif final
        traj_values = [max(v, target_weight) for v in traj_values]

        fig.add_scatter(x=traj_dates, y=traj_values, mode="lines",
                        name="Trajectoire cible (-0.5 kg/sem)",
                        line=dict(color="#27AE60", width=2, dash="dashdot"))

        # Corridor ±1 kg
        traj_upper = [v + 1.0 for v in traj_values]
        traj_lower = [max(v - 1.0, target_weight) for v in traj_values]
        fig.add_scatter(x=traj_dates, y=traj_upper, mode="lines",
                        line=dict(width=0), showlegend=False)
        fig.add_scatter(x=traj_dates, y=traj_lower, mode="lines",
                        fill="tonexty", fillcolor="rgba(39,174,96,0.08)",
                        line=dict(width=0), name="Corridor cible (±1 kg)")

        # Zone effort
        fig.add_vrect(x0=effort_start, x1=last_date, fillcolor="rgba(39,174,96,0.05)",
                      annotation_text="Effort actuel", line_width=0)

    st.plotly_chart(fig, use_container_width=True)

    # ── Vue hebdomadaire consolidée (AN3 — NOUVEAU) ─────────────────────
    with st.expander("📊 Vue hebdomadaire consolidée", expanded=False):
        _render_weekly_view(df, targets)

    # ── Graphique multi-MA (existant) ───────────────────────────────────
    with st.expander("📊 Moyennes mobiles multiples (7/14/30 mesures)", expanded=False):
        df_ma = multi_rolling_averages(df, windows=(7, 14, 30))
        fig_ma = go.Figure()
        fig_ma.add_scatter(x=df_ma["Date"], y=df_ma["Poids (Kgs)"], mode="markers", name="Mesures", opacity=0.4, marker=dict(size=4))
        colors = {"MA_7m": "#2E7BCF", "MA_14m": "#E67E22", "MA_30m": "#27AE60"}
        labels = {"MA_7m": "MA 7 mesures", "MA_14m": "MA 14 mesures", "MA_30m": "MA 30 mesures"}
        for col, color in colors.items():
            if col in df_ma.columns:
                fig_ma.add_scatter(x=df_ma["Date"], y=df_ma[col], mode="lines", name=labels.get(col, col), line=dict(color=color, width=2))
        for idx, target in enumerate(targets, start=1):
            fig_ma.add_hline(y=float(target), line_dash="dash", annotation_text=f"Obj. {idx}")
        fig_ma.update_layout(title="Moyennes mobiles (glissantes sur N mesures consécutives)", hovermode="x unified")
        st.plotly_chart(fig_ma, use_container_width=True)
        st.caption("ℹ️ Les moyennes mobiles glissent sur N mesures consécutives, pas sur N jours calendaires.")

    # ── Comparaison hebdo (existant) ────────────────────────────────────
    wk_data = df[df["Date"] >= last_date - pd.Timedelta(days=7)]["Poids (Kgs)"]
    prev_wk_data = df[(df["Date"] >= last_date - pd.Timedelta(days=14)) & (df["Date"] < last_date - pd.Timedelta(days=7))]["Poids (Kgs)"]
    wk = float(wk_data.mean()) if not wk_data.empty else current
    prev_wk = float(prev_wk_data.mean()) if not prev_wk_data.empty else wk
    delta_wk = wk - prev_wk
    st.metric("Comparaison hebdo (7j calendaires)", f"{wk:.2f} kg", f"{delta_wk:+.2f} kg", delta_color="inverse",
             help=f"Semaine courante: {len(wk_data)} mesure(s) · Semaine précédente: {len(prev_wk_data)} mesure(s)")

    # ── Volatilité (existant) ───────────────────────────────────────────
    vol = weight_volatility(analysis_df, window=14)
    st.caption(f"📊 Volatilité (14 derniers jours) : {vol['interpretation']} (σ={vol['std']:.2f} kg, amplitude={vol['range']:.1f} kg, {vol.get('nb_mesures', '?')} mesures)")

    # ── Distribution et IMC (existant) ──────────────────────────────────
    with st.expander("Distribution et évolution IMC"):
        hist = px.histogram(df, x="Poids (Kgs)", nbins=25, title="Distribution du poids")
        st.plotly_chart(hist, use_container_width=True)
        df["IMC"] = df["Poids (Kgs)"] / (height_m**2)
        bmi_line = px.line(df, x="Date", y="IMC", title="Évolution de l'IMC")
        st.plotly_chart(bmi_line, use_container_width=True)

    # ── Score de progression détaillé (existant) ────────────────────────
    with st.expander("🏆 Détail du score de progression"):
        components = prog.get("components", {})
        if components:
            comp_cols = st.columns(len(components))
            for i, (key, val) in enumerate(components.items()):
                with comp_cols[i]:
                    st.metric(key.title(), f"{val:.0f}")

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
        text=[f"{w:.1f}" for w in display_weeks["poids_moyen"]],
        textposition="outside",
        hovertext=[f"Moy: {row.poids_moyen:.1f} kg | Min: {row.poids_min:.1f} | Max: {row.poids_max:.1f} | {row.nb_mesures} mes."
                   for _, row in display_weeks.iterrows()],
        hoverinfo="text",
    )
    for idx, target in enumerate(targets, start=1):
        fig_wk.add_hline(y=float(target), line_dash="dash", annotation_text=f"Obj. {idx}")
    fig_wk.update_layout(
        title="Poids moyen par semaine (vert = baisse, rouge = hausse)",
        yaxis_title="Poids moyen (kg)",
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig_wk, use_container_width=True)
    st.caption("ℹ️ Chaque barre = moyenne des mesures de la semaine. La couleur indique le sens de la variation par rapport à la semaine précédente.")


main()
