from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.analytics import (
    compute_trend_ema,
    detect_current_effort,
    discipline_score,
    generate_insights_text,
    multi_rolling_averages,
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

    # ── Bannière période d'effort (A1 — NOUVEAU) ───────────────────────
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
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Poids actuel", f"{current:.2f} kg", f"{current-prev:+.2f}")
    with c2:
        kpi_card("Moy. 7 derniers jours", f"{short:.2f} kg", help_text=f"Basé sur {short_n} mesure(s) des 7 derniers jours calendaires")
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
        kpi_card("Vitesse 7j", v_display, help_text="Variation de poids en kg par semaine sur les 7 derniers jours")
    with c7:
        kpi_card("Discipline", f"{disc['score']}/100", help_text=f"{disc['interpretation'].title()} — {disc['measured_days']}/{disc['expected_days']} jours mesurés")
    with c8:
        kpi_card("Score global", f"{prog['score']}/100 ({prog['grade']})", help_text="Score composite : progression, vitesse, discipline, cohérence")
    with c9:
        streak_icon = "🔥" if streaks["current_type"] == "perte" else "📈" if streaks["current_type"] == "gain" else "➡️"
        streak_txt = f"{streak_icon} {streaks['current_streak']} mesures en {streaks['current_type']}"
        kpi_card("Série en cours", streak_txt, help_text=f"Record perte : {streaks['longest_loss']} mesures · Record gain : {streaks['longest_gain']} mesures")

    # ── Barre de progression (existant — recalibrée sur effort si possible) ──
    if has_effort_period:
        effort_initial_weight = float(effort_df["Poids (Kgs)"].iloc[0])
        # Si l'effort commence au-dessus de l'objectif, progression = combien on a perdu
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

    # ── Signal de tendance cohérent (A5 — remplace le plateau contradictoire) ──
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

    # ── Alertes intelligentes contextuelles (A9 — NOUVEAU) ──────────────
    _render_smart_alerts(df, analysis_df, streaks, last_date)

    # ── Insights automatiques textuels (A2 — réécrit) ───────────────────
    st.markdown("---")
    st.subheader("💡 Insights automatiques")
    insights = generate_insights_text(df, target_weight)
    for insight in insights:
        st.markdown(f"> {insight}")

    # Accélération
    acc = weight_acceleration(analysis_df)
    if acc["interpretation"] != "données insuffisantes":
        st.caption(f"📐 Accélération : {acc['interpretation']}")

    # ── Comparaison rythme actuel vs nécessaire (A7 — NOUVEAU) ──────────
    pace = pace_comparison(analysis_df, target_weight)
    if pace.get("current_pace") is not None:
        st.caption(f"🏎️ {pace['interpretation']}")

    # ── Graphique principal avec MA + EMA (existant + A6) ──────────────
    st.markdown("---")
    ma_type = st.session_state.get("ma_type", "Simple")
    window_size = int(st.session_state.get("window_size", 7))
    df["Poids_MA"] = _moving_average(df["Poids (Kgs)"], window_size, ma_type)

    # A6: Ajouter la tendance EMA
    df_ema = compute_trend_ema(df, span=window_size)

    fig = px.line(df, x="Date", y="Poids (Kgs)", title="Évolution du poids")
    fig.add_scatter(x=df["Date"], y=df["Poids_MA"], mode="lines", name=f"MM {ma_type} ({window_size}j)",
                    line=dict(width=1.5, dash="dot"))
    fig.add_scatter(x=df_ema["Date"], y=df_ema["Tendance_EMA"], mode="lines", name=f"Tendance EMA ({window_size})",
                    line=dict(color="#E74C3C", width=2.5))
    fig.add_hline(y=float(df["Poids (Kgs)"].mean()), line_dash="dot", annotation_text="Moyenne globale")

    for idx, target in enumerate(targets, start=1):
        fig.add_hline(y=float(target), line_dash="dash", annotation_text=f"Objectif {idx}: {target:.1f} kg")

    # Marquer la période d'effort
    if has_effort_period:
        fig.add_vrect(x0=effort_start, x1=last_date, fillcolor="rgba(39,174,96,0.08)",
                      annotation_text="Effort actuel", line_width=0)

    st.plotly_chart(fig, use_container_width=True)

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

    # ── Comparaison hebdo (existant — corrigé dates calendaires) ────────
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
            comp_cols = st.columns(4)
            labels = {"progression": "Progression", "vitesse": "Vitesse", "discipline": "Discipline", "cohérence": "Cohérence"}
            max_pts = {"progression": 40, "vitesse": 25, "discipline": 20, "cohérence": 15}
            for i, (key, label) in enumerate(labels.items()):
                with comp_cols[i]:
                    val = components.get(key, 0)
                    mx = max_pts[key]
                    st.metric(label, f"{val:.0f}/{mx}")

    help_box("Résumé hebdo", "Cette vue est analytique et informative; elle ne remplace pas un avis médical.")


def _render_smart_alerts(df: pd.DataFrame, analysis_df: pd.DataFrame, streaks: dict, last_date: pd.Timestamp) -> None:
    """Alertes intelligentes contextuelles (A9)."""
    alerts_shown = False

    # Alerte : pas de mesure récente
    days_since_last = (pd.Timestamp.now().normalize() - last_date).days
    if days_since_last >= 3:
        alert_banner(f"📏 Dernière mesure il y a {days_since_last} jours. Pensez à vous peser !", "info")
        alerts_shown = True

    # Alerte : série de perte en cours
    if streaks["current_streak"] >= 3 and streaks["current_type"] == "perte":
        alert_banner(f"🔥 Belle série ! {streaks['current_streak']} mesures consécutives en baisse.", "success")
        alerts_shown = True

    # Alerte : nouveau plus bas récent
    if len(analysis_df) >= 5:
        recent_min = float(analysis_df["Poids (Kgs)"].tail(3).min())
        older_min = float(analysis_df["Poids (Kgs)"].iloc[:-3].min()) if len(analysis_df) > 3 else recent_min + 1
        if recent_min < older_min:
            alert_banner(f"🎉 Nouveau plus bas atteint : {recent_min:.1f} kg !", "success")
            alerts_shown = True

    # Alerte : remontée après série de perte (rassurer)
    if streaks["current_streak"] >= 1 and streaks["current_type"] == "gain" and streaks["longest_loss"] >= 3:
        if len(analysis_df) >= 3:
            alert_banner("📊 Petite remontée après une bonne série — c'est normal, les fluctuations quotidiennes sont naturelles.", "info")
            alerts_shown = True


main()
