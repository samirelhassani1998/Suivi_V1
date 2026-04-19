from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

from app.core.analytics import (
    best_worst_weeks,
    consistency_score,
    day_of_week_analysis,
    detect_current_effort,
    detect_trend_breaks,
    discipline_score,
    period_comparison,
    segment_phases,
    streak_analysis,
    weight_velocity,
    weight_volatility,
)
from app.core.data import data_quality_report
from app.core.insights import detect_anomalies_robust, detect_plateau
from app.ui.components import empty_state, kpi_card


def _df() -> pd.DataFrame:
    return st.session_state.get("filtered_data", st.session_state.get("working_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"])))


def main() -> None:
    st.title("Insights / Analyse")
    df = _df()
    if df.empty:
        empty_state("Pas encore de données à analyser.")
        return

    df = df.sort_values("Date").copy()

    # ── Toggle effort / historique (UX1 — NOUVEAU) ──────────────────────
    effort = detect_current_effort(df, gap_threshold_days=21)
    has_effort = effort["is_subset"] and len(effort["effort_df"]) >= 2

    if has_effort:
        scope = st.radio(
            "📊 Périmètre d'analyse",
            ["Effort actuel", "Historique complet"],
            horizontal=True,
            help=f"Effort actuel : depuis le {effort['start_date'].strftime('%d/%m/%Y')} ({effort['measurements']} mesures, {effort['days']} jours)",
        )
        analysis_df = effort["effort_df"] if scope == "Effort actuel" else df
        if scope == "Effort actuel":
            st.info(
                f"📅 Calculs sur la période d'effort : {effort['start_date'].strftime('%d/%m/%Y')} → "
                f"{analysis_df['Date'].max().strftime('%d/%m/%Y')} ({effort['measurements']} mesures)"
            )
    else:
        analysis_df = df

    # ══════════════════════════════════════════════════════════════════════
    # Section 1 : Qualité et Plateau (EXISTANT — amélioré visuellement)
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("📊 Qualité des données & Plateau")
    quality = data_quality_report(analysis_df)

    # Remplacer le JSON brut par des cards visuelles
    q_cols = st.columns(4)
    with q_cols[0]:
        score = quality["score"]
        color = "🟢" if score > 70 else "🟡" if score > 50 else "🔴"
        kpi_card("Score qualité", f"{color} {score}/100")
    with q_cols[1]:
        cov = quality.get("coverage_pct", 0)
        kpi_card("Couverture", f"{cov:.0f}%", help_text=f"{quality['missing_days']} jours manquants")
    with q_cols[2]:
        kpi_card("Régularité", f"{quality['weekly_measurements']:.1f}/sem", help_text=f"Irrégularité: {quality['irregularity']:.2f}")
    with q_cols[3]:
        kpi_card("Anomalies", f"{quality['anomalies']}", help_text=f"Doublons: {quality['duplicates']}")

    with st.expander("📋 Détails qualité (JSON)", expanded=False):
        st.json(quality)

    plateau14 = detect_plateau(analysis_df, 14)
    plateau30 = detect_plateau(analysis_df, 30)

    p_cols = st.columns(2)
    with p_cols[0]:
        p14_icon = "🟠" if plateau14["status"] == "plateau probable" else "🟢" if "baisse" in plateau14["status"] else "🔴" if "reprise" in plateau14["status"] else "⚪"
        nb14 = plateau14.get("nb_mesures", "?")
        st.metric("Plateau (14 derniers jours)", f"{p14_icon} {plateau14['status']}", f"Pente: {plateau14['slope']:.3f} ({nb14} mesures)")
    with p_cols[1]:
        p30_icon = "🟠" if plateau30["status"] == "plateau probable" else "🟢" if "baisse" in plateau30["status"] else "🔴" if "reprise" in plateau30["status"] else "⚪"
        nb30 = plateau30.get("nb_mesures", "?")
        st.metric("Plateau (30 derniers jours)", f"{p30_icon} {plateau30['status']}", f"Pente: {plateau30['slope']:.3f} ({nb30} mesures)")

    # ══════════════════════════════════════════════════════════════════════
    # Section 2 : Scores avancés (NOUVEAU)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🎯 Scores & Discipline")

    disc = discipline_score(analysis_df, window_days=30)
    cons = consistency_score(analysis_df, n_weeks=4)
    vol = weight_volatility(analysis_df, window=14)

    sc_cols = st.columns(3)
    with sc_cols[0]:
        d_color = "🟢" if disc["score"] >= 70 else "🟡" if disc["score"] >= 40 else "🔴"
        st.metric(
            "🏅 Discipline (30j)",
            f"{d_color} {disc['score']}/100",
            help=f"{disc['interpretation'].title()} — {disc['measured_days']}/{disc['expected_days']} jours",
        )
        st.progress(disc["score"] / 100)
    with sc_cols[1]:
        c_color = "🟢" if cons["score"] >= 70 else "🟡" if cons["score"] >= 40 else "🔴"
        st.metric(
            "🔄 Cohérence",
            f"{c_color} {cons['score']}/100",
            help=f"{cons['interpretation'].title()} — σ hebdo moyen: {cons['avg_weekly_std']:.2f} kg",
        )
        st.progress(cons["score"] / 100)
    with sc_cols[2]:
        v_color = "🟢" if vol["cv"] < 1 else "🟡" if vol["cv"] < 2 else "🔴"
        nb_vol = vol.get("nb_mesures", "?")
        st.metric(
            "📊 Volatilité (14 derniers jours)",
            f"{v_color} {vol['interpretation'].title()}",
            help=f"CV={vol['cv']:.2f}% · σ={vol['std']:.2f} kg · Amplitude: {vol['range']:.1f} kg · {nb_vol} mesures",
        )

    # ══════════════════════════════════════════════════════════════════════
    # Section 3 : Phases du parcours (NOUVEAU)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📈 Phases du parcours")

    phases = segment_phases(analysis_df, min_days=7)
    if phases:
        phase_data = []
        phase_colors = {"perte": "#27AE60", "plateau": "#F39C12", "reprise": "#E74C3C"}
        phase_icons = {"perte": "📉", "plateau": "➡️", "reprise": "📈"}
        for p in phases:
            phase_data.append({
                "Début": p.start.strftime("%d/%m/%Y"),
                "Fin": p.end.strftime("%d/%m/%Y"),
                "Phase": f"{phase_icons.get(p.phase_type, '❓')} {p.phase_type.title()}",
                "Durée (j)": p.duration_days,
                "Pente (kg/j)": f"{p.slope:+.4f}",
                "Poids moyen": f"{p.mean_weight:.1f} kg",
            })
        st.dataframe(pd.DataFrame(phase_data), use_container_width=True, hide_index=True)

        # Timeline visuelle des phases
        fig_phases = go.Figure()
        fig_phases.add_scatter(x=analysis_df["Date"], y=analysis_df["Poids (Kgs)"], mode="markers", name="Mesures", marker=dict(size=3, color="#999"), showlegend=False)
        for p in phases:
            mask = (analysis_df["Date"] >= p.start) & (analysis_df["Date"] <= p.end)
            phase_df = analysis_df[mask]
            if not phase_df.empty:
                fig_phases.add_scatter(
                    x=phase_df["Date"], y=phase_df["Poids (Kgs)"],
                    mode="lines", name=f"{p.phase_type.title()} ({p.duration_days}j)",
                    line=dict(color=phase_colors.get(p.phase_type, "#999"), width=3),
                )
        fig_phases.update_layout(title="Segmentation par phases", hovermode="x unified")
        st.plotly_chart(fig_phases, use_container_width=True)
    else:
        st.info("Pas assez de données pour segmenter en phases.")

    # ══════════════════════════════════════════════════════════════════════
    # Section 4 : Ruptures de tendance (NOUVEAU)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔀 Ruptures de tendance")

    breaks = detect_trend_breaks(analysis_df, threshold=2.0)
    if breaks:
        for b in breaks:
            icon = "📈" if b["type"] == "reprise" else "🚀"
            st.markdown(f"- {icon} **{b['date'].strftime('%d/%m/%Y')}** — {b['description']}")
    else:
        st.success("Aucune rupture majeure de tendance détectée.")

    # ══════════════════════════════════════════════════════════════════════
    # Section 5 : Meilleures / Pires semaines (NOUVEAU)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏆 Meilleures & Pires semaines")

    bw = best_worst_weeks(analysis_df, n=5)
    bw_cols = st.columns(2)
    with bw_cols[0]:
        st.markdown("**✅ Meilleures semaines** (plus grande perte)")
        if not bw["best"].empty:
            display_best = bw["best"].copy()
            display_best["Semaine"] = display_best["Semaine"].dt.strftime("%d/%m/%Y")
            display_best["Variation (kg)"] = display_best["Variation (kg)"].apply(lambda x: f"{x:+.2f}")
            display_best["Poids moyen"] = display_best["Poids moyen"].apply(lambda x: f"{x:.1f}")
            st.dataframe(display_best, use_container_width=True, hide_index=True)
        else:
            st.info("Pas assez de données hebdomadaires.")
    with bw_cols[1]:
        st.markdown("**❌ Pires semaines** (plus grand gain)")
        if not bw["worst"].empty:
            display_worst = bw["worst"].copy()
            display_worst["Semaine"] = display_worst["Semaine"].dt.strftime("%d/%m/%Y")
            display_worst["Variation (kg)"] = display_worst["Variation (kg)"].apply(lambda x: f"{x:+.2f}")
            display_worst["Poids moyen"] = display_worst["Poids moyen"].apply(lambda x: f"{x:.1f}")
            st.dataframe(display_worst, use_container_width=True, hide_index=True)
        else:
            st.info("Pas assez de données hebdomadaires.")

    # ══════════════════════════════════════════════════════════════════════
    # Section 6 : Comparaison périodique (NOUVEAU)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📅 Comparaison périodique")

    period = period_comparison(analysis_df)
    cmp_cols = st.columns(2)
    with cmp_cols[0]:
        st.markdown("**Semaine courante vs précédente**")
        if period["week"]:
            w = period["week"]
            delta = w["delta"]
            icon = "📉" if delta < -0.1 else "📈" if delta > 0.1 else "➡️"
            st.metric("Delta hebdo", f"{icon} {delta:+.2f} kg",
                      f"{w['current_mean']:.1f} vs {w['previous_mean']:.1f} kg")
        else:
            st.info("Données insuffisantes")
    with cmp_cols[1]:
        st.markdown("**Mois courant vs précédent**")
        if period["month"]:
            m = period["month"]
            delta = m["delta"]
            icon = "📉" if delta < -0.1 else "📈" if delta > 0.1 else "➡️"
            st.metric("Delta mensuel", f"{icon} {delta:+.2f} kg",
                      f"{m['current_mean']:.1f} vs {m['previous_mean']:.1f} kg")
        else:
            st.info("Données insuffisantes")

    # ══════════════════════════════════════════════════════════════════════
    # Section 7 : Patterns par jour de la semaine (NOUVEAU)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📆 Patterns par jour de la semaine")

    dow = day_of_week_analysis(analysis_df)
    if not dow.empty:
        dow_display = dow.copy()
        dow_display["Poids moyen"] = dow_display["Poids moyen"].apply(lambda x: f"{x:.2f}")
        dow_display["Écart-type"] = dow_display["Écart-type"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        st.dataframe(dow_display, use_container_width=True, hide_index=True)

        fig_dow = px.bar(dow, x="Jour", y="Poids moyen", title="Poids moyen par jour de la semaine",
                         color="Poids moyen", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig_dow, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # Section 8 : Streaks (NOUVEAU)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔥 Séries consécutives (Streaks)")

    streaks = streak_analysis(analysis_df)
    sk_cols = st.columns(4)
    with sk_cols[0]:
        icon = "🔥" if streaks["current_type"] == "perte" else "📈" if streaks["current_type"] == "gain" else "➡️"
        st.metric("Série actuelle", f"{icon} {streaks['current_streak']} mesures", streaks["current_type"])
    with sk_cols[1]:
        st.metric("Record perte", f"📉 {streaks['longest_loss']} mesures")
    with sk_cols[2]:
        st.metric("Record gain", f"📈 {streaks['longest_gain']} mesures")
    with sk_cols[3]:
        vel = weight_velocity(analysis_df, windows=(7,))
        v7 = vel.get(7)
        st.metric("Vitesse 7j", f"{v7:+.2f} kg/sem" if v7 is not None else "N/A")

    # ══════════════════════════════════════════════════════════════════════
    # Section 9 : Anomalies & Clustering (EXISTANT — préservé intégralement)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    t1, t2 = st.tabs(["Anomalies", "Clustering"])
    with t1:
        anomalies = detect_anomalies_robust(df, use_iforest=st.toggle("Activer IsolationForest", value=True))
        st.dataframe(anomalies[["Date", "Poids (Kgs)", "anomalie", "raison", "decision"]], use_container_width=True)
        fig_anom = px.scatter(anomalies, x="Date", y="Poids (Kgs)", color="anomalie", title="Anomalies détectées")
        st.plotly_chart(fig_anom, use_container_width=True)

    with t2:
        max_clusters = min(6, len(df))
        if max_clusters < 2:
            st.warning("Données insuffisantes pour KMeans.")
        else:
            k = st.slider("Nombre de clusters", 2, max_clusters, 3)
            clustered = df.copy()
            clustered["cluster"] = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(clustered[["Poids (Kgs)"]])
            fig_cluster = px.scatter(clustered, x="Date", y="Poids (Kgs)", color="cluster", title="Clusters KMeans")
            st.plotly_chart(fig_cluster, use_container_width=True)


main()
