"""Page Insights : qualité, phases, régularité, effets de calendrier et pesées atypiques."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    weekday_effect,
    weight_velocity,
    weight_volatility,
)
from app.core.data import data_quality_report
from app.core.formatting import format_fr_date, format_fr_kg, format_fr_kg_per_week, format_fr_number
from app.core.insights import detect_anomalies_robust, detect_plateau
from app.core.session_state import get_filtered_or_working_data
from app.core.trend import noise_level, trend_weight
from app.ui.charts import (
    ACCENT_COLOR,
    MEASURE_COLOR,
    NEUTRAL_LINE,
    STATUS_CRITICAL,
    STATUS_GOOD,
    STATUS_WARNING,
    TREND_BAND_FILL,
    TREND_COLOR,
    add_line_trace,
    add_measure_trace,
    apply_layout,
    bar_figure,
    histogram_figure,
)
from app.ui.components import empty_state, insight_card, kpi_card, page_hero, section_header

PHASE_COLORS = {"perte": STATUS_GOOD, "plateau": STATUS_WARNING, "reprise": STATUS_CRITICAL}
PHASE_ICONS = {"perte": "📉", "plateau": "➡️", "reprise": "📈"}


def _df() -> pd.DataFrame:
    return get_filtered_or_working_data()


def _metric_delta(value: float | None, unit: str = "kg", decimals: int = 2) -> str | None:
    """Delta ASCII pour ``st.metric`` : seul un « - » initial est lu comme négatif."""
    if value is None or not np.isfinite(float(value)):
        return None
    sign = "-" if value < 0 else "+" if value > 0 else ""
    return f"{sign}{format_fr_number(abs(float(value)), decimals=decimals)} {unit}".strip()


def _score_badge(score: float, *, good: float = 70, fair: float = 40) -> str:
    return "🟢" if score >= good else "🟡" if score >= fair else "🔴"


# ──────────────────────────────────────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────────────────────────────────────


def _quality_section(analysis_df: pd.DataFrame) -> None:
    section_header("Qualité des données & plateau", "Ce que valent les mesures avant d'en tirer quoi que ce soit.", "📊")
    quality = data_quality_report(analysis_df)
    q_cols = st.columns(4)
    with q_cols[0]:
        kpi_card("Score qualité", f"{_score_badge(quality['score'], good=70, fair=50)} {quality['score']}/100", help_text="Couverture, régularité, valeurs atypiques et doublons combinés.")
    with q_cols[1]:
        kpi_card("Couverture", f"{format_fr_number(quality.get('coverage_pct', 0), decimals=0)} %", help_text=f"{quality['missing_days']} jour(s) sans mesure sur la période.")
    with q_cols[2]:
        kpi_card("Régularité", f"{format_fr_number(quality['weekly_measurements'], decimals=1)}/sem", help_text=f"Irrégularité des intervalles : {format_fr_number(quality['irregularity'], decimals=2)} (0 = parfaitement régulier).")
    with q_cols[3]:
        kpi_card("Valeurs atypiques", f"{quality['anomalies']}", help_text=f"Doublons de date : {quality['duplicates']}.")
    with st.expander("Détails qualité", expanded=False):
        st.json(quality)

    plateau14 = detect_plateau(analysis_df, 14)
    plateau30 = detect_plateau(analysis_df, 30)
    p_cols = st.columns(2)
    for column, plateau, label in ((p_cols[0], plateau14, "14 derniers jours"), (p_cols[1], plateau30, "30 derniers jours")):
        with column:
            status = str(plateau["status"])
            icon = "🟠" if status == "plateau probable" else "🟢" if "baisse" in status else "🔴" if "reprise" in status else "⚪"
            kpi_card(
                f"Plateau ({label})",
                f"{icon} {status}",
                help_text=f"Pente : {format_fr_kg_per_week(plateau['slope'], decimals=3, sign=True)} sur {plateau.get('nb_mesures', '?')} mesures. Fenêtre calendaire, pas un nombre fixe de mesures.",
            )
    st.caption("Un plateau est déclaré quand la pente reste sous 0,15 kg/semaine et l'amplitude sous 0,5 kg sur la fenêtre.")


def _scores_section(analysis_df: pd.DataFrame) -> None:
    section_header("Scores & discipline", "Régularité de la saisie et cohérence des pesées.", "🎯")
    disc = discipline_score(analysis_df, window_days=30)
    cons = consistency_score(analysis_df, n_weeks=4)
    vol = weight_volatility(analysis_df, window=14)
    sc_cols = st.columns(3)
    with sc_cols[0]:
        kpi_card("Discipline (30 j)", f"{_score_badge(disc['score'])} {disc['score']}/100", help_text=f"{disc['interpretation'].title()} — {disc['measured_days']}/{disc['expected_days']} jours mesurés.")
        st.progress(disc["score"] / 100)
    with sc_cols[1]:
        kpi_card("Cohérence", f"{_score_badge(cons['score'])} {cons['score']}/100", help_text=f"{cons['interpretation'].title()} — écart-type intra-semaine moyen : {format_fr_kg(cons['avg_weekly_std'], decimals=2)}.")
        st.progress(cons["score"] / 100)
    with sc_cols[2]:
        v_badge = "🟢" if vol["cv"] < 1 else "🟡" if vol["cv"] < 2 else "🔴"
        kpi_card(
            "Volatilité (14 j)",
            f"{v_badge} {vol['interpretation'].title()}",
            help_text=f"CV = {format_fr_number(vol['cv'], decimals=2)} % · σ = {format_fr_kg(vol['std'], decimals=2)} · amplitude {format_fr_kg(vol['range'], decimals=1)} · {vol.get('nb_mesures', '?')} mesures.",
        )


def _phases_section(analysis_df: pd.DataFrame) -> None:
    section_header("Phases du parcours", "Segments de perte, de plateau et de reprise, détectés sur la pente locale.", "📈")
    phases = segment_phases(analysis_df, min_days=7)
    if not phases:
        st.info("Segmentation disponible à partir de 14 mesures sans interruption.")
        return
    rows = [
        {
            "Début": format_fr_date(p.start),
            "Fin": format_fr_date(p.end),
            "Phase": f"{PHASE_ICONS.get(p.phase_type, '❓')} {p.phase_type.title()}",
            "Durée (j)": p.duration_days,
            "Pente": format_fr_kg_per_week(p.slope, decimals=2, sign=True),
            "Poids moyen": format_fr_kg(p.mean_weight, decimals=1),
        }
        for p in phases
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    fig = go.Figure()
    add_measure_trace(fig, analysis_df["Date"], analysis_df["Poids (Kgs)"], name="Mesures", mode="markers", opacity=0.45, marker_size=4)
    seen: set[str] = set()
    for p in phases:
        mask = (analysis_df["Date"] >= p.start) & (analysis_df["Date"] <= p.end)
        phase_df = analysis_df[mask]
        if phase_df.empty:
            continue
        label = p.phase_type.title()
        fig.add_scatter(
            x=phase_df["Date"],
            y=phase_df["Poids (Kgs)"],
            mode="lines",
            name=label,
            legendgroup=label,
            showlegend=label not in seen,
            line=dict(color=PHASE_COLORS.get(p.phase_type, NEUTRAL_LINE), width=3),
            hovertemplate=f"{label} ({p.duration_days} j) : %{{y:.2f}} kg<extra></extra>",
        )
        seen.add(label)
    apply_layout(fig, "Segmentation par phases", y_title="Poids (kg)", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Vert : perte (pente < −0,14 kg/sem) · ambre : plateau · rouge : reprise. Les segments sont coupés aux interruptions de plus de 21 jours.")


def _breaks_section(analysis_df: pd.DataFrame) -> None:
    section_header("Ruptures de tendance", "Changements de régime détectés par CUSUM sur les variations quotidiennes.", "🔀")
    breaks = detect_trend_breaks(analysis_df, threshold=2.0)
    if breaks:
        for b in breaks:
            icon = "📈" if b["type"] == "reprise" else "🚀"
            st.markdown(f"- {icon} **{format_fr_date(b['date'])}** — {b['description']}")
    else:
        st.success("Aucune rupture majeure de tendance détectée.")


def _weeks_section(analysis_df: pd.DataFrame) -> None:
    section_header("Meilleures & pires semaines", "Variation entre la première et la dernière pesée de chaque semaine.", "🏆")
    bw = best_worst_weeks(analysis_df, n=5)

    def _display(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["Semaine"] = out["Semaine"].apply(lambda d: f"Semaine du {format_fr_date(d)}")
        out["Variation (kg)"] = out["Variation (kg)"].apply(lambda x: format_fr_kg(x, decimals=2, sign=True))
        out["Poids moyen"] = out["Poids moyen"].apply(lambda x: format_fr_kg(x, decimals=1))
        return out

    bw_cols = st.columns(2)
    with bw_cols[0]:
        st.markdown("**✅ Meilleures semaines** (plus grande perte)")
        if not bw["best"].empty:
            st.dataframe(_display(bw["best"]), use_container_width=True, hide_index=True)
        else:
            st.info("Pas assez de données hebdomadaires (14 mesures, deux pesées par semaine).")
    with bw_cols[1]:
        st.markdown("**❌ Pires semaines** (plus grand gain)")
        if not bw["worst"].empty:
            st.dataframe(_display(bw["worst"]), use_container_width=True, hide_index=True)
        else:
            st.info("Pas assez de données hebdomadaires (14 mesures, deux pesées par semaine).")


def _period_section(analysis_df: pd.DataFrame) -> None:
    section_header("Comparaison périodique", "Moyennes calendaires : semaine et mois en cours face aux précédents.", "📅")
    period = period_comparison(analysis_df)
    cmp_cols = st.columns(2)
    for column, key, title, unit_label in ((cmp_cols[0], "week", "Semaine courante", "hebdo"), (cmp_cols[1], "month", "Mois courant", "mensuel")):
        with column:
            data = period.get(key)
            if data:
                st.metric(
                    f"Delta {unit_label}",
                    format_fr_kg(data["current_mean"], decimals=1),
                    _metric_delta(data["delta"]),
                    delta_color="inverse",
                    help=f"{title} : {format_fr_kg(data['current_mean'], decimals=2)} ({data['current_count']} mesures) contre {format_fr_kg(data['previous_mean'], decimals=2)} ({data['previous_count']} mesures) la période précédente.",
                )
            else:
                kpi_card(f"Delta {unit_label}", "—", help_text="Données insuffisantes sur l'une des deux périodes.")
    st.caption("Base de calcul : moyennes calendaires ; une baisse s'affiche en vert.")


def _weekday_section(analysis_df: pd.DataFrame) -> None:
    section_header("Effet du jour de la semaine", "Écart moyen à la tendance pour chaque jour, testé contre les six autres.", "📆")
    effect = weekday_effect(analysis_df)
    if not effect["ready"]:
        st.info(f"Analyse disponible plus tard : {effect['reason']}.")
        dow = day_of_week_analysis(analysis_df)
        if not dow.empty:
            dow_display = dow.copy()
            dow_display["Poids moyen"] = dow_display["Poids moyen"].apply(lambda x: format_fr_kg(x, decimals=2))
            dow_display["Écart-type"] = dow_display["Écart-type"].apply(lambda x: format_fr_kg(x, decimals=2) if pd.notna(x) else "—")
            st.dataframe(dow_display, use_container_width=True, hide_index=True)
        return

    table = effect["table"]
    colors = [STATUS_CRITICAL if (pd.notna(p) and p < 0.05 and m > 0) else STATUS_GOOD if (pd.notna(p) and p < 0.05 and m < 0) else MEASURE_COLOR for m, p in zip(table["Écart moyen (kg)"], table["p ajustée"])]
    hover = [
        f"{day} : {format_fr_kg(mean, decimals=2, sign=True)} à la tendance<br>{count} mesure(s)"
        + (f"<br>p ajustée = {format_fr_number(p, decimals=3)}" if pd.notna(p) else "<br>trop peu de mesures pour tester")
        for day, mean, count, p in zip(table["Jour"], table["Écart moyen (kg)"], table["Mesures"], table["p ajustée"])
    ]
    fig = bar_figure(table["Jour"], table["Écart moyen (kg)"].fillna(0.0), "Écart moyen à la tendance par jour de la semaine", y_title="Écart (kg)", colors=colors, hover=hover, height=340)
    fig.add_hline(y=0, line=dict(color=NEUTRAL_LINE, width=1))
    st.plotly_chart(fig, use_container_width=True)

    if effect["significant"]:
        direction = "plus lourd" if effect["mean"] > 0 else "plus léger"
        insight_card(
            f"Le {effect['day'].lower()} est votre jour le plus {direction}",
            f"Écart moyen de {format_fr_kg(effect['mean'], decimals=2, sign=True)} à la tendance, et l'écart résiste à un test qui tient compte des sept jours candidats "
            f"(p ajustée = {format_fr_number(effect['p_adjusted'], decimals=3)}). Ce genre d'effet tient le plus souvent au sel, à l'alcool ou au repas du week-end : "
            "c'est de l'eau, pas de la masse grasse.",
            tone="info",
            icon="📆",
        )
    else:
        insight_card(
            "Aucun jour ne se distingue",
            f"Le {effect['day'].lower()} est le jour le plus éloigné de la tendance ({format_fr_kg(effect['mean'], decimals=2, sign=True)}), mais l'écart ne se distingue pas du hasard "
            f"(p ajustée = {format_fr_number(effect['p_adjusted'], decimals=2)}) : le plus extrême de sept jours est toujours loin de la moyenne.",
            tone="neutral",
            icon="📆",
        )
    display = table.copy()
    display["Écart moyen (kg)"] = display["Écart moyen (kg)"].apply(lambda x: format_fr_kg(x, decimals=2, sign=True) if pd.notna(x) else "—")
    display["p ajustée"] = display["p ajustée"].apply(lambda x: format_fr_number(x, decimals=3) if pd.notna(x) else "—")
    with st.expander("Voir le tableau par jour", expanded=False):
        st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Écarts calculés par rapport au poids de tendance (LOWESS) pour retirer la pente générale ; test de Welch par jour, seuil divisé par sept (Bonferroni).")


def _streaks_section(analysis_df: pd.DataFrame) -> None:
    section_header("Séries consécutives", "Nombre de pesées d'affilée dans le même sens.", "🔥")
    streaks = streak_analysis(analysis_df)
    sk_cols = st.columns(4)
    with sk_cols[0]:
        icon = "🔥" if streaks["current_type"] == "perte" else "📈" if streaks["current_type"] == "gain" else "➡️"
        kpi_card("Série actuelle", f"{icon} {streaks['current_streak']} mesures", help_text=f"Type : {streaks['current_type']}.")
    with sk_cols[1]:
        kpi_card("Record perte", f"📉 {streaks['longest_loss']} mesures")
    with sk_cols[2]:
        kpi_card("Record gain", f"📈 {streaks['longest_gain']} mesures")
    with sk_cols[3]:
        vel = weight_velocity(analysis_df, windows=(7,))
        v7 = vel.get(7)
        kpi_card("Vitesse 7 j", format_fr_kg_per_week(v7, decimals=2, sign=True) if v7 is not None else "—")
    st.caption("Une série compte les pesées, pas les jours : deux pesées séparées d'une semaine comptent pour deux.")


def _anomalies_section(df: pd.DataFrame) -> None:
    st.markdown("**Pesées atypiques**")
    use_iforest = st.toggle(
        "Ajouter IsolationForest",
        value=False,
        help="IsolationForest est réglé pour signaler 10 % des pesées quoi qu'il arrive : utile pour explorer, pas pour conclure.",
    )
    anomalies = detect_anomalies_robust(df, use_iforest=use_iforest)
    flagged = anomalies[anomalies["anomalie"]]
    if flagged.empty:
        st.success("Aucune pesée ne s'écarte anormalement de la tendance (z robuste ≤ 3,5).")
    else:
        display = flagged[["Date", "Poids (Kgs)", "ecart_tendance", "z_robuste", "raison"]].copy()
        display["Date"] = display["Date"].apply(format_fr_date)
        display["Poids (Kgs)"] = display["Poids (Kgs)"].apply(lambda x: format_fr_kg(x, decimals=2))
        display["ecart_tendance"] = display["ecart_tendance"].apply(lambda x: format_fr_kg(x, decimals=2, sign=True))
        display["z_robuste"] = display["z_robuste"].apply(lambda x: format_fr_number(x, decimals=1, sign=True))
        display = display.rename(columns={"ecart_tendance": "Écart à la tendance", "z_robuste": "z robuste", "raison": "Raison"})
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(f"{len(flagged)} pesée(s) à revoir dans le Journal : erreur de saisie, balance différente ou journée particulière.")

    fig = go.Figure()
    frame = trend_weight(df)
    if not frame.empty:
        add_line_trace(fig, frame["Date"], frame["Tendance"], name="Poids de tendance", color=TREND_COLOR, width=2.2)
    normal = anomalies[~anomalies["anomalie"]]
    add_measure_trace(fig, normal["Date"], normal["Poids (Kgs)"], name="Pesées", mode="markers", marker_size=5, opacity=0.7)
    if not flagged.empty:
        fig.add_scatter(
            x=flagged["Date"],
            y=flagged["Poids (Kgs)"],
            mode="markers",
            name="Atypiques",
            marker=dict(size=11, color=STATUS_CRITICAL, symbol="x", line=dict(width=1, color="#ffffff")),
            hovertemplate="Atypique : %{y:.2f} kg<extra></extra>",
        )
    apply_layout(fig, "Pesées atypiques face à la tendance", y_title="Poids (kg)", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Méthode : z-score robuste d'Iglewicz & Hoaglin sur l'écart à la tendance (0,6745 × écart / MAD), seuil 3,5. Comparer à la médiane globale signalait à tort les extrêmes d'une perte régulière.")


def _fluctuations_section(df: pd.DataFrame) -> None:
    st.markdown("**Fluctuations d'un jour à l'autre**")
    frame = trend_weight(df)
    noise = noise_level(df)
    if not noise["ready"] or len(frame) < 6:
        st.info("Disponible après cinq pesées.")
        return
    diffs = frame["Poids (Kgs)"].diff().dropna()
    fig = histogram_figure(diffs, "Variation entre deux pesées consécutives", x_title="Variation (kg)", nbins=25, reference=0.0, reference_label="aucune variation")
    fig.add_vrect(x0=-noise["band"], x1=noise["band"], fillcolor=TREND_BAND_FILL, line_width=0, layer="below", annotation_text="bruit habituel", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)
    share_inside = float((diffs.abs() <= noise["band"]).mean() * 100)
    cols = st.columns(3)
    with cols[0]:
        kpi_card("Bruit quotidien", f"± {format_fr_kg(noise['band'], decimals=1)}", help_text=f"σ robuste des écarts à la tendance : {format_fr_kg(noise['sigma'], decimals=2)}.")
    with cols[1]:
        kpi_card("Variation médiane", format_fr_kg(float(diffs.abs().median()), decimals=2), help_text="Valeur absolue médiane d'une variation entre deux pesées consécutives.")
    with cols[2]:
        kpi_card("Variations dans le bruit", f"{format_fr_number(share_inside, decimals=0)} %", help_text="Part des variations entre pesées consécutives plus petites que le bruit habituel.")
    st.caption(
        "L'eau, le glycogène et le contenu digestif font varier le poids de plusieurs centaines de grammes sans que la masse grasse change. "
        "Une variation dans la bande n'est pas un événement ; seule la tendance en dit quelque chose."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    df = _df()
    page_hero(
        "Analyse",
        "Insights",
        "Qualité des mesures, phases du parcours, régularité, effets de calendrier et pesées atypiques.",
        meta=f"{len(df)} mesure(s) analysée(s)" if not df.empty else "Aucune mesure",
    )
    if df.empty:
        empty_state("Pas encore de données à analyser.")
        return

    df = df.sort_values("Date").copy()
    effort = detect_current_effort(df, gap_threshold_days=21)
    has_effort = effort["is_subset"] and len(effort["effort_df"]) >= 2

    if has_effort:
        scope = st.radio(
            "Périmètre d'analyse",
            ["Effort actuel", "Historique complet"],
            horizontal=True,
            help=f"Effort actuel : depuis le {format_fr_date(effort['start_date'])} ({effort['measurements']} mesures, {effort['days']} jours).",
        )
        analysis_df = effort["effort_df"] if scope == "Effort actuel" else df
        if scope == "Effort actuel":
            st.info(
                f"📅 Calculs sur la période d'effort : {format_fr_date(effort['start_date'])} → "
                f"{format_fr_date(analysis_df['Date'].max())} ({effort['measurements']} mesures)."
            )
    else:
        analysis_df = df

    _quality_section(analysis_df)
    _scores_section(analysis_df)
    _phases_section(analysis_df)
    _breaks_section(analysis_df)
    _weeks_section(analysis_df)
    _period_section(analysis_df)
    _weekday_section(analysis_df)
    _streaks_section(analysis_df)

    section_header("Pesées atypiques & fluctuations", "Ce qui mérite une vérification, et ce qui n'est que du bruit.", "🔍")
    t1, t2 = st.tabs(["Anomalies", "Fluctuations"])
    with t1:
        _anomalies_section(df)
    with t2:
        _fluctuations_section(df)


main()
