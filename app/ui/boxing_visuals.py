"""Figures Plotly de l'onglet Boxe.

Même contrat que :mod:`app.ui.whoop_visuals` : aucune dépendance à Streamlit,
chaque fonction renvoie une ``go.Figure`` ou ``None`` quand rien n'est
exploitable, ce qui rend les règles de lisibilité vérifiables par des tests.

Couleurs : la zone de récupération du matin est une signification (bon /
moyen / mauvais), elle prend donc la palette de statut et toujours son nom en
légende ; une série unique prend le premier emplacement catégoriel ; les
intensités reprennent le dégradé facile → dur de l'onglet WHOOP.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.core.boxing_analytics import LATE_SESSION_HOURS, sleep_nights
from app.core.date_labels import format_clock_hour, format_duration_minutes, format_long_date
from app.core.whoop_analytics import INTENSITY_BANDS
from app.ui.whoop_visuals import (
    AXIS,
    INK_MUTED,
    INK_SECONDARY,
    INTENSITY_COLORS,
    LINE_WIDTH,
    MARKER_SIZE,
    RECOVERY_BANDS,
    SERIES_COLORS,
    STRAIN_BANDS,
    SURFACE,
    ZONE_COLORS,
    _add_bands,
    _base_layout,
    _calendar_axis,
)

NO_ZONE_COLOR = "#b5b4ad"
# Largeur de barre fixée à 70 % d'une journée : laissée à Plotly, elle se
# calcule trace par trace, et une zone à deux séances espacées d'un mois
# s'étalait sur plusieurs semaines.
DAY_BAR_WIDTH_MS = 0.7 * 24 * 3600 * 1000
ZONE_ORDER: tuple[tuple[str, str], ...] = (
    ("Vert", "Matin vert"),
    ("Jaune", "Matin jaune"),
    ("Rouge", "Matin rouge"),
    ("—", "Matin non noté"),
)
# Dégradé séquentiel à une teinte (bleu, du clair au foncé) pour des effectifs.
SEQUENTIAL_BLUES: tuple[str, ...] = ("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95")


def _fr(value: Any, decimals: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    return f"{numeric:.{decimals}f}".replace(".", ",").replace("-", "−")


def _session_hover(row: pd.Series) -> str:
    parts = [format_long_date(row["Date"])]
    start = row.get("Début")
    if start is not None and pd.notna(start):
        parts[0] += f" à {pd.Timestamp(start).strftime('%H:%M')}"
    parts.append(f"Strain {_fr(row.get('Strain séance'))} · {format_duration_minutes(row.get('Durée (min)'))}")
    if np.isfinite(float(row.get("Intensité (% FCR)", np.nan))):
        parts.append(f"Intensité : {_fr(row.get('Intensité (% FCR)'), 0)} % de réserve cardiaque")
    morning = float(row.get("Récupération du matin (%)", np.nan))
    following = float(row.get("Récupération du lendemain (%)", np.nan))
    parts.append(f"Matin : {_fr(morning, 0)} %" if np.isfinite(morning) else "Matin : non noté")
    parts.append(f"Lendemain : {_fr(following, 0)} %" if np.isfinite(following) else "Lendemain : non noté")
    return "<br>".join(parts)


def boxing_sessions_chart(table: pd.DataFrame | None) -> go.Figure | None:
    """Chaque séance à sa date, hauteur = strain, couleur = zone du matin.

    La couleur dit dans quel état vous êtes monté sur le ring ; le survol
    donne la récupération du lendemain, ce que la séance a laissé.
    """
    if table is None or table.empty or "Strain séance" not in table.columns:
        return None
    usable = table.dropna(subset=["Strain séance"])
    if usable.empty:
        return None
    figure = go.Figure()
    _add_bands(figure, STRAIN_BANDS)
    zones = usable["Zone du matin"].fillna("—") if "Zone du matin" in usable.columns else pd.Series("—", index=usable.index)
    for zone, label in ZONE_ORDER:
        chunk = usable[zones == zone]
        if chunk.empty:
            continue
        figure.add_bar(
            x=chunk["Date"],
            y=chunk["Strain séance"],
            name=label,
            width=DAY_BAR_WIDTH_MS,
            marker=dict(color=ZONE_COLORS.get(zone, NO_ZONE_COLOR), line=dict(width=2, color=SURFACE)),
            customdata=[_session_hover(row) for _, row in chunk.iterrows()],
            hovertemplate="%{customdata}<extra></extra>",
        )
    figure = _base_layout(figure, "Séances de boxe, colorées selon la récupération du matin", y_title="Strain de la séance", show_legend=len(figure.data) > 1)
    figure.update_layout(barmode="overlay", bargap=0.35, hovermode="closest")
    figure.update_yaxes(range=[0, 21])
    return _calendar_axis(figure, usable["Date"])


def weekly_frequency_chart(weekly: pd.DataFrame | None) -> go.Figure | None:
    """Séances par semaine, semaines vides comprises : la régularité se voit aux trous."""
    if weekly is None or weekly.empty or "Séances" not in weekly.columns:
        return None
    hover = [
        f"Semaine du {format_long_date(week, with_weekday=False)}<br>{int(count)} séance(s) · {format_duration_minutes(minutes)}"
        + (f"<br>TRIMP {_fr(trimp, 0)}" if np.isfinite(float(trimp)) and count else "")
        for week, count, minutes, trimp in zip(weekly["Semaine"], weekly["Séances"], weekly["Durée totale (min)"], weekly["TRIMP"])
    ]
    figure = go.Figure(
        go.Bar(
            x=weekly["Semaine"],
            y=weekly["Séances"],
            name="Séances",
            marker=dict(color=SERIES_COLORS[0], line=dict(width=2, color=SURFACE)),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    figure = _base_layout(figure, "Séances par semaine", y_title="Séances", show_legend=False)
    figure.update_layout(bargap=0.35, hovermode="closest")
    figure.update_yaxes(dtick=1, rangemode="tozero")
    return _calendar_axis(figure, weekly["Semaine"])


def recovery_profile_chart(profile: pd.DataFrame | None, overall: float) -> go.Figure | None:
    """Récupération moyenne du matin de la séance à J+3, face à votre moyenne générale."""
    if profile is None or profile.empty:
        return None
    lows = profile["IC 95 % bas"].to_numpy(dtype=float)
    highs = profile["IC 95 % haut"].to_numpy(dtype=float)
    means = profile["Récupération moyenne (%)"].to_numpy(dtype=float)
    figure = go.Figure()
    _add_bands(figure, RECOVERY_BANDS)
    figure.add_scatter(
        x=profile["Jour"],
        y=means,
        mode="lines+markers",
        name="Après une séance",
        connectgaps=False,
        line=dict(color=SERIES_COLORS[0], width=LINE_WIDTH),
        marker=dict(size=MARKER_SIZE + 2, line=dict(width=2, color=SURFACE)),
        error_y=dict(
            type="data",
            symmetric=False,
            array=np.where(np.isfinite(highs), highs - means, np.nan),
            arrayminus=np.where(np.isfinite(lows), means - lows, np.nan),
            color=INK_MUTED,
            thickness=1.5,
            width=6,
        ),
        customdata=profile["Observations"],
        hovertemplate="%{x} : %{y:.0f} %<br>%{customdata} séance(s)<extra></extra>",
    )
    if np.isfinite(overall):
        figure.add_hline(
            y=float(overall),
            line=dict(color=INK_SECONDARY, width=1.5),
            annotation_text=f"Votre moyenne : {_fr(overall, 0)} %",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color=INK_SECONDARY),
        )
    figure = _base_layout(figure, "Récupération après une séance de boxe", y_title="Récupération (%)", show_legend=False)
    figure.update_layout(hovermode="closest")
    figure.update_yaxes(range=[0, 100])
    return figure


def readiness_scatter(table: pd.DataFrame | None) -> go.Figure | None:
    """Récupération du matin (avant la séance) en abscisse, strain de la séance en ordonnée."""
    if table is None or table.empty:
        return None
    usable = table.dropna(subset=["Récupération du matin (%)", "Strain séance"])
    if usable.empty:
        return None
    figure = go.Figure()
    for low, high, color, label in RECOVERY_BANDS:
        figure.add_vrect(x0=low, x1=high, fillcolor=color, line_width=0, layer="below", annotation_text=label, annotation_position="top left", annotation_font=dict(size=10, color=INK_MUTED))
    for zone, label in ZONE_ORDER[:3]:
        chunk = usable[usable["Zone du matin"] == zone]
        if chunk.empty:
            continue
        figure.add_scatter(
            x=chunk["Récupération du matin (%)"],
            y=chunk["Strain séance"],
            mode="markers",
            name=label,
            marker=dict(size=MARKER_SIZE + 3, color=ZONE_COLORS[zone], line=dict(width=2, color=SURFACE)),
            customdata=[_session_hover(row) for _, row in chunk.iterrows()],
            hovertemplate="%{customdata}<extra></extra>",
        )
    figure = _base_layout(figure, "État du matin et intensité de la séance", y_title="Strain de la séance", show_legend=len(figure.data) > 1)
    figure.update_layout(hovermode="closest")
    figure.update_xaxes(range=[0, 100], title="Récupération du matin (%)", ticksuffix=" %")
    return figure


def late_sessions_chart(table: pd.DataFrame | None, *, cutoff: float = LATE_SESSION_HOURS) -> go.Figure | None:
    """Marge entre fin de séance et coucher habituel, face à la nuit qui a suivi."""
    if table is None or table.empty:
        return None
    usable = sleep_nights(table).dropna(subset=["Marge avant coucher habituel (h)", "Sommeil suivant (heures)"])
    if usable.empty:
        return None
    hover = [
        f"{format_long_date(date)}<br>Dernière fin de séance {_fr(margin)} h avant votre coucher habituel"
        f"<br>Nuit suivante : {_fr(sleep)} h, coucher à {format_clock_hour(bedtime)}"
        for date, margin, sleep, bedtime in zip(usable["Date"], usable["Marge avant coucher habituel (h)"], usable["Sommeil suivant (heures)"], usable["Coucher suivant"])
    ]
    figure = go.Figure(
        go.Scatter(
            x=usable["Marge avant coucher habituel (h)"],
            y=usable["Sommeil suivant (heures)"],
            mode="markers",
            name="Nuits après boxe",
            marker=dict(size=MARKER_SIZE + 3, color=SERIES_COLORS[0], line=dict(width=2, color=SURFACE)),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    figure.add_vline(
        x=float(cutoff),
        line=dict(color=INK_SECONDARY, width=1.5),
        annotation_text=f"{_fr(cutoff, 0)} h avant le coucher",
        annotation_position="top right",
        annotation_font=dict(size=11, color=INK_SECONDARY),
    )
    figure = _base_layout(figure, "Dernière séance de la journée et nuit suivante", y_title="Sommeil de la nuit suivante (h)", show_legend=False)
    figure.update_layout(hovermode="closest")
    figure.update_xaxes(title="Heures entre la fin de séance et votre coucher habituel", ticksuffix=" h")
    return figure


def habits_heatmap(matrix: pd.DataFrame | None) -> go.Figure | None:
    """Jour de semaine × moment de la journée : quand vous boxez réellement.

    Une case sans séance reste vide plutôt que peinte de la couleur la plus
    claire : zéro séance n'est pas « un peu de séances ».
    """
    if matrix is None or matrix.empty or float(matrix.to_numpy().sum()) == 0:
        return None
    values = matrix.to_numpy(dtype=float)
    shown = np.where(values > 0, values, np.nan)
    top = max(1.0, float(np.nanmax(shown)))
    scale = [[index / (len(SEQUENTIAL_BLUES) - 1), color] for index, color in enumerate(SEQUENTIAL_BLUES)]
    figure = go.Figure(
        go.Heatmap(
            z=shown,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=scale,
            zmin=1,
            zmax=top,
            xgap=3,
            ygap=3,
            hovertemplate="%{x}, %{y} : %{z:.0f} séance(s)<extra></extra>",
            colorbar=dict(title=dict(text="Séances", side="right"), thickness=12, outlinewidth=0, tickfont=dict(color=INK_MUTED, size=10), dtick=1),
        )
    )
    figure = _base_layout(figure, "Quand vous boxez", y_title="", height=300, show_legend=False)
    figure.update_layout(hovermode="closest")
    figure.update_xaxes(gridcolor=SURFACE, linecolor=SURFACE)
    figure.update_yaxes(gridcolor=SURFACE, linecolor=SURFACE, autorange="reversed")
    return figure


def session_intensity_chart(table: pd.DataFrame | None) -> go.Figure | None:
    """Minutes faciles, modérées et dures de chaque séance, empilées.

    La boxe alterne rounds et récupérations : la part du temps passé au-dessus
    de 80 % de la réserve cardiaque dit ce qu'une séance a vraiment demandé,
    là où sa durée seule ne le dit pas.
    """
    if table is None or table.empty:
        return None
    zone_columns = [column for _, members in INTENSITY_BANDS for column in members]
    if not all(column in table.columns for column in zone_columns):
        return None
    usable = table.dropna(subset=zone_columns, how="all")
    if usable.empty:
        return None
    figure = go.Figure()
    labels = [format_long_date(date) for date in usable["Date"]]
    for index, (band, members) in enumerate(INTENSITY_BANDS):
        minutes = usable[list(members)].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        figure.add_bar(
            x=usable["Date"],
            y=minutes,
            name=band,
            marker=dict(color=INTENSITY_COLORS[index % len(INTENSITY_COLORS)], line=dict(width=2, color=SURFACE)),
            customdata=labels,
            hovertemplate=f"%{{customdata}}<br>{band} : %{{y:.0f}} min<extra></extra>",
        )
    figure = _base_layout(figure, "Intensité de chaque séance", y_title="Minutes", show_legend=True)
    figure.update_layout(barmode="stack", bargap=0.35, hovermode="closest")
    return _calendar_axis(figure, usable["Date"])


def progression_chart(table: pd.DataFrame | None, metric: str) -> go.Figure | None:
    """Une mesure de séance dans le temps, avec sa droite de tendance."""
    if table is None or table.empty or metric not in table.columns:
        return None
    usable = table[["Date", metric]].copy()
    usable[metric] = pd.to_numeric(usable[metric], errors="coerce")
    usable = usable.dropna()
    if len(usable) < 2:
        return None
    figure = go.Figure()
    figure.add_scatter(
        x=usable["Date"],
        y=usable[metric],
        mode="markers",
        name="Chaque séance",
        connectgaps=False,
        marker=dict(size=MARKER_SIZE + 2, color=SERIES_COLORS[0], line=dict(width=2, color=SURFACE)),
        customdata=[format_long_date(date) for date in usable["Date"]],
        hovertemplate=f"%{{customdata}}<br>{metric} : %{{y:.1f}}<extra></extra>",
    )
    elapsed = (pd.to_datetime(usable["Date"]) - pd.to_datetime(usable["Date"]).min()).dt.days.to_numpy(dtype=float)
    if np.ptp(elapsed) > 0:
        slope, intercept = np.polyfit(elapsed, usable[metric].to_numpy(dtype=float), 1)
        ends = np.array([elapsed.min(), elapsed.max()])
        figure.add_scatter(
            x=[usable["Date"].min(), usable["Date"].max()],
            y=intercept + slope * ends,
            mode="lines",
            name="Tendance linéaire",
            connectgaps=False,
            line=dict(color=INK_MUTED, width=LINE_WIDTH, dash="dot"),
            hovertemplate="Tendance : %{y:.1f}<extra></extra>",
        )
    figure = _base_layout(figure, f"{metric}, séance après séance", y_title=metric, show_legend=True)
    figure.update_layout(hovermode="closest")
    figure.update_xaxes(linecolor=AXIS)
    return _calendar_axis(figure, usable["Date"])
