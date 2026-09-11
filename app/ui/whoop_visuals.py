"""Figures Plotly de l'onglet WHOOP.

Module volontairement sans dépendance à Streamlit : chaque fonction renvoie une
``go.Figure``, ce qui rend la mise en forme vérifiable par des tests plutôt que
par relecture visuelle.

Choix de couleurs : palette catégorielle validée pour la vision des couleurs
(séparation ΔE suffisante entre teintes adjacentes) et palette de statut
réservée aux significations bon/mauvais, jamais à une identité de série.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Palette catégorielle : ordre fixe, jamais recyclé ni réattribué au filtrage.
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# Palette de statut : réservée aux significations, jamais à une série.
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

ZONE_COLORS: Mapping[str, str] = {
    "Vert": STATUS_GOOD,
    "Jaune": STATUS_WARNING,
    "Rouge": STATUS_CRITICAL,
}

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_MUTED = "#898781"
INK_SECONDARY = "#52514e"

LINE_WIDTH = 2
MARKER_SIZE = 8
CHART_HEIGHT = 340
# Hauteur augmentée pour laisser la place à la bande d'axe et éviter qu'un
# conteneur trop court ne provoque une barre de défilement interne.
TALL_CHART_HEIGHT = 420


def _base_layout(figure: go.Figure, title: str, *, y_title: str = "", height: int = CHART_HEIGHT, show_legend: bool = True) -> go.Figure:
    """Grille discrète, axes en filet, marges laissant respirer les libellés."""
    figure.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0b0b0b")),
        height=height,
        margin=dict(t=56, b=52, l=56, r=24),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY, size=12),
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        yaxis=dict(
            title=y_title,
            gridcolor=GRID,
            griddash="solid",
            zeroline=False,
            linecolor=AXIS,
            tickfont=dict(color=INK_MUTED),
        ),
        xaxis=dict(
            gridcolor=GRID,
            griddash="solid",
            zeroline=False,
            linecolor=AXIS,
            tickfont=dict(color=INK_MUTED),
        ),
    )
    return figure


def _calendar_axis(figure: go.Figure) -> go.Figure:
    """Graduations au jour, au format jour/mois.

    Avec peu de points, Plotly bascule spontanément en graduations horaires
    (``03:00``, ``06:00``) sur des mesures pourtant quotidiennes.
    """
    figure.update_xaxes(tickformat="%d/%m", ticklabelmode="period", hoverformat="%d/%m/%Y")
    return figure


def series_chart(
    grid: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    y_title: str,
    *,
    trend: pd.DataFrame | None = None,
    height: int = CHART_HEIGHT,
) -> go.Figure | None:
    """Courbes quotidiennes sur calendrier continu, avec tendance lissée optionnelle.

    Renvoie ``None`` quand aucune métrique n'est exploitable : la page décide
    alors quoi afficher, plutôt que d'exposer un cadre vide.
    """
    usable = [metric for metric in metrics if metric in grid.columns and grid[metric].notna().any()]
    if not usable:
        return None

    figure = go.Figure()
    for index, metric in enumerate(usable):
        figure.add_scatter(
            x=grid["Date"],
            y=grid[metric],
            mode="lines+markers",
            name=metric,
            # Un jour sans mesure reste un trou : relier deux points distants
            # fabriquerait une tendance que personne n'a mesurée.
            connectgaps=False,
            line=dict(color=SERIES_COLORS[index % len(SERIES_COLORS)], width=LINE_WIDTH),
            marker=dict(size=MARKER_SIZE, line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{metric} : %{{y:.2f}}<extra></extra>",
        )

    if trend is not None and not trend.empty:
        trend_metric = [column for column in trend.columns if column != "Date"]
        if trend_metric and trend[trend_metric[0]].notna().any():
            figure.add_scatter(
                x=trend["Date"],
                y=trend[trend_metric[0]],
                mode="lines",
                name="Tendance 7 jours",
                # La tendance ne doit pas davantage enjamber les jours manquants.
                connectgaps=False,
                line=dict(color=INK_MUTED, width=LINE_WIDTH, dash="dot"),
                hovertemplate="Tendance 7 j : %{y:.2f}<extra></extra>",
            )

    # Une série unique se passe de légende : le titre la nomme déjà.
    show_legend = len(figure.data) > 1
    return _calendar_axis(_base_layout(figure, title, y_title=y_title, height=height, show_legend=show_legend))


def recovery_gauge(value: float, zone: str | None) -> go.Figure:
    """Jauge du dernier score de récupération, seuils WHOOP matérialisés."""
    raw = float(value) if value is not None and np.isfinite(float(value)) else 0.0
    # Un score de récupération vit entre 0 et 100 : borner évite qu'une valeur
    # aberrante ne dessine une aiguille hors du cadran.
    numeric = min(100.0, max(0.0, raw))
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=numeric,
            number=dict(suffix=" %", font=dict(size=34, color="#0b0b0b")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=AXIS, tickfont=dict(color=INK_MUTED, size=10)),
                bar=dict(color=ZONE_COLORS.get(str(zone), SERIES_COLORS[0]), thickness=0.7),
                bgcolor=SURFACE,
                borderwidth=0,
                steps=[
                    dict(range=[0, 34], color="#fbe3e3"),
                    dict(range=[34, 67], color="#fdf1d6"),
                    dict(range=[67, 100], color="#dcf3dc"),
                ],
            ),
        )
    )
    figure.update_layout(
        height=220,
        margin=dict(t=16, b=16, l=24, r=24),
        paper_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY),
    )
    return figure


def zone_distribution_chart(counts: pd.DataFrame) -> go.Figure | None:
    """Répartition des jours par zone : couleurs de statut, pas d'identité."""
    if counts is None or counts.empty:
        return None
    figure = go.Figure(
        go.Bar(
            x=counts["Zone"],
            y=counts["Jours"],
            marker=dict(color=[ZONE_COLORS.get(zone, SERIES_COLORS[0]) for zone in counts["Zone"]]),
            text=counts["Jours"],
            textposition="outside",
            hovertemplate="%{x} : %{y} jour(s)<extra></extra>",
        )
    )
    figure = _base_layout(figure, "Jours par zone de récupération", y_title="Jours", show_legend=False)
    figure.update_layout(hovermode="closest", bargap=0.45)
    return figure


def recovery_calendar(matrix: Mapping[str, Any]) -> go.Figure | None:
    """Calendrier semaine × jour : un mois de récupération d'un seul coup d'œil.

    L'échelle rouge → ambre → vert reprend la convention de WHOOP. C'est une
    échelle de chaleur sémantique, accompagnée de sa barre de légende ; les
    valeurs restent lisibles au survol et dans la vue tableau.
    """
    if not matrix or not matrix.get("values"):
        return None

    week_labels = [pd.Timestamp(week).strftime("%d/%m") for week in matrix["weeks"]]
    hover = [
        [
            "Aucune mesure" if date is None else f"{pd.Timestamp(date).strftime('%d/%m/%Y')}"
            for date in row
        ]
        for row in matrix["dates"]
    ]
    figure = go.Figure(
        go.Heatmap(
            z=matrix["values"],
            x=matrix["weekdays"],
            y=week_labels,
            customdata=hover,
            colorscale=[[0.0, STATUS_CRITICAL], [0.34, STATUS_WARNING], [0.67, STATUS_GOOD], [1.0, "#046b04"]],
            zmin=0,
            zmax=100,
            xgap=3,
            ygap=3,
            hovertemplate="%{customdata}<br>Récupération : %{z:.0f} %<extra></extra>",
            colorbar=dict(title=dict(text="%", side="right"), thickness=12, outlinewidth=0, tickfont=dict(color=INK_MUTED, size=10)),
        )
    )
    figure = _base_layout(
        figure,
        "Calendrier de récupération",
        y_title="Semaine du",
        height=max(220, 60 + 42 * len(week_labels)),
        show_legend=False,
    )
    figure.update_layout(hovermode="closest")
    figure.update_xaxes(gridcolor=SURFACE, linecolor=SURFACE)
    figure.update_yaxes(gridcolor=SURFACE, linecolor=SURFACE, autorange="reversed")
    return figure


def indexed_comparison_chart(indexed: pd.DataFrame, metrics: Sequence[str], title: str) -> go.Figure | None:
    """Séries ramenées à une base 100 commune, sur un axe unique.

    Superposer un poids et une métrique WHOOP sur deux axes verticaux
    fabriquerait une corrélation visuelle arbitraire : le calage des échelles
    serait un choix d'affichage, pas une donnée.
    """
    usable = [metric for metric in metrics if metric in indexed.columns and indexed[metric].notna().any()]
    if not usable:
        return None

    figure = go.Figure()
    for index, metric in enumerate(usable):
        figure.add_scatter(
            x=indexed["Date"],
            y=indexed[metric],
            mode="lines+markers",
            name=metric,
            connectgaps=False,
            line=dict(color=SERIES_COLORS[index % len(SERIES_COLORS)], width=LINE_WIDTH),
            marker=dict(size=MARKER_SIZE, line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{metric} : %{{y:.1f}} (base 100)<extra></extra>",
        )
    figure.add_hline(y=100, line=dict(color=AXIS, width=1))
    return _calendar_axis(_base_layout(figure, title, y_title="Base 100 au départ", show_legend=True))


def sleep_stages_chart(grid: pd.DataFrame) -> go.Figure | None:
    """Stades de sommeil empilés, séparés par un filet de surface."""
    stages = [stage for stage in ("Sommeil profond (heures)", "Sommeil REM (heures)") if stage in grid.columns and grid[stage].notna().any()]
    if not stages:
        return None

    figure = go.Figure()
    for index, stage in enumerate(stages):
        figure.add_bar(
            x=grid["Date"],
            y=grid[stage],
            name=stage,
            marker=dict(color=SERIES_COLORS[index % len(SERIES_COLORS)], line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{stage} : %{{y:.2f}} h<extra></extra>",
        )
    figure = _base_layout(figure, "Répartition des stades de sommeil", y_title="Heures", show_legend=True)
    figure.update_layout(barmode="stack", bargap=0.35)
    return _calendar_axis(figure)


def weekday_chart(profile: pd.DataFrame, metric_label: str) -> go.Figure | None:
    """Profil par jour de semaine : série unique, donc une seule couleur.

    Colorer chaque barre selon sa hauteur redoublerait l'information déjà portée
    par la longueur et consommerait le seul canal encore libre.
    """
    if profile is None or profile.empty:
        return None
    usable = profile.dropna(subset=["Moyenne"])
    if usable.empty:
        return None

    figure = go.Figure(
        go.Bar(
            x=usable["Jour"],
            y=usable["Moyenne"],
            marker=dict(color=SERIES_COLORS[0]),
            customdata=usable["Observations"],
            hovertemplate="%{x} : %{y:.0f}<br>%{customdata} mesure(s)<extra></extra>",
        )
    )
    figure = _base_layout(figure, f"{metric_label} par jour de la semaine", y_title=metric_label, show_legend=False)
    figure.update_layout(hovermode="closest", bargap=0.4)
    return figure


def sparkline(values: Sequence[float], *, positive: bool = True) -> go.Figure:
    """Micro-courbe d'accompagnement d'un indicateur, sans axes ni légende."""
    color = SERIES_COLORS[0] if positive else STATUS_SERIOUS
    figure = go.Figure(
        go.Scatter(
            y=list(values),
            mode="lines",
            line=dict(color=color, width=LINE_WIDTH),
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        height=60,
        margin=dict(t=4, b=4, l=4, r=4),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return figure
