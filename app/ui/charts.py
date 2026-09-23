"""Style Plotly partagé par les pages poids (Dashboard, Prévisions, Insights).

Une seule définition des couleurs, des marges, de la légende et des axes de
dates : les graphiques de toutes les pages se lisent comme un même système,
aligné sur celui de l'onglet WHOOP. Aucune dépendance à Streamlit : chaque
fonction reçoit ou renvoie une ``go.Figure`` et reste vérifiable par des tests.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.ui.whoop_visuals import (
    AXIS,
    GRID,
    INK_MUTED,
    INK_SECONDARY,
    SERIES_COLORS,
    STATUS_CRITICAL,
    STATUS_GOOD,
    STATUS_SERIOUS,
    STATUS_WARNING,
    SURFACE,
)

# Rôles fixes : la mesure est toujours bleue, la tendance toujours orange, la
# cible toujours verte. Un lecteur qui passe d'une page à l'autre n'a pas à
# réapprendre la légende.
MEASURE_COLOR = SERIES_COLORS[0]
TREND_COLOR = SERIES_COLORS[1]
TARGET_COLOR = SERIES_COLORS[2]
ACCENT_COLOR = SERIES_COLORS[3]
TRAJECTORY_COLOR = "#0f766e"
NEUTRAL_LINE = "#8b8a83"
BAND_FILL = "rgba(42,120,214,0.12)"
TREND_BAND_FILL = "rgba(235,104,52,0.12)"
FONT_FAMILY = "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

CHART_HEIGHT = 420
SHORT_CHART_HEIGHT = 340

# Graduations de dates sans mois anglais : formats numériques adaptés au zoom.
DATE_TICK_STOPS = (
    dict(dtickrange=[None, 1000 * 60 * 60 * 24 * 15], value="%d/%m"),
    dict(dtickrange=[1000 * 60 * 60 * 24 * 15, "M1"], value="%d/%m"),
    dict(dtickrange=["M1", "M12"], value="%m/%Y"),
    dict(dtickrange=["M12", None], value="%Y"),
)


def apply_layout(
    figure: go.Figure,
    title: str,
    *,
    y_title: str = "",
    x_title: str = "",
    height: int = CHART_HEIGHT,
    show_legend: bool = True,
    hovermode: str = "x unified",
    date_axis: bool = True,
) -> go.Figure:
    """Mise en forme commune : surface claire, grille discrète, légende en tête."""
    # Légende sous le graphique : elle peut occuper deux ou trois lignes
    # (objectifs, trajectoire, tendance, bruit) sans jamais recouvrir le titre.
    legend_offset = -0.26 if x_title else -0.16
    figure.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.97, yanchor="top", font=dict(size=16, color="#0b0b0b")),
        height=height,
        margin=dict(t=56, b=110 if show_legend else 48, l=56, r=24),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, color=INK_SECONDARY, size=12),
        hovermode=hovermode,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_offset,
            xanchor="left",
            x=0,
            font=dict(size=11),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
    )
    figure.update_yaxes(
        title=y_title,
        gridcolor=GRID,
        griddash="solid",
        zeroline=False,
        linecolor=AXIS,
        tickfont=dict(color=INK_MUTED),
        title_font=dict(color=INK_MUTED),
    )
    figure.update_xaxes(
        title=x_title,
        gridcolor=GRID,
        griddash="solid",
        showgrid=False,
        zeroline=False,
        linecolor=AXIS,
        tickfont=dict(color=INK_MUTED),
        title_font=dict(color=INK_MUTED),
    )
    if date_axis:
        figure.update_xaxes(hoverformat="%d/%m/%Y", tickformatstops=list(DATE_TICK_STOPS))
    return figure


def add_measure_trace(
    figure: go.Figure,
    dates: Sequence[Any] | pd.Series,
    values: Sequence[float] | pd.Series,
    *,
    name: str = "Poids mesuré",
    color: str = MEASURE_COLOR,
    mode: str = "lines+markers",
    opacity: float = 1.0,
    marker_size: int = 6,
    width: float = 2.2,
) -> go.Figure:
    """Série de pesées : ligne fine et points, jamais de lissage implicite."""
    figure.add_scatter(
        x=list(dates),
        y=list(values),
        mode=mode,
        name=name,
        opacity=opacity,
        line=dict(color=color, width=width),
        marker=dict(size=marker_size, color=color, line=dict(width=1, color=SURFACE)),
        hovertemplate=f"{name} : %{{y:.2f}} kg<extra></extra>",
    )
    return figure


def add_line_trace(
    figure: go.Figure,
    dates: Sequence[Any] | pd.Series,
    values: Sequence[float] | pd.Series,
    *,
    name: str,
    color: str,
    dash: str | None = None,
    width: float = 2.2,
    unit: str = "kg",
    decimals: int = 2,
    showlegend: bool = True,
) -> go.Figure:
    figure.add_scatter(
        x=list(dates),
        y=list(values),
        mode="lines",
        name=name,
        showlegend=showlegend,
        line=dict(color=color, width=width, dash=dash) if dash else dict(color=color, width=width),
        hovertemplate=f"{name} : %{{y:.{decimals}f}} {unit}<extra></extra>",
    )
    return figure


def add_band(
    figure: go.Figure,
    dates: Sequence[Any] | pd.Series,
    low: Sequence[float] | pd.Series,
    high: Sequence[float] | pd.Series,
    *,
    name: str,
    fill: str = BAND_FILL,
    unit: str = "kg",
) -> go.Figure:
    """Bande entre deux bornes, dessinée sous les courbes et lisible au survol."""
    x = list(dates)
    figure.add_scatter(
        x=x,
        y=list(high),
        mode="lines",
        name=f"{name} (haut)",
        line=dict(width=0),
        showlegend=False,
        hovertemplate=f"{name}, haut : %{{y:.2f}} {unit}<extra></extra>",
    )
    figure.add_scatter(
        x=x,
        y=list(low),
        mode="lines",
        name=name,
        fill="tonexty",
        fillcolor=fill,
        line=dict(width=0),
        hovertemplate=f"{name}, bas : %{{y:.2f}} {unit}<extra></extra>",
    )
    return figure


def add_horizontal_reference(
    figure: go.Figure,
    dates: Sequence[Any] | pd.Series | pd.Index,
    value: float,
    *,
    name: str,
    color: str = TARGET_COLOR,
    dash: str = "dash",
    width: float = 1.4,
    showlegend: bool = True,
) -> go.Figure:
    """Ligne horizontale tracée comme une série : elle reste dans la légende et au survol."""
    if dates is None or len(dates) == 0:
        return figure
    x_min, x_max = min(dates), max(dates)
    figure.add_scatter(
        x=[x_min, x_max],
        y=[float(value), float(value)],
        mode="lines",
        name=name,
        showlegend=showlegend,
        line=dict(color=color, width=width, dash=dash),
        hovertemplate=f"{name}<extra></extra>",
    )
    return figure


def histogram_figure(
    values: Sequence[float] | pd.Series,
    title: str,
    *,
    x_title: str,
    nbins: int = 25,
    color: str = MEASURE_COLOR,
    reference: float | None = None,
    reference_label: str = "",
    height: int = SHORT_CHART_HEIGHT,
) -> go.Figure:
    """Histogramme monochrome, une ligne de référence optionnelle."""
    figure = go.Figure(
        go.Histogram(
            x=list(values),
            nbinsx=nbins,
            marker=dict(color=color, line=dict(width=1, color=SURFACE)),
            hovertemplate="%{x} : %{y} mesure(s)<extra></extra>",
        )
    )
    if reference is not None and np.isfinite(reference):
        figure.add_vline(
            x=float(reference),
            line=dict(color=TREND_COLOR, width=2, dash="dash"),
            annotation_text=reference_label,
            annotation_position="top right",
            annotation_font=dict(size=11, color=INK_MUTED),
        )
    figure = apply_layout(figure, title, y_title="Mesures", x_title=x_title, height=height, show_legend=False, hovermode="closest", date_axis=False)
    figure.update_layout(bargap=0.08)
    return figure


def bar_figure(
    labels: Sequence[Any],
    values: Sequence[float],
    title: str,
    *,
    y_title: str = "",
    colors: Sequence[str] | str = MEASURE_COLOR,
    hover: Sequence[str] | None = None,
    height: int = SHORT_CHART_HEIGHT,
    text: Sequence[str] | None = None,
    date_axis: bool = False,
) -> go.Figure:
    """Barres d'une seule série ; la couleur ne code une signification que si elle est fournie."""
    figure = go.Figure(
        go.Bar(
            x=list(labels),
            y=list(values),
            marker=dict(color=list(colors) if not isinstance(colors, str) else colors, line=dict(width=0)),
            customdata=list(hover) if hover is not None else None,
            hovertemplate="%{customdata}<extra></extra>" if hover is not None else "%{x} : %{y:.2f}<extra></extra>",
            text=list(text) if text is not None else None,
            textposition="outside" if text is not None else None,
        )
    )
    figure = apply_layout(figure, title, y_title=y_title, height=height, show_legend=False, hovermode="closest", date_axis=date_axis)
    figure.update_layout(bargap=0.3)
    return figure


def status_color(value: float, *, good_below: float | None = None, bad_above: float | None = None) -> str:
    """Couleur de statut d'une variation : baisse verte, hausse rouge, quasi-nulle neutre."""
    if value is None or not np.isfinite(value):
        return NEUTRAL_LINE
    if good_below is not None and value < good_below:
        return STATUS_GOOD
    if bad_above is not None and value > bad_above:
        return STATUS_CRITICAL
    return NEUTRAL_LINE


__all__ = [
    "ACCENT_COLOR",
    "BAND_FILL",
    "CHART_HEIGHT",
    "MEASURE_COLOR",
    "NEUTRAL_LINE",
    "SHORT_CHART_HEIGHT",
    "STATUS_CRITICAL",
    "STATUS_GOOD",
    "STATUS_SERIOUS",
    "STATUS_WARNING",
    "TARGET_COLOR",
    "TRAJECTORY_COLOR",
    "TREND_BAND_FILL",
    "TREND_COLOR",
    "add_band",
    "add_horizontal_reference",
    "add_line_trace",
    "add_measure_trace",
    "apply_layout",
    "bar_figure",
    "histogram_figure",
    "status_color",
]
