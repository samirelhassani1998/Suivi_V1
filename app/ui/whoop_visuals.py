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

from app.core.date_labels import format_clock_hour, format_day_month, format_long_date
from app.core.whoop import sport_label

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

# Bandes de référence publiées par WHOOP, dessinées en fond des courbes pour
# que la lecture d'une valeur n'exige plus de se rappeler les seuils.
RECOVERY_BANDS: tuple[tuple[float, float, str, str], ...] = (
    (0.0, 34.0, "#fbe3e3", "Rouge"),
    (34.0, 67.0, "#fdf1d6", "Jaune"),
    (67.0, 100.0, "#dcf3dc", "Vert"),
)
# Échelle de strain : léger 0–9,9, modéré 10–13,9, élevé 14–17,9, maximal 18–21
# (https://www.whoop.com/us/en/thelocker/how-does-whoop-strain-work-101/).
STRAIN_BANDS: tuple[tuple[float, float, str, str], ...] = (
    (0.0, 10.0, "#eef3f8", "Léger"),
    (10.0, 14.0, "#e2ecf7", "Modéré"),
    (14.0, 18.0, "#fdf1d6", "Élevé"),
    (18.0, 21.0, "#fbe3e3", "Maximal"),
)

LINE_WIDTH = 2
MARKER_SIZE = 8
CHART_HEIGHT = 340
# Hauteur augmentée pour laisser la place à la bande d'axe et éviter qu'un
# conteneur trop court ne provoque une barre de défilement interne.
TALL_CHART_HEIGHT = 420
MAX_CALENDAR_HEIGHT = 640


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


MAX_AXIS_TICKS = 8


def french_date_ticks(dates: Any, *, max_ticks: int = MAX_AXIS_TICKS) -> tuple[list, list[str]]:
    """Positions et étiquettes d'axe en français (``3 sept.``).

    Plotly ne connaît que les mois anglais : laisser ses graduations
    automatiques affiche « Sep 3 » au milieu d'une interface française.
    """
    parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
    unique = parsed.dt.normalize().drop_duplicates().sort_values()
    if unique.empty:
        return [], []
    step = max(1, int(np.ceil(len(unique) / max(1, int(max_ticks)))))
    selected = list(unique)[::step]
    return selected, [format_day_month(date) for date in selected]


def _calendar_axis(figure: go.Figure, dates: Any = None) -> go.Figure:
    """Graduations au jour, étiquetées en français.

    Avec peu de points, Plotly bascule spontanément en graduations horaires
    (``03:00``, ``06:00``) sur des mesures pourtant quotidiennes.
    """
    figure.update_xaxes(ticklabelmode="period", hoverformat="%d/%m/%Y")
    if dates is not None:
        tickvals, ticktext = french_date_ticks(dates)
        if tickvals:
            figure.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            return figure
    figure.update_xaxes(tickformat="%d/%m")
    return figure


def _add_bands(figure: go.Figure, bands: Sequence[tuple[float, float, str, str]] | None) -> go.Figure:
    """Bandes horizontales de référence, étiquetées à droite, sous les données."""
    for low, high, color, label in bands or ():
        figure.add_hrect(
            y0=low,
            y1=high,
            fillcolor=color,
            line_width=0,
            layer="below",
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(size=10, color=INK_MUTED),
        )
    return figure


def series_chart(
    grid: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    y_title: str,
    *,
    trend: pd.DataFrame | None = None,
    height: int = CHART_HEIGHT,
    bands: Sequence[tuple[float, float, str, str]] | None = None,
) -> go.Figure | None:
    """Courbes quotidiennes sur calendrier continu, avec tendance lissée optionnelle.

    Renvoie ``None`` quand aucune métrique n'est exploitable : la page décide
    alors quoi afficher, plutôt que d'exposer un cadre vide. ``bands`` dessine
    des plages de référence en fond (zones de récupération, niveaux de strain).
    """
    usable = [metric for metric in metrics if metric in grid.columns and grid[metric].notna().any()]
    if not usable:
        return None

    figure = go.Figure()
    _add_bands(figure, bands)
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
    return _calendar_axis(_base_layout(figure, title, y_title=y_title, height=height, show_legend=show_legend), grid["Date"])


def recovery_gauge(value: float, zone: str | None) -> go.Figure:
    """Jauge du dernier score de récupération, seuils WHOOP matérialisés."""
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = float("nan")
    missing = not np.isfinite(raw)
    # Une mesure absente affichée à 0 % se lisait comme une récupération
    # catastrophique, au centre de la zone rouge.
    numeric = 0.0 if missing else min(100.0, max(0.0, raw))
    figure = go.Figure(
        go.Indicator(
            mode="gauge" if missing else "gauge+number",
            value=numeric,
            title=dict(text="Récupération du jour", font=dict(size=13, color=INK_SECONDARY)),
            number=dict(suffix=" %", font=dict(size=34, color="#0b0b0b")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=AXIS, tickfont=dict(color=INK_MUTED, size=10)),
                bar=dict(color="rgba(0,0,0,0)" if missing else ZONE_COLORS.get(str(zone), SERIES_COLORS[0]), thickness=0.7),
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
        height=240,
        margin=dict(t=44, b=16, l=24, r=24),
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

    week_labels = [format_day_month(week) for week in matrix["weeks"]]
    hover = [
        [
            "Aucune mesure" if date is None else format_long_date(date)
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
        # Sans plafond, un an d'historique produisait un graphique de plus de
        # deux mille pixels de haut, illisible et interminable à faire défiler.
        height=min(MAX_CALENDAR_HEIGHT, max(220, 60 + 42 * len(week_labels))),
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
    return _calendar_axis(_base_layout(figure, title, y_title="Base 100 au départ", show_legend=True), indexed["Date"])


# Du plus profond au plus léger, l'éveil en dernier : l'empilement se lit de
# bas en haut comme une nuit, et sa hauteur totale est le temps passé au lit.
SLEEP_STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("Sommeil profond (heures)", SERIES_COLORS[0]),
    ("Sommeil REM (heures)", SERIES_COLORS[2]),
    ("Sommeil léger (heures)", "#9ec3ec"),
    ("Éveil (heures)", "#d9d8d0"),
)


def sleep_stages_chart(grid: pd.DataFrame) -> go.Figure | None:
    """Stades de sommeil empilés, séparés par un filet de surface.

    Sans le sommeil léger et l'éveil, l'empilement plafonnait à trois heures
    sur une nuit de sept : le lecteur cherchait où était passé le reste.
    """
    stages = [(stage, color) for stage, color in SLEEP_STAGE_ORDER if stage in grid.columns and grid[stage].notna().any()]
    if not stages:
        return None

    figure = go.Figure()
    for stage, color in stages:
        figure.add_bar(
            x=grid["Date"],
            y=grid[stage],
            name=stage.replace(" (heures)", ""),
            marker=dict(color=color, line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{stage.replace(' (heures)', '')} : %{{y:.2f}} h<extra></extra>",
        )
    figure = _base_layout(figure, "Composition de chaque nuit", y_title="Heures", show_legend=True)
    figure.update_layout(barmode="stack", bargap=0.35)
    return _calendar_axis(figure, grid["Date"])


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


def bedtime_chart(grid: pd.DataFrame) -> go.Figure | None:
    """Heures de coucher, lues en horloge et non sur l'échelle interne signée.

    Les couchers d'avant minuit sont stockés en négatif pour rester continus avec
    ceux d'après minuit ; exposer ce codage sur l'axe afficherait « −1,5 » là où
    le lecteur attend « 22:30 ».
    """
    if "Heure de coucher" not in grid.columns or not grid["Heure de coucher"].notna().any():
        return None

    values = grid["Heure de coucher"]
    figure = go.Figure()
    figure.add_scatter(
        x=grid["Date"],
        y=values,
        mode="lines+markers",
        name="Heure de coucher",
        connectgaps=False,
        line=dict(color=SERIES_COLORS[0], width=LINE_WIDTH),
        marker=dict(size=MARKER_SIZE, line=dict(width=2, color=SURFACE)),
        customdata=[format_clock_hour(value) for value in values],
        hovertemplate="Coucher : %{customdata}<extra></extra>",
    )
    low, high = float(values.min()), float(values.max())
    ticks = [tick / 2 for tick in range(int(np.floor(low * 2)), int(np.ceil(high * 2)) + 1)]
    figure.update_yaxes(tickmode="array", tickvals=ticks, ticktext=[format_clock_hour(tick) for tick in ticks])
    return _calendar_axis(
        _base_layout(figure, "Heure de coucher", y_title="Heure locale", show_legend=False),
        grid["Date"],
    )


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


def recovery_bars_chart(grid: pd.DataFrame, *, trend: pd.DataFrame | None = None) -> go.Figure | None:
    """Une barre par matin noté, colorée de la zone WHOOP du score.

    La courbe dit la forme ; la barre colorée dit le verdict de chaque jour
    sans obliger à comparer une hauteur aux seuils. La tendance sur sept
    jours reste en filigrane.
    """
    if "Récupération (%)" not in grid.columns or not grid["Récupération (%)"].notna().any():
        return None
    values = grid["Récupération (%)"]
    # Les seuils des barres sont ceux des bandes de fond : une seule définition.
    zones = [
        next((label for low, high, _color, label in RECOVERY_BANDS if low <= value < high or (value == high == 100.0)), None)
        if np.isfinite(value)
        else None
        for value in values
    ]
    figure = go.Figure()
    _add_bands(figure, RECOVERY_BANDS)
    figure.add_bar(
        x=grid["Date"],
        y=values,
        name="Récupération",
        marker=dict(color=[ZONE_COLORS.get(zone, SERIES_COLORS[0]) for zone in zones], line=dict(width=0)),
        customdata=[format_long_date(date) for date in grid["Date"]],
        hovertemplate="%{customdata}<br>Récupération : %{y:.0f} %<extra></extra>",
    )
    if trend is not None and not trend.empty:
        trend_metric = [column for column in trend.columns if column != "Date"]
        if trend_metric and trend[trend_metric[0]].notna().any():
            figure.add_scatter(
                x=trend["Date"],
                y=trend[trend_metric[0]],
                mode="lines",
                name="Tendance 7 jours",
                connectgaps=False,
                line=dict(color=INK_SECONDARY, width=LINE_WIDTH, dash="dot"),
                hovertemplate="Tendance 7 j : %{y:.0f} %<extra></extra>",
            )
    figure = _base_layout(figure, "Récupération, jour par jour", y_title="%", show_legend=len(figure.data) > 1)
    figure.update_layout(bargap=0.25, hovermode="closest")
    figure.update_yaxes(range=[0, 100])
    return _calendar_axis(figure, grid["Date"])


def smoothed_baseline_chart(series: pd.DataFrame, metric: str, title: str, unit: str) -> go.Figure | None:
    """Valeurs quotidiennes en points, moyenne mobile en trait, plage habituelle en fond."""
    if series is None or series.empty or metric not in series.columns or not series[metric].notna().any():
        return None
    figure = go.Figure()
    low, high = series["Bas"].dropna(), series["Haut"].dropna()
    if not low.empty and not high.empty:
        figure.add_hrect(
            y0=float(low.iloc[0]),
            y1=float(high.iloc[0]),
            fillcolor="#e8f1e8",
            line_width=0,
            layer="below",
            annotation_text="Plage habituelle",
            annotation_position="top left",
            annotation_font=dict(size=10, color=INK_MUTED),
        )
    figure.add_scatter(
        x=series["Date"],
        y=series[metric],
        mode="markers",
        name="Chaque nuit",
        # Des points isolés n'ont rien à relier ; la déclaration reste explicite
        # pour que la règle « aucune série nommée ne comble un trou » se vérifie.
        connectgaps=False,
        marker=dict(size=MARKER_SIZE - 2, color=INK_MUTED, opacity=0.7),
        hovertemplate=f"Nuit : %{{y:.0f}} {unit}<extra></extra>",
    )
    figure.add_scatter(
        x=series["Date"],
        y=series["Lissé"],
        mode="lines",
        name="Moyenne 7 jours",
        connectgaps=False,
        line=dict(color=SERIES_COLORS[0], width=LINE_WIDTH + 1),
        hovertemplate=f"Moyenne 7 j : %{{y:.0f}} {unit}<extra></extra>",
    )
    return _calendar_axis(_base_layout(figure, title, y_title=unit, show_legend=True), series["Date"])


def sessions_timeline_chart(sessions: pd.DataFrame) -> go.Figure | None:
    """Chaque séance à sa date, hauteur = strain de la séance, couleur = sport.

    Le survol donne la récupération du lendemain matin : l'effort et sa
    conséquence se lisent au même endroit.
    """
    if sessions is None or sessions.empty or "Strain séance" not in sessions.columns:
        return None
    usable = sessions.dropna(subset=["Strain séance"])
    if usable.empty:
        return None
    figure = go.Figure()
    labelled = usable["Sport"].map(sport_label)
    sports = list(dict.fromkeys(labelled))
    for index, sport in enumerate(sports):
        chunk = usable[labelled == sport]
        following = chunk["Récupération du lendemain (%)"] if "Récupération du lendemain (%)" in chunk.columns else pd.Series(np.nan, index=chunk.index)
        hover = [
            f"{format_long_date(date)}<br>{sport} : strain {strain:.1f}"
            + (f"<br>Lendemain : {value:.0f} %" if np.isfinite(value) else "<br>Lendemain : non noté")
            for date, strain, value in zip(chunk["Date"], chunk["Strain séance"], following.fillna(np.nan))
        ]
        figure.add_bar(
            x=chunk["Date"],
            y=chunk["Strain séance"],
            name=sport,
            marker=dict(color=SERIES_COLORS[index % len(SERIES_COLORS)], line=dict(width=1, color=SURFACE)),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    figure = _base_layout(figure, "Séances dans le temps", y_title="Strain de la séance", show_legend=True)
    figure.update_layout(barmode="stack", bargap=0.3, hovermode="closest")
    return _calendar_axis(figure, usable["Date"])


INTENSITY_COLORS: tuple[str, ...] = ("#9ec3ec", "#eda100", "#d03b3b")


def intensity_profile_chart(profile: pd.DataFrame) -> go.Figure | None:
    """Barres horizontales à 100 % : part facile / modérée / dure, par sport."""
    if profile is None or profile.empty or "Sport" not in profile.columns:
        return None
    bands = [column for column in profile.columns if column.startswith(("Facile", "Modéré", "Dur"))]
    if not bands:
        return None
    figure = go.Figure()
    labels = [sport_label(sport) if sport != "Toutes séances" else sport for sport in profile["Sport"]]
    for index, band in enumerate(bands):
        figure.add_bar(
            y=labels,
            x=profile[band],
            name=band,
            orientation="h",
            marker=dict(color=INTENSITY_COLORS[index % len(INTENSITY_COLORS)], line=dict(width=1, color=SURFACE)),
            text=[f"{value:.0f} %" if np.isfinite(value) and value >= 8 else "" for value in profile[band]],
            textposition="inside",
            hovertemplate=f"{band} : %{{x:.0f}} %<extra></extra>",
        )
    figure = _base_layout(
        figure,
        "Répartition de l'intensité, par sport",
        y_title="",
        height=max(220, 80 + 48 * len(labels)),
        show_legend=True,
    )
    figure.update_layout(barmode="stack", bargap=0.35, hovermode="closest")
    figure.update_xaxes(range=[0, 100], title="% du temps en zones", ticksuffix=" %")
    figure.update_yaxes(autorange="reversed")
    return figure
