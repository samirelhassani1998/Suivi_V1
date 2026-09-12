"""Composants UI réutilisables Streamlit."""

from __future__ import annotations

from html import escape

import streamlit as st


def kpi_card(title: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    st.metric(label=title, value=value, delta=delta, help=help_text)


def confidence_badge(label: str, level: str) -> None:
    color = {"élevée": "🟢", "moyenne": "🟡", "faible": "🔴"}.get(level, "⚪")
    st.caption(f"{color} **{label}** : {level.title()}")


def alert_banner(message: str, kind: str = "info") -> None:
    getattr(st, kind if kind in {"info", "warning", "error", "success"} else "info")(message)


def empty_state(message: str) -> None:
    st.info(f"📭 {message}")


def help_box(title: str, text: str) -> None:
    with st.expander(f"ℹ️ {title}", expanded=False):
        st.write(text)


def page_hero(eyebrow: str, title: str, subtitle: str, meta: str | None = None) -> None:
    """Render a polished page header without relying on raw Streamlit title spacing."""
    meta_html = f"<div class='suivi-hero-meta'>{meta}</div>" if meta else ""
    st.markdown(
        f"""
        <section class="suivi-hero suivi-fade-in">
            <div>
                <span class="suivi-eyebrow">{eyebrow}</span>
                <h1>{title}</h1>
                <p>{subtitle}</p>
                {meta_html}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None, icon: str = "") -> None:
    """Consistent section title with optional helper copy."""
    st.markdown(
        f"""
        <div class="suivi-section-title">
            <h2>{icon} {title}</h2>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, body: str, tone: str = "neutral", icon: str = "💡") -> None:
    """Small editorial card for user-facing advice/alerts."""
    st.markdown(
        f"""
        <div class="suivi-insight-card suivi-insight-{tone}">
            <div class="suivi-insight-icon">{icon}</div>
            <div>
                <strong>{title}</strong>
                <p>{body}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def progress_panel(title: str, percent: float, caption: str, tone: str = "primary") -> None:
    """Render a responsive progress card with Streamlit-friendly HTML/CSS."""
    safe_percent = max(0.0, min(100.0, float(percent)))
    st.markdown(
        f"""
        <div class="suivi-progress-panel suivi-progress-{tone}">
            <div class="suivi-progress-top">
                <strong>{title}</strong>
                <span>{safe_percent:.1f}%</span>
            </div>
            <div class="suivi-progress-track"><div style="width:{safe_percent:.1f}%"></div></div>
            <p>{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def day_card(
    date_label: str,
    relative_label: str,
    zone: str | None,
    stats: list[tuple[str, str, str]],
    sessions: list[tuple[str, str, str]],
    *,
    empty_message: str | None = None,
) -> None:
    """Carte d'une journée : date en tête, mesures du jour, séances rattachées.

    ``stats`` reçoit des triplets (libellé, valeur, unité) déjà formatés, et
    ``sessions`` des triplets (heure, intitulé, détail). La mise en forme des
    nombres et des dates appartient à l'appelant : ce composant n'assemble que
    le balisage.
    """
    tone = str(zone).lower() if zone else "vide"
    head = f"<span class='suivi-day-date'>{escape(date_label)}</span>"
    if relative_label:
        head += f"<span class='suivi-day-ago'>{escape(relative_label)}</span>"

    if empty_message:
        body = f"<p class='suivi-day-empty'>{escape(empty_message)}</p>"
    else:
        cells = "".join(
            f"<div class='suivi-day-stat'><b>{escape(value)}</b>"
            f"<span>{escape(unit)}</span><span>· {escape(label)}</span></div>"
            for label, value, unit in stats
        )
        body = f"<div class='suivi-day-stats'>{cells}</div>"

    if sessions:
        items = "".join(
            f"<li><time>{escape(when)}</time><strong>{escape(title)}</strong>"
            f"<span>{escape(detail)}</span></li>"
            for when, title, detail in sessions
        )
        body += f"<ul class='suivi-day-sessions'>{items}</ul>"

    st.markdown(
        f"<div class='suivi-day-card suivi-day-{tone}'>"
        f"<div class='suivi-day-head'>{head}</div>{body}</div>",
        unsafe_allow_html=True,
    )
