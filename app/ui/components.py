"""Composants UI réutilisables Streamlit."""

from __future__ import annotations

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
