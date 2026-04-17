"""Thème visuel léger pour Suivi V2."""

from __future__ import annotations

import streamlit as st


def apply_global_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem;}
        h1, h2, h3 {letter-spacing: 0.2px;}
        div[data-testid="stMetric"] {background:#f7f9fc;border-radius:12px;padding:10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
