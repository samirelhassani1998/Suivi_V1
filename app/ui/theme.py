"""Thème visuel léger pour Suivi V2."""

from __future__ import annotations

import streamlit as st


def apply_global_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --suivi-blue: #2E7BCF;
            --suivi-green: #27AE60;
            --suivi-orange: #F39C12;
            --suivi-red: #E74C3C;
            --suivi-ink: #1f2937;
            --suivi-muted: #667085;
            --suivi-card: #f7f9fc;
            --suivi-border: #e5e7eb;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        h1, h2, h3 {letter-spacing: 0.2px; color: var(--suivi-ink);}
        h1 {margin-bottom: 0.25rem;}
        .suivi-hero {
            margin: 0.25rem 0 1rem 0;
            padding: 1rem 1.1rem;
            border: 1px solid var(--suivi-border);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(46,123,207,0.10), rgba(39,174,96,0.08));
        }
        .suivi-hero p {
            margin: 0.25rem 0 0 0;
            color: var(--suivi-muted);
            font-size: 1rem;
            line-height: 1.45;
        }
        .suivi-eyebrow {
            display: inline-block;
            color: var(--suivi-blue);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.75rem;
            margin-bottom: 0.2rem;
        }
        div[data-testid="stMetric"] {
            background: var(--suivi-card);
            border: 1px solid var(--suivi-border);
            border-radius: 14px;
            padding: 0.85rem 0.9rem;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }
        div[data-testid="stMetricLabel"] p {color: var(--suivi-muted); font-size: 0.88rem;}
        div[data-testid="stMetricValue"] {font-size: 1.35rem;}
        div[data-testid="stMetricDelta"] {font-size: 0.88rem;}
        .stPlotlyChart {
            border: 1px solid var(--suivi-border);
            border-radius: 16px;
            padding: 0.35rem;
            background: #ffffff;
        }
        @media (max-width: 768px) {
            .block-container {padding-left: 0.75rem; padding-right: 0.75rem;}
            .suivi-hero {padding: 0.9rem; border-radius: 14px;}
            div[data-testid="stMetric"] {padding: 0.75rem;}
            div[data-testid="stMetricValue"] {font-size: 1.15rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
