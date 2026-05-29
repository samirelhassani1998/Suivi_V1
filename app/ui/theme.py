"""Thème visuel moderne pour Suivi V2."""

from __future__ import annotations

import streamlit as st


def apply_global_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --suivi-blue: #2563eb;
            --suivi-blue-soft: #dbeafe;
            --suivi-green: #16a34a;
            --suivi-green-soft: #dcfce7;
            --suivi-orange: #f97316;
            --suivi-orange-soft: #ffedd5;
            --suivi-red: #dc2626;
            --suivi-purple: #7c3aed;
            --suivi-ink: #111827;
            --suivi-muted: #667085;
            --suivi-subtle: #98a2b3;
            --suivi-card: rgba(255, 255, 255, 0.92);
            --suivi-card-soft: #f8fafc;
            --suivi-border: #e5e7eb;
            --suivi-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 32rem),
                linear-gradient(180deg, #f8fbff 0%, #ffffff 34%, #f8fafc 100%);
            color: var(--suivi-ink);
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {color: rgba(255,255,255,0.90);}
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            background: rgba(255,255,255,0.055);
            overflow: hidden;
            margin-bottom: 0.7rem;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] details summary {
            font-weight: 800;
        }
        section[data-testid="stSidebar"] .stButton button,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.10);
            color: #ffffff;
            font-weight: 800;
        }
        section[data-testid="stSidebar"] .stButton button:hover,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
            border-color: rgba(255,255,255,0.36);
            background: rgba(37,99,235,0.45);
            color: #ffffff;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: rgba(255,255,255,0.06);
            border: 1px dashed rgba(255,255,255,0.22);
            border-radius: 16px;
        }
        .suivi-sidebar-card {
            margin: 0.25rem 0 1rem 0;
            padding: 0.85rem 0.95rem;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
            box-shadow: 0 14px 30px rgba(0,0,0,0.12);
        }
        .suivi-sidebar-card strong {
            display: block;
            margin-top: 0.2rem;
            color: #ffffff;
            font-size: 0.98rem;
        }
        .suivi-sidebar-eyebrow {
            color: rgba(255,255,255,0.62) !important;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }
        h1, h2, h3 {letter-spacing: -0.02em; color: var(--suivi-ink);}
        h1 {font-size: clamp(2rem, 4vw, 3.25rem); margin: 0.1rem 0 0.25rem 0; line-height: 1.04;}
        h2 {font-size: clamp(1.25rem, 2vw, 1.7rem); margin-top: 0.2rem;}
        h3 {font-size: 1.05rem;}
        .suivi-fade-in {animation: suiviFadeIn 0.35s ease-out both;}
        @keyframes suiviFadeIn {from {opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);}}
        .suivi-hero {
            position: relative;
            overflow: hidden;
            margin: 0.15rem 0 1.15rem 0;
            padding: 1.35rem 1.45rem;
            border: 1px solid rgba(37, 99, 235, 0.14);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,246,255,0.92)),
                radial-gradient(circle at 90% 20%, rgba(22,163,74,0.16), transparent 16rem);
            box-shadow: var(--suivi-shadow);
        }
        .suivi-hero:after {
            content: "";
            position: absolute;
            width: 18rem;
            height: 18rem;
            right: -8rem;
            top: -9rem;
            background: radial-gradient(circle, rgba(37,99,235,0.18), transparent 65%);
        }
        .suivi-hero p {
            max-width: 760px;
            margin: 0.25rem 0 0 0;
            color: var(--suivi-muted);
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .suivi-hero-meta {
            display: inline-flex;
            margin-top: 0.8rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            color: #1e3a8a;
            background: rgba(219,234,254,0.85);
            font-size: 0.86rem;
            font-weight: 700;
        }
        .suivi-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--suivi-blue);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-size: 0.76rem;
            margin-bottom: 0.2rem;
        }
        .suivi-section-title {
            margin: 1.05rem 0 0.65rem 0;
        }
        .suivi-section-title h2 {margin-bottom: 0.05rem;}
        .suivi-section-title p {margin: 0; color: var(--suivi-muted); line-height: 1.45;}
        div[data-testid="stMetric"] {
            background: var(--suivi-card);
            border: 1px solid rgba(226, 232, 240, 0.9);
            border-radius: 20px;
            padding: 1rem 1rem;
            box-shadow: 0 10px 28px rgba(15,23,42,0.055);
            min-height: 112px;
        }
        div[data-testid="stMetricLabel"] p {color: var(--suivi-muted); font-size: 0.88rem; font-weight: 700;}
        div[data-testid="stMetricValue"] {font-size: clamp(1.25rem, 2.2vw, 1.75rem); font-weight: 800; color: var(--suivi-ink);}
        div[data-testid="stMetricDelta"] {font-size: 0.88rem; font-weight: 700;}
        div[data-testid="stTabs"] button {font-weight: 800; color: var(--suivi-muted);}
        div[data-testid="stTabs"] button[aria-selected="true"] {color: var(--suivi-blue);}
        .stPlotlyChart {
            border: 1px solid rgba(226,232,240,0.95);
            border-radius: 24px;
            padding: 0.6rem;
            background: #ffffff;
            box-shadow: 0 16px 36px rgba(15,23,42,0.055);
        }
        .suivi-insight-card, .suivi-progress-panel {
            display: flex;
            gap: 0.8rem;
            align-items: flex-start;
            padding: 1rem;
            margin: 0.45rem 0;
            border: 1px solid var(--suivi-border);
            border-radius: 20px;
            background: var(--suivi-card);
            box-shadow: 0 10px 24px rgba(15,23,42,0.045);
        }
        .suivi-insight-card p, .suivi-progress-panel p {margin: 0.25rem 0 0 0; color: var(--suivi-muted); line-height: 1.45;}
        .suivi-insight-icon {font-size: 1.35rem; line-height: 1;}
        .suivi-insight-success {border-color: rgba(22,163,74,0.22); background: linear-gradient(135deg, #ffffff, var(--suivi-green-soft));}
        .suivi-insight-warning {border-color: rgba(249,115,22,0.25); background: linear-gradient(135deg, #ffffff, var(--suivi-orange-soft));}
        .suivi-insight-info {border-color: rgba(37,99,235,0.20); background: linear-gradient(135deg, #ffffff, var(--suivi-blue-soft));}
        .suivi-progress-panel {display: block;}
        .suivi-progress-top {display: flex; justify-content: space-between; gap: 1rem; align-items: center;}
        .suivi-progress-top span {font-size: 1.25rem; font-weight: 900; color: var(--suivi-blue);}
        .suivi-progress-track {height: 0.72rem; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 0.85rem 0 0.35rem 0;}
        .suivi-progress-track div {height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--suivi-blue), var(--suivi-green));}
        .suivi-progress-success .suivi-progress-track div {background: linear-gradient(90deg, var(--suivi-green), #86efac);}
        div[data-testid="stAlert"] {border-radius: 18px; border: 1px solid rgba(226,232,240,0.9);}
        .stDataFrame, div[data-testid="stDataEditor"] {border-radius: 18px; overflow: hidden;}
        @media (max-width: 768px) {
            .block-container {padding-left: 0.8rem; padding-right: 0.8rem;}
            .suivi-hero {padding: 1rem; border-radius: 20px;}
            .suivi-hero p {font-size: 0.95rem;}
            div[data-testid="stMetric"] {padding: 0.85rem; min-height: 96px;}
            .stPlotlyChart {border-radius: 16px;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
