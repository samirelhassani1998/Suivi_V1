from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.data import validate_journal
from app.core.session_state import get_working_data, set_working_data
from app.ui.components import alert_banner, empty_state


def _ensure_df() -> pd.DataFrame:
    df = get_working_data()
    if df.empty:
        return pd.DataFrame(columns=["Date", "Poids (Kgs)"])
    return df


def _format_dates_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Crée une copie avec les dates formatées en dd/mm/yyyy pour l'affichage."""
    display = df.copy()
    if "Date" in display.columns and not display.empty:
        display["Date"] = pd.to_datetime(display["Date"], errors="coerce").dt.strftime("%d/%m/%Y")
    return display


def main() -> None:
    st.title("Journal")
    df = _ensure_df()
    if df.empty:
        empty_state("Commencez par ajouter une première ligne.")

    st.caption("Le journal conserve toutes les colonnes du CSV. Les lignes invalides sont signalées, pas supprimées silencieusement.")

    with st.expander("Filtre date (aperçu)", expanded=False):
        if not df.empty and "Date" in df.columns:
            min_d, max_d = df["Date"].min(), df["Date"].max()
            date_range = st.date_input("Période", value=(min_d.date(), max_d.date()))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                preview = df[(df["Date"] >= start) & (df["Date"] <= end)]
                st.dataframe(_format_dates_for_display(preview.tail(20)), use_container_width=True)

    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="journal_editor")
    report = validate_journal(edited)

    for err in report.errors:
        alert_banner(err, "error")
    for w in report.warnings:
        alert_banner(w, "warning")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enregistrer les modifications", type="primary"):
            if report.errors:
                st.error("Impossible d'enregistrer tant que les erreurs bloquantes persistent.")
            else:
                set_working_data(report.cleaned)
                st.success("Modifications enregistrées dans la session.")
    with c2:
        st.download_button(
            "Exporter CSV",
            data=edited.to_csv(index=False).encode("utf-8"),
            file_name="journal_poids.csv",
            mime="text/csv",
        )

    st.subheader("Aperçu des dernières lignes")
    st.dataframe(_format_dates_for_display(edited.tail(10)), use_container_width=True)


main()

