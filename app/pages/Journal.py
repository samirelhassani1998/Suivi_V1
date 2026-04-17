from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import ALL_COLUMNS
from app.core.data import validate_journal
from app.ui.components import alert_banner, empty_state


def _ensure_df() -> pd.DataFrame:
    if "raw_data" not in st.session_state:
        st.session_state["raw_data"] = pd.DataFrame(columns=ALL_COLUMNS)
    return st.session_state["raw_data"].copy()


def main() -> None:
    st.title("Journal")
    df = _ensure_df()
    if df.empty:
        empty_state("Commencez par ajouter une première ligne.")

    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="journal_editor")
    report = validate_journal(edited)

    for err in report.errors:
        alert_banner(err, "error")
    for w in report.warnings:
        alert_banner(w, "warning")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enregistrer dans la session", type="primary"):
            st.session_state["raw_data"] = report.cleaned.copy()
            st.session_state["filtered_data"] = report.cleaned.copy()
            st.success("Journal enregistré en session.")
    with c2:
        st.download_button(
            "Exporter CSV",
            data=report.cleaned.to_csv(index=False).encode("utf-8"),
            file_name="journal_poids.csv",
            mime="text/csv",
        )


main()
