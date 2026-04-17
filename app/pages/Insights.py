from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.data import data_quality_report
from app.core.insights import detect_anomalies_robust, detect_plateau
from app.ui.components import empty_state


def _df() -> pd.DataFrame:
    return st.session_state.get("filtered_data", pd.DataFrame(columns=["Date", "Poids (Kgs)"]))


def main() -> None:
    st.title("Insights / Analyse")
    df = _df()
    if df.empty:
        empty_state("Pas encore de données à analyser.")
        return

    quality = data_quality_report(df)
    st.json(quality)

    plateau14 = detect_plateau(df, 14)
    plateau30 = detect_plateau(df, 30)
    st.write(f"14j: **{plateau14['status']}** | pente={plateau14['slope']:.3f}")
    st.write(f"30j: **{plateau30['status']}** | pente={plateau30['slope']:.3f}")

    anomalies = detect_anomalies_robust(df, use_iforest=st.toggle("Activer IsolationForest", value=False))
    st.dataframe(anomalies[["Date", "Poids (Kgs)", "anomalie", "raison", "decision"]], use_container_width=True)


main()
