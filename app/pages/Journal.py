from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from app.core.data import validate_journal
from app.core.data_editing import has_unsaved_changes
from app.core.formatting import format_fr_date, format_fr_kg
from app.core.session_state import get_working_data, set_working_data
from app.ui.components import alert_banner, empty_state, kpi_card, page_hero, section_header


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
    for column in display.columns:
        if display[column].dtype == object:
            display[column] = display[column].fillna("")
    return display


def _dataframes_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return not has_unsaved_changes(left, right)


def _column_config(df: pd.DataFrame) -> dict:
    """Dates en jour/mois/année et poids à une décimale dans l'éditeur."""
    config: dict = {}
    if "Date" in df.columns:
        config["Date"] = st.column_config.DateColumn("Date", format="DD/MM/YYYY", help="Jour de la pesée.")
    if "Poids (Kgs)" in df.columns:
        config["Poids (Kgs)"] = st.column_config.NumberColumn("Poids (kg)", format="%.1f", min_value=20.0, max_value=400.0, step=0.1)
    for column in df.columns:
        if column in config:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            config[column] = st.column_config.NumberColumn(column, format="%.1f")
    return config


def _quick_add_form(df: pd.DataFrame) -> pd.DataFrame:
    """Ajout d'une pesée en trois champs, sans faire défiler l'éditeur jusqu'en bas."""
    section_header("Ajouter une pesée", "Le geste du matin : une date, un poids, une note facultative.", "➕")
    last_weight = float(pd.to_numeric(df["Poids (Kgs)"], errors="coerce").dropna().iloc[-1]) if not df.empty and "Poids (Kgs)" in df.columns and pd.to_numeric(df["Poids (Kgs)"], errors="coerce").notna().any() else 80.0
    with st.form("journal_quick_add", clear_on_submit=False):
        cols = st.columns([1, 1, 2, 1])
        with cols[0]:
            date_value = st.date_input("Date", value=dt.date.today(), format="DD/MM/YYYY")
        with cols[1]:
            weight_value = st.number_input("Poids (kg)", min_value=20.0, max_value=400.0, value=round(last_weight, 1), step=0.1, format="%.1f")
        with cols[2]:
            note_value = st.text_input("Note (facultatif)", value="", placeholder="à jeun, après le sport…")
        with cols[3]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Ajouter", type="primary", use_container_width=True)
    if not submitted:
        return df

    new_row: dict = {column: np.nan for column in df.columns}
    new_row["Date"] = pd.Timestamp(date_value)
    new_row["Poids (Kgs)"] = float(weight_value)
    if note_value.strip():
        new_row["Notes"] = note_value.strip()
    combined = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    report = validate_journal(combined)
    if report.errors:
        for err in report.errors:
            alert_banner(err, "error")
        return df
    same_day = int((pd.to_datetime(df["Date"], errors="coerce").dt.normalize() == pd.Timestamp(date_value)).sum()) if not df.empty and "Date" in df.columns else 0
    set_working_data(report.cleaned)
    message = f"Pesée du {format_fr_date(date_value)} ajoutée : {format_fr_kg(weight_value, decimals=1)}."
    if same_day:
        message += f" Le journal contenait déjà {same_day} mesure(s) ce jour-là ; toutes sont conservées."
    st.success(message)
    return get_working_data()


def _status_strip(df: pd.DataFrame) -> None:
    q = st.session_state.get("data_quality", {})
    dates = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
    cols = st.columns(4)
    with cols[0]:
        kpi_card("Mesures en session", f"{len(df)}")
    with cols[1]:
        if dates.notna().any():
            kpi_card("Période couverte", f"{format_fr_date(dates.min())} → {format_fr_date(dates.max())}", help_text=f"{dates.dt.normalize().nunique()} jour(s) distinct(s).")
        else:
            kpi_card("Période couverte", "—")
    with cols[2]:
        kpi_card("Dernière pesée", format_fr_date(dates.max()) if dates.notna().any() else "—")
    with cols[3]:
        kpi_card("Lignes rejetées à l'import", f"{q.get('invalid_rows', 0)}", help_text=f"Source : {q.get('source', 'n/a')} · {q.get('duplicate_dates', 0)} date(s) en double conservée(s).")


def main() -> None:
    df = _ensure_df()
    page_hero(
        "Données",
        "Journal",
        "Ajoutez, modifiez et exportez vos mesures tout en conservant les colonnes personnalisées de votre CSV.",
        meta=f"{len(df)} ligne(s) en session" if not df.empty else "Aucune mesure en session",
    )
    if df.empty:
        empty_state("Commencez par ajouter une première ligne.")

    q = st.session_state.get("data_quality", {})
    if q:
        st.info(
            f"Qualité données ({q.get('source', 'n/a')}) : {q.get('raw_rows', 0)} lues, "
            f"{q.get('valid_rows', 0)} valides conservées, {q.get('invalid_rows', 0)} invalides, "
            f"{q.get('duplicate_dates', 0)} dates dupliquées, {q.get('columns_kept', 0)} colonnes conservées. "
            f"Colonnes additionnelles : {', '.join(q.get('extra_columns', [])) or 'aucune'}."
        )

    _status_strip(df)
    df = _quick_add_form(df)

    section_header("Édition des mesures", "Filtrez une période, éditez les lignes puis enregistrez en session. Toutes les colonnes du CSV sont conservées.", "🧾")

    with st.expander("Filtre date (aperçu)", expanded=False):
        if not df.empty and "Date" in df.columns:
            min_d, max_d = df["Date"].min(), df["Date"].max()
            date_range = st.date_input("Période", value=(min_d.date(), max_d.date()), format="DD/MM/YYYY")
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                preview = df[(df["Date"] >= start) & (df["Date"] <= end)]
                st.dataframe(_format_dates_for_display(preview.tail(20)), use_container_width=True, hide_index=True)

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="journal_editor",
        column_config=_column_config(df),
    )
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
        if not _dataframes_equal(report.cleaned, get_working_data()):
            st.warning(
                "Des modifications visibles ne sont pas encore enregistrées.\n"
                "Enregistrez-les avant l'export pour les inclure."
            )
        st.download_button(
            "Exporter les données enregistrées",
            data=get_working_data().to_csv(index=False).encode("utf-8"),
            file_name="journal_poids.csv",
            mime="text/csv",
        )

    section_header("Dernières lignes", "Contrôle rapide après édition ou import, de la plus récente à la plus ancienne.", "👀")
    recent = edited.copy()
    if "Date" in recent.columns:
        recent = recent.assign(_order=pd.to_datetime(recent["Date"], errors="coerce")).sort_values("_order", ascending=False).drop(columns="_order")
    st.dataframe(_format_dates_for_display(recent.head(10)), use_container_width=True, hide_index=True)


main()
