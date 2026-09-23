from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import AppDefaults, DUPLICATE_STRATEGIES
from app.core.business import (
    FINAL_TARGET_WEIGHT_KG,
    TARGET_TRAJECTORY_END_DATE,
    TARGET_TRAJECTORY_START_DATE,
    TARGET_TRAJECTORY_START_WEIGHT_KG,
)
from app.core.formatting import format_fr_date, format_fr_kg
from app.core.session_state import (
    DEFAULT_ZOOM_TARGET_END_DATE,
    DEFAULT_ZOOM_TARGET_START_DATE,
    ensure_session_defaults,
)
from app.core.targets import get_target_weights, normalise_target_weights
from app.ui.components import kpi_card, page_hero, section_header

DUPLICATE_LABELS = {
    "garder_la_derniere": "Garder la dernière mesure du jour",
    "moyenne_journaliere": "Moyenne des mesures du jour",
    "mediane_journaliere": "Médiane des mesures du jour",
}


def main() -> None:
    ensure_session_defaults()
    defaults = AppDefaults()
    page_hero(
        "Configuration",
        "Paramètres",
        "Objectifs intermédiaires, taille, fenêtre de zoom et règles de calcul. Les changements vivent dans la session Streamlit.",
        meta=f"Trajectoire de référence : {format_fr_kg(TARGET_TRAJECTORY_START_WEIGHT_KG, decimals=1)} le {format_fr_date(TARGET_TRAJECTORY_START_DATE)} → {format_fr_kg(FINAL_TARGET_WEIGHT_KG, decimals=1)} le {format_fr_date(TARGET_TRAJECTORY_END_DATE)}",
    )

    padded_goals = get_target_weights(st.session_state)
    with st.form("settings_form"):
        section_header("Objectifs", "Cinq paliers du plus haut au plus bas ; le cinquième est l'objectif final.", "🎯")
        goal_cols = st.columns(5)
        goals = []
        for index, column in enumerate(goal_cols, start=1):
            with column:
                goals.append(st.number_input(f"Objectif {index} (kg)", value=float(padded_goals[index - 1]), step=0.5, format="%.1f"))
        st.number_input(
            "Poids objectif final (kg)",
            value=float(padded_goals[-1]),
            disabled=True,
            help="Synchronisé avec l'objectif 5 pour garder une cible finale unique.",
        )

        section_header("Profil", "Utilisé pour l'indice de masse corporelle.", "🧍")
        height_cm = st.number_input("Taille (cm)", min_value=120.0, max_value=230.0, value=float(st.session_state.get("height_cm", defaults.height_cm)), step=1.0, format="%.0f")

        section_header("Affichage", "Fenêtre de zoom du Dashboard et lissage optionnel.", "🖥️")
        zoom_cols = st.columns(2)
        with zoom_cols[0]:
            zoom_start = st.date_input(
                "Début du zoom trajectoire",
                value=pd.Timestamp(st.session_state.get("zoom_target_start_date", DEFAULT_ZOOM_TARGET_START_DATE)).date(),
                format="DD/MM/YYYY",
            )
        with zoom_cols[1]:
            zoom_end = st.date_input(
                "Fin du zoom trajectoire",
                value=pd.Timestamp(st.session_state.get("zoom_target_end_date", DEFAULT_ZOOM_TARGET_END_DATE)).date(),
                format="DD/MM/YYYY",
            )
        window_size = st.slider(
            "Fenêtre de la tendance long terme (EMA, mesures)",
            min_value=3,
            max_value=60,
            value=int(st.session_state.get("window_size", 7)),
            help="N'affecte que l'option « Tendance long terme » du graphique principal. Le poids de tendance LOWESS n'en dépend pas.",
        )

        section_header("Données", "Règle appliquée aux calculs quand plusieurs pesées tombent le même jour.", "🗂️")
        current_strategy = st.session_state.get("duplicate_strategy", defaults.duplicate_strategy)
        duplicate = st.selectbox(
            "Gestion des doublons journaliers",
            DUPLICATE_STRATEGIES,
            index=DUPLICATE_STRATEGIES.index(current_strategy) if current_strategy in DUPLICATE_STRATEGIES else 0,
            format_func=lambda key: DUPLICATE_LABELS.get(key, key),
            help="Le journal conserve toujours toutes les lignes ; cette règle ne concerne que les analyses.",
        )
        submitted = st.form_submit_button("Enregistrer", type="primary")

    if submitted:
        target_weights = normalise_target_weights(tuple(goals))
        if list(target_weights) != sorted(target_weights, reverse=True):
            st.warning("Les objectifs ne sont pas décroissants : vérifiez l'ordre des paliers (le 5ᵉ doit être le plus bas).")
        st.session_state["target_weights"] = target_weights
        st.session_state["target_weight"] = float(target_weights[-1])
        st.session_state["height_cm"] = float(height_cm)
        st.session_state["height_m"] = float(height_cm) / 100
        st.session_state["duplicate_strategy"] = duplicate
        zoom_start_ts = pd.Timestamp(zoom_start)
        zoom_end_ts = pd.Timestamp(zoom_end)
        if zoom_start_ts > zoom_end_ts:
            st.warning("La date de début du zoom doit être antérieure ou égale à la date de fin.")
        else:
            st.session_state["window_size"] = int(window_size)
            st.session_state["zoom_target_start_date"] = zoom_start_ts
            st.session_state["zoom_target_end_date"] = zoom_end_ts
            st.success("Paramètres enregistrés.")

    section_header("Diagnostic système", "Informations utiles pour vérifier la session active et le déploiement.", "🧪")
    working = st.session_state.get("working_data", pd.DataFrame())
    cols = st.columns(4)
    with cols[0]:
        kpi_card("Streamlit", st.__version__)
    with cols[1]:
        kpi_card("Source", str(st.session_state.get("data_source", "n/a")))
    with cols[2]:
        kpi_card("Lignes en session", f"{len(working)}")
    with cols[3]:
        kpi_card("WHOOP", "connecté" if st.session_state.get("whoop_token") else "non connecté")
    with st.expander("Clés de session", expanded=False):
        st.write(sorted(str(key) for key in st.session_state.keys()))


main()
