
"""Entry point for the Streamlit multi-page application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.utils import (
    DATA_URL,
    filter_by_dates,
    get_date_range,
    load_data,
)
from app.auth import check_password


st.set_page_config(page_title="Suivi du Poids Amélioré", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container {
        font-family: 'Segoe UI', sans-serif;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf, #2e7bcf);
        color: white;
    }
    .stButton>button {
        background-color: #2e7bcf;
        color: white;
        border: none;
    }
    .stMetric {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_dataset() -> None:
    """Load the dataset and store both raw and filtered versions."""
    if "data_url" not in st.session_state:
        st.session_state["data_url"] = st.secrets.get("data_url", DATA_URL)

    if st.session_state.get("reload_requested"):
        load_data.clear()
        st.session_state.pop("reload_requested")

    try:
        df = load_data(st.session_state["data_url"])
    except RuntimeError as error:
        st.error(
            "Impossible de charger les données distantes. "
            "Veuillez vérifier la connexion réseau ou réessayer plus tard."
        )
        st.caption(str(error))
        empty_df = pd.DataFrame(columns=["Date", "Poids (Kgs)"])
        st.session_state["raw_data"] = empty_df
        st.session_state["filtered_data"] = empty_df
        st.stop()

    st.session_state["raw_data"] = df
    st.session_state["filtered_data"] = df


def _configure_sidebar() -> None:
    """Setup sidebar controls and apply filtering to the dataset."""
    st.sidebar.header("Paramètres Généraux")

    if st.sidebar.button("Recharger les données"):
        st.session_state["reload_requested"] = True
        _load_dataset()

    theme = st.sidebar.selectbox(
        "Choisir un thème",
        ["Default", "Dark", "Light", "Solar", "Seaborn"],
        key="theme",
    )
    st.session_state["theme"] = theme

    ma_type = st.sidebar.selectbox(
        "Type de moyenne mobile",
        ["Simple", "Exponentielle"],
        key="ma_type",
    )
    st.session_state["ma_type"] = ma_type

    window_size = st.sidebar.slider(
        "Taille de la moyenne mobile (jours)",
        1,
        30,
        st.session_state.get("window_size", 7),
        key="window_size",
    )
    st.session_state["window_size"] = window_size

    df = st.session_state.get("raw_data")
    if df is not None and not df.empty:
        try:
            date_min, date_max = get_date_range(df)
            date_range = st.sidebar.date_input(
                "Sélectionnez une plage de dates",
                (date_min, date_max),
                key="date_range",
            )
            st.session_state["filtered_data"] = filter_by_dates(df, date_range)
        except ValueError as error:
            st.sidebar.error(f"Impossible de déterminer la plage de dates : {error}")
            st.session_state["filtered_data"] = df
    else:
        st.session_state["filtered_data"] = df

    st.sidebar.markdown("---")
    st.sidebar.subheader("Objectifs et Infos Personnelles")
    target1 = st.sidebar.number_input(
        "Objectif 1 (Kgs)",
        value=st.session_state.get("target_weight_1", 95.0),
        key="target_weight_1",
    )
    target2 = st.sidebar.number_input(
        "Objectif 2 (Kgs)",
        value=st.session_state.get("target_weight_2", 90.0),
        key="target_weight_2",
    )
    target3 = st.sidebar.number_input(
        "Objectif 3 (Kgs)",
        value=st.session_state.get("target_weight_3", 85.0),
        key="target_weight_3",
    )
    target4 = st.sidebar.number_input(
        "Objectif 4 (Kgs)",
        value=st.session_state.get("target_weight_4", 80.0),
        key="target_weight_4",
    )
    st.session_state["target_weights"] = (target1, target2, target3, target4)

    height_cm = st.sidebar.number_input(
        "Votre taille (cm)",
        value=st.session_state.get("height_cm", 182),
        key="height_cm",
    )
    st.session_state["height_m"] = height_cm / 100.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("Anomalies & Activité")
    anomaly_method = st.sidebar.selectbox(
        "Méthode de détection",
        ["Z-score", "IsolationForest"],
        key="anomaly_method",
    )
    st.session_state["anomaly_method"] = anomaly_method

    z_threshold = st.sidebar.slider(
        "Seuil Z-score",
        1.0,
        5.0,
        st.session_state.get("z_threshold", 2.0),
        step=0.5,
        key="z_threshold",
    )
    st.session_state["z_threshold"] = z_threshold

    calories = st.sidebar.number_input(
        "Calories consommées aujourd'hui",
        min_value=0,
        value=st.session_state.get("calories", 2000),
        key="calories",
    )
    calories_burned = st.sidebar.number_input(
        "Calories brûlées (approximatif)",
        min_value=0,
        value=st.session_state.get("calories_burned", 500),
        key="calories_burned",
    )
    st.sidebar.write(
        "Bilan calorique estimé :",
        calories - calories_burned,
        "kcal",
    )


def _register_pages() -> None:
    """Register the Streamlit pages for navigation."""
    if hasattr(st, "Page") and hasattr(st, "navigation"):
        pages = [
            st.Page("pages/1_Analyse.py", title="Analyse", icon="📊"),
            st.Page("pages/2_Modeles.py", title="Modèles", icon="🤖"),
            st.Page("pages/3_Predictions.py", title="Prédictions", icon="📈"),
        ]
        navigator = st.navigation(pages)
        navigator.run()
    else:
        st.sidebar.info(
            "Utilisez le menu de navigation Streamlit (colonne de gauche) "
            "pour accéder aux autres pages."
        )


def main() -> None:
    if not check_password():
        st.stop()

    _load_dataset()
    _configure_sidebar()

    st.title("Suivi de l'Évolution du Poids")
    st.markdown(
        """
        Suivez votre poids, calculez votre IMC, visualisez des tendances, réalisez des prévisions et détectez des anomalies.
        Utilisez la navigation pour accéder aux différentes analyses.
        """
    )

    _register_pages()

    st.markdown("---")
    st.markdown("**Sources et Références :**")
    st.markdown("- [Streamlit Documentation](https://docs.streamlit.io/)")
    st.markdown("- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)")
    st.markdown("- [Plotly Graph Objects](https://plotly.com/python/)")
    st.markdown("- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)")
    st.markdown("- [Statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)")
    st.markdown("- [Pandas Documentation](https://pandas.pydata.org/docs/)")
    st.markdown("- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)")
    st.markdown("- [OMS - Obésité et Surpoids](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)")


if __name__ == "__main__":
    main()
