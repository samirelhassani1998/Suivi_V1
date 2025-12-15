
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


st.set_page_config(page_title="Suivi & Analyses du Poids", layout="wide")

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

    with st.spinner("Chargement des données de poids..."):
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

    ma_type = st.sidebar.selectbox(
        "Type de moyenne mobile",
        ["Simple", "Exponentielle"],
        key="ma_type",
    )

    window_size = st.sidebar.slider(
        "Taille de la moyenne mobile (jours)",
        1,
        30,
        st.session_state.get("window_size", 7),
        key="window_size",
    )

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

    z_threshold = st.sidebar.slider(
        "Seuil Z-score",
        1.0,
        5.0,
        st.session_state.get("z_threshold", 2.0),
        step=0.5,
        key="z_threshold",
    )

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
    """Register the Streamlit pages for navigation with explicit labels."""

    analyse_page = ("pages/1_Analyse.py", "Suivi & Analyses du poids", "📊")
    model_page = ("pages/2_Modeles.py", "Modèles prédictifs", "🤖")
    prediction_page = ("pages/3_Predictions.py", "Prévisions du poids", "📈")

    if hasattr(st, "Page") and hasattr(st, "navigation"):
        pages = [
            st.Page(analyse_page[0], title=analyse_page[1], icon=analyse_page[2]),
            st.Page(model_page[0], title=model_page[1], icon=model_page[2]),
            st.Page(prediction_page[0], title=prediction_page[1], icon=prediction_page[2]),
        ]
        navigator = st.navigation(pages)
        navigator.run()
    else:
        if hasattr(st.sidebar, "page_link"):
            st.sidebar.page_link(
                analyse_page[0], label=f"{analyse_page[2]} {analyse_page[1]}"
            )
            st.sidebar.page_link(
                model_page[0], label=f"{model_page[2]} {model_page[1]}"
            )
            st.sidebar.page_link(
                prediction_page[0], label=f"{prediction_page[2]} {prediction_page[1]}"
            )
        else:
            st.sidebar.markdown("## Navigation")
            st.sidebar.markdown(f"[{analyse_page[2]} {analyse_page[1]}]({analyse_page[0]})")
            st.sidebar.markdown(f"[{model_page[2]} {model_page[1]}]({model_page[0]})")
            st.sidebar.markdown(
                f"[{prediction_page[2]} {prediction_page[1]}]({prediction_page[0]})"
            )


def main() -> None:
    if not check_password():
        st.stop()

    _load_dataset()
    _configure_sidebar()
    _register_pages()

    st.title("Suivi & Analyses du Poids")
    st.markdown(
        """
        Retrouvez vos courbes de poids, IMC, anomalies et prévisions en un clic.
        La page d'analyses rassemble toutes les visualisations clés pour suivre vos progrès.
        """
    )

    if hasattr(st, "page_link"):
        st.page_link(
            "pages/1_Analyse.py",
            label="📊 Accéder directement aux analyses du poids",
            icon="📊",
        )
    else:
        st.markdown("[📊 Accéder directement aux analyses du poids](pages/1_Analyse.py)")

    st.markdown("---")
    st.subheader("Navigation rapide")
    if hasattr(st, "page_link"):
        st.page_link("pages/1_Analyse.py", label="📊 Suivi & Analyses du poids", icon="📊")
        st.page_link("pages/2_Modeles.py", label="🤖 Modèles prédictifs", icon="🤖")
        st.page_link("pages/3_Predictions.py", label="📈 Prévisions du poids", icon="📈")
    else:
        st.markdown("[📊 Suivi & Analyses du poids](pages/1_Analyse.py)")
        st.markdown("[🤖 Modèles prédictifs](pages/2_Modeles.py)")
        st.markdown("[📈 Prévisions du poids](pages/3_Predictions.py)")

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
