"""Tests de rendu de l'onglet Boxe (Streamlit AppTest, sans réseau)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

PAGE = "app/pages/Boxe.py"


def _token() -> dict:
    return {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_at": (pd.Timestamp.utcnow() + pd.Timedelta(hours=2)).isoformat(),
        "scopes": ["offline", "read:workout"],
        "token_type": "Bearer",
    }


def _rich_state(at: AppTest, *, days: int = 60, sport: str = "boxing") -> None:
    """Deux mois de données se terminant aujourd'hui, une séance tous les trois jours."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=days, freq="D")
    daily = pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": rng.uniform(25, 95, days),
            "HRV (ms)": rng.normal(45, 6, days),
            "FC repos (bpm)": rng.normal(56, 3, days),
            "Sommeil (heures)": rng.normal(7, 0.6, days),
            "Efficacité sommeil (%)": rng.normal(90, 3, days),
            "Heure de coucher": rng.normal(-0.5, 0.7, days),
            "Calories (kcal)": rng.normal(2800, 200, days),
            "Strain": rng.uniform(6, 16, days),
        }
    )
    rows = []
    for index, date in enumerate(dates):
        if index % 3:
            continue
        rows.append(
            {
                "Date": date,
                "Début": date + pd.Timedelta(hours=19 if index % 2 else 12, minutes=15),
                "Sport": sport,
                "Durée (min)": 58.6432,
                "Strain séance": float(rng.uniform(10, 16)),
                "Calories séance (kcal)": float(rng.uniform(500, 800)),
                "FC moyenne (bpm)": float(rng.uniform(130, 160)),
                "FC max (bpm)": float(rng.uniform(170, 190)),
                "Part enregistrée (%)": 100.0,
                **{f"Zone {zone} (min)": float(rng.uniform(2, 15)) for zone in range(6)},
            }
        )
    rows.append({"Date": dates[4], "Début": dates[4] + pd.Timedelta(hours=8), "Sport": "running", "Durée (min)": 30.0, "Strain séance": 8.0})
    weights = pd.DataFrame({"Date": dates, "Poids (Kgs)": 104 - np.arange(days) * 0.06 + rng.normal(0, 0.3, days)})

    at.session_state["source_data"] = weights.copy()
    at.session_state["working_data"] = weights.copy()
    at.session_state["filtered_data"] = pd.DataFrame()
    at.session_state["filter_active"] = False
    at.session_state["whoop_token"] = _token()
    at.session_state["whoop_daily"] = daily
    at.session_state["whoop_workouts"] = pd.DataFrame(rows)
    at.session_state["whoop_body"] = {"max_heart_rate": 192}
    at.session_state["whoop_last_sync"] = pd.Timestamp.now()


def _text(at: AppTest) -> str:
    parts = [str(element.value) for element in at.markdown]
    parts += [str(element.value) for element in at.caption]
    parts += [str(element.value) for element in at.info]
    return " ".join(parts)


def test_boxing_page_points_to_the_whoop_tab_without_a_connected_account():
    at = AppTest.from_file(PAGE)
    at.run(timeout=20)

    assert not at.exception
    assert any("Aucun compte WHOOP connecté" in str(info.value) for info in at.info)


def test_boxing_page_offers_a_sync_when_connected_without_data():
    at = AppTest.from_file(PAGE)
    at.session_state["whoop_token"] = _token()
    at.run(timeout=20)

    assert not at.exception
    assert any("Lancez une synchronisation" in str(info.value) for info in at.info)
    assert any(button.label == "Synchroniser maintenant" for button in at.button)


def test_boxing_page_renders_every_tab_with_its_analyses():
    at = AppTest.from_file(PAGE)
    _rich_state(at)
    at.run(timeout=60)

    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Vue d'ensemble", "Séances", "Récupération", "Sommeil", "Charge & progression", "Poids & énergie"]
    rendered = _text(at)
    for heading in (
        "Repère du jour",
        "Journal des séances",
        "Ce que la boxe coûte au lendemain",
        "Séances tardives et sommeil",
        "Charge boxe",
        "Progression",
        "Ce que la boxe pèse dans votre objectif",
        "La balance du lendemain",
    ):
        assert heading in rendered
    assert len(at.get("plotly_chart")) >= 6
    assert len(at.metric) >= 15


def test_boxing_page_counts_only_boxing_sessions():
    at = AppTest.from_file(PAGE)
    _rich_state(at)
    at.run(timeout=60)

    assert not at.exception
    sessions = next(metric for metric in at.metric if metric.label == "Séances")
    # Vingt séances de boxe sur soixante jours ; la course du cinquième jour n'est pas comptée.
    assert sessions.value == "20"


def test_boxing_page_formats_the_session_journal_without_raw_precision():
    at = AppTest.from_file(PAGE)
    _rich_state(at)
    at.run(timeout=60)

    assert not at.exception
    journal = next(frame.value for frame in at.dataframe if "Matin" in frame.value.columns)
    assert "58.6432" not in journal.to_string()
    assert journal["Durée (min)"].iloc[0] == "59 min"
    assert journal["Matin"].iloc[0].split(" ")[-1] in {"Vert", "Jaune", "Rouge", "—"}


def test_boxing_page_cites_its_sources():
    at = AppTest.from_file(PAGE)
    _rich_state(at)
    at.run(timeout=60)

    rendered = _text(at)
    for source in ("doi.org/10.1038/s41467-025-58271-x", "pubmed.ncbi.nlm.nih.gov/11708692", "doi.org/10.3390/jpm7020003", "developer.whoop.com"):
        assert source in rendered


def test_boxing_page_falls_back_to_combat_sports_and_explains_an_empty_selection():
    at = AppTest.from_file(PAGE)
    _rich_state(at, sport="kickboxing")
    at.run(timeout=60)

    assert not at.exception
    assert at.multiselect[0].value == ["Kickboxing"]
    assert any("Aucune séance étiquetée" in str(caption.value) for caption in at.caption)


def test_boxing_page_never_touches_the_weight_data():
    at = AppTest.from_file(PAGE)
    _rich_state(at)
    before = at.session_state["working_data"].copy()
    at.run(timeout=60)

    assert not at.exception
    pd.testing.assert_frame_equal(at.session_state["working_data"], before)


def test_main_navigation_exposes_the_boxing_page():
    source = Path("Suivi_V1.py").read_text(encoding="utf-8")

    assert 'st.Page("app/pages/Boxe.py", title="Boxe"' in source
