from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.core.formatting import format_fr_kg
from app.core.data_editing import has_unsaved_changes
from streamlit.testing.v1 import AppTest

from app.core.target_trajectory import compare_to_target_trajectory


def _state(at: AppTest) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=80),
            "Poids (Kgs)": [95 - i * 0.08 for i in range(80)],
            "Extra Col": [f"v{i}" for i in range(80)],
            "Moment": ["08:00" for _ in range(80)],
        }
    )
    at.session_state["source_data"] = df.copy()
    at.session_state["working_data"] = df.copy()
    at.session_state["filtered_data"] = df.copy()
    at.session_state["raw_data"] = df.copy()
    at.session_state["target_weights"] = (100.0, 95.0, 90.0, 85.0, 80.0)
    at.session_state["fast_mode"] = True
    at.session_state["data_quality"] = {"source": "test", "raw_rows": 80, "valid_rows": 80, "invalid_rows": 0, "duplicate_dates": 0, "columns_kept": len(df.columns), "extra_columns": ["Extra Col", "Moment"]}


def _aligned_offset(dates: pd.DatetimeIndex, weights: list[float]) -> float:
    """Écart à appliquer pour que la dernière mesure tombe sur la trajectoire.

    Le calculer plutôt que le figer évite que ce test redevienne faux à la
    prochaine évolution des paramètres de trajectoire cible.
    """
    reference = pd.DataFrame({"Date": dates, "Poids (Kgs)": weights})
    return -float(compare_to_target_trajectory(reference)["gap_kg"])


def _active_trajectory_state(at: AppTest, *, offset_kg: float | None = 0.0) -> None:
    dates = pd.date_range(
        "2026-09-20",
        periods=40,
        freq="D",
    )

    weights = [
        106.1 - index * 0.3
        for index in range(len(dates))
    ]
    # None demande explicitement une série alignée sur la trajectoire cible.
    weights[-1] += _aligned_offset(dates, weights) if offset_kg is None else offset_kg

    df = pd.DataFrame(
        {
            "Date": dates,
            "Poids (Kgs)": weights,
            "Moment": ["08:00"] * len(dates),
            "Extra Col": [
                f"v{index}"
                for index in range(len(dates))
            ],
        }
    )

    at.session_state["source_data"] = df.copy()
    at.session_state["working_data"] = df.copy()
    at.session_state["filtered_data"] = pd.DataFrame()
    at.session_state["filter_active"] = False
    at.session_state["raw_data"] = df.copy()
    at.session_state["target_weights"] = (
        100.0,
        95.0,
        90.0,
        85.0,
        80.0,
    )
    at.session_state["fast_mode"] = True
    at.session_state["data_quality"] = {
        "source": "test",
        "raw_rows": len(df),
        "valid_rows": len(df),
        "invalid_rows": 0,
        "duplicate_dates": 0,
        "columns_kept": len(df.columns),
        "extra_columns": ["Moment", "Extra Col"],
    }


def test_dashboard_renders_active_target_trajectory_without_exception():
    at = AppTest.from_file(
        "app/pages/Dashboard.py"
    )

    _active_trajectory_state(at)
    at.run(timeout=15)

    assert not at.exception

    rendered = " ".join(
        str(element.value)
        for element in at.markdown
    )

    assert "Trajectoire cible" in rendered
    assert "Rythme moyen requis" in rendered
    assert "Écart" in rendered
    assert "statut" in rendered


@pytest.mark.parametrize(
    ("offset_kg", "expected_status"),
    [
        (-2.0, "en avance"),
        (None, "aligné"),
    ],
)
def test_dashboard_renders_active_target_trajectory_status_variants_without_exception(offset_kg, expected_status):
    at = AppTest.from_file("app/pages/Dashboard.py")

    _active_trajectory_state(at, offset_kg=offset_kg)
    at.run(timeout=15)

    assert not at.exception

    rendered = " ".join(str(element.value) for element in at.markdown)
    assert "Trajectoire cible" in rendered
    assert f"statut : {expected_status}" in rendered

def test_dashboard_renders_kpis_and_sections():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    markdown_values = [str(m.value) for m in at.markdown]
    assert any("Dashboard" in value for value in markdown_values)
    assert len(at.metric) >= 3
    plotly_elements = at.get("plotly_chart")
    assert len(plotly_elements) >= 2
    trace_names = []
    for chart in plotly_elements:
        spec = json.loads(chart.proto.spec)
        trace_names.extend(trace.get("name", "") for trace in spec.get("data", []))
    normalized_trace_names = " ".join(trace_names).lower()
    assert "poids" in normalized_trace_names
    assert "objectif" in normalized_trace_names or "cible" in normalized_trace_names
    target_traces = []
    for chart in plotly_elements:
        spec = json.loads(chart.proto.spec)
        target_traces.extend(
            trace for trace in spec.get("data", [])
            if "Trajectoire cible vers 80 kg au 16/12/2026" in trace.get("name", "")
        )
    assert target_traces
    trace = next(trace for trace in target_traces if trace["x"][0].startswith("2026-09-20"))
    assert trace["x"][0].startswith("2026-09-20")
    assert trace["y"][0] == 106.1
    assert trace["x"][-1].startswith("2026-12-16")
    assert trace["y"][-1] == 80.0
    assert len(trace["x"]) == 88
    assert all(not x.startswith("2026-12-17") for x in trace["x"])
    assert any("objectif" in str(c.value).lower() for c in at.caption)


def test_journal_loads_and_keeps_state_on_navigation():
    at = AppTest.from_file("app/pages/Journal.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    assert len(at.dataframe) >= 1
    assert "Extra Col" in at.session_state["working_data"].columns
    assert any("Qualité données" in str(info.value) for info in at.info)
    button_labels = [b.label for b in at.button]
    assert "Enregistrer les modifications" in button_labels
    downloads = at.get("download_button")
    download_labels = [d.label for d in downloads]
    assert "Exporter les données enregistrées" in download_labels

    # navigation simulée vers une autre page puis retour
    at_dash = AppTest.from_file("app/pages/Dashboard.py")
    for k in ["source_data", "working_data", "filtered_data", "raw_data", "target_weights", "fast_mode"]:
        at_dash.session_state[k] = at.session_state[k]
    at_dash.run()
    assert not at_dash.exception

    at_back = AppTest.from_file("app/pages/Journal.py")
    for k in ["source_data", "working_data", "filtered_data", "raw_data", "target_weights", "fast_mode"]:
        at_back.session_state[k] = at_dash.session_state[k]
    at_back.run()
    assert not at_back.exception
    assert len(at_back.session_state["working_data"]) == 80
    assert "Extra Col" in at_back.session_state["working_data"].columns


def test_predictions_render_multiple_sections_even_if_submodel_fails():
    at = AppTest.from_file("app/pages/Predictions.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    rendered_text = " ".join([str(s.value) for s in at.subheader] + [str(m.value) for m in at.markdown])
    assert "Leaderboard" in rendered_text
    assert "Projection selon vos mesures" in rendered_text
    tab_labels = [t.label for t in at.tabs]
    assert any("SARIMA" in t for t in tab_labels)
    assert any("Auto-ARIMA" in t for t in tab_labels)


def test_settings_exposes_five_goals():
    at = AppTest.from_file("app/pages/Settings.py")
    _state(at)
    at.run(timeout=10)
    assert not at.exception
    labels = [n.label for n in at.number_input]
    assert "Objectif 1 (kg)" in labels
    assert "Objectif 2 (kg)" in labels
    assert "Objectif 3 (kg)" in labels
    assert "Objectif 4 (kg)" in labels
    assert "Objectif 5 (kg)" in labels
    date_labels = [d.label for d in at.date_input]
    assert "Début du zoom trajectoire" in date_labels
    assert "Fin du zoom trajectoire" in date_labels
    assert at.session_state["zoom_target_start_date"] == pd.Timestamp("2026-09-20")
    assert at.session_state["zoom_target_end_date"] == pd.Timestamp("2026-12-16")


def test_dashboard_zoom_chart_uses_configured_period_and_handles_empty_data():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.run(timeout=10)

    assert not at.exception
    assert any("Aucune donnée de poids disponible sur cette période." in str(info.value) for info in at.info)

    plotly_elements = at.get("plotly_chart")
    assert len(plotly_elements) >= 2
    zoom_spec = json.loads(plotly_elements[1].proto.spec)
    xaxis = zoom_spec.get("layout", {}).get("xaxis", {})
    assert xaxis.get("range", [])[0].startswith("2026-09-20")
    assert xaxis.get("range", [])[1].startswith("2026-12-16")
    measured_traces = [trace for trace in zoom_spec.get("data", []) if trace.get("name") == "Poids mesuré"]
    assert measured_traces == []


def test_dashboard_zoom_chart_filters_measured_data_to_configured_period():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _active_trajectory_state(at)
    at.session_state["zoom_target_start_date"] = pd.Timestamp("2026-09-27")
    at.session_state["zoom_target_end_date"] = pd.Timestamp("2026-10-02")
    at.run(timeout=15)

    assert not at.exception
    plotly_elements = at.get("plotly_chart")
    assert len(plotly_elements) >= 2
    zoom_spec = json.loads(plotly_elements[1].proto.spec)
    measured_trace = next(trace for trace in zoom_spec.get("data", []) if trace.get("name") == "Poids mesuré")
    assert measured_trace["x"][0].startswith("2026-09-27")
    assert measured_trace["x"][-1].startswith("2026-10-02")
    assert all("2026-09-20" not in x for x in measured_trace["x"])


def test_dashboard_zoom_invalid_period_warns_without_exception():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _active_trajectory_state(at)
    at.session_state["zoom_target_start_date"] = pd.Timestamp("2026-12-16")
    at.session_state["zoom_target_end_date"] = pd.Timestamp("2026-09-20")
    at.run(timeout=15)

    assert not at.exception
    assert any("La date de début du zoom" in str(w.value) for w in at.warning)


def test_dashboard_migrates_legacy_four_goals_to_requested_five_goals():
    at = AppTest.from_file("app/pages/Dashboard.py")
    _state(at)
    at.session_state["target_weights"] = (100.0, 95.0, 90.0, 85.0)
    at.session_state["target_weight"] = 85.0
    at.run(timeout=10)
    assert not at.exception
    assert at.session_state["target_weights"] == (100.0, 95.0, 90.0, 85.0, 80.0)
    assert at.session_state["target_weight"] == 80.0
    assert any(f"Objectif 5: {format_fr_kg(80.0)}" in str(c.value) for c in at.caption)


def test_has_unsaved_changes_detects_real_dataframe_differences():
    saved = pd.DataFrame({"Date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")], "Poids (Kgs)": [80.0, 79.8]})
    assert has_unsaved_changes(saved.copy(), saved) is False
    changed = saved.copy(); changed.loc[0, "Poids (Kgs)"] = 80.2
    assert has_unsaved_changes(changed, saved) is True
    added = pd.concat([saved, pd.DataFrame({"Date": [pd.Timestamp("2026-01-03")], "Poids (Kgs)": [79.6]})], ignore_index=True)
    assert has_unsaved_changes(added, saved) is True
    assert has_unsaved_changes(saved.iloc[:1], saved) is True
    typed = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "Poids (Kgs)": ["80", "79,8"]})
    assert has_unsaved_changes(typed, saved) is False
    typed_decimal = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "Poids (Kgs)": ["80,0", "79,8"]})
    assert has_unsaved_changes(typed_decimal, saved) is False
    french_date = pd.DataFrame({"Date": ["01/01/2026", "02/01/2026"], "Poids (Kgs)": [80, 79.8]})
    assert has_unsaved_changes(french_date, saved) is False
    november_date = pd.DataFrame({"Date": ["01/11/2026"], "Poids (Kgs)": [80]})
    november_ts = pd.DataFrame({"Date": [pd.Timestamp("2026-11-01")], "Poids (Kgs)": [80.0]})
    assert has_unsaved_changes(november_date, november_ts) is False
    reindexed = saved.copy(); reindexed.index = [10, 11]
    assert has_unsaved_changes(reindexed, saved) is False
    reordered = saved.iloc[::-1].reset_index(drop=True)
    assert has_unsaved_changes(reordered, saved) is True
    custom_saved = saved.assign(Note=["a", "b"])
    custom_changed = custom_saved.copy(); custom_changed.loc[1, "Note"] = "c"
    assert has_unsaved_changes(custom_changed, custom_saved) is True
    different_date = saved.copy(); different_date.loc[1, "Date"] = pd.Timestamp("2026-01-03")
    assert has_unsaved_changes(different_date, saved) is True


def _whoop_daily_frame(days: int = 14) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": [55 + (i % 6) * 5 for i in range(days)],
            "HRV (ms)": [40 + (i % 5) * 2.5 for i in range(days)],
            "FC repos (bpm)": [54 + (i % 3) for i in range(days)],
            "Sommeil (heures)": [6.8 + (i % 4) * 0.3 for i in range(days)],
            "Performance sommeil (%)": [80 + (i % 5) * 3 for i in range(days)],
            "Sommeil profond (heures)": [1.2 + (i % 3) * 0.2 for i in range(days)],
            "Sommeil REM (heures)": [1.6 + (i % 4) * 0.15 for i in range(days)],
            "Strain": [9 + (i % 7) for i in range(days)],
            "Calories (kcal)": [2400 + (i % 6) * 90 for i in range(days)],
        }
    )


def _whoop_connected_state(at: AppTest) -> None:
    _state(at)
    at.session_state["whoop_token"] = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_at": (pd.Timestamp.utcnow() + pd.Timedelta(hours=2)).isoformat(),
        "scopes": ["offline", "read:recovery"],
        "token_type": "Bearer",
    }
    at.session_state["whoop_daily"] = _whoop_daily_frame()
    at.session_state["whoop_workouts"] = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-07")],
            "Début": [pd.Timestamp("2026-01-03 07:30"), pd.Timestamp("2026-01-07 18:10")],
            "Sport": ["Running", "Weightlifting"],
            "Durée (min)": [45.0, 60.0],
            "Strain séance": [9.4, 7.2],
            "Calories séance (kcal)": [520.0, 430.0],
            "FC moyenne (bpm)": [142.0, 118.0],
            "FC max (bpm)": [178.0, 150.0],
            "Distance (km)": [8.2, float("nan")],
        }
    )
    at.session_state["whoop_profile"] = {"first_name": "Test", "last_name": "Utilisateur"}
    at.session_state["whoop_last_sync"] = pd.Timestamp("2026-01-14")


def test_whoop_page_renders_connection_panel_without_credentials():
    at = AppTest.from_file("app/pages/Whoop.py")
    _state(at)
    at.run(timeout=15)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "WHOOP" in rendered
    assert "Connexion WHOOP" in rendered
    # Sans identifiants, aucune donnée WHOOP n'est inventée.
    assert at.session_state["whoop_daily"].empty


def test_whoop_page_renders_tabs_and_charts_when_session_holds_data():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    at.run(timeout=20)

    assert not at.exception
    tab_labels = [t.label for t in at.tabs]
    for expected in ["Vue d'ensemble", "Récupération", "Sommeil", "Effort", "Poids × WHOOP"]:
        assert expected in tab_labels
    assert len(at.get("plotly_chart")) >= 3
    assert len(at.metric) >= 3


def test_whoop_page_crosses_weight_and_whoop_without_touching_weight_data():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    before = at.session_state["working_data"].copy()
    at.run(timeout=20)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption)
    assert "Corrélations" in rendered
    # Non-régression : l'onglet WHOOP est en lecture seule sur les données de poids.
    pd.testing.assert_frame_equal(at.session_state["working_data"], before)
    assert len(at.session_state["working_data"]) == 80


def test_whoop_page_handles_connected_account_without_synced_data():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    at.session_state["whoop_daily"] = pd.DataFrame()
    at.session_state["whoop_workouts"] = pd.DataFrame()
    at.run(timeout=15)

    assert not at.exception
    assert any("Lancez une synchronisation" in str(info.value) for info in at.info)


def test_main_navigation_exposes_whoop_page_without_dropping_existing_pages():
    source = Path("Suivi_V1.py").read_text(encoding="utf-8")
    for page in ["Dashboard.py", "Journal.py", "Predictions.py", "Insights.py", "Settings.py", "Whoop.py"]:
        assert f"app/pages/{page}" in source
    assert 'title="Whoop"' in source


def _whoop_rich_state(at: AppTest, *, whoop_days: int = 45) -> None:
    """Historique suffisant pour débloquer toutes les analyses croisées.

    Les dates se terminent aujourd'hui : le filtre de période compte à partir
    du jour courant, comme le lecteur l'attend d'un libellé « 7 jours ».
    """
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=whoop_days, freq="D")
    at.session_state["whoop_daily"] = pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": [45 + (i % 9) * 6 for i in range(whoop_days)],
            "HRV (ms)": [38 + (i % 7) * 2.0 for i in range(whoop_days)],
            "FC repos (bpm)": [54 + (i % 4) for i in range(whoop_days)],
            "Sommeil (heures)": [6.4 + (i % 5) * 0.35 for i in range(whoop_days)],
            "Besoin de sommeil (heures)": [8.1] * whoop_days,
            "Dette de sommeil (heures)": [8.1 - (6.4 + (i % 5) * 0.35) for i in range(whoop_days)],
            "Performance sommeil (%)": [78 + (i % 6) * 3 for i in range(whoop_days)],
            "Efficacité sommeil (%)": [90 + (i % 3) for i in range(whoop_days)],
            "Régularité sommeil (%)": [0.0] * whoop_days,
            "Sommeil profond (heures)": [1.1 + (i % 4) * 0.2 for i in range(whoop_days)],
            "Sommeil REM (heures)": [1.5 + (i % 3) * 0.2 for i in range(whoop_days)],
            "Sommeil léger (heures)": [3.4 + (i % 5) * 0.1 for i in range(whoop_days)],
            "Éveil (heures)": [0.3 + (i % 3) * 0.1 for i in range(whoop_days)],
            "Fréquence respiratoire (resp/min)": [14.4 + (i % 5) * 0.15 for i in range(whoop_days)],
            "Heure de coucher": [-1.5 + (i % 4) * 0.4 for i in range(whoop_days)],
            "Strain": [8 + (i % 8) for i in range(whoop_days)],
            "Calories (kcal)": [2500 + (i % 7) * 120 for i in range(whoop_days)],
        }
    )
    # Deux jours de séance choisis dans la fenêtre, quelle que soit sa longueur.
    session_days = [dates[min(4, len(dates) - 1)], dates[min(8, len(dates) - 1)]]
    at.session_state["whoop_workouts"] = pd.DataFrame(
        {
            "Date": [session_days[0].normalize(), session_days[1].normalize()],
            "Début": [
                session_days[0] + pd.Timedelta(hours=18, minutes=30),
                session_days[1] + pd.Timedelta(hours=7, minutes=15),
            ],
            "Sport": ["boxing", "weightlifting"],
            "Durée (min)": [43.4038, 60.1234],
            "Strain séance": [13.9323, 7.812],
            "Calories séance (kcal)": [505.512, 236.0414],
            "FC moyenne (bpm)": [142.0, 115.0],
            "FC max (bpm)": [185.0, 170.0],
            "Distance (km)": [float("nan"), float("nan")],
            "Part enregistrée (%)": [100.0, 64.0],
            "Zone 0 (min)": [1.0, 5.0],
            "Zone 1 (min)": [4.0, 15.0],
            "Zone 2 (min)": [8.0, 20.0],
            "Zone 3 (min)": [12.0, 12.0],
            "Zone 4 (min)": [13.0, 6.0],
            "Zone 5 (min)": [5.0, 2.0],
        }
    )
    at.session_state["whoop_profile"] = {"first_name": "Test", "last_name": "Utilisateur"}
    at.session_state["whoop_last_sync"] = pd.Timestamp.now()

    # Le croisement poids × WHOOP exige que les deux sources couvrent la même
    # fenêtre : la fixture de poids partagée reste ancrée en janvier pour les
    # autres pages, celle-ci est donc posée localement.
    weights = pd.DataFrame(
        {
            "Date": dates,
            "Poids (Kgs)": [104 - index * 0.06 for index in range(whoop_days)],
        }
    )
    at.session_state["source_data"] = weights.copy()
    at.session_state["working_data"] = weights.copy()
    at.session_state["filtered_data"] = pd.DataFrame()
    at.session_state["filter_active"] = False
    at.session_state["raw_data"] = weights.copy()


def test_whoop_page_unlocks_energy_balance_and_lagged_correlations_with_enough_history():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Bilan énergétique estimé" in rendered
    assert "Corrélations décalées" in rendered
    assert "Charge d'entraînement" in rendered
    assert "Synthèse hebdomadaire" in rendered
    assert "Zones de récupération" in rendered
    assert "Dette de sommeil" in rendered
    # L'estimation est réellement calculée, pas seulement annoncée.
    assert not any("Estimation disponible à partir de" in str(info.value) for info in at.info)


def test_whoop_page_states_what_is_still_missing_on_a_short_history():
    """Cas réel d'un bracelet tout juste acheté : quelques jours de mesures."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at, whoop_days=4)
    at.run(timeout=30)

    assert not at.exception
    messages = " ".join(str(info.value) for info in at.info)
    # Plutôt que d'afficher une statistique non fiable, la page dit ce qui manque.
    assert "Estimation disponible à partir de" in messages
    assert "Corrélations calculées à partir de" in messages
    assert "Indicateur disponible à partir de" in messages


def test_whoop_page_breaks_lines_on_days_without_measurement():
    """Non-régression : un trou de mesure ne doit pas être relié par une droite."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    daily = at.session_state["whoop_daily"]
    holed = daily.drop(index=[5, 6, 7]).reset_index(drop=True)
    at.session_state["whoop_daily"] = holed
    # Un jour pesé sans mesure WHOOP reste une journée à montrer : pour obtenir
    # des journées réellement vides, la pesée doit manquer aussi.
    weights = at.session_state["working_data"]
    without = weights[weights["Date"].isin(holed["Date"])].reset_index(drop=True)
    at.session_state["working_data"] = without
    at.session_state["source_data"] = without.copy()
    at.session_state["raw_data"] = without.copy()
    at.run(timeout=30)

    assert not at.exception
    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    # Les sparklines d'accompagnement sont des tracés anonymes et sans trou par
    # construction : seules les séries nommées portent des données quotidiennes.
    named_traces = [
        trace
        for spec in specs
        for trace in spec.get("data", [])
        if trace.get("type") in {"scatter", "scattergl"} and "y" in trace and trace.get("name")
    ]
    assert named_traces
    # Les courbes déclarent explicitement ne pas combler les trous...
    assert all(trace.get("connectgaps") is False for trace in named_traces)
    # ...et les jours retirés sont bien présents en valeur vide.
    assert any(any(value is None for value in trace["y"]) for trace in named_traces)


def test_whoop_page_formats_workout_tables_without_raw_precision():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    tables = [frame.value for frame in at.dataframe]
    flattened = " ".join(table.to_csv(index=False) for table in tables)
    # Les décimales brutes de l'API ne sont plus affichées telles quelles.
    assert "43.4038" not in flattened
    assert "505.512" not in flattened
    # Les noms de sport sont traduits et les dates écrites en français.
    assert "Boxe" in flattened
    assert "2026-01-05 00:00:00" not in flattened
    assert any(month in flattened for month in ("janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."))
    # Une durée se lit en heures et minutes, pas en décimales de minute.
    assert "43 min" in flattened


def test_whoop_page_shows_narrative_insights_and_a_recovery_gauge():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Ce que disent vos données" in rendered
    # Les constats sont rédigés, pas seulement tabulés.
    assert "suivi-insight-card" in rendered

    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    gauges = [trace for spec in specs for trace in spec.get("data", []) if trace.get("type") == "indicator"]
    assert gauges, "la jauge de récupération doit être rendue"
    assert gauges[0]["gauge"]["axis"]["range"] == [0, 100]


def test_whoop_page_period_filter_scopes_every_tab():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at, whoop_days=60)
    at.run(timeout=30)
    assert not at.exception

    period = next(radio for radio in at.radio if radio.label == "Période analysée")
    assert period.options == ["7 jours", "30 jours", "90 jours", "Tout"]

    period.set_value("7 jours").run(timeout=30)
    assert not at.exception

    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    dated = [
        trace
        for spec in specs
        for trace in spec.get("data", [])
        if trace.get("name") and isinstance(trace.get("x"), list) and trace["x"] and str(trace["x"][0]).startswith("20")
    ]
    assert dated
    # 60 jours importés, 7 demandés : aucune série ne doit dépasser la tranche.
    assert all(len(trace["x"]) <= 7 for trace in dated)


def test_whoop_page_compares_weight_on_a_single_indexed_axis():
    """Non-régression : deux échelles verticales fabriqueraient une corrélation visuelle."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    assert all("yaxis2" not in spec.get("layout", {}) for spec in specs)
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "base 100" in rendered.lower()


def test_whoop_page_offers_a_table_view_for_its_charts():
    """Une valeur portée par une couleur doit rester lisible autrement."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    assert len(at.dataframe) >= 3


def test_whoop_page_warns_when_measurements_stopped_days_ago():
    """Rien n'indiquait que les moyennes « récentes » portaient sur des jours anciens."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    stale = at.session_state["whoop_daily"].copy()
    stale["Date"] = stale["Date"] - pd.Timedelta(days=8)
    at.session_state["whoop_daily"] = stale
    at.run(timeout=30)

    assert not at.exception
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "Dernière mesure" in warnings
    assert "il y a 8 jours" in warnings


def test_whoop_page_period_counts_from_today_not_from_the_last_measurement():
    """Après une semaine sans porter le bracelet, « 7 jours » doit rester 7 jours."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    stale = at.session_state["whoop_daily"].copy()
    stale["Date"] = stale["Date"] - pd.Timedelta(days=30)
    at.session_state["whoop_daily"] = stale
    at.run(timeout=30)
    assert not at.exception

    next(radio for radio in at.radio if radio.label == "Période analysée").set_value("7 jours").run(timeout=30)

    assert not at.exception
    # Aucune mesure dans les 7 derniers jours réels : la page le dit au lieu
    # d'afficher silencieusement une autre tranche.
    assert any("Aucune mesure WHOOP sur cette période" in str(info.value) for info in at.info)


def test_whoop_page_writes_dates_in_french():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    labels = [
        label
        for spec in specs
        for label in spec.get("layout", {}).get("xaxis", {}).get("ticktext", []) or []
    ]
    assert labels, "les axes temporels doivent porter des étiquettes explicites"
    assert all(month not in label for label in labels for month in ("Jan", "Feb", "Aug", "Sep", "Oct", "Dec"))


def test_whoop_page_leads_with_the_readings_the_whoop_app_cannot_give():
    """La lecture jour par jour et le croisement avec le poids passent devant
    les onglets que l'application WHOOP fournit déjà."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    labels = [tab.label for tab in at.tabs]
    assert labels[:3] == ["Vue d'ensemble", "Jour par jour", "Poids × WHOOP"]
    # Les onglets redondants avec l'app WHOOP viennent après.
    assert labels.index("Jour par jour") < labels.index("Récupération")


def test_whoop_page_survives_two_recoveries_on_the_same_day():
    """Des dates dupliquées faisaient remonter une ValueError jusqu'à la page."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    daily = at.session_state["whoop_daily"]
    duplicated = pd.concat([daily, daily.tail(1)], ignore_index=True)
    at.session_state["whoop_daily"] = duplicated
    at.run(timeout=30)

    assert not at.exception


def test_whoop_page_reads_bedtimes_as_a_clock_never_as_signed_decimals():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    bedtime_axes = [
        spec["layout"]["yaxis"]["ticktext"]
        for spec in specs
        if spec.get("layout", {}).get("yaxis", {}).get("ticktext")
        and any(trace.get("name") == "Heure de coucher" for trace in spec.get("data", []))
    ]
    assert bedtime_axes, "le graphique des couchers doit porter un axe en horloge"
    assert all(":" in label for label in bedtime_axes[0])


def test_whoop_page_hides_the_connection_plumbing_behind_the_data():
    """Le bouton de déconnexion occupait le haut de page avant toute mesure."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    # La déconnexion reste accessible, mais dans le panneau replié.
    assert "Déconnecter WHOOP" in [button.label for button in at.button]
    assert any("synchronisation" in str(exp.label).lower() for exp in at.get("expander"))


def test_whoop_page_renders_a_dated_journal_with_sessions_under_their_day():
    """Demande explicite : voir les récupérations par date, et les séances aussi."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    session_day = at.session_state["whoop_workouts"]["Date"].iloc[0]
    at.run(timeout=30)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "suivi-day-card" in rendered
    # Chaque journée porte sa date en toutes lettres.
    assert any(weekday in rendered for weekday in ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"))
    # La séance apparaît avec son heure de début, sous sa date.
    assert "18:30" in rendered
    assert "Boxe" in rendered
    assert pd.Timestamp(session_day).strftime("%d") in rendered


def test_whoop_journal_hides_empty_days_until_asked():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    daily = at.session_state["whoop_daily"]
    holed = daily.drop(index=[5, 6, 7]).reset_index(drop=True)
    at.session_state["whoop_daily"] = holed
    # Un jour pesé sans mesure WHOOP reste une journée à montrer : pour obtenir
    # des journées réellement vides, la pesée doit manquer aussi.
    weights = at.session_state["working_data"]
    without = weights[weights["Date"].isin(holed["Date"])].reset_index(drop=True)
    at.session_state["working_data"] = without
    at.session_state["source_data"] = without.copy()
    at.session_state["raw_data"] = without.copy()
    at.run(timeout=30)

    assert not at.exception
    toggle = next(item for item in at.toggle if "sans aucune mesure" in item.label)
    assert toggle.value is False
    rendered_before = " ".join(str(m.value) for m in at.markdown)

    toggle.set_value(True).run(timeout=30)
    assert not at.exception
    rendered_after = " ".join(str(m.value) for m in at.markdown)
    # Les jours vides apparaissent, signalés comme tels.
    assert rendered_after.count("suivi-day-card") > rendered_before.count("suivi-day-card")
    assert "Aucune mesure ce jour-là" in rendered_after


def test_whoop_page_shows_the_physiological_watch_without_diagnosing():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    daily = at.session_state["whoop_daily"].copy()
    last = daily.index[-1]
    daily.loc[last, "FC repos (bpm)"] = 72.0
    daily.loc[last, "HRV (ms)"] = 22.0
    daily.loc[last, "Fréquence respiratoire (resp/min)"] = 18.5
    at.session_state["whoop_daily"] = daily
    at.run(timeout=30)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption)
    assert "Veille physiologique" in rendered
    # L'avertissement est explicite et aucune pathologie n'est nommée.
    assert "ne constituent pas un diagnostic" in rendered
    assert "professionnel de santé" in rendered


def test_whoop_page_exports_the_journal():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)

    assert not at.exception
    assert "Exporter le journal (CSV)" in [button.label for button in at.get("download_button")]


def test_whoop_page_states_where_the_weight_is_going_before_anything_else():
    """Une prise de poids ne doit pas rester invisible derrière un constat sur le sommeil."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at, whoop_days=45)
    weights = at.session_state["working_data"].copy()
    # Profil de prise de poids réelle sur la période.
    weights["Poids (Kgs)"] = [100.0 + index * 0.06 for index in range(len(weights))]
    for key in ("working_data", "source_data", "raw_data"):
        at.session_state[key] = weights.copy()
    at.run(timeout=30)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Votre poids augmente" in rendered
    assert "kg par semaine" in rendered


def test_whoop_page_projects_the_arrival_against_the_deadline():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at, whoop_days=45)
    at.run(timeout=30)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Au rythme actuel" in rendered


def test_whoop_page_shows_the_recovery_streaks():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at, whoop_days=45)
    at.run(timeout=30)

    assert not at.exception
    labels = [metric.label for metric in at.metric]
    assert "Série en cours" in labels
    assert "Plus longue série rouge" in labels


def test_whoop_page_reports_what_training_really_costs():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at, whoop_days=45)
    at.run(timeout=30)

    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Ce que pèsent vraiment vos séances" in rendered
    assert "Part de l'entraînement" in [metric.label for metric in at.metric]


def test_whoop_effort_tab_shows_load_tolerance_without_any_recorded_workout():
    """WHOOP mesure un strain continu même sans séance enregistrée.

    Le panneau de tolérance à la charge ne dépend que des données
    quotidiennes ; placé après le retour anticipé « aucune séance », il
    devenait inatteignable pour un porteur qui ne logue aucun entraînement.
    """
    at = AppTest.from_file("app/pages/Whoop.py", default_timeout=30)
    _state(at)
    _whoop_connected_state(at)
    _whoop_rich_state(at, whoop_days=45)
    at.session_state["whoop_workouts"] = pd.DataFrame()
    at.run()

    assert not at.exception
    rendered = " ".join(str(element.value) for element in at.markdown)
    assert "récupération du lendemain matin" in rendered, (
        "le panneau de charge doit précéder le retour anticipé sur les séances"
    )


def test_whoop_effort_tab_lists_each_session_by_date_with_the_next_morning():
    """Ce que le lecteur demande : ses séances de boxe, à leur date, et ce qu'elles ont laissé."""
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)
    assert not at.exception

    effort = at.tabs[5]
    rendered = " ".join(str(m.value) for m in effort.markdown)
    assert "Vos séances, par date" in rendered
    sessions = effort.dataframe[0].value
    assert list(sessions.columns)[:3] == ["Date", "Début", "Sport"]
    assert "Récupération du lendemain (%)" in sessions.columns
    assert "Zones 4–5 (min)" in sessions.columns
    # Du plus récent au plus ancien, jour de semaine en tête, heure de début seule.
    dates = sessions["Date"].tolist()
    assert dates[0].split(" ")[0] in {"lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."}
    assert sessions["Début"].tolist()[1] == "18:30"
    assert set(sessions["Sport"]) == {"Boxe", "Musculation"}
    # Une séance captée à 64 % est signalée plutôt que lue comme une séance légère.
    captions = " ".join(str(c.value) for c in effort.caption)
    assert "à moins de 80 %" in captions
    # Les zones de FC alimentent la répartition d'intensité.
    assert "Répartition de l'intensité" in rendered


def test_whoop_recovery_tab_lists_each_morning_by_date_with_its_zone():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)
    assert not at.exception

    recovery = at.tabs[3]
    rendered = " ".join(str(m.value) for m in recovery.markdown)
    assert rendered.index("Récupération, jour par jour") < rendered.index("Zones de récupération")
    log = recovery.dataframe[0].value
    assert list(log.columns) == ["Date", "Récupération (%)", "Zone", "HRV (ms)", "FC repos (bpm)", "Sommeil (heures)", "Strain de la veille", "Strain du jour"]
    # Quatorze lignes visibles d'emblée, l'historique complet derrière un expander.
    assert len(log) == 14
    assert any("Voir toutes les journées notées" in expander.label for expander in recovery.expander)
    assert log["Zone"].iloc[0] in {"🟢 Vert", "🟡 Jaune", "🔴 Rouge"}
    # La colonne Date porte le jour de semaine.
    assert log["Date"].iloc[0].split(" ")[0] in {"lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."}
    # HRV et FC repos lissées face à leur plage, avec un statut lisible.
    labels = [metric.label for metric in recovery.metric]
    assert "Plage habituelle" in labels
    assert any("Moyenne des 7 derniers jours" == label for label in labels)


def test_whoop_charts_carry_the_whoop_reference_bands_and_the_full_night():
    at = AppTest.from_file("app/pages/Whoop.py")
    _whoop_connected_state(at)
    _whoop_rich_state(at)
    at.run(timeout=30)
    assert not at.exception

    specs = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    band_labels = {
        annotation.get("text")
        for spec in specs
        for annotation in spec.get("layout", {}).get("annotations", [])
    }
    # Niveaux de strain et zones de récupération dessinés en fond des courbes.
    assert {"Léger", "Modéré", "Élevé", "Maximal"} <= band_labels
    assert {"Rouge", "Jaune", "Vert"} <= band_labels
    stage_names = {
        trace.get("name")
        for spec in specs
        for trace in spec.get("data", [])
        if trace.get("type") == "bar" and str(trace.get("name", "")).startswith(("Sommeil", "Éveil"))
    }
    assert stage_names == {"Sommeil profond", "Sommeil REM", "Sommeil léger", "Éveil"}
