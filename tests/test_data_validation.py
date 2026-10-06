from __future__ import annotations

import datetime as dt
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.core.data import clean_weight_dataframe_with_report, read_weight_csv


def _main_module():
    source = Path("Suivi_V1.py").read_text().split("st.set_page_config", 1)[0]
    namespace = {"__name__": "suivi_validation_test"}
    exec(compile(source, "Suivi_V1.py", "exec"), namespace)
    return namespace


def _saved():
    return pd.DataFrame({"Date": [pd.Timestamp("2026-10-06")], "Poids (Kgs)": [80.0], "Notes": ["conserver"]})


@pytest.mark.parametrize("payload", [
    b'Date,Poids,Notes\n06/10/2026,"80,2","avant, repas"\n',
    'Date;Poids;Notes\n06/10/2026;80,2;à jeun\n'.encode('utf-8-sig'),
    'Date\tPoids\tNotes\n06/10/2026\t80,2\tà jeun\n'.encode('cp1252'),
    '\n \t\r\nDate;Poids;Notes\n06/10/2026;80,2;à jeun\n'.encode('utf-8-sig'),
])
def test_csv_dialects_preserve_french_values_and_notes(payload):
    cleaned, quality = clean_weight_dataframe_with_report(read_weight_csv(BytesIO(payload)))
    assert quality.invalid_rows == 0
    assert cleaned["Date"].iloc[0] == pd.Timestamp("2026-10-06")
    assert cleaned["Poids (Kgs)"].iloc[0] == 80.2
    assert "Notes" in cleaned.columns


@pytest.mark.parametrize("payload", [
    b'Date,Poids,Poids\n06/10/2026,80,81\n',
    b'Date,Poids,Poids (kg)\n06/10/2026,80,81\n',
    b'Date,Notes\n06/10/2026,test\n',
    b'',
])
def test_csv_rejects_ambiguous_or_missing_columns(payload):
    with pytest.raises(ValueError):
        read_weight_csv(BytesIO(payload))


@pytest.mark.parametrize("payload", [
    b'Date,Poids\nbad-date,80\n',
    b'Date,Poids\n',
    b'Date,Poids\n06/10/2026,inf\n',
    b'Date,Poids,Poids (kg)\n06/10/2026,80,81\n',
])
def test_invalid_import_keeps_source_and_session_work(monkeypatch, payload):
    module = _main_module()
    original = _saved()
    state = {"source_data": original.copy(), "working_data": original.copy(), "data_source": "existing"}
    monkeypatch.setattr(module["st"], "session_state", state)
    with pytest.raises(ValueError):
        module["_import_local_csv"](BytesIO(payload))
    pd.testing.assert_frame_equal(state["working_data"], original)
    pd.testing.assert_frame_equal(state["source_data"], original)
    assert state["data_source"] == "existing"


def test_empty_remote_reload_keeps_session_work(monkeypatch):
    module = _main_module()
    original = _saved()
    state = {"source_data": original.copy(), "working_data": original.copy()}
    monkeypatch.setattr(module["st"], "session_state", state)
    monkeypatch.setattr(module["st"], "secrets", {})
    module["load_remote_csv_with_report"] = lambda _: (original.iloc[:0], {})
    with pytest.raises(ValueError, match="aucune mesure valide"):
        module["_load_from_source"]()
    pd.testing.assert_frame_equal(state["working_data"], original)
    pd.testing.assert_frame_equal(state["source_data"], original)


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_remote_request_failure_is_bounded_and_preserves_session(monkeypatch, failure):
    module = _main_module()
    original = _saved()
    state = {"source_data": original.copy(), "working_data": original.copy(), "data_source": "existing"}
    monkeypatch.setattr(module["st"], "session_state", state)
    monkeypatch.setattr(module["st"], "secrets", {"data_url": "https://example.test/failed.csv"})
    response = Mock(content=b"not a csv")
    response.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
    transport = Mock(side_effect=requests.Timeout("read timeout")) if failure == "timeout" else Mock(return_value=response)
    monkeypatch.setattr("requests.get", transport)
    module["load_remote_csv_with_report"].clear()
    with pytest.raises(requests.RequestException):
        module["_load_from_source"]()
    connect_timeout, read_timeout = transport.call_args.kwargs["timeout"]
    assert 0 < connect_timeout <= 10
    assert 0 < read_timeout <= 30
    pd.testing.assert_frame_equal(state["working_data"], original)
    pd.testing.assert_frame_equal(state["source_data"], original)
    assert state["data_source"] == "existing"


def test_remote_csv_accepts_the_same_french_dialect_as_local_import(monkeypatch):
    module = _main_module()
    payload = 'Date;Poids;Notes\n06/10/2026;80,2;à jeun\n'.encode("utf-8-sig")
    response = Mock(content=payload)
    monkeypatch.setattr("requests.get", Mock(return_value=response))
    module["load_remote_csv_with_report"].clear()
    remote, quality = module["load_remote_csv_with_report"]("https://example.test/french.csv")
    local, _ = clean_weight_dataframe_with_report(read_weight_csv(BytesIO(payload)))
    pd.testing.assert_frame_equal(remote, local)
    assert quality["source"] == "google_sheets"
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize("state", [
    {"working_data": _saved(), "source_data": _saved().iloc[:0]},
    {"working_data": _saved().iloc[:0], "source_data": _saved().iloc[:0], "data_initialized": True},
])
def test_init_does_not_replace_local_work_or_intentionally_empty_state(monkeypatch, state):
    module = _main_module()
    monkeypatch.setattr(module["st"], "session_state", state)
    module["_load_from_source"] = lambda: pytest.fail("initialisation intempestive")
    module["init_data_once"]()


def test_numeric_and_string_excel_dates_have_identical_calendar_days():
    frame = pd.DataFrame({"Date": [46000, "46000", 46000.5, "06/10/2026", "2026-10-06"], "Poids": [80.0] * 5})
    cleaned, _ = clean_weight_dataframe_with_report(frame)
    assert cleaned["Date"].tolist() == [pd.Timestamp("2025-12-09")] * 3 + [pd.Timestamp("2026-10-06")] * 2


def test_weight_cleaning_rejects_nonfinite_but_does_not_impose_clinical_ranges():
    frame = pd.DataFrame({"Date": ["06/10/2026"] * 6, "Poids": [np.inf, -np.inf, 0, -1, 1, 10000]})
    cleaned, quality = clean_weight_dataframe_with_report(frame)
    assert quality.invalid_rows == 4
    assert cleaned["Poids (Kgs)"].tolist() == [1, 10000]
    assert "poids non fini" in quality.rejected_rows[0].reasons


def test_invalid_journal_edit_blocks_saving_and_preserves_existing_measurement():
    at = AppTest.from_file("app/pages/Journal.py")
    original = _saved()
    at.session_state["working_data"] = original.copy()
    at.run()
    at.session_state["journal_editor_0"] = {"edited_rows": {0: {"Date": None}}, "added_rows": [], "deleted_rows": []}
    at.run()
    assert at.session_state["journal_has_unsaved_changes"] is True
    next(button for button in at.button if button.label == "Enregistrer les modifications").click().run()
    assert not at.exception
    assert at.error
    pd.testing.assert_frame_equal(at.session_state["working_data"], original)


def test_incomplete_new_journal_row_is_dirty_and_cannot_be_saved():
    at = AppTest.from_file("app/pages/Journal.py")
    original = _saved()
    at.session_state["working_data"] = original.copy()
    at.run()
    at.session_state["journal_editor_0"] = {"edited_rows": {}, "added_rows": [{"Poids (Kgs)": 81.0}], "deleted_rows": []}
    at.run()
    assert at.session_state["journal_has_unsaved_changes"] is True
    assert any("pas encore enregistrées" in message.value for message in at.warning)
    next(button for button in at.button if button.label == "Ajouter").click().run()
    pd.testing.assert_frame_equal(at.session_state["working_data"], original)
    assert not at.exception


@pytest.mark.parametrize("weight", [10000.0, float("inf")])
def test_journal_handles_preexisting_weight_outside_quick_add_bounds(weight):
    at = AppTest.from_file("app/pages/Journal.py")
    original = _saved()
    original["Poids (Kgs)"] = weight
    at.session_state["working_data"] = original
    at.run()
    assert not at.exception
    assert next(widget for widget in at.number_input if widget.label == "Poids (kg)").value == 80.0
    assert at.warning


@pytest.mark.parametrize("invalid_field", ["negative_goal", "increasing_goals", "reversed_zoom"])
def test_invalid_settings_are_atomic(invalid_field):
    at = AppTest.from_file("app/pages/Settings.py")
    before = {
        "target_weights": (100.0, 95.0, 90.0, 85.0, 80.0), "target_weight": 80.0,
        "height_cm": 182.0, "height_m": 1.82, "duplicate_strategy": "garder_la_derniere",
        "window_size": 7, "zoom_target_start_date": pd.Timestamp("2026-10-01"),
        "zoom_target_end_date": pd.Timestamp("2026-12-31"),
    }
    for key, value in before.items():
        at.session_state[key] = value
    at.run()
    next(widget for widget in at.number_input if widget.label == "Taille (cm)").set_value(190.0)
    at.slider[0].set_value(14)
    if invalid_field == "reversed_zoom":
        next(widget for widget in at.date_input if widget.label == "Début du zoom trajectoire").set_value(dt.date(2027, 1, 1))
    else:
        next(widget for widget in at.number_input if widget.label == "Objectif 5 (kg)").set_value(-5.0 if invalid_field == "negative_goal" else 110.0)
    next(button for button in at.button if button.label == "Enregistrer").click().run()
    assert not at.exception
    assert at.error and not at.success
    for key, value in before.items():
        assert at.session_state[key] == value


@pytest.mark.parametrize("changed", [False, True])
def test_sidebar_requires_replacement_confirmation_only_for_local_changes(changed):
    source = Path("Suivi_V1.py").read_text().split("st.set_page_config", 1)[0]
    source += '\nst.session_state["source_data"] = pd.DataFrame({"Date": [pd.Timestamp("2026-10-06")], "Poids (Kgs)": [80.0]})\n'
    source += f'\nst.session_state["working_data"] = pd.DataFrame({{"Date": [pd.Timestamp("2026-10-06")], "Poids (Kgs)": [{79.0 if changed else 80.0}]}})\n'
    at = AppTest.from_string(source + '\nsidebar_controls()\n').run()
    assert not at.exception
    replacement_buttons = [button for button in at.button if button.label in {"Recharger Google Sheets", "Réinitialiser la session"}]
    assert len(replacement_buttons) == 2
    assert all(button.disabled is changed for button in replacement_buttons)
    if changed:
        at.checkbox[0].check().run()
        assert all(not button.disabled for button in at.button)
    else:
        assert not at.checkbox


@pytest.mark.parametrize("action, expected_weight", [
    ("Réinitialiser la session", 80.0),
    ("Recharger Google Sheets", 79.0),
])
def test_sidebar_replacement_discards_editor_deltas_and_consumes_confirmation(action, expected_weight):
    source = Path("Suivi_V1.py").read_text().split("st.set_page_config", 1)[0]
    source += '''
from pathlib import Path
if not st.session_state.get("integration_initialized", False):
    set_source_data(pd.DataFrame({"Date": [pd.Timestamp("2026-10-06")], "Poids (Kgs)": [80.0]}), "csv_local")
    st.session_state["integration_initialized"] = True
# Transport remplacé uniquement dans le banc de test ; le vrai import reste utilisé.
def _load_from_source():
    _import_local_csv(BytesIO(b"Date,Poids\\n06/10/2026,79.0\\n"))
sidebar_controls()
exec(compile(Path("app/pages/Journal.py").read_text(), "app/pages/Journal.py", "exec"))
'''
    at = AppTest.from_string(source).run()
    assert not at.exception
    revision = at.session_state["journal_editor_revision"]
    at.session_state[f"journal_editor_{revision}"] = {
        "edited_rows": {0: {"Poids (Kgs)": 81.0}}, "added_rows": [], "deleted_rows": [],
    }
    at.run().run()
    assert at.session_state["journal_has_unsaved_changes"] is True
    next(widget for widget in at.checkbox if widget.label == "Remplacer mes modifications de session").check().run()
    next(button for button in at.button if button.label == action).click().run()
    assert not at.exception
    assert at.session_state["working_data"]["Poids (Kgs)"].tolist() == [expected_weight]
    assert at.session_state["journal_has_unsaved_changes"] is False
    # Sauver immédiatement après le remplacement ne ressuscite pas la saisie81.
    next(button for button in at.button if button.label == "Enregistrer les modifications").click().run()
    assert not at.exception
    assert at.session_state["working_data"]["Poids (Kgs)"].tolist() == [expected_weight]
    revision = at.session_state["journal_editor_revision"]
    at.session_state[f"journal_editor_{revision}"] = {
        "edited_rows": {0: {"Poids (Kgs)": 82.0}}, "added_rows": [], "deleted_rows": [],
    }
    at.run().run()
    assert at.session_state["journal_has_unsaved_changes"] is True
    confirmation = next(widget for widget in at.checkbox if widget.label == "Remplacer mes modifications de session")
    assert confirmation.value is False
    assert next(button for button in at.button if button.label == action).disabled is True
