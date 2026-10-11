import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.core.data import clean_weight_dataframe_with_report, data_quality_report, prepare_analysis_data
from app.core.data_editing import apply_editor_changes, has_unsaved_changes
from app.core.time_utils import normalize_datetime_series


def _saved():
    return pd.DataFrame({"Date": pd.to_datetime(["2026-10-01", "2026-10-02"]), "Poids (Kgs)": [80.0, 79.0], "Notes": ["matin", "soir"]})


@pytest.mark.parametrize("count", [1, 2, 7, 8, 30])
def test_quality_counts_calendar_days_inclusively(count):
    frame = pd.DataFrame({"Date": pd.date_range("2026-10-01", periods=count), "Poids (Kgs)": [80.0] * count})
    report = data_quality_report(frame)
    assert report["missing_days"] == 0
    assert report["coverage_pct"] == 100
    assert report["weekly_measurements"] == min(count, 7)
    assert np.isfinite(report["irregularity"])


def test_quality_gaps_are_between_distinct_valid_days():
    frame = pd.DataFrame({"Date": ["2026-10-01 08:00", "2026-10-01 20:00", "2026-10-03", "bad"], "Poids (Kgs)": [80, 81, 79, float("inf")]})
    original = frame.copy(deep=True)
    report = data_quality_report(frame)
    assert report["duplicates"] == 1
    assert report["coverage_pct"] == 66.7
    assert report["missing_days"] == 1
    assert report["irregularity"] == 0
    pd.testing.assert_frame_equal(frame, original)


def test_cleaning_duplicate_index_keeps_distinct_rejection_reasons():
    frame = pd.DataFrame({"Date": ["bad", "2026-10-01", "2026-10-02"], "Poids": [80, -1, 79]}, index=[4, 4, 4])
    cleaned, report = clean_weight_dataframe_with_report(frame)
    assert cleaned["Poids (Kgs)"].tolist() == [79]
    assert [(r.index, r.reasons) for r in report.rejected_rows] == [
        (4, ["date invalide ou vide"]), (4, ["poids nul ou négatif"]),
    ]


def test_last_daily_weight_uses_original_time_before_normalization():
    frame = pd.DataFrame({"Date": ["2026-10-01 18:00", "2026-10-01 08:00"], "Poids (Kgs)": [79.0, 80.0]})
    original = frame.copy(deep=True)
    cleaned, _ = clean_weight_dataframe_with_report(frame)
    assert len(cleaned) == 2
    for source in [frame, cleaned]:
        assert prepare_analysis_data(source)["Poids (Kgs)"].tolist() == [79.0]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("value, day", [
    ("2026-03-29T00:00:00+01:00", "2026-03-29"),
    ("2026-10-25T00:00:00+02:00", "2026-10-25"),
    ("2026-10-25T23:00:00-04:00", "2026-10-25"),
    (pd.Timestamp("2026-10-25", tz="Europe/Paris"), "2026-10-25"),
])
def test_journal_dates_keep_local_day_and_instant_parser_keeps_utc(value, day):
    frame = pd.DataFrame({"Date": [value], "Poids (Kgs)": [80.0]})
    cleaned, _ = clean_weight_dataframe_with_report(frame)
    assert cleaned["Date"].iloc[0] == pd.Timestamp(day)
    assert not has_unsaved_changes(frame, cleaned)
    assert normalize_datetime_series([value]).iloc[0] == pd.Timestamp(value).tz_convert(None)


def _journal():
    source = '''
import streamlit as st
from pathlib import Path
if st.checkbox("Afficher le journal", value=True, key="show_journal"):
    exec(compile(Path("app/pages/Journal.py").read_text(), "app/pages/Journal.py", "exec"))
else:
    st.write("Autre page")
'''
    at = AppTest.from_string(source)
    at.session_state["working_data"] = _saved()
    return at.run()


@pytest.mark.parametrize("edits", [
    {"edited_rows": {0: {"Poids (Kgs)": 78.0}}, "added_rows": [], "deleted_rows": []},
    {"edited_rows": {}, "added_rows": [{"Poids (Kgs)": 77.0}], "deleted_rows": [0]},
])
def test_journal_draft_survives_widget_cleanup_without_saving(edits):
    at = _journal()
    at.session_state["journal_editor_0"] = edits
    at.run()
    draft = at.session_state["journal_draft"].copy(deep=True)
    assert at.session_state["journal_has_unsaved_changes"]
    at.checkbox(key="show_journal").uncheck().run()
    assert "journal_editor_0" not in at.session_state
    at.checkbox(key="show_journal").check().run()
    assert not at.exception
    pd.testing.assert_frame_equal(at.session_state["journal_draft"], draft)
    pd.testing.assert_frame_equal(at.session_state["working_data"], _saved())
    assert at.session_state["journal_has_unsaved_changes"]


def test_last_edit_sent_with_navigation_is_not_lost():
    at = _journal()
    # AppTest has no data-editor control. Send the actual browser widget payload
    # so its callback runs; assigning session_state directly bypasses callbacks.
    states = at._tree.get_widget_states()
    next(widget for widget in states.widgets if widget.id.endswith("-show_journal")).bool_value = False
    editor = next(element for element in at.get("arrow_data_frame") if element.proto.id.endswith("-journal_editor_0"))
    states.widgets.add(id=editor.proto.id, string_value=json.dumps({"edited_rows": {"0": {"Poids (Kgs)": 77.0}}, "added_rows": [], "deleted_rows": []}))
    at._run(states)
    at.checkbox(key="show_journal").check().run()
    assert not at.exception
    assert at.session_state["journal_draft"]["Poids (Kgs)"].tolist() == [77, 79]
    pd.testing.assert_frame_equal(at.session_state["working_data"], _saved())


def test_editor_callback_applies_delta_once_and_keeps_dates_and_incomplete_rows():
    base = _saved()
    changes = {"edited_rows": {0: {"Date": "2026-10-04", "Notes": "corrigée"}},
               "deleted_rows": [1], "added_rows": [{"Date": "2026-10-05", "Poids (Kgs)": None}]}
    expected = pd.DataFrame({"Date": pd.to_datetime(["2026-10-04", "2026-10-05"]), "Poids (Kgs)": [80.0, np.nan], "Notes": ["corrigée", None]})
    pd.testing.assert_frame_equal(apply_editor_changes(base, changes), expected)
    pd.testing.assert_frame_equal(apply_editor_changes(base, changes), expected)
    pd.testing.assert_frame_equal(base, _saved())


def test_save_then_next_edit_preserves_both_changes_and_export_source():
    at = _journal()
    at.session_state["journal_editor_0"] = {"edited_rows": {0: {"Poids (Kgs)": 78.0}}, "added_rows": [], "deleted_rows": []}
    at.run()
    next(b for b in at.button if b.label == "Enregistrer les modifications").click().run()
    assert not at.exception
    assert at.success
    revision = at.session_state["journal_editor_revision"]
    at.session_state[f"journal_editor_{revision}"] = {"edited_rows": {1: {"Notes": "nouvelle note"}}, "added_rows": [], "deleted_rows": []}
    at.run()
    next(b for b in at.button if b.label == "Enregistrer les modifications").click().run()
    assert not at.exception
    saved = at.session_state["working_data"]
    assert saved["Poids (Kgs)"].tolist() == [78, 79]
    assert saved["Notes"].tolist() == ["matin", "nouvelle note"]
    assert not at.session_state["journal_has_unsaved_changes"]


def test_cancel_draft_restores_saved_work_without_resetting_source():
    at = _journal()
    at.session_state["source_data"] = _saved().assign(Notes="source différente")
    at.session_state["journal_editor_0"] = {"edited_rows": {}, "added_rows": [], "deleted_rows": [0]}
    at.run()
    next(b for b in at.button if b.label == "Annuler les modifications de l’éditeur").click().run()
    assert not at.exception
    pd.testing.assert_frame_equal(at.session_state["working_data"], _saved())
    pd.testing.assert_frame_equal(at.session_state["journal_draft"], _saved())
    assert not at.session_state["journal_has_unsaved_changes"]
