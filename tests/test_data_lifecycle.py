from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path

import pandas as pd

from app.core.data import clean_weight_dataframe_with_report, prepare_analysis_data


def _main_module():
    path = Path(__file__).resolve().parents[1] / "Suivi_V1.py"
    src = path.read_text().split("st.set_page_config", 1)[0]
    ns = {"__name__": "suivi_main_test"}
    exec(compile(src, str(path), "exec"), ns)
    return ns


def sample_df():
    return pd.DataFrame({
        "Date": ["01/01/2026", "01/01/2026", "02/01/2026"],
        "Poids (Kgs)": [80.1, 80.4, 80.0],
        "Moment": ["08:00", "20:00", "08:00"],
        "Calories": [2000, 2100, 2050],
        "Sommeil": [7, 8, 7.5],
        "Notes": ["a", "b", "c"],
        "Colonne personnalisée": ["x", "y", "z"],
    })


def test_load_remote_csv_entrypoint_preserves_duplicate_day_and_extra_columns(monkeypatch):
    mod = _main_module()
    monkeypatch.setattr("pandas.read_csv", lambda *a, **k: sample_df().copy())
    mod["load_remote_csv_with_report"].clear()
    out, quality = mod["load_remote_csv_with_report"]("fake://csv")
    wrapped = mod["load_remote_csv"]("fake://csv")
    assert isinstance(wrapped, pd.DataFrame)
    assert isinstance((out, quality), tuple)
    assert len(out) == 3
    assert len(wrapped) == 3
    assert out["Date"].nunique() == 2
    assert {"Moment", "Colonne personnalisée"}.issubset(out.columns)
    assert {"Moment", "Colonne personnalisée"}.issubset(wrapped.columns)
    assert quality["duplicate_dates"] == 1
    assert hasattr(mod["load_remote_csv_with_report"], "clear")


def test_local_csv_cleaning_preserves_all_valid_rows_columns_and_duplicate_days():
    cleaned, report = clean_weight_dataframe_with_report(sample_df(), source="csv_local")
    assert len(cleaned) == 3
    assert cleaned["Date"].duplicated().sum() == 1
    assert set(sample_df().columns) == set(cleaned.columns)
    assert list(cleaned.columns[:2]) == ["Date", "Poids (Kgs)"]
    assert report.raw_rows == 3
    assert report.invalid_rows == 0
    assert report.duplicate_dates == 1


def test_invalid_rows_are_rejected_with_reasons_and_indices():
    raw = pd.DataFrame({
        "Date": ["bad", "01/01/2026", "02/01/2026", "03/01/2026", ""],
        "Poids (Kgs)": [80, "abc", 0, -1, ""],
        "Notes": ["bad date", "bad weight", "zero", "negative", "partial"],
    })
    cleaned, report = clean_weight_dataframe_with_report(raw)
    assert cleaned.empty
    assert report.invalid_rows == 5
    reasons = {r.index: r.reasons for r in report.rejected_rows}
    assert "date invalide ou vide" in reasons[0]
    assert "poids non numérique ou vide" in reasons[1]
    assert "poids nul ou négatif" in reasons[2]
    assert "poids nul ou négatif" in reasons[3]
    assert set(reasons[4]) == {"date invalide ou vide", "poids non numérique ou vide"}


def test_session_state_copies_survive_rerun_navigation_and_isolate_layers(monkeypatch):
    import app.core.session_state as ss
    state = {}
    monkeypatch.setattr(ss.st, "session_state", state)
    ss.ensure_session_defaults()
    df = sample_df()
    ss.set_source_data(df, "csv_local", {"raw_rows": 3})
    state["working_data"].loc[0, "Notes"] = "edit"
    ss.ensure_session_defaults()  # simulated rerun/page import
    assert state["working_data"].loc[0, "Notes"] == "edit"
    assert state["source_data"].loc[0, "Notes"] == "a"

    filtered = ss.get_filtered_or_working_data()
    filtered.loc[0, "Notes"] = "filtered edit"
    assert state["working_data"].loc[0, "Notes"] == "edit"

    new_df = df.copy(); new_df.loc[0, "Notes"] = "reload"
    ss.set_source_data(new_df, "google_sheets", {"raw_rows": 3})
    assert state["source_data"].loc[0, "Notes"] == "reload"
    assert state["working_data"].loc[0, "Notes"] == "reload"


def test_analysis_data_duplicate_strategy_is_copy_and_uses_moment_for_last():
    df = sample_df()
    analysis = prepare_analysis_data(df, "garder_la_derniere")
    assert len(analysis) == 2
    assert analysis.loc[analysis["Date"] == pd.Timestamp("2026-01-01"), "Poids (Kgs)"].iloc[0] == 80.4
    assert "Moment" in analysis.columns
    assert len(df) == 3


def test_analysis_data_accepts_text_moment_clock_moment_and_datetime_dates():
    text = pd.DataFrame({
        "Date": ["01/01/2026", "01/01/2026"],
        "Poids (Kgs)": [80.1, 80.4],
        "Moment": ["matin", "soir"],
    })
    clock = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-01"],
        "Poids (Kgs)": [80.1, 80.4],
        "Moment": ["08:00", "20:00"],
    })
    typed = pd.DataFrame({
        "Date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
        "Poids (Kgs)": [80.1, 80.4],
        "Moment": ["matin", "soir"],
    })
    for frame in [text, clock, typed]:
        original = frame.copy(deep=True)
        out = prepare_analysis_data(frame, "garder_la_derniere")
        assert out["Poids (Kgs)"].iloc[0] == 80.4
        pd.testing.assert_frame_equal(frame, original)


def test_filtered_view_distinguishes_inactive_nonempty_and_empty(monkeypatch):
    import app.core.session_state as ss
    state = {}
    monkeypatch.setattr(ss.st, "session_state", state)
    ss.ensure_session_defaults()
    cleaned, _ = clean_weight_dataframe_with_report(sample_df(), source="csv_local")
    ss.set_working_data(cleaned)
    assert len(ss.get_filtered_or_working_data()) == 3
    ss.set_filtered_data(cleaned.iloc[:1])
    assert len(ss.get_filtered_or_working_data()) == 1
    ss.set_filtered_data(cleaned.iloc[0:0])
    assert ss.get_filtered_or_working_data().empty


def test_columns_survive_cleaning_session_filter_and_export(monkeypatch):
    import app.core.session_state as ss
    state = {}
    monkeypatch.setattr(ss.st, "session_state", state)
    cleaned, quality = clean_weight_dataframe_with_report(sample_df(), source="csv_local")
    ss.set_source_data(cleaned, "csv_local", quality.to_dict())
    ss.set_filtered_data(ss.get_working_data().iloc[:2])
    exported = ss.get_working_data().to_csv(index=False)
    roundtrip = pd.read_csv(StringIO(exported))
    expected = set(sample_df().columns)
    assert expected.issubset(cleaned.columns)
    assert expected.issubset(state["working_data"].columns)
    assert expected.issubset(state["filtered_data"].columns)
    assert expected.issubset(roundtrip.columns)
