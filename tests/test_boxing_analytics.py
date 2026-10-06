"""Tests des analyses boxe : rattachement aux dates, garde-fous et tests statistiques.

Chaque comparaison est vérifiée dans les deux sens : elle doit détecter un
effet réel, et se taire sur des séries où la boxe n'a aucun lien avec la mesure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.boxing_analytics import (
    BOXING_LABEL,
    LATE_SESSION_HOURS,
    MIN_GROUP_SIZE,
    available_sports,
    boxing_energy,
    boxing_habits,
    boxing_insights,
    boxing_load,
    boxing_progression,
    boxing_records,
    boxing_sessions,
    boxing_summary,
    default_sport_selection,
    late_session_sleep,
    next_day_cost,
    next_morning_weight,
    readiness_effect,
    recovery_profile,
    reference_max_hr,
    session_table,
    todays_guidance,
    weekly_sessions,
    zone_of,
)
from app.core.whoop import HttpResponse, WhoopToken
from app.core.whoop_sync import fetch_whoop_data, sync_window

TODAY = pd.Timestamp("2026-10-03")


def _daily(days: int = 70, *, seed: int = 0, end: pd.Timestamp = TODAY) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Récupération (%)": rng.uniform(30, 90, days),
            "HRV (ms)": rng.normal(45, 6, days),
            "FC repos (bpm)": rng.normal(56, 3, days),
            "Sommeil (heures)": rng.normal(7, 0.6, days),
            "Efficacité sommeil (%)": rng.normal(90, 3, days),
            "Heure de coucher": rng.normal(-0.5, 0.7, days),
            "Calories (kcal)": rng.normal(2800, 200, days),
            "Strain": rng.uniform(6, 16, days),
        }
    )


def _session(date, *, hour: float = 19.0, sport: str = "boxing", strain: float = 13.0, zones: tuple | None = (2, 4, 8, 12, 13, 5), **extra) -> dict:
    row = {
        "Date": pd.Timestamp(date).normalize(),
        "Début": pd.Timestamp(date).normalize() + pd.Timedelta(hours=hour),
        "Sport": sport,
        "Durée (min)": 60.0,
        "Strain séance": strain,
        "Calories séance (kcal)": 600.0,
        "FC moyenne (bpm)": 145.0,
        "FC max (bpm)": 182.0,
        "Part enregistrée (%)": 100.0,
    }
    if zones is not None:
        row.update({f"Zone {index} (min)": float(value) for index, value in enumerate(zones)})
    row.update(extra)
    return row


def _workouts(daily: pd.DataFrame, *, every: int = 3, seed: int = 0, late_share: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    rows = []
    for index, date in enumerate(daily["Date"]):
        if index % every:
            continue
        hour = 19.0 if rng.random() < late_share else 12.0
        rows.append(_session(date, hour=hour, strain=float(rng.uniform(10, 16))))
    return pd.DataFrame(rows)


# ── Sélection des séances ─────────────────────────────────────────────────────


def test_sports_are_listed_in_french_and_boxing_is_selected_by_default():
    workouts = pd.DataFrame([_session(TODAY), _session(TODAY - pd.Timedelta(days=2)), _session(TODAY, sport="running")])

    sports = available_sports(workouts)

    assert sports == ["Boxe", "Course à pied"]
    assert default_sport_selection(sports) == ["Boxe"]


def test_combat_sports_are_proposed_when_no_session_is_labelled_boxing():
    """Une séance rangée sous « Kickboxing » ne doit pas rendre l'onglet muet."""
    assert default_sport_selection(["Kickboxing", "Course à pied", "Arts martiaux"]) == ["Kickboxing", "Arts martiaux"]
    assert default_sport_selection(["Course à pied"]) == []


def test_boxing_sessions_keep_only_the_selected_sports():
    workouts = pd.DataFrame([_session(TODAY), _session(TODAY, sport="weightlifting"), _session(TODAY, sport="kickboxing")])

    assert boxing_sessions(workouts)["Sport"].tolist() == [BOXING_LABEL]
    assert sorted(boxing_sessions(workouts, ["Boxe", "Kickboxing"])["Sport"]) == ["Boxe", "Kickboxing"]
    assert boxing_sessions(None).empty


# ── FC maximale et table des séances ──────────────────────────────────────────


def test_max_hr_prefers_the_whoop_profile_then_the_highest_observed_value():
    workouts = pd.DataFrame([_session(TODAY, **{"FC max (bpm)": 188.0})])

    assert reference_max_hr({"max_heart_rate": 195}, workouts) == (195.0, "profil WHOOP")
    assert reference_max_hr({}, workouts) == (188.0, "plus haute FC observée")
    # Une valeur aberrante du profil est ignorée plutôt que de fausser toutes les intensités.
    assert reference_max_hr({"max_heart_rate": 40}, workouts)[1] == "plus haute FC observée"
    value, source = reference_max_hr({}, None)
    assert np.isnan(value) and source == "indisponible"


def test_session_intensity_uses_the_heart_rate_reserve_of_the_day():
    daily = pd.DataFrame({"Date": [TODAY], "FC repos (bpm)": [60.0], "Récupération (%)": [70.0]})
    sessions = boxing_sessions(pd.DataFrame([_session(TODAY, **{"FC moyenne (bpm)": 150.0, "FC max (bpm)": 171.0})]))

    table = session_table(daily, sessions, max_hr=180.0)

    # (150 − 60) / (180 − 60) = 75 % de la réserve cardiaque.
    assert table.loc[0, "Intensité (% FCR)"] == pytest.approx(75.0)
    assert table.loc[0, "Pic (% FC max)"] == pytest.approx(95.0)


def test_trimp_weights_each_zone_by_its_number_and_leaves_missing_zones_empty():
    sessions = boxing_sessions(pd.DataFrame([_session(TODAY, zones=(10, 10, 10, 10, 10, 10)), _session(TODAY - pd.Timedelta(days=1), zones=None)]))

    table = session_table(pd.DataFrame(), sessions)

    with_zones = table[table["Date"] == TODAY].iloc[0]
    without = table[table["Date"] != TODAY].iloc[0]
    assert with_zones["TRIMP"] == pytest.approx(10 * (0 + 1 + 2 + 3 + 4 + 5))
    assert with_zones["Zones 4–5 (min)"] == pytest.approx(20.0)
    assert with_zones["Part en zones 4–5 (%)"] == pytest.approx(100 * 20 / 60)
    # Zéro minute en zone haute est une information ; l'absence de zones n'en est pas une.
    assert np.isnan(without["TRIMP"]) and np.isnan(without["Zones 4–5 (min)"])


def test_session_reads_the_morning_score_before_and_the_next_morning_after():
    daily = pd.DataFrame(
        {
            "Date": [TODAY - pd.Timedelta(days=1), TODAY],
            "Récupération (%)": [80.0, 30.0],
            "HRV (ms)": [50.0, 38.0],
            "Sommeil (heures)": [7.5, 6.0],
            "Heure de coucher": [-1.0, 0.5],
        }
    )
    sessions = boxing_sessions(pd.DataFrame([_session(TODAY - pd.Timedelta(days=1), hour=20.0)]))

    row = session_table(daily, sessions).iloc[0]

    assert row["Récupération du matin (%)"] == 80.0 and row["Zone du matin"] == "Vert"
    assert row["Récupération du lendemain (%)"] == 30.0 and row["Zone du lendemain"] == "Rouge"
    assert row["HRV du lendemain (ms)"] == 38.0
    # La nuit qui suit la séance de la veille porte la date du réveil.
    assert row["Sommeil suivant (heures)"] == 6.0
    # Fin à 21 h, coucher à 00 h 30 : 3 h 30 d'écart.
    assert row["Délai avant coucher (h)"] == pytest.approx(3.5)


def test_session_table_is_chronological_unless_newest_first_is_asked():
    sessions = boxing_sessions(pd.DataFrame([_session(TODAY), _session(TODAY - pd.Timedelta(days=5))]))

    assert session_table(None, sessions)["Date"].is_monotonic_increasing
    assert session_table(None, sessions, newest_first=True)["Date"].is_monotonic_decreasing
    assert session_table(None, pd.DataFrame()).empty


# ── Volume, régularité, habitudes ─────────────────────────────────────────────


def test_summary_counts_sessions_per_week_and_days_since_the_last_one():
    dates = [TODAY - pd.Timedelta(days=offset) for offset in (2, 5, 9, 16)]
    table = session_table(None, boxing_sessions(pd.DataFrame([_session(date) for date in dates])))

    summary = boxing_summary(table, window_days=28, today=TODAY)

    assert summary["sessions"] == 4
    assert summary["per_week"] == pytest.approx(1.0)
    assert summary["days_since_last"] == 2
    # Repos entre séances : 6, 3 et 2 jours → médiane 3.
    assert summary["median_rest_days"] == pytest.approx(3.0)


def test_weekly_sessions_keep_empty_weeks_at_zero():
    """Une semaine sans boxe est une information : l'omettre ferait croire à une régularité."""
    table = session_table(None, boxing_sessions(pd.DataFrame([_session("2026-09-07"), _session("2026-09-09"), _session("2026-09-23")])))

    weekly = weekly_sessions(table, start="2026-09-07", end="2026-09-27")

    assert weekly["Séances"].tolist() == [2, 0, 1]
    assert weekly.loc[1, "Durée totale (min)"] == 0.0
    assert weekly["Semaine"].dt.dayofweek.eq(0).all()


def test_habits_place_each_session_in_its_weekday_and_time_slot():
    sessions = [_session("2026-09-07", hour=19.5), _session("2026-09-14", hour=19.0), _session("2026-09-16", hour=7.0)]
    habits = boxing_habits(session_table(None, boxing_sessions(pd.DataFrame(sessions))))

    assert habits["ready"]
    assert habits["matrix"].loc["Soir (18–21 h)", "lun."] == 2
    assert habits["matrix"].loc["Matin (avant 12 h)", "mer."] == 1
    assert habits["favorite_day"] == "lundi"


def test_records_carry_their_date():
    table = session_table(None, boxing_sessions(pd.DataFrame([_session("2026-09-07", strain=12.0), _session("2026-09-09", strain=17.5)])))

    strain = next(record for record in boxing_records(table) if record["label"] == "Strain record")

    assert strain["value"] == 17.5 and strain["date"] == pd.Timestamp("2026-09-09")


# ── Charge ────────────────────────────────────────────────────────────────────


def test_load_compares_the_last_week_to_the_typical_week_of_the_21_days_before():
    sessions = [_session(TODAY - pd.Timedelta(days=offset), zones=(0, 0, 0, 0, 25, 0)) for offset in (1, 3, 5, 9, 16, 23)]
    table = session_table(None, boxing_sessions(pd.DataFrame(sessions)))

    load = boxing_load(table, reference=TODAY, data_start=TODAY - pd.Timedelta(days=60))

    # Trois séances de 100 TRIMP cette semaine contre une par semaine avant : × 3.
    assert load["unit"] == "TRIMP"
    assert load["acute"] == pytest.approx(300.0)
    assert load["chronic_weekly"] == pytest.approx(100.0)
    assert load["ratio"] == pytest.approx(3.0)
    assert load["status"] == "montée en charge brutale"


def test_load_refuses_to_invent_zeros_before_the_history_starts():
    table = session_table(None, boxing_sessions(pd.DataFrame([_session(TODAY - pd.Timedelta(days=offset)) for offset in (1, 3, 5)])))

    load = boxing_load(table, reference=TODAY, data_start=TODAY - pd.Timedelta(days=10))

    assert load["status"] == "historique trop court"
    assert np.isnan(load["ratio"])


def test_load_falls_back_to_minutes_when_zones_are_missing():
    sessions = [_session(TODAY - pd.Timedelta(days=offset), zones=None) for offset in (1, 9, 16, 23)]
    load = boxing_load(session_table(None, boxing_sessions(pd.DataFrame(sessions))), reference=TODAY, data_start=TODAY - pd.Timedelta(days=60))

    assert load["unit"] == "minutes"
    assert load["ratio"] == pytest.approx(1.0)


# ── Comparaisons testées ──────────────────────────────────────────────────────


def _with_next_day_drop(daily: pd.DataFrame, workouts: pd.DataFrame, drop: float) -> pd.DataFrame:
    data = daily.copy()
    boxing_days = set(pd.to_datetime(workouts["Date"]).dt.normalize())
    for index, date in enumerate(data["Date"]):
        if pd.Timestamp(date) - pd.Timedelta(days=1) in boxing_days:
            data.loc[index, "Récupération (%)"] -= drop
    return data


def test_next_day_cost_detects_a_real_drop_after_boxing():
    daily = _daily(seed=4)
    workouts = _workouts(daily)
    table = session_table(daily, boxing_sessions(workouts))

    cost = next_day_cost(_with_next_day_drop(daily, workouts, 25.0), table)

    row = cost["table"][cost["table"]["Mesure"] == "Récupération du lendemain (%)"].iloc[0]
    assert cost["ready"]
    assert row["Écart"] < -15 and bool(row["Écart établi"])


@pytest.mark.parametrize("seed", range(10))
def test_next_day_cost_stays_silent_when_boxing_changes_nothing(seed):
    daily = _daily(seed=seed)
    table = session_table(daily, boxing_sessions(_workouts(daily, seed=seed)))

    cost = next_day_cost(daily, table)

    assert cost["ready"]
    # Seuil corrigé pour les trois mesures : sur dix séries sans lien, une
    # détection serait déjà suspecte ; on exige ici le silence sur la récupération.
    row = cost["table"][cost["table"]["Mesure"] == "Récupération du lendemain (%)"].iloc[0]
    assert not (bool(row["Écart établi"]) and abs(row["Écart"]) > 15)


def test_next_day_cost_waits_for_its_minimum_sample():
    daily = _daily(10)
    table = session_table(daily, boxing_sessions(pd.DataFrame([_session(daily["Date"].iloc[2])])))

    cost = next_day_cost(daily, table)

    assert not cost["ready"] and cost["boxing_days"] == 1 and cost["required"] == MIN_GROUP_SIZE


def test_two_sessions_the_same_day_count_as_one_next_morning():
    daily = _daily(20)
    day = daily["Date"].iloc[5]
    table = session_table(daily, boxing_sessions(pd.DataFrame([_session(day, hour=10), _session(day, hour=18)])))

    assert next_day_cost(daily, table)["boxing_days"] == 1


def test_recovery_profile_reads_the_session_morning_then_the_following_days():
    daily = _daily(seed=2)
    workouts = _workouts(daily)
    profile = recovery_profile(_with_next_day_drop(daily, workouts, 30.0), session_table(daily, boxing_sessions(workouts)))

    table = profile["table"].set_index("Décalage")
    assert list(table.index) == [0, 1, 2, 3]
    assert table.loc[0, "Jour"] == "Matin de la séance"
    assert table.loc[1, "Récupération moyenne (%)"] < table.loc[0, "Récupération moyenne (%)"] - 15


def test_readiness_effect_detects_harder_sessions_on_green_mornings():
    daily = _daily(seed=6)
    table = session_table(daily, boxing_sessions(_workouts(daily, every=2)))
    table["Strain séance"] = np.where(table["Zone du matin"] == "Vert", 17.0, 11.0) + np.random.default_rng(0).normal(0, 0.5, len(table))

    readiness = readiness_effect(table)

    row = readiness["table"][readiness["table"]["Mesure"] == "Strain séance"].iloc[0]
    assert readiness["ready"] and bool(row["Écart établi"]) and row["Écart"] > 4
    assert set(readiness["by_zone"]["Zone du matin"]) <= {"Vert", "Jaune", "Rouge"}


def test_late_sessions_are_judged_against_the_usual_bedtime_not_the_next_one():
    """Classer selon le coucher réel rendrait le test circulaire : un coucher
    avancé raccourcirait le délai et rangerait la séance parmi les tardives."""
    daily = _daily(seed=8)
    table = session_table(daily, boxing_sessions(_workouts(daily, seed=8)))

    habitual = float(np.median(daily["Heure de coucher"]))
    late = table[table["Marge avant coucher habituel (h)"] < LATE_SESSION_HOURS]
    for _, row in late.iterrows():
        expected = (row["Date"] + pd.Timedelta(days=1) + pd.Timedelta(hours=habitual) - row["Fin"]).total_seconds() / 3600
        assert row["Marge avant coucher habituel (h)"] == pytest.approx(expected)


def test_late_sessions_shortening_the_night_are_detected():
    daily = _daily(seed=9)
    table = session_table(daily, boxing_sessions(_workouts(daily, seed=9)))
    late = table["Marge avant coucher habituel (h)"] < LATE_SESSION_HOURS
    table.loc[late, "Sommeil suivant (heures)"] = 5.8 + np.random.default_rng(1).normal(0, 0.2, int(late.sum()))
    table.loc[~late, "Sommeil suivant (heures)"] = 7.4 + np.random.default_rng(2).normal(0, 0.2, int((~late).sum()))

    result = late_session_sleep(table)

    row = result["table"][result["table"]["Mesure"] == "Sommeil suivant (heures)"].iloc[0]
    assert result["ready"] and bool(row["Écart établi"]) and row["Écart"] < -1


def test_next_morning_weight_only_pairs_weigh_ins_one_day_apart():
    dates = pd.date_range(end=TODAY, periods=30, freq="D")
    weights = pd.DataFrame({"Date": dates, "Poids (Kgs)": 100.0})
    boxing_days = dates[::3]
    for date in boxing_days:
        index = dates.get_loc(date)
        if index + 1 < len(dates):
            weights.loc[index + 1 :, "Poids (Kgs)"] -= 0.4
            weights.loc[index + 2 :, "Poids (Kgs)"] += 0.4
    # Un trou de deux jours ne fait pas une paire.
    weights = weights.drop(index=[20])
    table = session_table(None, boxing_sessions(pd.DataFrame([_session(date) for date in boxing_days])))

    result = next_morning_weight(weights, table)

    assert result["ready"]
    assert result["after_boxing"] == pytest.approx(-0.4)
    assert result["gap"] < -0.5 and result["significant"]
    assert result["boxing_pairs"] + result["other_pairs"] < len(weights) - 1


def test_progression_is_uncertain_unless_the_slope_survives_the_corrected_test():
    dates = pd.date_range("2026-07-01", periods=20, freq="3D")
    rng = np.random.default_rng(5)
    table = pd.DataFrame(
        {
            "Date": dates,
            "Intensité (% FCR)": 55 + np.arange(20) * 0.8 + rng.normal(0, 1.0, 20),
            "Strain séance": rng.normal(13, 1.5, 20),
            "Durée (min)": rng.normal(60, 5, 20),
        }
    )

    progression = boxing_progression(table)

    readings = dict(zip(progression["table"]["Mesure"], progression["table"]["Lecture"]))
    assert readings["Intensité (% FCR)"] == "en hausse"
    assert readings["Strain séance"] == "tendance non déterminée"
    slope = progression["table"].set_index("Mesure").loc["Intensité (% FCR)", "Pente / 30 jours"]
    assert slope == pytest.approx(0.8 * 10, rel=0.2)


def test_progression_waits_for_enough_sessions_and_span():
    table = pd.DataFrame({"Date": pd.date_range("2026-09-01", periods=5, freq="2D"), "Strain séance": [10, 11, 12, 13, 14]})

    result = boxing_progression(table)

    assert not result["ready"] and result["table"].empty


def test_energy_removes_what_the_body_would_have_burnt_anyway():
    daily = pd.DataFrame({"Date": pd.date_range(end=TODAY, periods=14, freq="D"), "Calories (kcal)": 2880.0})
    workouts = pd.DataFrame([_session(TODAY - pd.Timedelta(days=offset)) for offset in (1, 8)])
    table = session_table(daily, boxing_sessions(workouts))

    energy = boxing_energy(daily, workouts, table, window_days=14, required_daily_kg=0.2)

    # 2 880 kcal / 1 440 min = 2 kcal/min ; une séance de 60 min à 600 kcal laisse 480 kcal nets.
    assert energy["baseline_per_minute"] == pytest.approx(2.0)
    assert energy["gross_weekly"] == pytest.approx(600.0)
    assert energy["net_weekly"] == pytest.approx(480.0)
    assert energy["required_weekly_deficit"] == pytest.approx(0.2 * 7700 * 7)
    assert energy["share_of_target"] == pytest.approx(480.0 / (0.2 * 7700 * 7) * 100)


# ── Repère du jour et constats ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("recovery", "expected_tone", "keyword"),
    [(20.0, "warning", "rouge"), (50.0, "info", "jaune"), (80.0, "success", "Feu vert")],
)
def test_today_guidance_follows_the_whoop_zone(recovery, expected_tone, keyword):
    daily = pd.DataFrame({"Date": [TODAY], "Récupération (%)": [recovery]})
    table = session_table(None, boxing_sessions(pd.DataFrame([_session(TODAY - pd.Timedelta(days=2))])))

    guidance = todays_guidance(daily, table, {}, today=TODAY)

    assert guidance["tone"] == expected_tone
    assert keyword.lower() in guidance["title"].lower()
    assert guidance["days_since"] == 2


def test_green_guidance_warns_when_the_week_already_jumped():
    daily = pd.DataFrame({"Date": [TODAY], "Récupération (%)": [85.0]})

    guidance = todays_guidance(daily, pd.DataFrame(), {"status": "montée en charge brutale", "ratio": 1.8}, today=TODAY)

    assert guidance["tone"] == "info" and "1,8" in guidance["body"]


def test_guidance_without_a_score_today_says_so():
    guidance = todays_guidance(pd.DataFrame({"Date": [TODAY - pd.Timedelta(days=1)], "Récupération (%)": [60.0]}), None, None, today=TODAY)

    assert guidance["zone"] is None and "pas encore disponible" in guidance["title"]


def test_zone_of_uses_the_whoop_thresholds():
    assert [zone_of(value) for value in (10, 33.9, 34, 66.9, 67, 100)] == ["Rouge", "Rouge", "Jaune", "Jaune", "Vert", "Vert"]
    assert zone_of(float("nan")) is None


def test_insights_stay_silent_on_untested_comparisons_and_rank_warnings_first():
    daily = _daily(seed=3)
    workouts = _workouts(daily)
    table = session_table(daily, boxing_sessions(workouts))
    summary = boxing_summary(table, window_days=70, today=TODAY)
    summary["days_since_last"] = 14
    cost = next_day_cost(_with_next_day_drop(daily, workouts, 25.0), table)

    insights = boxing_insights(
        summary=summary,
        load={"status": "charge maîtrisée"},
        cost=cost,
        readiness={"table": pd.DataFrame()},
        late={"ready": False, "table": pd.DataFrame()},
        weight={"ready": False},
        progression={"ready": False},
        energy={"ready": False},
    )

    titles = [insight.title for insight in insights]
    assert titles[0].startswith("Pas de boxe depuis 14 jours")
    assert any("points de moins après la boxe" in title for title in titles)
    priorities = [insight.priority for insight in insights]
    assert priorities == sorted(priorities, reverse=True)


def test_insights_are_empty_without_any_session():
    assert boxing_insights(summary={"sessions": 0}, load={}, cost={}, readiness={}, late={}, weight={}, progression={}, energy={}) == []


# ── Synchronisation partagée ──────────────────────────────────────────────────


class _Transport:
    """Répond à chaque route WHOOP par une réponse fixe, et note les URL appelées."""

    def __init__(self):
        self.urls: list[str] = []

    def __call__(self, method, url, *, params=None, data=None, headers=None, timeout=30):
        self.urls.append(url)
        if url.endswith("measurement/body"):
            return HttpResponse(200, {"height_meter": 1.8, "weight_kilogram": 100.0, "max_heart_rate": 191})
        if url.endswith("profile/basic"):
            return HttpResponse(200, {"first_name": "Test"})
        if url.endswith("activity/workout"):
            return HttpResponse(
                200,
                {
                    "records": [
                        {
                            "start": "2026-09-30T17:00:00.000Z",
                            "end": "2026-09-30T18:00:00.000Z",
                            "timezone_offset": "+02:00",
                            "sport_name": "boxing",
                            "score_state": "SCORED",
                            "score": {"strain": 14.2, "average_heart_rate": 150, "max_heart_rate": 186, "kilojoule": 2500, "percent_recorded": 100},
                        }
                    ],
                    "next_token": None,
                },
            )
        return HttpResponse(200, {"records": [], "next_token": None})


def test_sync_fetches_the_body_measurement_for_the_max_heart_rate():
    transport = _Transport()

    result = fetch_whoop_data(WhoopToken(access_token="at"), start="2026-09-01", end="2026-10-01", transport=transport)

    assert result.body["max_heart_rate"] == 191
    assert result.counts["workout"] == 1
    assert result.workouts.loc[0, "Sport"] == "boxing"
    assert any(url.endswith("measurement/body") for url in transport.urls)
    assert "séances importés" in result.summary()


def test_sync_window_ends_today_in_utc():
    start, end = sync_window(30, now=pd.Timestamp("2026-10-03 23:30", tz="Europe/Paris"))

    assert end == pd.Timestamp("2026-10-03")
    assert (end - start).days == 30


@pytest.mark.parametrize("missing_offset", [1, 16])
def test_load_uses_complete_minutes_on_both_windows_when_a_trimp_is_missing(missing_offset):
    sessions = [_session(TODAY - pd.Timedelta(days=offset), zones=None if offset == missing_offset else (0, 0, 0, 0, 25, 0)) for offset in (1, 9, 16, 23)]
    table = session_table(None, boxing_sessions(pd.DataFrame(sessions)))

    result = boxing_load(table, reference=TODAY, data_start=TODAY - pd.Timedelta(days=60))

    assert result["unit"] == "minutes"
    assert result["acute"] == 60
    assert result["chronic_weekly"] == 60
    assert result["ratio"] == 1
    assert result["trimp_sessions"] == 3 and result["total_sessions"] == 4


def test_load_stays_unavailable_without_a_common_complete_measure():
    table = pd.DataFrame({"Date": [TODAY - pd.Timedelta(days=offset) for offset in (1, 9, 16, 23)], "TRIMP": [np.nan, 100, 100, 100], "Durée (min)": [60, np.nan, 60, 60]})

    result = boxing_load(table, reference=TODAY, data_start=TODAY - pd.Timedelta(days=60))

    assert result["status"] == "données incomplètes"
    assert np.isnan(result["ratio"]) and np.isnan(result["acute"])
    assert result["trimp_sessions"] == result["duration_sessions"] == 3


def test_load_chooses_its_unit_only_from_the_comparison_windows():
    sessions = [_session(TODAY - pd.Timedelta(days=offset), zones=None if offset == 50 else (0, 0, 0, 0, 25, 0)) for offset in (1, 9, 16, 23, 50)]
    table = session_table(None, boxing_sessions(pd.DataFrame(sessions)))

    result = boxing_load(table, reference=TODAY, data_start=TODAY - pd.Timedelta(days=60))

    assert result["unit"] == "TRIMP" and result["ratio"] == 1
    assert result["total_sessions"] == 4


def test_a_partially_missing_zone_does_not_become_a_complete_trimp():
    row = _session(TODAY)
    row["Zone 4 (min)"] = np.nan
    table = session_table(None, boxing_sessions(pd.DataFrame([row])))
    assert np.isnan(table.loc[0, "TRIMP"])


def test_weight_comparison_limits_exposure_dates_and_keeps_their_next_morning():
    dates = pd.date_range(end=TODAY, periods=60)
    weights = pd.DataFrame({"Date": dates, "Poids (Kgs)": 100 + np.random.default_rng(4).normal(0, 0.3, 60)})
    sessions = pd.DataFrame({"Date": dates[::3]})
    result = next_morning_weight(weights, sessions, exposure_start=dates[30], exposure_end=dates[58])
    trimmed = next_morning_weight(weights.iloc[30:], sessions, exposure_start=dates[30], exposure_end=dates[58])

    assert (result["boxing_pairs"], result["other_pairs"]) == (10, 19)
    assert result["gap"] == pytest.approx(trimmed["gap"])
    assert result["p_value"] == pytest.approx(trimmed["p_value"])
    # The outcome on date 59 is retained for the final exposure on date 58.
    without_last = next_morning_weight(weights.iloc[:-1], sessions, exposure_start=dates[30], exposure_end=dates[58])
    assert without_last["other_pairs"] == 18
    # Later outcomes are not new exposure dates in this comparison.
    extra = pd.DataFrame({"Date": [TODAY + pd.Timedelta(days=1)], "Poids (Kgs)": [120]})
    extended = next_morning_weight(pd.concat([weights, extra]), sessions, exposure_start=dates[30], exposure_end=dates[58])
    assert extended["gap"] == pytest.approx(result["gap"])


def _sleep_comparison_table():
    return pd.DataFrame({
        "Date": pd.date_range(end=TODAY, periods=8),
        "Marge avant coucher habituel (h)": [2.0] * 4 + [6.0] * 4,
        "Coucher suivant": [0.0] * 8,
        "Sommeil suivant (heures)": [5.0] * 4 + [8.0] * 4,
        "Efficacité sommeil suivant (%)": [90.0] * 8,
        "FC repos du lendemain (bpm)": [60.0] * 8,
        "HRV du lendemain (ms)": [50.0] * 8,
    })


def test_sleep_comparison_counts_one_night_per_day_and_uses_the_latest_session():
    table = _sleep_comparison_table()
    additional_early = table.iloc[:4].copy()
    additional_early["Marge avant coucher habituel (h)"] = 10.0
    duplicated = pd.concat([additional_early, table, table], ignore_index=True)

    result = late_session_sleep(duplicated)

    assert result["late"] == result["early"] == 4
    assert len(result["nights"]) == result["nights"]["Date"].nunique() == 8
    assert result["table"]["Effectifs"].eq("4 / 4").all()
    assert result["table"].set_index("Mesure").loc["Sommeil suivant (heures)", "Écart"] == -3


def test_many_sessions_on_two_days_do_not_unlock_a_sleep_comparison():
    table = _sleep_comparison_table()
    table["Date"] = [TODAY - pd.Timedelta(days=1)] * 4 + [TODAY] * 4
    result = late_session_sleep(table)
    assert not result["ready"]
    assert result["late"] == result["early"] == 1


def test_sleep_does_not_classify_a_day_with_an_unknown_session_end():
    table = _sleep_comparison_table()
    unknown = table.iloc[[0]].copy()
    unknown["Marge avant coucher habituel (h)"] = np.nan
    result = late_session_sleep(pd.concat([table, unknown], ignore_index=True))
    assert result["late"] == 3 and result["early"] == 4
    assert not result["ready"]


def test_unavailable_sleep_inference_never_becomes_a_reassuring_insight():
    result = late_session_sleep(_sleep_comparison_table())
    assert result["ready"]  # Descriptive means remain useful.
    assert result["table"]["Inférence"].eq("indisponible").all()
    assert not result["table"]["Écart établi"].any()
    insights = boxing_insights(summary={"sessions": 8}, load={}, cost={}, readiness={}, late=result, weight={}, progression={}, energy={})
    assert not any(insight.icon == "🌙" for insight in insights)


def test_non_significant_sleep_result_does_not_claim_an_absence_of_effect():
    table = _sleep_comparison_table()
    table["Sommeil suivant (heures)"] = [6, 7, 8, 9] * 2
    result = late_session_sleep(table)
    insights = boxing_insights(summary={"sessions": 8}, load={}, cost={}, readiness={}, late=result, weight={}, progression={}, energy={})
    sleep = next(insight for insight in insights if insight.icon == "🌙")
    assert sleep.tone == "info"
    assert "incertain" in sleep.title and "ne démontre pas l'absence" in sleep.body


def test_energy_never_substitutes_gross_calories_for_a_missing_net_estimate():
    table = session_table(None, boxing_sessions(pd.DataFrame([_session(TODAY)])))
    result = boxing_energy(None, None, table, window_days=7, required_daily_kg=0.02)
    assert result["gross_weekly"] == 600
    assert np.isnan(result["net_weekly"])
    assert np.isnan(result["share_of_target"]) and np.isnan(result["kg_per_month"])


def test_weekly_unknown_measurements_stay_missing_in_weeks_with_sessions():
    table = session_table(None, boxing_sessions(pd.DataFrame([_session("2026-09-07", **{"Calories séance (kcal)": np.nan, "Durée (min)": np.nan})])))
    result = weekly_sessions(table, start="2026-09-07", end="2026-09-20")
    assert np.isnan(result.loc[0, "Calories (kcal)"])
    assert np.isnan(result.loc[0, "Durée totale (min)"])
    assert result.loc[1, "Calories (kcal)"] == result.loc[1, "Durée totale (min)"] == 0


def test_missing_session_duration_does_not_become_zero_net_energy():
    daily = pd.DataFrame({"Date": pd.date_range(end=TODAY, periods=14), "Calories (kcal)": 2880.0})
    workouts = pd.DataFrame([_session(TODAY, **{"Durée (min)": np.nan})])
    table = session_table(daily, boxing_sessions(workouts))
    result = boxing_energy(daily, workouts, table, window_days=14, required_daily_kg=0.02)
    assert result["rest_days"] >= 3
    assert np.isnan(result["net_weekly"]) and np.isnan(result["share_of_target"])
