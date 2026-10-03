"""Onglet Boxe : vos séances de boxe lues à travers les données WHOOP.

L'onglet Whoop décrit la journée ; celui-ci ne regarde que la boxe. Il reprend
les données de la même synchronisation et y ajoute ce que l'application WHOOP
ne met jamais en regard : l'intensité rapportée à votre réserve cardiaque, le
coût de la séance pour le lendemain, l'effet des séances tardives sur la nuit,
la charge, la progression, et ce que la boxe pèse dans l'objectif de poids.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

from app.core.boxing_analytics import (
    BOXING_LABEL,
    LATE_SESSION_HOURS,
    MIN_PERCENT_RECORDED,
    MIN_SESSIONS_PROGRESSION,
    MIN_SPAN_PROGRESSION_DAYS,
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
)
from app.core.date_labels import (
    describe_freshness,
    format_clock_hour,
    format_duration_minutes,
    format_long_date,
    format_relative_day,
    format_short_date,
)
from app.core.formatting import MISSING_VALUE as MISSING_TEXT, format_fr_number
from app.core.session_state import DEFAULT_WHOOP_SYNC_DAYS, ensure_session_defaults, get_filtered_or_working_data
from app.core.target_trajectory import required_daily_loss
from app.core.whoop_analytics import KCAL_PER_KG, filter_period, plural
from app.core.whoop_session import resolve_credentials, run_whoop_sync, stored_token
from app.ui.boxing_visuals import (
    boxing_sessions_chart,
    habits_heatmap,
    late_sessions_chart,
    progression_chart,
    readiness_scatter,
    recovery_profile_chart,
    session_intensity_chart,
    weekly_frequency_chart,
)
from app.ui.components import empty_state, insight_card, kpi_card, page_hero, section_header
from app.ui.tables import format_table

PERIOD_CHOICES: dict[str, int | None] = {"30 jours": 30, "90 jours": 90, "6 mois": 182, "Tout": None}
DATED_ROWS = 14

SESSION_DECIMALS = {
    "Durée (min)": 0,
    "Strain séance": 1,
    "Intensité (% FCR)": 0,
    "Pic (% FC max)": 0,
    "FC moyenne (bpm)": 0,
    "FC max (bpm)": 0,
    "Zones 4–5 (min)": 0,
    "Part en zones 4–5 (%)": 0,
    "TRIMP": 0,
    "Calories séance (kcal)": 0,
    "kcal/min": 1,
    "Récupération du matin (%)": 0,
    "Récupération du lendemain (%)": 0,
    "HRV du lendemain (ms)": 0,
    "FC repos du lendemain (bpm)": 0,
    "Sommeil suivant (heures)": 1,
    "Efficacité sommeil suivant (%)": 0,
    "Délai avant coucher (h)": 1,
    "Marge avant coucher habituel (h)": 1,
    "Part enregistrée (%)": 0,
}

# Décimales et unités des mesures comparées dans les tableaux de tests.
MEASURE_FORMATS: dict[str, tuple[int, str]] = {
    "Récupération du lendemain (%)": (0, "pts"),
    "HRV du lendemain (ms)": (0, "ms"),
    "FC repos du lendemain (bpm)": (1, "bpm"),
    "Strain séance": (1, ""),
    "Intensité (% FCR)": (0, "pts"),
    "Zones 4–5 (min)": (0, "min"),
    "Efficacité sommeil suivant (%)": (1, "pts"),
}


def _n(count: Any, singular: str, plural_form: str | None = None) -> str:
    return f"{count} {plural(count, singular, plural_form)}"


def _zone_label(zone: Any) -> str:
    badge = {"Vert": "🟢", "Jaune": "🟡", "Rouge": "🔴"}.get(str(zone))
    return f"{badge} {zone}" if badge else MISSING_TEXT


def _render_chart(figure, key: str, *, fallback: str = "Pas assez de données sur la période choisie.") -> None:
    if figure is None:
        st.info(fallback)
        return
    st.plotly_chart(figure, use_container_width=True, key=key)


def _table_view(frame: pd.DataFrame, label: str, decimals: dict[str, int] | None = None) -> None:
    """Contrepartie tabulaire d'un graphique : toute valeur colorée reste lisible autrement."""
    if frame is None or frame.empty:
        return
    with st.expander(label, expanded=False):
        st.dataframe(format_table(frame, decimals, default_decimals=SESSION_DECIMALS), use_container_width=True, hide_index=True)


def _signed(value: Any, decimals: int) -> str:
    """Nombre signé, arrondi AVANT le signe : −0,3 à zéro décimale s'écrit « 0 », pas « −0 »."""
    try:
        numeric = round(float(value), decimals)
    except (TypeError, ValueError):
        return MISSING_TEXT
    return format_fr_number(numeric + 0.0, decimals=decimals, sign=True)


def _signed_minutes(hours: float) -> str:
    if not np.isfinite(hours):
        return MISSING_TEXT
    return f"{_signed(hours * 60.0, 0)} min"


def _tests_display(table: pd.DataFrame) -> pd.DataFrame:
    """Tableau de comparaisons : chaque mesure dans son unité, l'heure de coucher en horloge."""
    if table is None or table.empty:
        return pd.DataFrame()
    groups = [column for column in table.columns[1:3]]
    rows = []
    for _, row in table.iterrows():
        measure = str(row["Mesure"])
        record: dict[str, str] = {"Mesure": measure}
        if measure == "Coucher suivant":
            for group in groups:
                record[group] = format_clock_hour(row[group])
            record["Écart"] = _signed_minutes(float(row["Écart"]))
            record["IC 95 %"] = f"{_signed_minutes(float(row['IC 95 % bas']))} à {_signed_minutes(float(row['IC 95 % haut']))}"
        elif measure == "Sommeil suivant (heures)":
            for group in groups:
                record[group] = format_duration_minutes(float(row[group]) * 60.0)
            record["Écart"] = _signed_minutes(float(row["Écart"]))
            record["IC 95 %"] = f"{_signed_minutes(float(row['IC 95 % bas']))} à {_signed_minutes(float(row['IC 95 % haut']))}"
        else:
            decimals, unit = MEASURE_FORMATS.get(measure, (1, ""))
            suffix = f" {unit}" if unit else ""
            for group in groups:
                record[group] = format_fr_number(row[group], decimals=decimals)
            record["Écart"] = _signed(row["Écart"], decimals) + suffix
            record["IC 95 %"] = f"{_signed(row['IC 95 % bas'], decimals)} à {_signed(row['IC 95 % haut'], decimals)}{suffix}"
        record["Effectifs"] = str(row["Effectifs"])
        record["Écart établi"] = "oui" if bool(row["Écart établi"]) else "non"
        rows.append(record)
    return pd.DataFrame(rows)


def _tests_table(table: pd.DataFrame) -> None:
    st.dataframe(_tests_display(table), use_container_width=True, hide_index=True)
    st.caption(
        "« Écart établi » : l'écart résiste à un test de Welch dont le seuil de 5 % est divisé par le nombre "
        "de mesures comparées (correction de Bonferroni). Sans cette correction, comparer plusieurs mesures "
        "en ferait « différer » une par pur hasard. Effectifs : premier groupe / second groupe."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Connexion et synchronisation
# ──────────────────────────────────────────────────────────────────────────────


def _link_to_whoop(label: str) -> None:
    try:
        st.page_link("app/pages/Whoop.py", label=label, icon="⌚")
    except (StreamlitAPIException, KeyError):
        st.caption(f"{label} : ouvrez l'onglet **Whoop** dans le menu.")


def _not_connected() -> None:
    section_header("Connexion WHOOP requise", "Cet onglet lit les séances importées depuis votre bracelet.", "🔗")
    st.info(
        "Aucun compte WHOOP connecté. La connexion se fait une seule fois, depuis l'onglet Whoop : "
        "les séances de boxe importées y sont ensuite lues ici, sans nouvelle autorisation."
    )
    _link_to_whoop("Connecter WHOOP")


def _sync_controls(token, *, expanded: bool) -> None:
    last_sync = st.session_state.get("whoop_last_sync")
    label = f"Synchronisation WHOOP · dernière : {format_relative_day(last_sync) if last_sync is not None else 'jamais'}"
    with st.expander(label, expanded=expanded):
        cols = st.columns([1, 1])
        with cols[0]:
            days = st.number_input(
                "Profondeur d'historique (jours)",
                min_value=7,
                max_value=365,
                value=int(st.session_state.get("whoop_sync_days", DEFAULT_WHOOP_SYNC_DAYS)),
                step=7,
                key="boxing_sync_days",
                help="Les analyses de coût, de sommeil et de progression demandent plusieurs semaines de séances : 90 jours ou plus est conseillé.",
            )
        with cols[1]:
            st.write("")
            launch = st.button("Synchroniser maintenant", use_container_width=True, type="primary", key="boxing_sync")
        st.session_state["whoop_sync_days"] = int(days)
        # Les identifiants ne sont lus qu'au moment de synchroniser : les lire à
        # chaque affichage ferait apparaître l'alerte « No secrets found » de
        # Streamlit sur une installation locale sans fichier de secrets.
        if launch and run_whoop_sync(resolve_credentials(), token, int(days)):
            st.rerun()


def _freshness_banner(daily: pd.DataFrame) -> None:
    if daily is None or daily.empty or "Date" not in daily.columns:
        return
    last = pd.to_datetime(daily["Date"], errors="coerce").max()
    freshness = describe_freshness(last)
    if freshness["days"] is None:
        return
    message = f"Dernière mesure WHOOP : **{freshness['label']}** — {format_long_date(last)}."
    if freshness["stale"]:
        st.warning(message + " Les séances récentes manquent peut-être : une synchronisation remettra la page à jour.")
    else:
        st.caption(message)


# ──────────────────────────────────────────────────────────────────────────────
# Onglets
# ──────────────────────────────────────────────────────────────────────────────


def _overview_tab(context: dict[str, Any]) -> None:
    summary = context["summary"]
    guidance = context["guidance"]
    table = context["table"]

    section_header("Repère du jour", "Ce que votre récupération de ce matin autorise, lu selon les zones WHOOP.", "🧭")
    insight_card(guidance["title"], guidance["body"], tone=guidance["tone"], icon="🥊")
    st.caption(
        "Repère tiré des zones de récupération WHOOP (vert : prêt pour un effort soutenu, jaune : effort modéré, "
        "rouge : repos — [WHOOP 101](https://developer.whoop.com/docs/whoop-101/)). Ce n'est pas une prescription : "
        "un entraîneur, une douleur ou une blessure priment sur un score."
    )

    section_header("En chiffres", f"{context['period_label']} · {_n(summary['sessions'], 'séance')}.", "🗓️")
    cols = st.columns(5)
    with cols[0]:
        kpi_card("Séances", f"{summary['sessions']}", help_text=f"Soit {format_fr_number(summary['per_week'], decimals=1)} par semaine sur la période.")
    with cols[1]:
        kpi_card("Durée moyenne", format_duration_minutes(summary["mean_minutes"]), help_text=f"Total : {format_duration_minutes(summary['total_minutes'])}.")
    with cols[2]:
        kpi_card("Strain moyen", format_fr_number(summary["mean_strain"], decimals=1), help_text="Strain WHOOP de la séance, sur une échelle de 0 à 21.")
    with cols[3]:
        kpi_card(
            "Intensité moyenne",
            f"{format_fr_number(summary['mean_intensity'], decimals=0)} % FCR",
            help_text=(
                "FC moyenne de la séance rapportée à votre réserve cardiaque (FC max − FC de repos du jour) : "
                f"FC max de référence {format_fr_number(context['max_hr'], decimals=0)} bpm ({context['max_hr_source']})."
            ),
        )
    with cols[4]:
        kpi_card(
            "Dernière séance",
            format_relative_day(summary["last_date"]),
            help_text=f"{format_long_date(summary['last_date'])}. Jours de repos entre deux séances : {format_fr_number(summary['median_rest_days'], decimals=0)} en médiane.",
        )

    section_header("Ce que disent vos séances", "Constats classés par importance ; chacun reste muet tant que son effectif n'est pas atteint.", "🧠")
    insights = context["insights"]
    for insight in insights:
        insight_card(insight.title, insight.body, tone=insight.tone, icon=insight.icon)

    section_header("Vos séances dans le temps", "Hauteur : strain de la séance. Couleur : récupération du matin, avant la séance.", "📈")
    _render_chart(boxing_sessions_chart(table), "boxing-sessions")
    _table_view(
        table[["Date", "Strain séance", "Durée (min)", "Récupération du matin (%)", "Zone du matin", "Récupération du lendemain (%)"]].iloc[::-1],
        "Voir les valeurs",
    )
    if summary["partial_sessions"]:
        st.caption(
            f"⚠️ {_n(summary['partial_sessions'], 'séance')} captée(s) à moins de {format_fr_number(MIN_PERCENT_RECORDED, decimals=0)} % : "
            "strain, calories et zones y sont sous-estimés. Le capteur optique au poignet perd plus souvent le signal "
            "quand le bras frappe et que le gant serre."
        )


def _sessions_tab(context: dict[str, Any]) -> None:
    table = context["table"]
    section_header("Journal des séances", "Chaque séance avec ce qui l'a précédée et ce qu'elle a laissé, du plus récent au plus ancien.", "📔")
    display = table.iloc[::-1].reset_index(drop=True)
    view = pd.DataFrame(
        {
            "Date": display["Date"],
            "Début": [pd.Timestamp(value).strftime("%H:%M") if pd.notna(value) else MISSING_TEXT for value in display["Début"]],
            "Durée (min)": display["Durée (min)"],
            "Strain séance": display["Strain séance"],
            "Intensité (% FCR)": display["Intensité (% FCR)"],
            "FC moyenne (bpm)": display["FC moyenne (bpm)"],
            "FC max (bpm)": display["FC max (bpm)"],
            "Zones 4–5 (min)": display["Zones 4–5 (min)"],
            "TRIMP": display["TRIMP"],
            "Calories séance (kcal)": display["Calories séance (kcal)"],
            "Matin": display["Zone du matin"].map(_zone_label),
            "Récupération du matin (%)": display["Récupération du matin (%)"],
            "Lendemain": display["Zone du lendemain"].map(_zone_label),
            "Récupération du lendemain (%)": display["Récupération du lendemain (%)"],
            "Sommeil suivant (heures)": display["Sommeil suivant (heures)"],
        }
    )
    for column in ("Intensité (% FCR)", "Zones 4–5 (min)", "TRIMP", "Sommeil suivant (heures)"):
        if not view[column].notna().any():
            view = view.drop(columns=[column])
    st.dataframe(format_table(view.head(DATED_ROWS), default_decimals=SESSION_DECIMALS), use_container_width=True, hide_index=True)
    if len(view) > DATED_ROWS:
        with st.expander(f"Voir toutes les séances ({len(view)})", expanded=False):
            st.dataframe(format_table(view, default_decimals=SESSION_DECIMALS), use_container_width=True, hide_index=True)
    st.caption(
        "**Intensité (% FCR)** : (FC moyenne − FC de repos du matin) / (FC max − FC de repos), la méthode de "
        "réserve cardiaque de Karvonen dont WHOOP tire ses zones. **TRIMP** : minutes de chaque zone multipliées "
        "par le numéro de la zone (méthode des zones d'Edwards, référence objective du RPE de séance chez "
        "[Foster et al., J Strength Cond Res 2001](https://pubmed.ncbi.nlm.nih.gov/11708692/), et interchangeable "
        "avec le TRIMP de Banister en sport de combat : r = 0,89 chez des pratiquants de taekwondo, "
        "[Haddad et al., J Strength Cond Res 2012](https://pubmed.ncbi.nlm.nih.gov/21904234)) — WHOOP exprimant ses zones en réserve "
        "cardiaque plutôt qu'en FC max, le score se compare d'une séance à l'autre, pas d'une personne à l'autre. "
        "**Matin** : récupération calculée au réveil, avant la séance. **Lendemain** : premier score qui porte sa trace."
    )
    st.download_button(
        "Exporter les séances de boxe (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="boxe_seances.csv",
        mime="text/csv",
    )

    section_header("Intensité de chaque séance", "Minutes faciles, modérées et dures, séance par séance.", "🎚️")
    _render_chart(session_intensity_chart(table), "boxing-intensity", fallback="Zones de fréquence cardiaque indisponibles pour ces séances.")
    st.caption(
        "Facile : sous 70 % de la réserve cardiaque (zones 0–2) ; modéré : 70–80 % (zone 3) ; dur : 80 % et plus "
        "(zones 4–5). Un combat amateur sollicite une fréquence cardiaque très élevée sur toute sa durée "
        "([Chaabène et al., Sports Med 2015](https://doi.org/10.1007/s40279-014-0274-7)) : la part de temps dur "
        "dit à quel point une séance s'en approche."
    )

    records = boxing_records(table)
    if records:
        section_header("Meilleures marques de la période", None, "🏆")
        cols = st.columns(len(records))
        for column, record in zip(cols, records):
            with column:
                value = (
                    format_duration_minutes(record["value"])
                    if record["label"] == "Plus longue séance"
                    else f"{format_fr_number(record['value'], decimals=record['decimals'])}{' ' + record['unit'] if record['unit'] else ''}"
                )
                kpi_card(record["label"], value, help_text=format_long_date(record["date"]))

    habits = boxing_habits(table)
    section_header("Vos habitudes", "Jour de la semaine et moment de la journée de vos séances.", "🕰️")
    if not habits["ready"]:
        st.info("Habitudes lisibles à partir de trois séances horodatées.")
    else:
        _render_chart(habits_heatmap(habits["matrix"]), "boxing-habits")
        st.caption(f"Jour le plus fréquent : **{habits['favorite_day']}** ; moment le plus fréquent : **{habits['favorite_slot'].lower()}**.")
        with st.expander("Voir le tableau des habitudes", expanded=False):
            st.dataframe(habits["matrix"], use_container_width=True)


def _recovery_tab(context: dict[str, Any]) -> None:
    cost = context["cost"]
    section_header("Ce que la boxe coûte au lendemain", "Récupération, HRV et FC de repos le matin qui suit une journée de boxe, face au matin qui suit vos autres journées.", "🌡️")
    if not cost["ready"]:
        st.info(
            f"Comparaison disponible à partir de {cost['required']} lendemains de boxe notés et {cost['required']} autres "
            f"(actuellement {cost['boxing_days']} et {cost['other_days']})."
        )
    else:
        row = cost["table"][cost["table"]["Mesure"] == "Récupération du lendemain (%)"].iloc[0]
        cols = st.columns(3)
        with cols[0]:
            kpi_card("Lendemain de boxe", f"{format_fr_number(row['Après boxe'], decimals=0)} %", help_text=f"Sur {_n(cost['boxing_days'], 'lendemain')}.")
        with cols[1]:
            kpi_card("Lendemain des autres jours", f"{format_fr_number(row['Autres jours'], decimals=0)} %", help_text=f"Sur {_n(cost['other_days'], 'lendemain')}.")
        with cols[2]:
            # « non établi » passé en delta s'afficherait avec une flèche verte
            # montante : le statut est donc porté par le libellé.
            kpi_card(
                "Écart établi" if bool(row["Écart établi"]) else "Écart (non établi)",
                f"{_signed(row['Écart'], 0)} pts",
                help_text=f"Intervalle de confiance à 95 % : {_signed(row['IC 95 % bas'], 0)} à {_signed(row['IC 95 % haut'], 0)} points.",
            )
        _tests_table(cost["table"])
        morning_gap = cost["morning_boxing"] - cost["morning_other"]
        if np.isfinite(morning_gap) and abs(morning_gap) >= 5:
            st.caption(
                f"⚠️ Vos matins de boxe sont déjà {format_fr_number(abs(morning_gap), decimals=0)} points "
                f"{'au-dessus' if morning_gap > 0 else 'en dessous'} de vos autres matins : si vous boxez surtout les jours "
                "où vous êtes en forme, le lendemain de boxe part de plus haut, et la comparaison minimise le coût réel de la séance."
            )

    profile = context["profile"]
    section_header("En combien de jours vous revenez", "Récupération moyenne du matin de la séance à trois jours plus tard.", "↩️")
    if profile["table"].empty:
        st.info("Profil disponible dès qu'une séance est suivie de matins notés.")
    else:
        _render_chart(recovery_profile_chart(profile["table"], profile["overall"]), "boxing-profile")
        _table_view(profile["table"], "Voir le profil chiffré", {"Décalage": 0, "Récupération moyenne (%)": 0, "IC 95 % bas": 0, "IC 95 % haut": 0, "Observations": 0})
        st.caption(
            "Barres : intervalle de confiance à 95 % de la moyenne. Deux séances rapprochées se recouvrent : le J+2 "
            "d'une séance peut être le lendemain de la suivante."
        )

    readiness = context["readiness"]
    section_header("Boxez-vous plus fort les matins verts ?", "La récupération du matin précède la séance : la comparaison suit le sens du temps.", "🚦")
    _render_chart(readiness_scatter(context["table"]), "boxing-readiness", fallback="Aucune séance précédée d'un matin noté sur la période.")
    if not readiness["by_zone"].empty:
        by_zone = readiness["by_zone"].copy()
        by_zone["Zone du matin"] = by_zone["Zone du matin"].map(_zone_label)
        st.dataframe(
            format_table(by_zone, {"Séances": 0, "Strain moyen": 1, "Intensité (% FCR)": 0, "Zones 4–5 (min)": 0, "Récupération du lendemain (%)": 0}),
            use_container_width=True,
            hide_index=True,
        )
    if not readiness["ready"]:
        st.info(
            f"Test disponible à partir de {readiness['required']} séances après un matin vert et autant après un autre matin "
            f"(actuellement {readiness['green']} et {readiness['other']})."
        )
    else:
        _tests_table(readiness["table"])


def _sleep_tab(context: dict[str, Any]) -> None:
    late = context["late"]
    section_header(
        "Séances tardives et sommeil",
        f"Les nuits qui suivent une séance terminée moins de {format_fr_number(LATE_SESSION_HOURS, decimals=0)} h avant votre coucher habituel, face aux autres.",
        "🌙",
    )
    cols = st.columns(3)
    with cols[0]:
        kpi_card("Séances tardives", f"{late['late']}", help_text=f"Terminées moins de {format_fr_number(LATE_SESSION_HOURS, decimals=0)} h avant votre coucher habituel.")
    with cols[1]:
        kpi_card("Séances plus tôt", f"{late['early']}")
    with cols[2]:
        kpi_card("Marge médiane", f"{format_fr_number(late['median_gap'], decimals=1)} h", help_text="Entre la fin de séance et votre heure de coucher habituelle.")
    _render_chart(late_sessions_chart(context["table"]), "boxing-late", fallback="Heures de coucher ou de séance indisponibles sur la période.")
    if not late["ready"]:
        st.info(
            f"Comparaison disponible à partir de {late['required']} séances tardives et {late['required']} plus tôt suivies "
            f"d'une nuit mesurée (actuellement {late['late']} et {late['early']})."
        )
    else:
        _tests_table(late["table"])
    st.caption(
        "Pourquoi 4 heures : sur 14 689 porteurs de WHOOP suivis un an, un effort terminé moins de quatre heures avant "
        "le coucher est associé à un endormissement plus tardif, une nuit plus courte, une FC nocturne plus haute et une "
        "HRV plus basse, d'autant plus que l'effort est intense "
        "([Leota et al., Nature Communications 2025](https://doi.org/10.1038/s41467-025-58271-x)). Une méta-analyse "
        "d'essais contrôlés concluait au contraire que l'exercice du soir ne dégrade pas le sommeil en général "
        "([Stutz et al., Sports Med 2019](https://doi.org/10.1007/s40279-018-1015-0)). Ce test tranche pour vous : "
        "la séance est dite tardive par rapport à votre coucher **habituel**, pour qu'un coucher avancé ne suffise pas "
        "à la classer ainsi."
    )


def _load_tab(context: dict[str, Any]) -> None:
    load = context["load"]
    section_header("Charge boxe", "Les 7 derniers jours rapportés à votre semaine type des 21 jours précédents.", "🏋️")
    if load["status"] == "historique trop court":
        st.info(
            f"Il faut 28 jours d'historique WHOOP pour comparer la semaine en cours à une habitude : "
            f"synchronisez au moins depuis le {format_long_date(load['chronic_start'], with_weekday=False)}."
        )
    elif load["status"] in ("habitude trop mince", "indisponible"):
        st.info(
            f"Moins de 3 séances dans les 21 jours précédant la semaine en cours ({load['chronic_sessions']}) : "
            "pas encore d'habitude à laquelle comparer la semaine."
        )
    else:
        unit = load["unit"]
        cols = st.columns(3)
        with cols[0]:
            kpi_card(
                "7 derniers jours",
                f"{format_fr_number(load['acute'], decimals=0)} {unit}",
                help_text=f"Du {format_short_date(load['acute_start'])} au {format_short_date(load['acute_end'])} · {_n(load['acute_sessions'], 'séance')}.",
            )
        with cols[1]:
            kpi_card(
                "Semaine type",
                f"{format_fr_number(load['chronic_weekly'], decimals=0)} {unit}",
                help_text=f"Moyenne hebdomadaire du {format_short_date(load['chronic_start'])} au {format_short_date(load['chronic_end'])} · {_n(load['chronic_sessions'], 'séance')}.",
            )
        with cols[2]:
            kpi_card("Rapport", format_fr_number(load["ratio"], decimals=2), help_text=str(load["status"]))
        tone = {"montée en charge brutale": "warning", "montée en charge soutenue": "info", "charge en retrait": "info"}.get(load["status"], "success")
        insight_card(
            str(load["status"]).capitalize(),
            "Entre 0,8 et 1,3, la semaine reste dans vos habitudes ; au-delà de 1,5, elle les dépasse nettement. "
            "Avec deux ou trois séances par semaine, une seule séance de plus suffit à franchir ces seuils.",
            tone=tone,
            icon="🏋️",
        )
        # Les liens vivent dans la légende : la carte est du HTML brut, où la
        # syntaxe Markdown d'un lien s'afficherait telle quelle.
        st.caption(
            "Le rapport décrit une rupture d'habitude, il ne prédit pas la blessure "
            "([Impellizzeri et al., Int J Sports Physiol Perform 2020](https://doi.org/10.1123/ijspp.2019-0864)). "
            "La semaine en cours n'entre pas dans la référence, sans quoi le rapport serait borné par construction "
            "([Windt & Gabbett, Br J Sports Med 2019](https://bjsm.bmj.com/content/53/16/988))."
        )
        if unit == "minutes":
            st.caption("Zones de fréquence cardiaque absentes : la charge est comptée en minutes de boxe plutôt qu'en TRIMP.")

    weekly = context["weekly"]
    section_header("Régularité", "Séances par semaine sur la période, semaines vides comprises.", "📅")
    _render_chart(weekly_frequency_chart(weekly), "boxing-weekly")
    _table_view(weekly.iloc[::-1], "Voir les semaines", {"Séances": 0, "Durée totale (min)": 0, "TRIMP": 0, "Calories (kcal)": 0, "Strain moyen": 1})

    progression = context["progression"]
    section_header("Progression", "Pente de chaque mesure de séance par tranche de 30 jours, avec son intervalle de confiance.", "📊")
    if not progression["ready"]:
        st.info(
            f"Tendance calculée à partir de {MIN_SESSIONS_PROGRESSION} séances étalées sur au moins "
            f"{MIN_SPAN_PROGRESSION_DAYS} jours (actuellement {len(context['table'])} sur {progression['span_days']} jours)."
        )
        return
    st.dataframe(
        format_table(progression["table"], {"Séances": 0, "Pente / 30 jours": 2, "IC 95 % bas": 2, "IC 95 % haut": 2}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Une mesure n'est dite « en hausse » ou « en baisse » que si sa pente résiste à un test dont le seuil est divisé "
        "par le nombre de mesures examinées ; « stable » signifie que le hasard suffit à produire la pente observée. "
        "Une intensité qui baisse à séance égale peut traduire une meilleure condition — ou des séances plus techniques : "
        "le chiffre ne dit pas laquelle."
    )
    metrics = list(progression["table"]["Mesure"])
    metric = st.selectbox("Mesure à afficher", metrics, index=0, key="boxing-progression-metric")
    _render_chart(progression_chart(context["table"], metric), "boxing-progression")


def _weight_tab(context: dict[str, Any]) -> None:
    energy = context["energy"]
    section_header("Ce que la boxe pèse dans votre objectif", "Calories des séances, rapportées à la dépense mesurée et au déficit que suppose la trajectoire cible.", "🔥")
    if not energy["ready"]:
        st.info("Calories des séances indisponibles sur la période.")
    else:
        cols = st.columns(4)
        with cols[0]:
            kpi_card("Boxe, brut", f"{format_fr_number(energy['gross_weekly'], decimals=0)} kcal/sem", help_text="Calories WHOOP des séances, ramenées à une semaine.")
        with cols[1]:
            kpi_card(
                "Boxe, excédent net",
                f"{format_fr_number(energy['net_weekly'], decimals=0)} kcal/sem",
                help_text=(
                    f"Brut moins la dépense que vous auriez eue de toute façon pendant la séance : "
                    f"{format_fr_number(energy['baseline_per_minute'], decimals=2)} kcal/min, médiane de vos "
                    f"{energy['rest_days']} jours sans aucune séance."
                    if np.isfinite(energy["net_weekly"])
                    else "Il faut au moins trois jours sans séance avec une dépense mesurée pour estimer l'excédent net."
                ),
            )
        with cols[2]:
            kpi_card("Part de la dépense", f"{format_fr_number(energy['share_of_burn'], decimals=0)} %", help_text="Calories des séances sur la dépense totale des jours mesurés.")
        with cols[3]:
            kpi_card(
                "Part du déficit visé",
                f"{format_fr_number(energy['share_of_target'], decimals=0)} %",
                help_text=f"Déficit hebdomadaire que suppose la trajectoire cible : {format_fr_number(energy['required_weekly_deficit'], decimals=0)} kcal.",
            )
        st.caption(
            f"Équivalent indicatif : {format_fr_number(energy['kg_per_month'], decimals=1)} kg par mois sur la base de "
            f"{format_fr_number(KCAL_PER_KG, decimals=0)} kcal par kilogramme. Deux réserves : les bracelets du commerce "
            "mesurent correctement la fréquence cardiaque mais estiment mal la dépense énergétique "
            "([Shcherbina et al., J Pers Med 2017](https://doi.org/10.3390/jpm7020003), sept appareils testés, WHOOP n'en "
            "faisait pas partie) ; et l'équivalence de 7 700 kcal par kilogramme ne vaut pas pour l'eau et le glycogène "
            "qui dominent les variations de quelques jours."
        )

    weight = context["weight"]
    section_header("La balance du lendemain", "Variation de poids d'un matin au suivant, selon qu'une séance a eu lieu entre les deux.", "⚖️")
    if not weight["ready"]:
        st.info(
            f"Comparaison disponible à partir de {weight['required']} paires de pesées à un jour d'écart encadrant une séance "
            f"et autant sans séance (actuellement {weight['boxing_pairs']} et {weight['other_pairs']})."
        )
        return
    cols = st.columns(3)
    with cols[0]:
        kpi_card("Après une séance", f"{_signed(weight['after_boxing'], 2)} kg", help_text=f"Sur {_n(weight['boxing_pairs'], 'paire')} de pesées.")
    with cols[1]:
        kpi_card("Les autres jours", f"{_signed(weight['other'], 2)} kg", help_text=f"Sur {_n(weight['other_pairs'], 'paire')} de pesées.")
    with cols[2]:
        kpi_card(
            "Écart établi" if weight["significant"] else "Écart (non établi)",
            f"{_signed(weight['gap'], 2)} kg",
            help_text=f"Intervalle à 95 % : {_signed(weight['low'], 2)} à {_signed(weight['high'], 2)} kg. Test de Welch au seuil de 5 %.",
        )
    st.caption(
        "Une séance fait perdre de l'eau par la transpiration, que la réhydratation rend en un à deux jours : un creux "
        "le lendemain de la boxe n'est pas de la graisse perdue, et sa remontée n'est pas une reprise. La tendance de "
        "fond se lit sur le Dashboard. L'heure des pesées n'étant pas connue, une pesée prise après la séance brouille la comparaison."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ensure_session_defaults()
    page_hero(
        "Sport",
        "Boxe",
        "Vos séances de boxe lues à travers WHOOP : intensité réelle, coût pour le lendemain, sommeil, "
        "charge, progression et poids.",
        meta="Lecture seule — vos données de poids ne sont jamais modifiées",
    )

    token = stored_token()
    if token is None:
        _not_connected()
        return

    full_daily = st.session_state.get("whoop_daily", pd.DataFrame())
    full_workouts = st.session_state.get("whoop_workouts", pd.DataFrame())
    has_data = not (full_daily is None or full_daily.empty) or not (full_workouts is None or full_workouts.empty)
    _sync_controls(token, expanded=not has_data)
    if not has_data:
        empty_state("Lancez une synchronisation pour importer vos séances de boxe.")
        return

    sports = available_sports(full_workouts)
    if not sports:
        empty_state("Aucune séance WHOOP importée : enregistrez vos séances de boxe dans l'application WHOOP, puis synchronisez.")
        return

    default = default_sport_selection(sports)
    selection = st.multiselect(
        "Activités WHOOP comptées comme de la boxe",
        sports,
        default=default,
        help="Par défaut, les séances enregistrées sous « Boxing ». Ajoutez une activité si vous y avez rangé des séances de boxe.",
    )
    if BOXING_LABEL not in sports:
        st.caption(
            "Aucune séance étiquetée « Boxe » dans WHOOP : "
            + (
                f"{', '.join(default)} retenu par défaut, modifiable ci-dessus."
                if default
                else "choisissez ci-dessus l'activité sous laquelle vous enregistrez vos séances."
            )
        )
    if not selection:
        st.info("Choisissez au moins une activité pour lancer l'analyse.")
        return

    _freshness_banner(full_daily)
    period_label = st.radio("Période analysée", list(PERIOD_CHOICES), index=len(PERIOD_CHOICES) - 1, horizontal=True, help="S'applique à tous les onglets ci-dessous.")
    days = PERIOD_CHOICES[period_label]
    today = pd.Timestamp.now().normalize()
    daily = filter_period(full_daily, days, today=today)
    workouts = filter_period(full_workouts, days, today=today)
    sessions = boxing_sessions(workouts, selection)
    if sessions.empty:
        empty_state(f"Aucune séance de {', '.join(selection).lower()} sur la période choisie.")
        others = [sport for sport in available_sports(workouts) if sport not in selection]
        if others:
            st.caption("Activités présentes sur la période : " + ", ".join(others) + ".")
        return

    max_hr, max_hr_source = reference_max_hr(st.session_state.get("whoop_body"), full_workouts, full_daily)
    # La table s'appuie sur l'historique complet : le lendemain de la dernière
    # séance de la période, ou la FC de repos d'un jour charnière, restent lisibles.
    table = session_table(full_daily, sessions, max_hr=max_hr)

    first_day = pd.to_datetime(full_daily["Date"], errors="coerce").min() if not full_daily.empty else pd.to_datetime(sessions["Date"]).min()
    period_start = today - pd.Timedelta(days=days - 1) if days else first_day
    period_start = max(period_start, first_day) if pd.notna(first_day) else period_start
    window_days = max(1, int((today - period_start).days) + 1)
    last_daily = pd.to_datetime(full_daily["Date"], errors="coerce").max() if not full_daily.empty else today
    reference = min(today, last_daily) if pd.notna(last_daily) else today

    summary = boxing_summary(table, window_days=window_days, today=today)
    # La charge et le repère du jour décrivent le présent : ils portent sur les
    # quatre dernières semaines quelle que soit la période affichée.
    all_sessions = boxing_sessions(full_workouts, selection)
    load = boxing_load(session_table(full_daily, all_sessions, max_hr=max_hr), reference=reference, data_start=first_day)
    cost = next_day_cost(daily, table)
    profile = recovery_profile(daily, table)
    readiness = readiness_effect(table)
    late = late_session_sleep(table)
    weight = next_morning_weight(get_filtered_or_working_data(), table)
    progression = boxing_progression(table)
    energy = boxing_energy(daily, workouts, table, window_days=window_days, required_daily_kg=required_daily_loss())
    context = {
        "table": table,
        "summary": summary,
        "load": load,
        "cost": cost,
        "profile": profile,
        "readiness": readiness,
        "late": late,
        "weight": weight,
        "progression": progression,
        "energy": energy,
        "weekly": weekly_sessions(table, start=period_start, end=today),
        "guidance": todays_guidance(full_daily, all_sessions, load, today=today),
        "insights": boxing_insights(
            summary=summary, load=load, cost=cost, readiness=readiness, late=late, weight=weight, progression=progression, energy=energy
        ),
        "max_hr": max_hr,
        "max_hr_source": max_hr_source,
        "period_label": "Toute la période importée" if days is None else f"{days} derniers jours",
    }

    tabs = st.tabs(["Vue d'ensemble", "Séances", "Récupération", "Sommeil", "Charge & progression", "Poids & énergie"])
    with tabs[0]:
        _overview_tab(context)
    with tabs[1]:
        _sessions_tab(context)
    with tabs[2]:
        _recovery_tab(context)
    with tabs[3]:
        _sleep_tab(context)
    with tabs[4]:
        _load_tab(context)
    with tabs[5]:
        _weight_tab(context)


main()
