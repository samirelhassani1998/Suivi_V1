"""Moteur analytique avancé pour le suivi de poids.

Fonctions pures, sans dépendance Streamlit, pour :
- Vitesse de perte/gain
- Rolling averages multiples
- Volatilité du poids
- Score de discipline / régularité
- Score de cohérence
- Détection de rupture de tendance (CUSUM)
- Meilleures / pires semaines
- Segmentation par phases
- Comparaison périodique
- Scénarios prospectifs
- Projection objectif multi-scénario
- Streak analysis
- Day-of-week patterns
- Interprétation textuelle automatique
- Score de progression global
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Vitesse de perte / gain (kg/semaine)
# ---------------------------------------------------------------------------

def weight_velocity(df: pd.DataFrame, windows: tuple[int, ...] = (7, 14, 30, 90)) -> dict[int, float | None]:
    """Calcule la vitesse de variation du poids en kg/semaine sur plusieurs fenêtres.

    Retourne un dict {window_days: kg_per_week} pour chaque fenêtre.
    None si données insuffisantes pour la fenêtre.
    """
    if df.empty:
        return {w: None for w in windows}
    data = df.sort_values("Date")
    current = data["Poids (Kgs)"].iloc[-1]
    result: dict[int, float | None] = {}
    for w in windows:
        if len(data) < 2:
            result[w] = None
            continue
        target_date = data["Date"].iloc[-1] - pd.Timedelta(days=w)
        past = data[data["Date"] <= target_date]
        if past.empty:
            # Utiliser la première mesure si la fenêtre est plus large que l'historique
            past_weight = data["Poids (Kgs)"].iloc[0]
            actual_days = (data["Date"].iloc[-1] - data["Date"].iloc[0]).days
        else:
            past_weight = past["Poids (Kgs)"].iloc[-1]
            actual_days = (data["Date"].iloc[-1] - past["Date"].iloc[-1]).days
        if actual_days <= 0:
            result[w] = None
        else:
            result[w] = (current - past_weight) / actual_days * 7  # kg/semaine
    return result


# ---------------------------------------------------------------------------
# 2. Rolling averages multiples
# ---------------------------------------------------------------------------

def multi_rolling_averages(df: pd.DataFrame, windows: tuple[int, ...] = (7, 14, 30)) -> pd.DataFrame:
    """Ajoute plusieurs colonnes de moyenne mobile au DataFrame."""
    data = df.sort_values("Date").copy()
    for w in windows:
        data[f"MA_{w}j"] = data["Poids (Kgs)"].rolling(w, min_periods=1).mean()
    return data


# ---------------------------------------------------------------------------
# 3. Volatilité du poids
# ---------------------------------------------------------------------------

def weight_volatility(df: pd.DataFrame, window: int = 14) -> dict[str, float]:
    """Mesure la volatilité du poids sur une fenêtre (écart-type + coeff. variation)."""
    if len(df) < 3:
        return {"std": 0.0, "cv": 0.0, "range": 0.0, "interpretation": "données insuffisantes"}
    recent = df.sort_values("Date").tail(window)["Poids (Kgs)"]
    std = float(recent.std())
    mean = float(recent.mean())
    cv = std / mean * 100 if mean > 0 else 0.0
    rng = float(recent.max() - recent.min())
    if cv < 0.5:
        interp = "très stable"
    elif cv < 1.0:
        interp = "stable"
    elif cv < 2.0:
        interp = "modéré"
    else:
        interp = "volatil"
    return {"std": round(std, 3), "cv": round(cv, 2), "range": round(rng, 2), "interpretation": interp}


# ---------------------------------------------------------------------------
# 4. Score de discipline (régularité de saisie)
# ---------------------------------------------------------------------------

def discipline_score(df: pd.DataFrame, window_days: int = 30) -> dict[str, Any]:
    """Score de discipline basé sur la régularité de saisie.

    Retourne un score 0-100, le taux de mesures, et une interprétation.
    """
    if df.empty:
        return {"score": 0, "rate": 0.0, "interpretation": "aucune donnée", "measured_days": 0, "expected_days": window_days}
    data = df.sort_values("Date")
    last_date = data["Date"].max()
    start = last_date - pd.Timedelta(days=window_days)
    recent = data[data["Date"] >= start]
    measured = recent["Date"].nunique()
    rate = measured / window_days * 100
    score = min(100, int(rate))
    if score >= 85:
        interp = "excellente"
    elif score >= 60:
        interp = "bonne"
    elif score >= 40:
        interp = "moyenne"
    else:
        interp = "faible"
    return {
        "score": score,
        "rate": round(rate, 1),
        "interpretation": interp,
        "measured_days": int(measured),
        "expected_days": window_days,
    }


# ---------------------------------------------------------------------------
# 5. Score de cohérence (stabilité intra-semaine)
# ---------------------------------------------------------------------------

def consistency_score(df: pd.DataFrame, n_weeks: int = 4) -> dict[str, Any]:
    """Score de cohérence basé sur l'écart-type intra-semaine moyen.

    Plus le std est faible, plus les mesures sont cohérentes.
    """
    if len(df) < 7:
        return {"score": 0, "avg_weekly_std": 0.0, "interpretation": "données insuffisantes"}
    data = df.sort_values("Date").copy()
    data["week"] = data["Date"].dt.isocalendar().week.astype(int)
    data["year"] = data["Date"].dt.year
    recent_weeks = data.drop_duplicates(subset=["year", "week"]).tail(n_weeks)
    if recent_weeks.empty:
        return {"score": 0, "avg_weekly_std": 0.0, "interpretation": "données insuffisantes"}
    week_keys = list(zip(recent_weeks["year"], recent_weeks["week"]))
    stds = []
    for y, w in week_keys:
        week_data = data[(data["year"] == y) & (data["week"] == w)]["Poids (Kgs)"]
        if len(week_data) >= 2:
            stds.append(float(week_data.std()))
    if not stds:
        return {"score": 50, "avg_weekly_std": 0.0, "interpretation": "pas assez de mesures par semaine"}
    avg_std = float(np.mean(stds))
    # Score inversé : plus le std est faible, plus le score est élevé
    score = max(0, min(100, int(100 - avg_std * 50)))
    if score >= 80:
        interp = "très cohérent"
    elif score >= 60:
        interp = "cohérent"
    elif score >= 40:
        interp = "variable"
    else:
        interp = "incohérent"
    return {"score": score, "avg_weekly_std": round(avg_std, 3), "interpretation": interp}


# ---------------------------------------------------------------------------
# 6. Détection de ruptures de tendance (CUSUM simplifié)
# ---------------------------------------------------------------------------

def detect_trend_breaks(df: pd.DataFrame, threshold: float = 2.0) -> list[dict[str, Any]]:
    """Détecte les ruptures de tendance via un CUSUM simplifié.

    Retourne une liste de points de rupture avec date et type.
    """
    if len(df) < 14:
        return []
    data = df.sort_values("Date").copy()
    diff = data["Poids (Kgs)"].diff().dropna()
    mean_diff = float(diff.mean())
    std_diff = float(diff.std()) if float(diff.std()) > 0 else 0.01

    cusum_pos = np.zeros(len(diff))
    cusum_neg = np.zeros(len(diff))
    breaks: list[dict[str, Any]] = []

    for i in range(1, len(diff)):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + (diff.iloc[i] - mean_diff) / std_diff - 0.5)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - (diff.iloc[i] - mean_diff) / std_diff - 0.5)

        if cusum_pos[i] > threshold:
            breaks.append({
                "date": data["Date"].iloc[i + 1],
                "type": "reprise",
                "description": "Début de reprise de poids détecté",
            })
            cusum_pos[i] = 0
        elif cusum_neg[i] > threshold:
            breaks.append({
                "date": data["Date"].iloc[i + 1],
                "type": "accélération_perte",
                "description": "Accélération de la perte de poids détectée",
            })
            cusum_neg[i] = 0

    return breaks


# ---------------------------------------------------------------------------
# 7. Meilleures / pires semaines
# ---------------------------------------------------------------------------

def best_worst_weeks(df: pd.DataFrame, n: int = 5) -> dict[str, pd.DataFrame]:
    """Identifie les N meilleures et pires semaines (par variation de poids)."""
    if len(df) < 14:
        return {"best": pd.DataFrame(), "worst": pd.DataFrame()}
    data = df.sort_values("Date").copy()
    data["week_start"] = data["Date"].dt.to_period("W").apply(lambda x: x.start_time)
    weekly = data.groupby("week_start").agg(
        poids_moyen=("Poids (Kgs)", "mean"),
        poids_debut=("Poids (Kgs)", "first"),
        poids_fin=("Poids (Kgs)", "last"),
        nb_mesures=("Poids (Kgs)", "count"),
    ).reset_index()
    weekly["variation"] = weekly["poids_fin"] - weekly["poids_debut"]
    weekly = weekly[weekly["nb_mesures"] >= 2]
    if weekly.empty:
        return {"best": pd.DataFrame(), "worst": pd.DataFrame()}
    best = weekly.nsmallest(n, "variation")[["week_start", "variation", "poids_moyen", "nb_mesures"]].copy()
    worst = weekly.nlargest(n, "variation")[["week_start", "variation", "poids_moyen", "nb_mesures"]].copy()
    best.columns = ["Semaine", "Variation (kg)", "Poids moyen", "Mesures"]
    worst.columns = ["Semaine", "Variation (kg)", "Poids moyen", "Mesures"]
    return {"best": best.reset_index(drop=True), "worst": worst.reset_index(drop=True)}


# ---------------------------------------------------------------------------
# 8. Segmentation par phases
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    start: pd.Timestamp
    end: pd.Timestamp
    phase_type: str  # "perte", "plateau", "reprise"
    slope: float
    mean_weight: float
    duration_days: int


def segment_phases(df: pd.DataFrame, min_days: int = 7) -> list[Phase]:
    """Découpe l'historique en phases (perte, plateau, reprise) basées sur la pente locale."""
    if len(df) < min_days * 2:
        return []
    data = df.sort_values("Date").reset_index(drop=True)
    phases: list[Phase] = []
    window = max(min_days, 7)
    i = 0
    while i + window <= len(data):
        end_idx = min(i + window, len(data))
        chunk = data.iloc[i:end_idx]
        x = np.arange(len(chunk))
        slope = float(np.polyfit(x, chunk["Poids (Kgs)"], 1)[0])

        # Étendre la phase tant que la tendance reste la même
        while end_idx < len(data):
            next_chunk = data.iloc[i:end_idx + 1]
            next_slope = float(np.polyfit(np.arange(len(next_chunk)), next_chunk["Poids (Kgs)"], 1)[0])
            if _same_trend(slope, next_slope):
                end_idx += 1
                slope = next_slope
            else:
                break

        chunk = data.iloc[i:end_idx]
        if abs(slope) < 0.02:
            phase_type = "plateau"
        elif slope < 0:
            phase_type = "perte"
        else:
            phase_type = "reprise"

        phases.append(Phase(
            start=chunk["Date"].iloc[0],
            end=chunk["Date"].iloc[-1],
            phase_type=phase_type,
            slope=round(slope, 4),
            mean_weight=round(float(chunk["Poids (Kgs)"].mean()), 2),
            duration_days=int((chunk["Date"].iloc[-1] - chunk["Date"].iloc[0]).days) + 1,
        ))
        i = end_idx

    return _merge_consecutive_phases(phases)


def _same_trend(s1: float, s2: float) -> bool:
    if abs(s1) < 0.02 and abs(s2) < 0.02:
        return True  # both plateau
    if s1 < -0.02 and s2 < -0.02:
        return True  # both loss
    if s1 > 0.02 and s2 > 0.02:
        return True  # both gain
    return False


def _merge_consecutive_phases(phases: list[Phase]) -> list[Phase]:
    if not phases:
        return []
    merged = [phases[0]]
    for p in phases[1:]:
        if p.phase_type == merged[-1].phase_type:
            # Merge
            prev = merged[-1]
            merged[-1] = Phase(
                start=prev.start,
                end=p.end,
                phase_type=p.phase_type,
                slope=round((prev.slope + p.slope) / 2, 4),
                mean_weight=round((prev.mean_weight + p.mean_weight) / 2, 2),
                duration_days=int((p.end - prev.start).days) + 1,
            )
        else:
            merged.append(p)
    return merged


# ---------------------------------------------------------------------------
# 9. Comparaison périodique
# ---------------------------------------------------------------------------

def period_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Compare semaine courante vs précédente, mois courant vs précédent."""
    if len(df) < 7:
        return {"week": None, "month": None}
    data = df.sort_values("Date")
    last_date = data["Date"].max()

    # Semaine courante vs précédente
    week_end = last_date
    week_start = week_end - pd.Timedelta(days=6)
    prev_week_end = week_start - pd.Timedelta(days=1)
    prev_week_start = prev_week_end - pd.Timedelta(days=6)

    this_week = data[(data["Date"] >= week_start) & (data["Date"] <= week_end)]["Poids (Kgs)"]
    prev_week = data[(data["Date"] >= prev_week_start) & (data["Date"] <= prev_week_end)]["Poids (Kgs)"]

    week_cmp = None
    if not this_week.empty and not prev_week.empty:
        week_cmp = {
            "current_mean": round(float(this_week.mean()), 2),
            "previous_mean": round(float(prev_week.mean()), 2),
            "delta": round(float(this_week.mean() - prev_week.mean()), 2),
            "current_count": len(this_week),
            "previous_count": len(prev_week),
        }

    # Mois courant vs précédent
    month_start = last_date.replace(day=1)
    prev_month_end = month_start - pd.Timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)

    this_month = data[(data["Date"] >= month_start) & (data["Date"] <= last_date)]["Poids (Kgs)"]
    prev_month = data[(data["Date"] >= prev_month_start) & (data["Date"] <= prev_month_end)]["Poids (Kgs)"]

    month_cmp = None
    if not this_month.empty and not prev_month.empty:
        month_cmp = {
            "current_mean": round(float(this_month.mean()), 2),
            "previous_mean": round(float(prev_month.mean()), 2),
            "delta": round(float(this_month.mean() - prev_month.mean()), 2),
            "current_count": len(this_month),
            "previous_count": len(prev_month),
        }

    return {"week": week_cmp, "month": month_cmp}


# ---------------------------------------------------------------------------
# 10. Scénarios prospectifs
# ---------------------------------------------------------------------------

def prospective_scenarios(df: pd.DataFrame, target_weight: float) -> dict[str, dict[str, Any]]:
    """Génère 3 scénarios (optimiste/réaliste/pessimiste) basés sur les tendances récentes."""
    if len(df) < 14:
        return {}
    data = df.sort_values("Date")
    current = float(data["Poids (Kgs)"].iloc[-1])
    last_date = data["Date"].max()

    velocities = weight_velocity(df, windows=(7, 14, 30))
    v7 = velocities.get(7)
    v14 = velocities.get(14)
    v30 = velocities.get(30)

    valid = [v for v in [v7, v14, v30] if v is not None]
    if not valid:
        return {}

    # Scénarios basés sur les différentes vitesses
    scenarios = {}
    best_v = min(valid)   # plus grande perte (plus négatif)
    worst_v = max(valid)   # plus grand gain ou plus petite perte
    median_v = float(np.median(valid))

    for name, velocity in [("optimiste", best_v), ("réaliste", median_v), ("pessimiste", worst_v)]:
        if velocity is None:
            continue
        proj_30 = current + velocity * (30 / 7)
        proj_60 = current + velocity * (60 / 7)
        proj_90 = current + velocity * (90 / 7)

        eta_days = None
        eta_date = None
        if velocity < -0.01 and current > target_weight:
            remaining = current - target_weight
            eta_days = int(remaining / abs(velocity) * 7)
            eta_date = last_date + pd.Timedelta(days=eta_days)
        elif current <= target_weight:
            eta_days = 0
            eta_date = last_date

        scenarios[name] = {
            "velocity_kg_week": round(velocity, 3),
            "proj_30j": round(proj_30, 2),
            "proj_60j": round(proj_60, 2),
            "proj_90j": round(proj_90, 2),
            "eta_days": eta_days,
            "eta_date": eta_date,
        }

    return scenarios


# ---------------------------------------------------------------------------
# 11. Streak analysis (séries consécutives)
# ---------------------------------------------------------------------------

def streak_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Analyse les séries consécutives de perte / gain de poids."""
    if len(df) < 3:
        return {"current_streak": 0, "current_type": "neutre", "longest_loss": 0, "longest_gain": 0}
    data = df.sort_values("Date")
    diffs = data["Poids (Kgs)"].diff().dropna()

    # Streak courante
    current_streak = 0
    current_type = "neutre"
    for d in reversed(diffs.values):
        if d < -0.01 and (current_type in ("perte", "neutre")):
            current_streak += 1
            current_type = "perte"
        elif d > 0.01 and (current_type in ("gain", "neutre")):
            current_streak += 1
            current_type = "gain"
        else:
            if current_type == "neutre":
                continue
            break

    # Plus longues séries
    longest_loss = 0
    longest_gain = 0
    temp_loss = 0
    temp_gain = 0
    for d in diffs.values:
        if d < -0.01:
            temp_loss += 1
            temp_gain = 0
            longest_loss = max(longest_loss, temp_loss)
        elif d > 0.01:
            temp_gain += 1
            temp_loss = 0
            longest_gain = max(longest_gain, temp_gain)
        else:
            temp_loss = 0
            temp_gain = 0

    return {
        "current_streak": current_streak,
        "current_type": current_type,
        "longest_loss": longest_loss,
        "longest_gain": longest_gain,
    }


# ---------------------------------------------------------------------------
# 12. Day-of-week patterns
# ---------------------------------------------------------------------------

def day_of_week_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse les patterns par jour de la semaine."""
    if df.empty:
        return pd.DataFrame()
    data = df.sort_values("Date").copy()
    data["jour"] = data["Date"].dt.day_name()
    data["jour_num"] = data["Date"].dt.weekday
    stats = data.groupby(["jour_num", "jour"]).agg(
        poids_moyen=("Poids (Kgs)", "mean"),
        poids_std=("Poids (Kgs)", "std"),
        nb_mesures=("Poids (Kgs)", "count"),
    ).reset_index().sort_values("jour_num")
    jour_fr = {
        "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
        "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche",
    }
    stats["Jour"] = stats["jour"].map(jour_fr)
    return stats[["Jour", "poids_moyen", "poids_std", "nb_mesures"]].rename(columns={
        "poids_moyen": "Poids moyen",
        "poids_std": "Écart-type",
        "nb_mesures": "Mesures",
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 13. Score de progression global
# ---------------------------------------------------------------------------

def progression_score(df: pd.DataFrame, target_weight: float) -> dict[str, Any]:
    """Score composite 0-100 combinant progression, vitesse, discipline, cohérence."""
    if len(df) < 7:
        return {"score": 0, "grade": "N/A", "components": {}}

    data = df.sort_values("Date")
    initial = float(data["Poids (Kgs)"].iloc[0])
    current = float(data["Poids (Kgs)"].iloc[-1])

    # Composante progression (0-40 pts)
    total_to_lose = initial - target_weight
    if total_to_lose > 0:
        progress_pct = max(0, min(100, (initial - current) / total_to_lose * 100))
    else:
        progress_pct = 100.0
    progress_pts = progress_pct * 0.4

    # Composante vitesse (0-25 pts)
    vel = weight_velocity(df, windows=(14,))
    v14 = vel.get(14)
    if v14 is not None and v14 < 0:
        speed_pts = min(25, abs(v14) * 25)  # -1kg/sem = 25 pts
    else:
        speed_pts = 0

    # Composante discipline (0-20 pts)
    disc = discipline_score(df, window_days=30)
    disc_pts = disc["score"] * 0.2

    # Composante cohérence (0-15 pts)
    cons = consistency_score(df, n_weeks=4)
    cons_pts = cons["score"] * 0.15

    total = int(min(100, progress_pts + speed_pts + disc_pts + cons_pts))

    if total >= 85:
        grade = "A+"
    elif total >= 70:
        grade = "A"
    elif total >= 55:
        grade = "B"
    elif total >= 40:
        grade = "C"
    else:
        grade = "D"

    return {
        "score": total,
        "grade": grade,
        "components": {
            "progression": round(progress_pts, 1),
            "vitesse": round(speed_pts, 1),
            "discipline": round(disc_pts, 1),
            "cohérence": round(cons_pts, 1),
        },
    }


# ---------------------------------------------------------------------------
# 14. Rate of change acceleration
# ---------------------------------------------------------------------------

def weight_acceleration(df: pd.DataFrame) -> dict[str, Any]:
    """Détermine si la perte/gain de poids accélère ou ralentit."""
    vel = weight_velocity(df, windows=(7, 30))
    v7 = vel.get(7)
    v30 = vel.get(30)
    if v7 is None or v30 is None:
        return {"acceleration": 0, "interpretation": "données insuffisantes"}
    # Accélération = différence entre vitesse récente et vitesse globale
    acc = v7 - v30  # positif = la tendance est plus vers le gain récemment
    if acc < -0.2:
        interp = "La perte de poids s'accélère 🚀"
    elif acc > 0.2:
        interp = "La perte de poids ralentit ⚠️" if v30 < 0 else "Le gain de poids s'accélère ⚠️"
    else:
        interp = "Rythme stable"
    return {"acceleration": round(acc, 3), "interpretation": interp}


# ---------------------------------------------------------------------------
# 15. Interprétation textuelle intelligente
# ---------------------------------------------------------------------------

def generate_insights_text(df: pd.DataFrame, target_weight: float) -> list[str]:
    """Génère des insights textuels en français à partir des données."""
    if len(df) < 7:
        return ["📊 Continuez à enregistrer vos mesures pour obtenir des insights personnalisés."]

    insights: list[str] = []
    data = df.sort_values("Date")
    current = float(data["Poids (Kgs)"].iloc[-1])
    initial = float(data["Poids (Kgs)"].iloc[0])
    first_date = data["Date"].iloc[0].strftime("%d/%m/%Y")
    last_date = data["Date"].iloc[-1].strftime("%d/%m/%Y")

    # Progression globale
    total_lost = initial - current
    if total_lost > 0:
        insights.append(f"📉 Vous avez perdu **{total_lost:.1f} kg** depuis le {first_date} ({initial:.1f} → {current:.1f} kg).")
    elif total_lost < 0:
        insights.append(f"📈 Vous avez pris **{abs(total_lost):.1f} kg** depuis le {first_date} ({initial:.1f} → {current:.1f} kg).")

    # Vitesse
    vel = weight_velocity(df, windows=(7, 14))
    v7 = vel.get(7)
    v14 = vel.get(14)
    if v7 is not None:
        if v7 < -0.5:
            insights.append(f"🏃 Excellente vitesse de perte ces 7 derniers jours : **{v7:.2f} kg/sem**.")
        elif v7 < 0:
            insights.append(f"👍 Perte progressive ces 7 derniers jours : **{v7:.2f} kg/sem**.")
        elif v7 > 0.3:
            insights.append(f"⚠️ Reprise de poids ces 7 derniers jours : **+{v7:.2f} kg/sem**.")

    # Volatilité
    vol = weight_volatility(df, window=14)
    if vol["cv"] > 1.5:
        insights.append(f"📊 Poids assez volatil sur les 14 dernières mesures (amplitude de {vol['range']:.1f} kg).")
    elif vol["cv"] < 0.5:
        insights.append("✅ Poids très stable récemment.")

    # Discipline
    disc = discipline_score(df, window_days=30)
    if disc["score"] >= 85:
        insights.append(f"🏅 Discipline exemplaire : {disc['measured_days']}/{disc['expected_days']} jours mesurés ce mois.")
    elif disc["score"] < 50:
        insights.append(f"📝 Pensez à mesurer plus régulièrement ({disc['measured_days']}/{disc['expected_days']} jours ce mois).")

    # Streaks (mesures consécutives, pas jours calendaires)
    streaks = streak_analysis(df)
    if streaks["current_streak"] >= 3 and streaks["current_type"] == "perte":
        insights.append(f"🔥 Série en cours : **{streaks['current_streak']} mesures** consécutives en baisse !")
    if streaks["longest_loss"] >= 5:
        insights.append(f"🏆 Record de série de perte : {streaks['longest_loss']} mesures consécutives en baisse.")

    # Proximité objectif
    remaining = current - target_weight
    if remaining > 0:
        if remaining < 2:
            insights.append(f"🎯 Plus que **{remaining:.1f} kg** pour atteindre l'objectif ! Vous y êtes presque !")
        elif remaining < 5:
            insights.append(f"🎯 Il reste **{remaining:.1f} kg** pour atteindre l'objectif de {target_weight} kg.")
    elif remaining <= 0:
        insights.append("🎉 **Objectif atteint !** Félicitations !")

    return insights if insights else ["📊 Analyses en cours, continuez à saisir vos données."]
