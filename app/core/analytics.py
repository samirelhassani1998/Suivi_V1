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
    """Ajoute plusieurs colonnes de moyenne mobile au DataFrame.

    Note : les moyennes glissent sur N mesures consécutives, pas N jours calendaires.
    """
    data = df.sort_values("Date").copy()
    for w in windows:
        data[f"MA_{w}m"] = data["Poids (Kgs)"].rolling(w, min_periods=1).mean()
    return data


# ---------------------------------------------------------------------------
# 3. Volatilité du poids
# ---------------------------------------------------------------------------

def weight_volatility(df: pd.DataFrame, window: int = 14) -> dict[str, float]:
    """Mesure la volatilité du poids sur les `window` derniers jours calendaires."""
    if len(df) < 3:
        return {"std": 0.0, "cv": 0.0, "range": 0.0, "interpretation": "données insuffisantes", "nb_mesures": 0}
    data = df.sort_values("Date")
    cutoff = data["Date"].max() - pd.Timedelta(days=window)
    recent = data[data["Date"] >= cutoff]["Poids (Kgs)"]
    if len(recent) < 2:
        return {"std": 0.0, "cv": 0.0, "range": 0.0, "interpretation": "données insuffisantes", "nb_mesures": len(recent)}
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
    return {"std": round(std, 3), "cv": round(cv, 2), "range": round(rng, 2), "interpretation": interp, "nb_mesures": len(recent)}


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


def segment_phases(df: pd.DataFrame, min_days: int = 7, gap_threshold_days: int = 21) -> list[Phase]:
    """Découpe l'historique en phases (perte, plateau, reprise) basées sur la pente locale.

    Gap-aware: les données sont d'abord découpées aux trous > gap_threshold_days
    puis chaque bloc continu est segmenté indépendamment.
    """
    if len(df) < min_days * 2:
        return []
    data = df.sort_values("Date").reset_index(drop=True)

    # Découper aux gaps > gap_threshold_days
    blocks: list[pd.DataFrame] = []
    block_start = 0
    for i in range(1, len(data)):
        gap = (data["Date"].iloc[i] - data["Date"].iloc[i - 1]).days
        if gap > gap_threshold_days:
            if i - block_start >= min_days:
                blocks.append(data.iloc[block_start:i].reset_index(drop=True))
            block_start = i
    # Dernier bloc
    if len(data) - block_start >= min_days:
        blocks.append(data.iloc[block_start:].reset_index(drop=True))
    elif not blocks and len(data) >= 3:
        # Si aucun bloc assez grand, prendre tout
        blocks.append(data)

    all_phases: list[Phase] = []
    for block in blocks:
        all_phases.extend(_segment_block(block, min_days))

    return _merge_consecutive_phases(all_phases)


def _segment_block(data: pd.DataFrame, min_days: int = 7) -> list[Phase]:
    """Segmente un bloc continu (sans gaps) en phases."""
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

    return phases


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
    """Score composite 0-100 combinant progression, vitesse, discipline, cohérence.

    Fallback pour < 7 mesures: score simplifié basé sur direction + discipline.
    """
    if len(df) < 3:
        return {"score": 0, "grade": "N/A", "components": {}}

    data = df.sort_values("Date")
    initial = float(data["Poids (Kgs)"].iloc[0])
    current = float(data["Poids (Kgs)"].iloc[-1])

    if len(df) < 7:
        # Fallback simplifié : direction (0-30) + discipline (0-30), plafonné à 60
        direction_delta = initial - current  # positif = perte
        if direction_delta > 0:
            direction_pts = min(30, direction_delta * 12)  # -2.5 kg = 30 pts
        else:
            direction_pts = 0

        days_span = max((data["Date"].max() - data["Date"].min()).days, 1)
        disc_pts = min(30, (len(df) / max(days_span, 1)) * 100 * 0.3)

        total = int(min(60, direction_pts + disc_pts))  # plafonné à 60
        grade = _score_to_grade(total)
        return {
            "score": total,
            "grade": grade,
            "confidence": "fragile",
            "components": {
                "direction": round(direction_pts, 1),
                "discipline": round(disc_pts, 1),
            },
        }

    # Score complet (>= 7 mesures)
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
    grade = _score_to_grade(total)

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


def _score_to_grade(score: int) -> str:
    if score >= 85:
        return "A+"
    if score >= 70:
        return "A"
    if score >= 55:
        return "B"
    if score >= 40:
        return "C"
    return "D"


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
# 15. Détection de la période d'effort actuelle
# ---------------------------------------------------------------------------

def detect_current_effort(df: pd.DataFrame, gap_threshold_days: int = 21) -> dict[str, Any]:
    """Détecte la période d'effort actuelle en cherchant le dernier 'trou' significatif.

    Un trou > gap_threshold_days est interprété comme un arrêt du suivi.
    Retourne un dict avec le sous-DataFrame de la période d'effort, la date de début,
    le nombre de jours et de mesures.
    """
    if df.empty:
        return {"effort_df": df, "start_date": None, "days": 0, "measurements": 0, "is_subset": False}

    data = df.sort_values("Date").copy()
    if len(data) < 2:
        return {
            "effort_df": data,
            "start_date": data["Date"].iloc[0],
            "days": 1,
            "measurements": 1,
            "is_subset": False,
        }

    # Parcourir en ordre inverse pour trouver le dernier trou
    dates = data["Date"].values
    gap_idx = None
    for i in range(len(dates) - 1, 0, -1):
        gap = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
        if gap > gap_threshold_days:
            gap_idx = i
            break

    if gap_idx is not None:
        effort_df = data.iloc[gap_idx:].copy().reset_index(drop=True)
        is_subset = True
    else:
        effort_df = data.copy()
        is_subset = False

    start_date = effort_df["Date"].iloc[0]
    end_date = effort_df["Date"].iloc[-1]
    duration_days = (end_date - start_date).days + 1

    return {
        "effort_df": effort_df,
        "start_date": start_date,
        "days": duration_days,
        "measurements": len(effort_df),
        "is_subset": is_subset,
    }


# ---------------------------------------------------------------------------
# 16. Tendance robuste EMA
# ---------------------------------------------------------------------------

def compute_trend_ema(df: pd.DataFrame, span: int = 7) -> pd.DataFrame:
    """Calcule une tendance EMA (Exponential Moving Average) robuste.

    Fonctionne correctement même avec des données éparses car EWM
    s'adapte naturellement aux intervalles irréguliers.
    """
    data = df.sort_values("Date").copy()
    data["Tendance_EMA"] = data["Poids (Kgs)"].ewm(span=max(span, 2), adjust=False).mean()
    return data


# ---------------------------------------------------------------------------
# 17. Comparaison rythme actuel vs rythme nécessaire
# ---------------------------------------------------------------------------

def pace_comparison(df: pd.DataFrame, target_weight: float, target_date: pd.Timestamp | None = None) -> dict[str, Any]:
    """Compare le rythme actuel de perte avec le rythme nécessaire pour atteindre l'objectif.

    Si target_date n'est pas fourni, on utilise 6 mois par défaut.
    """
    if len(df) < 3:
        return {"current_pace": None, "required_pace": None, "interpretation": "données insuffisantes"}

    data = df.sort_values("Date")
    current = float(data["Poids (Kgs)"].iloc[-1])
    last_date = data["Date"].max()

    if current <= target_weight:
        return {
            "current_pace": 0.0,
            "required_pace": 0.0,
            "interpretation": "Objectif déjà atteint !",
            "ratio": 1.0,
        }

    # Rythme actuel : vitesse sur 14j (ou toute la période d'effort si < 14j)
    vel = weight_velocity(df, windows=(14,))
    current_pace = vel.get(14)  # kg/semaine (négatif = perte)

    # Rythme nécessaire
    remaining = current - target_weight
    if target_date is None:
        target_date = last_date + pd.Timedelta(days=180)  # 6 mois par défaut
    days_left = max((target_date - last_date).days, 1)
    weeks_left = days_left / 7
    required_pace = -(remaining / weeks_left)  # négatif = perte nécessaire

    # Interprétation
    if current_pace is None:
        interp = "Pas assez de données pour calculer le rythme actuel."
        ratio = None
    elif current_pace >= 0:
        interp = f"Vous êtes actuellement en reprise (+{current_pace:.2f} kg/sem). Objectif : perdre {abs(required_pace):.2f} kg/sem."
        ratio = 0.0
    else:
        ratio = abs(current_pace) / abs(required_pace) if abs(required_pace) > 0.001 else 999.0
        if ratio >= 2.0:
            interp = f"Excellent ! Vous perdez **{abs(current_pace):.2f} kg/sem**, soit {ratio:.1f}x le rythme nécessaire ({abs(required_pace):.2f} kg/sem)."
        elif ratio >= 1.0:
            interp = f"Bon rythme ! Vous perdez **{abs(current_pace):.2f} kg/sem**, aligné avec le rythme nécessaire ({abs(required_pace):.2f} kg/sem)."
        elif ratio >= 0.5:
            interp = f"Rythme un peu lent : **{abs(current_pace):.2f} kg/sem** vs {abs(required_pace):.2f} kg/sem nécessaire."
        else:
            interp = f"Rythme insuffisant : **{abs(current_pace):.2f} kg/sem** vs {abs(required_pace):.2f} kg/sem nécessaire."

    return {
        "current_pace": current_pace,
        "required_pace": required_pace,
        "remaining_kg": round(remaining, 1),
        "days_left": days_left,
        "interpretation": interp,
        "ratio": ratio,
    }


# ---------------------------------------------------------------------------
# 18. Interprétation textuelle intelligente (contextuelle)
# ---------------------------------------------------------------------------

def generate_insights_text(df: pd.DataFrame, target_weight: float) -> list[str]:
    """Génère des insights textuels en français, priorisés sur la période d'effort actuelle.

    L'ordre est conçu pour être motivant :
    1. D'abord ce qui se passe MAINTENANT (effort en cours)
    2. Ensuite les signaux positifs
    3. En dernier l'historique global (contexte)
    """
    if len(df) < 3:
        return ["📊 Continuez à enregistrer vos mesures pour obtenir des insights personnalisés."]

    insights: list[str] = []
    data = df.sort_values("Date")
    current = float(data["Poids (Kgs)"].iloc[-1])
    initial_global = float(data["Poids (Kgs)"].iloc[0])

    # Détection période d'effort
    effort = detect_current_effort(df, gap_threshold_days=21)
    effort_df = effort["effort_df"]
    effort_start = effort["start_date"]
    effort_measurements = effort["measurements"]
    effort_days = effort["days"]

    if effort["is_subset"] and len(effort_df) >= 2:
        effort_initial = float(effort_df["Poids (Kgs)"].iloc[0])
        effort_delta = effort_initial - current
        start_str = effort_start.strftime("%d/%m/%Y")

        # -- PRIORITÉ 1 : Effort en cours --
        if effort_delta > 0:
            insights.append(
                f"💪 Effort en cours : **-{effort_delta:.1f} kg** en {effort_days} jours "
                f"depuis le {start_str} ({effort_initial:.1f} → {current:.1f} kg)."
            )
        elif effort_delta < 0:
            insights.append(
                f"📊 Depuis le {start_str} : **+{abs(effort_delta):.1f} kg** en {effort_days} jours "
                f"({effort_initial:.1f} → {current:.1f} kg)."
            )
        else:
            insights.append(f"➡️ Poids stable depuis le {start_str} ({effort_days} jours).")

        insights.append(f"📅 Période de suivi actuelle : **{effort_measurements} mesures** sur {effort_days} jours.")
    else:
        effort_df = data  # pas de trou détecté, on utilise tout

    # -- PRIORITÉ 2 : Signaux positifs récents --

    # Streaks
    streaks = streak_analysis(effort_df)
    if streaks["current_streak"] >= 2 and streaks["current_type"] == "perte":
        insights.append(f"🔥 Série en cours : **{streaks['current_streak']} mesures** consécutives en baisse !")
    if streaks["longest_loss"] >= 5:
        insights.append(f"🏆 Record de série de perte : {streaks['longest_loss']} mesures consécutives en baisse.")

    # Vitesse sur l'effort
    vel = weight_velocity(effort_df, windows=(7, 14))
    v7 = vel.get(7)
    if v7 is not None:
        if v7 < -1.0:
            insights.append(f"🚀 Perte rapide ces 7 derniers jours : **{v7:.2f} kg/sem**.")
        elif v7 < -0.3:
            insights.append(f"🏃 Bonne vitesse de perte : **{v7:.2f} kg/sem** sur 7 jours.")
        elif v7 < 0:
            insights.append(f"👍 Perte progressive : **{v7:.2f} kg/sem** sur 7 jours.")
        elif v7 > 0.5:
            insights.append(f"⚠️ Reprise notable ces 7 derniers jours : **+{v7:.2f} kg/sem**.")

    # Volatilité
    vol = weight_volatility(effort_df, window=14)
    if vol["nb_mesures"] >= 3:
        if vol["cv"] > 1.5:
            insights.append(f"📊 Poids fluctuant (amplitude {vol['range']:.1f} kg sur 14j). C'est normal, la tendance compte plus.")
        elif vol["cv"] < 0.5:
            insights.append("✅ Poids très stable récemment.")

    # -- PRIORITÉ 3 : Objectif --
    remaining = current - target_weight
    if remaining > 0:
        if remaining < 2:
            insights.append(f"🎯 Plus que **{remaining:.1f} kg** pour atteindre l'objectif ! Vous y êtes presque !")
        else:
            insights.append(f"🎯 Il reste **{remaining:.1f} kg** pour atteindre l'objectif de {target_weight} kg.")
    elif remaining <= 0:
        insights.append("🎉 **Objectif atteint !** Félicitations !")

    # -- PRIORITÉ 4 : Discipline (encourageante) --
    disc = discipline_score(effort_df, window_days=min(effort_days, 30) if effort_days > 0 else 30)
    if disc["score"] >= 85:
        insights.append(f"🏅 Discipline exemplaire : {disc['measured_days']}/{disc['expected_days']} jours mesurés.")
    elif disc["score"] >= 50:
        insights.append(f"👍 Bonne régularité : {disc['measured_days']} mesures sur {disc['expected_days']} jours.")
    elif effort_measurements >= 3:
        insights.append(f"📝 Essayez de mesurer plus souvent pour des insights plus fiables ({disc['measured_days']}/{disc['expected_days']} jours).")

    # -- CONTEXTE : Historique global (en dernier, si différent de l'effort) --
    if effort["is_subset"]:
        total_delta = initial_global - current
        first_date = data["Date"].iloc[0].strftime("%d/%m/%Y")
        if total_delta > 0:
            insights.append(f"📜 Historique global : **-{total_delta:.1f} kg** depuis le {first_date}.")
        elif total_delta < 0:
            insights.append(f"📜 Historique global depuis le {first_date} : +{abs(total_delta):.1f} kg. L'important est la tendance actuelle.")

    return insights if insights else ["📊 Analyses en cours, continuez à saisir vos données."]


# ---------------------------------------------------------------------------
# 18b. Résumé actionnable : Situation / Interprétation / Action
# ---------------------------------------------------------------------------

def generate_action_summary(df: pd.DataFrame, target_weight: float) -> dict[str, str]:
    """Génère un résumé structuré en 3 niveaux pour l'utilisateur.

    Retourne un dict avec 'situation', 'interpretation', 'action'.
    Contexte-aware : adapté à la phase de l'effort (démarrage, actif, établi).
    """
    if len(df) < 2:
        return {
            "situation": "Première mesure enregistrée. Bienvenue !",
            "interpretation": "Les analyses se débloquent à partir de 3 mesures.",
            "action": "Pesez-vous demain matin à jeun pour commencer le suivi.",
        }

    data = df.sort_values("Date")
    current = float(data["Poids (Kgs)"].iloc[-1])
    effort = detect_current_effort(df, gap_threshold_days=21)
    effort_df = effort["effort_df"]
    effort_days = effort["days"]
    effort_measurements = effort["measurements"]
    is_startup = effort_days < 7

    if effort["is_subset"] and len(effort_df) >= 2:
        effort_initial = float(effort_df["Poids (Kgs)"].iloc[0])
        delta = effort_initial - current  # positif = perte
        start_str = effort["start_date"].strftime("%d/%m/%Y")
    else:
        delta = float(data["Poids (Kgs)"].iloc[0]) - current
        start_str = data["Date"].iloc[0].strftime("%d/%m/%Y")

    # ── SITUATION (factuelle) ──
    if effort["is_subset"]:
        if delta > 0:
            situation = f"Vous avez repris le suivi le {start_str} — **-{delta:.1f} kg** en {effort_days} jours ({effort_measurements} mesures)."
        elif delta < 0:
            situation = f"Vous avez repris le suivi le {start_str} — **+{abs(delta):.1f} kg** en {effort_days} jours ({effort_measurements} mesures)."
        else:
            situation = f"Vous avez repris le suivi le {start_str} — poids stable ({effort_measurements} mesures)."
    else:
        situation = f"Suivi continu depuis le {start_str} — {len(df)} mesures."

    # ── INTERPRÉTATION (ce que ça veut dire) ──
    if is_startup:
        if delta > 1.5:
            interpretation = (
                f"Cette perte rapide ({delta:.1f} kg en {effort_days}j) est très probablement de l'eau "
                f"et de la vidange digestive. **Ce n'est pas de la graisse.** "
                f"La vraie tendance se vérifiera à partir du jour 7-10."
            )
        elif delta > 0:
            interpretation = (
                f"Bonne direction pour le début. "
                f"C'est encore trop tôt pour extrapoler une tendance fiable (jour {effort_days}/7)."
            )
        elif delta < 0:
            interpretation = (
                f"Reprise de {abs(delta):.1f} kg en début de suivi. "
                f"Les premiers jours sont souvent bruités — attendez le jour 7 pour évaluer."
            )
        else:
            interpretation = "Poids stable en début de suivi. C'est normal, attendez quelques jours."
    else:
        # Effort établi (>= 7 jours)
        vel = weight_velocity(effort_df, windows=(7,))
        v7 = vel.get(7)
        if v7 is not None and v7 < -1.0:
            interpretation = f"Perte soutenue de **{v7:.2f} kg/sem** — rythme rapide mais tenable si bien encadré."
        elif v7 is not None and v7 < -0.3:
            interpretation = f"Perte régulière de **{v7:.2f} kg/sem** — c'est le rythme idéal pour une perte durable."
        elif v7 is not None and v7 < 0:
            interpretation = f"Perte lente ({v7:.2f} kg/sem) mais dans la bonne direction. La patience paie."
        elif v7 is not None and v7 > 0.5:
            interpretation = f"Reprise récente de **+{v7:.2f} kg/sem**. Vérifiez si c'est du bruit ou une tendance."
        elif v7 is not None and v7 > 0:
            interpretation = "Légère remontée récente — probablement des fluctuations naturelles."
        else:
            interpretation = "Tendance neutre. Continuez le suivi pour des insights plus précis."

    # ── ACTION (quoi faire maintenant) ──
    remaining = current - target_weight

    if is_startup:
        action = (
            f"**Cette semaine** : pesez-vous chaque matin à jeun. "
            f"L'objectif n'est pas la perte mais la régularité du suivi. "
            f"Les analyses fiables arrivent au jour 7."
        )
    elif remaining > 20:
        action = (
            f"**Prochain objectif** : passer sous {int(current / 5) * 5} kg. "
            f"Concentrez-vous sur la constance du suivi quotidien."
        )
    elif remaining > 5:
        ms_target = int(current) if int(current) < current else int(current) - 1
        action = (
            f"**Prochain palier** : {ms_target} kg (reste {current - ms_target:.1f} kg). "
            f"Maintenez le rythme actuel."
        )
    elif remaining > 0:
        action = f"**Vous approchez de l'objectif** ({remaining:.1f} kg restants). Gardez le cap !"
    else:
        action = "**Objectif atteint !** Passez en mode maintien — l'enjeu est de rester stable."

    return {
        "situation": situation,
        "interpretation": interpretation,
        "action": action,
    }


# ---------------------------------------------------------------------------
# 19. Analyse historique des efforts (détection yo-yo)
# ---------------------------------------------------------------------------

def analyze_effort_history(df: pd.DataFrame, gap_threshold_days: int = 21) -> dict[str, Any]:
    """Analyse l'historique des périodes d'effort pour détecter le pattern yo-yo.

    Retourne un résumé des efforts passés, le rebond moyen après chaque pause,
    et un insight textuel sur le pattern global.
    """
    if len(df) < 5:
        return {"efforts": [], "pattern": "données insuffisantes", "insight": None}

    data = df.sort_values("Date").reset_index(drop=True)

    # Découper en périodes d'effort (blocs séparés par gaps > threshold)
    effort_starts = [0]
    for i in range(1, len(data)):
        gap = (data.iloc[i]["Date"] - data.iloc[i - 1]["Date"]).days
        if gap > gap_threshold_days:
            effort_starts.append(i)

    efforts: list[dict[str, Any]] = []
    for i, start_idx in enumerate(effort_starts):
        end_idx = (effort_starts[i + 1] - 1) if (i + 1 < len(effort_starts)) else (len(data) - 1)
        chunk = data.iloc[start_idx:end_idx + 1]
        if len(chunk) < 2:
            continue

        start_w = float(chunk["Poids (Kgs)"].iloc[0])
        end_w = float(chunk["Poids (Kgs)"].iloc[-1])
        min_w = float(chunk["Poids (Kgs)"].min())
        days = (chunk["Date"].iloc[-1] - chunk["Date"].iloc[0]).days + 1

        efforts.append({
            "start_date": chunk["Date"].iloc[0],
            "end_date": chunk["Date"].iloc[-1],
            "start_weight": start_w,
            "end_weight": end_w,
            "min_weight": min_w,
            "delta": round(start_w - end_w, 1),  # positif = perte
            "days": days,
            "measurements": len(chunk),
        })

    if len(efforts) < 2:
        return {"efforts": efforts, "pattern": "historique insuffisant", "insight": None}

    # Calculer les rebonds : différence entre fin d'un effort et début du suivant
    rebounds: list[float] = []
    for i in range(len(efforts) - 1):
        end_w_prev = efforts[i]["end_weight"]
        start_w_next = efforts[i + 1]["start_weight"]
        rebound = start_w_next - end_w_prev  # positif = reprise
        rebounds.append(round(rebound, 1))

    avg_rebound = float(np.mean(rebounds)) if rebounds else 0.0
    positive_rebounds = [r for r in rebounds if r > 0]
    pct_rebounds = len(positive_rebounds) / len(rebounds) * 100 if rebounds else 0

    # Déterminer le pattern
    if pct_rebounds >= 70 and avg_rebound > 2:
        pattern = "yo-yo confirmé"
        insight = (
            f"⚠️ **Pattern yo-yo détecté** : après chaque pause de suivi, "
            f"votre poids remonte en moyenne de **+{avg_rebound:.1f} kg** "
            f"({len(positive_rebounds)}/{len(rebounds)} reprises). "
            f"Le maintien du suivi régulier est votre plus grande arme contre ce cycle."
        )
    elif pct_rebounds >= 50:
        pattern = "tendance au rebond"
        insight = (
            f"📊 Vos données montrent un rebond moyen de **+{avg_rebound:.1f} kg** "
            f"après les pauses de suivi ({len(positive_rebounds)}/{len(rebounds)} cas). "
            f"Maintenir la régularité est essentiel."
        )
    elif avg_rebound < 0:
        pattern = "progression maintenue"
        insight = (
            f"✅ Bonne nouvelle : vous maintenez globalement vos acquis entre les périodes "
            f"de suivi (variation moyenne : {avg_rebound:+.1f} kg)."
        )
    else:
        pattern = "neutre"
        insight = None

    # Meilleur effort historique
    best_effort = max(efforts, key=lambda e: e["delta"]) if efforts else None

    return {
        "efforts": efforts,
        "rebounds": rebounds,
        "avg_rebound": round(avg_rebound, 1),
        "pct_rebounds": round(pct_rebounds, 0),
        "pattern": pattern,
        "insight": insight,
        "best_effort": best_effort,
        "total_efforts": len(efforts),
    }


# ---------------------------------------------------------------------------
# 20. Prochain milestone intelligent
# ---------------------------------------------------------------------------

def next_milestone(current_weight: float, targets: tuple[float, ...], velocity: float | None = None, measurements: int = 0) -> dict[str, Any]:
    """Trouve le prochain palier atteignable pour la motivation.

    Cherche d'abord les objectifs configurés, sinon le prochain chiffre rond en dessous.
    Ne retourne un ETA que si les données sont suffisantes (>= 7 mesures)
    et la vitesse est physiologiquement plausible (<= 2 kg/sem).
    """
    # Chercher parmi les objectifs configurés
    reachable_targets = sorted([t for t in targets if t < current_weight], reverse=True)

    if reachable_targets:
        next_target = reachable_targets[0]
        remaining = current_weight - next_target
        label = f"Objectif {next_target:.0f} kg"
    else:
        # Prochain chiffre rond en dessous
        next_target = float(int(current_weight))  # ex: 102.6 → 102
        if next_target >= current_weight:
            next_target -= 1
        remaining = current_weight - next_target
        label = f"Palier {next_target:.0f} kg"

    # Si le prochain objectif est trop loin (> 10 kg), proposer un palier intermédiaire
    if remaining > 10:
        intermediate = float(int(current_weight / 5) * 5)  # arrondi au 5 inférieur
        if intermediate < current_weight:
            next_target = intermediate
            remaining = current_weight - next_target
            label = f"Palier intermédiaire {next_target:.0f} kg"

    # ETA basé sur la vitesse actuelle — avec garde-fous
    eta_days = None
    eta_confidence = None
    if velocity is not None and velocity < -0.01:
        # Garde-fou 1: pas d'ETA si < 7 mesures
        if measurements < 7:
            eta_confidence = "fragile"
        # Garde-fou 2: si la vitesse implique > 2 kg/sem, c'est du bruit
        elif abs(velocity) > 2.0:
            eta_confidence = "optimiste"
            # Utiliser une vitesse plafonnée réaliste de 0.75 kg/sem
            eta_days = int(remaining / 0.75 * 7)
        else:
            eta_days = int(remaining / abs(velocity) * 7)
            eta_confidence = "solide" if measurements >= 14 else "modérée"

    return {
        "target": next_target,
        "remaining": round(remaining, 1),
        "label": label,
        "eta_days": eta_days,
        "eta_confidence": eta_confidence,
    }


