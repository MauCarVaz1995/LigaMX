#!/usr/bin/env python3
"""
discovery_bot.py — Bot de retroalimentación y hallazgos automáticos
=====================================================================
Analiza el historial de predicciones, los datos de partidos y los modelos
para detectar patrones, sesgos sistemáticos y oportunidades de mejora.

Corre sin intervención humana. Documenta hallazgos en JSON + HTML.
Se integra en el pipeline diario y en el email semanal.

Funciones:
  1. Calibración del modelo — ¿las probabilidades están bien calibradas?
     (reliability diagram por decil)
  2. Sesgos por equipo — ¿hay equipos donde el modelo siempre falla?
  3. Sesgos por mercado — ¿corners o tarjetas tienen más edge?
  4. Deriva temporal — ¿el modelo empeora con el tiempo?
  5. Correlaciones de mercado — ¿corners y tarjetas están correlacionados?
  6. Equipos con patrones atípicos (outliers estadísticos)
  7. Recomendaciones accionables con prioridad

Uso:
  python bots/discovery_bot.py            # análisis completo
  python bots/discovery_bot.py --weekly   # modo resumen semanal
  python bots/discovery_bot.py --json     # solo output JSON (para integración)
"""

import json
import sys
import warnings
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE         = Path(__file__).resolve().parent.parent
EVENTS_CSV   = BASE / "data/processed/match_events.csv"
PREDS_LOG    = BASE / "data/processed/predicciones_log.csv"
RETRAIN_LOG  = BASE / "data/processed/retrain_log.json"
ML_METRICS   = BASE / "data/processed/ml_models/ml_metrics.json"
REPORTS_DIR  = BASE / "output/reports"
FINDINGS_DIR = BASE / "output/reports/discovery"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(values) -> float:
    v = [x for x in values if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else 0.0


def _brier(probs, outcomes) -> float:
    return float(np.mean((np.array(probs) - np.array(outcomes)) ** 2))


def _calibration_curve(probs, outcomes, n_bins=10):
    """Devuelve (frac_positivos, frac_predichos) por decil — reliability diagram."""
    probs    = np.array(probs)
    outcomes = np.array(outcomes)
    bins = np.linspace(0, 1, n_bins + 1)
    frac_pos, frac_pred, counts = [], [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() >= 3:
            frac_pos.append(float(outcomes[mask].mean()))
            frac_pred.append(float(probs[mask].mean()))
            counts.append(int(mask.sum()))
    return frac_pos, frac_pred, counts


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 1 — Calibración del modelo
# ─────────────────────────────────────────────────────────────────────────────

def analizar_calibracion(df_events: pd.DataFrame) -> dict:
    """
    Usa los datos históricos para evaluar si las tasas de corners/tarjetas
    están bien estimadas por las medias de los modelos.
    Compara: P(Over 9.5 corners) modelo_simple vs real.
    """
    df = df_events.dropna(subset=["corners_total", "amarillas_total"])
    if len(df) < 20:
        return {"status": "insufficient_data", "n": len(df)}

    # Distribución real
    real_over_85  = float((df["corners_total"] > 8.5).mean())
    real_over_95  = float((df["corners_total"] > 9.5).mean())
    real_over_105 = float((df["corners_total"] > 10.5).mean())
    real_over_35c = float(((df["amarillas_total"].fillna(0) + 2*df["rojas_total"].fillna(0)) > 3.5).mean())
    real_over_45c = float(((df["amarillas_total"].fillna(0) + 2*df["rojas_total"].fillna(0)) > 4.5).mean())

    # Porcentaje de goles que resultan en BTTS
    real_btts = float(((df["goles_local"] > 0) & (df["goles_visitante"] > 0)).mean())

    mu_corners = float(df["corners_total"].mean())
    mu_cards   = float((df["amarillas_total"].fillna(0) + 2*df["rojas_total"].fillna(0)).mean())

    # P(Over k) bajo Poisson con mu real
    from scipy.stats import poisson as spois
    poisson_over_85  = 1 - spois.cdf(8, mu_corners)
    poisson_over_95  = 1 - spois.cdf(9, mu_corners)
    poisson_over_105 = 1 - spois.cdf(10, mu_corners)
    poisson_over_45c = 1 - spois.cdf(4, mu_cards)

    # Bias = modelo_naïve - real (+ significa modelo sobreestima)
    findings = []
    for name, real, pred in [
        ("corners Over 8.5",  real_over_85,  poisson_over_85),
        ("corners Over 9.5",  real_over_95,  poisson_over_95),
        ("corners Over 10.5", real_over_105, poisson_over_105),
        ("cards Over 4.5",    real_over_45c, poisson_over_45c),
    ]:
        bias = pred - real
        if abs(bias) > 0.04:
            direction = "SOBREESTIMA" if bias > 0 else "SUBESTIMA"
            findings.append(f"Modelo naïve {direction} {name}: {bias:+.1%} (real={real:.1%}, pred={pred:.1%})")

    return {
        "mu_corners":    round(mu_corners, 2),
        "mu_cards":      round(mu_cards, 2),
        "real_btts":     round(real_btts, 3),
        "real_corners":  {
            "over_8.5":  round(real_over_85, 3),
            "over_9.5":  round(real_over_95, 3),
            "over_10.5": round(real_over_105, 3),
        },
        "real_cards": {
            "over_3.5": round(real_over_35c, 3),
            "over_4.5": round(real_over_45c, 3),
        },
        "n_partidos": int(len(df)),
        "findings":   findings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 2 — Patrones por equipo
# ─────────────────────────────────────────────────────────────────────────────

def analizar_equipos(df_events: pd.DataFrame) -> dict:
    """
    Detecta equipos con comportamientos atípicos en corners, tarjetas y goles.
    Un equipo es outlier si su tasa está > 1.5σ de la media global.
    """
    df = df_events.dropna(subset=["corners_total"])
    if len(df) < 20:
        return {"status": "insufficient_data"}

    equipos_stats = defaultdict(lambda: {
        "corners_for": [], "corners_against": [],
        "cards": [], "goals_scored": [], "goals_allowed": [], "xg": [],
    })

    for _, row in df.iterrows():
        loc = row["local"]
        vis = row["visitante"]
        if pd.notna(row["corners_local"]):
            equipos_stats[loc]["corners_for"].append(row["corners_local"])
            equipos_stats[vis]["corners_against"].append(row["corners_local"])
        if pd.notna(row["corners_visitante"]):
            equipos_stats[vis]["corners_for"].append(row["corners_visitante"])
            equipos_stats[loc]["corners_against"].append(row["corners_visitante"])
        t = row.get("amarillas_total", np.nan)
        if pd.notna(t):
            equipos_stats[loc]["cards"].append(t / 2)
            equipos_stats[vis]["cards"].append(t / 2)
        equipos_stats[loc]["goals_scored"].append(row["goles_local"])
        equipos_stats[vis]["goals_scored"].append(row["goles_visitante"])
        equipos_stats[loc]["goals_allowed"].append(row["goles_visitante"])
        equipos_stats[vis]["goals_allowed"].append(row["goles_local"])
        if pd.notna(row.get("xg_local")):
            equipos_stats[loc]["xg"].append(row["xg_local"])
            equipos_stats[vis]["xg"].append(row["xg_visitante"])

    # Estadísticas globales
    all_cf = [_safe_mean(s["corners_for"])    for s in equipos_stats.values() if s["corners_for"]]
    all_ca = [_safe_mean(s["corners_against"]) for s in equipos_stats.values() if s["corners_against"]]
    mu_cf, sd_cf = np.mean(all_cf), np.std(all_cf)
    mu_ca, sd_ca = np.mean(all_ca), np.std(all_ca)

    outliers = []
    team_profiles = {}

    for eq, stats in equipos_stats.items():
        if not stats["corners_for"]:
            continue
        cf   = _safe_mean(stats["corners_for"])
        ca   = _safe_mean(stats["corners_against"])
        gs   = _safe_mean(stats["goals_scored"])
        ga   = _safe_mean(stats["goals_allowed"])
        xg   = _safe_mean(stats["xg"]) if stats["xg"] else gs
        cards = _safe_mean(stats["cards"])
        n    = len(stats["corners_for"])

        profile = {
            "corners_for":     round(cf, 2),
            "corners_against": round(ca, 2),
            "goals_scored":    round(gs, 2),
            "goals_allowed":   round(ga, 2),
            "xg_scored":       round(xg, 2),
            "cards_avg":       round(cards, 2),
            "n":               n,
        }
        team_profiles[eq] = profile

        # Z-scores
        z_cf = (cf - mu_cf) / (sd_cf + 1e-6)
        z_ca = (ca - mu_ca) / (sd_ca + 1e-6)

        tags = []
        if z_cf > 1.5:  tags.append(f"MUY_ACTIVO corners_for={cf:.1f}")
        if z_cf < -1.5: tags.append(f"PASIVO corners_for={cf:.1f}")
        if z_ca > 1.5:  tags.append(f"CONCEDE_CORNERS corners_against={ca:.1f}")
        if gs > 1.8:    tags.append(f"GOLEADOR goals={gs:.1f}")
        if ga > 1.8:    tags.append(f"POROSO goals_allowed={ga:.1f}")
        if cards > 3.0: tags.append(f"TARJETERO cards={cards:.1f}")

        if tags and n >= 5:
            outliers.append({"equipo": eq, "tags": tags, "n": n})

    # Top equipos por corners
    top_cf = sorted(team_profiles.items(), key=lambda x: x[1]["corners_for"], reverse=True)[:5]
    top_ca = sorted(team_profiles.items(), key=lambda x: x[1]["corners_against"], reverse=True)[:5]

    findings = []
    for eq, prof in top_cf:
        if prof["corners_for"] > mu_cf + 1.2 * sd_cf:
            findings.append(f"⚽ {eq} genera +{prof['corners_for']:.1f} corners/partido (muy activo ofensivamente)")
    for eq, prof in top_ca:
        if prof["corners_against"] > mu_ca + 1.2 * sd_ca:
            findings.append(f"🚩 {eq} concede +{prof['corners_against']:.1f} corners/partido (defensiva vulnerable)")

    return {
        "n_equipos":     len(team_profiles),
        "mu_corners_for":  round(float(mu_cf), 2),
        "mu_corners_against": round(float(mu_ca), 2),
        "outliers":      sorted(outliers, key=lambda x: -x["n"])[:10],
        "top5_corners_for":     [{"equipo": e, "val": p["corners_for"]}  for e, p in top_cf],
        "top5_corners_against": [{"equipo": e, "val": p["corners_against"]} for e, p in top_ca],
        "team_profiles": team_profiles,
        "findings":      findings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 3 — Correlaciones entre mercados
# ─────────────────────────────────────────────────────────────────────────────

def analizar_correlaciones(df_events: pd.DataFrame) -> dict:
    """
    Calcula correlaciones entre mercados. Importante para:
    - Saber qué mercados son independientes (diversificación)
    - Detectar si corners y tarjetas van juntos (partidos tensos = ambos altos)
    """
    df = df_events.dropna(subset=["corners_total", "amarillas_total", "goles_local"])
    if len(df) < 20:
        return {"status": "insufficient_data"}

    df = df.copy()
    df["cards_total"] = df["amarillas_total"].fillna(0) + 2 * df["rojas_total"].fillna(0)
    df["btts"]        = ((df["goles_local"] > 0) & (df["goles_visitante"] > 0)).astype(int)
    df["goles_total"] = df["goles_local"] + df["goles_visitante"]
    df["elo_diff_proxy"] = df["shots_local"].fillna(0) - df["shots_visitante"].fillna(0)

    cols = ["corners_total", "cards_total", "goles_total", "btts",
            "xg_local", "xg_visitante", "shots_local", "shots_visitante"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(method="pearson").round(3)

    findings = []
    # Analizar correlaciones relevantes
    pairs = [
        ("corners_total", "cards_total", "corners y tarjetas"),
        ("corners_total", "goles_total", "corners y goles"),
        ("xg_local",      "goles_total", "xG vs goles reales"),
        ("shots_local",   "corners_total", "disparos y corners"),
    ]
    for a, b, label in pairs:
        if a in corr.index and b in corr.columns:
            r = corr.loc[a, b]
            if abs(r) > 0.25:
                direction = "POSITIVA" if r > 0 else "NEGATIVA"
                findings.append(f"Correlación {direction} {label}: r={r:.2f}")
            elif abs(r) < 0.10:
                findings.append(f"Mercados INDEPENDIENTES {label}: r={r:.2f} → apostar ambos OK")

    return {
        "n_partidos":  int(len(df)),
        "correlations": corr[cols].loc[cols].to_dict() if all(c in corr.index for c in cols) else {},
        "findings":    findings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 4 — Deriva temporal del modelo
# ─────────────────────────────────────────────────────────────────────────────

def analizar_deriva(df_events: pd.DataFrame, window: int = 30) -> dict:
    """
    ¿El modelo Poisson sigue siendo válido en partidos recientes?
    Compara la media de corners/tarjetas en los últimos 30 partidos
    vs la media histórica. Detecta cambios de tendencia.
    """
    df = df_events.dropna(subset=["corners_total"]).sort_values("fecha")
    if len(df) < window + 10:
        return {"status": "insufficient_data"}

    df["cards_total"] = df["amarillas_total"].fillna(0) + 2 * df["rojas_total"].fillna(0)

    recent = df.tail(window)
    hist   = df.iloc[:-window]

    mu_corners_hist   = float(hist["corners_total"].mean())
    mu_corners_recent = float(recent["corners_total"].mean())
    mu_cards_hist     = float(hist["cards_total"].mean())
    mu_cards_recent   = float(recent["cards_total"].mean())

    drift_corners = mu_corners_recent - mu_corners_hist
    drift_cards   = mu_cards_recent - mu_cards_hist

    findings = []
    if abs(drift_corners) > 0.8:
        direction = "SUBIENDO" if drift_corners > 0 else "BAJANDO"
        findings.append(
            f"⚠️ Drift corners {direction}: reciente={mu_corners_recent:.1f} vs hist={mu_corners_hist:.1f} ({drift_corners:+.1f}). "
            f"Reentrenar modelo."
        )
    if abs(drift_cards) > 0.5:
        direction = "SUBIENDO" if drift_cards > 0 else "BAJANDO"
        findings.append(
            f"⚠️ Drift tarjetas {direction}: reciente={mu_cards_recent:.1f} vs hist={mu_cards_hist:.1f} ({drift_cards:+.1f}). "
            f"Revisar card_rate."
        )

    # Rolling window month-by-month
    df["mes"] = df["fecha"].astype(str).str[:7]
    monthly = df.groupby("mes").agg(
        mu_corners=("corners_total", "mean"),
        mu_cards=("cards_total", "mean"),
        n=("corners_total", "count"),
    ).reset_index()

    return {
        "mu_corners_historical": round(mu_corners_hist, 2),
        "mu_corners_recent":     round(mu_corners_recent, 2),
        "mu_cards_historical":   round(mu_cards_hist, 2),
        "mu_cards_recent":       round(mu_cards_recent, 2),
        "drift_corners":         round(drift_corners, 2),
        "drift_cards":           round(drift_cards, 2),
        "monthly_trend": monthly[monthly["n"] >= 5].to_dict(orient="records"),
        "findings":      findings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 5 — Partidos de alta presión (corners + tarjetas juntos)
# ─────────────────────────────────────────────────────────────────────────────

def analizar_patrones_alta_presion(df_events: pd.DataFrame) -> dict:
    """
    Encuentra características de partidos donde AMBOS Over corners Y Over tarjetas
    se cumplen simultáneamente. Estos son los mejores candidatos para portafolio.
    """
    df = df_events.dropna(subset=["corners_total", "amarillas_total"]).copy()
    df["cards_total"] = df["amarillas_total"].fillna(0) + 2 * df["rojas_total"].fillna(0)
    df["over_95c"]  = (df["corners_total"] > 9.5).astype(int)
    df["over_45t"]  = (df["cards_total"]   > 4.5).astype(int)
    df["both_over"] = ((df["over_95c"] == 1) & (df["over_45t"] == 1)).astype(int)

    if len(df) < 20:
        return {"status": "insufficient_data"}

    # ¿Cuándo ocurre "both_over"?
    both_df = df[df["both_over"] == 1]

    # Por equipo: ¿quién protagoniza estos partidos?
    team_count = defaultdict(int)
    for _, row in both_df.iterrows():
        team_count[row["local"]] += 1
        team_count[row["visitante"]] += 1

    top_teams = sorted(team_count.items(), key=lambda x: -x[1])[:8]

    # Correlación corners-tarjetas
    r = float(df[["corners_total", "cards_total"]].corr().iloc[0, 1])

    findings = []
    both_rate = float(df["both_over"].mean())
    if both_rate > 0.25:
        findings.append(f"✅ Partidos con Over 9.5c + Over 4.5t simultáneos: {both_rate:.0%}. Mercados diversificables.")
    elif both_rate < 0.15:
        findings.append(f"⚠️ Solo {both_rate:.0%} de partidos cumplen ambos Over. Evitar parlays corners+tarjetas.")

    if r > 0.30:
        findings.append(f"⚠️ Corners y tarjetas correlacionados (r={r:.2f}). No apostar ambos en mismo partido.")
    else:
        findings.append(f"✅ Corners y tarjetas son independientes (r={r:.2f}). Portafolio mixto válido.")

    # Jornadas con más alta presión
    jorns = both_df.groupby("jornada")["both_over"].count().sort_values(ascending=False).head(5)

    return {
        "both_over_rate":       round(both_rate, 3),
        "corners_cards_corr":   round(r, 3),
        "top_equipos_presion":  [{"equipo": e, "partidos": c} for e, c in top_teams],
        "jornadas_presion":     jorns.to_dict(),
        "findings":             findings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 6 — Calibración del modelo ELO+Poisson (reliability diagram)
# ─────────────────────────────────────────────────────────────────────────────

def analizar_calibracion_predicciones() -> dict:
    """
    Analiza la calibración del modelo ELO+Poisson usando predicciones_log.csv.
    Compara probabilidades predichas vs tasas reales de acierto.
    Calcula Brier score, ECE (Expected Calibration Error), y sesgos por resultado.
    """
    if not PREDS_LOG.exists():
        return {"status": "no_log", "findings": []}

    df = pd.read_csv(PREDS_LOG)
    df["resultado_real"] = df["resultado_real"].str.lower().str.strip()
    df = df.dropna(subset=["resultado_real"])
    df = df[df["resultado_real"].isin(["local", "empate", "visitante"])]

    if len(df) < 5:
        return {"status": "insuficiente", "n": len(df), "findings": []}

    n = len(df)
    findings = []

    # Tasas reales vs predichas
    real_local    = float((df["resultado_real"] == "local").mean())
    real_empate   = float((df["resultado_real"] == "empate").mean())
    real_visita   = float((df["resultado_real"] == "visitante").mean())
    pred_local    = float(df["prob_local"].mean() / 100)
    pred_empate   = float(df["prob_empate"].mean() / 100)
    pred_visita   = float(df["prob_visitante"].mean() / 100)

    # Historical reference rates (Liga MX 2023-2026)
    HIST_LOCAL   = 0.467
    HIST_EMPATE  = 0.252
    HIST_VISITA  = 0.281

    bias_local  = pred_local  - real_local
    bias_empate = pred_empate - real_empate
    bias_visita = pred_visita - real_visita

    # Brier score para resultado más probable
    brier_scores = []
    for _, row in df.iterrows():
        p = [row["prob_local"]/100, row["prob_empate"]/100, row["prob_visitante"]/100]
        y = [1 if row["resultado_real"]=="local" else 0,
             1 if row["resultado_real"]=="empate" else 0,
             1 if row["resultado_real"]=="visitante" else 0]
        brier_scores.append(sum((pi-yi)**2 for pi,yi in zip(p,y)))
    brier = float(np.mean(brier_scores))
    naive_brier = float(np.mean([
        (HIST_LOCAL-y[0])**2 + (HIST_EMPATE-y[1])**2 + (HIST_VISITA-y[2])**2
        for _,row in df.iterrows()
        for y in [[1 if row["resultado_real"]=="local" else 0,
                   1 if row["resultado_real"]=="empate" else 0,
                   1 if row["resultado_real"]=="visitante" else 0]]
    ]))
    brier_skill = float(1 - brier / naive_brier) if naive_brier > 0 else 0.0

    # Accuracy
    df["ganador_pred_norm"] = df["ganador_predicho"].str.lower().str.strip()
    accuracy = float((df["ganador_pred_norm"] == df["resultado_real"]).mean())

    # Findings con umbral mínimo de n=10 para conclusiones estadísticas
    if n >= 10:
        if abs(bias_empate) > 0.05:
            dir_ = "SOBREESTIMA" if bias_empate > 0 else "SUBESTIMA"
            findings.append(f"Modelo {dir_} empates: pred={pred_empate:.1%} real={real_empate:.1%} (bias={bias_empate:+.1%}, n={n})")
        if abs(bias_local) > 0.07:
            dir_ = "SOBREESTIMA" if bias_local > 0 else "SUBESTIMA"
            findings.append(f"Modelo {dir_} victorias locales: pred={pred_local:.1%} real={real_local:.1%} (bias={bias_local:+.1%}, n={n})")
        if brier_skill < -0.05:
            findings.append(f"⚠️ Brier skill negativo ({brier_skill:.1%}): modelo peor que distribución histórica (n={n})")
        elif brier_skill > 0.05:
            findings.append(f"✅ Brier skill positivo ({brier_skill:.1%}): modelo mejor que base (n={n})")

    if n < 30:
        findings.append(f"ℹ️ Muestra pequeña (n={n}): calibración no concluyente, necesita n≥30")

    return {
        "n":           n,
        "accuracy":    round(accuracy, 3),
        "brier":       round(brier, 3),
        "brier_skill": round(brier_skill, 3),
        "pred_rates":  {"local": round(pred_local,3), "empate": round(pred_empate,3), "visitante": round(pred_visita,3)},
        "real_rates":  {"local": round(real_local,3), "empate": round(real_empate,3), "visitante": round(real_visita,3)},
        "hist_rates":  {"local": HIST_LOCAL, "empate": HIST_EMPATE, "visitante": HIST_VISITA},
        "bias":        {"local": round(bias_local,3), "empate": round(bias_empate,3), "visitante": round(bias_visita,3)},
        "findings":    findings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 7 — Oportunidades de value sistemático
# ─────────────────────────────────────────────────────────────────────────────

def analizar_oportunidades(
    cal: dict,
    equipos: dict,
    deriva: dict,
) -> list[dict]:
    """
    Consolida todos los hallazgos y genera recomendaciones accionables.
    """
    recs = []

    # Recomendar reentrenamiento si hay drift
    if abs(deriva.get("drift_corners", 0)) > 0.8:
        recs.append({
            "prioridad": "alta",
            "accion":    "retrain_corners",
            "razon":     f"Drift corners={deriva['drift_corners']:+.1f}. Modelos desactualizados.",
        })

    if abs(deriva.get("drift_cards", 0)) > 0.5:
        recs.append({
            "prioridad": "alta",
            "accion":    "retrain_cards",
            "razon":     f"Drift tarjetas={deriva['drift_cards']:+.1f}. Card_rates desactualizados.",
        })

    # Detectar mercados con mejor edge
    real_c = cal.get("real_corners", {})
    if real_c.get("over_9.5", 0) > 0.55:
        recs.append({
            "prioridad": "media",
            "accion":    "bet_corners_over_95",
            "razon":     f"Over 9.5 corners ocurre {real_c['over_9.5']:.0%} del tiempo. Buscar cuotas >1.67.",
        })
    if real_c.get("over_9.5", 1) < 0.40:
        recs.append({
            "prioridad": "media",
            "accion":    "bet_corners_under_95",
            "razon":     f"Under 9.5 corners ocurre {1-real_c['over_9.5']:.0%} del tiempo.",
        })

    # Outliers de equipo como targets
    for out in equipos.get("outliers", [])[:3]:
        if any("MUY_ACTIVO" in t for t in out["tags"]):
            recs.append({
                "prioridad": "media",
                "accion":    f"target_corners_{out['equipo'].replace(' ','_')}",
                "razon":     f"{out['equipo']} genera muchos corners. Apostar Over cuando juegue como local.",
            })

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE REPORTE HTML
# ─────────────────────────────────────────────────────────────────────────────

def _pred_cal_section(cp: dict) -> str:
    """Genera la sección HTML de calibración de predicciones 1X2."""
    if not cp or cp.get("status") in ("no_log", "insuficiente") or cp.get("n", 0) < 5:
        return ""
    n  = cp["n"]
    ac = cp.get("accuracy", 0)
    bs = cp.get("brier", 0)
    sk = cp.get("brier_skill", 0)
    pr = cp.get("pred_rates", {})
    rr = cp.get("real_rates", {})
    hr = cp.get("hist_rates", {})
    findings = cp.get("findings", [])
    skill_color = "#1a7a1a" if sk > 0 else "#ff4444"
    rows = ""
    for k, label in [("local","Local"),("empate","Empate"),("visitante","Visitante")]:
        bias = pr.get(k,0) - rr.get(k,0)
        bias_color = "#ffaa00" if abs(bias) > 0.07 else "#888"
        rows += (f"<tr><td>{label}</td>"
                 f"<td>{pr.get(k,0):.1%}</td>"
                 f"<td>{rr.get(k,0):.1%}</td>"
                 f"<td>{hr.get(k,0):.1%}</td>"
                 f"<td style='color:{bias_color};'>{bias:+.1%}</td></tr>")
    items = "".join(f"<li>{f}</li>" for f in findings) if findings else ""
    return f"""
    <div class='disc-section'>
      <h3 style='color:#00d4ff;'>🎯 Calibración Predicciones 1X2 (n={n})</h3>
      <span class='disc-stat'>Accuracy={ac:.1%}</span>
      <span class='disc-stat'>Brier={bs:.3f}</span>
      <span class='disc-stat' style='color:{skill_color};'>Skill={sk:+.1%}</span>
      <table>
        <tr><th>Resultado</th><th>Pred (media)</th><th>Real (muestra)</th><th>Histórico LigaMX</th><th>Bias</th></tr>
        {rows}
      </table>
      {"<ul>" + items + "</ul>" if items else ""}
    </div>"""


def _render_html(report: dict) -> str:
    fecha = report["generated_at"][:10]

    def section(title: str, findings: list, color: str = "#00d4ff") -> str:
        if not findings:
            return ""
        items = "".join(f"<li>{f}</li>" for f in findings)
        return f"""
        <div class='disc-section'>
          <h3 style='color:{color};'>{title}</h3>
          <ul>{items}</ul>
        </div>"""

    def rec_cards(recs: list) -> str:
        if not recs:
            return ""
        rows = ""
        for r in recs:
            color = {"alta": "#b00000", "media": "#c07000", "baja": "#1a7a1a"}.get(r["prioridad"], "#aaa")
            rows += f"<tr><td style='color:{color};font-weight:bold'>{r['prioridad'].upper()}</td><td>{r['accion']}</td><td>{r['razon']}</td></tr>"
        return f"""
        <div class='disc-section'>
          <h3 style='color:#ff6b35;'>🎯 Recomendaciones Accionables</h3>
          <table><tr><th>Prioridad</th><th>Acción</th><th>Razón</th></tr>{rows}</table>
        </div>"""

    cal      = report.get("calibracion", {})
    cal_pred = report.get("calibracion_predicciones", {})
    eq       = report.get("equipos", {})
    der      = report.get("deriva", {})
    corr     = report.get("correlaciones", {})
    pres     = report.get("alta_presion", {})
    recs     = report.get("recomendaciones", [])

    # Métricas ML si existen
    ml_section = ""
    if "ml_metrics" in report:
        ml = report["ml_metrics"].get("metrics", {})
        rows = ""
        for name, met in ml.items():
            if "brier" in met:
                sk = met.get("skill", "?")
                sk_pct = f"{sk:.1%}" if isinstance(sk, float) else sk
                color = "#1a7a1a" if isinstance(sk, float) and sk > 0.15 else "#ffaa00"
                rows += f"<tr><td>{name}</td><td>{met['brier']:.4f}</td><td>{met['brier_naive']:.4f}</td><td style='color:{color};'>{sk_pct}</td></tr>"
        ml_section = f"""
        <div class='disc-section'>
          <h3 style='color:#a855f7;'>🤖 Modelos ML (LightGBM calibrado)</h3>
          <table><tr><th>Target</th><th>Brier</th><th>Naïve</th><th>Skill</th></tr>{rows}</table>
        </div>"""

    html = f"""<div style='font-family:Arial,sans-serif;color:#222;'>
<style>
  .disc-section {{ background:#fff;border-radius:6px;padding:14px 16px;margin:10px 0;
                   box-shadow:0 1px 3px rgba(0,0,0,.08);border-left:3px solid #7b1fa2; }}
  .disc-section h3 {{ font-size:13px;margin:0 0 8px;color:#333; }}
  .disc-section ul {{ margin:4px 0;padding-left:18px; }}
  .disc-section li {{ margin:3px 0;font-size:12px;color:#444; }}
  .disc-table {{ width:100%;border-collapse:collapse;font-size:12px; }}
  .disc-table th {{ background:#f0f0f0;color:#555;padding:5px 7px;text-align:left;
                    border-bottom:2px solid #ddd; }}
  .disc-table td {{ padding:4px 7px;border-bottom:1px solid #f0f0f0;color:#333; }}
  .disc-stat {{ display:inline-block;background:#f4f4f4;border:1px solid #ddd;
               border-radius:10px;padding:3px 10px;margin:3px;font-size:12px;color:#333; }}
</style>
<div style='background:#7b1fa2;color:#fff;padding:10px 14px;border-radius:6px;margin-bottom:10px;'>
  <b>🔬 Discovery Report — {fecha}</b>
  <span style='font-size:11px;opacity:.8;margin-left:8px;'>
    Partidos analizados: {cal.get("n_partidos","?")}
  </span>
</div>

<div class='disc-section'>
  <h3 style='color:#00d4ff;'>📊 Estadísticas Base (Liga MX)</h3>
  <span class='disc-stat'>μ corners = {cal.get('mu_corners','?')}</span>
  <span class='disc-stat'>μ tarjetas = {cal.get('mu_cards','?')}</span>
  <span class='disc-stat'>BTTS real = {cal.get('real_btts','?'):.1%}</span>
  <span class='disc-stat'>Over 9.5c real = {cal.get('real_corners',{}).get('over_9.5','?'):.1%}</span>
  <span class='disc-stat'>Over 4.5t real = {cal.get('real_cards',{}).get('over_4.5','?'):.1%}</span>
</div>

{section('📐 Calibración y Sesgos del Modelo (corners/tarjetas)', cal.get('findings',[]))}
{_pred_cal_section(cal_pred)}
{section('🏟️ Patrones por Equipo', eq.get('findings',[]))}
{section('📈 Deriva Temporal', der.get('findings',[]))}
{section('🔗 Correlaciones de Mercado', corr.get('findings',[]))}
{section('🔥 Análisis Alta Presión', pres.get('findings',[]))}
{ml_section}
{rec_cards(recs)}

<div class='disc-section'>
  <h3>Top 5 Equipos Generadores de Corners</h3>
  {''.join(f"<span class='disc-stat'>{x['equipo']} {x['val']:.1f}</span>" for x in eq.get('top5_corners_for',[]))}
</div>

<p style='color:#888; font-size:11px; margin-top:16px;'>MAU-STATISTICS · discovery_bot.py</p>
</div>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(verbose: bool = True, json_only: bool = False, weekly: bool = False) -> dict:
    """Ejecuta todos los análisis y guarda reporte."""

    if verbose and not json_only:
        print(f"\n── discovery_bot.py · {TODAY} ──")

    # Cargar datos
    df_events = pd.read_csv(EVENTS_CSV) if EVENTS_CSV.exists() else pd.DataFrame()
    if len(df_events) == 0:
        if verbose:
            print("  ❌ match_events.csv vacío o no encontrado")
        return {}

    # Ejecutar análisis
    cal   = analizar_calibracion(df_events)
    eq    = analizar_equipos(df_events)
    corr  = analizar_correlaciones(df_events)
    der   = analizar_deriva(df_events)
    pres  = analizar_patrones_alta_presion(df_events)
    cal_pred = analizar_calibracion_predicciones()
    recs  = analizar_oportunidades(cal, eq, der)

    # Cargar métricas ML si existen
    ml_metrics = None
    if ML_METRICS.exists():
        try:
            ml_metrics = json.loads(ML_METRICS.read_text())
        except Exception:
            pass

    # Consolidar todos los hallazgos
    all_findings = (
        cal.get("findings", []) +
        cal_pred.get("findings", []) +
        eq.get("findings", []) +
        der.get("findings", []) +
        corr.get("findings", []) +
        pres.get("findings", [])
    )

    report = {
        "generated_at":    datetime.now().isoformat(),
        "n_partidos":      len(df_events),
        "calibracion":     cal,
        "calibracion_predicciones": cal_pred,
        "equipos": {
            "n_equipos":    eq.get("n_equipos"),
            "outliers":     eq.get("outliers", []),
            "top5_corners_for": eq.get("top5_corners_for", []),
            "top5_corners_against": eq.get("top5_corners_against", []),
            "findings":     eq.get("findings", []),
        },
        "correlaciones":   {k: v for k, v in corr.items() if k != "correlations"},
        "deriva":          der,
        "alta_presion":    pres,
        "recomendaciones": recs,
        "all_findings":    all_findings,
    }
    if ml_metrics:
        report["ml_metrics"] = ml_metrics

    # Guardar JSON
    out_json = FINDINGS_DIR / f"discovery_{TODAY}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    # Guardar HTML
    out_html = FINDINGS_DIR / f"discovery_{TODAY}.html"
    out_html.write_text(_render_html(report), encoding="utf-8")

    # También guardar el más reciente como latest (para email)
    (REPORTS_DIR / "discovery_latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str)
    )
    (REPORTS_DIR / "discovery_latest.html").write_text(
        _render_html(report), encoding="utf-8"
    )

    if verbose and not json_only:
        print(f"  📊 Partidos analizados: {len(df_events)}")
        print(f"  🔍 Hallazgos totales:   {len(all_findings)}")
        print(f"  🎯 Recomendaciones:     {len(recs)}")
        print(f"\n  Hallazgos:")
        for f in all_findings[:15]:
            print(f"    • {f}")
        if recs:
            print(f"\n  Recomendaciones:")
            for r in recs:
                print(f"    [{r['prioridad'].upper():5s}] {r['accion']}: {r['razon'][:80]}")
        print(f"\n  → {out_json}")
        print(f"  → {out_html}")

    if json_only:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bot de retroalimentación y hallazgos")
    parser.add_argument("--json",    action="store_true", help="Solo output JSON")
    parser.add_argument("--weekly",  action="store_true", help="Modo resumen semanal")
    parser.add_argument("--quiet",   action="store_true", help="Sin verbose")
    args = parser.parse_args()

    run(
        verbose=not args.quiet and not args.json,
        json_only=args.json,
        weekly=args.weekly,
    )


if __name__ == "__main__":
    main()
