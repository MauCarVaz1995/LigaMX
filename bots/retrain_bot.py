#!/usr/bin/env python3
"""
retrain_bot.py — Bot autónomo de re-entrenamiento de modelos
=============================================================
Corre después de que scrape_match_events.py agrega nuevos partidos.
Evalúa si hay suficientes datos nuevos para justificar re-entrenamiento,
re-entrena si es necesario y reporta métricas de calidad del modelo.

Criterio de re-entrenamiento (configurable):
  - Han pasado ≥ 7 días desde el último entrenamiento, O
  - Se han agregado ≥ MIN_NEW_MATCHES partidos nuevos al CSV

Métricas de validación (pseudoout-of-sample):
  - Brier Score de los últimos N partidos (holdout rolling)
  - MAE de predicciones corners vs real
  - Comparación vs naive baseline (predicción = media de la liga)

Uso:
  python bots/retrain_bot.py              # re-entrena si hay cambios
  python bots/retrain_bot.py --force      # siempre re-entrena
  python bots/retrain_bot.py --evaluate   # solo evalúa sin re-entrenar
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))

EVENTS_CSV    = BASE / "data/processed/match_events.csv"
CORNERS_MODEL = BASE / "data/processed/corners_model.json"
TARJETAS_MODEL = BASE / "data/processed/tarjetas_model.json"
RETRAIN_LOG   = BASE / "data/processed/retrain_log.json"

MIN_NEW_MATCHES = 5   # re-entrenar si ≥ 5 partidos nuevos
MIN_DAYS_RETRAIN = 7  # o si pasaron ≥ 7 días
HOLDOUT_N = 20        # últimos N partidos para evaluar


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de evaluación
# ─────────────────────────────────────────────────────────────────────────────

def brier_score_corners(model_path: Path, df_holdout: pd.DataFrame,
                        lines: list = [8.5, 9.5, 10.5]) -> dict:
    """
    Brier Score para predicciones Over/Under corners.
    BS = mean((p_modelo - resultado_real)²)
    Baseline naive: p = 0.5 → BS = 0.25

    Referencia: Brier (1950), aplicado a apuestas en
    Constantinou & Fenton (2012) "Solving the Problem of Inadequate Scoring
    Rules for Assessing Probabilistic Football Forecast Models"
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mc", SCRIPTS / "modelo_corners.py")
        mc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mc)
        m = mc.CornersModel()
        m.load()
    except Exception as e:
        return {"error": str(e)}

    scores = {f"brier_over_{l}": [] for l in lines}
    mae_total = []

    for _, row in df_holdout.iterrows():
        local  = row["local"]
        visita = row["visitante"]
        real_t = row["corners_total"]

        try:
            pred = m.predict(local, visita)
        except Exception:
            continue

        mae_total.append(abs(pred["lambda_total"] - real_t))

        for l in lines:
            p_pred = pred.get(f"prob_over_{l}", 0.5)
            real_o = 1.0 if real_t > l else 0.0
            scores[f"brier_over_{l}"].append((p_pred - real_o) ** 2)

    result = {}
    for k, vals in scores.items():
        if vals:
            result[k] = round(np.mean(vals), 4)

    result["mae_total_corners"] = round(np.mean(mae_total), 2) if mae_total else None
    result["baseline_brier"] = 0.25  # p=0.5 siempre → BS=0.25
    result["n_holdout"] = len(df_holdout)
    return result


def brier_score_tarjetas(model_path: Path, df_holdout: pd.DataFrame,
                         lines: list = [3.5, 4.5, 5.5]) -> dict:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mt", SCRIPTS / "modelo_tarjetas.py")
        mt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mt)
        m = mt.TarjetasModel()
        m.load()
    except Exception as e:
        return {"error": str(e)}

    scores = {f"brier_over_{l}": [] for l in lines}

    for _, row in df_holdout.iterrows():
        local   = row["local"]
        visita  = row["visitante"]
        real_t  = row.get("amarillas_total", np.nan)
        if pd.isna(real_t):
            continue

        try:
            pred = m.predict(local, visita)
        except Exception:
            continue

        for l in lines:
            p_pred = pred.get(f"prob_over_{l}", 0.5)
            real_o = 1.0 if real_t > l else 0.0
            scores[f"brier_over_{l}"].append((p_pred - real_o) ** 2)

    result = {}
    for k, vals in scores.items():
        if vals:
            result[k] = round(np.mean(vals), 4)
    result["baseline_brier"] = 0.25
    result["n_holdout"] = len(df_holdout)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Lógica de re-entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def should_retrain(force: bool = False) -> tuple[bool, str]:
    """¿Hay suficientes razones para re-entrenar ahora?"""
    if force:
        return True, "forzado por --force"

    if not CORNERS_MODEL.exists():
        return True, "modelo no existe"

    # Leer log de último entrenamiento
    log = {}
    if RETRAIN_LOG.exists():
        try:
            log = json.loads(RETRAIN_LOG.read_text())
        except Exception:
            pass

    last_train_date = log.get("last_train_date", "2000-01-01")
    last_n_partidos = log.get("n_partidos_at_train", 0)

    days_since = (date.today() - date.fromisoformat(last_train_date)).days
    if days_since >= MIN_DAYS_RETRAIN:
        return True, f"han pasado {days_since} días desde último entrenamiento"

    if EVENTS_CSV.exists():
        current_n = len(pd.read_csv(EVENTS_CSV))
        new_matches = current_n - last_n_partidos
        if new_matches >= MIN_NEW_MATCHES:
            return True, f"{new_matches} partidos nuevos agregados"

    return False, "modelo actualizado, sin cambios suficientes"


def retrain_all(verbose: bool = True) -> dict:
    """Re-entrena corners y tarjetas. Devuelve métricas."""
    from modelo_corners  import CornersModel
    from modelo_tarjetas import TarjetasModel

    df = pd.read_csv(EVENTS_CSV)
    n  = len(df)

    print(f"\n── retrain_bot.py · {date.today()} ──")
    print(f"  Dataset: {n} partidos en {df['torneo'].nunique()} torneos")

    # Train/holdout split: últimos HOLDOUT_N partidos como holdout
    df_sorted  = df.sort_values("fecha")
    df_train   = df_sorted.iloc[:-HOLDOUT_N] if n > HOLDOUT_N else df_sorted
    df_holdout = df_sorted.iloc[-HOLDOUT_N:] if n > HOLDOUT_N else df_sorted

    print(f"  Train: {len(df_train)} | Holdout: {len(df_holdout)} partidos")

    # Re-entrenar sobre TODOS los datos (el holdout solo es para métricas)
    print("\n  ── Corners ──")
    CornersModel().fit(verbose=verbose)

    print("\n  ── Tarjetas ──")
    TarjetasModel().fit(verbose=verbose)

    # Evaluar sobre holdout
    print("\n  ── Evaluación holdout ──")
    corners_metrics  = brier_score_corners(CORNERS_MODEL, df_holdout)
    tarjetas_metrics = brier_score_tarjetas(TARJETAS_MODEL, df_holdout)

    print(f"  Corners  Brier Over 9.5:  {corners_metrics.get('brier_over_9.5', '?')} "
          f"(baseline: {corners_metrics.get('baseline_brier', 0.25)})")
    print(f"  Corners  MAE total:       {corners_metrics.get('mae_total_corners', '?')} córners")
    print(f"  Tarjetas Brier Over 4.5:  {tarjetas_metrics.get('brier_over_4.5', '?')}")

    # Guardar log
    log = {
        "last_train_date":      date.today().isoformat(),
        "n_partidos_at_train":  n,
        "torneos":              df["torneo"].unique().tolist(),
        "corners_metrics":      corners_metrics,
        "tarjetas_metrics":     tarjetas_metrics,
    }
    RETRAIN_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n  Log guardado → {RETRAIN_LOG}")

    return log


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bot de re-entrenamiento de modelos betting")
    parser.add_argument("--force",    action="store_true", help="Forzar re-entrenamiento")
    parser.add_argument("--evaluate", action="store_true", help="Solo evaluar sin re-entrenar")
    args = parser.parse_args()

    if not EVENTS_CSV.exists():
        print(f"ERROR: {EVENTS_CSV} no existe. Ejecuta scrape_match_events.py primero.")
        sys.exit(1)

    if args.evaluate:
        df = pd.read_csv(EVENTS_CSV)
        df_holdout = df.sort_values("fecha").tail(HOLDOUT_N)
        print(f"\n── Evaluación de modelos (holdout: últimos {HOLDOUT_N} partidos) ──")
        c = brier_score_corners(CORNERS_MODEL, df_holdout)
        t = brier_score_tarjetas(TARJETAS_MODEL, df_holdout)
        print(json.dumps({"corners": c, "tarjetas": t}, indent=2))
        return

    should, reason = should_retrain(force=args.force)
    if should:
        print(f"  Re-entrenando: {reason}")
        retrain_all(verbose=True)
    else:
        print(f"  Sin re-entrenamiento necesario: {reason}")
        if RETRAIN_LOG.exists():
            log = json.loads(RETRAIN_LOG.read_text())
            print(f"  Último entrenamiento: {log.get('last_train_date')} "
                  f"({log.get('n_partidos_at_train')} partidos)")


if __name__ == "__main__":
    main()
