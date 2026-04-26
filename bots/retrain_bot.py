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
    print("\n  ── Corners (Dixon-Coles) ──")
    CornersModel().fit(verbose=verbose)

    print("\n  ── Tarjetas (Poisson) ──")
    TarjetasModel().fit(verbose=verbose)

    # ML models (LightGBM calibrado)
    ml_metrics = {}
    try:
        sys.path.insert(0, str(SCRIPTS))
        import modelo_ml as mml
        print("\n  ── ML models (LightGBM) ──")
        ml_result = mml.train(verbose=verbose)
        ml_metrics = ml_result.get("metrics", {})
    except Exception as e:
        print(f"  [warn] modelo_ml.py falló: {e}")

    # Evaluar sobre holdout
    print("\n  ── Evaluación holdout (Poisson) ──")
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
        "ml_metrics":           ml_metrics,
    }
    RETRAIN_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n  Log guardado → {RETRAIN_LOG}")

    return log


# ─────────────────────────────────────────────────────────────────────────────
# Auto-calibración: detecta y aplica correcciones de drift sin intervención manual
# ─────────────────────────────────────────────────────────────────────────────

DRIFT_LOG = BASE / "data/processed/drift_log.json"

def detect_and_fix_drift(verbose: bool = True) -> dict:
    """
    Lee match_events.csv y detecta desviaciones entre parámetros históricos
    y recientes (últimas 3 jornadas). Si el drift es > umbral, actualiza
    los parámetros globales en los modelos guardados SIN re-entrenamiento completo.

    Parámetros monitoreados:
    - mu_tarjetas: media tarjetas/partido (umbral drift > 0.5)
    - mu_corners:  media corners/partido (umbral drift > 0.8)
    - ratio_local: fracción partidos que gana local (umbral drift > 0.08)

    Devuelve dict con cambios aplicados.
    """
    if not EVENTS_CSV.exists():
        return {}

    df = pd.read_csv(EVENTS_CSV)
    if len(df) < 30:
        return {}

    df = df.sort_values("fecha").reset_index(drop=True)

    # Calcular columnas derivadas ANTES de slicear (evita SettingWithCopyWarning)
    if "tarjetas_total" not in df.columns:
        df["tarjetas_total"] = (
            df["amarillas_local"].fillna(0) +
            df["amarillas_visitante"].fillna(0) +
            2 * df["rojas_local"].fillna(0) +
            2 * df["rojas_visitante"].fillna(0)
        )
    if "corners_total" not in df.columns and \
       "corners_local" in df.columns and "corners_visitante" in df.columns:
        df["corners_total"] = df["corners_local"].fillna(0) + df["corners_visitante"].fillna(0)

    df_recent = df.tail(27).copy()   # ~3 jornadas × 9 partidos
    df_hist   = df                   # dataset completo

    fixes_applied = {}
    drift_report  = {}

    # ── Tarjetas ──
    for _df in [df, df_recent, df_hist]:
        if "tarjetas_total" not in _df.columns:
            pass  # ya calculado arriba
    mu_hist_t   = df_hist["tarjetas_total"].mean()
    mu_recent_t = df_recent["tarjetas_total"].mean()
    drift_t = abs(mu_recent_t - mu_hist_t)
    drift_report["tarjetas"] = {
        "mu_hist": round(mu_hist_t, 3),
        "mu_recent": round(mu_recent_t, 3),
        "drift": round(mu_recent_t - mu_hist_t, 3),
    }

    if drift_t > 0.5 and TARJETAS_MODEL.exists():
        # Actualizar mu en el modelo guardado con blend 70% hist + 30% reciente
        mu_blend = 0.70 * mu_hist_t + 0.30 * mu_recent_t
        try:
            model_data = json.loads(TARJETAS_MODEL.read_text())
            old_mu = model_data.get("mu", mu_hist_t)
            model_data["mu"] = round(mu_blend, 4)
            model_data["_drift_corrected"] = date.today().isoformat()
            TARJETAS_MODEL.write_text(json.dumps(model_data, ensure_ascii=False, indent=2))
            fixes_applied["tarjetas_mu"] = {
                "antes": round(old_mu, 3),
                "despues": round(mu_blend, 3),
                "razon": f"drift={mu_recent_t-mu_hist_t:+.2f} > 0.5",
            }
            if verbose:
                print(f"  [auto-fix] tarjetas mu: {old_mu:.3f} → {mu_blend:.3f} "
                      f"(drift {mu_recent_t-mu_hist_t:+.2f})")
        except Exception as e:
            if verbose:
                print(f"  [warn] no se pudo actualizar tarjetas_model.json: {e}")

    # ── Corners ──
    if "corners_total" in df.columns:
        mu_hist_c   = df_hist["corners_total"].mean()
        mu_recent_c = df_recent["corners_total"].mean()
        drift_c = abs(mu_recent_c - mu_hist_c)
        drift_report["corners"] = {
            "mu_hist": round(mu_hist_c, 3),
            "mu_recent": round(mu_recent_c, 3),
            "drift": round(mu_recent_c - mu_hist_c, 3),
        }

        if drift_c > 0.8 and CORNERS_MODEL.exists():
            mu_blend_c = 0.70 * mu_hist_c + 0.30 * mu_recent_c
            try:
                model_data = json.loads(CORNERS_MODEL.read_text())
                old_mu_c = model_data.get("mu_total", mu_hist_c)
                model_data["mu_total"] = round(mu_blend_c, 4)
                model_data["_drift_corrected"] = date.today().isoformat()
                CORNERS_MODEL.write_text(json.dumps(model_data, ensure_ascii=False, indent=2))
                fixes_applied["corners_mu"] = {
                    "antes": round(old_mu_c, 3),
                    "despues": round(mu_blend_c, 3),
                    "razon": f"drift={mu_recent_c-mu_hist_c:+.2f} > 0.8",
                }
                if verbose:
                    print(f"  [auto-fix] corners mu: {old_mu_c:.3f} → {mu_blend_c:.3f} "
                          f"(drift {mu_recent_c-mu_hist_c:+.2f})")
            except Exception as e:
                if verbose:
                    print(f"  [warn] no se pudo actualizar corners_model.json: {e}")

    # ── Guardar log de drift ──
    drift_entry = {
        "fecha": date.today().isoformat(),
        "drift_detectado": drift_report,
        "fixes_aplicados": fixes_applied,
        "n_partidos_recientes": len(df_recent),
        "n_partidos_hist": len(df_hist),
    }
    history = []
    if DRIFT_LOG.exists():
        try:
            history = json.loads(DRIFT_LOG.read_text())
        except Exception:
            pass
    history.append(drift_entry)
    history = history[-30:]   # guardar solo últimas 30 entradas
    DRIFT_LOG.write_text(json.dumps(history, ensure_ascii=False, indent=2))

    if not fixes_applied and verbose:
        print(f"  [drift] Sin correcciones: tarjetas Δ={mu_recent_t-mu_hist_t:+.2f}, "
              f"dentro del umbral")

    return fixes_applied


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

    # 1. Siempre detectar y corregir drift (no requiere re-entrenamiento completo)
    print(f"\n── Detección automática de drift ──")
    fixes = detect_and_fix_drift(verbose=True)
    if fixes:
        print(f"  ✓ {len(fixes)} correcciones aplicadas automáticamente")

    # 2. Re-entrenar si hay suficientes cambios
    should, reason = should_retrain(force=args.force)
    if should:
        print(f"\n  Re-entrenando: {reason}")
        retrain_all(verbose=True)
    else:
        print(f"\n  Sin re-entrenamiento necesario: {reason}")
        if RETRAIN_LOG.exists():
            log = json.loads(RETRAIN_LOG.read_text())
            print(f"  Último entrenamiento: {log.get('last_train_date')} "
                  f"({log.get('n_partidos_at_train')} partidos)")


if __name__ == "__main__":
    main()
