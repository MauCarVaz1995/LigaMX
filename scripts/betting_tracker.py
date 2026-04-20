#!/usr/bin/env python3
"""
betting_tracker.py — Tracker de predicciones de mercados de apuestas
=====================================================================
Registra y evalúa predicciones de corners, tarjetas, BTTS y goles
en data/processed/betting_log.csv — una fila por mercado por partido.

Flujo:
  1. --log   → lee el último betting_*.json y agrega filas al CSV
  2. --update → llena resultado_real + acierto desde match_events.csv

CSV: data/processed/betting_log.csv
  fecha_prediccion, fecha_partido, partido, liga, jornada,
  mercado, linea, prob_modelo, cuota_vista, ev_estimado,
  resultado_real (True/False), acierto (True/False), corners_real,
  tarjetas_real, goles_local_real, goles_visita_real

Uso:
  python scripts/betting_tracker.py --log             # loguear predicciones
  python scripts/betting_tracker.py --update          # resolver resultados
  python scripts/betting_tracker.py --log --update    # ambos
  python scripts/betting_tracker.py --stats           # métricas por mercado
"""

import argparse
import json
import os
import tempfile
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE         = Path(__file__).resolve().parent.parent
BETTING_LOG  = BASE / "data/processed/betting_log.csv"
REPORTS_DIR  = BASE / "output/reports"
EVENTS_CSV   = BASE / "data/processed/match_events.csv"

TODAY = date.today().isoformat()

# Mercados a loguear y sus claves en el JSON del betting bot
MERCADOS = [
    # (key_en_json,  sub_key,       mercado_label,       linea)
    ("corners",  "over_8.5",   "corners_over_8.5",   8.5),
    ("corners",  "over_9.5",   "corners_over_9.5",   9.5),
    ("corners",  "over_10.5",  "corners_over_10.5",  10.5),
    ("tarjetas", "over_3.5",   "tarjetas_over_3.5",  3.5),
    ("tarjetas", "over_4.5",   "tarjetas_over_4.5",  4.5),
    ("tarjetas", "over_5.5",   "tarjetas_over_5.5",  5.5),
    ("btts",     "btts_si",    "btts_si",             None),
    ("btts",     "over_2.5",   "goles_over_2.5",      2.5),
    ("btts",     "over_1.5",   "goles_over_1.5",      1.5),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_log() -> pd.DataFrame:
    if BETTING_LOG.exists():
        return pd.read_csv(BETTING_LOG, low_memory=False)
    # Crear vacío con el schema correcto
    return pd.DataFrame(columns=[
        "fecha_prediccion", "fecha_partido", "partido", "liga", "jornada",
        "equipo_local", "equipo_visita", "mercado", "linea", "prob_modelo",
        "cuota_vista", "ev_estimado", "resultado_real", "acierto",
        "corners_real", "tarjetas_real", "goles_local_real", "goles_visita_real",
    ])


def _save_log(df: pd.DataFrame):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(BETTING_LOG.parent), suffix=".tmp")
    try:
        df.to_csv(tmp_path, index=False)
        os.close(tmp_fd)
        os.replace(tmp_path, str(BETTING_LOG))
    except Exception:
        os.close(tmp_fd)
        if Path(tmp_path).exists():
            os.unlink(tmp_path)
        raise


def _norm(s: str) -> str:
    return str(s).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOG — leer betting JSON y registrar predicciones
# ─────────────────────────────────────────────────────────────────────────────
def log_predictions(verbose: bool = True) -> int:
    """
    Lee el último betting_*.json generado por daily_betting_bot.py
    y agrega filas nuevas a betting_log.csv.
    Deduplicación: no duplica (partido, mercado, fecha_partido).
    """
    reports = sorted(REPORTS_DIR.glob("betting_*.json"))
    if not reports:
        if verbose:
            print("  [warn] No hay betting_*.json en output/reports/")
        return 0

    latest = reports[-1]
    if verbose:
        print(f"  Leyendo {latest.name}")

    data = json.loads(latest.read_text())
    partidos = data if isinstance(data, list) else data.get("partidos", [])

    df = _load_log()

    # Key de deduplicación existente
    existing_keys = set(
        zip(df["fecha_partido"].astype(str),
            df["equipo_local"].astype(str).str.strip().str.lower(),
            df["equipo_visita"].astype(str).str.strip().str.lower(),
            df["mercado"].astype(str))
    ) if not df.empty else set()

    nuevas = []
    for p in partidos:
        local    = p.get("local", "")
        visita   = p.get("visita", p.get("visitante", ""))
        fecha_p  = str(p.get("fecha", ""))[:10]
        liga     = p.get("torneo", p.get("liga", "Liga MX"))
        jornada  = p.get("jornada", "")
        partido  = f"{local} vs {visita}"

        for cat, sub_key, mercado_label, linea in MERCADOS:
            prob = p.get(cat, {}).get(sub_key)
            if prob is None:
                continue

            key = (fecha_p, _norm(local), _norm(visita), mercado_label)
            if key in existing_keys:
                continue

            nuevas.append({
                "fecha_prediccion":  TODAY,
                "fecha_partido":     fecha_p,
                "partido":           partido,
                "liga":              liga,
                "jornada":           jornada,
                "equipo_local":      local,
                "equipo_visita":     visita,
                "mercado":           mercado_label,
                "linea":             linea,
                "prob_modelo":       round(float(prob), 4),
                "cuota_vista":       None,
                "ev_estimado":       None,
                "resultado_real":    None,
                "acierto":           None,
                "corners_real":      None,
                "tarjetas_real":     None,
                "goles_local_real":  None,
                "goles_visita_real": None,
            })
            existing_keys.add(key)

    if nuevas:
        df = pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True)
        _save_log(df)
        if verbose:
            print(f"  → {len(nuevas)} nuevas predicciones en betting_log.csv")
    else:
        if verbose:
            print("  → Sin predicciones nuevas (ya logueadas o sin datos)")

    return len(nuevas)


# ─────────────────────────────────────────────────────────────────────────────
# 2. UPDATE — resolver resultados desde match_events.csv
# ─────────────────────────────────────────────────────────────────────────────
def update_results(verbose: bool = True) -> int:
    """
    Para cada fila de betting_log sin resultado_real, busca en match_events.csv
    el partido correspondiente y calcula si el mercado se cumplió.
    """
    if not BETTING_LOG.exists():
        if verbose:
            print("  [warn] betting_log.csv no existe aún")
        return 0
    if not EVENTS_CSV.exists():
        if verbose:
            print("  [warn] match_events.csv no existe")
        return 0

    df_log    = _load_log()
    df_events = pd.read_csv(EVENTS_CSV, low_memory=False)

    # Solo intentar resolver partidos que ya pasaron hace al menos 1 día
    cutoff = (date.today() - timedelta(days=1)).isoformat()
    pendientes = df_log[
        df_log["resultado_real"].isna() &
        (df_log["fecha_partido"].astype(str) <= cutoff)
    ]

    if pendientes.empty:
        if verbose:
            print("  Sin predicciones pendientes de resolver")
        return 0

    actualizados = 0

    for idx, row in pendientes.iterrows():
        local  = _norm(row["equipo_local"])
        visita = _norm(row["equipo_visita"])
        fecha  = str(row["fecha_partido"])[:10]

        # Buscar en match_events (con tolerancia ±1 día)
        ev = None
        for delta in [0, -1, 1, -2]:
            try:
                d = datetime.strptime(fecha, "%Y-%m-%d")
                alt = (d + timedelta(days=delta)).strftime("%Y-%m-%d")
            except ValueError:
                continue

            mask = (
                (df_events["fecha"].astype(str).str[:10] == alt) &
                (df_events["local"].str.strip().str.lower() == local) &
                (df_events["visitante"].str.strip().str.lower() == visita)
            )
            if mask.any():
                ev = df_events[mask].iloc[0]
                break

        if ev is None:
            continue  # partido no encontrado aún

        # Extraer stats reales
        corners_real  = int(ev["corners_total"])  if pd.notna(ev.get("corners_total"))  else None
        amarillas     = int(ev["amarillas_total"]) if pd.notna(ev.get("amarillas_total")) else None
        rojas         = int(ev["rojas_total"])     if pd.notna(ev.get("rojas_total"))     else 0
        tarjetas_real = (amarillas + 2 * rojas)    if amarillas is not None else None
        goles_local   = int(ev["goles_local"])     if pd.notna(ev.get("goles_local"))     else None
        goles_visita  = int(ev["goles_visitante"]) if pd.notna(ev.get("goles_visitante")) else None

        # Calcular resultado para el mercado específico
        mercado = row["mercado"]
        linea   = row["linea"]
        resultado = None

        if mercado.startswith("corners_over_") and corners_real is not None:
            resultado = corners_real > linea

        elif mercado.startswith("tarjetas_over_") and tarjetas_real is not None:
            resultado = tarjetas_real > linea

        elif mercado == "btts_si" and goles_local is not None and goles_visita is not None:
            resultado = (goles_local > 0) and (goles_visita > 0)

        elif mercado.startswith("goles_over_") and goles_local is not None and goles_visita is not None:
            resultado = (goles_local + goles_visita) > linea

        if resultado is None:
            continue  # datos insuficientes para este mercado

        # Actualizar fila
        df_log.at[idx, "resultado_real"]    = bool(resultado)
        df_log.at[idx, "acierto"]           = bool(resultado == (row["prob_modelo"] >= 0.5))
        df_log.at[idx, "corners_real"]      = corners_real
        df_log.at[idx, "tarjetas_real"]     = tarjetas_real
        df_log.at[idx, "goles_local_real"]  = goles_local
        df_log.at[idx, "goles_visita_real"] = goles_visita
        actualizados += 1

    if actualizados > 0:
        _save_log(df_log)
        if verbose:
            print(f"  → {actualizados} predicciones actualizadas con resultado real")
    else:
        if verbose:
            print("  → 0 actualizaciones (sin datos nuevos en match_events.csv)")

    return actualizados


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATS — métricas por mercado
# ─────────────────────────────────────────────────────────────────────────────
def show_stats() -> dict:
    """
    Muestra hit rate, Brier score y ROI hipotético por mercado.
    """
    if not BETTING_LOG.exists():
        print("  betting_log.csv no existe aún — corre --log primero")
        return {}

    df = _load_log()
    df_eval = df[df["resultado_real"].notna()].copy()

    if df_eval.empty:
        print(f"  Total predicciones: {len(df)} | Sin resultados resueltos aún")
        return {}

    df_eval["resultado_real"] = df_eval["resultado_real"].astype(bool)
    df_eval["prob_modelo"]    = df_eval["prob_modelo"].astype(float)

    print(f"\n── Betting Tracker — {len(df)} predicciones ({len(df_eval)} evaluadas) ──\n")
    print(f"{'Mercado':<22} {'N':>4} {'Hit%':>6} {'Brier':>6} {'vs base':>8} {'Skill':>7}")
    print("─" * 60)

    stats = {}
    for mercado, g in df_eval.groupby("mercado"):
        n       = len(g)
        hit     = g["resultado_real"].mean()
        brier   = float(((g["prob_modelo"] - g["resultado_real"].astype(float)) ** 2).mean())
        base_rate = g["resultado_real"].mean()           # baseline: siempre predice la media
        brier_naive = float((base_rate * (1 - base_rate)))  # Brier de predictor constante
        skill   = (brier_naive - brier) / brier_naive if brier_naive > 0 else 0

        # ROI hipotético: si apostaste cuando prob >= 0.60 a cuota justa
        picks_ev = g[g["prob_modelo"] >= 0.60]
        if len(picks_ev) > 0:
            # Cuota justa = 1 / prob_modelo
            roi = float(((picks_ev["prob_modelo"].apply(lambda p: 1/p) *
                          picks_ev["resultado_real"].astype(float)) - 1).mean())
        else:
            roi = None

        skill_str = f"{skill:+.1%}" if n >= 10 else "n<10"
        roi_str   = f"{roi:+.1%}" if roi is not None else "—"
        print(f"{mercado:<22} {n:>4} {hit:>6.1%} {brier:>6.3f} {brier_naive:>8.3f} {skill_str:>7}")

        stats[mercado] = {"n": n, "hit_rate": round(hit, 3), "brier": round(brier, 3),
                          "skill": round(skill, 3), "roi_hipotetico": roi}

    print()

    # Pendientes
    pendientes = df[df["resultado_real"].isna()]
    if not pendientes.empty:
        print(f"⏳ Pendientes de resolver: {len(pendientes)} predicciones")
        prox = pendientes.groupby("mercado")["fecha_partido"].count()
        for m, c in prox.items():
            print(f"   {m}: {c}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tracker de predicciones de mercados de apuestas"
    )
    parser.add_argument("--log",    action="store_true", help="Loguear predicciones del último betting JSON")
    parser.add_argument("--update", action="store_true", help="Resolver resultados desde match_events.csv")
    parser.add_argument("--stats",  action="store_true", help="Mostrar métricas por mercado")
    parser.add_argument("--all",    action="store_true", help="Log + update + stats")
    args = parser.parse_args()

    if args.all:
        args.log = args.update = args.stats = True

    if not any([args.log, args.update, args.stats]):
        # Por defecto: todo
        args.log = args.update = args.stats = True

    if args.log:
        print("\n── Logueando predicciones ──")
        log_predictions(verbose=True)

    if args.update:
        print("\n── Actualizando resultados ──")
        update_results(verbose=True)

    if args.stats:
        show_stats()


if __name__ == "__main__":
    main()
