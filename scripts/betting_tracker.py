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
    Lee el último betting_*.json (Liga MX) y el último betting_intl_*.json
    (ligas internacionales) y agrega filas nuevas a betting_log.csv.
    Para ligas internacionales: solo btts_si, goles_over_1.5, goles_over_2.5.
    Deduplicación por (fecha_partido, equipo_local, equipo_visita, mercado).
    """
    # Buscar archivos: Liga MX (betting_YYYY-MM-DD*) y Intl (betting_intl_*)
    mx_reports   = [f for f in sorted(REPORTS_DIR.glob("betting_*.json"))
                    if not f.name.startswith("betting_intl_")]
    intl_reports = sorted(REPORTS_DIR.glob("betting_intl_*.json"))

    sources = []
    if mx_reports:
        sources.append(("mx", mx_reports[-1]))
    if intl_reports:
        sources.append(("intl", intl_reports[-1]))

    if not sources:
        if verbose:
            print("  [warn] No hay betting_*.json en output/reports/")
        return 0

    total_nuevas = 0
    for source_type, report_file in sources:
        if verbose:
            print(f"  Leyendo {report_file.name} [{source_type}]")
        data = json.loads(report_file.read_text())
        partidos = data if isinstance(data, list) else data.get("partidos", [])
        total_nuevas += _log_from_partidos(partidos, source_type, verbose)

    return total_nuevas


def _log_from_partidos(partidos: list, source_type: str, verbose: bool) -> int:
    """Agrega predicciones al betting_log desde una lista de partidos."""
    # Para intl: solo btts/goles (sin corners ni tarjetas)
    MERCADOS_INTL = [
        ("btts",  "btts_si",   "btts_si",        None),
        ("btts",  "over_2.5",  "goles_over_2.5",  2.5),
        ("btts",  "over_1.5",  "goles_over_1.5",  1.5),
    ]
    mercados_a_usar = MERCADOS_INTL if source_type == "intl" else MERCADOS

    if not partidos:
        return 0

    df = _load_log()

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

        for cat, sub_key, mercado_label, linea in mercados_a_usar:
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
            print(f"  → {len(nuevas)} nuevas predicciones [{source_type}]")
    else:
        if verbose:
            print(f"  → Sin predicciones nuevas [{source_type}]")

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
    """Muestra hit rate, Brier score y skill por mercado."""
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
        n           = len(g)
        hit         = g["resultado_real"].mean()
        brier       = float(((g["prob_modelo"] - g["resultado_real"].astype(float)) ** 2).mean())
        base_rate   = g["resultado_real"].mean()
        brier_naive = float(base_rate * (1 - base_rate))
        skill       = (brier_naive - brier) / brier_naive if brier_naive > 0 else 0

        skill_str = f"{skill:+.1%}" if n >= 10 else "n<10"
        print(f"{mercado:<22} {n:>4} {hit:>6.1%} {brier:>6.3f} {brier_naive:>8.3f} {skill_str:>7}")
        stats[mercado] = {"n": n, "hit_rate": round(hit, 3), "brier": round(brier, 3), "skill": round(skill, 3)}

    print()

    # Pendientes
    pendientes = df[df["resultado_real"].isna()]
    if not pendientes.empty:
        print(f"⏳ Pendientes de resolver: {len(pendientes)} predicciones")
        for m, c in pendientes.groupby("mercado")["fecha_partido"].count().items():
            print(f"   {m}: {c}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. P&L — ganancia/pérdida simulada
# ─────────────────────────────────────────────────────────────────────────────
def show_pnl(unidad: float = 100.0) -> dict:
    """
    Muestra P&L simulado por mercado en dos modalidades:
      - P&L Real:     usa cuota_vista cuando está disponible
      - P&L Justo:    usa cuota teórica = 1/prob_modelo (sin margen de casa)
    Solo incluye picks donde prob_modelo >= 0.55 (margen mínimo para apostar).
    Unidad = monto por apuesta en MXN (default 100).
    """
    if not BETTING_LOG.exists():
        print("  betting_log.csv no existe aún")
        return {}

    df = _load_log()
    df_eval = df[df["resultado_real"].notna()].copy()

    if df_eval.empty:
        print(f"  Sin resultados resueltos aún ({len(df)} predicciones pendientes)")
        return {}

    df_eval["resultado_real"] = df_eval["resultado_real"].astype(bool)
    df_eval["prob_modelo"]    = df_eval["prob_modelo"].astype(float)

    # Umbral: solo apostar cuando prob >= 0.55
    UMBRAL = 0.55
    picks = df_eval[df_eval["prob_modelo"] >= UMBRAL].copy()

    print(f"\n── P&L Simulado — picks con prob ≥ {UMBRAL:.0%} (unidad = ${unidad:.0f} MXN) ──\n")
    print(f"{'Mercado':<22} {'N':>4} {'Gan':>4} {'Per':>4}  {'P&L Real':>10}  {'P&L Justo':>10}  {'ROI Justo':>10}")
    print("─" * 75)

    totals = {"n": 0, "gan": 0, "per": 0, "pnl_real": 0.0, "pnl_justo": 0.0}
    resultado = {}

    for mercado, g in picks.groupby("mercado"):
        n   = len(g)
        gan = int(g["resultado_real"].sum())
        per = n - gan

        # P&L real: usa cuota_vista si existe, si no usa cuota justa
        pnl_real = 0.0
        for _, row in g.iterrows():
            cuota = row.get("cuota_vista")
            won   = bool(row["resultado_real"])
            if pd.notna(cuota) and float(cuota) > 1.0:
                c = float(cuota)
            else:
                c = 1.0 / float(row["prob_modelo"])   # cuota justa si no hay real
            pnl_real += (c - 1) * unidad if won else -unidad

        # P&L justo: siempre usa 1/prob (sin margen, upper bound teórico)
        pnl_justo = 0.0
        for _, row in g.iterrows():
            c   = 1.0 / float(row["prob_modelo"])
            won = bool(row["resultado_real"])
            pnl_justo += (c - 1) * unidad if won else -unidad

        roi_justo = pnl_justo / (n * unidad) if n > 0 else 0

        pnl_real_str  = f"${pnl_real:+.0f}"
        pnl_justo_str = f"${pnl_justo:+.0f}"
        roi_str       = f"{roi_justo:+.1%}"
        print(f"{mercado:<22} {n:>4} {gan:>4} {per:>4}  {pnl_real_str:>10}  {pnl_justo_str:>10}  {roi_str:>10}")

        totals["n"]         += n
        totals["gan"]       += gan
        totals["per"]       += per
        totals["pnl_real"]  += pnl_real
        totals["pnl_justo"] += pnl_justo
        resultado[mercado]   = {"n": n, "ganadas": gan, "pnl_real": round(pnl_real, 2),
                                 "pnl_justo": round(pnl_justo, 2), "roi_justo": round(roi_justo, 4)}

    print("─" * 75)
    roi_total = totals["pnl_justo"] / (totals["n"] * unidad) if totals["n"] > 0 else 0
    print(f"{'TOTAL':<22} {totals['n']:>4} {totals['gan']:>4} {totals['per']:>4}  "
          f"${totals['pnl_real']:>+9.0f}  ${totals['pnl_justo']:>+9.0f}  {roi_total:>+10.1%}")

    n_con_cuota = picks["cuota_vista"].notna().sum()
    print(f"\n  * P&L Real usa cuota_vista en {n_con_cuota}/{totals['n']} picks; resto usa cuota justa (1/prob)")
    print(f"  * P&L Justo = hipotético sin margen de casa (upper bound)")
    print(f"  * Umbral: prob ≥ {UMBRAL:.0%} | Unidad: ${unidad:.0f} MXN por apuesta")

    # Value bets reales con EV > 0
    vb = df_eval[df_eval["ev_estimado"].notna() & (df_eval["ev_estimado"] > 0)]
    if not vb.empty:
        print(f"\n  💰 Value bets con cuota real y EV > 0%:")
        for _, row in vb.iterrows():
            won = bool(row["resultado_real"]) if pd.notna(row["resultado_real"]) else None
            pnl_str = ""
            if won is not None:
                c = float(row["cuota_vista"])
                pnl = (c - 1) * unidad if won else -unidad
                pnl_str = f"→ {'GANÓ' if won else 'PERDIÓ'} ${pnl:+.0f}"
            print(f"    {row['fecha_partido']} {row['partido']} | {row['mercado']} "
                  f"@ {row['cuota_vista']:.2f} EV={float(row['ev_estimado']):.1%} {pnl_str}")

    print()
    return resultado


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
    parser.add_argument("--pnl",    action="store_true", help="Mostrar P&L simulado por mercado")
    parser.add_argument("--unidad", type=float, default=100.0, help="Monto por apuesta en MXN (default 100)")
    parser.add_argument("--all",    action="store_true", help="Log + update + stats + pnl")
    args = parser.parse_args()

    if args.all:
        args.log = args.update = args.stats = args.pnl = True

    if not any([args.log, args.update, args.stats, args.pnl]):
        args.stats = True   # Por defecto: solo stats

    if args.log:
        print("\n── Logueando predicciones ──")
        log_predictions(verbose=True)

    if args.update:
        print("\n── Actualizando resultados ──")
        update_results(verbose=True)

    if args.stats:
        show_stats()

    if args.pnl:
        show_pnl(unidad=args.unidad)


if __name__ == "__main__":
    main()
