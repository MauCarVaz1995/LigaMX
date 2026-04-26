#!/usr/bin/env python3
"""
parlay_analyzer.py — Análisis de apuestas combinadas (parlays)
==============================================================
Lee los picks de alta confianza del día y arma combinaciones 2, 3 y 4 selecciones.
Para cada combinación calcula:
  - Probabilidad combinada (producto de probs individuales)
  - Cuota combinada (producto de cuotas)
  - EV del parlay
  - Retorno esperado sobre un monto dado

Cuotas: usa cuota real de The Odds API si está disponible en betting_log.csv,
        si no, usa cuota justa (1/prob) como upper bound.

Uso:
  python bots/parlay_analyzer.py                     # picks del día, $1000 MXN
  python bots/parlay_analyzer.py --monto 500         # con $500 MXN
  python bots/parlay_analyzer.py --min-prob 0.60     # umbral mínimo de prob
  python bots/parlay_analyzer.py --max-legs 3        # solo hasta 3 selecciones
  python bots/parlay_analyzer.py --fecha 2026-04-27  # fecha específica
  python bots/parlay_analyzer.py --backtest          # simula historial completo
"""

import argparse
import json
import sys
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import pandas as pd

BASE    = Path(__file__).resolve().parent.parent
REPORTS = BASE / "output" / "reports"
LOG_CSV = BASE / "data" / "processed" / "betting_log.csv"

# Mercados que incluimos en parlays (key en JSON → descripción)
MERCADOS_PARLAY = {
    "corners":   [("over_8.5",  "Corners +8.5"),
                  ("over_9.5",  "Corners +9.5"),
                  ("over_10.5", "Corners +10.5"),
                  ("under_8.5", "Corners -8.5"),
                  ("under_9.5", "Corners -9.5"),
                  ("under_10.5","Corners -10.5")],
    "tarjetas":  [("over_3.5",  "Tarjetas +3.5"),
                  ("over_4.5",  "Tarjetas +4.5"),
                  ("over_5.5",  "Tarjetas +5.5")],
    "btts":      [("btts_si",   "BTTS Sí"),
                  ("over_1.5",  "Goles +1.5"),
                  ("over_2.5",  "Goles +2.5")],
}

# Nombres de campo en el JSON de betting
MERCADO_FIELD = {
    "corners":  "corners",
    "tarjetas": "tarjetas",
    "btts":     "btts",
}

# En btts el campo "over_1.5" se llama igual, "btts_si" también
BTTS_FIELD_MAP = {
    "btts_si":  "btts_si",
    "over_1.5": "over_1.5",
    "over_2.5": "over_2.5",
}


# ─────────────────────────────────────────────────────────────────────────────
def load_betting_json(fecha_str: str) -> list[dict]:
    """Carga el betting_*.json más cercano a la fecha dada."""
    candidates = sorted(REPORTS.glob("betting_*.json"), reverse=True)
    for path in candidates:
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                # Verifica que contenga partidos de esa fecha o cercanos
                fechas = [p.get("fecha", "") for p in data]
                if any(f >= fecha_str for f in fechas):
                    return data, path.name
        except Exception:
            continue
    return [], ""


def load_real_odds(partido_id: str, mercado: str, linea: str) -> float | None:
    """
    Busca la cuota real en betting_log.csv para un partido/mercado/linea.
    Devuelve None si no hay cuota real disponible.
    """
    if not LOG_CSV.exists():
        return None
    try:
        df = pd.read_csv(LOG_CSV, dtype=str)
        mask = (df["partido_id"].astype(str) == str(partido_id)) & \
               (df["mercado"] == mercado) & \
               (df["linea"].astype(str) == str(linea))
        rows = df[mask]
        if rows.empty:
            return None
        cuota = rows.iloc[-1].get("cuota_vista", "")
        if cuota and cuota not in ("", "nan"):
            return float(cuota)
    except Exception:
        pass
    return None


def extract_picks(partidos: list[dict], min_prob: float) -> list[dict]:
    """
    Extrae todos los picks con prob >= min_prob de la lista de partidos.
    Devuelve lista de dicts con info del pick.
    """
    picks = []
    for p in partidos:
        match_id  = p.get("match_id", "")
        local     = p.get("local", "?")
        visita    = p.get("visita", "?")
        fecha     = p.get("fecha", "?")
        jornada   = p.get("jornada", "?")
        partido_str = f"{local} vs {visita}"

        for cat, mercados in MERCADOS_PARLAY.items():
            cat_data = p.get(MERCADO_FIELD[cat], {})
            if not isinstance(cat_data, dict):
                continue

            for campo, descripcion in mercados:
                prob = cat_data.get(campo)
                if not isinstance(prob, float):
                    continue
                if prob < min_prob:
                    continue
                # Evitar duplicados: si over_8.5 > 0.65 y over_9.5 > 0.65, son picks distintos
                # pero dentro del mismo partido/cat solo tomamos el de mayor EV

                # Cuota: real si disponible, si no justa
                cuota_real = load_real_odds(match_id, cat, campo.replace("over_", "").replace("under_", ""))
                if cuota_real and cuota_real > 1.01:
                    cuota = cuota_real
                    cuota_tipo = "real"
                else:
                    cuota = round(1 / prob, 3)
                    cuota_tipo = "justa"

                ev = prob * cuota - 1

                picks.append({
                    "partido":     partido_str,
                    "fecha":       fecha,
                    "jornada":     jornada,
                    "match_id":    match_id,
                    "cat":         cat,
                    "campo":       campo,
                    "descripcion": descripcion,
                    "prob":        prob,
                    "cuota":       cuota,
                    "cuota_tipo":  cuota_tipo,
                    "ev":          ev,
                    "label":       f"{partido_str} → {descripcion}",
                })

    # Ordenar por EV descendente
    picks.sort(key=lambda x: x["ev"], reverse=True)
    return picks


def build_parlays(picks: list[dict], max_legs: int) -> list[dict]:
    """
    Construye todas las combinaciones de 2..max_legs picks.
    Restricción: máx 1 pick por partido (no combinar 2 mercados del mismo partido
    a menos que sean categorías distintas e independientes — aquí lo restringimos
    a máx 2 picks del mismo partido).
    """
    parlays = []

    for n_legs in range(2, max_legs + 1):
        for combo in combinations(picks, n_legs):
            # Restricción: no más de 2 picks del mismo partido
            partidos_en_combo = [c["partido"] for c in combo]
            if max(partidos_en_combo.count(x) for x in set(partidos_en_combo)) > 2:
                continue
            # Restricción: no mismo mercado del mismo partido
            keys = [(c["partido"], c["cat"], c["campo"]) for c in combo]
            if len(keys) != len(set(keys)):
                continue

            prob_combinada  = 1.0
            cuota_combinada = 1.0
            tiene_cuota_real = all(c["cuota_tipo"] == "real" for c in combo)

            for c in combo:
                prob_combinada  *= c["prob"]
                cuota_combinada *= c["cuota"]

            ev_parlay = prob_combinada * cuota_combinada - 1

            parlays.append({
                "legs":             n_legs,
                "picks":            list(combo),
                "prob_combinada":   round(prob_combinada, 4),
                "cuota_combinada":  round(cuota_combinada, 3),
                "tiene_cuota_real": tiene_cuota_real,
                "ev":               round(ev_parlay, 4),
            })

    # Ordenar por EV descendente
    parlays.sort(key=lambda x: x["ev"], reverse=True)
    return parlays


def print_parlay_report(parlays: list[dict], picks: list[dict],
                        monto: float, min_prob: float, top_n: int = 10):
    hoy = date.today().isoformat()
    print()
    print("=" * 70)
    print(f"  PARLAY ANALYZER — {hoy}")
    print(f"  Picks disponibles: {len(picks)}  |  Umbral: {min_prob:.0%}  |  Monto: ${monto:,.0f} MXN")
    print("=" * 70)

    if not picks:
        print(f"\n  Sin picks con prob ≥ {min_prob:.0%} para hoy.\n")
        return

    # ── Picks individuales ──
    print(f"\n── PICKS INDIVIDUALES (prob ≥ {min_prob:.0%}) ──")
    print(f"  {'Pick':<48} {'Prob':>6} {'Cuota':>7} {'EV':>7}  {'Retorno/monto':>14}")
    print("  " + "─" * 86)
    for pick in picks:
        retorno = monto * pick["cuota"] - monto
        tag = "*" if pick["cuota_tipo"] == "real" else "~"
        ev_str = f"{pick['ev']:+.1%}"
        print(f"  {pick['label']:<48} {pick['prob']:>5.1%} {pick['cuota']:>6.2f}{tag} {ev_str:>7}  "
              f"  +${retorno:>9,.0f}")
    print(f"\n  * cuota real (Pinnacle/Betsson) | ~ cuota justa (1/prob, upper bound)")

    # ── Parlays ──
    if not parlays:
        print("\n  Sin combinaciones con EV positivo.\n")
        return

    ev_pos = [p for p in parlays if p["ev"] > 0]
    print(f"\n── TOP PARLAYS CON EV POSITIVO ({len(ev_pos)} de {len(parlays)} combinaciones) ──\n")

    if not ev_pos:
        print("  Ninguna combinación tiene EV > 0 con las cuotas disponibles.\n")
        # Mostrar top 5 de todas formas
        ev_pos = parlays[:5]
        print("  Top 5 mejores (pueden ser cuota justa):\n")

    shown = 0
    for parlay in ev_pos[:top_n]:
        ganancia_si_gana = monto * parlay["cuota_combinada"] - monto
        ev_mxn = parlay["ev"] * monto
        cuota_tag = "" if parlay["tiene_cuota_real"] else " (cuotas justas ~)"

        print(f"  ── {parlay['legs']} selecciones{cuota_tag} ──")
        for p in parlay["picks"]:
            tag = "*" if p["cuota_tipo"] == "real" else "~"
            print(f"    • {p['label']:<48} {p['prob']:.1%}  @{p['cuota']:.2f}{tag}")
        print(f"    → Cuota combinada: {parlay['cuota_combinada']:.2f}  |  "
              f"Prob combinada: {parlay['prob_combinada']:.1%}  |  EV: {parlay['ev']:+.1%}")
        print(f"    → Invertir ${monto:,.0f} MXN → ganarías ${ganancia_si_gana:,.0f} MXN "
              f"(valor esperado: ${ev_mxn:+,.0f} MXN)")
        print()
        shown += 1

    # ── Resumen por número de legs ──
    print("── RESUMEN POR NÚMERO DE SELECCIONES ──")
    print(f"  {'Legs':>4}  {'Total combos':>13}  {'EV > 0':>7}  {'Mejor EV':>10}  "
          f"{'Mejor retorno/$1000':>20}")
    print("  " + "─" * 60)
    for n in range(2, max(p["legs"] for p in parlays) + 1):
        subset = [p for p in parlays if p["legs"] == n]
        ev_pos_subset = [p for p in subset if p["ev"] > 0]
        if not subset:
            continue
        mejor = subset[0]
        retorno_mejor = 1000 * mejor["cuota_combinada"] - 1000
        print(f"  {n:>4}  {len(subset):>13}  {len(ev_pos_subset):>7}  "
              f"{mejor['ev']:>+9.1%}  ${retorno_mejor:>18,.0f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Backtest: simula cómo habrían ido los parlays en el historial
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(min_prob: float, monto: float, max_legs: int):
    """Lee todas las fechas del betting_log y simula parlays."""
    if not LOG_CSV.exists():
        print("No hay betting_log.csv para backtest.")
        return

    df = pd.read_csv(LOG_CSV)
    # Solo picks resueltos con resultado conocido
    df_res = df.dropna(subset=["acierto"])
    if df_res.empty:
        print("Sin predicciones resueltas en betting_log.csv.")
        return

    fechas = sorted(df_res["fecha_partido"].unique())
    resultados_parlay = []

    print(f"\n── BACKTEST PARLAYS — {len(fechas)} jornadas con resultados ──\n")
    print(f"  {'Fecha':<12} {'Legs':>4} {'Cuota':>7} {'Prob':>7} {'EV':>7} "
          f"{'Resultado':>10} {'P&L':>10}")
    print("  " + "─" * 65)

    total_pnl = 0.0
    total_apuestas = 0

    for fecha in fechas:
        df_fecha = df_res[df_res["fecha_partido"] == fecha]
        # Construir picks desde betting_log
        picks_bf = []
        for _, row in df_fecha.iterrows():
            prob = float(row.get("prob_modelo", 0) or 0)
            if prob < min_prob:
                continue
            cuota_vista = row.get("cuota_vista")
            if cuota_vista and str(cuota_vista) not in ("", "nan"):
                cuota = float(cuota_vista)
                cuota_tipo = "real"
            else:
                cuota = round(1 / prob, 3) if prob > 0 else 2.0
                cuota_tipo = "justa"

            # acierto: True si el modelo acertó
            acierto_val = row.get("acierto", "")
            resultado = "si" if str(acierto_val).lower() in ("true", "1", "si") else "no"

            picks_bf.append({
                "partido":     str(row.get("partido", "?")),
                "fecha":       str(row.get("fecha_partido", "?")),
                "jornada":     row.get("jornada", "?"),
                "match_id":    str(row.get("partido", "")),
                "cat":         str(row.get("mercado", "?")),
                "campo":       str(row.get("linea", "?")),
                "descripcion": f"{row.get('mercado','?')} {row.get('linea','?')}",
                "prob":        prob,
                "cuota":       cuota,
                "cuota_tipo":  cuota_tipo,
                "ev":          prob * cuota - 1,
                "label":       f"{row.get('partido','?')} → {row.get('mercado','?')} {row.get('linea','?')}",
                "resultado":   resultado,
            })

        if len(picks_bf) < 2:
            continue

        parlays_bf = build_parlays(picks_bf, max_legs)
        if not parlays_bf:
            continue

        # Tomar el mejor parlay de cada jornada
        mejor_parlay = parlays_bf[0]
        # Evaluar resultado: todas las piernas deben haber acertado
        gano = all(
            p.get("resultado", "") == "si"
            for p in mejor_parlay["picks"]
        )
        pnl = monto * mejor_parlay["cuota_combinada"] - monto if gano else -monto
        total_pnl += pnl
        total_apuestas += 1

        res_str = "GANÓ ✓" if gano else "PERDIÓ ✗"
        print(f"  {fecha:<12} {mejor_parlay['legs']:>4} "
              f"{mejor_parlay['cuota_combinada']:>7.2f} "
              f"{mejor_parlay['prob_combinada']:>7.1%} "
              f"{mejor_parlay['ev']:>+7.1%} "
              f"{res_str:>10} "
              f"${pnl:>+9,.0f}")
        resultados_parlay.append({"fecha": fecha, "gano": gano, "pnl": pnl,
                                  "legs": mejor_parlay["legs"],
                                  "cuota": mejor_parlay["cuota_combinada"]})

    if total_apuestas == 0:
        print("  Sin jornadas con ≥2 picks resueltos.\n")
        return

    roi = total_pnl / (total_apuestas * monto)
    ganadas = sum(1 for r in resultados_parlay if r["gano"])
    print("  " + "─" * 65)
    print(f"  TOTAL: {total_apuestas} parlays | {ganadas} ganados | "
          f"ROI: {roi:+.1%} | P&L: ${total_pnl:+,.0f} MXN")
    print(f"  (umbral prob ≥ {min_prob:.0%}, monto ${monto:,.0f}/jornada)\n")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Parlay Analyzer")
    parser.add_argument("--monto",    type=float, default=1000.0,
                        help="Monto a invertir por parlay en MXN (default: 1000)")
    parser.add_argument("--min-prob", type=float, default=0.65,
                        help="Probabilidad mínima por selección (default: 0.65)")
    parser.add_argument("--max-legs", type=int,   default=4,
                        help="Máximo número de selecciones por parlay (default: 4)")
    parser.add_argument("--fecha",    type=str,   default=None,
                        help="Fecha objetivo YYYY-MM-DD (default: hoy)")
    parser.add_argument("--backtest", action="store_true",
                        help="Simular parlays sobre historial resuelto")
    parser.add_argument("--top",      type=int,   default=10,
                        help="Mostrar top N parlays (default: 10)")
    args = parser.parse_args()

    fecha_str = args.fecha or date.today().isoformat()

    if args.backtest:
        run_backtest(args.min_prob, args.monto, args.max_legs)
        return

    partidos, fname = load_betting_json(fecha_str)
    if not partidos:
        print(f"No se encontró betting JSON para {fecha_str} en {REPORTS}/")
        sys.exit(1)

    print(f"[fuente] {fname} — {len(partidos)} partidos")

    picks   = extract_picks(partidos, args.min_prob)
    parlays = build_parlays(picks, args.max_legs)
    print_parlay_report(parlays, picks, args.monto, args.min_prob, args.top)


if __name__ == "__main__":
    main()
