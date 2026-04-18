#!/usr/bin/env python3
"""
calcular_ev.py — Calculadora unificada de EV para value betting
===============================================================
Combina los modelos de corners, tarjetas y BTTS para un partido dado.
Imprime todas las probabilidades y, si se dan cuotas, los EVs.

Uso:
  python scripts/calcular_ev.py --local "Cruz Azul" --visita "América"
  python scripts/calcular_ev.py --local "Toluca" --visita "Monterrey" \\
      --corners-over 1.90 --corners-linea 9.5 \\
      --tarjetas-over 1.85 --tarjetas-linea 4.5 \\
      --btts-si 1.80

Output: tabla con todas las apuestas disponibles y su EV.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modelo_corners   import predecir_corners
from modelo_tarjetas  import predecir_tarjetas
from modelo_btts      import predecir_btts


def print_section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def main():
    parser = argparse.ArgumentParser(description="Calculadora de EV para value betting")
    parser.add_argument("--local",  required=True)
    parser.add_argument("--visita", required=True)

    # Corners
    parser.add_argument("--corners-linea",  type=float, default=9.5)
    parser.add_argument("--corners-over",   type=float, help="Cuota Over corners")
    parser.add_argument("--corners-under",  type=float, help="Cuota Under corners")

    # Tarjetas
    parser.add_argument("--tarjetas-linea",  type=float, default=4.5)
    parser.add_argument("--tarjetas-over",   type=float, help="Cuota Over tarjetas")
    parser.add_argument("--tarjetas-under",  type=float, help="Cuota Under tarjetas")

    # BTTS
    parser.add_argument("--btts-si",    type=float, help="Cuota BTTS Sí")
    parser.add_argument("--btts-no",    type=float, help="Cuota BTTS No")
    parser.add_argument("--over25",     type=float, help="Cuota Over 2.5 goles")
    parser.add_argument("--under25",    type=float, help="Cuota Under 2.5 goles")

    # ELOs manuales (opcionales)
    parser.add_argument("--elo-local",  type=float)
    parser.add_argument("--elo-visita", type=float)

    args = parser.parse_args()

    print(f"\n{'═'*50}")
    print(f"  VALUE BETS: {args.local} vs {args.visita}")
    print(f"{'═'*50}")

    # ── Corners ─────────────────────────────────────────────────────────────
    print_section(f"CORNERS (línea {args.corners_linea})")
    try:
        c = predecir_corners(args.local, args.visita,
                             cuota_over=args.corners_over,
                             cuota_under=args.corners_under,
                             linea=args.corners_linea)
        print(f"  λ local={c['lambda_local']} | λ visita={c['lambda_visita']} | λ total={c['lambda_total']}")
        print(f"  Over  7.5: {c['prob_over_7.5']:.1%}")
        print(f"  Over  8.5: {c['prob_over_8.5']:.1%}  |  Under 8.5: {c['prob_under_8.5']:.1%}")
        print(f"  Over  9.5: {c['prob_over_9.5']:.1%}  |  Under 9.5: {c['prob_under_9.5']:.1%}")
        print(f"  Over 10.5: {c['prob_over_10.5']:.1%}")
        linea = args.corners_linea
        if f"ev_over_{linea}" in c:
            ev = c[f"ev_over_{linea}"]
            tag = "✅ VALUE" if ev > 0.05 else ("⚠️ borderline" if ev > 0 else "❌")
            print(f"\n  → Over  {linea} @ {args.corners_over}: EV {ev:+.1%}  {tag}")
        if f"ev_under_{linea}" in c:
            ev = c[f"ev_under_{linea}"]
            tag = "✅ VALUE" if ev > 0.05 else ("⚠️ borderline" if ev > 0 else "❌")
            print(f"  → Under {linea} @ {args.corners_under}: EV {ev:+.1%}  {tag}")
    except Exception as e:
        print(f"  ERROR corners: {e}")

    # ── Tarjetas ─────────────────────────────────────────────────────────────
    print_section(f"TARJETAS (línea {args.tarjetas_linea})")
    try:
        t = predecir_tarjetas(args.local, args.visita,
                              cuota_over=args.tarjetas_over,
                              cuota_under=args.tarjetas_under,
                              linea=args.tarjetas_linea)
        print(f"  λ total={t['lambda_total']} (local {t['cr_local']} + visita {t['cr_visita']} + rival {t['rival_bonus']})")
        print(f"  Over 3.5: {t['prob_over_3.5']:.1%}")
        print(f"  Over 4.5: {t['prob_over_4.5']:.1%}  |  Under 4.5: {t['prob_under_4.5']:.1%}")
        print(f"  Over 5.5: {t['prob_over_5.5']:.1%}  |  Under 5.5: {t['prob_under_5.5']:.1%}")
        linea = args.tarjetas_linea
        if f"ev_over_{linea}" in t:
            ev = t[f"ev_over_{linea}"]
            tag = "✅ VALUE" if ev > 0.05 else ("⚠️ borderline" if ev > 0 else "❌")
            print(f"\n  → Over  {linea} @ {args.tarjetas_over}: EV {ev:+.1%}  {tag}")
        if f"ev_under_{linea}" in t:
            ev = t[f"ev_under_{linea}"]
            tag = "✅ VALUE" if ev > 0.05 else ("⚠️ borderline" if ev > 0 else "❌")
            print(f"  → Under {linea} @ {args.tarjetas_under}: EV {ev:+.1%}  {tag}")
    except Exception as e:
        print(f"  ERROR tarjetas: {e}")

    # ── BTTS & Goles ─────────────────────────────────────────────────────────
    print_section("BTTS & GOLES")
    try:
        b = predecir_btts(args.local, args.visita,
                          elo_local=args.elo_local, elo_visita=args.elo_visita,
                          cuota_btts_si=args.btts_si, cuota_btts_no=args.btts_no,
                          cuota_over25=args.over25, cuota_under25=args.under25)
        print(f"  ELO {b['elo_local']} vs {b['elo_visita']}  |  λ {b['lambda_local']} vs {b['lambda_visita']}")
        print(f"  BTTS Sí: {b['p_btts_si']:.1%}  |  No: {b['p_btts_no']:.1%}")
        print(f"  Over 1.5: {b['p_over_1.5']:.1%}")
        print(f"  Over 2.5: {b['p_over_2.5']:.1%}  |  Under 2.5: {b['p_under_2.5']:.1%}")
        print(f"  BTTS + Over 2.5: {b['p_btts_over_2.5']:.1%}")

        for ev_key, cuota, label in [
            ("ev_btts_si",  args.btts_si,  "BTTS Sí"),
            ("ev_btts_no",  args.btts_no,  "BTTS No"),
            ("ev_over25",   args.over25,   "Over 2.5"),
            ("ev_under25",  args.under25,  "Under 2.5"),
        ]:
            if ev_key in b:
                ev  = b[ev_key]
                tag = "✅ VALUE" if ev > 0.05 else ("⚠️ borderline" if ev > 0 else "❌")
                print(f"\n  → {label} @ {cuota}: EV {ev:+.1%}  {tag}")
    except Exception as e:
        print(f"  ERROR BTTS: {e}")

    print(f"\n{'═'*50}\n")


if __name__ == "__main__":
    main()
