#!/usr/bin/env python3
"""
modelo_btts.py — Modelo BTTS (Both Teams To Score) para Liga MX
================================================================
Usa los λ del modelo ELO+Poisson existente para calcular P(BTTS).

  P(local anota)  = 1 - e^(-λ_local)
  P(visita anota) = 1 - e^(-λ_visita)
  P(BTTS = Sí)    = P(local anota) × P(visita anota)

También predice Over/Under 2.5 goles y BTTS+Over 2.5.

Uso:
  from scripts.modelo_btts import predecir_btts
  result = predecir_btts("Cruz Azul", "América", elo_local=1700, elo_visita=1750)

  # O desde CLI:
  python scripts/modelo_btts.py --local "Cruz Azul" --visita "América"
  python scripts/modelo_btts.py --local "Toluca" --visita "Monterrey" --cuota-btts 1.85
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent

# Parámetros del modelo ELO+Poisson (deben coincidir con 15_prediccion_elo_poisson.py)
HOME_ADV     = 100    # ELO bonus para local
K_SCALE      = 400    # denominador logístico
MU_GOALS     = 1.421  # media global de goles (del modelo Poisson Liga MX)
ELO_CSV      = BASE / "data/processed/elo_historico.csv"


# ─────────────────────────────────────────────────────────────────────────────
def _get_latest_elos(local: str, visita: str) -> tuple[float, float]:
    """Obtiene los ELOs más recientes de ambos equipos."""
    try:
        df = pd.read_csv(ELO_CSV)
        df = df.sort_values("fecha")

        elo_l = df[df["equipo"] == local]["elo"].iloc[-1]  if local  in df["equipo"].values else 1500.0
        elo_v = df[df["equipo"] == visita]["elo"].iloc[-1] if visita in df["equipo"].values else 1500.0
        return float(elo_l), float(elo_v)
    except Exception:
        return 1500.0, 1500.0


def _lambdas_from_elo(elo_local: float, elo_visita: float) -> tuple[float, float]:
    """
    Convierte ELOs a λ_local y λ_visita usando fórmula aditiva.

    Fórmula: diferencia de ELO determina la diferencia esperada de goles.
    Calibración: 300 ELO points ≈ 1 gol de diferencia esperado.
    Preserva el total de goles (MU_TOTAL=2.858) como suma constante.

    Ejemplos con HOME_ADV=100:
      ELOs iguales:              lam_l=1.595, lam_v=1.263  → modo 1-1
      ELO diff=200 (home fav):   lam_l=1.928, lam_v=0.930  → modo 1-0
      ELO diff=-200 (away fav):  lam_l=1.262, lam_v=1.596  → modo 1-1 o 0-1
    """
    MU_TOTAL = 1.636 + 1.222     # 2.858 — total goles esperado (Liga MX histórico)
    elo_diff  = (elo_local + HOME_ADV) - elo_visita
    goal_diff = elo_diff / 300.0  # calibrado: 300 ELO ≈ 1 gol de diferencia
    lam_l = (MU_TOTAL + goal_diff) / 2.0
    lam_v = (MU_TOTAL - goal_diff) / 2.0
    return max(lam_l, 0.20), max(lam_v, 0.20)


def _dixon_coles_rho(lam_l: float, lam_v: float, rho: float = -0.22) -> np.ndarray:
    """Matriz de probabilidades Dixon-Coles hasta max_g×max_g."""
    max_g = 8
    tau = np.ones((max_g, max_g))
    tau[0, 0] = 1 - lam_l * lam_v * rho
    tau[0, 1] = 1 + lam_l * rho
    tau[1, 0] = 1 + lam_v * rho
    tau[1, 1] = 1 - rho

    probs = np.zeros((max_g, max_g))
    for i in range(max_g):
        for j in range(max_g):
            probs[i, j] = tau[i, j] * poisson.pmf(i, lam_l) * poisson.pmf(j, lam_v)
    probs /= probs.sum()
    return probs


# ─────────────────────────────────────────────────────────────────────────────
def predecir_btts(local: str, visita: str,
                  elo_local: float = None, elo_visita: float = None,
                  cuota_btts_si: float = None, cuota_btts_no: float = None,
                  cuota_over25: float = None, cuota_under25: float = None,
                  rho: float = -0.22) -> dict:
    """
    Predice BTTS y mercados relacionados.

    Args:
        local, visita: nombres de equipo
        elo_local, elo_visita: ELOs manuales (si None, los lee del CSV)
        cuota_btts_si/no, cuota_over25/under25: cuotas de casa (para EV)
        rho: parámetro Dixon-Coles

    Returns:
        dict con probabilidades, lambdas y EV si hay cuotas
    """
    if elo_local is None or elo_visita is None:
        elo_l, elo_v = _get_latest_elos(local, visita)
        elo_local  = elo_local  or elo_l
        elo_visita = elo_visita or elo_v

    lam_l, lam_v = _lambdas_from_elo(elo_local, elo_visita)
    probs = _dixon_coles_rho(lam_l, lam_v, rho)

    # BTTS = P(local ≥ 1) × P(visita ≥ 1) — ajustado con DC
    # En la matriz probs[i,j]: BTTS = 1 - P(i=0,*) - P(*,j=0) + P(0,0)
    p_local_0  = probs[0, :].sum()   # local no anota
    p_visita_0 = probs[:, 0].sum()   # visita no anota
    p_00       = probs[0, 0]
    p_btts_si  = 1 - p_local_0 - p_visita_0 + p_00
    p_btts_no  = 1 - p_btts_si

    # Over/Under goles
    p_over25  = float(np.sum(probs[i, j] for i in range(8) for j in range(8) if i + j > 2))
    p_under25 = 1 - p_over25
    p_over15  = float(np.sum(probs[i, j] for i in range(8) for j in range(8) if i + j > 1))

    # BTTS + Over 2.5 (mercado combinado popular)
    p_btts_over25 = float(np.sum(probs[i, j] for i in range(8) for j in range(8)
                                 if i > 0 and j > 0 and i + j > 2))

    result = {
        "local":           local,
        "visita":          visita,
        "elo_local":       round(elo_local, 0),
        "elo_visita":      round(elo_visita, 0),
        "lambda_local":    round(lam_l, 3),
        "lambda_visita":   round(lam_v, 3),
        "p_btts_si":       round(p_btts_si,  3),
        "p_btts_no":       round(p_btts_no,  3),
        "p_over_2.5":      round(p_over25,   3),
        "p_under_2.5":     round(p_under25,  3),
        "p_over_1.5":      round(p_over15,   3),
        "p_btts_over_2.5": round(p_btts_over25, 3),
    }

    def add_ev(prob, cuota, label):
        if cuota:
            ev = round(prob * cuota - 1, 4)
            result[f"ev_{label}"] = ev
            result[f"apuesta_{label}"] = "✅ VALUE" if ev > 0.05 else "❌ sin valor"

    add_ev(p_btts_si,  cuota_btts_si,  "btts_si")
    add_ev(p_btts_no,  cuota_btts_no,  "btts_no")
    add_ev(p_over25,   cuota_over25,   "over25")
    add_ev(p_under25,  cuota_under25,  "under25")

    return result


# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local",        default="Cruz Azul")
    parser.add_argument("--visita",       default="América")
    parser.add_argument("--elo-local",    type=float)
    parser.add_argument("--elo-visita",   type=float)
    parser.add_argument("--cuota-btts",   type=float, help="Cuota para BTTS Sí")
    parser.add_argument("--cuota-no-btts",type=float, help="Cuota para BTTS No")
    parser.add_argument("--cuota-over25", type=float)
    parser.add_argument("--cuota-under25",type=float)
    args = parser.parse_args()

    r = predecir_btts(
        args.local, args.visita,
        elo_local=args.elo_local, elo_visita=args.elo_visita,
        cuota_btts_si=args.cuota_btts, cuota_btts_no=args.cuota_no_btts,
        cuota_over25=args.cuota_over25, cuota_under25=args.cuota_under25,
    )

    print(f"\n── BTTS: {r['local']} vs {r['visita']} ──")
    print(f"  ELO  {r['elo_local']} vs {r['elo_visita']}")
    print(f"  λ    local={r['lambda_local']} | visita={r['lambda_visita']}")
    print(f"  BTTS Sí:  {r['p_btts_si']:.1%}  |  No: {r['p_btts_no']:.1%}")
    print(f"  Over 1.5: {r['p_over_1.5']:.1%}")
    print(f"  Over 2.5: {r['p_over_2.5']:.1%}  |  Under 2.5: {r['p_under_2.5']:.1%}")
    print(f"  BTTS + Over 2.5: {r['p_btts_over_2.5']:.1%}")

    for k in ["ev_btts_si", "ev_btts_no", "ev_over25", "ev_under25"]:
        if k in r:
            label = k.replace("ev_", "")
            print(f"  EV {label}: {r[k]:+.1%}  {r.get(f'apuesta_{label}','')}")


if __name__ == "__main__":
    main()
