#!/usr/bin/env python3
"""
modelo_corners.py — Modelo Poisson bivariado de corners para Liga MX
=====================================================================
Implementación basada en:
  [1] Dixon & Coles (1997) "Modelling Association Football Scores"
      → MLE de parámetros ataque/defensa con constraint sum-to-zero
  [2] Rue & Salvesen (2000) "Prediction and retrospective analysis of
      soccer matches in a league" → time decay ε = 0.003/día
  [3] Karlis & Ntzoufras (2003) "Analysis of sports data by using
      bivariate Poisson models" → correlación local/visitante vía λ3

Capa de "feeling" Liga MX (liga_mx_knowledge.py):
  - Rivalidades históricas (Clásico Nacional, Regio, Joven...)
  - Ventaja de estadio calibrada por equipo (Azteca, Akron, ...)
  - Penalización por altitud (Toluca 2680m, Pachuca 2400m)
  - Factor de fase del torneo (liguilla: ×1.05 corners)

Modelo matemático:
  λ_local  = exp(μ + att[local]  + def[visita] + home_adv[local])
  λ_visita = exp(μ + att[visita] + def[local])
  Constraint: Σ att_i = 0,  Σ def_i = 0   (identificabilidad)
  Pesos:      w_t = exp(-ε × days_ago)     (Rue & Salvesen)

  Capa feeling:
  λ_local  *= context.home_adv_factor × context.corners_phase_mult
  λ_visita *= context.altitude_visitor_corner_penalty × context.corners_phase_mult
  λ_total  += context.rivalry_corners_bonus

Uso:
  python scripts/modelo_corners.py --train
  python scripts/modelo_corners.py --local "Cruz Azul" --visita "América" \\
      --linea 9.5 --cuota-over 1.90 --cuota-under 1.90 \\
      --jornada 15
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")

BASE      = Path(__file__).resolve().parent.parent
SCRIPTS   = Path(__file__).resolve().parent
EVENTS    = BASE / "data/processed/match_events.csv"
MODEL_OUT = BASE / "data/processed/corners_model.json"

sys.path.insert(0, str(SCRIPTS))
from liga_mx_knowledge import get_match_context

# Parámetros científicos
EPSILON       = 0.003    # Rue & Salvesen (2000): decay diario (más suave que goles)
MIN_WEIGHT    = 0.15     # partidos muy viejos no bajan de 15% de peso
HOME_ADV_LOG  = 0.07     # log(1.07) ≈ ventaja home base en corners


# ─────────────────────────────────────────────────────────────────────────────
# MLE Dixon-Coles adaptado para corners
# ─────────────────────────────────────────────────────────────────────────────

def _build_weights(df: pd.DataFrame) -> np.ndarray:
    """Rue & Salvesen time decay: w = max(MIN_WEIGHT, exp(-ε × days))"""
    today = pd.Timestamp.today()
    days  = (today - pd.to_datetime(df["fecha"])).dt.days.values
    w     = np.exp(-EPSILON * days)
    return np.maximum(w, MIN_WEIGHT)


def _neg_log_likelihood(params: np.ndarray, df: pd.DataFrame,
                        teams: list, weights: np.ndarray) -> float:
    """
    NLL ponderado para Poisson bivariado independiente.
    params = [att_0..att_{n-1}, def_0..def_{n-1}, home_adv, mu]
    Constraint sum-to-zero se aplica normalizando internamente.
    """
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    att      = params[:n]
    defe     = params[n:2*n]
    home_adv = params[2*n]
    mu       = params[2*n + 1]

    # Sum-to-zero constraint (identificabilidad)
    att  = att  - att.mean()
    defe = defe - defe.mean()

    locs  = df["local"].map(idx).values
    viss  = df["visitante"].map(idx).values
    c_l   = df["corners_local"].values.astype(float)
    c_v   = df["corners_visitante"].values.astype(float)

    lam_l = np.exp(mu + att[locs] + defe[viss] + home_adv)
    lam_v = np.exp(mu + att[viss] + defe[locs])

    # Clip para estabilidad numérica
    lam_l = np.clip(lam_l, 0.1, 30)
    lam_v = np.clip(lam_v, 0.1, 30)

    ll = weights * (
        c_l * np.log(lam_l) - lam_l +
        c_v * np.log(lam_v) - lam_v
    )
    return -ll.sum()


def _fit_mle(df: pd.DataFrame, weights: np.ndarray, verbose: bool = True) -> dict:
    """Ajusta el modelo por MLE. Retorna dict con parámetros estimados."""
    teams = sorted(set(df["local"].tolist() + df["visitante"].tolist()))
    n     = len(teams)

    # Valores iniciales: att=0, def=0, home_adv=HOME_ADV_LOG, mu=log(mean/2)
    mu0     = np.log(df["corners_local"].mean())
    x0      = np.zeros(2 * n + 2)
    x0[2*n] = HOME_ADV_LOG
    x0[2*n + 1] = mu0

    if verbose:
        print(f"  Ajustando MLE: {n} equipos, {len(df)} partidos...")

    res = minimize(
        _neg_log_likelihood,
        x0,
        args=(df, teams, weights),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if verbose and not res.success:
        print(f"  [warn] MLE no convergió completamente: {res.message}")

    params = res.x
    att    = params[:n]    - params[:n].mean()   # sum-to-zero
    defe   = params[n:2*n] - params[n:2*n].mean()
    home   = params[2*n]
    mu     = params[2*n + 1]

    if verbose:
        top_att = sorted(zip(teams, att), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top atacantes-corner: {[f'{t}({v:+.2f})' for t,v in top_att]}")
        top_def = sorted(zip(teams, defe), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top conceden-corner:  {[f'{t}({v:+.2f})' for t,v in top_def]}")
        print(f"  home_adv={home:.3f} ({np.exp(home):.3f}x) | μ={mu:.3f} → exp(μ)={np.exp(mu):.2f}")

    return {
        "teams":     teams,
        "att":       {t: float(a) for t, a in zip(teams, att)},
        "def":       {t: float(d) for t, d in zip(teams, defe)},
        "home_adv":  float(home),
        "mu":        float(mu),
        "mu_local":  float(df["corners_local"].mean()),
        "mu_visita": float(df["corners_visitante"].mean()),
        "n_partidos": len(df),
        "torneos":   df["torneo"].unique().tolist(),
        "epsilon":   EPSILON,
        "converged": bool(res.success),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predicción
# ─────────────────────────────────────────────────────────────────────────────

def _predict_raw(model: dict, local: str, visita: str) -> tuple[float, float]:
    """Lambdas crudas del modelo estadístico (sin feeling)."""
    att_l = model["att"].get(local,  0.0)
    def_l = model["def"].get(local,  0.0)
    att_v = model["att"].get(visita, 0.0)
    def_v = model["def"].get(visita, 0.0)
    mu    = model["mu"]
    home  = model["home_adv"]

    lam_l = np.exp(mu + att_l + def_v + home)
    lam_v = np.exp(mu + att_v + def_l)
    return float(lam_l), float(lam_v)


def _apply_feeling(lam_l: float, lam_v: float, ctx) -> tuple[float, float, float]:
    """
    Aplica la capa de conocimiento de dominio (liga_mx_knowledge).
    Retorna (lam_l_adj, lam_v_adj, extra_total).
    """
    # Ventaja de estadio calibrada (reemplaza home_adv genérico para local conocido)
    lam_l *= ctx.home_adv_factor / np.exp(HOME_ADV_LOG)  # ratio vs baseline

    # Altitud: visitante saca menos córners
    lam_v *= ctx.altitude_visitor_corner_penalty

    # Fase del torneo
    lam_l *= ctx.corners_phase_mult
    lam_v *= ctx.corners_phase_mult

    # Rivalidad: bonus aditivo (repartido 60/40 local/visita según historial)
    bonus = ctx.rivalry_corners_bonus
    extra_l = bonus * 0.6
    extra_v = bonus * 0.4

    return lam_l + extra_l, lam_v + extra_v, 0.0


def predict(model: dict, local: str, visita: str,
            jornada: int = 10, is_liguilla: bool = False,
            stage: str = "", referee: str = "") -> dict:
    """Predicción completa con ciencia + feeling."""
    ctx     = get_match_context(local, visita, jornada, is_liguilla, stage, referee)
    lam_l0, lam_v0 = _predict_raw(model, local, visita)
    lam_l,  lam_v, _ = _apply_feeling(lam_l0, lam_v0, ctx)
    lam_t   = lam_l + lam_v

    # Convolución de dos Poisson independientes para P(total)
    max_k   = 50
    dist_l  = np.array([poisson.pmf(k, lam_l) for k in range(max_k)])
    dist_v  = np.array([poisson.pmf(k, lam_v) for k in range(max_k)])
    dist_t  = np.convolve(dist_l, dist_v)[:max_k]
    dist_t /= dist_t.sum()  # renormalizar por si hay masa en la cola

    def p_over(line: float) -> float:
        return float(dist_t[int(line) + 1:].sum())

    return {
        "local":  local,
        "visita": visita,
        # Lambdas
        "lambda_local":       round(lam_l, 2),
        "lambda_visita":      round(lam_v, 2),
        "lambda_total":       round(lam_t, 2),
        # Over/Under
        "prob_over_7.5":      round(p_over(7.5),  3),
        "prob_over_8.5":      round(p_over(8.5),  3),
        "prob_over_9.5":      round(p_over(9.5),  3),
        "prob_over_10.5":     round(p_over(10.5), 3),
        "prob_over_11.5":     round(p_over(11.5), 3),
        "prob_under_8.5":     round(1 - p_over(8.5),  3),
        "prob_under_9.5":     round(1 - p_over(9.5),  3),
        "prob_under_10.5":    round(1 - p_over(10.5), 3),
        # Contexto
        "rivalry_bonus":      ctx.rivalry_corners_bonus,
        "rivalry_intensity":  ctx.rivalry_intensity,
        "phase":              ctx.phase,
        "home_adv_factor":    ctx.home_adv_factor,
        "notes":              ctx.notes,
        # Crudo (sin feeling) para debug
        "_lambda_raw_local":  round(lam_l0, 2),
        "_lambda_raw_visita": round(lam_v0, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

class CornersModel:
    def __init__(self):
        self._model: dict = {}

    def fit(self, verbose: bool = True) -> "CornersModel":
        df = pd.read_csv(EVENTS)
        df = df.dropna(subset=["corners_local", "corners_visitante"])
        df["corners_local"]     = df["corners_local"].astype(int)
        df["corners_visitante"] = df["corners_visitante"].astype(int)

        if verbose:
            print(f"\n── Entrenando CornersModel (Dixon-Coles + Rue-Salvesen) ──")
            print(f"  Dataset: {len(df)} partidos en {df['torneo'].nunique()} torneos")
            print(f"  μ corners = {df['corners_local'].mean():.2f} local | "
                  f"{df['corners_visitante'].mean():.2f} visita")

        w = _build_weights(df)
        self._model = _fit_mle(df, w, verbose)
        MODEL_OUT.write_text(json.dumps(self._model, ensure_ascii=False, indent=2))
        if verbose:
            print(f"  Modelo guardado → {MODEL_OUT}")
        return self

    def load(self) -> "CornersModel":
        self._model = json.loads(MODEL_OUT.read_text())
        return self

    def predict(self, local: str, visita: str, **kwargs) -> dict:
        if not self._model:
            self.load()
        return predict(self._model, local, visita, **kwargs)

    @property
    def n_partidos(self) -> int:
        return self._model.get("n_partidos", 0)

    @property
    def torneos(self) -> list:
        return self._model.get("torneos", [])


def predecir_corners(local: str, visita: str,
                     cuota_over: float = None, cuota_under: float = None,
                     linea: float = 9.5, jornada: int = 10,
                     is_liguilla: bool = False, stage: str = "",
                     referee: str = "") -> dict:
    """API pública simplificada."""
    m = CornersModel()
    if MODEL_OUT.exists():
        m.load()
    else:
        m.fit(verbose=False)

    result = m.predict(local, visita, jornada=jornada,
                       is_liguilla=is_liguilla, stage=stage, referee=referee)

    # Calcular EV si hay cuotas
    over_key  = f"prob_over_{linea}"
    under_key = f"prob_under_{linea}"

    if cuota_over and over_key in result:
        ev = round(result[over_key] * cuota_over - 1, 4)
        result[f"ev_over_{linea}"]  = ev
        result["apuesta_over"]      = "✅ VALUE" if ev > 0.05 else ("⚠️" if ev > 0 else "❌")

    if cuota_under and under_key in result:
        ev = round(result[under_key] * cuota_under - 1, 4)
        result[f"ev_under_{linea}"] = ev
        result["apuesta_under"]     = "✅ VALUE" if ev > 0.05 else ("⚠️" if ev > 0 else "❌")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Modelo Poisson corners Liga MX (Dixon-Coles + Rue-Salvesen)"
    )
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--local",       default="Cruz Azul")
    parser.add_argument("--visita",      default="América")
    parser.add_argument("--linea",       type=float, default=9.5)
    parser.add_argument("--cuota-over",  type=float)
    parser.add_argument("--cuota-under", type=float)
    parser.add_argument("--jornada",     type=int,   default=10)
    parser.add_argument("--liguilla",    action="store_true")
    parser.add_argument("--stage",       default="")
    parser.add_argument("--arbitro",     default="")
    args = parser.parse_args()

    m = CornersModel()
    if args.train or not MODEL_OUT.exists():
        m.fit(verbose=True)
    else:
        m.load()
        print(f"Modelo cargado: {m.n_partidos} partidos, {len(m.torneos)} torneos")

    r = m.predict(args.local, args.visita,
                  jornada=args.jornada, is_liguilla=args.liguilla,
                  stage=args.stage, referee=args.arbitro)

    print(f"\n── Corners: {r['local']} vs {r['visita']} ──")
    if r["notes"]:
        for note in r["notes"]:
            print(f"  📌 {note}")
    print(f"  λ local={r['lambda_local']} (raw {r['_lambda_raw_local']}) "
          f"| visita={r['lambda_visita']} (raw {r['_lambda_raw_visita']}) "
          f"| total={r['lambda_total']}")
    print(f"  Over  7.5: {r['prob_over_7.5']:.1%}")
    print(f"  Over  8.5: {r['prob_over_8.5']:.1%}  |  Under 8.5: {r['prob_under_8.5']:.1%}")
    print(f"  Over  9.5: {r['prob_over_9.5']:.1%}  |  Under 9.5: {r['prob_under_9.5']:.1%}")
    print(f"  Over 10.5: {r['prob_over_10.5']:.1%}  |  Under 10.5: {r['prob_under_10.5']:.1%}")

    if args.cuota_over or args.cuota_under:
        r2 = predecir_corners(args.local, args.visita,
                              cuota_over=args.cuota_over,
                              cuota_under=args.cuota_under,
                              linea=args.linea, jornada=args.jornada)
        linea = args.linea
        if f"ev_over_{linea}" in r2:
            print(f"\n  → Over  {linea} @ {args.cuota_over}: EV {r2[f'ev_over_{linea}']:+.1%}  {r2.get('apuesta_over','')}")
        if f"ev_under_{linea}" in r2:
            print(f"  → Under {linea} @ {args.cuota_under}: EV {r2[f'ev_under_{linea}']:+.1%}  {r2.get('apuesta_under','')}")


if __name__ == "__main__":
    main()
