#!/usr/bin/env python3
"""
modelo_tarjetas.py — Modelo Poisson de tarjetas para Liga MX
=============================================================
Calcula probabilidades Over/Under de tarjetas usando Poisson simple
sobre el card_rate histórico por equipo.

  λ_tarjetas = card_rate[local] + card_rate[visita] + factor_rivalidad

Uso:
  from scripts.modelo_tarjetas import predecir_tarjetas
  result = predecir_tarjetas("Cruz Azul", "América")

  # CLI con cuotas para EV:
  python scripts/modelo_tarjetas.py --local "Cruz Azul" --visita "América" --cuota-over 1.90
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")

BASE      = Path(__file__).resolve().parent.parent
EVENTS    = BASE / "data/processed/match_events.csv"
MODEL_OUT = BASE / "data/processed/tarjetas_model.json"


# ─────────────────────────────────────────────────────────────────────────────
def _load_data(min_partidos: int = 3) -> pd.DataFrame:
    df = pd.read_csv(EVENTS)
    df = df.dropna(subset=["amarillas_local", "amarillas_visitante"])
    df["amarillas_local"]    = df["amarillas_local"].astype(int)
    df["amarillas_visitante"] = df["amarillas_visitante"].astype(int)
    df["rojas_local"]    = df["rojas_local"].fillna(0).astype(int)
    df["rojas_visitante"] = df["rojas_visitante"].fillna(0).astype(int)
    # Tarjetas totales = amarillas + 2×rojas (valor bookmaker estándar)
    df["tarjetas_local"]    = df["amarillas_local"]    + 2 * df["rojas_local"]
    df["tarjetas_visitante"] = df["amarillas_visitante"] + 2 * df["rojas_visitante"]
    df["tarjetas_total"] = df["tarjetas_local"] + df["tarjetas_visitante"]
    return df


class TarjetasModel:
    def __init__(self):
        self.mu            = None   # media global tarjetas ponderadas
        self.card_rate     = {}     # card_rate[equipo] = media tarjetas por partido
        self.rivalidades   = {}     # (local, visita) → factor adicional
        self.n_partidos    = 0

    def fit(self, df: pd.DataFrame = None, verbose: bool = True) -> "TarjetasModel":
        if df is None:
            df = _load_data()
        self.n_partidos = len(df)
        self.mu = df["tarjetas_total"].mean()

        if verbose:
            print(f"  Dataset: {self.n_partidos} partidos")
            print(f"  μ tarjetas ponderadas = {self.mu:.2f} por partido")
            print(f"  μ amarillas = {df['amarillas_local'].mean() + df['amarillas_visitante'].mean():.2f}")

        equipos = set(df["local"].tolist() + df["visitante"].tolist())
        for eq in equipos:
            as_local  = df[df["local"]    == eq]["tarjetas_local"]
            as_visita = df[df["visitante"] == eq]["tarjetas_visitante"]
            todas = pd.concat([as_local, as_visita])
            if len(todas) >= 3:
                self.card_rate[eq] = float(todas.mean())
            else:
                self.card_rate[eq] = float(self.mu / 2)

        # Rivalidades: pares con más tarjetas que la media
        for _, row in df.iterrows():
            pair = (row["local"], row["visitante"])
            if pair not in self.rivalidades:
                self.rivalidades[pair] = []
            self.rivalidades[pair].append(row["tarjetas_total"])

        # Normalizar rivalidades a factor aditivo (0 si ≤ 2 encuentros)
        rivalidad_factors = {}
        for pair, totales in self.rivalidades.items():
            if len(totales) >= 2:
                avg = np.mean(totales)
                rival_factor = max(avg - self.mu, 0)  # solo positivo
                if rival_factor > 0.5:
                    rivalidad_factors[pair] = round(rival_factor, 2)
        self.rivalidades = rivalidad_factors

        if verbose and rivalidad_factors:
            top = sorted(rivalidad_factors.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Rivalidades calientes: {top}")

        if MODEL_OUT.parent.exists():
            MODEL_OUT.write_text(json.dumps({
                "mu": self.mu, "n_partidos": self.n_partidos,
                "card_rate": self.card_rate,
                "rivalidades": {f"{k[0]}|{k[1]}": v for k, v in self.rivalidades.items()},
            }, ensure_ascii=False, indent=2))
            if verbose:
                print(f"  Modelo guardado → {MODEL_OUT}")
        return self

    def load(self) -> "TarjetasModel":
        d = json.loads(MODEL_OUT.read_text())
        self.mu         = d["mu"]
        self.n_partidos = d["n_partidos"]
        self.card_rate  = d["card_rate"]
        self.rivalidades = {tuple(k.split("|")): v for k, v in d["rivalidades"].items()}
        return self

    def predict(self, local: str, visita: str) -> dict:
        cr_l = self.card_rate.get(local,  self.mu / 2)
        cr_v = self.card_rate.get(visita, self.mu / 2)
        rival_bonus = self.rivalidades.get((local, visita), 0.0)

        lam_total = cr_l + cr_v + rival_bonus

        max_k = 25
        dist  = np.array([poisson.pmf(k, lam_total) for k in range(max_k)])

        def p_over(linea: float) -> float:
            return float(dist[int(linea) + 1:].sum())

        return {
            "local": local, "visita": visita,
            "lambda_total":   round(lam_total, 2),
            "cr_local":       round(cr_l, 2),
            "cr_visita":      round(cr_v, 2),
            "rival_bonus":    round(rival_bonus, 2),
            "prob_over_3.5":  round(p_over(3.5),  3),
            "prob_over_4.5":  round(p_over(4.5),  3),
            "prob_over_5.5":  round(p_over(5.5),  3),
            "prob_under_4.5": round(1 - p_over(4.5), 3),
            "prob_under_5.5": round(1 - p_over(5.5), 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
def predecir_tarjetas(local: str, visita: str,
                      cuota_over: float = None, cuota_under: float = None,
                      linea: float = 4.5) -> dict:
    m = TarjetasModel()
    if MODEL_OUT.exists():
        m.load()
    else:
        df = _load_data()
        if len(df) == 0:
            raise RuntimeError(f"No hay datos en {EVENTS}")
        m.fit(df, verbose=False)

    result = m.predict(local, visita)

    over_key  = f"prob_over_{linea}"
    under_key = f"prob_under_{linea}"

    if cuota_over and over_key in result:
        ev = round(result[over_key] * cuota_over - 1, 4)
        result[f"ev_over_{linea}"]  = ev
        result["apuesta_over"]      = "✅ VALUE" if ev > 0.05 else "❌ sin valor"
    if cuota_under and under_key in result:
        ev = round(result[under_key] * cuota_under - 1, 4)
        result[f"ev_under_{linea}"] = ev
        result["apuesta_under"]     = "✅ VALUE" if ev > 0.05 else "❌ sin valor"

    return result


# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",        action="store_true")
    parser.add_argument("--local",        default="Cruz Azul")
    parser.add_argument("--visita",       default="América")
    parser.add_argument("--linea",        type=float, default=4.5)
    parser.add_argument("--cuota-over",   type=float)
    parser.add_argument("--cuota-under",  type=float)
    args = parser.parse_args()

    if args.train or not MODEL_OUT.exists():
        print("\n── Entrenando modelo de tarjetas ──")
        df = _load_data()
        m = TarjetasModel()
        m.fit(df, verbose=True)
    else:
        m = TarjetasModel().load()
        print(f"\n── Modelo cargado: {m.n_partidos} partidos ──")

    print(f"\n── Predicción tarjetas: {args.local} vs {args.visita} ──")
    r = m.predict(args.local, args.visita)
    print(f"  λ total = {r['lambda_total']} (local {r['cr_local']} + visita {r['cr_visita']} + rival {r['rival_bonus']})")
    print(f"  Over 3.5: {r['prob_over_3.5']:.1%}")
    print(f"  Over 4.5: {r['prob_over_4.5']:.1%}  |  Under 4.5: {r['prob_under_4.5']:.1%}")
    print(f"  Over 5.5: {r['prob_over_5.5']:.1%}  |  Under 5.5: {r['prob_under_5.5']:.1%}")

    if args.cuota_over or args.cuota_under:
        r2 = predecir_tarjetas(args.local, args.visita,
                               cuota_over=args.cuota_over,
                               cuota_under=args.cuota_under,
                               linea=args.linea)
        for k in [f"ev_over_{args.linea}", f"ev_under_{args.linea}"]:
            if k in r2:
                label = k.replace("ev_", "")
                apuesta_key = "apuesta_over" if "over" in k else "apuesta_under"
                print(f"  EV {label}: {r2[k]:+.1%}  {r2.get(apuesta_key, '')}")


if __name__ == "__main__":
    main()
