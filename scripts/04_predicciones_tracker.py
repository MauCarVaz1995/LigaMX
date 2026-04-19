#!/usr/bin/env python3
"""
04_predicciones_tracker.py
Sistema de seguimiento y evaluación del modelo ELO+Poisson.

Guarda cada predicción en:  data/processed/predicciones_log.csv
Permite registrar resultados reales y reportar el desempeño histórico.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTACIÓN desde otro script (nombre con dígito → usar importlib):

    import importlib, sys
    sys.path.insert(0, 'scripts')
    tracker = importlib.import_module('04_predicciones_tracker')
    tracker.registrar_prediccion(...)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uso standalone:

    # Registrar resultado de un partido ya predicho
    python scripts/04_predicciones_tracker.py resultado \
        "Mexico vs Portugal" 2026-03-29 0 0

    # Ver reporte de desempeño
    python scripts/04_predicciones_tracker.py reporte

    # Listar predicciones pendientes
    python scripts/04_predicciones_tracker.py pendientes
"""

import argparse
import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data" / "processed" / "predicciones_log.csv"
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "fecha_prediccion",       # YYYY-MM-DD HH:MM:SS — cuándo se generó
    "partido",                # "Mexico vs Portugal"
    "equipo_local",
    "equipo_visitante",
    "elo_local",
    "elo_visitante",
    "prob_local",             # % (0-100)
    "prob_empate",
    "prob_visitante",
    "ganador_predicho",       # "Local" | "Empate" | "Visitante"
    "marcador_mas_probable",  # "1-1"
    "lambda_local",
    "lambda_visitante",
    "paleta_usada",
    "fecha_partido",          # YYYY-MM-DD — cuándo se juega
    # ── campos que se llenan al conocerse el resultado ──
    "resultado_real",         # "Local" | "Empate" | "Visitante"
    "goles_local_real",
    "goles_visitante_real",
    "acierto_ganador",        # True | False
    "error_marcador",         # |g_L_pred - g_L_real| + |g_V_pred - g_V_real|
]


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ─────────────────────────────────────────────────────────────────────────────
def _load() -> pd.DataFrame:
    """Carga el CSV o devuelve un DataFrame vacío con las columnas correctas."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, dtype=str)
        # Asegurar que existen todas las columnas (compatibilidad hacia adelante)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[CSV_COLUMNS]
    return pd.DataFrame(columns=CSV_COLUMNS)


def _save(df: pd.DataFrame) -> None:
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")


def _ganador_predicho(prob_local: float, prob_empate: float, prob_visitante: float) -> str:
    """Determina el resultado predicho basándose en la probabilidad más alta."""
    if prob_local >= prob_empate and prob_local >= prob_visitante:
        return "Local"
    elif prob_empate >= prob_local and prob_empate >= prob_visitante:
        return "Empate"
    else:
        return "Visitante"


def _resultado_real(goles_local: int, goles_visitante: int) -> str:
    if goles_local > goles_visitante:
        return "local"
    elif goles_local == goles_visitante:
        return "empate"
    else:
        return "visitante"


def _error_marcador(marcador_predicho: str, goles_local_real: int, goles_visitante_real: int) -> int:
    """
    Distancia Manhattan entre marcador predicho y real.
    marcador_predicho: "1-1"  →  (1, 1)
    """
    try:
        gl_pred, gv_pred = map(int, marcador_predicho.split("-"))
        return abs(gl_pred - goles_local_real) + abs(gv_pred - goles_visitante_real)
    except Exception:
        return -1


def _match_key(partido: str, fecha_partido: str) -> str:
    """Clave única para identificar un partido."""
    return f"{partido.strip().lower()}|{fecha_partido.strip()}"


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA 1 — registrar_prediccion
# ─────────────────────────────────────────────────────────────────────────────
def registrar_prediccion(
    equipo_local: str,
    equipo_visitante: str,
    elo_local: float,
    elo_visitante: float,
    prob_local: float,
    prob_empate: float,
    prob_visitante: float,
    marcador_probable: str,
    lambda_local: float,
    lambda_visitante: float,
    fecha_partido: str,
    paleta: str = "",
    sobreescribir: bool = False,
) -> None:
    """
    Registra una nueva predicción en el CSV de seguimiento.

    Si el partido ya existe para esa fecha (y sobreescribir=False),
    lo omite silenciosamente para no duplicar.

    Args:
        equipo_local:     nombre del equipo local (en inglés)
        equipo_visitante: nombre del equipo visitante
        elo_local:        ELO del equipo local al momento de predecir
        elo_visitante:    ELO del equipo visitante
        prob_local:       probabilidad de ganar el local (0-1 o 0-100, se normaliza)
        prob_empate:      probabilidad de empate
        prob_visitante:   probabilidad de ganar el visitante
        marcador_probable: "G_L-G_V", ej. "1-1"
        lambda_local:     lambda de Poisson del equipo local
        lambda_visitante: lambda de Poisson del equipo visitante
        fecha_partido:    "YYYY-MM-DD" — fecha del partido
        paleta:           nombre de la paleta visual usada (opcional)
        sobreescribir:    si True, reemplaza la predicción existente
    """
    # Normalizar probabilidades a porcentaje si vienen como fracción (0-1)
    if max(prob_local, prob_empate, prob_visitante) <= 1.0:
        prob_local     *= 100
        prob_empate    *= 100
        prob_visitante *= 100

    partido = f"{equipo_local} vs {equipo_visitante}"
    key     = _match_key(partido, fecha_partido)

    df = _load()

    # Verificar duplicados
    if len(df) > 0:
        existing_keys = df.apply(
            lambda r: _match_key(r["partido"], r["fecha_partido"]), axis=1
        )
        if key in existing_keys.values:
            if not sobreescribir:
                print(f"  [tracker] Predicción ya existe: {partido} ({fecha_partido}) — omitida")
                return
            else:
                df = df[existing_keys != key]

    nueva = {
        "fecha_prediccion":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "partido":             partido,
        "equipo_local":        equipo_local,
        "equipo_visitante":    equipo_visitante,
        "elo_local":           round(elo_local, 1),
        "elo_visitante":       round(elo_visitante, 1),
        "prob_local":          round(prob_local, 1),
        "prob_empate":         round(prob_empate, 1),
        "prob_visitante":      round(prob_visitante, 1),
        "ganador_predicho":    _ganador_predicho(prob_local, prob_empate, prob_visitante),
        "marcador_mas_probable": marcador_probable,
        "lambda_local":        round(lambda_local, 4),
        "lambda_visitante":    round(lambda_visitante, 4),
        "paleta_usada":        paleta,
        "fecha_partido":       fecha_partido,
        # Campos pendientes
        "resultado_real":      "",
        "goles_local_real":    "",
        "goles_visitante_real": "",
        "acierto_ganador":     "",
        "error_marcador":      "",
    }

    df = pd.concat([df, pd.DataFrame([nueva])], ignore_index=True)
    _save(df)
    print(f"  [tracker] ✓ Predicción registrada: {partido} ({fecha_partido})")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA 2 — registrar_resultado
# ─────────────────────────────────────────────────────────────────────────────
def registrar_resultado(
    partido: str,
    fecha_partido: str,
    goles_local: int,
    goles_visitante: int,
) -> bool:
    """
    Llena las columnas de resultado real para un partido ya predicho.
    Calcula automáticamente acierto_ganador y error_marcador.

    Args:
        partido:          "Mexico vs Portugal" (debe coincidir exactamente con el CSV)
        fecha_partido:    "YYYY-MM-DD"
        goles_local:      goles del equipo local en el partido real
        goles_visitante:  goles del equipo visitante

    Returns:
        True si se encontró y actualizó, False si no se encontró.

    Ejemplo:
        registrar_resultado("Mexico vs Portugal", "2026-03-29", 0, 0)
    """
    df  = _load()
    key = _match_key(partido, fecha_partido)

    mask = df.apply(
        lambda r: _match_key(r["partido"], r["fecha_partido"]) == key, axis=1
    )

    if not mask.any():
        print(
            f"  [tracker] No se encontró: '{partido}' ({fecha_partido})\n"
            f"  Partidos disponibles:\n"
            + "\n".join(f"    · {r['partido']} ({r['fecha_partido']})" for _, r in df.iterrows()),
            file=sys.stderr,
        )
        return False

    idx = df[mask].index[0]
    res = _resultado_real(goles_local, goles_visitante)
    err = _error_marcador(df.at[idx, "marcador_mas_probable"], goles_local, goles_visitante)

    ganador_pred = df.at[idx, "ganador_predicho"]
    acierto      = (str(ganador_pred).lower().strip() == res)

    df.at[idx, "resultado_real"]         = res
    df.at[idx, "goles_local_real"]       = goles_local
    df.at[idx, "goles_visitante_real"]   = goles_visitante
    df.at[idx, "acierto_ganador"]        = acierto
    df.at[idx, "error_marcador"]         = err

    _save(df)

    mark = "✓" if acierto else "✗"
    print(
        f"  [tracker] {mark} Resultado registrado: {partido} ({fecha_partido})\n"
        f"     Predicho : {ganador_pred}  ({df.at[idx, 'marcador_mas_probable']})\n"
        f"     Real     : {res}  ({goles_local}-{goles_visitante})\n"
        f"     Acierto  : {acierto}  |  Error marcador: {err}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA 3 — reporte_desempeno
# ─────────────────────────────────────────────────────────────────────────────
def reporte_desempeno(n_ultimos: int = 10) -> None:
    """
    Imprime un resumen del desempeño histórico del modelo.

    Args:
        n_ultimos: cuántos partidos recientes mostrar en la tabla (default 10)
    """
    df = _load()
    total = len(df)
    if total == 0:
        print("Sin predicciones registradas aún.")
        return

    # Separar con y sin resultado
    con_res   = df[df["resultado_real"].notna() & (df["resultado_real"] != "")]
    sin_res   = df[df["resultado_real"].isna()  | (df["resultado_real"] == "")]
    n_con     = len(con_res)
    n_sin     = len(sin_res)

    # Métricas (solo partidos con resultado)
    aciertos     = 0
    error_total  = 0.0
    errores_vals = []

    if n_con > 0:
        for _, r in con_res.iterrows():
            if str(r["acierto_ganador"]).lower() == "true":
                aciertos += 1
            try:
                ev = float(r["error_marcador"])
                if ev >= 0:
                    errores_vals.append(ev)
                    error_total += ev
            except (ValueError, TypeError):
                pass

    pct_acierto = (aciertos / n_con * 100) if n_con > 0 else 0.0
    error_prom  = (sum(errores_vals) / len(errores_vals)) if errores_vals else 0.0

    # Breakdown por tipo de resultado predicho vs real
    breakdown = {}
    if n_con > 0:
        for _, r in con_res.iterrows():
            gp  = r["ganador_predicho"]
            res = r["resultado_real"]
            key = (gp, res)
            breakdown[key] = breakdown.get(key, 0) + 1

    # ── Impresión ─────────────────────────────────────────────────────────────
    W = 62
    print("\n" + "═" * W)
    print(f"  REPORTE DE DESEMPEÑO — MODELO ELO+POISSON")
    print("═" * W)
    print(f"  Total predicciones   : {total}")
    print(f"  Con resultado        : {n_con}")
    print(f"  Pendientes           : {n_sin}")
    print("─" * W)

    if n_con > 0:
        print(f"  % Acierto ganador    : {pct_acierto:.1f}%  ({aciertos}/{n_con})")
        print(f"  Error promedio marc. : {error_prom:.2f} goles")

        if breakdown:
            print("\n  Matriz de confusión (predicho → real):")
            for (gp, res), cnt in sorted(breakdown.items()):
                mark = "✓" if gp == res else "✗"
                print(f"    {mark}  {gp:<12} → {res:<12}  {cnt:>3}×")

    # ── Distribución de probabilidades predichas ───────────────────────────
    if n_con > 0:
        pl_mean  = pd.to_numeric(con_res["prob_local"],     errors="coerce").mean()
        pe_mean  = pd.to_numeric(con_res["prob_empate"],    errors="coerce").mean()
        pv_mean  = pd.to_numeric(con_res["prob_visitante"], errors="coerce").mean()
        print(f"\n  Prob. promedio predicha (con resultado):")
        print(f"    Local {pl_mean:.1f}%  |  Empate {pe_mean:.1f}%  |  Visitante {pv_mean:.1f}%")

    # ── Tabla últimos N partidos ───────────────────────────────────────────
    print(f"\n  Últimos {min(n_ultimos, total)} partidos predichos:")
    print(f"  {'Fecha':>10}  {'Partido':<28}  {'Pred':^8}  {'Real':^8}  {'Err':>4}  {'✓'}")
    print(f"  {'─'*10}  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'}")

    recientes = df.sort_values("fecha_prediccion", ascending=False).head(n_ultimos)
    for _, r in recientes.iterrows():
        res    = r["resultado_real"] if r["resultado_real"] else "—"
        err    = r["error_marcador"] if r["error_marcador"] else "—"
        ac_str = ""
        if str(r["acierto_ganador"]).lower() == "true":
            ac_str = "✓"
        elif str(r["acierto_ganador"]).lower() == "false":
            ac_str = "✗"

        # Acortar nombre del partido si es muy largo
        partido_short = r["partido"][:27] if len(r["partido"]) > 27 else r["partido"]
        fecha_p = str(r["fecha_partido"])[:10]

        print(
            f"  {fecha_p:>10}  {partido_short:<28}  "
            f"{str(r['ganador_predicho']):^8}  {str(res):^8}  "
            f"{str(err):>4}  {ac_str}"
        )

    print("═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT STANDALONE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tracker de predicciones ELO+Poisson",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Registrar resultado de un partido
  python scripts/04_predicciones_tracker.py resultado "Mexico vs Portugal" 2026-03-29 0 0

  # Ver reporte completo
  python scripts/04_predicciones_tracker.py reporte

  # Ver predicciones pendientes
  python scripts/04_predicciones_tracker.py pendientes
        """,
    )
    sub = parser.add_subparsers(dest="cmd")

    # Subcomando: resultado
    p_res = sub.add_parser("resultado", help="Registrar resultado real de un partido")
    p_res.add_argument("partido",         help='"Equipo1 vs Equipo2"')
    p_res.add_argument("fecha",           help="YYYY-MM-DD")
    p_res.add_argument("goles_local",     type=int)
    p_res.add_argument("goles_visitante", type=int)

    # Subcomando: reporte
    p_rep = sub.add_parser("reporte", help="Mostrar reporte de desempeño")
    p_rep.add_argument("--n", type=int, default=10, help="Nº de partidos recientes (default 10)")

    # Subcomando: pendientes
    sub.add_parser("pendientes", help="Listar predicciones sin resultado registrado")

    # Subcomando: listar
    sub.add_parser("listar", help="Listar todas las predicciones")

    args = parser.parse_args()

    if args.cmd == "resultado":
        ok = registrar_resultado(args.partido, args.fecha, args.goles_local, args.goles_visitante)
        sys.exit(0 if ok else 1)

    elif args.cmd == "reporte":
        reporte_desempeno(n_ultimos=args.n)

    elif args.cmd == "pendientes":
        df = _load()
        sin_res = df[df["resultado_real"].isna() | (df["resultado_real"] == "")]
        if sin_res.empty:
            print("Sin predicciones pendientes.")
        else:
            print(f"\n{len(sin_res)} predicción(es) pendiente(s) de resultado:\n")
            for _, r in sin_res.sort_values("fecha_partido").iterrows():
                print(f"  · {r['fecha_partido']:>10}  {r['partido']:<30}  "
                      f"Pred: {r['ganador_predicho']}  ({r['marcador_mas_probable']})")

    elif args.cmd == "listar":
        df = _load()
        if df.empty:
            print("Sin predicciones.")
        else:
            print(df.to_string(max_cols=8))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
