#!/usr/bin/env python3
"""
scrape_odds.py — Descarga cuotas históricas de football-data.co.uk
===================================================================
Descarga cuotas 1X2 y Over/Under 2.5 de Bet365/Pinnacle para ligas europeas
y sudamericanas. Guarda en data/processed/odds_historico.csv.

Estas cuotas permiten:
  1. Backtesting CLV: ¿nuestro modelo supera consistentemente las cuotas?
  2. Detección de value: partido donde prob_modelo > prob_implícita
  3. Calibración: comparar distribución de probabilidades modelo vs mercado

Fuente: football-data.co.uk (GRATIS, sin API key, desde 1993)
  Cobertura: Bundesliga, La Liga, Serie A, Ligue 1, Premier League,
             Liga Argentina (parcial), Brasileirao (parcial)
  Columnas clave: B365H/D/A (Bet365 1X2), B365>2.5, B365<2.5

NOTA: Liga MX NO está en football-data.co.uk.
Para Liga MX en producción → usar The Odds API (500 req/mes gratis).

Uso:
  python scripts/scrape_odds.py                        # todas las ligas, temporada actual
  python scripts/scrape_odds.py --liga bundesliga      # solo Bundesliga
  python scripts/scrape_odds.py --temporadas 3         # últimas 3 temporadas
  python scripts/scrape_odds.py --backfill             # descarga todo el histórico
  python scripts/scrape_odds.py --clv                  # calcula CLV vs predicciones_log
"""

import argparse
import io
import time
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE      = Path(__file__).resolve().parent.parent
ODDS_CSV  = BASE / "data/processed/odds_historico.csv"
PRED_LOG  = BASE / "data/processed/predicciones_log.csv"

TODAY = date.today()

# ─────────────────────────────────────────────────────────────────────────────
# Catálogo de ligas
# ─────────────────────────────────────────────────────────────────────────────
LIGAS = {
    "bundesliga":    {"code": "D1",   "nombre": "Bundesliga",       "pais": "Germany",    "desde": 1994},
    "laliga":        {"code": "SP1",  "nombre": "La Liga",          "pais": "Spain",      "desde": 1994},
    "seriea":        {"code": "I1",   "nombre": "Serie A",          "pais": "Italy",      "desde": 1994},
    "ligue1":        {"code": "F1",   "nombre": "Ligue 1",          "pais": "France",     "desde": 1994},
    "premier":       {"code": "E0",   "nombre": "Premier League",   "pais": "England",    "desde": 1994},
    "eredivisie":    {"code": "N1",   "nombre": "Eredivisie",       "pais": "Netherlands","desde": 2000},
    "portuguesa":    {"code": "P1",   "nombre": "Primeira Liga",    "pais": "Portugal",   "desde": 1995},
    "argentina":     {"code": "ARG",  "nombre": "Liga Argentina",   "pais": "Argentina",  "desde": 2012},
    "brasileirao":   {"code": "BRA",  "nombre": "Brasileirao",      "pais": "Brazil",     "desde": 2012},
}

# Columnas de cuotas que nos interesan (en orden de preferencia)
# football-data.co.uk tiene Bet365, Pinnacle, Betfair, promedio de varias casas
ODD_COLS_MAP = {
    # 1X2 — primero Pinnacle (mejor mercado), luego Bet365
    "odd_1": ["PSH", "B365H", "BbAvH"],    # local gana
    "odd_X": ["PSD", "B365D", "BbAvD"],    # empate
    "odd_2": ["PSA", "B365A", "BbAvA"],    # visitante gana
    # Over/Under 2.5
    "odd_over25":  ["PSC>2.5", "B365>2.5", "BbAv>2.5"],
    "odd_under25": ["PSC<2.5", "B365<2.5", "BbAv<2.5"],
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _season_codes(desde_año: int, hasta_año: int = None) -> list[str]:
    """Genera códigos de temporada: 9394, 9495, ..., 2526."""
    hasta = hasta_año or TODAY.year
    result = []
    for y in range(desde_año, hasta + 1):
        s = f"{str(y)[-2:]}{str(y+1)[-2:]}"
        result.append(s)
    return result


def _pick_col(df: pd.DataFrame, candidates: list[str]):
    """Devuelve la primera columna candidata que exista en df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_team(name: str) -> str:
    """Normalización básica de nombres para comparación."""
    return str(name).strip().lower()


def fetch_season(liga_key: str, season_code: str, verbose: bool = True) -> pd.DataFrame | None:
    """
    Descarga CSV de una liga/temporada. Retorna DataFrame normalizado o None si falla.
    """
    liga = LIGAS[liga_key]
    url  = BASE_URL.format(season=season_code, code=liga["code"])

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return None  # temporada no existe aún
        resp.raise_for_status()

        # football-data.co.uk a veces tiene encoding latin-1
        try:
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception:
            df = pd.read_csv(io.StringIO(resp.content.decode("latin-1")))

        if df.empty or "HomeTeam" not in df.columns:
            return None

        # ── Normalizar estructura ─────────────────────────────────────────
        rows = []
        for _, r in df.iterrows():
            # Parsear fecha (football-data usa DD/MM/YY o DD/MM/YYYY)
            try:
                fecha_raw = str(r.get("Date", "")).strip()
                if len(fecha_raw) == 8:   # DD/MM/YY
                    fecha = pd.to_datetime(fecha_raw, format="%d/%m/%y").date()
                else:
                    fecha = pd.to_datetime(fecha_raw, dayfirst=True).date()
            except Exception:
                continue

            local   = str(r.get("HomeTeam", "")).strip()
            visita  = str(r.get("AwayTeam",  "")).strip()
            if not local or not visita:
                continue

            # Goles reales
            fthg = r.get("FTHG", r.get("HG"))
            ftag = r.get("FTAG", r.get("AG"))

            row = {
                "fecha":       str(fecha),
                "liga":        liga_key,
                "nombre_liga": liga["nombre"],
                "pais":        liga["pais"],
                "temporada":   f"20{season_code[:2]}/20{season_code[2:]}",
                "local":       local,
                "visitante":   visita,
                "goles_local": int(fthg) if pd.notna(fthg) else None,
                "goles_visita":int(ftag) if pd.notna(ftag) else None,
            }

            # Cuotas — tomar primera columna disponible
            for out_col, candidates in ODD_COLS_MAP.items():
                col = _pick_col(df, candidates)
                val = pd.to_numeric(r.get(col), errors="coerce") if col else None
                row[out_col] = round(float(val), 3) if pd.notna(val) and val else None

            rows.append(row)

        result = pd.DataFrame(rows)
        if verbose and not result.empty:
            print(f"    {liga['nombre']} {season_code}: {len(result)} partidos")
        return result

    except requests.RequestException as e:
        if verbose:
            print(f"    [warn] {liga_key} {season_code}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Guardar — append incremental
# ─────────────────────────────────────────────────────────────────────────────
def save_odds(df_new: pd.DataFrame, verbose: bool = True):
    """
    Agrega cuotas nuevas al CSV histórico. Deduplicación por (fecha, liga, local, visitante).
    Escritura atómica para no corromper el archivo.
    """
    import os, tempfile

    if df_new.empty:
        return 0

    if ODDS_CSV.exists():
        df_old = pd.read_csv(ODDS_CSV, low_memory=False)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new.copy()

    # Deduplicar
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(
        subset=["fecha", "liga", "local", "visitante"], keep="last"
    )
    df_combined = df_combined.sort_values(["liga", "fecha"]).reset_index(drop=True)
    nuevos = len(df_combined) - (len(pd.read_csv(ODDS_CSV, low_memory=False)) if ODDS_CSV.exists() else 0)

    # Escritura atómica
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(ODDS_CSV.parent), suffix=".tmp")
    try:
        df_combined.to_csv(tmp_path, index=False)
        os.close(tmp_fd)
        os.replace(tmp_path, str(ODDS_CSV))
    except Exception:
        os.close(tmp_fd)
        os.unlink(tmp_path)
        raise

    if verbose:
        print(f"  → odds_historico.csv: {len(df_combined):,} partidos totales (+{nuevos} nuevos)")
    return nuevos


# ─────────────────────────────────────────────────────────────────────────────
# Scraping principal
# ─────────────────────────────────────────────────────────────────────────────
def scrape(ligas: list[str], n_temporadas: int = 1, backfill: bool = False,
           verbose: bool = True) -> int:
    """
    Descarga cuotas para las ligas y temporadas indicadas.
    Retorna el número de partidos nuevos guardados.
    """
    total_nuevos = 0

    for liga_key in ligas:
        if liga_key not in LIGAS:
            print(f"  [warn] Liga desconocida: {liga_key}")
            continue

        liga = LIGAS[liga_key]
        if verbose:
            print(f"\n── {liga['nombre']} ({liga['pais']}) ──")

        if backfill:
            temporadas = _season_codes(liga["desde"])
        else:
            # Última N temporadas
            año_inicio = max(liga["desde"], TODAY.year - n_temporadas)
            temporadas = _season_codes(año_inicio)

        frames = []
        for season in temporadas:
            df = fetch_season(liga_key, season, verbose=verbose)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(0.3)  # ser amable con el servidor

        if frames:
            df_liga = pd.concat(frames, ignore_index=True)
            nuevos  = save_odds(df_liga, verbose=verbose)
            total_nuevos += nuevos

    return total_nuevos


# ─────────────────────────────────────────────────────────────────────────────
# Análisis CLV (Closing Line Value)
# ─────────────────────────────────────────────────────────────────────────────
def calcular_clv(verbose: bool = True) -> dict:
    """
    Compara las probabilidades del modelo vs las cuotas de cierre.

    CLV = prob_modelo / prob_implícita - 1
    Si CLV > 0 de forma consistente → el modelo tiene edge real sobre el mercado.

    Retorna dict con métricas agregadas.
    """
    if not PRED_LOG.exists() or not ODDS_CSV.exists():
        return {"error": "Faltan archivos (predicciones_log.csv u odds_historico.csv)"}

    preds = pd.read_csv(PRED_LOG)
    odds  = pd.read_csv(ODDS_CSV, low_memory=False)

    # Solo predicciones con resultado real conocido
    preds = preds[preds["resultado_real"].isin(["local", "empate", "visitante"])].copy()
    if preds.empty:
        return {"n_comparaciones": 0, "msg": "Sin predicciones evaluadas aún"}

    # Normalizar nombres para join
    preds["_local_n"] = preds["equipo_local"].str.strip().str.lower()
    preds["_visit_n"] = preds["equipo_visitante"].str.strip().str.lower()
    preds["_fecha"]   = preds["fecha_partido"].str[:10]
    odds["_local_n"]  = odds["local"].str.strip().str.lower()
    odds["_visit_n"]  = odds["visitante"].str.strip().str.lower()
    odds["_fecha"]    = odds["fecha"].str[:10]

    merged = preds.merge(
        odds[["_fecha", "_local_n", "_visit_n",
              "odd_1", "odd_X", "odd_2", "odd_over25", "odd_under25"]],
        on=["_fecha", "_local_n", "_visit_n"],
        how="left"
    )

    n_match = merged["odd_1"].notna().sum()
    if n_match == 0:
        return {
            "n_comparaciones": 0,
            "msg": "Sin coincidencias entre predicciones y cuotas. "
                   "Las ligas en odds_historico.csv probablemente no coinciden con las predicciones."
        }

    # Calcular prob implícita (con margen de la casa → sobrerronda)
    m = merged[merged["odd_1"].notna()].copy()
    m["pi_1"] = 1 / m["odd_1"]
    m["pi_X"] = 1 / m["odd_X"]
    m["pi_2"] = 1 / m["odd_2"]
    m["overround"] = m["pi_1"] + m["pi_X"] + m["pi_2"]
    # Quitar margen (fair odds)
    m["fair_1"] = m["pi_1"] / m["overround"]
    m["fair_X"] = m["pi_X"] / m["overround"]
    m["fair_2"] = m["pi_2"] / m["overround"]

    # CLV por resultado predicho
    def get_clv(row):
        pred = str(row.get("ganador_predicho", "")).lower()
        if pred == "local":
            return row["prob_local"] / 100 / row["fair_1"] - 1 if row["fair_1"] > 0 else None
        elif pred == "empate":
            return row["prob_empate"] / 100 / row["fair_X"] - 1 if row["fair_X"] > 0 else None
        elif pred == "visitante":
            return row["prob_visitante"] / 100 / row["fair_2"] - 1 if row["fair_2"] > 0 else None
        return None

    m["clv"] = m.apply(get_clv, axis=1)
    clv_vals = m["clv"].dropna()

    resultado = {
        "n_predicciones_total":  len(preds),
        "n_comparaciones":       int(n_match),
        "clv_promedio":          round(float(clv_vals.mean()), 4) if len(clv_vals) else None,
        "clv_mediana":           round(float(clv_vals.median()), 4) if len(clv_vals) else None,
        "pct_clv_positivo":      round(float((clv_vals > 0).mean()), 3) if len(clv_vals) else None,
        "interpretacion": (
            "✅ Modelo supera el mercado" if clv_vals.mean() > 0.02
            else "⚠️ Modelo al nivel del mercado" if clv_vals.mean() > -0.02
            else "❌ Mercado supera al modelo"
        ) if len(clv_vals) > 5 else "n insuficiente para conclusiones"
    }

    if verbose:
        print(f"\n── CLV Analysis ──")
        print(f"  Predicciones evaluadas: {len(preds)}")
        print(f"  Con cuotas disponibles: {n_match}")
        if len(clv_vals):
            print(f"  CLV promedio:  {clv_vals.mean():+.1%}")
            print(f"  CLV mediana:   {clv_vals.median():+.1%}")
            print(f"  % CLV > 0:     {(clv_vals > 0).mean():.0%}")
            print(f"  → {resultado['interpretacion']}")

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Descarga cuotas históricas de football-data.co.uk"
    )
    parser.add_argument("--liga",        help="Liga específica (bundesliga, laliga, etc.)")
    parser.add_argument("--all",         action="store_true", help="Todas las ligas")
    parser.add_argument("--temporadas",  type=int, default=2,
                        help="Número de temporadas hacia atrás (default: 2)")
    parser.add_argument("--backfill",    action="store_true",
                        help="Descargar TODO el histórico disponible (~2000 en adelante)")
    parser.add_argument("--clv",         action="store_true",
                        help="Calcular CLV (requiere odds_historico.csv y predicciones_log.csv)")
    parser.add_argument("--verbose",     action="store_true", default=True)
    args = parser.parse_args()

    if args.clv:
        calcular_clv(verbose=True)
        return

    if args.liga:
        ligas = [args.liga]
    elif args.all:
        ligas = list(LIGAS.keys())
    else:
        # Default: las más útiles para backtest (más historial, más eficientes → mejor benchmark)
        ligas = ["bundesliga", "laliga", "seriea", "premier", "argentina", "brasileirao"]

    print(f"\n── Descargando cuotas: {len(ligas)} ligas, "
          f"{'backfill completo' if args.backfill else f'últimas {args.temporadas} temporadas'} ──")

    nuevos = scrape(
        ligas=ligas,
        n_temporadas=args.temporadas,
        backfill=args.backfill,
        verbose=args.verbose,
    )

    print(f"\n✅ Listo — {nuevos} partidos nuevos en odds_historico.csv")

    # Mostrar resumen del archivo
    if ODDS_CSV.exists():
        df = pd.read_csv(ODDS_CSV, low_memory=False)
        print(f"\n── Resumen odds_historico.csv ──")
        summary = df.groupby("nombre_liga").agg(
            partidos=("fecha", "count"),
            desde=("fecha", "min"),
            hasta=("fecha", "max"),
            pct_odds=("odd_1", lambda x: f"{x.notna().mean():.0%}")
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
