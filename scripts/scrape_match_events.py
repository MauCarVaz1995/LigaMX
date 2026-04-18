#!/usr/bin/env python3
"""
scrape_match_events.py — Extrae corners, tarjetas, shots de FotMob matchfacts
==============================================================================
Para cada partido del historico Liga MX, descarga los detalles de FotMob
y extrae los eventos que necesitamos para los modelos de betting:
  - Corners (por equipo y total)
  - Tarjetas amarillas y rojas (por equipo y total)
  - Shots / shots on target
  - xG (si FotMob lo tiene disponible)

Estrategia:
  1. Lee historico_clausura_2026.json para la lista de partidos + match_ids
  2. Para cada partido terminado, usa match_id directo (ya lo tenemos)
  3. Cachea el JSON crudo en data/raw/fotmob/ para no re-descargar
  4. Extrae los campos y guarda en data/processed/match_events.csv

Uso:
  python scripts/scrape_match_events.py                # Clausura 2026 (default)
  python scripts/scrape_match_events.py --all          # todos los torneos del historico
  python scripts/scrape_match_events.py --days 7       # solo últimos 7 días
  python scripts/scrape_match_events.py --force        # re-descargar aunque exista caché

Output:
  data/processed/match_events.csv
  Columnas: match_id, fecha, jornada, torneo, local, visitante,
            goles_local, goles_visitante,
            corners_local, corners_visitante, corners_total,
            amarillas_local, amarillas_visitante, amarillas_total,
            rojas_local, rojas_visitante, rojas_total,
            shots_local, shots_visitante,
            shots_ot_local, shots_ot_visitante,
            xg_local, xg_visitante
"""

import argparse
import importlib
import json
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent

HIST_DIR  = BASE / "data/raw/historico"
RAW_DIR   = BASE / "data/raw/fotmob"
OUT_CSV   = BASE / "data/processed/match_events.csv"

# Importar 03 con importlib
import sys
sys.path.insert(0, str(SCRIPTS))
_m03 = importlib.import_module("03_get_match_stats_fotmob")
fetch_match_details = _m03.fetch_match_details

SLEEP_BETWEEN = 1.5  # segundos entre requests


# ─────────────────────────────────────────────────────────────────────────────
def _flatten_stats(content: dict) -> dict:
    """
    Aplana stats de FotMob: navega Periods.All.stats (lista de grupos)
    y devuelve {key: [home_val, away_val]} para todos los stats disponibles.
    """
    flat = {}
    try:
        periods = content.get("stats", {}).get("Periods", {})
        all_period = periods.get("All", {})
        for group in all_period.get("stats", []):
            for stat in group.get("stats", []) if isinstance(group.get("stats"), list) else [group]:
                key  = stat.get("key", "")
                vals = stat.get("stats")
                if key and isinstance(vals, list) and len(vals) == 2:
                    flat[key] = vals
    except Exception:
        pass
    return flat


def _get(flat: dict, *keys) -> tuple[float | None, float | None]:
    """Busca el primer key encontrado y retorna (home, away) como float."""
    for key in keys:
        if key in flat:
            vals = flat[key]
            try:
                h = float(str(vals[0]).split(" ")[0]) if vals[0] is not None else None
                a = float(str(vals[1]).split(" ")[0]) if vals[1] is not None else None
                return h, a
            except (ValueError, TypeError, IndexError):
                pass
    return None, None


def _extract_events(content: dict) -> dict:
    """Extrae corners, tarjetas, shots, xG del content raw de FotMob."""
    r: dict = {}

    flat = _flatten_stats(content)

    r["corners_local"],    r["corners_visitante"]   = _get(flat, "corners", "corner_kicks")
    r["amarillas_local"],  r["amarillas_visitante"]  = _get(flat, "yellow_cards")
    r["rojas_local"],      r["rojas_visitante"]      = _get(flat, "red_cards")
    r["shots_local"],      r["shots_visitante"]      = _get(flat, "total_shots")
    r["shots_ot_local"],   r["shots_ot_visitante"]   = _get(flat, "ShotsOnTarget", "shots_on_target")
    r["xg_local"],         r["xg_visitante"]         = _get(flat, "expected_goals", "xG", "xg")

    def total(a, b): return (a + b) if a is not None and b is not None else None
    r["corners_total"]   = total(r["corners_local"],   r["corners_visitante"])
    r["amarillas_total"] = total(r["amarillas_local"],  r["amarillas_visitante"])
    r["rojas_total"]     = total(r["rojas_local"],      r["rojas_visitante"])

    return r


def process_partido(partido: dict, force: bool = False, torneo: str = "") -> dict | None:
    """Descarga y extrae eventos de un partido. Retorna fila para el CSV o None."""
    match_id = int(partido["id"])
    local    = partido["local"]
    visita   = partido["visitante"]
    fecha    = partido["fecha"][:10]

    # Caché del JSON crudo
    import re
    safe_l = re.sub(r"[^\w]", "_", local)
    safe_v = re.sub(r"[^\w]", "_", visita)
    cache  = RAW_DIR / f"{safe_l}_{safe_v}_{fecha}.json"

    content = None

    if cache.exists() and not force:
        try:
            cached = json.loads(cache.read_text())
            # Si tiene el formato del resultado de get_match_stats_by_id, extraer team_stats
            if "team_stats" in cached:
                # Reconstruir content desde caché — descargar de nuevo para tener raw
                pass  # usaremos fetch_match_details de todas formas si no hay raw
        except Exception:
            pass

    # Descargar si no hay caché o es force
    raw_cache = RAW_DIR / f"raw_matchfacts_{match_id}.json"
    if not raw_cache.exists() or force:
        match_info = {"match_id": match_id, "home": local, "away": visita}
        content = fetch_match_details(match_info)
        if content:
            raw_cache.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
            time.sleep(SLEEP_BETWEEN)
    else:
        content = json.loads(raw_cache.read_text())

    if not content:
        return None

    events = _extract_events(content)

    return {
        "match_id":          match_id,
        "fecha":             fecha,
        "jornada":           partido.get("jornada", ""),
        "torneo":            torneo or partido.get("_torneo", "Liga MX"),
        "local":             local,
        "visitante":         visita,
        "goles_local":       partido.get("goles_local"),
        "goles_visitante":   partido.get("goles_visit"),
        **events,
    }


# ─────────────────────────────────────────────────────────────────────────────
def load_historico_partidos(hist_file: Path, since_date: str = None) -> tuple[list[dict], str]:
    with open(hist_file, encoding="utf-8") as f:
        d = json.load(f)
    torneo_raw = d.get("torneo", hist_file.stem.replace("historico_", ""))
    # Normalizar: "2025/2026 - Clausura" → "Liga MX Clausura 2026"
    torneo = f"Liga MX {torneo_raw}" if not torneo_raw.startswith("Liga") else torneo_raw
    partidos = [p for p in d["partidos"] if p.get("terminado")]
    if since_date:
        partidos = [p for p in partidos if p["fecha"][:10] >= since_date]
    return partidos, torneo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",     action="store_true", help="Procesar todos los torneos del historico")
    parser.add_argument("--seasons", type=int, default=None,
                        help="Solo las últimas N temporadas (recomendado: 5-6). "
                             "Prioriza las más recientes. Ignora --all si se especifica.")
    parser.add_argument("--days",    type=int, help="Solo partidos de los últimos N días")
    parser.add_argument("--force",   action="store_true", help="Re-descargar aunque exista caché")
    args = parser.parse_args()

    # Determinar archivos de historico a procesar
    if args.seasons:
        # Las últimas N temporadas (más recientes primero)
        all_files = sorted(HIST_DIR.glob("historico_*.json"), reverse=True)
        hist_files = all_files[:args.seasons]
        hist_files = sorted(hist_files)  # cronológico para procesar
        print(f"  Temporadas seleccionadas ({args.seasons}): {[f.name for f in hist_files]}")
    elif args.all:
        hist_files = sorted(HIST_DIR.glob("historico_*.json"))
    else:
        hist_files = [HIST_DIR / "historico_clausura_2026.json"]

    since_date = None
    if args.days:
        since_date = (date.today() - timedelta(days=args.days)).isoformat()

    # Cargar CSV existente para no re-procesar
    existing_ids: set[int] = set()
    if OUT_CSV.exists() and not args.force:
        existing_df = pd.read_csv(OUT_CSV)
        existing_ids = set(existing_df["match_id"].astype(int).tolist())

    all_partidos = []  # list of (partido_dict, torneo_str)
    for hf in hist_files:
        if not hf.exists():
            print(f"  [skip] {hf.name} no existe")
            continue
        partidos, torneo = load_historico_partidos(hf, since_date)
        all_partidos.extend((p, torneo) for p in partidos)

    # Filtrar ya procesados
    pendientes = [(p, t) for p, t in all_partidos if int(p["id"]) not in existing_ids]

    print(f"\n{'═'*60}")
    print(f"  scrape_match_events.py — {len(pendientes)} partidos pendientes")
    print(f"  (de {len(all_partidos)} totales, {len(existing_ids)} ya en CSV)")
    print(f"{'═'*60}\n")

    if not pendientes:
        print("  Todo al día. Sin partidos pendientes.")
        return

    rows = []
    ok = 0
    for i, (p, torneo) in enumerate(sorted(pendientes, key=lambda x: x[0]["fecha"]), 1):
        print(f"  [{i}/{len(pendientes)}] {p['local']} vs {p['visitante']} ({p['fecha'][:10]})", end=" ")
        row = process_partido(p, force=args.force, torneo=torneo)
        if row:
            # Verificar que conseguimos al menos corners
            has_data = row.get("corners_total") is not None or row.get("amarillas_total") is not None
            if has_data:
                print(f"✓  corners={row.get('corners_total','?')} amrillas={row.get('amarillas_total','?')}")
            else:
                print(f"⚠️  sin corners/tarjetas (FotMob no los tiene)")
            rows.append(row)
            ok += 1
        else:
            print("✗  no disponible")

    if rows:
        new_df = pd.DataFrame(rows)
        if OUT_CSV.exists():
            existing_df = pd.read_csv(OUT_CSV)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["match_id"])
        else:
            combined = new_df
        combined.to_csv(OUT_CSV, index=False)
        print(f"\n  ✅ Guardado: {OUT_CSV} ({len(combined)} partidos totales)")
    else:
        print("\n  Sin datos nuevos para guardar.")

    print(f"\n  Resultado: {ok}/{len(pendientes)} partidos procesados")


if __name__ == "__main__":
    main()
