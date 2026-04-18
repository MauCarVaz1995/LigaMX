#!/usr/bin/env python3
"""
gen_postpartido.py — Infografías post-partido automáticas
=========================================================
Para cada partido de Liga MX terminado recientemente descarga stats
de FotMob y genera:
  output/charts/partidos/{LOC}_{VIS}_{fecha}_ratings_{local}.png
  output/charts/partidos/{LOC}_{VIS}_{fecha}_ratings_{visitante}.png
  output/charts/partidos/{LOC}_{VIS}_{fecha}_team_stats.png

Uso:
  python scripts/gen_postpartido.py              # últimos 2 días
  python scripts/gen_postpartido.py --days 5
  python scripts/gen_postpartido.py --force
"""

import argparse
import importlib
import importlib.util
import json
import re
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

HIST_FILE = BASE / "data/raw/historico/historico_clausura_2026.json"
OUT_DIR   = BASE / "output/charts/partidos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Importar 03 con importlib (nombre empieza con dígito)
_m03 = importlib.import_module("03_get_match_stats_fotmob")
get_match_stats       = _m03.get_match_stats
get_match_stats_by_id = _m03.get_match_stats_by_id

# Importar funciones de render de 05
_spec = importlib.util.spec_from_file_location(
    "viz", SCRIPTS / "05_viz_player_performance.py"
)
_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_viz)
render_ratings    = _viz.render_ratings
render_team_stats = _viz.render_team_stats


# ─────────────────────────────────────────────────────────────────────────────
PALETAS_POOL = ["rojo_fuego", "medianoche_neon", "oceano_esmeralda"]

def safe_code(s: str) -> str:
    return re.sub(r"[^\w]", "_", s).strip("_")

def code_for(local: str, visita: str, fecha: str) -> str:
    return f"{safe_code(local)}_{safe_code(visita)}_{fecha}"

def already_done(code: str) -> bool:
    return (OUT_DIR / f"{code}_team_stats.png").exists()


def process_match(partido: dict, force: bool = False) -> bool:
    local  = partido["local"]
    visita = partido["visitante"]
    fecha  = partido["fecha"][:10]
    code   = code_for(local, visita, fecha)

    if already_done(code) and not force:
        print(f"  [skip] {local} vs {visita} ({fecha}) — ya existe")
        return True

    print(f"\n  ── {local} vs {visita}  ({fecha}) ──")

    # Intentar primero con match_id directo (no depende del endpoint de fechas)
    match_id   = int(partido["id"])
    score_home = partido.get("goles_local")
    score_away = partido.get("goles_visit")
    data = get_match_stats_by_id(
        match_id, local, visita, fecha,
        score_home=score_home, score_away=score_away,
        force=force,
    )
    # Fallback: búsqueda normal por nombre+fecha
    if not data or data.get("error"):
        data = get_match_stats(local, visita, fecha, force=force)

    if not data:
        print(f"    [warn] No se encontraron stats en FotMob")
        return False

    if data.get("error") == "matchfacts_unavailable":
        print(f"    [warn] Partido encontrado pero stats no disponibles aún")
        return False

    if not data.get("players_home"):
        print(f"    [warn] Sin datos de jugadores (¿partido sin ratings?)")
        return False

    paleta = PALETAS_POOL[int(partido.get("jornada", 1)) % len(PALETAS_POOL)]

    try:
        render_ratings(data, "home", data["players_home"], code, paleta)
        render_ratings(data, "away", data["players_away"], code, paleta)
        render_team_stats(data, code, paleta)
        print(f"    [ok] 3 imágenes → output/charts/partidos/{code}_*.png  [{paleta}]")
        return True
    except Exception as e:
        print(f"    [error] {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",  type=int, default=2,
                        help="Días hacia atrás a procesar (default: 2)")
    parser.add_argument("--force", action="store_true",
                        help="Re-generar aunque ya existan las imágenes")
    args = parser.parse_args()

    if not HIST_FILE.exists():
        print(f"ERROR: {HIST_FILE} no encontrado", file=sys.stderr)
        sys.exit(1)

    with open(HIST_FILE, encoding="utf-8") as f:
        historico = json.load(f)

    since = (date.today() - timedelta(days=args.days)).isoformat()
    candidatos = [
        p for p in historico["partidos"]
        if p.get("terminado") and p["fecha"][:10] >= since
    ]

    if not candidatos:
        print(f"  Sin partidos terminados desde {since}")
        sys.exit(0)

    print(f"\n{'═'*56}")
    print(f"  Post-partido — {len(candidatos)} partido(s) desde {since}")
    print(f"{'═'*56}")

    ok = sum(process_match(p, force=args.force) for p in sorted(candidatos, key=lambda x: x["fecha"]))
    print(f"\n  Resultado: {ok}/{len(candidatos)} OK")


if __name__ == "__main__":
    main()
