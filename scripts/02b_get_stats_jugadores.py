"""
02b_get_stats_jugadores.py
Para cada jugador en data/raw/jugadores/ descarga las estadísticas detalladas
de su página individual en FotMob (firstSeasonStats.statsSection).

Incluye: goles/90, asistencias/90, xG, xA, tiros/90, pases completados,
duelos ganados, intercepciones y cualquier otra métrica disponible.

Guarda un JSON por equipo en data/raw/stats_detalladas/.
Delay de 3 segundos entre requests para evitar bloqueos.

Uso:
    python 02b_get_stats_jugadores.py
    python 02b_get_stats_jugadores.py --equipo toluca   # solo un equipo (debug)
    python 02b_get_stats_jugadores.py --resume          # salta equipos ya procesados
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

# ── Rutas ──────────────────────────────────────────────────────────────────────
JUGADORES_DIR = Path(__file__).parent.parent / "data" / "raw" / "jugadores"
OUTPUT_DIR    = Path(__file__).parent.parent / "data" / "raw" / "stats_detalladas"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-MX,es;q=0.9",
}

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)

DELAY_SECONDS = 3


# ── Extracción de datos ────────────────────────────────────────────────────────

def fetch_player_stats(player_id: int) -> dict:
    """
    Descarga la página del jugador y extrae firstSeasonStats.statsSection.
    Retorna dict con metadatos y stats aplanadas.
    """
    url = f"https://www.fotmob.com/es/players/{player_id}/overview/player"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    match = NEXT_DATA_RE.search(r.text)
    if not match:
        raise RuntimeError("No se encontró __NEXT_DATA__")

    page    = json.loads(match.group(1))
    props   = page["props"]["pageProps"]

    # El nodo de datos puede estar en 'fallback' o directamente en 'data'
    fallback = props.get("fallback", {}) or {}
    pdata    = fallback.get(f"player:{player_id}") or props.get("data") or {}

    if not pdata or not isinstance(pdata, dict):
        raise RuntimeError("Estructura de datos no encontrada o vacía")

    # ── Temporada de referencia ───────────────────────────────────────────────
    stat_seasons = pdata.get("statSeasons", [])
    torneo_ref = None
    if stat_seasons:
        s0 = stat_seasons[0]
        t0 = s0.get("tournaments", [{}])[0]
        torneo_ref = {
            "season":     s0.get("seasonName", ""),
            "tournament": t0.get("name", ""),
            "entry_id":   t0.get("entryId", ""),
        }

    # ── firstSeasonStats ─────────────────────────────────────────────────────
    fss      = pdata.get("firstSeasonStats", {})
    top_card = fss.get("topStatCard", {})
    stat_sec = fss.get("statsSection", {})

    # Resumen del topStatCard (partidos, minutos, rating, etc.)
    top_stats = {}
    for item in top_card.get("items", []):
        key = item.get("localizedTitleId", "")
        top_stats[key] = {
            "value":       _parse_val(item.get("statValue")),
            "per90":       item.get("per90"),
            "percentile":  item.get("percentileRank"),
        }

    # Grupos de statsSection (shooting, passing, defending, etc.)
    grupos = {}
    for grupo in stat_sec.get("items", []):
        grupo_key   = grupo.get("localizedTitleId") or grupo.get("title", "misc")
        grupo_items = {}
        for item in grupo.get("items", []):
            key = item.get("localizedTitleId", "")
            if not key:
                continue
            grupo_items[key] = {
                "title":      item.get("title", ""),
                "value":      _parse_val(item.get("statValue")),
                "per90":      item.get("per90"),
                "percentile": item.get("percentileRank"),
                "format":     item.get("statFormat", ""),
            }
        if grupo_items:
            grupos[grupo_key] = {
                "title":  grupo.get("title", grupo_key),
                "stats":  grupo_items,
            }

    return {
        "torneo_ref": torneo_ref,
        "top_stats":  top_stats,
        "grupos":     grupos,
    }


def _parse_val(raw):
    """Convierte '7.02' o '12' a float/int según corresponda."""
    if raw is None:
        return None
    try:
        f = float(raw)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return raw


# ── Carga de equipos ───────────────────────────────────────────────────────────

def cargar_equipos(filtro: str = None) -> list[dict]:
    """Lee los JSONs de jugadores y retorna lista de equipos con sus jugadores."""
    equipos = []
    for path in sorted(JUGADORES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        nombre = data["equipo"].lower()
        if filtro and filtro.lower() not in nombre:
            continue
        jugadores = [j for j in data["jugadores"] if j["rol_grupo"] != "coach"]
        equipos.append({
            "equipo":    data["equipo"],
            "equipo_id": data["equipo_id"],
            "slug":      path.stem,
            "jugadores": jugadores,
        })
    return equipos


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equipo",  type=str, default=None, help="Filtrar por nombre de equipo")
    parser.add_argument("--resume",  action="store_true",    help="Saltar equipos ya procesados")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    equipos = cargar_equipos(args.equipo)

    total_jug = sum(len(e["jugadores"]) for e in equipos)
    print(f"Equipos: {len(equipos)}  |  Jugadores: {total_jug}")
    print(f"Tiempo estimado: ~{total_jug * DELAY_SECONDS // 60} min {total_jug * DELAY_SECONDS % 60} seg\n")

    jug_num = 0
    for equipo in equipos:
        out_path = OUTPUT_DIR / f"{equipo['slug']}.json"

        if args.resume and out_path.exists():
            print(f"  [SKIP] {equipo['equipo']} (ya procesado)")
            jug_num += len(equipo["jugadores"])
            continue

        print(f"{'─'*60}")
        print(f"  {equipo['equipo']}  ({len(equipo['jugadores'])} jugadores)")
        print(f"{'─'*60}")

        resultados = []
        errores    = []

        for j in equipo["jugadores"]:
            jug_num += 1
            pid    = j["id"]
            nombre = j["nombre"]
            pos    = j.get("posicion", "?")
            print(f"  [{jug_num:>3}/{total_jug}] {nombre:<30} ({pos}) ... ", end="", flush=True)

            try:
                stats = fetch_player_stats(pid)

                # Resumen en consola
                top = stats["top_stats"]
                min_  = top.get("minutes_played", {}).get("value", 0) or 0
                goles = top.get("goals", {}).get("value", 0) or 0
                asis  = top.get("assists", {}).get("value", 0) or 0
                rat   = top.get("rating", {}).get("value", "-")
                torneo = stats["torneo_ref"]["tournament"] if stats["torneo_ref"] else "?"
                print(f"OK  | {min_:>4}' | {goles}G {asis}A | rat={rat} | [{torneo[:20]}]")

                resultados.append({
                    "id":       pid,
                    "nombre":   nombre,
                    "posicion": pos,
                    **stats,
                })

            except Exception as e:
                print(f"ERROR: {e}")
                errores.append({"id": pid, "nombre": nombre, "error": str(e)})

            time.sleep(DELAY_SECONDS)

        # Guardar JSON del equipo
        salida = {
            "equipo":    equipo["equipo"],
            "equipo_id": equipo["equipo_id"],
            "fuente":    "FotMob firstSeasonStats",
            "jugadores": resultados,
            "errores":   errores,
        }
        out_path.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = len(resultados)
        print(f"\n  ✓ {ok} jugadores guardados → {out_path.name}  ({len(errores)} errores)\n")

    print(f"\n{'═'*60}")
    print(f"  Proceso completado. Archivos en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
