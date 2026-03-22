"""
01_get_equipos.py
Obtiene la lista de equipos del Clausura 2026 de Liga MX (league ID 230)
extrayendo el JSON embebido en la página de FotMob (__NEXT_DATA__).
Guarda el resultado en data/raw/equipos_clausura2026.json
"""

import json
import re
from pathlib import Path

import requests

LEAGUE_ID = 230
URL = f"https://www.fotmob.com/es/leagues/{LEAGUE_ID}/overview/liga-mx"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "equipos_clausura2026.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-MX,es;q=0.9",
}


def main():
    print(f"Consultando: {URL}")
    r = requests.get(URL, headers=HEADERS, timeout=15)
    r.raise_for_status()

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        r.text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("No se encontró __NEXT_DATA__ en la página.")

    page_data = json.loads(match.group(1))
    props = page_data["props"]["pageProps"]

    details = props.get("details", {})
    tabla = props.get("table", [])

    equipos = []
    sub_tables = tabla[0].get("data", {}).get("tables", []) if tabla else []
    for entry in sub_tables:
        rows = entry.get("table", {}).get("all", [])
        for row in rows:
            equipos.append({
                "id":     row.get("id"),
                "nombre": row.get("name"),
                "short":  row.get("shortName"),
                "pos":    row.get("idx"),
                "pj":     row.get("played"),
                "g":      row.get("wins"),
                "e":      row.get("draws"),
                "p":      row.get("losses"),
                "gf":     row.get("scoresStr", "").split("-")[0] if row.get("scoresStr") else None,
                "gc":     row.get("scoresStr", "").split("-")[1] if row.get("scoresStr") and "-" in row.get("scoresStr","") else None,
                "pts":    row.get("pts"),
            })

    resultado = {
        "liga":          details.get("name", "Liga MX"),
        "temporada":     details.get("selectedSeason", ""),
        "total_equipos": len(equipos),
        "equipos":       equipos,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"\nGuardado en: {OUTPUT_PATH}")
    print(f"Liga:        {resultado['liga']}")
    print(f"Temporada:   {resultado['temporada']}")
    print(f"Equipos:     {resultado['total_equipos']}\n")

    header = f"{'Pos':>3}  {'ID':>8}  {'Equipo':<28}  {'Short':<6}  {'PJ':>2}  {'G':>2}  {'E':>2}  {'P':>2}  {'PTS':>3}"
    print(header)
    print("-" * len(header))
    for eq in equipos:
        print(
            f"{eq['pos']:>3}  {eq['id']:>8}  {eq['nombre']:<28}  {eq['short']:<6}  "
            f"{eq['pj']:>2}  {eq['g']:>2}  {eq['e']:>2}  {eq['p']:>2}  {eq['pts']:>3}"
        )


if __name__ == "__main__":
    main()
