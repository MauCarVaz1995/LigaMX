"""
02_get_jugadores.py
Lee los IDs de equipos de data/raw/equipos_clausura2026.json, consulta FotMob
para cada uno y descarga las estadísticas de sus jugadores.
Guarda un JSON por equipo en data/raw/jugadores/{id}_{nombre}.json
"""

import json
import re
import time
from pathlib import Path

import requests

EQUIPOS_PATH = Path(__file__).parent.parent / "data" / "raw" / "equipos_clausura2026.json"
OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "raw" / "jugadores"

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


def slugify(nombre: str) -> str:
    return nombre.lower().replace(" ", "-").replace("á","a").replace("é","e") \
                         .replace("í","i").replace("ó","o").replace("ú","u")


def fetch_team_players(team_id: int, team_name: str) -> dict:
    url = f"https://www.fotmob.com/es/teams/{team_id}/squad/team"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    match = NEXT_DATA_RE.search(r.text)
    if not match:
        raise RuntimeError(f"Sin __NEXT_DATA__ para equipo {team_id}")

    page = json.loads(match.group(1))
    team_data = page["props"]["pageProps"]["fallback"][f"team-{team_id}"]

    details = team_data.get("details", {})
    squad_groups = team_data.get("squad", {}).get("squad", [])

    jugadores = []
    for grupo in squad_groups:
        rol = grupo.get("title", "")
        for m in grupo.get("members", []):
            jugadores.append({
                "id":            m.get("id"),
                "nombre":        m.get("name"),
                "camiseta":      m.get("shirtNumber"),
                "posicion":      m.get("positionIdsDesc"),
                "posicion_id":   m.get("positionId"),
                "rol_grupo":     rol,
                "pais":          m.get("cname"),
                "pais_cod":      m.get("ccode"),
                "edad":          m.get("age"),
                "fecha_nac":     m.get("dateOfBirth"),
                "altura_cm":     m.get("height"),
                "rating":        m.get("rating"),
                "goles":         m.get("goals"),
                "asistencias":   m.get("assists"),
                "tarj_amarillas":m.get("ycards"),
                "tarj_rojas":    m.get("rcards"),
                "valor_mercado": m.get("transferValue"),
                "lesion":        m.get("injury"),
            })

    return {
        "equipo_id":   team_id,
        "equipo":      details.get("name", team_name),
        "temporada":   details.get("latestSeason", ""),
        "total":       len(jugadores),
        "jugadores":   jugadores,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EQUIPOS_PATH, encoding="utf-8") as f:
        equipos_data = json.load(f)

    # Deduplica por ID
    vistos = {}
    for eq in equipos_data["equipos"]:
        vistos.setdefault(eq["id"], eq["nombre"])

    equipos = list(vistos.items())
    print(f"Equipos a procesar: {len(equipos)}\n")

    ok, errores = 0, []

    for i, (team_id, team_name) in enumerate(equipos, 1):
        slug = slugify(team_name)
        out_path = OUTPUT_DIR / f"{team_id}_{slug}.json"
        print(f"[{i:02d}/{len(equipos)}] {team_name} (ID {team_id}) ...", end=" ", flush=True)

        try:
            datos = fetch_team_players(team_id, team_name)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(datos, f, ensure_ascii=False, indent=2)

            jugadores = [j for j in datos["jugadores"] if j["rol_grupo"] != "coach"]
            print(f"{len(jugadores)} jugadores  ->  {out_path.name}")
            ok += 1

        except Exception as e:
            print(f"ERROR: {e}")
            errores.append((team_name, str(e)))

        if i < len(equipos):
            time.sleep(1.2)  # cortesía con el servidor

    print(f"\nListo: {ok}/{len(equipos)} equipos descargados.")
    if errores:
        print("Errores:")
        for nombre, err in errores:
            print(f"  - {nombre}: {err}")


if __name__ == "__main__":
    main()
