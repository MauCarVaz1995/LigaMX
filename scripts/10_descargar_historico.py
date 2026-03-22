#!/usr/bin/env python3
"""
10_descargar_historico.py
Descarga histórico de Liga MX desde FotMob (tabla + partidos por torneo).

Uso:
  python 10_descargar_historico.py                     ← sólo lista temporadas
  python 10_descargar_historico.py --download           ← descarga los torneos configurados en TARGETS
  python 10_descargar_historico.py --download --all     ← descarga TODO lo disponible
"""

import sys
import json
import time
import re
import argparse
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LEAGUE_ID   = 230
LEAGUE_SLUG = "liga-mx"
DELAY       = 3   # segundos entre requests

BASE      = Path(__file__).resolve().parent.parent
OUT_DIR   = BASE / "data/raw/historico"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-MX,es;q=0.9",
    "Accept": "application/json, text/plain, */*",
}

# Torneos objetivo (confirmar con el usuario antes de descargar)
# Se completan automáticamente con los IDs reales obtenidos de allAvailableSeasons
# Torneos objetivo con IDs confirmados de FotMob
# season_id: ID de la temporada (año), torneo: Clausura/Apertura
TARGETS = [
    {"nombre": "Clausura 2026",  "season_id": 27048, "torneo": "Clausura", "year": "2025/2026"},
    {"nombre": "Apertura 2025",  "season_id": 27048, "torneo": "Apertura", "year": "2025/2026"},
    {"nombre": "Clausura 2025",  "season_id": 23651, "torneo": "Clausura", "year": "2024/2025"},
    {"nombre": "Apertura 2024",  "season_id": 23651, "torneo": "Apertura", "year": "2024/2025"},
    {"nombre": "Clausura 2024",  "season_id": 20894, "torneo": "Clausura", "year": "2023/2024"},
    {"nombre": "Apertura 2023",  "season_id": 20894, "torneo": "Apertura", "year": "2023/2024"},
]

# Mapa completo de IDs disponibles (para referencia y --all)
ALL_SEASON_IDS = {
    "2025/2026": 27048,
    "2024/2025": 23651,
    "2023/2024": 20894,
    "2022/2023": 17709,
    "2021/2022": 16528,
    "2020/2021": 15328,
    "2019/2020": 14101,
    "2018/2019": 12828,
}

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: OBTENER TEMPORADAS DISPONIBLES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_available_seasons() -> list[dict]:
    """
    Consulta FotMob via __NEXT_DATA__ y extrae allAvailableSeasons + seasonStatLinks.
    Retorna lista normalizada de {nombre, season_id, torneo, year}.
    """
    page_url = f"https://www.fotmob.com/es/leagues/{LEAGUE_ID}/overview/{LEAGUE_SLUG}"
    print(f"Consultando: {page_url}")
    r = requests.get(page_url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        r.text, re.DOTALL
    )
    if not match:
        raise RuntimeError("No se encontró __NEXT_DATA__ en la página.")
    data  = json.loads(match.group(1))
    props = data["props"]["pageProps"]

    # Lista de nombres: ['2025/2026 - Clausura', '2025/2026 - Apertura', ...]
    nombres = props.get("allAvailableSeasons", [])

    # Links con IDs: [{TournamentId, Name (year), RelativePath}, ...]
    stat_links = props.get("stats", {}).get("seasonStatLinks", [])

    # Construir mapa year → season_id
    year_to_id: dict[str, int] = {}
    for lnk in stat_links:
        year  = lnk.get("Name", "")
        sid   = lnk.get("TournamentId")
        if year and sid:
            year_to_id[year] = sid

    seasons = []
    for nombre in nombres:
        # Formato: "2025/2026 - Clausura"
        parts = nombre.split(" - ", 1)
        year   = parts[0].strip() if parts else ""
        torneo = parts[1].strip() if len(parts) > 1 else ""
        sid    = year_to_id.get(year)
        seasons.append({
            "nombre":    nombre,
            "year":      year,
            "torneo":    torneo,
            "season_id": sid,
        })

    return seasons


def print_seasons(seasons: list[dict]):
    """Imprime la tabla de temporadas disponibles."""
    print(f"\n{'─'*65}")
    print(f"  TEMPORADAS DISPONIBLES – Liga MX (ID {LEAGUE_ID})")
    print(f"{'─'*65}")
    print(f"  {'#':>3}  {'Season ID':>10}  {'Year':<14}  {'Torneo':<12}  Nombre completo")
    print(f"  {'─'*3}  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*25}")
    for i, s in enumerate(seasons, 1):
        sid = str(s['season_id']) if s['season_id'] else '???'
        print(
            f"  {i:>3}  {sid:>10}  {s['year']:<14}  {s['torneo']:<12}  {s['nombre']}"
        )
    print(f"{'─'*65}")
    print(f"  Total: {len(seasons)} torneos disponibles\n")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: DESCARGAR DATOS DE UN TORNEO
# ─────────────────────────────────────────────────────────────────────────────

def fetch_next_data(url: str) -> dict:
    """Descarga una página FotMob y extrae el __NEXT_DATA__."""
    print(f"    GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        r.text, re.DOTALL
    )
    if not m:
        raise RuntimeError(f"No se encontró __NEXT_DATA__ en {url}")
    return json.loads(m.group(1))


def _season_param(year: str, torneo: str) -> str:
    return requests.utils.quote(f"{year} - {torneo}")


def fetch_table_for_torneo(season_id, torneo: str, year: str) -> list[dict]:
    """Descarga tabla de posiciones. Intenta página de standings y API."""
    season_q = _season_param(year, torneo)

    # 1) Página standings con filtro de temporada
    urls_try = [
        f"https://www.fotmob.com/es/leagues/{LEAGUE_ID}/standings/{LEAGUE_SLUG}?season={season_q}",
        f"https://www.fotmob.com/es/leagues/{LEAGUE_ID}/overview/{LEAGUE_SLUG}?season={season_q}",
    ]
    for url in urls_try:
        try:
            data  = fetch_next_data(url)
            props = data.get("props", {}).get("pageProps", {})
            tabla = _extract_tabla(props)
            if tabla:
                return tabla
        except Exception as e:
            print(f"     ⚠ {e}")
        time.sleep(1)

    # 2) API de tabla (solo si tenemos season_id numérico)
    if season_id:
        url2 = f"https://www.fotmob.com/api/table?leagueId={LEAGUE_ID}&seasonId={season_id}"
        try:
            print(f"    GET {url2}")
            r = requests.get(url2, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return _extract_tabla(r.json())
        except Exception as e:
            print(f"     ⚠ API tabla: {e}")
    return []


def fetch_matches_for_torneo(season_id, torneo: str, year: str) -> list[dict]:
    """Descarga partidos desde la página de matches con filtro de temporada."""
    season_q = _season_param(year, torneo)
    urls_try = [
        f"https://www.fotmob.com/es/leagues/{LEAGUE_ID}/matches/{LEAGUE_SLUG}?season={season_q}",
        f"https://www.fotmob.com/es/leagues/{LEAGUE_ID}/fixtures/{LEAGUE_SLUG}?season={season_q}",
    ]
    for url in urls_try:
        try:
            data  = fetch_next_data(url)
            props = data.get("props", {}).get("pageProps", {})
            matches = _extract_matches_from_props(props)
            if matches:
                return matches
        except Exception as e:
            print(f"     ⚠ {e}")
        time.sleep(1)
    return []


def _extract_tabla(data: dict) -> list[dict]:
    """Extrae la tabla de posiciones del JSON de FotMob."""
    result = []

    # Desde respuesta de API de tabla
    tables = data.get("tables") or data.get("table") or []
    if isinstance(tables, list) and tables:
        for group in tables:
            sub = group.get("tables") or [group]
            for t in sub:
                rows = (t.get("table") or {}).get("all") or t.get("all") or []
                for row in rows:
                    result.append({
                        "pos":     row.get("idx"),
                        "equipo":  row.get("name"),
                        "equipo_id": row.get("id"),
                        "pj":      row.get("played"),
                        "g":       row.get("wins"),
                        "e":       row.get("draws"),
                        "p":       row.get("losses"),
                        "gf":      _split_score(row.get("scoresStr"), 0),
                        "gc":      _split_score(row.get("scoresStr"), 1),
                        "pts":     row.get("pts"),
                    })
                if result:
                    return result

    # Desde __NEXT_DATA__
    def search_tables(obj, depth=0):
        if depth > 5 or not isinstance(obj, (dict, list)):
            return []
        if isinstance(obj, dict):
            if "table" in obj and "all" in (obj.get("table") or {}):
                rows = obj["table"]["all"]
                return [{
                    "pos":     r.get("idx"),
                    "equipo":  r.get("name"),
                    "equipo_id": r.get("id"),
                    "pj":      r.get("played"),
                    "g":       r.get("wins"),
                    "e":       r.get("draws"),
                    "p":       r.get("losses"),
                    "gf":      _split_score(r.get("scoresStr"), 0),
                    "gc":      _split_score(r.get("scoresStr"), 1),
                    "pts":     r.get("pts"),
                } for r in rows]
            for v in obj.values():
                r = search_tables(v, depth+1)
                if r:
                    return r
        elif isinstance(obj, list):
            for item in obj:
                r = search_tables(item, depth+1)
                if r:
                    return r
        return []

    return search_tables(data)


def _extract_matches_from_props(props: dict) -> list[dict]:
    """
    Extrae partidos del pageProps de FotMob.
    Estructura real: props.fixtures.allMatches (lista plana de dicts de partido).
    """
    # Estructura confirmada: fixtures es un dict con allMatches
    fixtures = props.get("fixtures") or {}
    if isinstance(fixtures, dict):
        all_matches = fixtures.get("allMatches") or []
        if all_matches:
            return [p for p in (_parse_match(m) for m in all_matches) if p]

    # Fallbacks
    for cand in [props.get("matches"), props.get("overview", {}).get("leagueOverviewMatches")]:
        if isinstance(cand, list) and cand:
            result = [p for p in (_parse_match(m) for m in cand) if p]
            if result:
                return result
    return []


def _parse_match(m: dict, ronda=None) -> dict | None:
    """Convierte un dict de partido FotMob a formato normalizado."""
    if not isinstance(m, dict):
        return None
    home   = m.get("home") or {}
    away   = m.get("away") or {}
    status = m.get("status") or {}
    if not home or not away:
        return None

    # Score en status.scoreStr: "0 - 1" o "2-0"
    score_str = status.get("scoreStr") or m.get("score") or m.get("result") or ""
    gf, gc    = _parse_score(str(score_str))
    terminado = status.get("finished", False)

    return {
        "id":           m.get("id") or m.get("matchId"),
        "fecha":        status.get("utcTime") or m.get("utcTime") or m.get("date"),
        "jornada":      ronda or m.get("round") or m.get("roundId"),
        "local":        home.get("name") or home.get("longName"),
        "local_id":     home.get("id"),
        "visitante":    away.get("name") or away.get("longName"),
        "visitante_id": away.get("id"),
        "goles_local":  gf if terminado else None,
        "goles_visit":  gc if terminado else None,
        "score":        score_str,
        "terminado":    terminado,
    }


def _extract_matches(data: dict) -> list[dict]:
    """Extrae partidos del JSON de FotMob."""
    matches = []

    def search_matches(obj, depth=0):
        if depth > 6 or not isinstance(obj, (dict, list)):
            return
        if isinstance(obj, dict):
            # Patrón matches: lista con home/away/score
            if "home" in obj and "away" in obj and ("id" in obj or "matchId" in obj):
                match_id = obj.get("id") or obj.get("matchId")
                home = obj.get("home", {})
                away = obj.get("away", {})
                score = obj.get("score") or obj.get("result") or ""
                # Extraer goles del score "X - Y"
                gf, gc = _parse_score(str(score))
                matches.append({
                    "id":           match_id,
                    "fecha":        obj.get("status", {}).get("utcTime") or obj.get("date") or obj.get("utcTime"),
                    "jornada":      obj.get("round") or obj.get("roundId"),
                    "local":        home.get("name") or home.get("longName"),
                    "local_id":     home.get("id"),
                    "visitante":    away.get("name") or away.get("longName"),
                    "visitante_id": away.get("id"),
                    "goles_local":  gf,
                    "goles_visit":  gc,
                    "score":        score,
                    "terminado":    obj.get("status", {}).get("finished", False),
                })
                return
            for v in obj.values():
                search_matches(v, depth+1)
        elif isinstance(obj, list):
            for item in obj:
                search_matches(item, depth+1)

    search_matches(data)
    return matches


def _split_score(s, idx):
    if not s or "-" not in str(s):
        return None
    parts = str(s).split("-")
    try:
        return int(parts[idx].strip())
    except (ValueError, IndexError):
        return None


def _parse_score(score_str: str):
    """Extrae goles local y visitante de 'X - Y' o 'X-Y'."""
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)', score_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: ORQUESTADOR DE DESCARGA
# ─────────────────────────────────────────────────────────────────────────────

def download_season(target: dict) -> dict:
    """Descarga tabla + partidos para un torneo y retorna el dict completo."""
    sid    = target["season_id"]
    torneo = target["torneo"]
    year   = target["year"]
    nombre = target["nombre"]
    print(f"\n  [{nombre}]  season_id={sid}")

    result = {
        "torneo":    nombre,
        "season_id": sid,
        "torneo_tipo": torneo,
        "year":      year,
        "tabla":     [],
        "partidos":  [],
    }

    # ── Tabla de posiciones ─────────────────────────────────────────────────
    print(f"  → Tabla...")
    result["tabla"] = fetch_table_for_torneo(sid, torneo, year)
    time.sleep(DELAY)
    print(f"     {len(result['tabla'])} equipos")

    # ── Partidos ─────────────────────────────────────────────────────────────
    print(f"  → Partidos...")
    result["partidos"] = fetch_matches_for_torneo(sid, torneo, year)
    time.sleep(DELAY)
    print(f"     {len(result['partidos'])} partidos")

    return result


def save_season(data: dict):
    nombre_archivo = (
        data["torneo"].lower()
        .replace("/", "-").replace(" ", "_").replace("–", "-")
        .replace("é", "e").replace("á", "a").replace("ó", "o")
        .replace("ú", "u").replace("í", "i").replace("ñ", "n")
    )
    out = OUT_DIR / f"historico_{nombre_archivo}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Guardado: {out}  "
          f"({out.stat().st_size/1024:.1f} KB, "
          f"{len(data['tabla'])} equipos, "
          f"{len(data['partidos'])} partidos)")


# ─────────────────────────────────────────────────────────────────────────────
# MATCH DE NOMBRES OBJETIVO
# ─────────────────────────────────────────────────────────────────────────────

def build_all_targets(seasons: list[dict]) -> list[dict]:
    """Construye lista de targets desde TODAS las temporadas disponibles (con o sin season_id)."""
    return [
        {
            "nombre":    s["nombre"],
            "season_id": s["season_id"],   # puede ser None
            "torneo":    s["torneo"],
            "year":      s["year"],
        }
        for s in seasons
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Descargar los torneos configurados en TARGETS_NOMBRES")
    parser.add_argument("--all",      action="store_true",
                        help="Con --download: descargar TODAS las temporadas disponibles")
    parser.add_argument("--force",    action="store_true",
                        help="Re-descargar aunque el archivo ya exista")
    args = parser.parse_args()

    # ── Paso 1: listar temporadas ──────────────────────────────────────────
    print("=" * 65)
    print("  10_descargar_historico.py – Liga MX histórico FotMob")
    print("=" * 65)

    seasons = fetch_available_seasons()
    print_seasons(seasons)

    if not args.download:
        print("  ℹ  Para descargar, corre con --download")
        print("  ℹ  Los torneos objetivo configurados (TARGETS) son:")
        for t in TARGETS:
            print(f"       · {t['nombre']:20s}  season_id={t['season_id']}  torneo={t['torneo']}")
        print()
        return

    # ── Paso 2: seleccionar torneos ────────────────────────────────────────
    if args.all:
        to_download = build_all_targets(seasons)
        print(f"  Modo --all: descargando {len(to_download)} torneos\n")
    else:
        to_download = TARGETS
        print(f"\n  Torneos a descargar ({len(to_download)}):")
        for t in to_download:
            print(f"    · {t['nombre']:20s}  season_id={t['season_id']}  [{t['year']} - {t['torneo']}]")

    if not to_download:
        print("  ⚠ Nada que descargar. Verifica los IDs/nombres.")
        return

    # ── Paso 3: descargar ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Descargando {len(to_download)} torneos (delay={DELAY}s entre requests)")
    print(f"{'─'*65}")

    for i, season in enumerate(to_download, 1):
        # Verificar si ya existe
        nombre_archivo = (
            season["nombre"].lower()
            .replace("/", "-").replace(" ", "_").replace("–", "-")
            .replace("é", "e").replace("á", "a").replace("ó", "o")
            .replace("ú", "u").replace("í", "i").replace("ñ", "n")
        )
        dest = OUT_DIR / f"historico_{nombre_archivo}.json"
        if dest.exists() and dest.stat().st_size > 1000 and not args.force:
            print(f"\n[{i}/{len(to_download)}] SKIP (ya existe): {dest.name}")
            continue

        print(f"\n[{i}/{len(to_download)}]", end="")
        data = download_season(season)
        save_season(data)
        if i < len(to_download):
            time.sleep(DELAY)

    print(f"\n{'─'*65}")
    print(f"  ✓ Completado. Archivos en: {OUT_DIR}")
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()
