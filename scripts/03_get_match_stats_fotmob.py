#!/usr/bin/env python3
"""
03_get_match_stats_fotmob.py
Descarga estadísticas de partido desde FotMob usando el HTML de la página
(datos en __NEXT_DATA__, sin necesidad de la API protegida por Turnstile).

Salida: data/raw/fotmob/{local}_{visitante}_{fecha}.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTA SOBRE IMPORTACIÓN:
Los módulos Python no pueden importarse directamente si empiezan con dígito.
Usa importlib para llamar desde otro script:

    import importlib, sys
    sys.path.insert(0, 'scripts')
    _m = importlib.import_module('03_get_match_stats_fotmob')
    get_match_stats = _m.get_match_stats

    data = get_match_stats('Mexico', 'Portugal', '2026-03-26')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uso standalone:
    python scripts/03_get_match_stats_fotmob.py Mexico Portugal 2026-03-26
    python scripts/03_get_match_stats_fotmob.py Mexico Portugal          # fecha = hoy
"""

import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime, date
from difflib import get_close_matches
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE / "data" / "raw" / "fotmob"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOTMOB   = "https://www.fotmob.com"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fotmob.com/",
}

# ─────────────────────────────────────────────────────────────────────────────
# HTTP session (reutilizable entre llamadas)
# ─────────────────────────────────────────────────────────────────────────────
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
        try:
            _session.get(FOTMOB + "/", timeout=10)  # establece cookies de sesión
        except Exception:
            pass
    return _session


def _get_json(url: str, retries: int = 3, delay: float = 1.5) -> dict | None:
    """GET con reintentos y backoff exponencial."""
    sess = _get_session()
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(delay * (2 ** attempt))
                continue
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (2 ** attempt))
    return None


def _get_html(url: str, retries: int = 3, delay: float = 1.5) -> str | None:
    """GET HTML con reintentos."""
    sess = _get_session()
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=20, headers={"Accept": "text/html"})
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503):
                time.sleep(delay * (2 ** attempt))
                continue
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (2 ** attempt))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA DE PARTIDO
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(name: str) -> str:
    """Lowercase sin acentos para comparar nombres."""
    return name.lower().strip()


def _slug(name: str) -> str:
    """'Mexico' → 'mexico', 'US Virgin Islands' → 'us-virgin-islands'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")


def find_match(local: str, visitante: str, fecha: str) -> dict | None:
    """
    Busca el partido en la lista de matches de FotMob para la fecha dada.
    Devuelve el dict del partido o None si no se encuentra.

    Args:
        local:     nombre del equipo local (en inglés, aproximado)
        visitante: nombre del equipo visitante
        fecha:     'YYYY-MM-DD'
    Returns:
        {
            'match_id': int,
            'home': str, 'away': str,
            'score_home': int|None, 'score_away': int|None,
            'finished': bool,
            'utc_time': str,
            'league': str,
            'league_id': int,
        }
        o None si no se encuentra.
    """
    date_fmt = fecha.replace("-", "")   # 20260326
    url = f"{FOTMOB}/api/data/matches?date={date_fmt}"
    data = _get_json(url)
    if not data:
        print(f"[!] No se pudo obtener partidos para {fecha}", file=sys.stderr)
        return None

    local_n     = _normalize(local)
    visitante_n = _normalize(visitante)

    # Recorrer todas las ligas del día
    all_matches = []
    for league in data.get("leagues", []):
        for m in league.get("matches", []):
            home_name = m.get("home", {}).get("name", "")
            away_name = m.get("away", {}).get("name", "")
            all_matches.append((home_name, away_name, m, league))

    # 1. Búsqueda exacta (case-insensitive)
    for home_name, away_name, m, league in all_matches:
        if _normalize(home_name) == local_n and _normalize(away_name) == visitante_n:
            return _build_match_dict(m, league)

    # 2. Búsqueda fuzzy (difflib) con umbral 0.7
    all_home_names = [_normalize(h) for h, _, _, _ in all_matches]
    all_away_names = [_normalize(a) for _, a, _, _ in all_matches]

    home_candidates = get_close_matches(local_n, all_home_names, n=5, cutoff=0.6)
    away_candidates = get_close_matches(visitante_n, all_away_names, n=5, cutoff=0.6)

    for home_name, away_name, m, league in all_matches:
        h_n = _normalize(home_name)
        a_n = _normalize(away_name)
        if h_n in home_candidates and a_n in away_candidates:
            print(
                f"[~] Fuzzy match: '{home_name}' vs '{away_name}' "
                f"(buscado: '{local}' vs '{visitante}')",
                file=sys.stderr,
            )
            return _build_match_dict(m, league)

    print(
        f"[!] Partido no encontrado: '{local}' vs '{visitante}' ({fecha})",
        file=sys.stderr,
    )
    return None


def _build_match_dict(m: dict, league: dict) -> dict:
    status = m.get("status", {})
    home_score = m.get("home", {}).get("score")
    away_score = m.get("away", {}).get("score")
    return {
        "match_id":   m["id"],
        "home":       m["home"]["name"],
        "away":       m["away"]["name"],
        "home_id":    m["home"]["id"],
        "away_id":    m["away"]["id"],
        "score_home": home_score,
        "score_away": away_score,
        "score_str":  status.get("scoreStr", ""),
        "finished":   status.get("finished", False),
        "utc_time":   status.get("utcTime", ""),
        "league":     league.get("name", ""),
        "league_id":  league.get("id"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE STATS DESDE __NEXT_DATA__
# ─────────────────────────────────────────────────────────────────────────────
def _extract_next_data(html: str) -> dict | None:
    """Extrae el JSON de __NEXT_DATA__ del HTML de la página."""
    m = re.search(
        r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _flatten_player_stats(stats_list: list) -> dict:
    """
    Aplana los grupos de stats de un jugador en un dict plano:
    {'rating': 7.3, 'minutes_played': 90, 'goals': 1, ...}
    """
    flat = {}
    for group in stats_list:
        stats_dict = group.get("stats", {})
        if isinstance(stats_dict, dict):
            for stat_name, stat_data in stats_dict.items():
                stat_info = stat_data.get("stat", {})
                value = stat_info.get("value")
                total = stat_info.get("total")          # p.ej. pases: 22/28
                if value is not None:
                    flat[stat_data.get("key", stat_name)] = (
                        {"value": value, "total": total} if total is not None else value
                    )
    return flat


def _parse_players(starters: list, subs: list, player_stats: dict) -> list:
    """
    Combina lineup + playerStats para cada jugador.
    Devuelve lista de dicts con toda la info disponible.
    """
    players = []
    for p in starters + subs:
        pid = str(p.get("id"))
        ps  = player_stats.get(pid, {})
        flat_stats = _flatten_player_stats(ps.get("stats", []))
        players.append({
            "id":           p.get("id"),
            "name":         p.get("name"),
            "shirt_number": p.get("shirtNumber"),
            "position_id":  p.get("positionId"),
            "usual_position": p.get("usualPlayingPositionId"),
            "is_starter":   p in starters,
            "country":      p.get("countryName"),
            "country_code": p.get("countryCode"),
            "club":         p.get("primaryTeamName"),
            "age":          p.get("age"),
            "market_value": p.get("marketValue"),
            "rating":       p.get("performance", {}).get("rating"),
            "stats":        flat_stats,
        })
    return players


def _flatten_team_stats(stats_payload: dict) -> dict:
    """
    Aplana las stats de equipo de la sección 'stats' del content.
    Devuelve {'ball_possession': [33, 67], 'total_shots': [7, 10], ...}
    Índice 0 = local, índice 1 = visitante.
    """
    flat = {}
    periods = stats_payload.get("Periods", {})
    all_period = periods.get("All", {})
    for group in all_period.get("stats", []):
        for stat in group.get("stats", []):
            key   = stat.get("key", stat.get("title", "?")).lower().replace(" ", "_")
            value = stat.get("stats")    # [home_val, away_val]
            if value is not None:
                flat[key] = value
    return flat


def fetch_match_details(match_info: dict) -> dict | None:
    """
    Descarga la página HTML del partido y extrae __NEXT_DATA__.
    Devuelve el bloque 'content' del pageProps o None en caso de error.
    """
    mid  = match_info["match_id"]
    home = match_info["home"]
    away = match_info["away"]
    slug = f"{_slug(home)}-vs-{_slug(away)}"

    url = f"{FOTMOB}/match/{mid}/matchfacts/{slug}"
    print(f"  → Descargando: {url}", file=sys.stderr)

    html = _get_html(url)
    if not html:
        print(f"[!] No se pudo descargar la página del partido {mid}", file=sys.stderr)
        return None

    nd = _extract_next_data(html)
    if not nd:
        print(f"[!] __NEXT_DATA__ no encontrado en {url}", file=sys.stderr)
        return None

    return nd.get("props", {}).get("pageProps", {}).get("content")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def get_match_stats(
    equipo_local: str,
    equipo_visitante: str,
    fecha: str | None = None,
    force: bool = False,
) -> dict | None:
    """
    Busca el partido en FotMob, extrae estadísticas y guarda el JSON.

    Args:
        equipo_local:     Nombre del equipo local en inglés (aprox.)
        equipo_visitante: Nombre del equipo visitante en inglés (aprox.)
        fecha:            'YYYY-MM-DD'. Si es None, usa la fecha de hoy.
        force:            Si True, re-descarga aunque el JSON ya exista.

    Returns:
        dict con las estadísticas completas, o None si el partido no se encuentra.

    Ejemplo:
        data = get_match_stats('Mexico', 'Portugal', '2026-03-29')
    """
    if fecha is None:
        fecha = date.today().isoformat()

    # ── Nombre de archivo de caché ─────────────────────────────────────────
    local_safe     = re.sub(r"[^\w]", "_", equipo_local)
    visitante_safe = re.sub(r"[^\w]", "_", equipo_visitante)
    cache_path     = OUT_DIR / f"{local_safe}_{visitante_safe}_{fecha}.json"

    if cache_path.exists() and not force:
        print(f"[cache] {cache_path.name}", file=sys.stderr)
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    # ── Paso 1: buscar el partido ──────────────────────────────────────────
    print(f"[1/3] Buscando partido: {equipo_local} vs {equipo_visitante} ({fecha})")
    match_info = find_match(equipo_local, equipo_visitante, fecha)
    if not match_info:
        return None

    print(
        f"      Encontrado: {match_info['home']} vs {match_info['away']} "
        f"[ID {match_info['match_id']}] — {match_info['league']}"
    )

    # ── Paso 2: descargar detalles ─────────────────────────────────────────
    print(f"[2/3] Descargando estadísticas...")
    content = fetch_match_details(match_info)
    if not content:
        # Guardar solo la info básica disponible sin stats
        result = {**match_info, "players_home": [], "players_away": [],
                  "team_stats": {}, "events": [], "error": "matchfacts_unavailable"}
        _save(result, cache_path)
        return result

    # ── Paso 3: parsear y estructurar ──────────────────────────────────────
    print(f"[3/3] Procesando datos...")
    lineup       = content.get("lineup") or {}
    player_stats = content.get("playerStats") or {}
    team_stats   = content.get("stats") or {}
    match_facts  = content.get("matchFacts") or {}

    home_lineup  = lineup.get("homeTeam", {})
    away_lineup  = lineup.get("awayTeam", {})

    players_home = _parse_players(
        home_lineup.get("starters", []),
        home_lineup.get("subs",     []),
        player_stats,
    )
    players_away = _parse_players(
        away_lineup.get("starters", []),
        away_lineup.get("subs",     []),
        player_stats,
    )

    # Eventos del partido (goles, tarjetas, sustituciones)
    events = match_facts.get("events", [])

    # Player of the match
    potm = match_facts.get("playerOfTheMatch")

    result = {
        # Metadata del partido
        "match_id":     match_info["match_id"],
        "date":         fecha,
        "home_team":    match_info["home"],
        "away_team":    match_info["away"],
        "home_team_id": match_info["home_id"],
        "away_team_id": match_info["away_id"],
        "score_home":   match_info["score_home"],
        "score_away":   match_info["score_away"],
        "score_str":    match_info["score_str"],
        "finished":     match_info["finished"],
        "utc_time":     match_info["utc_time"],
        "league":       match_info["league"],
        "league_id":    match_info["league_id"],
        # Formaciones
        "formation_home": home_lineup.get("formation"),
        "formation_away": away_lineup.get("formation"),
        # Calificación media alineación
        "avg_rating_home": home_lineup.get("rating"),
        "avg_rating_away": away_lineup.get("rating"),
        # Player of the match
        "player_of_the_match": potm,
        # Stats de equipo (aplanadas)
        "team_stats": _flatten_team_stats(team_stats),
        # Jugadores
        "players_home": players_home,
        "players_away": players_away,
        # Eventos
        "events": events,
        # Metadato de extracción
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "source": "fotmob/__NEXT_DATA__",
    }

    _save(result, cache_path)
    return result


def get_match_stats_by_id(
    match_id: int,
    local: str,
    visitante: str,
    fecha: str,
    score_home: int = None,
    score_away: int = None,
    force: bool = False,
) -> dict | None:
    """
    Igual que get_match_stats pero usando el match_id directamente.
    Útil cuando ya tenemos el ID del historico y el endpoint de fechas no responde.
    """
    local_safe     = re.sub(r"[^\w]", "_", local)
    visitante_safe = re.sub(r"[^\w]", "_", visitante)
    cache_path     = OUT_DIR / f"{local_safe}_{visitante_safe}_{fecha}.json"

    if cache_path.exists() and not force:
        print(f"[cache] {cache_path.name}", file=sys.stderr)
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    score_str = f"{score_home}-{score_away}" if score_home is not None else ""

    match_info = {
        "match_id":   match_id,
        "home":       local,
        "away":       visitante,
        "home_id":    None,
        "away_id":    None,
        "score_home": score_home,
        "score_away": score_away,
        "score_str":  score_str,
        "finished":   True,
        "utc_time":   fecha,
        "league":     "Liga MX",
        "league_id":  230,
    }

    print(f"[1/3] Usando match_id={match_id} directamente")
    content = fetch_match_details(match_info)
    if not content:
        result = {**match_info, "players_home": [], "players_away": [],
                  "team_stats": {}, "events": [], "error": "matchfacts_unavailable"}
        _save(result, cache_path)
        return result

    print("[2/3] Procesando datos...")
    lineup       = content.get("lineup") or {}
    player_stats = content.get("playerStats") or {}
    team_stats   = content.get("stats") or {}
    match_facts  = content.get("matchFacts") or {}

    home_lineup  = lineup.get("homeTeam", {})
    away_lineup  = lineup.get("awayTeam", {})

    players_home = _parse_players(
        home_lineup.get("starters", []),
        home_lineup.get("subs",     []),
        player_stats,
    )
    players_away = _parse_players(
        away_lineup.get("starters", []),
        away_lineup.get("subs",     []),
        player_stats,
    )

    result = {
        "match_id":          match_id,
        "date":              fecha,
        "home_team":         local,
        "away_team":         visitante,
        "home_team_id":      match_info["home_id"],
        "away_team_id":      match_info["away_id"],
        "score_home":        score_home,
        "score_away":        score_away,
        "score_str":         score_str,
        "finished":          True,
        "utc_time":          fecha,
        "league":            "Liga MX",
        "league_id":         230,
        "formation_home":    home_lineup.get("formation"),
        "formation_away":    away_lineup.get("formation"),
        "avg_rating_home":   home_lineup.get("rating"),
        "avg_rating_away":   away_lineup.get("rating"),
        "player_of_the_match": match_facts.get("playerOfTheMatch"),
        "team_stats":        _flatten_team_stats(team_stats),
        "players_home":      players_home,
        "players_away":      players_away,
        "events":            match_facts.get("events", []),
        "fetched_at":        datetime.utcnow().isoformat() + "Z",
        "source":            "fotmob/__NEXT_DATA__/by_id",
    }

    _save(result, cache_path)
    return result


def _save(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] Guardado: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(data: dict) -> None:
    if not data:
        print("Sin datos.")
        return
    print(f"\n{'═'*58}")
    print(f"  {data['home_team']} {data['score_home']} - {data['score_away']} {data['away_team']}")
    print(f"  {data['date']}  ·  {data['league']}  ·  ID {data['match_id']}")
    print(f"{'═'*58}")

    ts = data.get("team_stats", {})
    if ts:
        print("\nEstadísticas de equipo (local / visitante):")
        for k, v in list(ts.items())[:12]:
            if isinstance(v, list) and len(v) == 2 and v[0] is not None and v[1] is not None:
                print(f"  {k:<30} {str(v[0]):>8} / {v[1]}")

    for side, key in [("Local", "players_home"), ("Visitante", "players_away")]:
        players = data.get(key, [])
        starters = [p for p in players if p.get("is_starter")]
        print(f"\n{side} — {data['home_team' if key=='players_home' else 'away_team']}")
        print(f"  {'#':>3}  {'Jugador':<25}  {'Rating':>6}  {'Min':>4}  {'Goles':>5}")
        print(f"  {'─'*3}  {'─'*25}  {'─'*6}  {'─'*4}  {'─'*5}")
        for p in starters:
            s = p.get("stats", {})
            rating = p.get("rating", s.get("rating_title", "-"))
            mins   = s.get("minutes_played", "-")
            goals  = s.get("goals", "-")
            print(
                f"  {p['shirt_number'] or '-':>3}  {p['name']:<25}  "
                f"{str(rating):>6}  {str(mins):>4}  {str(goals):>5}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT STANDALONE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Descarga stats de partido de FotMob"
    )
    parser.add_argument("local",      help="Nombre equipo local (inglés)")
    parser.add_argument("visitante",  help="Nombre equipo visitante (inglés)")
    parser.add_argument(
        "fecha", nargs="?", default=None,
        help="Fecha YYYY-MM-DD (default: hoy)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-descargar aunque el JSON ya exista"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Imprimir JSON completo en stdout"
    )
    args = parser.parse_args()

    data = get_match_stats(args.local, args.visitante, args.fecha, force=args.force)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print_summary(data)


if __name__ == "__main__":
    main()
