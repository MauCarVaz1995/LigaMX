"""
02c_get_stats_liga.py
Descarga estadísticas de jugadores de Liga MX Clausura 2026 directamente
desde data.fotmob.com (fuente oficial, stats por torneo específico).

Ventaja sobre 02b: las stats son únicamente de Liga MX Clausura,
no mezcla con Champions Cup, Apertura u otros torneos.

Salida:
  data/processed/stats_clausura2026.csv   ← stats per90 de Liga MX Clausura
  data/processed/jugadores_clausura2026.csv ← reemplaza el anterior (datos completos)
  data/processed/jugadores_clausura2026.pkl
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Configuración ───────────────────────────────────────────────────────────────
LIGA_ID       = 230
SEASON_ID     = 27048
TORNEO_SLUG   = "Clausura"
BASE_URL      = f"https://data.fotmob.com/stats/{LIGA_ID}/season/{SEASON_ID}/{TORNEO_SLUG}"

JUGADORES_DIR = Path("data/raw/jugadores")
PROCESSED_DIR = Path("data/processed")
CACHE_DIR     = Path(".cache/stats_liga")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"}

DELAY = 0.5  # segundos entre requests

# ── Mapeo: (archivo_json, col_df, ya_es_per90)
# ya_es_per90=True  → StatValue se usa directo
# ya_es_per90=False → StatValue es total; per90 = StatValue / MinutesPlayed * 90
STATS_MAP = [
    # ─ Goles / xG ─────────────────────────────────────────────────────────────
    ("goals_per_90",              "goles_p90",                    True),
    ("expected_goals_per_90",     "xG_p90",                       True),
    ("total_scoring_att",         "tiros_p90",                    True),
    ("ontarget_scoring_att",      "tiros_a_puerta_p90",           True),
    # ─ Pases / creación ────────────────────────────────────────────────────────
    ("goal_assist",               "asistencias_p90",              False),
    ("expected_assists_per_90",   "xA_p90",                       True),
    ("total_att_assist",          "chances_creadas_p90",          True),
    ("big_chance_created",        "grandes_chances_p90",          True),
    ("accurate_pass",             "pases_precisos_p90",           True),
    ("accurate_long_balls",       "pases_largos_p90",             True),
    # ─ Duelos / posesión ───────────────────────────────────────────────────────
    ("won_contest",               "duelos_tierra_ganados_p90",    True),
    ("poss_won_att_3rd",          "recuperaciones_campo_rival_p90", True),
    # ─ Defensa ─────────────────────────────────────────────────────────────────
    ("total_tackle",              "entradas_p90",                 True),
    ("interception",              "intercepciones_p90",           True),
    ("effective_clearance",       "despejes_p90",                 True),
    ("outfielder_block",          "tiros_bloqueados_p90",         True),
    # ─ Disciplina ──────────────────────────────────────────────────────────────
    ("fouls",                     "faltas_cometidas_p90",         True),
    ("yellow_card",               "tarjetas_amarillas_p90",       False),
    ("red_card",                  "tarjetas_rojas_p90",           False),
    # ─ Portería ────────────────────────────────────────────────────────────────
    ("saves",                     "paradas_p90",                  True),
    ("_save_percentage",          "porcentaje_paradas_p90",       True),
    ("_goals_prevented",          "goles_evitados_p90",           True),
    ("goals_conceded",            "goles_recibidos_p90",          True),
    ("clean_sheet",               "porterias_cero_p90",           False),
    ("penalty_won",               "penales_ganados_p90",          True),
]

# Stats de resumen (no per90)
SUMMARY_STATS = [
    ("mins_played", "minutos_stats", "partidos_stats"),
    ("rating",      "rating",        None),
    ("goals",       "goles",         None),
    ("goal_assist", "asistencias",   None),
]


# ── Descarga de JSONs ───────────────────────────────────────────────────────────

def fetch_stat_json(stat_name: str) -> dict | None:
    """Descarga un JSON de stat desde data.fotmob.com con caché local."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{stat_name}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    url = f"{BASE_URL}/{stat_name}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        time.sleep(DELAY)
        return data
    except Exception as e:
        print(f"  [WARN] {stat_name}.json → {e}")
        return None


def extract_stat_list(data: dict) -> list[dict]:
    """Extrae la StatList del primer TopList."""
    try:
        return data["TopLists"][0]["StatList"]
    except (KeyError, IndexError, TypeError):
        return []


# ── Construcción del DataFrame de stats ────────────────────────────────────────

def build_stats_df() -> pd.DataFrame:
    """Descarga todas las stats y construye un DataFrame por jugador."""
    rows: dict[int, dict] = {}  # player_id → {col: value}

    def upsert(pid: int, key: str, val):
        if pid not in rows:
            rows[pid] = {"id": pid}
        rows[pid][key] = val

    # ── Minutos y partidos ─────────────────────────────────────────────────────
    print("  mins_played.json ...", end=" ", flush=True)
    mins_data = fetch_stat_json("mins_played")
    if mins_data:
        for s in extract_stat_list(mins_data):
            pid = s["ParticiantId"]
            upsert(pid, "minutos_stats",  s["StatValue"])
            upsert(pid, "partidos_stats", s.get("MatchesPlayed") or s.get("SubStatValue"))
            upsert(pid, "_min_ref",       s["StatValue"])  # referencia para calcular per90
        print(f"OK ({len(rows)} jugadores)")

    # ── Rating ────────────────────────────────────────────────────────────────
    print("  rating.json ...", end=" ", flush=True)
    rat_data = fetch_stat_json("rating")
    if rat_data:
        for s in extract_stat_list(rat_data):
            upsert(s["ParticiantId"], "rating", s["StatValue"])
        print("OK")

    # ── Goles y asistencias totales ────────────────────────────────────────────
    for stat_file, col in [("goals", "goles"), ("goal_assist", "asistencias")]:
        print(f"  {stat_file}.json ...", end=" ", flush=True)
        d = fetch_stat_json(stat_file)
        if d:
            for s in extract_stat_list(d):
                upsert(s["ParticiantId"], col, int(s["StatValue"]))
                # Actualizar minutos si el jugador no apareció en mins_played
                pid = s["ParticiantId"]
                if "_min_ref" not in rows.get(pid, {}):
                    upsert(pid, "_min_ref", s["MinutesPlayed"])
                    upsert(pid, "minutos_stats", s["MinutesPlayed"])
            print("OK")

    # ── Métricas per90 ────────────────────────────────────────────────────────
    for stat_file, col_name, ya_per90 in STATS_MAP:
        print(f"  {stat_file}.json ...", end=" ", flush=True)
        d = fetch_stat_json(stat_file)
        if not d:
            print("SKIP")
            continue
        count = 0
        for s in extract_stat_list(d):
            pid  = s["ParticiantId"]
            minp = s.get("MinutesPlayed") or rows.get(pid, {}).get("_min_ref") or 0
            val  = s["StatValue"]
            if ya_per90:
                p90 = float(val)
            else:
                p90 = round(float(val) / minp * 90, 3) if minp > 0 else np.nan
            upsert(pid, col_name, p90)
            # Actualizar minutos si no estaban
            if "_min_ref" not in rows.get(pid, {}):
                upsert(pid, "_min_ref", minp)
                upsert(pid, "minutos_stats", minp)
            count += 1
        print(f"OK ({count})")

    # Limpiar columna auxiliar
    for r in rows.values():
        r.pop("_min_ref", None)

    return pd.DataFrame(list(rows.values()))


# ── Carga de datos base de jugadores ───────────────────────────────────────────

def cargar_jugadores_base() -> dict[int, dict]:
    jugadores = {}
    for path in sorted(JUGADORES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        equipo    = data["equipo"]
        equipo_id = data.get("equipo_id")
        for j in data["jugadores"]:
            if j["rol_grupo"] == "coach":
                continue
            pid = j["id"]
            jugadores[pid] = {
                "id":               pid,
                "nombre":           j["nombre"],
                "equipo":           equipo,
                "equipo_id":        equipo_id,
                "posicion":         j.get("posicion") or "",
                "edad":             j.get("edad"),
                "nacionalidad":     j.get("pais") or "",
                "pais_cod":         j.get("pais_cod") or "",
                "altura_cm":        j.get("altura_cm"),
                "camiseta":         j.get("camiseta"),
                "valor_mercado_eur":j.get("valor_mercado") or 0,
            }
    return jugadores


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Descargando stats de Liga MX Clausura {TORNEO_SLUG} (temporada {SEASON_ID})...")
    df_stats = build_stats_df()
    print(f"\n  → {len(df_stats)} jugadores con stats de Liga MX Clausura")

    print("\nCargando datos base de jugadores...")
    base = cargar_jugadores_base()
    print(f"  → {len(base)} jugadores en data/raw/jugadores/")

    # Merge: stats de liga + info base
    df_base = pd.DataFrame(list(base.values()))
    df = df_base.merge(df_stats, on="id", how="left")

    # Consolidar columnas de goles/asistencias: preferir las de Liga MX Clausura
    for col_liga, col_base in [("goles", "goles"), ("asistencias", "asistencias"), ("rating", "rating")]:
        if col_liga in df.columns:
            # ya está mergeado
            pass

    # Ordenar columnas
    cols_base = [
        "id", "nombre", "equipo", "equipo_id", "posicion", "edad",
        "nacionalidad", "pais_cod", "altura_cm", "camiseta",
        "valor_mercado_eur",
        "rating", "goles", "asistencias",
        "minutos_stats", "partidos_stats",
    ]
    cols_p90 = [col for _, col, _ in STATS_MAP]
    cols_disponibles = [c for c in cols_base + cols_p90 if c in df.columns]
    df = df[cols_disponibles]

    # Redondear per90
    for col in cols_p90:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    # Guardar
    csv_path = PROCESSED_DIR / "jugadores_clausura2026.csv"
    pkl_path = PROCESSED_DIR / "jugadores_clausura2026.pkl"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_pickle(pkl_path)

    print(f"\n  CSV → {csv_path}  ({csv_path.stat().st_size / 1024:.1f} KB)")
    print(f"  PKL → {pkl_path}  ({pkl_path.stat().st_size / 1024:.1f} KB)")
    print(f"\n  Shape: {df.shape[0]} filas × {df.shape[1]} columnas")

    # Reporte de cobertura
    print(f"\n── Cobertura de minutos en Liga MX Clausura ──")
    tiene_min = df["minutos_stats"].notna() & (df["minutos_stats"] > 0)
    print(f"  Jugadores con minutos en Clausura: {tiene_min.sum()} / {len(df)}")
    print(f"  Jugadores sin minutos (no jugaron): {(~tiene_min).sum()}")

    print(f"\n── Distribución de minutos ──")
    min_series = df["minutos_stats"].dropna()
    for umbral in [90, 180, 270, 300, 450, 630]:
        n = (min_series >= umbral).sum()
        print(f"  >= {umbral:>4} min: {n:>3} jugadores")


if __name__ == "__main__":
    main()
