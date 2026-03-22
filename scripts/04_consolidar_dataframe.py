"""
04_consolidar_dataframe.py
Cruza data/raw/jugadores/ con data/raw/stats_detalladas/ por ID de jugador
y genera un DataFrame maestro con info básica + todas las métricas per90.

Salida:
  data/processed/jugadores_clausura2026.csv
  data/processed/jugadores_clausura2026.pkl
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ──────────────────────────────────────────────────────────────────────
JUGADORES_DIR  = Path("data/raw/jugadores")
STATS_DIR      = Path("data/raw/stats_detalladas")
PROCESSED_DIR  = Path("data/processed")

# ── Mapeo: stat_key de FotMob → nombre de columna en el DataFrame ──────────────
# Formato: (grupo_preferido_o_None, stat_key) → columna
# grupo=None significa que se acepta de cualquier grupo
METRICAS_P90 = [
    # shooting
    ("shooting",    "goals",                    "goles_p90"),
    ("shooting",    "expected_goals",            "xG_p90"),
    ("shooting",    "non_penalty_xg",            "xG_np_p90"),
    ("shooting",    "expected_goals_on_target",  "xGOT_p90"),
    ("shooting",    "shots",                     "tiros_p90"),
    ("shooting",    "ShotsOnTarget",             "tiros_a_puerta_p90"),
    ("shooting",    "headed_shots",              "cabezazos_p90"),
    # passing
    ("passing",     "assists",                   "asistencias_p90"),
    ("passing",     "expected_assists",          "xA_p90"),
    ("passing",     "chances_created",           "chances_creadas_p90"),
    ("passing",     "big_chance_created_team_title", "grandes_chances_p90"),
    ("passing",     "successful_passes",         "pases_precisos_p90"),
    ("passing",     "successful_passes_accuracy","precision_pases_p90"),
    ("passing",     "long_balls_accurate",       "pases_largos_p90"),
    ("passing",     "long_ball_succeeeded_accuracy", "precision_pases_largos_p90"),
    ("passing",     "crosses_succeeeded",        "centros_precisos_p90"),
    ("passing",     "crosses_succeeeded_accuracy","precision_centros_p90"),
    # possession
    ("possession",  "duel_won",                  "duelos_ganados_p90"),
    ("possession",  "duel_won_percent",          "duelos_ganados_pct_p90"),
    ("possession",  "dribbles_succeeded",        "regates_p90"),
    ("possession",  "won_contest_subtitle",      "duelos_tierra_ganados_p90"),
    ("possession",  "aerials_won",               "duelos_aereos_ganados_p90"),
    ("possession",  "aerials_won_percent",       "duelos_aereos_ganados_pct_p90"),
    ("possession",  "touches",                   "toques_p90"),
    ("possession",  "touches_opp_box",           "toques_area_rival_p90"),
    ("possession",  "fouls_won",                 "faltas_recibidas_p90"),
    ("possession",  "dispossessed",              "perdidas_balon_p90"),
    # defending
    ("defending",   "interceptions",             "intercepciones_p90"),
    ("defending",   "clearances",                "despejes_p90"),
    ("defending",   "blocked_shots",             "tiros_bloqueados_p90"),
    ("defending",   "recoveries",                "recuperaciones_p90"),
    ("defending",   "fouls",                     "faltas_cometidas_p90"),
    ("defending",   "dribbled_past",             "regateado_p90"),
    ("defending",   "matchstats.headers.tackles","entradas_p90"),
    ("defending",   "poss_won_att_3rd_team_title","recuperaciones_campo_rival_p90"),
    # discipline
    ("discipline",  "yellow_cards",              "tarjetas_amarillas_p90"),
    ("discipline",  "red_cards",                 "tarjetas_rojas_p90"),
    # goalkeeping (solo porteros)
    ("goalkeeping", "saves",                     "paradas_p90"),
    ("goalkeeping", "save_percentage",           "porcentaje_paradas_p90"),
    ("goalkeeping", "goals_conceded",            "goles_recibidos_p90"),
    ("goalkeeping", "goals_prevented",           "goles_evitados_p90"),
    ("goalkeeping", "keeper_sweeper",            "salidas_p90"),
    ("goalkeeping", "keeper_high_claim",         "balones_al_aire_p90"),
    ("goalkeeping", "clean_sheet_team_title",    "porterias_cero_p90"),
    ("goalkeeping", "penalty_saves",             "penales_atajados_p90"),
    ("goalkeeping", "penalty_save_percent",      "pct_penales_atajados_p90"),
    # distribution (porteros)
    ("distribution","successful_passes",         "dist_pases_precisos_p90"),
    ("distribution","successful_passes_accuracy","dist_precision_pases_p90"),
    ("distribution","long_balls_accurate",       "dist_pases_largos_p90"),
    ("distribution","long_ball_succeeeded_accuracy","dist_precision_pases_largos_p90"),
    ("distribution","expected_assists",          "dist_xA_p90"),
]

# ── Carga jugadores base ────────────────────────────────────────────────────────

def cargar_jugadores_base() -> dict[int, dict]:
    """Retorna {player_id: {campos básicos}} desde data/raw/jugadores/."""
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
                "id":             pid,
                "nombre":         j["nombre"],
                "equipo":         equipo,
                "equipo_id":      equipo_id,
                "posicion":       j.get("posicion") or "",
                "edad":           j.get("edad"),
                "nacionalidad":   j.get("pais") or "",
                "pais_cod":       j.get("pais_cod") or "",
                "altura_cm":      j.get("altura_cm"),
                "camiseta":       j.get("camiseta"),
                "rating":         j.get("rating"),
                "goles":          j.get("goles") or 0,
                "asistencias":    j.get("asistencias") or 0,
                "tarj_amarillas": j.get("tarj_amarillas") or 0,
                "tarj_rojas":     j.get("tarj_rojas") or 0,
                "valor_mercado_eur": j.get("valor_mercado") or 0,
            }
    return jugadores


# ── Extrae valor per90 de un jugador de stats_detalladas ──────────────────────

def extraer_per90(jug_stats: dict, grupo: str, stat_key: str):
    """Busca stat_key en el grupo indicado y devuelve su per90."""
    grupos = jug_stats.get("grupos", {})
    grupo_data = grupos.get(grupo, {})
    stat = grupo_data.get("stats", {}).get(stat_key, {})
    val = stat.get("per90")
    if val is None:
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


# ── Carga stats detalladas y cruza ─────────────────────────────────────────────

def cargar_stats_detalladas(jugadores_base: dict) -> list[dict]:
    """Para cada jugador en stats_detalladas, extrae las métricas per90."""
    registros = []

    for path in sorted(STATS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for jug in data["jugadores"]:
            pid = jug["id"]
            base = jugadores_base.get(pid)
            if base is None:
                continue  # jugador no encontrado en base

            row = dict(base)  # copia campos básicos

            # Torneo de referencia de firstSeasonStats
            tref = jug.get("torneo_ref") or {}
            row["torneo_stats"] = tref.get("tournament", "")
            row["temporada_stats"] = tref.get("season", "")

            # Minutos y partidos del top_stats
            ts = jug.get("top_stats", {})
            row["minutos_stats"]  = ts.get("minutes_played", {}).get("value")
            row["partidos_stats"] = ts.get("matches_uppercase", {}).get("value")

            # Métricas per90
            for grupo, stat_key, col_name in METRICAS_P90:
                row[col_name] = extraer_per90(jug, grupo, stat_key)

            registros.append(row)

    return registros


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando jugadores base (data/raw/jugadores/)...")
    jugadores_base = cargar_jugadores_base()
    print(f"  → {len(jugadores_base)} jugadores únicos")

    print("Cargando y cruzando stats detalladas (data/raw/stats_detalladas/)...")
    registros = cargar_stats_detalladas(jugadores_base)
    print(f"  → {len(registros)} registros cruzados")

    df = pd.DataFrame(registros)

    # Ordenar columnas: primero info básica, luego métricas
    cols_base = [
        "id", "nombre", "equipo", "equipo_id", "posicion", "edad", "nacionalidad",
        "pais_cod", "altura_cm", "camiseta",
        "rating", "goles", "asistencias", "tarj_amarillas", "tarj_rojas",
        "valor_mercado_eur",
        "torneo_stats", "temporada_stats", "minutos_stats", "partidos_stats",
    ]
    cols_p90 = [col for _, _, col in METRICAS_P90]
    cols_orden = cols_base + [c for c in cols_p90 if c in df.columns]
    df = df[cols_orden]

    # Redondear métricas numéricas a 3 decimales
    for col in cols_p90:
        if col in df.columns:
            df[col] = df[col].round(3)

    # Guardar
    csv_path = PROCESSED_DIR / "jugadores_clausura2026.csv"
    pkl_path = PROCESSED_DIR / "jugadores_clausura2026.pkl"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_pickle(pkl_path)

    print(f"\n  CSV  → {csv_path}  ({csv_path.stat().st_size / 1024:.1f} KB)")
    print(f"  PKL  → {pkl_path}  ({pkl_path.stat().st_size / 1024:.1f} KB)")

    # ── Reporte ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Shape: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"{'═'*70}")

    print("\n── Primeras 5 filas (columnas base) ──")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(df[cols_base].head().to_string(index=False))

    print("\n── Lista completa de columnas ──")
    for i, col in enumerate(df.columns, 1):
        nulos = df[col].isna().sum()
        dtype = str(df[col].dtype)
        print(f"  {i:>3}. {col:<40} [{dtype:<8}]  nulos: {nulos}/{len(df)}")

    print(f"\n── Cobertura de métricas per90 ──")
    p90_coverage = {}
    for col in cols_p90:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            p90_coverage[col] = pct
    for col, pct in sorted(p90_coverage.items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 5)
        print(f"  {col:<45} {pct:>5.1f}%  {bar}")


if __name__ == "__main__":
    main()
