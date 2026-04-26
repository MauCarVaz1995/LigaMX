#!/usr/bin/env python3
"""
ligas_internacionales.py — Bot de predicción para ligas internacionales de clubes
==================================================================================
Obtiene fixtures de FotMob, mantiene ELO por liga, genera predicciones usando el
modelo ELO+Poisson con corrección Dixon-Coles y registra resultados en predicciones_log.csv.

Uso:
  python3 scripts/ligas_internacionales.py --liga champions
  python3 scripts/ligas_internacionales.py --all
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE             = Path(__file__).resolve().parent.parent
ELOS_DIR         = BASE / "data/processed/elos_ligas"
PRED_LOG         = BASE / "data/processed/predicciones_log.csv"
INTL_CSV         = BASE / "data/raw/internacional/results.csv"
ODDS_HISTORICO   = BASE / "data/processed/odds_historico.csv"
BETTING_LOG      = BASE / "data/processed/betting_log.csv"
REPORTS_DIR      = BASE / "output/reports"

ELOS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# LIGAS
# ─────────────────────────────────────────────────────────────────────────────
LIGAS: dict[str, dict] = {
    # Copas / torneos de clubes
    "champions":    {"id": 42,    "nombre": "UEFA Champions League", "espn": "UEFA.CHAMPIONS"},
    "libertadores": {"id": 384,   "nombre": "Copa Libertadores",     "espn": "CONMEBOL.LIBERTADORES"},
    "mls":          {"id": 913550,"nombre": "MLS",                   "espn": "usa.1"},
    "argentina":    {"id": 905256,"nombre": "Liga Profesional Argentina", "espn": "arg.1"},
    "brasileirao":  {"id": 268,   "nombre": "Brasileirao Serie A",   "espn": "bra.1"},
    # Ligas europeas (ELO desde odds_historico.csv — 5 años reales)
    "premier":      {"id": 47,    "nombre": "Premier League",  "odds_key": "premier",    "espn": "eng.1"},
    "laliga":       {"id": 87,    "nombre": "La Liga",          "odds_key": "laliga",     "espn": "esp.1"},
    "bundesliga":   {"id": 54,    "nombre": "Bundesliga",       "odds_key": "bundesliga", "espn": "ger.1"},
    "seriea":       {"id": 55,    "nombre": "Serie A",          "odds_key": "seriea",     "espn": "ita.1"},
    "ligue1":       {"id": 53,    "nombre": "Ligue 1",          "odds_key": "ligue1",     "espn": "fra.1"},
}

# Ligas cuyo ELO se bootstrapea desde odds_historico.csv (no desde results.csv)
LIGAS_ODDS_BOOTSTRAP = {"premier", "laliga", "bundesliga", "seriea", "ligue1"}

# Para Champions: ligas domésticas desde las que se toma el ELO de cada club
LIGAS_EUROPEAS_KEYS = ["premier", "laliga", "bundesliga", "seriea", "ligue1"]

# Palabras clave en results.csv para mapear a cada liga (solo las que tienen datos reales)
LIGA_TOURNAMENT_KEYWORDS: dict[str, list[str]] = {
    "libertadores": ["copa libertadores", "libertadores"],
    "mls":          ["major league soccer", "mls"],
    "argentina":    ["primera division", "liga profesional", "superliga argentina"],
    "brasileirao":  ["brasileirao", "campeonato brasileiro"],
}

# ─────────────────────────────────────────────────────────────────────────────
# MODELO PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────
HOME_ADV_ELO = 80    # ELO bonus para local (menor que Liga MX=100)
MU_HOME      = 1.5   # media global goles local
MU_AWAY      = 1.1   # media global goles visitante
ELO_BASE     = 1500.0
ELO_INIT     = 1500.0
K_ELO        = 32
SCALE        = 400
DC_RHO       = -0.20  # Dixon-Coles rho calibrado de datos reales Liga MX (1-1 ratio=1.22)

# ─────────────────────────────────────────────────────────────────────────────
# FOTMOB HEADERS
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.fotmob.com/",
}

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE NOMBRES
# ─────────────────────────────────────────────────────────────────────────────
# Alias conocidos entre FotMob y results.csv
TEAM_NAME_ALIASES: dict[str, str] = {
    # Champions
    "Man City":        "Manchester City",
    "Man Utd":         "Manchester United",
    "Atleti":          "Atletico Madrid",
    "Atletico Madrid": "Atletico de Madrid",
    "Atletico de Madrid": "Atletico de Madrid",
    "PSG":             "Paris Saint-Germain",
    "Paris SG":        "Paris Saint-Germain",
    "Inter":           "Inter Milan",
    "Internazionale":  "Inter Milan",
    "RB Leipzig":      "RB Leipzig",
    "Club Brugge":     "Club Brugge",
    # Libertadores
    "Flamengo":        "Flamengo",
    "River Plate":     "River Plate",
    "Boca Juniors":    "Boca Juniors",
    "Fluminense":      "Fluminense",
    # MLS
    "LA Galaxy":       "LA Galaxy",
    "Atlanta Utd":     "Atlanta United",
    "Atlanta United FC": "Atlanta United",
    # Argentina
    "Racing Club":     "Racing Club",
    "San Lorenzo":     "San Lorenzo",
    # Brasileirao
    "Atlético Mineiro": "Atletico Mineiro",
    "Atletico-MG":     "Atletico Mineiro",
}


def norm_team(name: str) -> str:
    """Normaliza nombre de equipo eliminando variantes conocidas."""
    s = str(name).strip()
    return TEAM_NAME_ALIASES.get(s, s)


# ─────────────────────────────────────────────────────────────────────────────
# ELO HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def elo_csv_path(liga_key: str) -> Path:
    return ELOS_DIR / f"{liga_key}.csv"


def load_elos(liga_key: str) -> dict[str, float]:
    """Carga ELOs más recientes para una liga. Devuelve dict equipo→elo."""
    p = elo_csv_path(liga_key)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        if df.empty:
            return {}
        # Último registro por equipo
        df = df.sort_values("fecha")
        return df.groupby("equipo")["elo"].last().to_dict()
    except Exception as e:
        print(f"  [WARN] load_elos({liga_key}): {e}")
        return {}


def save_elos_atomic(liga_key: str, registros: list[dict]):
    """Agrega registros al CSV de ELO de forma atómica (write temp + rename)."""
    p = elo_csv_path(liga_key)
    if registros:
        df_new = pd.DataFrame(registros)
        if p.exists():
            df_old = pd.read_csv(p)
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_out = df_new
        # Escritura atómica
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(ELOS_DIR), suffix=".tmp")
        try:
            df_out.to_csv(tmp_path, index=False)
            os.close(tmp_fd)
            os.replace(tmp_path, str(p))
        except Exception:
            os.close(tmp_fd)
            os.unlink(tmp_path)
            raise


def elo_expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / SCALE))


def update_elo(elo_team: float, elo_opp: float, result: float) -> float:
    """result: 1=win, 0.5=draw, 0=loss. Retorna nuevo ELO."""
    expected = elo_expected(elo_team, elo_opp)
    return elo_team + K_ELO * (result - expected)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP ELO DESDE HISTORICO
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_elo_from_history(liga_key: str) -> dict[str, float]:
    """
    Calcula ELO desde cero usando results.csv histórico.
    Aplica time-decay: partidos más recientes tienen más peso (K efectivo mayor).
    Guarda el resultado en el CSV de la liga.
    """
    print(f"  [bootstrap] Calculando ELO histórico para {liga_key}...")
    keywords = LIGA_TOURNAMENT_KEYWORDS.get(liga_key, [])
    if not keywords:
        print(f"  [WARN] Sin keywords para liga {liga_key}")
        return {}

    try:
        df = pd.read_csv(INTL_CSV, parse_dates=["date"])
    except Exception as e:
        print(f"  [WARN] No se pudo leer {INTL_CSV}: {e}")
        return {}

    # Filtrar por torneo
    mask = df["tournament"].str.lower().apply(
        lambda t: any(kw in str(t).lower() for kw in keywords)
    )
    df_liga = df[mask].copy().sort_values("date")
    if df_liga.empty:
        print(f"  [WARN] Sin datos históricos para {liga_key}")
        return {}

    print(f"  [bootstrap] {liga_key}: {len(df_liga)} partidos históricos encontrados")

    # Time-decay: K aumenta para partidos más recientes
    # Normalizar fechas: 0 (más antiguo) → 1 (más reciente)
    dates = pd.to_datetime(df_liga["date"])
    min_date = dates.min()
    max_date = dates.max()
    date_range = (max_date - min_date).days or 1

    elos: dict[str, float] = {}
    registros: list[dict] = []

    for _, row in df_liga.iterrows():
        home = norm_team(str(row["home_team"]))
        away = norm_team(str(row["away_team"]))
        hs   = int(row["home_score"]) if pd.notna(row["home_score"]) else None
        as_  = int(row["away_score"]) if pd.notna(row["away_score"]) else None
        if hs is None or as_ is None:
            continue

        fecha_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]

        # Time-decay factor: 0.5 → 1.5 (más reciente = K más alto)
        age_ratio = (row["date"] - min_date).days / date_range
        k_eff = K_ELO * (0.5 + age_ratio)

        elo_h = elos.get(home, ELO_INIT)
        elo_a = elos.get(away, ELO_INIT)

        neutral = bool(row.get("neutral", False))
        adv = 0 if neutral else HOME_ADV_ELO

        exp_h = elo_expected(elo_h + adv, elo_a)
        res_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        res_a = 1.0 - res_h

        elos[home] = elo_h + k_eff * (res_h - exp_h)
        elos[away] = elo_a + k_eff * (res_a - (1.0 - exp_h))

        registros.append({"equipo": home, "elo": round(elos[home], 2), "fecha": fecha_str, "liga": liga_key})
        registros.append({"equipo": away, "elo": round(elos[away], 2), "fecha": fecha_str, "liga": liga_key})

    if registros:
        save_elos_atomic(liga_key, registros)
        print(f"  [bootstrap] {liga_key}: {len(elos)} equipos inicializados")

    return elos


def bootstrap_elo_from_odds_historico(liga_key: str) -> dict[str, float]:
    """
    Bootstrap ELO para ligas europeas usando odds_historico.csv (5 años reales).
    Mucho más preciso que results.csv para clubes.
    """
    odds_key = LIGAS.get(liga_key, {}).get("odds_key", liga_key)
    print(f"  [bootstrap] ELO desde odds_historico para {liga_key} (key={odds_key})...")

    if not ODDS_HISTORICO.exists():
        print(f"  [WARN] odds_historico.csv no encontrado")
        return {}

    try:
        df = pd.read_csv(ODDS_HISTORICO, parse_dates=["fecha"])
    except Exception as e:
        print(f"  [WARN] Error leyendo odds_historico: {e}")
        return {}

    df_liga = df[df["liga"] == odds_key].copy().sort_values("fecha")
    if df_liga.empty:
        print(f"  [WARN] Sin datos para liga {odds_key} en odds_historico")
        return {}

    df_liga = df_liga[df_liga["goles_local"].notna() & df_liga["goles_visita"].notna()]
    print(f"  [bootstrap] {liga_key}: {len(df_liga)} partidos en odds_historico")

    dates    = df_liga["fecha"]
    min_date = dates.min()
    max_date = dates.max()
    date_range = max(1, (max_date - min_date).days)

    elos: dict[str, float] = {}
    registros: list[dict] = []

    for _, row in df_liga.iterrows():
        home = norm_team(str(row["local"]))
        away = norm_team(str(row["visitante"]))
        hs   = int(row["goles_local"])
        as_  = int(row["goles_visita"])

        fecha_str = row["fecha"].strftime("%Y-%m-%d") if hasattr(row["fecha"], "strftime") else str(row["fecha"])[:10]
        age_ratio = (row["fecha"] - min_date).days / date_range
        k_eff     = K_ELO * (0.5 + age_ratio)

        elo_h = elos.get(home, ELO_INIT)
        elo_a = elos.get(away, ELO_INIT)

        exp_h = elo_expected(elo_h + HOME_ADV_ELO, elo_a)
        res_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)

        elos[home] = elo_h + k_eff * (res_h - exp_h)
        elos[away] = elo_a + k_eff * ((1 - res_h) - (1 - exp_h))

        registros.append({"equipo": home, "elo": round(elos[home], 2), "fecha": fecha_str, "liga": liga_key})
        registros.append({"equipo": away, "elo": round(elos[away], 2), "fecha": fecha_str, "liga": liga_key})

    if registros:
        save_elos_atomic(liga_key, registros)
        elo_vals = list(elos.values())
        print(f"  [bootstrap] {liga_key}: {len(elos)} equipos | ELO {min(elo_vals):.0f}–{max(elo_vals):.0f}")

    return elos


# Cache de ELOs europeos para lookup Champions/Libertadores
_EUROPEAN_ELO_CACHE: dict[str, float] = {}

def load_all_european_elos() -> dict[str, float]:
    """
    Carga ELOs de todas las ligas europeas en un solo dict para
    usarlos en Champions League (lookup cross-liga).
    """
    global _EUROPEAN_ELO_CACHE
    if _EUROPEAN_ELO_CACHE:
        return _EUROPEAN_ELO_CACHE

    combined: dict[str, float] = {}
    for key in LIGAS_EUROPEAS_KEYS:
        p = elo_csv_path(key)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            # Solo el ELO más reciente por equipo
            latest = df.sort_values("fecha").groupby("equipo")["elo"].last()
            for team, elo in latest.items():
                # Si el equipo ya está, tomar el mayor (mejor estimación)
                if team not in combined or float(elo) > combined[team]:
                    combined[team] = float(elo)
        except Exception:
            pass

    _EUROPEAN_ELO_CACHE = combined
    print(f"  [cache] ELOs europeos cargados: {len(combined)} equipos")
    return combined


def get_or_bootstrap_elos(liga_key: str) -> dict[str, float]:
    """
    Si el CSV ya existe, lo carga.
    Si no, ejecuta bootstrap desde la fuente correcta:
      - Ligas europeas (premier/laliga/bundesliga/seriea/ligue1): desde odds_historico.csv
      - Champions: desde ELOs europeos ya cargados (cross-liga)
      - Resto: desde results.csv de selecciones (fallback)
    """
    p = elo_csv_path(liga_key)
    if p.exists() and p.stat().st_size > 100:
        return load_elos(liga_key)

    if liga_key in LIGAS_ODDS_BOOTSTRAP:
        return bootstrap_elo_from_odds_historico(liga_key)

    if liga_key == "champions":
        # Para Champions, primero asegurar que las ligas europeas estén bootstrapeadas
        eu_elos = load_all_european_elos()
        if eu_elos:
            print(f"  [champions] Usando ELOs de ligas europeas ({len(eu_elos)} clubes)")
            # Guardar el cross-elo como base para Champions
            registros = [{"equipo": t, "elo": round(e, 2), "fecha": TODAY, "liga": "champions"}
                         for t, e in eu_elos.items()]
            save_elos_atomic("champions", registros)
            return eu_elos

    return bootstrap_elo_from_history(liga_key)


# ─────────────────────────────────────────────────────────────────────────────
# MODELO ELO + POISSON + DIXON-COLES
# ─────────────────────────────────────────────────────────────────────────────
def dixon_coles_correction(home_goals: int, away_goals: int,
                           lam_l: float, lam_v: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam_l * lam_v * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + lam_l * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + lam_v * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def lambdas_from_elo(elo_local: float, elo_visitante: float) -> tuple[float, float]:
    """
    Convierte ELOs a λ_local y λ_visitante para el modelo Poisson.

    Fórmula aditiva: la diferencia de ELO determina cuántos goles separa a los equipos.
    Escala calibrada: 300 ELO points ≈ 1 gol de diferencia esperado.
    Preserva el total de goles esperado (MU_HOME + MU_AWAY = 2.6) como suma constante.

    Ejemplos:
      ELOs iguales + home adv 80:     lam_l=1.43, lam_v=1.17  → modo 1-0 o 1-1
      ELO diff=200 (home+adv=280):    lam_l=1.77, lam_v=0.83  → modo 1-0
      ELO diff=-200 (away stronger):  lam_l=1.10, lam_v=1.50  → modo 1-1 o 0-1
    """
    mu_total   = MU_HOME + MU_AWAY          # 2.6 goles totales esperados
    elo_diff   = (elo_local + HOME_ADV_ELO) - elo_visitante
    goal_diff  = elo_diff / 300.0           # calibración: 300 ELO ≈ 1 gol de diferencia
    lam_l = (mu_total + goal_diff) / 2.0
    lam_v = (mu_total - goal_diff) / 2.0
    return max(lam_l, 0.20), max(lam_v, 0.20)


def compute_probs(lam_l: float, lam_v: float,
                  max_goals: int = 6, rho: float = DC_RHO) -> dict:
    """
    Calcula probabilidades de resultado usando Poisson + corrección Dixon-Coles.
    Retorna dict con prob_local, prob_empate, prob_visitante, marcador más probable.
    """
    n = max_goals + 1
    mat = np.zeros((n, n))
    for gl in range(n):
        for gv in range(n):
            dc = dixon_coles_correction(gl, gv, lam_l, lam_v, rho)
            mat[gl, gv] = poisson.pmf(gl, lam_l) * poisson.pmf(gv, lam_v) * dc

    # Normalizar
    total = mat.sum()
    if total > 0:
        mat /= total

    p_local = float(np.tril(mat, -1).sum())
    p_empate = float(np.trace(mat))
    p_visita = float(np.triu(mat, 1).sum())

    # Marcador más probable
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    marcador = f"{idx[0]}-{idx[1]}"

    return {
        "prob_local":    round(p_local * 100, 1),
        "prob_empate":   round(p_empate * 100, 1),
        "prob_visitante": round(p_visita * 100, 1),
        "marcador":      marcador,
        "lambda_local":  round(lam_l, 4),
        "lambda_visitante": round(lam_v, 4),
    }


def ganador_predicho(prob_local: float, prob_empate: float, prob_visitante: float) -> str:
    m = max(prob_local, prob_empate, prob_visitante)
    if m == prob_local:
        return "local"
    elif m == prob_empate:
        return "empate"
    return "visitante"


# ─────────────────────────────────────────────────────────────────────────────
# ESPN API — fuente principal de fixtures (sin API key, cobre todas las ligas)
# ─────────────────────────────────────────────────────────────────────────────
ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}/scoreboard"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LigaMXBot/1.0)"}


def fetch_espn(liga_key: str, days_ahead: int = 10) -> list[dict]:
    """
    Obtiene fixtures próximos de una liga via ESPN API (sin API key).
    Retorna lista de dicts con: home, away, fecha, match_id.
    """
    espn_code = LIGAS.get(liga_key, {}).get("espn")
    if not espn_code:
        return []

    fixtures = []
    today = date.today()

    # ESPN /scoreboard incluye partidos del día actual y recientes.
    # Para cubrir días futuros hacemos requests por fecha.
    for delta in range(days_ahead):
        d = (today + timedelta(days=delta)).strftime("%Y%m%d")
        url = f"{ESPN_BASE.format(espn_code)}?dates={d}"
        try:
            r = requests.get(url, headers=ESPN_HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [WARN] ESPN {espn_code} fecha {d}: {e}")
            time.sleep(1)
            continue

        for event in data.get("events", []):
            try:
                comps = event.get("competitions", [{}])[0]
                status = comps.get("status", {}).get("type", {})
                # Solo partidos no terminados
                if status.get("completed", False):
                    continue
                competitors = comps.get("competitors", [])
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue
                home_name = norm_team(home["team"]["displayName"])
                away_name = norm_team(away["team"]["displayName"])
                fecha_str  = event.get("date", "")[:10]
                match_id   = event.get("id", "")
                fixtures.append({
                    "home":     home_name,
                    "away":     away_name,
                    "fecha":    fecha_str,
                    "match_id": match_id,
                })
            except (KeyError, TypeError, IndexError):
                continue
        time.sleep(0.3)

    # Deduplicar por match_id
    seen = set()
    unique = []
    for f in fixtures:
        key = f.get("match_id") or f"{f['home']}_{f['away']}_{f['fecha']}"
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# FOTMOB API — fallback (leagues endpoint da 404 desde abril 2026)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fotmob_by_date(league_id: int, days_ahead: int = 7) -> list[dict]:
    """
    Obtiene fixtures próximos de una liga usando el endpoint /api/data/matches?date=
    (el endpoint /api/leagues?id= da 404 en la versión actual de FotMob).
    Retorna lista de raw match dicts del JSON de FotMob.
    """
    all_matches = []
    today = date.today()

    for delta in range(days_ahead):
        d = (today + timedelta(days=delta)).strftime("%Y%m%d")
        url = f"https://www.fotmob.com/api/data/matches?date={d}"
        try:
            r = requests.get(url, headers=FOTMOB_HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [WARN] FotMob fecha {d}: {e}")
            time.sleep(2)
            continue

        for league in data.get("leagues", []):
            if league.get("id") == league_id:
                for m in league.get("matches", []):
                    if not m.get("status", {}).get("finished", True):
                        all_matches.append({**m, "_date": d})
        time.sleep(0.5)

    return all_matches


def fetch_fotmob_from_cache(league_id: int, max_age_days: int = 3) -> list[dict]:
    """
    Lee fixtures del intl JSON más reciente en caché.
    Útil cuando FotMob está caído o rate-limited.
    """
    raw_dir = BASE / "data/raw/fotmob"
    cutoff  = (date.today() - timedelta(days=max_age_days)).isoformat().replace("-", "")
    cached  = sorted(raw_dir.glob("intl_*.json"), reverse=True)

    for f in cached:
        # intl_YYYYMMDD.json
        name = f.stem.replace("intl_", "")
        if name < cutoff:
            break
        try:
            data = json.loads(f.read_text())
            matches = []
            for league in data.get("leagues", []):
                if league.get("id") == league_id:
                    for m in league.get("matches", []):
                        if not m.get("status", {}).get("finished", True):
                            matches.append(m)
            if matches:
                print(f"  [cache] {len(matches)} fixtures desde {f.name}")
                return matches
        except Exception:
            continue
    return []


def fetch_fotmob(league_id: int) -> dict | None:
    """
    Obtiene fixtures próximos de una liga.
    Intenta primero el API live; si falla, usa caché local.
    """
    matches = fetch_fotmob_by_date(league_id, days_ahead=10)
    if not matches:
        matches = fetch_fotmob_from_cache(league_id)
    if not matches:
        return None
    return {"_raw_matches": matches, "league_id": league_id}


def parse_fixtures(data: dict, liga_key: str) -> list[dict]:
    """
    Extrae fixtures próximos desde la respuesta de FotMob.
    Soporta el nuevo formato (_raw_matches) y el formato antiguo (matches dict).
    Retorna lista de dicts con: home, away, fecha, match_id.
    """
    fixtures = []
    try:
        # Nuevo formato: _raw_matches (lista directa de matches)
        if "_raw_matches" in data:
            match_list = data["_raw_matches"]
            for m in match_list:
                try:
                    home = norm_team(m["home"]["name"])
                    away = norm_team(m["away"]["name"])
                    utc  = m.get("status", {}).get("utcTime", "")
                    fecha = utc[:10] if utc else m.get("_date", TODAY)[:4] + "-" + m.get("_date", "")[4:6] + "-" + m.get("_date", "")[6:8]
                    fixtures.append({
                        "home":     home,
                        "away":     away,
                        "fecha":    fecha,
                        "match_id": m.get("id"),
                    })
                except (KeyError, TypeError):
                    continue
            return fixtures

        # Formato antiguo (para compatibilidad)
        # Estructura FotMob: data["matches"]["upcoming"] o data["matches"]["allMatches"]
        matches_section = data.get("matches", {})

        # Intentar diferentes claves según versión de la API
        match_list = (
            matches_section.get("upcoming")
            or matches_section.get("nextMatches")
            or []
        )

        # Si no hay upcoming, buscar en allMatches los no terminados
        if not match_list:
            all_matches = matches_section.get("allMatches", [])
            match_list = [
                m for m in all_matches
                if not m.get("status", {}).get("finished", False)
                and not m.get("status", {}).get("cancelled", False)
            ]

        for m in match_list:
            home_info = m.get("home") or {}
            away_info = m.get("away") or {}
            home_name = home_info.get("name") or home_info.get("longName", "")
            away_name = away_info.get("name") or away_info.get("longName", "")
            if not home_name or not away_name:
                continue

            # Fecha del partido
            status = m.get("status", {})
            utc_time = status.get("utcTime") or m.get("utcTime", "")
            if utc_time:
                try:
                    # Formato: "2026-04-20T20:00:00.000Z"
                    fecha = utc_time[:10]
                except Exception:
                    fecha = TODAY
            else:
                fecha = TODAY

            fixtures.append({
                "home":     norm_team(home_name),
                "away":     norm_team(away_name),
                "fecha":    fecha,
                "match_id": m.get("id", ""),
                "liga":     liga_key,
            })

    except Exception as e:
        print(f"  [WARN] parse_fixtures({liga_key}): {e}")

    return fixtures


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIONES
# ─────────────────────────────────────────────────────────────────────────────
def load_pred_log() -> pd.DataFrame:
    """Carga el tracker de predicciones. Agrega columna 'liga' si no existe."""
    if not PRED_LOG.exists():
        cols = [
            "fecha_prediccion", "partido", "equipo_local", "equipo_visitante",
            "elo_local", "elo_visitante", "prob_local", "prob_empate", "prob_visitante",
            "ganador_predicho", "marcador_mas_probable", "lambda_local", "lambda_visitante",
            "paleta_usada", "fecha_partido", "resultado_real", "goles_local_real",
            "goles_visitante_real", "acierto_ganador", "error_marcador", "liga",
        ]
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(PRED_LOG, dtype=str)
    if "liga" not in df.columns:
        df["liga"] = ""
    return df


def save_pred_log(df: pd.DataFrame):
    """Guarda el tracker de predicciones."""
    df.to_csv(PRED_LOG, index=False)


def already_predicted(df: pd.DataFrame, local: str, visitante: str, fecha: str) -> bool:
    """Verifica si ya existe una predicción para este partido."""
    mask = (
        (df["equipo_local"] == local) &
        (df["equipo_visitante"] == visitante) &
        (df["fecha_partido"] == fecha)
    )
    return bool(mask.any())


def generate_prediction(
    liga_key: str,
    local: str,
    visitante: str,
    fecha: str,
    elo_local: float,
    elo_visitante: float,
) -> dict:
    """Genera predicción completa para un partido."""
    lam_l, lam_v = lambdas_from_elo(elo_local, elo_visitante)
    probs = compute_probs(lam_l, lam_v)

    winner = ganador_predicho(
        probs["prob_local"],
        probs["prob_empate"],
        probs["prob_visitante"],
    )

    return {
        "fecha_prediccion":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "partido":             f"{local} vs {visitante}",
        "equipo_local":        local,
        "equipo_visitante":    visitante,
        "elo_local":           round(elo_local, 1),
        "elo_visitante":       round(elo_visitante, 1),
        "prob_local":          probs["prob_local"],
        "prob_empate":         probs["prob_empate"],
        "prob_visitante":      probs["prob_visitante"],
        "ganador_predicho":    winner,
        "marcador_mas_probable": probs["marcador"],
        "lambda_local":        probs["lambda_local"],
        "lambda_visitante":    probs["lambda_visitante"],
        "paleta_usada":        "ligas_internacionales",
        "fecha_partido":       fecha,
        "resultado_real":      None,
        "goles_local_real":    None,
        "goles_visitante_real": None,
        "acierto_ganador":     None,
        "error_marcador":      None,
        "liga":                liga_key,
    }


def generate_betting_dict(
    liga_key: str,
    liga_nombre: str,
    local: str,
    visitante: str,
    fecha: str,
    elo_local: float,
    elo_visitante: float,
) -> dict:
    """
    Genera un dict de predicción de apuestas (btts/goles) en el mismo formato
    que daily_betting_bot.py — para que betting_tracker.py lo pueda loguear.
    No incluye corners ni tarjetas (sin datos para ligas internacionales).
    """
    lam_l, lam_v = lambdas_from_elo(elo_local, elo_visitante)

    # Cálculo Poisson de probs para btts y goles
    import math as _math
    def _poisson_pmf(k: int, lam: float) -> float:
        return (_math.exp(-lam) * (lam ** k)) / _math.factorial(k)

    # P(BTTS) = P(local >= 1) * P(visitante >= 1)
    p_l0 = _poisson_pmf(0, lam_l)
    p_v0 = _poisson_pmf(0, lam_v)
    p_btts = (1 - p_l0) * (1 - p_v0)

    # P(Over 1.5) = 1 - P(0 goles) - P(1 gol total)
    # P(Over 2.5) = 1 - P(0) - P(1) - P(2)
    lam_total = lam_l + lam_v
    p_over_1_5 = 1 - _poisson_pmf(0, lam_total) - _poisson_pmf(1, lam_total)
    p_over_2_5 = p_over_1_5 - _poisson_pmf(2, lam_total)

    return {
        "match_id":  None,
        "fecha":     fecha,
        "jornada":   0,
        "local":     local,
        "visita":    visitante,
        "torneo":    liga_nombre,
        "liga":      liga_key,
        "elo_local":   round(elo_local, 1),
        "elo_visita":  round(elo_visitante, 1),
        "corners":  {},    # sin datos para ligas internacionales
        "tarjetas": {},    # sin datos para ligas internacionales
        "btts": {
            "btts_si":      round(p_btts, 4),
            "btts_no":      round(1 - p_btts, 4),
            "over_1.5":     round(p_over_1_5, 4),
            "over_2.5":     round(p_over_2_5, 4),
            "under_2.5":    round(1 - p_over_2_5, 4),
            "lambda_local":   round(lam_l, 3),
            "lambda_visita":  round(lam_v, 3),
        },
        "value_bets": [],
        "errors":    [],
    }


def save_betting_json_intl(partidos: list[dict]):
    """
    Guarda betting_intl_{fecha}.json con predicciones de ligas internacionales.
    El betting_tracker.py lo leerá para loguear btts/goles en betting_log.csv.
    """
    if not partidos:
        return None
    today_compact = TODAY.replace("-", "")
    path = REPORTS_DIR / f"betting_intl_{today_compact}.json"
    path.write_text(json.dumps(partidos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [betting_intl] {len(partidos)} partidos → {path.name}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ACTUALIZAR RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
def build_results_index() -> dict[tuple, tuple]:
    """Construye índice (fecha, home, away) → (goles_l, goles_v) desde results.csv."""
    if not INTL_CSV.exists():
        return {}
    try:
        df = pd.read_csv(INTL_CSV)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        idx = {}
        for _, r in df.iterrows():
            key = (str(r["date"]), norm_team(str(r["home_team"])), norm_team(str(r["away_team"])))
            idx[key] = (int(r["home_score"]), int(r["away_score"]))
        return idx
    except Exception as e:
        print(f"  [WARN] build_results_index: {e}")
        return {}


def update_results(df: pd.DataFrame, results_idx: dict) -> tuple[pd.DataFrame, int]:
    """
    Completa predicciones pendientes con resultados reales.
    Retorna (df_actualizado, n_actualizados).
    """
    pending = df[df["resultado_real"].isna()].index.tolist()
    updated = 0

    for idx in pending:
        row   = df.loc[idx]
        fecha = str(row.get("fecha_partido", ""))[:10]
        local = norm_team(str(row.get("equipo_local", "")))
        visit = norm_team(str(row.get("equipo_visitante", "")))

        # Buscar exacto
        score = results_idx.get((fecha, local, visit))

        # Tolerancia ±1 día
        if score is None:
            try:
                d = datetime.strptime(fecha, "%Y-%m-%d")
                for delta in [-1, 1]:
                    alt = (d + timedelta(days=delta)).strftime("%Y-%m-%d")
                    score = results_idx.get((alt, local, visit))
                    if score is not None:
                        break
            except ValueError:
                pass

        if score is None:
            continue

        gl, gv = score
        resultado = "local" if gl > gv else ("empate" if gl == gv else "visitante")
        pred = str(row.get("ganador_predicho", "")).lower().strip()
        acierto = (pred == resultado)

        df.at[idx, "resultado_real"]        = resultado
        df.at[idx, "goles_local_real"]      = gl
        df.at[idx, "goles_visitante_real"]  = gv
        df.at[idx, "acierto_ganador"]       = acierto
        updated += 1

    return df, updated


# ─────────────────────────────────────────────────────────────────────────────
# ELO UPDATE DESPUÉS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
def update_elo_from_results(liga_key: str, results_idx: dict):
    """
    Actualiza el ELO de la liga con partidos nuevos del results.csv
    que no estén ya reflejados (fecha posterior al último registro en el CSV).
    """
    p = elo_csv_path(liga_key)
    if not p.exists():
        return  # bootstrap no ha corrido aún

    try:
        df_elo = pd.read_csv(p)
        if df_elo.empty:
            return
        ultima_fecha = df_elo["fecha"].max()
    except Exception:
        return

    keywords = LIGA_TOURNAMENT_KEYWORDS.get(liga_key, [])
    if not keywords:
        return

    try:
        df_intl = pd.read_csv(INTL_CSV, parse_dates=["date"])
    except Exception:
        return

    mask = df_intl["tournament"].str.lower().apply(
        lambda t: any(kw in str(t).lower() for kw in keywords)
    )
    df_liga = df_intl[mask].copy().sort_values("date")
    df_liga = df_liga[df_liga["date"].dt.strftime("%Y-%m-%d") > ultima_fecha]

    if df_liga.empty:
        return

    elos = load_elos(liga_key)
    registros: list[dict] = []

    for _, row in df_liga.iterrows():
        home = norm_team(str(row["home_team"]))
        away = norm_team(str(row["away_team"]))
        hs   = int(row["home_score"]) if pd.notna(row["home_score"]) else None
        as_  = int(row["away_score"]) if pd.notna(row["away_score"]) else None
        if hs is None or as_ is None:
            continue

        fecha_str = row["date"].strftime("%Y-%m-%d")
        neutral   = bool(row.get("neutral", False))
        adv       = 0 if neutral else HOME_ADV_ELO

        elo_h = elos.get(home, ELO_INIT)
        elo_a = elos.get(away, ELO_INIT)

        exp_h = elo_expected(elo_h + adv, elo_a)
        res_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        res_a = 1.0 - res_h

        elos[home] = elo_h + K_ELO * (res_h - exp_h)
        elos[away] = elo_a + K_ELO * (res_a - (1.0 - exp_h))

        registros.append({"equipo": home, "elo": round(elos[home], 2), "fecha": fecha_str, "liga": liga_key})
        registros.append({"equipo": away, "elo": round(elos[away], 2), "fecha": fecha_str, "liga": liga_key})

    if registros:
        save_elos_atomic(liga_key, registros)
        print(f"  [{liga_key}] ELO actualizado con {len(registros)//2} partidos nuevos")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE POR LIGA
# ─────────────────────────────────────────────────────────────────────────────
def run_liga(liga_key: str, df_log: pd.DataFrame, results_idx: dict) -> dict:
    """
    Ejecuta el pipeline completo para una liga.
    Retorna dict con resumen de la liga para el JSON de reporte.
    """
    liga_info = LIGAS[liga_key]
    league_id = liga_info["id"]
    nombre    = liga_info["nombre"]

    print(f"\n[{liga_key}] {nombre} (id={league_id})")

    # 1. Cargar o bootstrapear ELOs
    elos = get_or_bootstrap_elos(liga_key)

    # 2. Actualizar ELOs con resultados recientes
    update_elo_from_results(liga_key, results_idx)
    elos = load_elos(liga_key)  # recargar tras actualización

    # 3. Obtener fixtures — ESPN primero, FotMob como fallback
    fixtures = fetch_espn(liga_key, days_ahead=10)
    if fixtures:
        print(f"  [{liga_key}] {len(fixtures)} fixtures desde ESPN")
    else:
        print(f"  [WARN] ESPN sin fixtures para {liga_key}, intentando FotMob...")
        data = fetch_fotmob(league_id)
        if data is not None:
            fixtures = parse_fixtures(data, liga_key)
        else:
            print(f"  [{liga_key}] Sin fixtures disponibles (ESPN y FotMob fallaron)")

    n_fixtures = len(fixtures)

    # 4. Generar predicciones (1X2 para predicciones_log + btts/goles para betting_log)
    nuevas_preds:   list[dict] = []
    betting_dicts:  list[dict] = []
    picks:          list[dict] = []

    for fix in fixtures:
        local     = fix["home"]
        visitante = fix["away"]
        fecha     = fix["fecha"]

        # Para Champions: buscar ELO en ligas europeas si no está en el CSV de champions
        elo_l = elos.get(local, ELO_INIT)
        elo_v = elos.get(visitante, ELO_INIT)

        if liga_key == "champions" and (elo_l == ELO_INIT or elo_v == ELO_INIT):
            eu = load_all_european_elos()
            elo_l = eu.get(local, elo_l)
            elo_v = eu.get(visitante, elo_v)

        # Predicción 1X2 → predicciones_log.csv
        if not already_predicted(df_log, local, visitante, fecha):
            pred = generate_prediction(liga_key, local, visitante, fecha, elo_l, elo_v)
            nuevas_preds.append(pred)
            picks.append({
                "partido":       pred["partido"],
                "fecha":         fecha,
                "ganador":       pred["ganador_predicho"],
                "prob_local":    pred["prob_local"],
                "prob_empate":   pred["prob_empate"],
                "prob_visitante":pred["prob_visitante"],
                "marcador":      pred["marcador_mas_probable"],
                "elo_local":     pred["elo_local"],
                "elo_visitante": pred["elo_visitante"],
            })

        # Predicción btts/goles → betting_log.csv (siempre, no deduplicar aquí)
        bet = generate_betting_dict(liga_key, nombre, local, visitante, fecha, elo_l, elo_v)
        betting_dicts.append(bet)

    # Agregar nuevas predicciones al log 1X2
    if nuevas_preds:
        df_new = pd.DataFrame(nuevas_preds)
        df_log = pd.concat([df_log, df_new], ignore_index=True)

    n_pred = len(nuevas_preds)
    print(f"  [{liga_key}] {n_fixtures} fixtures | {n_pred} preds 1X2 | {len(betting_dicts)} betting dicts")

    return {
        "liga_key":      liga_key,
        "nombre":        nombre,
        "n_fixtures":    n_fixtures,
        "n_predicciones":n_pred,
        "picks":         picks,
        "betting_dicts": betting_dicts,
        "df_log":        df_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bot de predicción para ligas internacionales de clubes"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--liga",
        choices=list(LIGAS.keys()),
        help="Ejecutar para una liga específica",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Ejecutar para todas las ligas",
    )
    args = parser.parse_args()

    ligas_a_correr = list(LIGAS.keys()) if args.all else [args.liga]

    print("=" * 60)
    print(f"ligas_internacionales.py — {TODAY}")
    print(f"Ligas: {', '.join(ligas_a_correr)}")
    print("=" * 60)

    # Cargar tracker y construir índice de resultados una sola vez
    df_log = load_pred_log()
    results_idx = build_results_index()

    # Actualizar resultados pendientes antes de generar nuevas predicciones
    df_log, updated = update_results(df_log, results_idx)
    if updated > 0:
        print(f"\n[tracker] {updated} predicciones actualizadas con resultados reales")
        save_pred_log(df_log)
        df_log = load_pred_log()  # recargar

    # Para Champions: precargar ELOs europeos una sola vez
    if "champions" in ligas_a_correr or any(k in LIGAS_ODDS_BOOTSTRAP for k in ligas_a_correr):
        for eu_key in LIGAS_EUROPEAS_KEYS:
            if not elo_csv_path(eu_key).exists():
                get_or_bootstrap_elos(eu_key)
        load_all_european_elos()

    # Ejecutar pipeline por liga
    reporte: dict = {
        "generated_at": datetime.now().isoformat(),
        "ligas": {},
    }
    all_betting_dicts: list[dict] = []

    for liga_key in ligas_a_correr:
        try:
            result  = run_liga(liga_key, df_log, results_idx)
            df_log  = result.pop("df_log")
            betting = result.pop("betting_dicts", [])
            all_betting_dicts.extend(betting)
            reporte["ligas"][liga_key] = {
                "nombre":     result["nombre"],
                "n_fixtures": result["n_fixtures"],
                "picks":      result["picks"],
            }
        except Exception as e:
            print(f"  [ERROR] Liga {liga_key} falló: {e}")
            reporte["ligas"][liga_key] = {"error": str(e)}

    # Guardar betting JSON para que betting_tracker.py lo procese
    save_betting_json_intl(all_betting_dicts)

    # Guardar tracker final
    save_pred_log(df_log)
    print(f"\n[tracker] Predicciones guardadas en {PRED_LOG}")

    # Guardar JSON de reporte
    today_compact = TODAY.replace("-", "")
    report_path = REPORTS_DIR / f"ligas_internacionales_{today_compact}.json"
    report_path.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[reporte] {report_path}")

    # Resumen final
    print("\n── RESUMEN ──────────────────────────────────────────")
    total_fixtures = 0
    total_preds    = 0
    for key, info in reporte["ligas"].items():
        if "error" in info:
            print(f"  {key}: ERROR — {info['error']}")
        else:
            nf = info.get("n_fixtures", 0)
            np_ = len(info.get("picks", []))
            total_fixtures += nf
            total_preds    += np_
            print(f"  {key}: {nf} fixtures | {np_} predicciones")
    print(f"  TOTAL: {total_fixtures} fixtures | {total_preds} predicciones nuevas")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
