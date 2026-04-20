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
BASE         = Path(__file__).resolve().parent.parent
ELOS_DIR     = BASE / "data/processed/elos_ligas"
PRED_LOG     = BASE / "data/processed/predicciones_log.csv"
INTL_CSV     = BASE / "data/raw/internacional/results.csv"
REPORTS_DIR  = BASE / "output/reports"

ELOS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# LIGAS
# ─────────────────────────────────────────────────────────────────────────────
LIGAS: dict[str, dict] = {
    "champions":   {"id": 42,  "nombre": "UEFA Champions League"},
    "libertadores": {"id": 384, "nombre": "Copa Libertadores"},
    "mls":         {"id": 130, "nombre": "MLS"},
    "argentina":   {"id": 112, "nombre": "Liga Argentina"},
    "brasileirao": {"id": 268, "nombre": "Brasileirao"},
}

# Palabras clave en results.csv para mapear a cada liga
LIGA_TOURNAMENT_KEYWORDS: dict[str, list[str]] = {
    "champions":   ["champions league", "uefa champions"],
    "libertadores": ["copa libertadores", "libertadores"],
    "mls":         ["major league soccer", "mls"],
    "argentina":   ["primera division", "liga profesional", "superliga argentina"],
    "brasileirao": ["brasileirao", "campeonato brasileiro", "serie a"],
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
DC_RHO       = -0.10  # Dixon-Coles rho (draw correction)

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


def get_or_bootstrap_elos(liga_key: str) -> dict[str, float]:
    """Si el CSV ya existe, lo carga. Si no, ejecuta bootstrap desde histórico."""
    p = elo_csv_path(liga_key)
    if p.exists() and p.stat().st_size > 100:
        return load_elos(liga_key)
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
    """Convierte ELOs a λ_local y λ_visitante para el modelo Poisson."""
    elo_eff_l = elo_local    + HOME_ADV_ELO * 0.5
    elo_eff_v = elo_visitante - HOME_ADV_ELO * 0.5
    lam_l = MU_HOME * (elo_eff_l / ELO_BASE)
    lam_v = MU_AWAY * (elo_eff_v / ELO_BASE)
    return max(lam_l, 0.15), max(lam_v, 0.15)


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
# FOTMOB API
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fotmob(league_id: int) -> dict | None:
    """Obtiene datos de la liga desde FotMob API. Retorna JSON o None."""
    url = f"https://www.fotmob.com/api/leagues?id={league_id}&ccode3=MEX"
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=FOTMOB_HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            print(f"  [WARN] HTTP {e.response.status_code} para liga {league_id} (intento {attempt}/3)")
        except Exception as e:
            print(f"  [WARN] Error fetch liga {league_id} (intento {attempt}/3): {e}")
        if attempt < 3:
            time.sleep(5 * attempt)
    return None


def parse_fixtures(data: dict, liga_key: str) -> list[dict]:
    """
    Extrae fixtures próximos (no terminados) desde la respuesta de FotMob.
    Retorna lista de dicts con: home, away, fecha, match_id.
    """
    fixtures = []
    try:
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

    # 3. Obtener fixtures de FotMob
    data = fetch_fotmob(league_id)
    fixtures = []
    if data is not None:
        fixtures = parse_fixtures(data, liga_key)
    else:
        print(f"  [{liga_key}] No se obtuvieron fixtures de FotMob")

    n_fixtures = len(fixtures)

    # 4. Generar predicciones
    nuevas_preds: list[dict] = []
    picks: list[dict] = []

    for fix in fixtures:
        local    = fix["home"]
        visitante = fix["away"]
        fecha    = fix["fecha"]

        # Saltar si ya está predicho
        if already_predicted(df_log, local, visitante, fecha):
            continue

        elo_l = elos.get(local,    ELO_INIT)
        elo_v = elos.get(visitante, ELO_INIT)

        pred = generate_prediction(liga_key, local, visitante, fecha, elo_l, elo_v)
        nuevas_preds.append(pred)
        picks.append({
            "partido":   pred["partido"],
            "fecha":     fecha,
            "ganador":   pred["ganador_predicho"],
            "prob_local": pred["prob_local"],
            "prob_empate": pred["prob_empate"],
            "prob_visitante": pred["prob_visitante"],
            "marcador":  pred["marcador_mas_probable"],
            "elo_local": pred["elo_local"],
            "elo_visitante": pred["elo_visitante"],
        })

    # Agregar nuevas predicciones al log
    if nuevas_preds:
        df_new = pd.DataFrame(nuevas_preds)
        df_log = pd.concat([df_log, df_new], ignore_index=True)

    n_pred = len(nuevas_preds)
    print(f"  [{liga_key}] {n_fixtures} fixtures | {n_pred} predicciones generadas")

    return {
        "liga_key":   liga_key,
        "nombre":     nombre,
        "n_fixtures": n_fixtures,
        "n_predicciones": n_pred,
        "picks":      picks,
        "df_log":     df_log,
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

    # Ejecutar pipeline por liga
    reporte: dict = {
        "generated_at": datetime.now().isoformat(),
        "ligas": {},
    }

    for liga_key in ligas_a_correr:
        try:
            result = run_liga(liga_key, df_log, results_idx)
            df_log = result.pop("df_log")  # propagar el log actualizado
            reporte["ligas"][liga_key] = {
                "nombre":     result["nombre"],
                "n_fixtures": result["n_fixtures"],
                "picks":      result["picks"],
            }
        except Exception as e:
            print(f"  [ERROR] Liga {liga_key} falló: {e}")
            reporte["ligas"][liga_key] = {"error": str(e)}

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
