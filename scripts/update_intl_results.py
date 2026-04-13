#!/usr/bin/env python3
"""
update_intl_results.py
Descarga partidos internacionales terminados de FotMob para fechas dadas
y agrega los nuevos a data/raw/internacional/results.csv
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

BASE     = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data/raw/internacional/results.csv"
RAW_DIR  = BASE / "data/raw/fotmob"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.fotmob.com/",
}

# Ligas que NO son selecciones (Liga MX, Premier League, etc.) — filtramos por exclusión
# FotMob marca partidos internacionales con country/type; usamos la lista de torneos de
# "international" que incluye: WC Qual, Nations League, Friendlies internacionales, etc.
# Heurística: excluimos leagueId conocidos de clubes (< 100 son grandes ligas de club en fotmob)
# En lugar de eso, cargamos todo y marcamos 'neutral' basándonos en el campo de FotMob.

EXCL_LEAGUE_IDS = {
    # Liga MX y ligas de club (IDs típicos en FotMob) — lista conservadora
    # Dejamos pasar todo y filtramos solo lo que claramente sea de club local
}

# ─── Ligas internacionales conocidas (incluir siempre) ───────────────────────
# Dejaremos pasar todo lo que NO sea ligas de club de un país específico.
# FotMob separa "international" en su API; aprovechamos el campo parentLeagueName.

def fetch_matches(date_str: str) -> list[dict]:
    """Descarga todos los partidos de una fecha desde FotMob."""
    date_fmt = date_str.replace("-", "")  # 20260330
    url = f"https://www.fotmob.com/api/data/matches?date={date_fmt}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [ERROR] {date_str}: {e}")
        return []
    return data

def parse_intl_matches(data: dict, date_str: str) -> list[dict]:
    """Extrae partidos de selecciones (internacionales) de la respuesta de FotMob."""
    matches = []
    leagues = data.get("leagues", [])
    for league in leagues:
        league_name = league.get("name", "")
        # Solo queremos ligas internacionales (ccode == "INT")
        if league.get("ccode", "") != "INT":
            continue

        for m in league.get("matches", []):
            status = m.get("status", {})
            if not status.get("finished", False):
                continue

            home = m.get("home", {})
            away = m.get("away", {})
            home_score = home.get("score")
            away_score = away.get("score")
            if home_score is None or away_score is None:
                continue

            matches.append({
                "date": date_str,
                "home_team": home.get("name", ""),
                "away_team": away.get("name", ""),
                "home_score": int(home_score),
                "away_score": int(away_score),
                "tournament": league_name or parent_name,
                "city": "",
                "country": "",
                "neutral": True,
            })
    return matches

def main():
    dates = ["2026-03-30", "2026-03-31", "2026-04-01"]

    # Cargar CSV existente
    df_existing = pd.read_csv(CSV_PATH)
    df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.strftime("%Y-%m-%d")

    # Clave de deduplicación
    existing_keys = set(
        zip(df_existing["date"], df_existing["home_team"], df_existing["away_team"])
    )

    new_rows = []
    by_date = {}

    for date_str in dates:
        print(f"\n{'='*50}")
        print(f"  Descargando FotMob: {date_str}")
        data = fetch_matches(date_str)

        # Guardar raw
        raw_file = RAW_DIR / f"intl_{date_str.replace('-','')}.json"
        raw_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        if not data:
            print(f"  [!] Sin datos para {date_str}")
            by_date[date_str] = 0
            time.sleep(1)
            continue

        parsed = parse_intl_matches(data, date_str)
        added = 0
        for row in parsed:
            key = (row["date"], row["home_team"], row["away_team"])
            if key not in existing_keys:
                new_rows.append(row)
                existing_keys.add(key)
                added += 1
                print(f"    + {row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']}  [{row['tournament']}]")

        by_date[date_str] = added
        print(f"  → {added} partidos nuevos en {date_str}")
        time.sleep(1.2)  # respetar rate limit

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_PATH, index=False)
        print(f"\n✓ CSV actualizado: {len(new_rows)} partidos nuevos agregados")
    else:
        print("\n→ No hay partidos nuevos que agregar")

    print("\n── Resumen por fecha ──────────────────────────────")
    for d, n in by_date.items():
        print(f"  {d}: {n} partidos nuevos")

    return by_date

if __name__ == "__main__":
    main()
