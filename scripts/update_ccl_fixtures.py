#!/usr/bin/env python3
"""
update_ccl_fixtures.py
Descarga y guarda todos los fixtures/resultados de la CONCACAF Champions Cup
desde FotMob (league id=915924).

Salida: data/raw/fotmob/ccl/ccl_fixtures_YYYYMMDD.json

Uso:
  python update_ccl_fixtures.py
"""

import json
import time
import requests
from datetime import date
from pathlib import Path

BASE     = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE / 'data/raw/fotmob/ccl'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CCL_ID = 915924

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept':     'application/json',
    'Referer':    'https://www.fotmob.com/',
}

# Todas las fechas posibles de la CCL 2025-26
# Rondas: play-in (feb), grupos (feb-mar), cuartos (mar-abr), semis (abr), final (may)
DATES = [
    # Play-in / primera ronda
    '20260218', '20260219', '20260220',
    '20260225', '20260226', '20260227',
    # Cuartos de final ida
    '20260311', '20260312', '20260313',
    # Cuartos de final vuelta
    '20260318', '20260319', '20260320',
    # Semifinales ida
    '20260408', '20260409',
    # Semifinales vuelta
    '20260415', '20260416',
    # Final
    '20260429', '20260430',
    '20260506', '20260507',
]


def fetch_matches(date_str: str) -> list[dict]:
    url = f'https://www.fotmob.com/api/data/matches?date={date_str}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        d = r.json()
        for league in d.get('leagues', []):
            if league.get('id') == CCL_ID:
                return league.get('matches', [])
    except Exception as e:
        print(f'  [warn] {date_str}: {e}')
    return []


def parse_match(m: dict, date_str: str) -> dict:
    h = m.get('home', {})
    a = m.get('away', {})
    status = m.get('status', {})
    return {
        'date':       date_str,
        'match_id':   m.get('id'),
        'home':       h.get('name', ''),
        'home_id':    h.get('id'),
        'away':       a.get('name', ''),
        'away_id':    a.get('id'),
        'home_score': h.get('score'),
        'away_score': a.get('score'),
        'finished':   status.get('finished', False),
        'started':    status.get('started', False),
        'round':      m.get('roundId', ''),
        'tournament': 'CONCACAF Champions Cup',
    }


def main():
    print('Actualizando fixtures CCL 2025-26...')
    all_matches = []

    for date_str in DATES:
        raw = fetch_matches(date_str)
        if raw:
            parsed = [parse_match(m, date_str) for m in raw]
            # Filtrar placeholders (partidos con nombres tipo "Team A/Team B")
            parsed = [p for p in parsed if '/' not in p['home'] and '/' not in p['away']]
            all_matches.extend(parsed)
            for p in parsed:
                score = f"{p['home_score']}-{p['away_score']}" if p['finished'] else 'pendiente'
                print(f'  [{date_str}] {p["home"]} vs {p["away"]} — {score}')
        time.sleep(0.5)

    # Clasificar por ronda según fecha
    for m in all_matches:
        d = m['date']
        if d <= '20260227':
            m['stage'] = 'play_in'
        elif d <= '20260320':
            m['stage'] = 'quarterfinals'
        elif d <= '20260416':
            m['stage'] = 'semifinals'
        else:
            m['stage'] = 'final'

    out = {
        'league_id':   CCL_ID,
        'tournament':  'CONCACAF Champions Cup',
        'season':      '2025-26',
        'updated':     date.today().isoformat(),
        'total':       len(all_matches),
        'matches':     all_matches,
    }

    today = date.today().strftime('%Y%m%d')
    out_file = OUT_DIR / f'ccl_fixtures_{today}.json'
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f'\n{len(all_matches)} partidos guardados → {out_file}')


if __name__ == '__main__':
    main()
