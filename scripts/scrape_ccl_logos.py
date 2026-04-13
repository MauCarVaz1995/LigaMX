#!/usr/bin/env python3
"""
scrape_ccl_logos.py
Descarga logos de todos los equipos de la CONCACAF Champions Cup 2025-26.

Fuente: FotMob CDN  →  images.fotmob.com/image_resources/logo/teamlogo/{id}.png
Destino: data/raw/logos/ccl/{NombreEquipo}.png

Uso:
  python scrape_ccl_logos.py           # descarga faltantes
  python scrape_ccl_logos.py --force   # re-descarga todos
"""

import argparse
import time
import urllib.request
from pathlib import Path

BASE     = Path(__file__).resolve().parent.parent
LOGO_DIR = BASE / 'data/raw/logos/ccl'
LOGO_DIR.mkdir(parents=True, exist_ok=True)

# ─── Equipos CCL 2025-26 con FotMob IDs ──────────────────────────────────────
# Liga MX: ya teníamos algunos en ligamx/, pero los replicamos aquí para
# tener un directorio ccl/ completo e independiente.
TEAMS = {
    # Liga MX
    'América':          6576,
    'Cruz Azul':        6578,
    'Tigres':           8561,
    'Toluca':           6618,
    'Monterrey':        7849,
    # MLS
    'Nashville SC':     915807,
    'LAFC':             867280,
    'LA Galaxy':        6637,
    'Seattle Sounders': 130394,
    'Cincinnati':       722265,
    'Inter Miami CF':   960720,
    'Philadelphia':     191716,
    'Vancouver':        307691,
    'San Diego FC':     1701119,
    # CONCACAF no-MLS
    'LD Alajuelense':   6335,
    'C.S. Cartaginés':  49732,
    'Atlético Ottawa':  1135780,
    'Real Espana':      49783,
    'Defence Force':    165465,
    'Mount Pleasant':   962034,
    'O&M FC':           1237060,
    'Sporting San Miguelito': 165472,
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer':    'https://www.fotmob.com/',
}

CDN_URL = 'https://images.fotmob.com/image_resources/logo/teamlogo/{}.png'


def download_logo(name: str, team_id: int, force: bool = False) -> bool:
    dest = LOGO_DIR / f'{name}.png'
    if dest.exists() and not force:
        print(f'  [skip] {name} — ya existe')
        return True

    url = CDN_URL.format(team_id)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        if len(data) < 500:
            print(f'  [warn] {name} — respuesta muy pequeña ({len(data)} bytes), posiblemente placeholder')
        dest.write_bytes(data)
        print(f'  [ok]   {name} ({len(data)//1024}KB) → {dest.name}')
        return True
    except Exception as e:
        print(f'  [err]  {name}: {e}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Re-descarga todos los logos')
    args = parser.parse_args()

    print(f'Descargando logos CCL 2025-26 → {LOGO_DIR}')
    print(f'Total equipos: {len(TEAMS)}')
    print()

    ok = 0
    for name, tid in TEAMS.items():
        if download_logo(name, tid, force=args.force):
            ok += 1
        time.sleep(0.3)

    print()
    print(f'Completado: {ok}/{len(TEAMS)} logos descargados en {LOGO_DIR}')


if __name__ == '__main__':
    main()
