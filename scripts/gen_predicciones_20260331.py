#!/usr/bin/env python3
"""
gen_predicciones_20260331.py
Genera imágenes de predicción para la fecha FIFA 2026-03-31 / 2026-04-01.
Usa ELOs actualizados al 2026-03-29.
"""

import sys, json, random, importlib, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# ── Cargar datos ──────────────────────────────────────────────────────────────
with open(BASE / 'data/raw/fotmob/fixtures_20260330_20260404.json') as f:
    fixtures = json.load(f)['fixtures']

with open(BASE / 'data/processed/elos_selecciones_20260329.json') as f:
    elos = json.load(f)

# ── Abreviaturas de torneo ────────────────────────────────────────────────────
TOURN_ABBREV = [
    ('World Cup Qualification UEFA',             'WC Qual UEFA'),
    ('Asian Cup Qualification',                  'Asian Cup Qual'),
    ('Asian Qualifiers',                         'Asian WC Qual'),
    ('UEFA Nations League C',                    'NL-C Qual'),
    ('UEFA Nations League',                      'UEFA Nations League'),
    ('World Cup Qualification Inter-Confederation', 'WC Inter-Conf Playoff'),
    ('Friendly',                                 'Amistoso'),
]

def abbrev_tourn(t):
    for key, short in TOURN_ABBREV:
        if key.lower() in t.lower():
            return short
    return t[:25]

MONTH_ES = {'01':'ene','02':'feb','03':'mar','04':'abr','05':'may',
            '06':'jun','07':'jul','08':'ago','09':'sep','10':'oct',
            '11':'nov','12':'dic'}

def fmt_date(d):
    day, mo, yr = d[8:], d[5:7], d[:4]
    return f'{int(day)} {MONTH_ES.get(mo, mo)} {yr}'

# ── Construir lista MATCHES ───────────────────────────────────────────────────
MATCHES = []
for fx in fixtures:
    h, a = fx['home_team'], fx['away_team']
    if h not in elos or a not in elos:
        print(f'⚠  Sin ELO: {h} / {a}')
        continue
    slug_h = ''.join(c for c in h[:4].upper() if c.isalpha())[:3]
    slug_a = ''.join(c for c in a[:4].upper() if c.isalpha())[:3]
    date_str = fx['date']
    label = f'{h} vs {a}  ·  {abbrev_tourn(fx["tournament"])} · {fmt_date(date_str)}'
    MATCHES.append(dict(
        team1=h, team2=a,
        elo1=elos[h], elo2=elos[a],
        file=f'pred_{slug_h}_{slug_a}_{date_str.replace("-","")}.png',
        label=label,
        fecha=date_str,
    ))

print(f'\n{"═"*60}')
print(f'  PREDICCIONES  2026-03-31 / 2026-04-01 — {len(MATCHES)} partido(s)')
print(f'{"═"*60}\n')

# ── Importar script 19 y parchear globals ─────────────────────────────────────
s19 = importlib.import_module('19_predicciones_hoy')

OUT_DIR_NEW = BASE / 'output/charts/predicciones'
OUT_DIR_NEW.mkdir(parents=True, exist_ok=True)

# Patch module globals so render_prediccion() uses correct OUT_DIR and ELOs
s19.OUT_DIR         = OUT_DIR_NEW
s19.ALL_ELOS_TODAY  = [m['elo1'] for m in MATCHES] + [m['elo2'] for m in MATCHES]

# ── Importar paletas y tracker ────────────────────────────────────────────────
from config_visual import PALETAS

tracker  = importlib.import_module('04_predicciones_tracker')
pal_keys = list(PALETAS.keys())
last_pal = None

results = []
for m in MATCHES:
    available = [k for k in pal_keys if k != last_pal]
    pal_key   = random.choice(available)
    last_pal  = pal_key
    pal       = PALETAS[pal_key]
    print(f'🎨  {pal_key}')
    print(f'⚽  {m["label"]}')

    ph, pd, pa, midx, bp = s19.render_prediccion(m, pal)
    lam1, lam2 = s19.elo_to_lambda(m['elo1'], m['elo2'], s19.ALL_ELOS_TODAY)

    winner = m['team1'] if ph > pa else (m['team2'] if pa > ph else 'Empate')
    print(f'    P(local)={ph*100:.1f}%  P(empate)={pd*100:.1f}%  P(visita)={pa*100:.1f}%')
    print(f'    Más probable: {m["team1"]} {midx[0]}-{midx[1]} {m["team2"]} ({bp:.1f}%)\n')

    tracker.registrar_prediccion(
        equipo_local      = m['team1'],
        equipo_visitante  = m['team2'],
        elo_local         = m['elo1'],
        elo_visitante     = m['elo2'],
        prob_local        = ph,
        prob_empate       = pd,
        prob_visitante    = pa,
        marcador_probable = f'{midx[0]}-{midx[1]}',
        lambda_local      = lam1,
        lambda_visitante  = lam2,
        fecha_partido     = m['fecha'],
        paleta            = pal_key,
    )
    results.append((m, ph, pd, pa, midx, bp, winner))

# ── Tabla resumen ─────────────────────────────────────────────────────────────
print(f'\n{"═"*95}')
print(f'{"LOCAL":<30} {"VISITANTE":<30} {"ELO-L":>6} {"ELO-V":>6} {"L%":>6} {"E%":>6} {"V%":>6} {"SCORE":>5} {"PRED.":>10}')
print(f'{"─"*95}')
for m, ph, pd, pa, midx, bp, winner in results:
    print(f'{m["team1"]:<30} {m["team2"]:<30} {m["elo1"]:>6.0f} {m["elo2"]:>6.0f} '
          f'{ph*100:>5.1f}% {pd*100:>5.1f}% {pa*100:>5.1f}% '
          f'{midx[0]}-{midx[1]}  {winner:<10}')
print(f'{"═"*95}\n')
print(f'Imágenes guardadas en: {OUT_DIR_NEW}')
print('Listo.')
