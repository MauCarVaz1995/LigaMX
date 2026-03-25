#!/usr/bin/env python3
"""
14_simulacion_montecarlo.py
Simulación de 10,000 escenarios del Clausura 2026.
Lee tabla actual + partidos restantes (jornadas 13-17) y simula con modelo Poisson.
Genera heatmap de probabilidades de posición final por equipo.
Salida: output/charts/montecarlo_clausura2026.png  (150 DPI)
"""

import json, glob, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from scipy.stats import poisson
from PIL import Image, ImageDraw
import urllib.request

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETTE, bebas, hex_rgba, hex_rgb

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
HIST_DIR   = BASE / 'data/raw/historico'
IMG_TEAMS  = BASE / 'data/raw/images/teams'
OUT_DIR    = BASE / 'output/charts'
BEBAS_TTF  = Path.home() / '.fonts/BebasNeue.ttf'

IMG_TEAMS.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

if BEBAS_TTF.exists():
    fm.fontManager.addfont(str(BEBAS_TTF))

N_SIM = 10_000

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
BG    = PALETTE['bg_main']
WHITE = PALETTE['text_primary']
GRAY  = PALETTE['text_secondary']
RED   = PALETTE['accent']

TEAM_IDS = {
    'Chivas': 7807, 'Cruz Azul': 6578, 'Toluca': 6618,
    'América': 6576, 'CF America': 6576,
    'Tigres': 8561, 'Monterrey': 7849, 'Pumas': 1946,
    'Santos Laguna': 7857, 'Pachuca': 7848, 'Atlas': 6577,
    'León': 1841, 'Necaxa': 1842, 'Tijuana': 162418,
    'Querétaro': 1943, 'FC Juárez': 649424, 'FC Juarez': 649424,
    'Mazatlán': 1170234, 'Mazatlan FC': 1170234,
    'San Luis': 6358, 'Atletico de San Luis': 6358,
    'Puebla': 7847,
}

TORNEO_WEIGHTS = {
    '2025/2026 - Clausura': 4,
    '2025/2026 - Apertura': 3,
    '2024/2025 - Clausura': 2,
    '2024/2025 - Apertura': 1,
}

_ALIAS = {
    'cf america': 'América',  'america': 'América',
    'chivas': 'Chivas',       'guadalajara': 'Chivas',
    'cruz azul': 'Cruz Azul',
    'tigres': 'Tigres',       'tigres uanl': 'Tigres',
    'monterrey': 'Monterrey', 'cf monterrey': 'Monterrey',
    'pumas': 'Pumas',         'pumas unam': 'Pumas',
    'toluca': 'Toluca',
    'santos laguna': 'Santos Laguna', 'santos': 'Santos Laguna',
    'pachuca': 'Pachuca',
    'atlas': 'Atlas',
    'león': 'León', 'leon': 'León',
    'necaxa': 'Necaxa',
    'tijuana': 'Tijuana',
    'querétaro': 'Querétaro', 'queretaro': 'Querétaro', 'queretaro fc': 'Querétaro',
    'fc juárez': 'FC Juárez', 'fc juarez': 'FC Juárez',
    'mazatlán': 'Mazatlán',   'mazatlan fc': 'Mazatlán', 'mazatlan': 'Mazatlán',
    'atletico de san luis': 'San Luis', 'san luis': 'San Luis',
    'puebla': 'Puebla',
}

def norm(n): return _ALIAS.get(str(n).lower().strip(), n.strip())

# ─────────────────────────────────────────────────────────────────────────────
# DATOS: TABLA ACTUAL + PARTIDOS RESTANTES
# ─────────────────────────────────────────────────────────────────────────────
def load_current_data():
    """Lee clausura_2026.json: tabla actual + partidos pendientes."""
    fpath = HIST_DIR / 'historico_clausura_2026.json'
    data  = json.load(open(fpath, encoding='utf-8'))

    # Tabla actual (después de J12)
    tabla = {}
    for row in data.get('tabla', []):
        equipo = norm(row['equipo'])
        tabla[equipo] = {
            'pts': int(row.get('pts', 0)),
            'pj':  int(row.get('pj',  0)),
            'g':   int(row.get('g',   0)),
            'e':   int(row.get('e',   0)),
            'p':   int(row.get('p',   0)),
            'gf':  int(row.get('gf',  0)),
            'gc':  int(row.get('gc',  0)),
        }

    # Partidos pendientes (terminado=False)
    pendientes = []
    for p in data.get('partidos', []):
        if p.get('terminado'):
            continue
        lo = norm(p.get('local', ''))
        vi = norm(p.get('visitante', ''))
        pendientes.append({'local': lo, 'visitante': vi})

    return tabla, pendientes

# ─────────────────────────────────────────────────────────────────────────────
# MODELO DE POISSON
# ─────────────────────────────────────────────────────────────────────────────
def load_model_matches():
    rows = []
    for fpath in sorted(glob.glob(str(HIST_DIR / '*.json'))):
        stem = Path(fpath).stem.replace('historico_', '')
        stem = stem.replace('-', '/', 1)
        parts = stem.split('_-_', 1)
        if len(parts) == 2:
            year  = parts[0].replace('_', '/')
            torneo = parts[1].replace('_', ' ').title()
            tkey  = f'{year} - {torneo}'
        else:
            continue
        w = TORNEO_WEIGHTS.get(tkey)
        if not w:
            continue
        d = json.load(open(fpath, encoding='utf-8'))
        for p in d.get('partidos', []):
            if not p.get('terminado'):
                continue
            if p.get('goles_local') is None or p.get('goles_visit') is None:
                continue
            rows.append({
                'peso': w,
                'local':     norm(p.get('local', '')),
                'visitante': norm(p.get('visitante', '')),
                'gl': int(p['goles_local']),
                'gv': int(p['goles_visit']),
            })
    return rows

def build_model(matches):
    sum_gl = sum_gv = sum_w = 0.0
    hs = defaultdict(float); hc = defaultdict(float)
    as_ = defaultdict(float); ac = defaultdict(float)
    hg = defaultdict(float);  ag = defaultdict(float)
    for m in matches:
        w = m['peso']; lo = m['local']; vi = m['visitante']
        gl = m['gl'];  gv = m['gv']
        sum_gl += gl*w; sum_gv += gv*w; sum_w += w
        hs[lo]  += gl*w; hc[lo] += gv*w
        as_[vi] += gv*w; ac[vi] += gl*w
        hg[lo]  += w;    ag[vi] += w
    mu_h = sum_gl / sum_w; mu_a = sum_gv / sum_w
    teams = set(hs) | set(as_)
    att = {}; defe = {}
    for t in teams:
        att[t]  = {'home': (hs[t]  / hg[t] / mu_h) if hg[t] > 0 else 1.0,
                   'away': (as_[t] / ag[t] / mu_a) if ag[t] > 0 else 1.0}
        defe[t] = {'home': (hc[t]  / hg[t] / mu_a) if hg[t] > 0 else 1.0,
                   'away': (ac[t]  / ag[t] / mu_h) if ag[t] > 0 else 1.0}
    return {'mu_home': mu_h, 'mu_away': mu_a, 'att': att, 'defe': defe}

def get_lambda(model, local, visitante):
    mu_h = model['mu_home']; mu_a = model['mu_away']
    att  = model['att'];     defe = model['defe']
    lam_l = att.get(local,     {}).get('home', 1.0) * defe.get(visitante, {}).get('away', 1.0) * mu_h
    lam_v = att.get(visitante, {}).get('away', 1.0) * defe.get(local,     {}).get('home', 1.0) * mu_a
    return max(lam_l, 0.2), max(lam_v, 0.2)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def simulate_one(model, tabla_base, pendientes, rng):
    """Un escenario: simula todos los partidos pendientes y devuelve ranking final."""
    pts = {t: v['pts'] for t, v in tabla_base.items()}
    gf  = {t: v['gf']  for t, v in tabla_base.items()}
    gc  = {t: v['gc']  for t, v in tabla_base.items()}

    for p in pendientes:
        lo = p['local']; vi = p['visitante']
        if lo not in pts: pts[lo] = gf[lo] = gc[lo] = 0
        if vi not in pts: pts[vi] = gf[vi] = gc[vi] = 0

        lam_l, lam_v = get_lambda(model, lo, vi)
        gl = rng.poisson(lam_l)
        gv = rng.poisson(lam_v)

        gf[lo] += gl; gc[lo] += gv
        gf[vi] += gv; gc[vi] += gl

        if gl > gv:        pts[lo] += 3
        elif gl == gv:     pts[lo] += 1; pts[vi] += 1
        else:              pts[vi] += 3

    # Ordenar: pts desc, luego diferencia de goles desc, luego goles a favor desc
    teams = sorted(pts.keys(), key=lambda t: (pts[t], gf[t] - gc[t], gf[t]), reverse=True)
    return {t: i + 1 for i, t in enumerate(teams)}

# ─────────────────────────────────────────────────────────────────────────────
# ESCUDO
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_TEAM = 'https://images.fotmob.com/image_resources/logo/teamlogo/{}.png'

def get_shield(team_name: str, size: int = 64) -> np.ndarray | None:
    name_key = {
        'América': 'CF America', 'FC Juárez': 'FC Juarez',
        'Mazatlán': 'Mazatlan FC', 'San Luis': 'Atletico de San Luis',
        'Querétaro': 'Queretaro FC', 'León': 'León',
    }.get(team_name, team_name)
    tid = TEAM_IDS.get(name_key) or TEAM_IDS.get(team_name)
    if tid is None:
        return None
    cache = IMG_TEAMS / f'{tid}.png'
    if not cache.exists():
        try:
            urllib.request.urlretrieve(FOTMOB_TEAM.format(tid), cache)
        except Exception:
            return None
    try:
        img = Image.open(cache).convert('RGBA')
        orig_w, orig_h = img.size
        target = min(size, orig_w, orig_h)   # never upscale beyond original
        if target != orig_w or target != orig_h:
            img = img.resize((target, target), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def render(teams_sorted, prob_matrix, tabla_base, output_path):
    N_TEAMS = len(teams_sorted)
    N_POS   = N_TEAMS   # 18

    FIG_W, FIG_H = 11.0, 16.0
    AR = FIG_H / FIG_W

    TEAM_ABBR = {
        'América': 'AMÉ', 'Chivas': 'CHI', 'Cruz Azul': 'CAZ',
        'Tigres': 'TIG', 'Monterrey': 'MTY', 'Pumas': 'PUM',
        'Toluca': 'TOL', 'Santos Laguna': 'SAN', 'Pachuca': 'PAC',
        'Atlas': 'ATL', 'León': 'LEO', 'Necaxa': 'NEC',
        'Tijuana': 'TIJ', 'Querétaro': 'QRO', 'FC Juárez': 'FCJ',
        'Mazatlán': 'MAZ', 'San Luis': 'SLP', 'Puebla': 'PUE',
    }

    # Current standings (for diagonal highlighting)
    teams_by_rank = sorted(
        tabla_base.keys(),
        key=lambda t: (tabla_base[t]['pts'],
                       tabla_base[t]['gf'] - tabla_base[t]['gc'],
                       tabla_base[t]['gf']),
        reverse=True)
    current_pos = {t: i + 1 for i, t in enumerate(teams_by_rank)}

    # Layout in inches
    HEADER_IN  = 1.55
    FOOTER_IN  = 0.55
    COLHDR_IN  = 0.42
    LEFT_IN    = 1.38
    MARGIN_IN  = 0.10
    CW_IN      = (FIG_W - LEFT_IN - MARGIN_IN) / (N_POS + 2 * 1.4)
    SUM_IN     = CW_IN * 1.4

    def n(val_in, total): return val_in / total

    HEADER_H = n(HEADER_IN, FIG_H)
    FOOTER_H = n(FOOTER_IN, FIG_H)
    COLHDR_H = n(COLHDR_IN, FIG_H)
    LEFT_X   = n(LEFT_IN, FIG_W)
    POS_W    = n(CW_IN, FIG_W)
    SUM_W    = n(SUM_IN, FIG_W)

    CONTENT_Y = FOOTER_H
    CONTENT_H = 1.0 - HEADER_H - FOOTER_H - COLHDR_H
    ROW_H     = CONTENT_H / N_TEAMS

    pos_col_x  = [LEFT_X + j * POS_W for j in range(N_POS)]
    sum4_x     = LEFT_X + N_POS * POS_W
    sum8_x     = sum4_x + SUM_W

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Background gradient
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = (np.array([0x0a,0x0e,0x12])/255*(1-t)
                   + np.array([0x13,0x1a,0x24])/255*t)
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bgax.axis('off')

    # ── HEADER ──────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(PALETTE['bg_secondary'])
    hax.axis('off')
    hax.axhline(0, color=RED, lw=2.5)
    hax.text(0.50, 0.88, '¿CÓMO TERMINA EL CLAUSURA 2026?',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(24))
    hax.text(0.50, 0.56,
             '10,000 simulaciones Monte Carlo · 45 partidos restantes · Jornadas 13–17',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=9.5)
    hax.text(0.50, 0.30,
             'Cada celda = % de simulaciones en que el equipo terminó en esa posición · '
             'Orden: P(Top 4) decreciente',
             color=PALETTE['text_secondary'], ha='center', va='top',
             transform=hax.transAxes, fontsize=7.5)

    # ── COLUMN HEADERS ───────────────────────────────────────────────────────
    col_hdr_y = CONTENT_Y + CONTENT_H

    # Zone labels (spanning group of columns)
    zone_defs = [
        (0,  3,  '#2ea043', '#0d2a18', 'LIGUILLA DIRECTA'),
        (4,  7,  '#d4a72c', '#2a2200', 'REPECHAJE'),
        (8,  17, '#f85149', '#1a0505', ''),
    ]
    for (c0, c1, fg, bg_z, zlbl) in zone_defs:
        zx0 = pos_col_x[c0]
        zx1 = pos_col_x[c1] + POS_W
        zax = fig.add_axes([zx0, col_hdr_y + COLHDR_H * 0.50, zx1 - zx0, COLHDR_H * 0.50])
        zax.set_facecolor(bg_z); zax.axis('off')
        if zlbl:
            zax.text(0.5, 0.5, zlbl, color=fg, ha='center', va='center',
                     fontsize=6.0, fontweight='bold', transform=zax.transAxes)

    # Individual position number cells
    for j in range(N_POS):
        pos = j + 1
        if pos <= 4:
            hdr_bg, hdr_fg = '#1a3a2a', '#2ea043'
        elif pos <= 8:
            hdr_bg, hdr_fg = '#3a3a1a', '#d4a72c'
        else:
            hdr_bg, hdr_fg = PALETTE['bg_card'], GRAY
        cax = fig.add_axes([pos_col_x[j], col_hdr_y, POS_W, COLHDR_H * 0.50])
        cax.set_facecolor(hdr_bg); cax.axis('off')
        for sp in cax.spines.values():
            sp.set_edgecolor(PALETTE['divider']); sp.set_linewidth(0.4)
        cax.text(0.5, 0.5, str(pos), color=hdr_fg,
                 ha='center', va='center', fontsize=8,
                 fontweight='bold' if pos <= 8 else 'normal',
                 transform=cax.transAxes)

    # Summary column headers
    for (sx, lbl, fg_c) in [(sum4_x, 'TOP\n4', '#2ea043'), (sum8_x, 'TOP\n8', '#d4a72c')]:
        shax = fig.add_axes([sx, col_hdr_y, SUM_W, COLHDR_H])
        shax.set_facecolor(PALETTE['bg_card']); shax.axis('off')
        shax.text(0.5, 0.5, lbl, color=fg_c, ha='center', va='center',
                  fontsize=7.5, fontweight='bold', transform=shax.transAxes)

    # ── TEAM ROWS ────────────────────────────────────────────────────────────
    def cell_style(prob):
        """Returns (face_color, text_color, bold, font_size)"""
        if prob > 0.15:   return '#D5001C', '#ffffff', True,  11
        elif prob > 0.05: return '#8b1a2b', '#ffffff', False,  9
        elif prob > 0.01: return '#1e2a3a', '#c9d1d9', False,  8
        else:             return '#161b22', '#30363d', False,  8

    BADGE_S = 64

    for i, team in enumerate(teams_sorted):
        prob_row = prob_matrix[team]
        row_y    = CONTENT_Y + (N_TEAMS - 1 - i) * ROW_H
        row_bg   = PALETTE['bg_secondary'] if i % 2 == 0 else PALETTE['bg_card']
        diag_col = current_pos.get(team, 99) - 1

        # Left column: crest + abbreviation
        lcol = fig.add_axes([0, row_y, LEFT_X, ROW_H])
        lcol.set_facecolor(row_bg); lcol.axis('off')
        lcol.axhline(1, color=PALETTE['divider'], lw=0.5)

        shield = get_shield(team, BADGE_S)
        if shield is not None:
            bh = ROW_H * 0.78
            bw = bh * AR
            sax = fig.add_axes([0.004, row_y + (ROW_H - bh) / 2, bw, bh])
            sax.set_facecolor('#f8f8fc')
            sax.imshow(shield); sax.axis('off')
        else:
            bw = 0.0

        abbr = TEAM_ABBR.get(team, team[:5].upper())
        name_x_in_ax = (0.004 + bw + 0.008) / LEFT_X
        lcol.text(name_x_in_ax, 0.50, abbr,
                  color=WHITE, ha='left', va='center', fontsize=8.5, fontweight='bold')

        # Position cells
        for j in range(N_POS):
            pos  = j + 1
            prob = prob_row.get(pos, 0.0)
            fc, tc, bold, fs = cell_style(prob)
            is_diag = (j == diag_col)

            cax = fig.add_axes([pos_col_x[j], row_y, POS_W, ROW_H])
            cax.set_facecolor(fc); cax.axis('off')

            for sp in cax.spines.values():
                sp.set_visible(True)
                if is_diag:
                    sp.set_edgecolor('#ffffff'); sp.set_linewidth(1.2)
                else:
                    sp.set_edgecolor(PALETTE['divider']); sp.set_linewidth(0.3)

            if prob >= 0.001:
                # >20%: 11pt bold  |  5-20%: 9pt  |  <5%: 8pt  |  0%: no text
                display_fs = 11 if prob > 0.20 else fs
                cax.text(0.5, 0.5, f'{prob*100:.0f}',
                         color=tc, ha='center', va='center',
                         fontsize=display_fs,
                         fontweight='bold' if (bold or prob > 0.20) else 'normal',
                         transform=cax.transAxes)

        # Summary: P(Top4) and P(Top8)
        p4 = sum(prob_row.get(p, 0) for p in range(1, 5))
        p8 = sum(prob_row.get(p, 0) for p in range(1, 9))

        for (sx, pval, hi_thresh, mid_thresh) in [
                (sum4_x, p4, 0.50, 0.20),
                (sum8_x, p8, 0.70, 0.35)]:
            sumax = fig.add_axes([sx, row_y, SUM_W, ROW_H])
            sumax.set_facecolor(row_bg); sumax.axis('off')
            sumax.axhline(1, color=PALETTE['divider'], lw=0.5)
            tc_s = ('#2ea043' if pval >= hi_thresh
                    else ('#d4a72c' if pval >= mid_thresh else GRAY))
            sumax.text(0.5, 0.5, f'{pval*100:.0f}%',
                       color=tc_s, ha='center', va='center', fontsize=8.5,
                       fontweight='bold' if pval >= hi_thresh else 'normal',
                       transform=sumax.transAxes)

    # ── LEGEND ───────────────────────────────────────────────────────────────
    leg_y = FOOTER_H * 0.58
    leg_h = FOOTER_H * 0.34
    lax = fig.add_axes([0.01, leg_y, 0.72, leg_h])
    lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis('off')
    items = [('>15%', '#D5001C', '#fff'), ('5–15%', '#8b1a2b', '#fff'),
             ('1–5%',  '#1e2a3a', '#c9d1d9'), ('<1%', '#161b22', '#30363d')]
    for k, (lbl, fc, tc) in enumerate(items):
        bx = k * 0.25
        lax.add_patch(mpatches.Rectangle((bx, 0.1), 0.055, 0.80,
                      facecolor=fc, edgecolor=PALETTE['divider'], lw=0.5))
        lax.text(bx + 0.07, 0.50, lbl, color=GRAY, va='center', fontsize=7.0)
    lax.text(0.86, 0.50, '□ posición actual  (borde blanco)',
             color=PALETTE['text_secondary'], va='center', fontsize=6.5)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.50])
    fax.set_facecolor(PALETTE['bg_secondary']); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.0)
    fax.text(0.015, 0.45, 'Fuente: FotMob · Clausura 2026',
             color=GRAY, fontsize=9, ha='left', va='center', transform=fax.transAxes)
    fax.text(0.985, 0.45, 'MAU-STATISTICS',
             color=RED, ha='right', va='center', transform=fax.transAxes, **bebas(20))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {output_path}')


def main():
    print('Cargando datos del Clausura 2026...')
    tabla_base, pendientes = load_current_data()
    print(f'  {len(tabla_base)} equipos en tabla, {len(pendientes)} partidos pendientes')

    print('Construyendo modelo Poisson...')
    matches = load_model_matches()
    model   = build_model(matches)
    print(f'  {len(matches)} partidos de entrenamiento')

    print(f'Simulando {N_SIM:,} escenarios...')
    rng = np.random.default_rng(42)
    # posiciones[equipo][pos] = conteo
    posiciones = {t: defaultdict(int) for t in tabla_base}
    for sim_i in range(N_SIM):
        pos_map = simulate_one(model, tabla_base, pendientes, rng)
        for team, pos in pos_map.items():
            if team in posiciones:
                posiciones[team][pos] += 1

    # Convertir a probabilidades
    prob_matrix = {}
    for team in posiciones:
        prob_matrix[team] = {pos: cnt / N_SIM for pos, cnt in posiciones[team].items()}

    # Ordenar por P(top 4)
    def p_top4(team):
        return sum(prob_matrix[team].get(p, 0) for p in range(1, 5))

    teams_sorted = sorted(prob_matrix.keys(), key=p_top4, reverse=True)

    print('\nTop 5 equipos por P(Liguilla directa):')
    for t in teams_sorted[:5]:
        print(f'  {t:20s}: {p_top4(t)*100:.1f}%')

    out = OUT_DIR / 'montecarlo_clausura2026.png'
    render(teams_sorted, prob_matrix, tabla_base, out)


if __name__ == '__main__':
    main()
