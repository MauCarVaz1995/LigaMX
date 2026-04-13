#!/usr/bin/env python3
"""
14_simulacion_montecarlo.py
Simulación de 10,000 escenarios del Clausura 2026.
Lee tabla actual + partidos restantes (jornadas 13-17) y simula con modelo Poisson.
Genera heatmap de probabilidades de posición final por equipo.
Salida: output/charts/montecarlo_clausura2026.png  (150 DPI)
"""

import argparse, json, glob, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from scipy.stats import poisson
from PIL import Image, ImageDraw
import urllib.request

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETTE, PALETAS, PALETA_ACTIVA, get_paleta, bebas, hex_rgba, hex_rgb

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
    try:
        fm.fontManager.addfont(str(BEBAS_TTF))
    except Exception:
        BEBAS_TTF = None

N_SIM = 10_000

# ─────────────────────────────────────────────────────────────────────────────
# PALETA  (se sobreescribe en main() según --paleta)
# ─────────────────────────────────────────────────────────────────────────────
_PAL  = get_paleta()          # paleta activa por defecto
BG    = _PAL['bg_primary']
WHITE = _PAL['text_primary']
GRAY  = _PAL['text_secondary']
RED   = _PAL['accent']

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
def render(teams_sorted, prob_matrix, tabla_base, output_path, pal=None):
    if pal is None:
        pal = get_paleta()
    _BG    = pal['bg_primary']
    _BG2   = pal['bg_secondary']
    _WHITE = pal['text_primary']
    _GRAY  = pal['text_secondary']
    _RED   = pal['accent']
    _ACC2  = pal['accent2']
    _BRAND = pal['brand_color']
    _CHIGH = pal['cell_high']
    _CMID  = pal['cell_mid']
    _CLOW  = pal['cell_low']
    _bg_rgb  = hex_rgb(_BG)
    _bg2_rgb = hex_rgb(_BG2)

    N_TEAMS = len(teams_sorted)
    N_POS   = N_TEAMS   # 18

    FIG_W, FIG_H = 12.0, 18.0
    AR = FIG_H / FIG_W     # 1.5  → used to make square badges

    TEAM_NAMES = {
        'América':       'AMÉRICA',   'Chivas':        'CHIVAS',
        'Cruz Azul':     'CRUZ AZUL', 'Tigres':        'TIGRES',
        'Monterrey':     'MONTERREY', 'Pumas':         'PUMAS',
        'Toluca':        'TOLUCA',    'Santos Laguna': 'SANTOS',
        'Pachuca':       'PACHUCA',   'Atlas':         'ATLAS',
        'León':          'LÉON',      'Necaxa':        'NECAXA',
        'Tijuana':       'TIJUANA',   'Querétaro':     'QUERÉTARO',
        'FC Juárez':     'JUÁREZ',    'Mazatlán':      'MAZATLÁN',
        'San Luis':      'SAN LUIS',  'Puebla':        'PUEBLA',
    }

    # Current standings (for diagonal highlighting)
    teams_by_rank = sorted(
        tabla_base.keys(),
        key=lambda t: (tabla_base[t]['pts'],
                       tabla_base[t]['gf'] - tabla_base[t]['gc'],
                       tabla_base[t]['gf']),
        reverse=True)
    current_pos = {t: i + 1 for i, t in enumerate(teams_by_rank)}

    # ── LAYOUT (inches) ───────────────────────────────────────────────────────
    HEADER_IN = 2.20
    FOOTER_IN = 1.20      # más alto para MAU-STATISTICS 28pt + leyenda 2 filas
    COLHDR_IN = 0.72
    LEFT_IN   = 1.92
    MARGIN_IN = 0.10
    # 18 pos cols + 2 summary cols (each 1.5× cell width)
    CW_IN  = (FIG_W - LEFT_IN - MARGIN_IN) / (N_POS + 2 * 1.5)
    SUM_IN = CW_IN * 1.5

    def n(v, total): return v / total

    HEADER_H = n(HEADER_IN, FIG_H)
    FOOTER_H = n(FOOTER_IN, FIG_H)
    COLHDR_H = n(COLHDR_IN, FIG_H)
    LEFT_X   = n(LEFT_IN,   FIG_W)
    POS_W    = n(CW_IN,     FIG_W)
    SUM_W    = n(SUM_IN,    FIG_W)

    CONTENT_Y = FOOTER_H
    CONTENT_H = 1.0 - HEADER_H - FOOTER_H - COLHDR_H
    ROW_H     = CONTENT_H / N_TEAMS

    pos_col_x = [LEFT_X + j * POS_W for j in range(N_POS)]
    sum4_x    = LEFT_X + N_POS * POS_W
    sum8_x    = sum4_x + SUM_W

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=_BG)

    # ── BACKGROUND (fondo sólido — bgax en zorder mínimo para no tapar celdas)
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = (np.array(_bg_rgb)/255*(1-t)
                   + np.array(_bg2_rgb)/255*t)
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.set_zorder(-100)          # SIEMPRE detrás de todas las celdas
    bgax.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bgax.axis('off')

    # ── HEADER ───────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(_BG2)
    hax.axis('off')
    hax.axhline(0, color=_RED, lw=4.0)
    hax.text(0.50, 0.90, '¿CÓMO TERMINA EL CLAUSURA 2026?',
             color=_WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(36))
    hax.text(0.50, 0.54,
             '10,000 simulaciones Monte Carlo · 45 partidos restantes · Jornadas 13–17',
             color=_GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=16)
    hax.text(0.50, 0.24,
             'Cada celda = % de simulaciones en que el equipo terminó en esa posición · '
             'Orden: P(Top 4) decreciente',
             color=_GRAY, ha='center', va='top',
             transform=hax.transAxes, fontsize=10)

    # ── HELPERS ───────────────────────────────────────────────────────────────
    def paint_cell(ax, fc):
        """Fondo sólido mediante Rectangle — evita el bug de axis('off') que oculta el patch."""
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=fc, edgecolor='none',
            zorder=0, transform=ax.transAxes, clip_on=False))
        ax.axis('off')

    # ── COLUMN HEADERS ───────────────────────────────────────────────────────
    col_hdr_y = CONTENT_Y + CONTENT_H

    # Zone banners (top half of colhdr) — paint_cell para fondo sólido
    zone_defs = [
        (0,  3,  '#2ea043', '#0a2210', 'LIGUILLA DIRECTA'),
        (4,  7,  '#d4a72c', '#261e00', 'REPECHAJE'),
        (8,  17, '#f85149', '#1a0404', ''),
    ]
    for c0, c1, fg, bg_z, zlbl in zone_defs:
        zx0 = pos_col_x[c0]
        zx1 = pos_col_x[c1] + POS_W
        zax = fig.add_axes([zx0, col_hdr_y + COLHDR_H*0.52, zx1-zx0, COLHDR_H*0.48])
        paint_cell(zax, bg_z)
        if zlbl:
            zax.text(0.5, 0.5, zlbl, color=fg, ha='center', va='center',
                     fontsize=8.5, fontweight='bold', transform=zax.transAxes)

    # Position number cells (bottom half of colhdr) — paint_cell
    for j in range(N_POS):
        pos = j + 1
        if pos <= 4:   hdr_bg, hdr_fg = '#1a4028', '#2ea043'
        elif pos <= 8: hdr_bg, hdr_fg = '#3a3200', '#d4a72c'
        else:          hdr_bg, hdr_fg = '#181e26', _GRAY
        cax = fig.add_axes([pos_col_x[j], col_hdr_y, POS_W, COLHDR_H*0.52])
        paint_cell(cax, hdr_bg)
        for sp in cax.spines.values():
            sp.set_edgecolor(_BG2); sp.set_linewidth(0.5)
        cax.text(0.5, 0.5, str(pos), color=hdr_fg, ha='center', va='center',
                 fontsize=10, fontweight='bold', transform=cax.transAxes)

    # Summary column headers — paint_cell + mismo nivel que pos cells
    for sx, lbl, fg_c, bg_c in [
            (sum4_x, 'TOP\n4', _CHIGH, _BG2),
            (sum8_x, 'TOP\n8', _CMID,  _BG)]:
        shax = fig.add_axes([sx, col_hdr_y, SUM_W, COLHDR_H])
        paint_cell(shax, bg_c)
        for sp in shax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(fg_c); sp.set_linewidth(1.8)
        shax.text(0.5, 0.5, lbl, color=fg_c, ha='center', va='center',
                  fontsize=11, fontweight='bold', transform=shax.transAxes)

    # ── CELL COLOR SCHEME — usa colores de la paleta activa ──────────────────
    def cell_style(prob):
        """(face_color, text_color, bold, font_size)"""
        if prob > 0.20:   return _CHIGH, '#000000', True,  15
        elif prob > 0.10: return _CMID,  _WHITE,    True,  13
        elif prob > 0.05: return _CLOW,  _CHIGH,    True,  11
        elif prob > 0.01: return _BG2,   _GRAY,     False, 10
        else:             return _BG,    _BG2,      False,  8

    BADGE_S = 96

    # ── DEBUG: primera fila (equipo con mayor P(top4)) ───────────────────────
    debug_team = teams_sorted[0]
    debug_row  = prob_matrix[debug_team]
    print(f'\n[DEBUG] Colores asignados — fila: {debug_team}')
    for j in range(min(6, N_POS)):
        p = debug_row.get(j + 1, 0.0)
        fc, tc, bold, fs = cell_style(p)
        print(f'  pos {j+1:2d}: prob={p*100:5.1f}%  fc={fc}  tc={tc}  fs={fs}')

    # ── TEAM ROWS ─────────────────────────────────────────────────────────────
    for i, team in enumerate(teams_sorted):
        prob_row = prob_matrix[team]
        row_y    = CONTENT_Y + (N_TEAMS - 1 - i) * ROW_H
        row_bg   = _BG2 if i % 2 == 0 else _BG
        diag_col = current_pos.get(team, 99) - 1

        # ── Left column: crest + full name ───────────────────────────────────
        lcol = fig.add_axes([0, row_y, LEFT_X, ROW_H])
        paint_cell(lcol, row_bg)
        lcol.axhline(1, color='#1e2631', lw=0.8)

        shield = get_shield(team, BADGE_S)
        if shield is not None:
            bh = ROW_H * 0.82
            bw = bh * AR
            sax = fig.add_axes([0.004, row_y + (ROW_H - bh) / 2, bw, bh])
            sax.set_facecolor('#f8f8fc')
            sax.patch.set_facecolor('#f8f8fc')
            sax.patch.set_visible(True)
            sax.imshow(shield); sax.axis('off')
        else:
            bw = 0.0

        full_name   = TEAM_NAMES.get(team, team.upper())
        name_x_frac = (0.004 + bw + 0.010) / LEFT_X
        lcol.text(name_x_frac, 0.50, full_name,
                  color=_WHITE, ha='left', va='center', fontsize=11.5, fontweight='bold')

        # ── Position cells ────────────────────────────────────────────────────
        for j in range(N_POS):
            prob  = prob_row.get(j + 1, 0.0)
            fc, tc, bold, fs = cell_style(prob)
            is_diag = (j == diag_col)

            cax = fig.add_axes([pos_col_x[j], row_y, POS_W, ROW_H])
            paint_cell(cax, fc)   # Rectangle sólido, nunca transparente

            # Bordes
            for sp in cax.spines.values():
                sp.set_visible(True)
                if is_diag:
                    sp.set_edgecolor('#ffffff'); sp.set_linewidth(2.0)
                else:
                    sp.set_edgecolor('#1e2631'); sp.set_linewidth(0.4)

            if prob >= 0.005:
                cax.text(0.5, 0.5, f'{prob*100:.0f}',
                         color=tc, ha='center', va='center',
                         fontsize=fs, fontweight='bold' if bold else 'normal',
                         transform=cax.transAxes, zorder=5)

        # ── Summary columns: P(Top4) y P(Top8) ───────────────────────────────
        p4 = sum(prob_row.get(p, 0) for p in range(1, 5))
        p8 = sum(prob_row.get(p, 0) for p in range(1, 9))

        for sx, pval, bg_s, border_c, hi_thresh, mid_thresh in [
                (sum4_x, p4, _BG2, _CHIGH, 0.50, 0.20),
                (sum8_x, p8, _BG,  _CMID,  0.70, 0.35)]:
            sumax = fig.add_axes([sx, row_y, SUM_W, ROW_H])
            paint_cell(sumax, bg_s)
            sumax.axhline(1, color='#1e2631', lw=0.5)
            for sp in sumax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(border_c); sp.set_linewidth(1.2)

            if pval >= 0.995:          # muestra "100%" redondeado → accent2
                tc_s, fs_s = _ACC2,  16
            elif pval >= hi_thresh:
                tc_s, fs_s = _CHIGH, 16
            elif pval >= mid_thresh:
                tc_s, fs_s = _CMID,  14
            else:
                tc_s, fs_s = _GRAY,  12

            sumax.text(0.5, 0.5, f'{pval*100:.0f}%',
                       color=tc_s, ha='center', va='center',
                       fontsize=fs_s, fontweight='bold',
                       transform=sumax.transAxes, zorder=5)

    # ── LEGEND (2 filas) ──────────────────────────────────────────────────────
    # Fila superior: cuadros de color
    row1_y = FOOTER_H * 0.72
    row1_h = FOOTER_H * 0.20
    lax = fig.add_axes([0.01, row1_y, 0.98, row1_h])
    lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis('off')
    legend_items = [
        ('>20%',  _CHIGH),
        ('10–20%', _CMID),
        ('5–10%',  _CLOW),
        ('1–5%',   _BG2),
        ('<1%',    _BG),
    ]
    step = 0.18          # cada item ocupa 18% del ancho
    for k, (lbl, fc) in enumerate(legend_items):
        bx = k * step
        lax.add_patch(mpatches.Rectangle(
            (bx, 0.05), step * 0.22, 0.90,
            facecolor=fc, edgecolor='#555', lw=1.0))
        lax.text(bx + step * 0.25, 0.50, lbl,
                 color=_WHITE, va='center', fontsize=11)
    # nota al final
    lax.text(0.92, 0.50, '□ posición actual',
             color=_GRAY, va='center', ha='left', fontsize=10)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.52])
    paint_cell(fax, _BG2)
    fax.axhline(1, color=_RED, lw=3.0)
    fax.text(0.015, 0.50, 'Fuente: FotMob · Clausura 2026',
             color=_GRAY, fontsize=11, ha='left', va='center', transform=fax.transAxes)

    shadow = [mpe.withStroke(linewidth=4, foreground='#000000')]
    fax.text(0.985, 0.50, 'MAU-STATISTICS',
             color=_BRAND, ha='right', va='center',
             transform=fax.transAxes,
             path_effects=shadow,
             **bebas(28))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'✓ Guardado: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paleta', default=None,
                        choices=list(PALETAS.keys()),
                        help='Nombre de la paleta (default: PALETA_ACTIVA)')
    args = parser.parse_args()
    pal = get_paleta(args.paleta)

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

    suffix = f'_{args.paleta}' if args.paleta else ''
    out = OUT_DIR / f'montecarlo_clausura2026{suffix}.png'
    render(teams_sorted, prob_matrix, tabla_base, out, pal=pal)


if __name__ == '__main__':
    main()
