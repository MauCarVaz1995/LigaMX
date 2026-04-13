#!/usr/bin/env python3
"""
14_paletas_montecarlo.py
Genera 6 versiones del heatmap Monte Carlo Clausura 2026, cada una con
una paleta de colores diferente. Corre la simulación UNA sola vez y reutiliza
los resultados para todas las versiones.
Salida: output/charts/paletas/montecarlo_*.png  (150 DPI)
"""

import json, glob, sys, warnings
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
from PIL import Image

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETTE, bebas, hex_rgba, hex_rgb

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
HIST_DIR  = BASE / 'data/raw/historico'
IMG_TEAMS = BASE / 'data/raw/images/teams'
OUT_DIR   = BASE / 'output/charts/paletas'
# Fuente centralizada desde config_visual (incluye assets/fonts/ como candidato)
try:
    from config_visual import BEBAS_TTF, BEBAS_AVAILABLE
except ImportError:
    BEBAS_TTF = None
    BEBAS_AVAILABLE = False

OUT_DIR.mkdir(parents=True, exist_ok=True)


N_SIM = 10_000

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
    '2025/2026 - Clausura': 4, '2025/2026 - Apertura': 3,
    '2024/2025 - Clausura': 2, '2024/2025 - Apertura': 1,
}
_ALIAS = {
    'cf america':'América','america':'América','chivas':'Chivas',
    'guadalajara':'Chivas','cruz azul':'Cruz Azul','tigres':'Tigres',
    'tigres uanl':'Tigres','monterrey':'Monterrey','cf monterrey':'Monterrey',
    'pumas':'Pumas','pumas unam':'Pumas','toluca':'Toluca',
    'santos laguna':'Santos Laguna','santos':'Santos Laguna',
    'pachuca':'Pachuca','atlas':'Atlas','león':'León','leon':'León',
    'necaxa':'Necaxa','tijuana':'Tijuana','querétaro':'Querétaro',
    'queretaro':'Querétaro','queretaro fc':'Querétaro',
    'fc juárez':'FC Juárez','fc juarez':'FC Juárez',
    'mazatlán':'Mazatlán','mazatlan fc':'Mazatlán','mazatlan':'Mazatlán',
    'atletico de san luis':'San Luis','san luis':'San Luis','puebla':'Puebla',
}
def norm(n): return _ALIAS.get(str(n).lower().strip(), n.strip())

TEAM_NAMES = {
    'América':'AMÉRICA','Chivas':'CHIVAS','Cruz Azul':'CRUZ AZUL',
    'Tigres':'TIGRES','Monterrey':'MONTERREY','Pumas':'PUMAS',
    'Toluca':'TOLUCA','Santos Laguna':'SANTOS','Pachuca':'PACHUCA',
    'Atlas':'ATLAS','León':'LÉON','Necaxa':'NECAXA','Tijuana':'TIJUANA',
    'Querétaro':'QUERÉTARO','FC Juárez':'JUÁREZ','Mazatlán':'MAZATLÁN',
    'San Luis':'SAN LUIS','Puebla':'PUEBLA',
}

# ─────────────────────────────────────────────────────────────────────────────
# 6 PALETAS
# ─────────────────────────────────────────────────────────────────────────────
# Cada paleta es un dict con TODAS las claves de color que render() consume.
# cell_thresholds: lista de (prob_min, face_color, text_color, bold, font_size)
#                  de MAYOR a MENOR probabilidad (primer match gana).
# brand: lista de (text, color) — se concatenan en el footer.
# legend_items: lista de (label, face_color)

PALETAS = [
    # ── A ─ Cyberpunk Quetzal ────────────────────────────────────────────────
    dict(
        filename   = 'montecarlo_cyberpunk_quetzal.png',
        label      = 'A · CYBERPUNK QUETZAL',
        bg_from    = (0x00, 0x00, 0x00),      # gradiente fondo superior
        bg_to      = (0x0a, 0x0a, 0x15),      # gradiente fondo inferior
        hdr_bg     = '#09091a',
        hdr_line   = '#e040fb',               # magenta
        row_bg_even= '#0a0a15',
        row_bg_odd = '#060612',
        cell_sep   = '#1a0f2e',
        cell_diag_border = '#e040fb',         # magenta para posición actual
        cell_thresholds  = [
            (0.20, '#00FF88', '#000000', True,  15),
            (0.10, '#00C853', '#ffffff', True,  13),
            (0.05, '#1a5c3a', '#00FF88', True,  11),
            (0.01, '#1a3a2a', '#555555', False, 10),
            (0.00, '#050510', '#111111', False,  8),
        ],
        zone_lig    = ('#0a1a26', '#e040fb'),  # (bg, fg)
        zone_rep    = ('#1a0a26', '#b040fb'),
        colhdr_lig  = ('#0f1a30', '#e040fb'),
        colhdr_rep  = ('#1a0f30', '#b040fb'),
        colhdr_rest = ('#08080f', '#444466'),
        sum4 = dict(bg='#091826', border='#00FF88', label='#00FF88',
                    hi_thresh=0.50, mid_thresh=0.20,
                    c100='#e040fb', c_hi='#00FF88', c_mid='#00C853', c_lo='#6e7681'),
        sum8 = dict(bg='#091826', border='#00C853', label='#00C853',
                    hi_thresh=0.70, mid_thresh=0.35,
                    c100='#e040fb', c_hi='#00FF88', c_mid='#00C853', c_lo='#6e7681'),
        legend = [('>20%','#00FF88'),('10–20%','#00C853'),('5–10%','#1a5c3a'),
                  ('1–5%','#1a3a2a'),('<1%','#050510')],
        footer_bg    = '#09091a',
        footer_line  = '#e040fb',
        text_primary = '#ffffff',
        text_muted   = '#9090c0',
        text_source  = '#6060a0',
        brand        = [('MAU-STATISTICS', '#00FF88')],
    ),

    # ── B ─ Matrix Neón ───────────────────────────────────────────────────────
    dict(
        filename   = 'montecarlo_matrix_neon.png',
        label      = 'B · MATRIX NEÓN',
        bg_from    = (0x05, 0x05, 0x10),
        bg_to      = (0x0d, 0x0d, 0x20),
        hdr_bg     = '#0a0a1a',
        hdr_line   = '#00e5ff',               # cyan
        row_bg_even= '#0d0d20',
        row_bg_odd = '#08081a',
        cell_sep   = '#0d1a0d',
        cell_diag_border = '#00e5ff',
        cell_thresholds  = [
            (0.20, '#76ff03', '#000000', True,  15),
            (0.10, '#4caf50', '#ffffff', True,  13),
            (0.05, '#2e7d32', '#ffffff', True,  11),
            (0.01, '#1a2a1a', '#a0a0a0', False, 10),
            (0.00, '#080808', '#111111', False,  8),
        ],
        zone_lig    = ('#001a0d', '#00e5ff'),
        zone_rep    = ('#001520', '#00b0cc'),
        colhdr_lig  = ('#001a0d', '#00e5ff'),
        colhdr_rep  = ('#001520', '#00b0cc'),
        colhdr_rest = ('#080810', '#334433'),
        sum4 = dict(bg='#001a0d', border='#76ff03', label='#76ff03',
                    hi_thresh=0.50, mid_thresh=0.20,
                    c100='#00e5ff', c_hi='#76ff03', c_mid='#4caf50', c_lo='#6e7681'),
        sum8 = dict(bg='#001a0d', border='#4caf50', label='#4caf50',
                    hi_thresh=0.70, mid_thresh=0.35,
                    c100='#00e5ff', c_hi='#76ff03', c_mid='#4caf50', c_lo='#6e7681'),
        legend = [('>20%','#76ff03'),('10–20%','#4caf50'),('5–10%','#2e7d32'),
                  ('1–5%','#1a2a1a'),('<1%','#080808')],
        footer_bg    = '#0a0a1a',
        footer_line  = '#00e5ff',
        text_primary = '#ffffff',
        text_muted   = '#80a080',
        text_source  = '#406040',
        brand        = [('MAU-STATISTICS', '#76ff03')],
    ),

    # ── C ─ Negro Selva (monocromático verde) ─────────────────────────────────
    dict(
        filename   = 'montecarlo_negro_selva.png',
        label      = 'C · NEGRO SELVA',
        bg_from    = (0x00, 0x00, 0x00),
        bg_to      = (0x07, 0x1a, 0x10),
        hdr_bg     = '#030e08',
        hdr_line   = '#00FF88',
        row_bg_even= '#071a10',
        row_bg_odd = '#040f09',
        cell_sep   = '#0a2015',
        cell_diag_border = '#b2dfdb',
        cell_thresholds  = [
            (0.20, '#00FF88', '#000000', True,  15),
            (0.10, '#00C853', '#b2dfdb', True,  13),
            (0.05, '#2e7d32', '#b2dfdb', True,  11),
            (0.01, '#0a2a15', '#336644', False, 10),
            (0.00, '#040f09', '#0d1f12', False,  8),
        ],
        zone_lig    = ('#0a2815', '#00FF88'),
        zone_rep    = ('#061a0e', '#00C853'),
        colhdr_lig  = ('#0a2815', '#00FF88'),
        colhdr_rep  = ('#061a0e', '#00C853'),
        colhdr_rest = ('#040f09', '#1a3a22'),
        sum4 = dict(bg='#0a2815', border='#00FF88', label='#00FF88',
                    hi_thresh=0.50, mid_thresh=0.20,
                    c100='#b2dfdb', c_hi='#00FF88', c_mid='#00C853', c_lo='#336644'),
        sum8 = dict(bg='#061a0e', border='#00C853', label='#00C853',
                    hi_thresh=0.70, mid_thresh=0.35,
                    c100='#b2dfdb', c_hi='#00FF88', c_mid='#00C853', c_lo='#336644'),
        legend = [('>20%','#00FF88'),('10–20%','#00C853'),('5–10%','#2e7d32'),
                  ('1–5%','#0a2a15'),('<1%','#040f09')],
        footer_bg    = '#030e08',
        footer_line  = '#00FF88',
        text_primary = '#b2dfdb',
        text_muted   = '#669977',
        text_source  = '#336644',
        brand        = [('MAU-STATISTICS', '#00FF88')],
    ),

    # ── D ─ Medianoche Neón Dual ──────────────────────────────────────────────
    dict(
        filename   = 'montecarlo_medianoche_neon.png',
        label      = 'D · MEDIANOCHE NEÓN DUAL',
        bg_from    = (0x08, 0x08, 0x0f),
        bg_to      = (0x12, 0x12, 0x1f),
        hdr_bg     = '#0c0c1a',
        hdr_line   = '#ff2d7b',              # rosa neón
        row_bg_even= '#10101e',
        row_bg_odd = '#0a0a16',
        cell_sep   = '#1a1a2e',
        cell_diag_border = '#ff2d7b',
        cell_thresholds  = [
            (0.20, '#00ffaa', '#000000', True,  15),
            (0.10, '#00aa77', '#ffffff', True,  13),
            (0.05, '#006644', '#00ffaa', True,  11),
            (0.01, '#0a1a15', '#445566', False, 10),
            (0.00, '#080810', '#131320', False,  8),
        ],
        zone_lig    = ('#1a0a18', '#ff2d7b'),  # rosa neón para top 4
        zone_rep    = ('#0a0a18', '#cc1a5e'),
        colhdr_lig  = ('#1a0a18', '#ff2d7b'),
        colhdr_rep  = ('#0a0a18', '#cc1a5e'),
        colhdr_rest = ('#08080f', '#333355'),
        sum4 = dict(bg='#0a1a15', border='#00ffaa', label='#00ffaa',
                    hi_thresh=0.50, mid_thresh=0.20,
                    c100='#ff2d7b', c_hi='#00ffaa', c_mid='#00aa77', c_lo='#6e7681'),
        sum8 = dict(bg='#0a1a15', border='#00aa77', label='#00aa77',
                    hi_thresh=0.70, mid_thresh=0.35,
                    c100='#ff2d7b', c_hi='#00ffaa', c_mid='#00aa77', c_lo='#6e7681'),
        legend = [('>20%','#00ffaa'),('10–20%','#00aa77'),('5–10%','#006644'),
                  ('1–5%','#0a1a15'),('<1%','#080810')],
        footer_bg    = '#0c0c1a',
        footer_line  = '#ff2d7b',
        text_primary = '#ffffff',
        text_muted   = '#8888bb',
        text_source  = '#555588',
        brand        = [('MAU', '#00ffaa'), ('STATISTICS', '#ff2d7b')],
    ),

    # ── E ─ Radioactivo ───────────────────────────────────────────────────────
    dict(
        filename   = 'montecarlo_radioactivo.png',
        label      = 'E · RADIOACTIVO',
        bg_from    = (0x00, 0x00, 0x00),
        bg_to      = (0x0f, 0x0f, 0x0f),
        hdr_bg     = '#080808',
        hdr_line   = '#39ff14',
        row_bg_even= '#0f0f0f',
        row_bg_odd = '#080808',
        cell_sep   = '#1a1a1a',
        cell_diag_border = '#b6ff00',         # lima ácida
        cell_thresholds  = [
            (0.20, '#39ff14', '#000000', True,  15),
            (0.10, '#2abf10', '#ffffff', True,  13),
            (0.05, '#1a7a0c', '#39ff14', True,  11),
            (0.01, '#0f1f0a', '#446633', False, 10),
            (0.00, '#080808', '#0f0f0f', False,  8),
        ],
        zone_lig    = ('#0f1a08', '#39ff14'),
        zone_rep    = ('#121208', '#b6ff00'),
        colhdr_lig  = ('#0f1a08', '#39ff14'),
        colhdr_rep  = ('#121208', '#b6ff00'),
        colhdr_rest = ('#080808', '#333322'),
        sum4 = dict(bg='#0f1a08', border='#39ff14', label='#39ff14',
                    hi_thresh=0.50, mid_thresh=0.20,
                    c100='#b6ff00', c_hi='#39ff14', c_mid='#2abf10', c_lo='#6e7681'),
        sum8 = dict(bg='#0f1a08', border='#2abf10', label='#2abf10',
                    hi_thresh=0.70, mid_thresh=0.35,
                    c100='#b6ff00', c_hi='#39ff14', c_mid='#2abf10', c_lo='#6e7681'),
        legend = [('>20%','#39ff14'),('10–20%','#2abf10'),('5–10%','#1a7a0c'),
                  ('1–5%','#0f1f0a'),('<1%','#080808')],
        footer_bg    = '#080808',
        footer_line  = '#39ff14',
        text_primary = '#ffffff',
        text_muted   = '#708060',
        text_source  = '#405030',
        brand        = [('MAU-STATISTICS', '#39ff14')],
    ),

    # ── F ─ Océano Esmeralda ──────────────────────────────────────────────────
    dict(
        filename   = 'montecarlo_oceano_esmeralda.png',
        label      = 'F · OCÉANO ESMERALDA',
        bg_from    = (0x04, 0x08, 0x12),
        bg_to      = (0x0c, 0x14, 0x20),
        hdr_bg     = '#060e1c',
        hdr_line   = '#69f0ae',              # menta para top 4
        row_bg_even= '#0a1220',
        row_bg_odd = '#060e18',
        cell_sep   = '#0f1e2e',
        cell_diag_border = '#69f0ae',
        cell_thresholds  = [
            (0.20, '#00e676', '#000000', True,  15),
            (0.10, '#009955', '#e0f2f1', True,  13),
            (0.05, '#00604a', '#e0f2f1', True,  11),
            (0.01, '#0a1a1a', '#336655', False, 10),
            (0.00, '#060e18', '#0d1a1a', False,  8),
        ],
        zone_lig    = ('#061a18', '#69f0ae'),  # menta para liguilla
        zone_rep    = ('#061518', '#00e676'),
        colhdr_lig  = ('#061a18', '#69f0ae'),
        colhdr_rep  = ('#061518', '#00e676'),
        colhdr_rest = ('#060e18', '#1a3344'),
        sum4 = dict(bg='#061a18', border='#69f0ae', label='#69f0ae',
                    hi_thresh=0.50, mid_thresh=0.20,
                    c100='#69f0ae', c_hi='#00e676', c_mid='#009955', c_lo='#6e7681'),
        sum8 = dict(bg='#061518', border='#00e676', label='#00e676',
                    hi_thresh=0.70, mid_thresh=0.35,
                    c100='#69f0ae', c_hi='#00e676', c_mid='#009955', c_lo='#6e7681'),
        legend = [('>20%','#00e676'),('10–20%','#009955'),('5–10%','#00604a'),
                  ('1–5%','#0a1a1a'),('<1%','#060e18')],
        footer_bg    = '#060e1c',
        footer_line  = '#69f0ae',
        text_primary = '#e0f2f1',
        text_muted   = '#669977',
        text_source  = '#336655',
        brand        = [('MAU-STATISTICS', '#00e676')],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL (igual que el script original)
# ─────────────────────────────────────────────────────────────────────────────
def load_current_data():
    fpath = HIST_DIR / 'historico_clausura_2026.json'
    data  = json.load(open(fpath, encoding='utf-8'))
    tabla = {}
    for row in data.get('tabla', []):
        eq = norm(row['equipo'])
        tabla[eq] = {
            'pts': int(row.get('pts', 0)), 'pj':  int(row.get('pj',  0)),
            'gf':  int(row.get('gf',  0)), 'gc':  int(row.get('gc',  0)),
        }
    pendientes = [
        {'local': norm(p['local']), 'visitante': norm(p['visitante'])}
        for p in data.get('partidos', []) if not p.get('terminado')
    ]
    return tabla, pendientes

def load_model_matches():
    rows = []
    for fpath in sorted(glob.glob(str(HIST_DIR / '*.json'))):
        stem  = Path(fpath).stem.replace('historico_', '')
        parts = stem.replace('-', '/', 1).split('_-_', 1)
        if len(parts) != 2: continue
        tkey = f"{parts[0].replace('_','/')} - {parts[1].replace('_',' ').title()}"
        w = TORNEO_WEIGHTS.get(tkey)
        if not w: continue
        d = json.load(open(fpath, encoding='utf-8'))
        for p in d.get('partidos', []):
            if not p.get('terminado'): continue
            if p.get('goles_local') is None or p.get('goles_visit') is None: continue
            rows.append({'peso': w, 'local': norm(p['local']),
                         'visitante': norm(p['visitante']),
                         'gl': int(p['goles_local']), 'gv': int(p['goles_visit'])})
    return rows

def build_model(matches):
    s_gl = s_gv = s_w = 0.0
    hs = defaultdict(float); hc = defaultdict(float)
    as_ = defaultdict(float); ac = defaultdict(float)
    hg = defaultdict(float);  ag = defaultdict(float)
    for m in matches:
        w,lo,vi,gl,gv = m['peso'],m['local'],m['visitante'],m['gl'],m['gv']
        s_gl+=gl*w; s_gv+=gv*w; s_w+=w
        hs[lo]+=gl*w; hc[lo]+=gv*w; as_[vi]+=gv*w; ac[vi]+=gl*w
        hg[lo]+=w;   ag[vi]+=w
    mu_h = s_gl/s_w; mu_a = s_gv/s_w
    att = {}; defe = {}
    for t in set(hs)|set(as_):
        att[t]  = {'home': (hs[t] /hg[t]/mu_h) if hg[t]>0 else 1.0,
                   'away': (as_[t]/ag[t]/mu_a) if ag[t]>0 else 1.0}
        defe[t] = {'home': (hc[t] /hg[t]/mu_a) if hg[t]>0 else 1.0,
                   'away': (ac[t] /ag[t]/mu_h) if ag[t]>0 else 1.0}
    return {'mu_home': mu_h, 'mu_away': mu_a, 'att': att, 'defe': defe}

def get_lambda(model, lo, vi):
    mu_h = model['mu_home']; mu_a = model['mu_away']
    att  = model['att'];     defe = model['defe']
    ll = att.get(lo,{}).get('home',1.)*defe.get(vi,{}).get('away',1.)*mu_h
    lv = att.get(vi,{}).get('away',1.)*defe.get(lo,{}).get('home',1.)*mu_a
    return max(ll, 0.2), max(lv, 0.2)

def simulate_one(model, tabla_base, pendientes, rng):
    pts = {t: v['pts'] for t,v in tabla_base.items()}
    gf  = {t: v['gf']  for t,v in tabla_base.items()}
    gc  = {t: v['gc']  for t,v in tabla_base.items()}
    for p in pendientes:
        lo,vi = p['local'],p['visitante']
        for k in (lo,vi):
            if k not in pts: pts[k]=gf[k]=gc[k]=0
        ll,lv = get_lambda(model, lo, vi)
        gl = rng.poisson(ll); gv = rng.poisson(lv)
        gf[lo]+=gl; gc[lo]+=gv; gf[vi]+=gv; gc[vi]+=gl
        if gl>gv: pts[lo]+=3
        elif gl==gv: pts[lo]+=1; pts[vi]+=1
        else: pts[vi]+=3
    teams = sorted(pts, key=lambda t:(pts[t],gf[t]-gc[t],gf[t]), reverse=True)
    return {t: i+1 for i,t in enumerate(teams)}

def get_shield(team_name, size=64):
    key = {'América':'CF America','FC Juárez':'FC Juarez',
           'Mazatlán':'Mazatlan FC','San Luis':'Atletico de San Luis',
           'Querétaro':'Queretaro FC'}.get(team_name, team_name)
    tid = TEAM_IDS.get(key) or TEAM_IDS.get(team_name)
    if not tid: return None
    cache = IMG_TEAMS / f'{tid}.png'
    if not cache.exists(): return None
    try:
        img = Image.open(cache).convert('RGBA')
        s = min(size, img.size[0], img.size[1])
        if s != img.size[0] or s != img.size[1]:
            img = img.resize((s, s), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# RENDER — COMPLETAMENTE PARAMETRIZADO POR PALETA
# ─────────────────────────────────────────────────────────────────────────────
def render(teams_sorted, prob_matrix, tabla_base, pal, output_path):
    N     = len(teams_sorted)
    FIG_W = 12.0; FIG_H = 18.0
    AR    = FIG_H / FIG_W

    # Current standings for diagonal marker
    teams_by_rank = sorted(
        tabla_base,
        key=lambda t:(tabla_base[t]['pts'],
                      tabla_base[t]['gf']-tabla_base[t]['gc'],
                      tabla_base[t]['gf']),
        reverse=True)
    current_pos = {t: i+1 for i,t in enumerate(teams_by_rank)}

    HEADER_IN = 2.20; FOOTER_IN = 1.20; COLHDR_IN = 0.72
    LEFT_IN   = 1.92; MARGIN_IN = 0.10
    CW_IN     = (FIG_W - LEFT_IN - MARGIN_IN) / (N + 2 * 1.5)
    SUM_IN    = CW_IN * 1.5

    def n(v, tot): return v / tot
    HEADER_H = n(HEADER_IN, FIG_H); FOOTER_H = n(FOOTER_IN, FIG_H)
    COLHDR_H = n(COLHDR_IN, FIG_H); LEFT_X   = n(LEFT_IN,   FIG_W)
    POS_W    = n(CW_IN,     FIG_W); SUM_W    = n(SUM_IN,    FIG_W)
    CONTENT_Y = FOOTER_H
    CONTENT_H = 1.0 - HEADER_H - FOOTER_H - COLHDR_H
    ROW_H     = CONTENT_H / N
    pos_col_x = [LEFT_X + j * POS_W for j in range(N)]
    sum4_x    = LEFT_X + N * POS_W
    sum8_x    = sum4_x + SUM_W

    bg_col = tuple(c/255 for c in pal['bg_from'])
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=bg_col)

    # ── Background gradient ──────────────────────────────────────────────────
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = (np.array(pal['bg_from'])/255*(1-t)
                   + np.array(pal['bg_to'])/255*t)
    bgax = fig.add_axes([0, 0, 1, 1]); bgax.set_zorder(-100)
    bgax.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bgax.axis('off')

    def paint_cell(ax, fc):
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.add_patch(mpatches.Rectangle(
            (0,0),1,1, facecolor=fc, edgecolor='none',
            zorder=0, transform=ax.transAxes, clip_on=False))
        ax.axis('off')

    def cell_color(prob):
        for thresh, fc, tc, bold, fs in pal['cell_thresholds']:
            if prob > thresh:
                return fc, tc, bold, fs
        fc, tc, bold, fs = pal['cell_thresholds'][-1][1:]
        return fc, tc, bold, fs

    # ── Header ───────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    hax.set_facecolor(pal['hdr_bg']); hax.axis('off')
    hax.axhline(0, color=pal['hdr_line'], lw=4.0)
    hax.text(0.50, 0.90, '¿CÓMO TERMINA EL CLAUSURA 2026?',
             color=pal['text_primary'], ha='center', va='top',
             transform=hax.transAxes, **bebas(36))
    hax.text(0.50, 0.54,
             '10,000 simulaciones Monte Carlo · 45 partidos restantes · Jornadas 13–17',
             color=pal['text_muted'], ha='center', va='top',
             transform=hax.transAxes, fontsize=16)
    hax.text(0.50, 0.24,
             'Cada celda = % de simulaciones en que el equipo terminó en esa posición · '
             'Orden: P(Top 4) decreciente',
             color=pal['text_source'], ha='center', va='top',
             transform=hax.transAxes, fontsize=10)

    # ── Zone banners (colhdr) ────────────────────────────────────────────────
    col_hdr_y = CONTENT_Y + CONTENT_H
    lig_bg, lig_fg = pal['zone_lig']
    rep_bg, rep_fg = pal['zone_rep']
    zone_defs = [
        (0,  3, lig_fg, lig_bg, 'LIGUILLA DIRECTA'),
        (4,  7, rep_fg, rep_bg, 'REPECHAJE'),
        (8, 17, '#888888', pal['colhdr_rest'][0], ''),
    ]
    for c0, c1, fg, bg_z, zlbl in zone_defs:
        zx0 = pos_col_x[c0]; zx1 = pos_col_x[c1] + POS_W
        zax = fig.add_axes([zx0, col_hdr_y + COLHDR_H*0.52, zx1-zx0, COLHDR_H*0.48])
        paint_cell(zax, bg_z)
        if zlbl:
            zax.text(0.5, 0.5, zlbl, color=fg, ha='center', va='center',
                     fontsize=8.5, fontweight='bold', transform=zax.transAxes)

    # Position number cells
    chdr_lig_bg, chdr_lig_fg = pal['colhdr_lig']
    chdr_rep_bg, chdr_rep_fg = pal['colhdr_rep']
    chdr_rest_bg, chdr_rest_fg = pal['colhdr_rest']
    for j in range(N):
        pos = j + 1
        if pos <= 4:   hbg, hfg = chdr_lig_bg, chdr_lig_fg
        elif pos <= 8: hbg, hfg = chdr_rep_bg, chdr_rep_fg
        else:          hbg, hfg = chdr_rest_bg, chdr_rest_fg
        cax = fig.add_axes([pos_col_x[j], col_hdr_y, POS_W, COLHDR_H*0.52])
        paint_cell(cax, hbg)
        for sp in cax.spines.values():
            sp.set_edgecolor(pal['cell_sep']); sp.set_linewidth(0.5)
        cax.text(0.5, 0.5, str(pos), color=hfg, ha='center', va='center',
                 fontsize=10, fontweight='bold', transform=cax.transAxes)

    # Summary col headers
    s4 = pal['sum4']; s8 = pal['sum8']
    for sx, cfg in [(sum4_x, s4), (sum8_x, s8)]:
        shax = fig.add_axes([sx, col_hdr_y, SUM_W, COLHDR_H])
        paint_cell(shax, cfg['bg'])
        for sp in shax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(cfg['border']); sp.set_linewidth(1.8)
        label_text = 'TOP\n4' if sx == sum4_x else 'TOP\n8'
        shax.text(0.5, 0.5, label_text, color=cfg['label'], ha='center', va='center',
                  fontsize=11, fontweight='bold', transform=shax.transAxes)

    # ── Team rows ─────────────────────────────────────────────────────────────
    BADGE_S = 96
    for i, team in enumerate(teams_sorted):
        prob_row = prob_matrix[team]
        row_y    = CONTENT_Y + (N - 1 - i) * ROW_H
        row_bg   = pal['row_bg_even'] if i%2==0 else pal['row_bg_odd']
        diag_col = current_pos.get(team, 99) - 1

        # Left column
        lcol = fig.add_axes([0, row_y, LEFT_X, ROW_H])
        paint_cell(lcol, row_bg)
        lcol.axhline(1, color=pal['cell_sep'], lw=0.8)
        shield = get_shield(team, BADGE_S)
        if shield is not None:
            bh = ROW_H * 0.82; bw = bh * AR
            sax = fig.add_axes([0.004, row_y + (ROW_H-bh)/2, bw, bh])
            sax.set_facecolor('#f8f8fc'); sax.patch.set_facecolor('#f8f8fc')
            sax.patch.set_visible(True); sax.imshow(shield); sax.axis('off')
        else:
            bw = 0.0
        full_name = TEAM_NAMES.get(team, team.upper())
        nx = (0.004 + bw + 0.010) / LEFT_X
        lcol.text(nx, 0.50, full_name, color=pal['text_primary'],
                  ha='left', va='center', fontsize=11.5, fontweight='bold')

        # Position cells
        for j in range(N):
            prob = prob_row.get(j+1, 0.0)
            fc, tc, bold, fs = cell_color(prob)
            is_diag = (j == diag_col)
            cax = fig.add_axes([pos_col_x[j], row_y, POS_W, ROW_H])
            paint_cell(cax, fc)
            for sp in cax.spines.values():
                sp.set_visible(True)
                if is_diag:
                    sp.set_edgecolor(pal['cell_diag_border']); sp.set_linewidth(2.0)
                else:
                    sp.set_edgecolor(pal['cell_sep']); sp.set_linewidth(0.4)
            if prob >= 0.005:
                cax.text(0.5, 0.5, f'{prob*100:.0f}',
                         color=tc, ha='center', va='center',
                         fontsize=fs, fontweight='bold' if bold else 'normal',
                         transform=cax.transAxes, zorder=5)

        # Summary columns
        p4 = sum(prob_row.get(p,0) for p in range(1,5))
        p8 = sum(prob_row.get(p,0) for p in range(1,9))
        for sx, pval, cfg in [(sum4_x, p4, s4), (sum8_x, p8, s8)]:
            sumax = fig.add_axes([sx, row_y, SUM_W, ROW_H])
            paint_cell(sumax, cfg['bg'])
            sumax.axhline(1, color=pal['cell_sep'], lw=0.5)
            for sp in sumax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(cfg['border']); sp.set_linewidth(1.2)
            if pval >= 0.995:              tc_s, fs_s = cfg['c100'], 16
            elif pval >= cfg['hi_thresh']: tc_s, fs_s = cfg['c_hi'],  16
            elif pval >= cfg['mid_thresh']:tc_s, fs_s = cfg['c_mid'], 14
            else:                          tc_s, fs_s = cfg['c_lo'],  12
            sumax.text(0.5, 0.5, f'{pval*100:.0f}%', color=tc_s,
                       ha='center', va='center', fontsize=fs_s, fontweight='bold',
                       transform=sumax.transAxes, zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────────
    row1_y = FOOTER_H * 0.72; row1_h = FOOTER_H * 0.20
    lax = fig.add_axes([0.01, row1_y, 0.98, row1_h])
    lax.set_xlim(0,1); lax.set_ylim(0,1); lax.axis('off')
    step = 0.18
    for k, (lbl, fc) in enumerate(pal['legend']):
        bx = k * step
        lax.add_patch(mpatches.Rectangle(
            (bx, 0.05), step*0.22, 0.90, facecolor=fc, edgecolor='#555', lw=1.0))
        lax.text(bx + step*0.25, 0.50, lbl,
                 color=pal['text_primary'], va='center', fontsize=11)
    lax.text(0.92, 0.50, '□ posición actual',
             color=pal['text_muted'], va='center', ha='left', fontsize=10)

    # ── Footer ────────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.52])
    paint_cell(fax, pal['footer_bg'])
    fax.axhline(1, color=pal['footer_line'], lw=3.0)
    fax.text(0.015, 0.50, 'Fuente: FotMob · Clausura 2026',
             color=pal['text_source'], fontsize=11,
             ha='left', va='center', transform=fax.transAxes)

    shadow = [mpe.withStroke(linewidth=4, foreground='#000000')]
    # Brand — puede ser una o dos partes (versión D)
    brand_parts = pal['brand']
    if len(brand_parts) == 1:
        fax.text(0.985, 0.50, brand_parts[0][0],
                 color=brand_parts[0][1], ha='right', va='center',
                 transform=fax.transAxes, path_effects=shadow, **bebas(28))
    else:
        # Dos partes: renderizar separadas y unidas
        t1, c1 = brand_parts[0]; t2, c2 = brand_parts[1]
        # Estimamos el ancho de t1 en fracciones del eje
        total   = len(t1) + len(t2)
        frac_t1 = len(t1) / total
        # Dibujamos ambas partes como si fueran consecutivas alineadas a la derecha
        combined = t1 + t2
        # Simple approach: renderizar el texto combinado con dos llamadas de texto
        # posicionadas aproximadamente. Usamos annotate para concatenar visualmente.
        fax.text(0.985, 0.50, t2, color=c2, ha='right', va='center',
                 transform=fax.transAxes, path_effects=shadow, **bebas(28))
        # Offset t1 a la izquierda de t2 (aprox char_width * len(t2))
        # Bebas Neue ~0.018 fracciones por carácter a tamaño 28pt en 12" fig
        char_w = 0.0155
        offset  = len(t2) * char_w
        fax.text(0.985 - offset, 0.50, t1, color=c1, ha='right', va='center',
                 transform=fax.transAxes, path_effects=shadow, **bebas(28))

    bg_c_tuple = tuple(c/255 for c in pal['bg_from'])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=bg_c_tuple)
    plt.close(fig)
    print(f'  ✓ {output_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('Cargando datos Clausura 2026...')
    tabla_base, pendientes = load_current_data()
    print(f'  {len(tabla_base)} equipos · {len(pendientes)} partidos pendientes')

    print('Construyendo modelo Poisson...')
    matches = load_model_matches()
    model   = build_model(matches)
    print(f'  {len(matches)} partidos de entrenamiento')

    print(f'Simulando {N_SIM:,} escenarios (seed=42)...')
    rng = np.random.default_rng(42)
    posiciones = {t: defaultdict(int) for t in tabla_base}
    for _ in range(N_SIM):
        pos_map = simulate_one(model, tabla_base, pendientes, rng)
        for team, pos in pos_map.items():
            if team in posiciones:
                posiciones[team][pos] += 1

    prob_matrix = {t: {p: c/N_SIM for p,c in posiciones[t].items()}
                   for t in posiciones}

    def p_top4(t): return sum(prob_matrix[t].get(p,0) for p in range(1,5))
    teams_sorted = sorted(prob_matrix, key=p_top4, reverse=True)

    print('\nTop 5 P(Liguilla directa):')
    for t in teams_sorted[:5]:
        print(f'  {t:20s}: {p_top4(t)*100:.1f}%')

    print(f'\nGenerando 6 versiones en {OUT_DIR}/')
    for pal in PALETAS:
        print(f'  [{pal["label"]}]')
        out = OUT_DIR / pal['filename']
        render(teams_sorted, prob_matrix, tabla_base, pal, out)

    print(f'\n✓ Listo — {len(PALETAS)} imágenes en output/charts/paletas/')


if __name__ == '__main__':
    main()
