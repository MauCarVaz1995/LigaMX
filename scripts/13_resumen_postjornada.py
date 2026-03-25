#!/usr/bin/env python3
"""
13_resumen_postjornada.py
Resumen visual post-jornada: predicción Poisson vs resultado real.
Jornada 12 — Clausura 2026.
Salida: output/charts/resumen_postjornada12.png  (150 DPI)
"""

import json, glob, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from scipy.stats import poisson
from PIL import Image, ImageDraw
import urllib.request

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETTE, bebas, hex_rgba, hex_rgb, make_h_gradient

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

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
BG      = PALETTE['bg_main']
WHITE   = PALETTE['text_primary']
GRAY    = PALETTE['text_secondary']
RED     = PALETTE['accent']
GREEN   = PALETTE['positive']
NEG     = PALETTE['negative']

TEAM_COLORS = {
    'Chivas':        '#CD1F2D', 'Guadalajara': '#CD1F2D',
    'Cruz Azul':     '#0047AB',
    'Toluca':        '#D5001C',
    'América':       '#FFD700', 'CF America':  '#FFD700',
    'Tigres':        '#F5A623', 'Tigres UANL': '#F5A623',
    'Monterrey':     '#003DA5', 'CF Monterrey':'#003DA5',
    'Pumas':         '#C8A84B', 'Pumas UNAM':  '#C8A84B',
    'Santos Laguna': '#2E8B57', 'Santos':      '#2E8B57',
    'Pachuca':       '#A8B8C8',
    'Atlas':         '#B22222',
    'León':          '#2D8C3C', 'Leon':        '#2D8C3C',
    'Necaxa':        '#D62828',
    'Tijuana':       '#C62828',
    'Querétaro':     '#1A7FCB', 'Queretaro FC':'#1A7FCB',
    'FC Juárez':     '#4CAF50', 'FC Juarez':   '#4CAF50',
    'Mazatlán':      '#9B59B6', 'Mazatlan FC': '#9B59B6',
    'San Luis':      '#D52B1E', 'Atletico de San Luis': '#D52B1E',
    'Puebla':        '#2563EB',
}

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

# ─────────────────────────────────────────────────────────────────────────────
# JORNADA 12 — PARTIDOS ANALIZADOS Y RESULTADOS REALES
# ─────────────────────────────────────────────────────────────────────────────
PARTIDOS_J12 = [
    {'local': 'Pachuca',       'visitante': 'Toluca',       'gl_real': 1, 'gv_real': 1},
    {'local': 'Santos Laguna', 'visitante': 'Puebla',       'gl_real': 2, 'gv_real': 1},
    {'local': 'FC Juarez',     'visitante': 'Tigres',       'gl_real': 2, 'gv_real': 1},
]

# ─────────────────────────────────────────────────────────────────────────────
# MODELO DE POISSON — mismo que script 11
# ─────────────────────────────────────────────────────────────────────────────
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
    'mazatlán': 'Mazatlán',   'mazatlan fc': 'Mazatlán',
    'atletico de san luis': 'San Luis', 'san luis': 'San Luis',
    'puebla': 'Puebla',
}

def norm(n): return _ALIAS.get(str(n).lower().strip(), n.strip())

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
            tkey = stem
        w = TORNEO_WEIGHTS.get(tkey)
        if not w:
            continue
        data = json.load(open(fpath, encoding='utf-8'))
        for p in data.get('partidos', []):
            if not p.get('terminado'):
                continue
            if p.get('goles_local') is None or p.get('goles_visit') is None:
                continue
            rows.append({
                'torneo': tkey, 'peso': w,
                'local':     norm(p.get('local', '')),
                'visitante': norm(p.get('visitante', '')),
                'gl': int(p['goles_local']),
                'gv': int(p['goles_visit']),
            })
    return rows

def build_model(matches):
    sum_gl = sum_gv = sum_w = 0.0
    home_scored   = defaultdict(float)
    home_conceded = defaultdict(float)
    away_scored   = defaultdict(float)
    away_conceded = defaultdict(float)
    home_games    = defaultdict(float)
    away_games    = defaultdict(float)
    for m in matches:
        w = m['peso']
        lo, vi = m['local'], m['visitante']
        gl, gv = m['gl'], m['gv']
        sum_gl += gl * w; sum_gv += gv * w; sum_w += w
        home_scored[lo]   += gl * w; home_conceded[lo] += gv * w
        away_scored[vi]   += gv * w; away_conceded[vi] += gl * w
        home_games[lo]    += w;      away_games[vi]    += w
    mu_h = sum_gl / sum_w; mu_a = sum_gv / sum_w
    teams = set(home_scored) | set(away_scored)
    attack = {}; defense = {}
    for t in teams:
        hg = home_games.get(t, 0); ag = away_games.get(t, 0)
        attack[t]  = {'home': (home_scored[t] / hg / mu_h) if hg > 0 else 1.0,
                      'away': (away_scored[t] / ag / mu_a) if ag > 0 else 1.0}
        defense[t] = {'home': (home_conceded[t] / hg / mu_a) if hg > 0 else 1.0,
                      'away': (away_conceded[t] / ag / mu_h) if ag > 0 else 1.0}
    return {'mu_home': mu_h, 'mu_away': mu_a, 'attack': attack, 'defense': defense}

def predict(model, local, visitante, max_goals=5):
    ln = norm(local); vn = norm(visitante)
    mu_h = model['mu_home']; mu_a = model['mu_away']
    att  = model['attack'];  defe = model['defense']
    att_h  = att.get(ln,  {}).get('home', 1.0)
    def_h  = defe.get(ln, {}).get('home', 1.0)
    att_a  = att.get(vn,  {}).get('away', 1.0)
    def_a  = defe.get(vn, {}).get('away', 1.0)
    lam_l  = att_h * def_a * mu_h
    lam_v  = att_a * def_h * mu_a
    n = max_goals + 1
    p_l = [poisson.pmf(g, lam_l) for g in range(n)]
    p_v = [poisson.pmf(g, lam_v) for g in range(n)]
    mat = np.array([[p_l[gl] * p_v[gv] for gv in range(n)] for gl in range(n)])
    total = mat.sum()
    if total > 0: mat /= total
    p_local = float(np.tril(mat, -1).sum())
    p_emp   = float(np.trace(mat))
    p_visit = float(np.triu(mat, 1).sum())
    # marcador más probable
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    return p_local, p_emp, p_visit, int(idx[0]), int(idx[1])

# ─────────────────────────────────────────────────────────────────────────────
# ESCUDO
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_TEAM = 'https://images.fotmob.com/image_resources/logo/teamlogo/{}.png'

def get_shield(team_name: str, size: int = 128) -> np.ndarray | None:
    name_key = {
        'FC Juarez': 'FC Juárez', 'FC Juárez': 'FC Juárez',
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
def resultado_real_str(gl, gv):
    if gl > gv:  return 'local'
    if gl == gv: return 'empate'
    return 'visitante'

def prediccion_str(p_l, p_e, p_v):
    mx = max(p_l, p_e, p_v)
    if mx == p_l:  return 'local'
    if mx == p_e:  return 'empate'
    return 'visitante'

def color_team(name):
    return TEAM_COLORS.get(name, '#888888')

def render(partidos_info, output_path):
    N = len(partidos_info)
    aciertos = sum(1 for p in partidos_info if p['res_real'] == p['res_pred'])

    FIG_W, FIG_H = 10.0, 8.0

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Background gradient
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = np.array([0x0a,0x0e,0x12])/255*(1-t) + np.array([0x13,0x1a,0x24])/255*t
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bg.axis('off')

    HEADER_H = 1.20 / FIG_H
    FOOTER_H = 0.90 / FIG_H
    ROW_H    = (1.0 - HEADER_H - FOOTER_H) / N

    # HEADER
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(PALETTE['bg_secondary'])
    hax.axis('off')
    hax.axhline(0, color=RED, lw=2.5)

    n_fails = N - aciertos
    hook = ('JORNADA 12 · ¿QUÉ TAN BIEN PREDIJIMOS?'
            if aciertos <= n_fails else f'¡{aciertos}/{N} ACIERTOS EN JORNADA 12!')
    hax.text(0.50, 0.84, hook, color=WHITE, ha='center', va='top',
             transform=hax.transAxes, **bebas(22))
    hax.text(0.50, 0.42,
             'Modelo Poisson · Predicción (probabilidad más alta) vs Resultado Real · Clausura 2026',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=9.0)

    # AR for square badges
    AR = FIG_H / FIG_W
    SHIELD_S = 128

    for idx, info in enumerate(partidos_info):
        row_y = FOOTER_H + (N - 1 - idx) * ROW_H
        lo = info['local']
        vi = info['visitante']
        p_l = info['p_local']
        p_e = info['p_empate']
        p_v = info['p_visit']
        res_pred = info['res_pred']
        res_real = info['res_real']
        acierto  = res_pred == res_real
        gl_real  = info['gl_real']
        gv_real  = info['gv_real']

        c_lo = color_team(lo)
        c_vi = color_team(vi)

        # Row background
        rax = fig.add_axes([0, row_y, 1, ROW_H])
        rax.set_facecolor(PALETTE['bg_card'] if idx % 2 == 0 else PALETTE['bg_secondary'])
        rax.set_xlim(0, 1); rax.set_ylim(0, 1)
        rax.axis('off')
        # Visible separator between rows
        rax.axhline(1.0, color='#2d333b', lw=1.5, zorder=10)

        # ── Local crest (left side)
        badge_h = ROW_H * 0.52
        badge_w = badge_h * AR
        badge_y = row_y + (ROW_H - badge_h) / 2
        sh_l = get_shield(lo, SHIELD_S)
        if sh_l is not None:
            sax = fig.add_axes([0.012, badge_y, badge_w, badge_h])
            sax.set_facecolor('#f8f8fc')
            sax.imshow(np.array(sh_l)); sax.axis('off')

        # Local name
        name_end_x = 0.012 + badge_w
        rax.text(name_end_x + 0.010, 0.56, lo.upper(),
                 color=WHITE, ha='left', va='center', **bebas(10))

        # ── Mini bars L / E / V  (center-left)
        BAR_X0  = 0.230
        BAR_LEN = 0.185
        BAR_H   = 0.110
        BAR_YS  = [0.76, 0.50, 0.24]
        max_p   = max(p_l, p_e, p_v)
        bar_colors = [
            c_lo if p_l == max_p else PALETTE['bar_loser'],
            (GRAY  if p_e == max_p else PALETTE['bar_loser']),
            c_vi if p_v == max_p else PALETTE['bar_loser'],
        ]
        bar_items = [('L', p_l, bar_colors[0]),
                     ('E', p_e, bar_colors[1]),
                     ('V', p_v, bar_colors[2])]

        for (lbl, val, col), by in zip(bar_items, BAR_YS):
            is_max = val == max_p
            rax.add_patch(Rectangle((BAR_X0, by - BAR_H/2), BAR_LEN, BAR_H,
                                    facecolor=PALETTE['bar_track'], lw=0, zorder=2))
            rax.add_patch(Rectangle((BAR_X0, by - BAR_H/2), BAR_LEN * val, BAR_H,
                                    facecolor=col, lw=0, zorder=3,
                                    alpha=0.92 if is_max else 0.55))
            rax.text(BAR_X0 - 0.006, by, lbl, color=GRAY,
                     ha='right', va='center', fontsize=6.5, fontweight='bold')
            txt_color = WHITE if is_max else GRAY
            rax.text(BAR_X0 + BAR_LEN + 0.008, by, f'{val*100:.0f}%',
                     color=txt_color, ha='left', va='center', fontsize=8.0,
                     fontweight='bold' if is_max else 'normal')

        # ── Result (center)
        cx = 0.545
        rax.text(cx, 0.78, 'RESULTADO REAL', color=GRAY,
                 ha='center', va='center', fontsize=6.5)
        rax.text(cx, 0.50, f'{gl_real}  –  {gv_real}',
                 color=WHITE, ha='center', va='center', **bebas(36))
        res_labels = {'local': f'VIC. {lo.upper()}', 'empate': 'EMPATE',
                      'visitante': f'VIC. {vi.upper()}'}
        res_col = (c_lo if res_real == 'local'
                   else (c_vi if res_real == 'visitante' else '#aaaaaa'))
        rax.text(cx, 0.16, res_labels[res_real],
                 color=res_col, ha='center', va='center', fontsize=6.5, fontweight='bold')

        # ── Check / X  (with space around it)
        mark_color = PALETTE['positive'] if acierto else PALETTE['negative']
        rax.text(0.660, 0.52, '✓' if acierto else '✗',
                 color=mark_color, ha='center', va='center', fontsize=24, fontweight='bold')

        # Note: prediction and 2nd most probable if fallo
        if not acierto:
            sorted_probs = sorted(
                [('local', p_l), ('empate', p_e), ('visitante', p_v)],
                key=lambda x: -x[1])
            pred_full = {
                'local':     f'Predicción: Victoria {lo}',
                'empate':    'Predicción: Empate',
                'visitante': f'Predicción: Victoria {vi}',
            }
            rax.text(0.660, 0.26, pred_full[res_pred],
                     color=GRAY, ha='center', va='center', fontsize=6.0)
            if sorted_probs[1][0] == res_real:
                rax.text(0.660, 0.10,
                         f'2° más probable: {res_real} ({sorted_probs[1][1]*100:.0f}%)',
                         color='#6e8aaa', ha='center', va='center', fontsize=5.5)

        # ── Visit crest (right side)
        visit_badge_x = 0.745
        sh_v = get_shield(vi, SHIELD_S)
        if sh_v is not None:
            sax = fig.add_axes([visit_badge_x, badge_y, badge_w, badge_h])
            sax.set_facecolor('#f8f8fc')
            sax.imshow(np.array(sh_v)); sax.axis('off')

        # Visit name
        vname_x = visit_badge_x + badge_w + 0.010
        rax.text(vname_x, 0.56, vi.upper(),
                 color=WHITE, ha='left', va='center', **bebas(10))

    # FOOTER
    fax = fig.add_axes([0, 0, 1, FOOTER_H])
    fax.set_facecolor(PALETTE['bg_secondary'])
    fax.axis('off')
    fax.axhline(1, color=RED, lw=2.0)

    pct = aciertos / N * 100
    pct_color = (PALETTE['positive'] if pct >= 67
                 else (WHITE if pct >= 34 else PALETTE['negative']))
    fax.text(0.50, 0.68,
             f'{aciertos}/{N} PREDICCIONES CORRECTAS  ·  {pct:.0f}% DE ACIERTO',
             color=pct_color, ha='center', va='center',
             transform=fax.transAxes, **bebas(18))
    fax.text(0.015, 0.24, 'Fuente: FotMob · Clausura 2026',
             color=GRAY, fontsize=9, ha='left', va='center', transform=fax.transAxes)
    fax.text(0.985, 0.24, 'MAU-STATISTICS',
             color=RED, ha='right', va='center', transform=fax.transAxes, **bebas(20))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {output_path}')


def main():
    print('Cargando modelo Poisson...')
    matches = load_model_matches()
    print(f'  {len(matches)} partidos de entrenamiento')
    model = build_model(matches)

    partidos_info = []
    for p in PARTIDOS_J12:
        lo, vi = p['local'], p['visitante']
        p_l, p_e, p_v, gl_pred, gv_pred = predict(model, lo, vi)
        res_real = resultado_real_str(p['gl_real'], p['gv_real'])
        print(f'  {lo} vs {vi}: P(L)={p_l:.1%} P(E)={p_e:.1%} P(V)={p_v:.1%} | Real: {p["gl_real"]}-{p["gv_real"]}')
        partidos_info.append({
            'local': lo, 'visitante': vi,
            'p_local': p_l, 'p_empate': p_e, 'p_visit': p_v,
            'gl_pred': gl_pred, 'gv_pred': gv_pred,
            'gl_real': p['gl_real'], 'gv_real': p['gv_real'],
            'res_real': res_real,
            'res_pred': prediccion_str(p_l, p_e, p_v),  # ADD THIS
        })

    out = OUT_DIR / 'resumen_postjornada12.png'
    render(partidos_info, out)


if __name__ == '__main__':
    main()
