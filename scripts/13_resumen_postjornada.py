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

def get_shield(team_name: str, size: int = 60) -> np.ndarray | None:
    tid = TEAM_IDS.get(team_name)
    if tid is None:
        return None
    cache = IMG_TEAMS / f'{tid}.png'
    if not cache.exists():
        try:
            urllib.request.urlretrieve(FOTMOB_TEAM.format(tid), cache)
        except Exception:
            return None
    try:
        img = Image.open(cache).convert('RGBA').resize((size, size), Image.LANCZOS)
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

    FIG_W, FIG_H = 9.0, 4.0 + N * 2.6
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Fondo gradiente
    grad = np.zeros((200, 2, 3))
    for i in range(200):
        t = i / 199
        grad[i] = np.array([0x0a, 0x0e, 0x12]) / 255 * (1 - t) + np.array([0x13, 0x1a, 0x24]) / 255 * t
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(grad, aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    bg.axis('off')

    HEADER_H = 0.160
    FOOTER_H = 0.080
    ROW_H    = (1.0 - HEADER_H - FOOTER_H) / N

    # ── HEADER ────────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(PALETTE['bg_secondary'])
    hax.axis('off')
    # Línea roja inferior
    hax.axhline(0, color=RED, lw=2.5)

    hax.text(0.50, 0.80, 'RESUMEN JORNADA 12 · ¿QUÉ TAN BIEN PREDIJIMOS?',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(22))
    hax.text(0.50, 0.40, 'Predicción Poisson vs Resultado Real  ·  Clausura 2026',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=9)

    # ── FILAS por partido ─────────────────────────────────────────────────────
    SHIELD_S = 64

    for idx, info in enumerate(partidos_info):
        row_y = FOOTER_H + (N - 1 - idx) * ROW_H
        p_l, p_e, p_v = info['p_local'], info['p_empate'], info['p_visit']
        res_real       = info['res_real']
        res_pred       = prediccion_str(p_l, p_e, p_v)
        acierto        = res_real == res_pred

        local     = info['local']
        visitante = info['visitante']
        gl_real   = info['gl_real']
        gv_real   = info['gv_real']

        c_local = color_team(local)
        c_visit = color_team(visitante)

        # Fondo de fila
        rax = fig.add_axes([0, row_y, 1, ROW_H])
        rax.set_facecolor(PALETTE['bg_card'] if idx % 2 == 0 else BG)
        rax.set_xlim(0, 1); rax.set_ylim(0, 1)
        rax.axis('off')
        # Línea separadora
        rax.axhline(1, color=PALETTE['divider'], lw=0.8)

        # ── Escudos (izquierda) ──────────────────────────────────────────────
        badge_h = ROW_H * 0.62
        badge_w = badge_h * (FIG_H / FIG_W)
        badge_y = row_y + ROW_H * 0.19

        shield_l = get_shield(local, SHIELD_S)
        if shield_l is not None:
            sax = fig.add_axes([0.025, badge_y, badge_w, badge_h])
            sax.set_facecolor('#f8f8fc'); sax.imshow(np.array(shield_l)); sax.axis('off')
        shield_v = get_shield(visitante, SHIELD_S)
        if shield_v is not None:
            sax = fig.add_axes([0.105, badge_y, badge_w, badge_h])
            sax.set_facecolor('#f8f8fc'); sax.imshow(np.array(shield_v)); sax.axis('off')

        # ── Nombres de equipo ────────────────────────────────────────────────
        rax.text(0.025, 0.82, local.upper(), color=WHITE,
                 ha='left', va='top', **bebas(9))
        rax.text(0.105, 0.82, 'VS', color=GRAY,
                 ha='left', va='top', fontsize=7.5, fontweight='bold')
        rax.text(0.170, 0.82, visitante.upper(), color=WHITE,
                 ha='left', va='top', **bebas(9))

        # ── Barras de predicción (centro) ────────────────────────────────────
        bar_x0  = 0.295
        bar_len = 0.360
        bar_h   = 0.135
        bar_gap = 0.032
        labels  = [('LOCAL', p_l, c_local), ('EMPATE', p_e, '#888888'), ('VISIT', p_v, c_visit)]
        for bi, (lbl, val, col) in enumerate(labels):
            by = 0.68 - bi * (bar_h + bar_gap)
            # track
            rax.add_patch(Rectangle((bar_x0, by - bar_h/2), bar_len, bar_h,
                                    facecolor=PALETTE['bar_track'], lw=0, zorder=2))
            # barra llena
            rax.add_patch(Rectangle((bar_x0, by - bar_h/2), bar_len * val, bar_h,
                                    facecolor=col, lw=0, zorder=3, alpha=0.85))
            # label izq
            rax.text(bar_x0 - 0.008, by, lbl, color=GRAY,
                     ha='right', va='center', fontsize=6.5, fontweight='bold')
            # porcentaje der
            rax.text(bar_x0 + bar_len + 0.010, by, f'{val*100:.1f}%',
                     color=WHITE, ha='left', va='center', fontsize=8,
                     fontweight='bold' if val == max(p_l, p_e, p_v) else 'normal')

        # Predicción destacada
        pred_labels = {'local': f'VICTORIA {local.upper()}',
                       'empate': 'EMPATE',
                       'visitante': f'VICTORIA {visitante.upper()}'}
        rax.text(0.475, 0.14, f'PREDICCIÓN: {pred_labels[res_pred]}',
                 color=GRAY, ha='center', va='center', fontsize=7)

        # ── Resultado real (derecha-centro) ──────────────────────────────────
        cx = 0.720
        rax.text(cx, 0.76, 'RESULTADO REAL', color=GRAY,
                 ha='center', va='center', fontsize=7)
        rax.text(cx, 0.50, f'{gl_real}  –  {gv_real}',
                 color=WHITE, ha='center', va='center', **bebas(28))
        res_labels = {'local':     f'VICTORIA {local.upper()}',
                      'empate':    'EMPATE',
                      'visitante': f'VICTORIA {visitante.upper()}'}
        rax.text(cx, 0.20, res_labels[res_real],
                 color=c_local if res_real == 'local' else (c_visit if res_real == 'visitante' else GRAY),
                 ha='center', va='center', fontsize=7.5, fontweight='bold')

        # ── Check / X ─────────────────────────────────────────────────────────
        mark_x = 0.880
        mark_color = GREEN if acierto else NEG
        mark_text  = '✓' if acierto else '✗'
        rax.text(mark_x, 0.50, mark_text, color=mark_color,
                 ha='center', va='center', fontsize=34, fontweight='bold')
        rax.text(mark_x, 0.14, 'ACIERTO' if acierto else 'FALLO',
                 color=mark_color, ha='center', va='center', fontsize=7, fontweight='bold')

    # ── FOOTER ───────────────────────────────────────────────────────────────
    n_aciertos = sum(1 for p in partidos_info if p['res_real'] == prediccion_str(p['p_local'], p['p_empate'], p['p_visit']))
    pct = n_aciertos / N * 100

    fax = fig.add_axes([0, 0, 1, FOOTER_H])
    fax.set_facecolor(PALETTE['bg_secondary'])
    fax.axis('off')
    fax.axhline(1, color=RED, lw=2.0)

    # Resumen de aciertos
    pct_color = GREEN if pct >= 67 else (PALETTE['text_primary'] if pct >= 34 else NEG)
    fax.text(0.50, 0.60,
             f'{n_aciertos}/{N} PREDICCIONES CORRECTAS  ·  {pct:.0f}% DE ACIERTO',
             color=pct_color, ha='center', va='center',
             transform=fax.transAxes, **bebas(18))

    # Branding
    fax.text(0.015, 0.25, 'Fuente: FotMob · Clausura 2026',
             color=GRAY, fontsize=9, ha='left', va='center', transform=fax.transAxes)
    fax.text(0.985, 0.25, 'MAU-STATISTICS',
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
        })

    out = OUT_DIR / 'resumen_postjornada12.png'
    render(partidos_info, out)


if __name__ == '__main__':
    main()
