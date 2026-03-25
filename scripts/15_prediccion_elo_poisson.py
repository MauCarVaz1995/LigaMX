#!/usr/bin/env python3
"""
15_prediccion_elo_poisson.py
Modelo combinado ELO + Poisson para predicción de partidos.
Compara con el Poisson puro para los partidos de la Jornada 13.
Salida: output/charts/predicciones_j13_comparativo.png  (150 DPI)
"""

import json, glob, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
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
ELO_CSV    = BASE / 'data/processed/elo_historico.csv'
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
BG    = PALETTE['bg_main']
WHITE = PALETTE['text_primary']
GRAY  = PALETTE['text_secondary']
RED   = PALETTE['accent']
GREEN = PALETTE['positive']

TEAM_COLORS = {
    'Chivas':        '#CD1F2D',
    'Cruz Azul':     '#0047AB',
    'Toluca':        '#D5001C',
    'América':       '#FFD700', 'CF America': '#FFD700',
    'Tigres':        '#F5A623',
    'Monterrey':     '#003DA5',
    'Pumas':         '#C8A84B',
    'Santos Laguna': '#2E8B57',
    'Pachuca':       '#A8B8C8',
    'Atlas':         '#B22222',
    'León':          '#2D8C3C',
    'Necaxa':        '#D62828',
    'Tijuana':       '#C62828',
    'Querétaro':     '#1A7FCB',
    'FC Juárez':     '#4CAF50', 'FC Juarez': '#4CAF50',
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
# JORNADA 13 — partidos (pendientes según el JSON)
# ─────────────────────────────────────────────────────────────────────────────
JORNADA_13 = [
    ('Puebla',     'FC Juárez'),
    ('Necaxa',     'Mazatlán'),
    ('Tijuana',    'Tigres'),
    ('Monterrey',  'San Luis'),
    ('Cruz Azul',  'Pachuca'),
    ('León',       'Atlas'),
    ('Santos Laguna', 'América'),
    ('Querétaro',  'Toluca'),
    ('Chivas',     'Pumas'),
]

# ─────────────────────────────────────────────────────────────────────────────
# MODELO POISSON PURO
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

def build_poisson_model(matches):
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

def predict_poisson(model, local, visitante, max_goals=5):
    mu_h = model['mu_home']; mu_a = model['mu_away']
    att  = model['att'];     defe = model['defe']
    lam_l = att.get(local, {}).get('home', 1.0) * defe.get(visitante, {}).get('away', 1.0) * mu_h
    lam_v = att.get(visitante, {}).get('away', 1.0) * defe.get(local, {}).get('home', 1.0) * mu_a
    return _compute_probs(lam_l, lam_v, max_goals)

def _compute_probs(lam_l, lam_v, max_goals=5):
    n = max_goals + 1
    p_l = [poisson.pmf(g, lam_l) for g in range(n)]
    p_v = [poisson.pmf(g, lam_v) for g in range(n)]
    mat = np.array([[p_l[gl] * p_v[gv] for gv in range(n)] for gl in range(n)])
    mat /= mat.sum()
    p_local = float(np.tril(mat, -1).sum())
    p_emp   = float(np.trace(mat))
    p_visit = float(np.triu(mat, 1).sum())
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    return p_local, p_emp, p_visit, lam_l, lam_v, int(idx[0]), int(idx[1])

# ─────────────────────────────────────────────────────────────────────────────
# MODELO ELO + POISSON
# ─────────────────────────────────────────────────────────────────────────────
def load_elo_ratings() -> dict[str, float]:
    """Lee el CSV y devuelve el rating ELO más reciente por equipo."""
    import csv
    elos: dict[str, float] = {}
    with open(ELO_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            elos[row['equipo']] = float(row['elo'])
    return elos

ELO_BASE      = 1500.0
HOME_ADV_ELO  = 100.0   # ventaja de localía en puntos ELO (mismo que 12_modelo_elo.py)

def predict_elo_poisson(mu_home, mu_away, elo_local, elo_visit,
                        def_factor_away, def_factor_home,
                        max_goals=5):
    """
    Combina ELO con Poisson.
    λ_local    = μ_home × (elo_local / ELO_BASE) × def_factor_away_visit
    λ_visitante = μ_away × (elo_visit / ELO_BASE) × def_factor_home_local
    Además aplica corrección por ventaja de localía vía ELO:
      elo_eff_local = elo_local + HOME_ADV_ELO / 2   (repartida mitad a cada lambda)
      elo_eff_visit = elo_visit - HOME_ADV_ELO / 2
    """
    elo_eff_l = elo_local  + HOME_ADV_ELO * 0.5
    elo_eff_v = elo_visit  - HOME_ADV_ELO * 0.5

    lam_l = mu_home * (elo_eff_l / ELO_BASE) * def_factor_away
    lam_v = mu_away * (elo_eff_v / ELO_BASE) * def_factor_home
    # Evitar lambdas negativos o cero
    lam_l = max(lam_l, 0.15)
    lam_v = max(lam_v, 0.15)
    return _compute_probs(lam_l, lam_v, max_goals)

# ─────────────────────────────────────────────────────────────────────────────
# ESCUDO
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_TEAM = 'https://images.fotmob.com/image_resources/logo/teamlogo/{}.png'

def get_shield(team_name: str, size: int = 52) -> np.ndarray | None:
    name_key = {
        'FC Juárez': 'FC Juarez',
        'Mazatlán': 'Mazatlan FC',
        'San Luis': 'Atletico de San Luis',
        'Querétaro': 'Queretaro FC',
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
        img = Image.open(cache).convert('RGBA').resize((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def render(partidos_data, output_path):
    N = len(partidos_data)
    FIG_W, FIG_H = 12.0, 2.0 + N * 1.10 + 1.40

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Fondo gradiente
    grad = np.zeros((200, 2, 3))
    for i in range(200):
        t = i / 199
        grad[i] = np.array([0x0a, 0x0e, 0x12]) / 255 * (1-t) + np.array([0x13, 0x1a, 0x24]) / 255 * t
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(grad, aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    bg.axis('off')

    HEADER_H  = 1.50 / FIG_H
    FOOTER_H  = 0.50 / FIG_H
    CONTENT_H = 1.0 - HEADER_H - FOOTER_H
    ROW_H     = CONTENT_H / N

    # ── HEADER ────────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(PALETTE['bg_secondary'])
    hax.axis('off')
    hax.axhline(0, color=RED, lw=2.5)
    hax.text(0.50, 0.88, 'PREDICCIONES JORNADA 13 · ELO + POISSON vs POISSON PURO',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(18))
    hax.text(0.50, 0.52, 'Comparación de modelos · Clausura 2026',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=9)
    hax.text(0.50, 0.28,
             'λ(ELO+P) = μ_liga × (ELO_equipo / 1500) × factor_defensa_rival · '
             'λ(Poisson) = ataque_equipo × defensa_rival × μ_liga',
             color=PALETTE['text_secondary'], ha='center', va='top',
             transform=hax.transAxes, fontsize=7)

    # ── CABECERA DE COLUMNAS ────────────────────────────────────────────────
    cols_y = 1 - HEADER_H - ROW_H * 0.25
    cols_h = ROW_H * 0.25
    col_labels = [
        (0.28, 'ELO LOCAL'),
        (0.38, 'ELO VISITA'),
        (0.50, '——— ELO+POISSON ———'),
        (0.735, '—— POISSON PURO ——'),
    ]
    cax = fig.add_axes([0, cols_y, 1, cols_h])
    cax.set_facecolor(PALETTE['bg_card'])
    cax.axis('off')
    for x, lbl in col_labels:
        cax.text(x, 0.45, lbl, color=GRAY, ha='center', va='center',
                 fontsize=6.5, fontweight='bold', transform=cax.transAxes)
    # sub-encabezados ELO+P
    for x, lbl in [(0.460, 'LOC%'), (0.510, 'EMP%'), (0.560, 'VIS%'), (0.618, 'PRED')]:
        cax.text(x, 0.45, lbl, color=PALETTE['text_secondary'],
                 ha='center', va='center', fontsize=6.0, transform=cax.transAxes)
    # sub-encabezados Poisson puro
    for x, lbl in [(0.690, 'LOC%'), (0.740, 'EMP%'), (0.790, 'VIS%'), (0.848, 'PRED')]:
        cax.text(x, 0.45, lbl, color=PALETTE['text_secondary'],
                 ha='center', va='center', fontsize=6.0, transform=cax.transAxes)
    cax.text(0.940, 0.45, 'ΔLOC%', color=PALETTE['text_secondary'],
             ha='center', va='center', fontsize=6.0, transform=cax.transAxes)

    # ── FILAS ────────────────────────────────────────────────────────────────
    SHIELD_S = 46
    AR = FIG_H / FIG_W

    for i, pd_data in enumerate(partidos_data):
        row_y = FOOTER_H + (N - 1 - i) * ROW_H
        lo    = pd_data['local']
        vi    = pd_data['visitante']
        ep_l, ep_e, ep_v = pd_data['ep_local'], pd_data['ep_empate'], pd_data['ep_visit']
        pp_l, pp_e, pp_v = pd_data['pp_local'], pd_data['pp_empate'], pd_data['pp_visit']
        elo_l = pd_data['elo_local']
        elo_v = pd_data['elo_visit']

        c_lo = TEAM_COLORS.get(lo, '#888888')
        c_vi = TEAM_COLORS.get(vi, '#888888')

        rax = fig.add_axes([0, row_y, 1, ROW_H])
        rax.set_facecolor(PALETTE['bg_secondary'] if i % 2 == 0 else PALETTE['bg_card'])
        rax.set_xlim(0, 1); rax.set_ylim(0, 1)
        rax.axis('off')
        rax.axhline(1, color=PALETTE['divider'], lw=0.6)

        # Escudos
        sh_l = get_shield(lo, SHIELD_S)
        sh_v = get_shield(vi, SHIELD_S)
        badge_h = ROW_H * 0.68
        badge_w = badge_h * AR
        badge_y = row_y + ROW_H * 0.16

        if sh_l is not None:
            sax = fig.add_axes([0.012, badge_y, badge_w, badge_h])
            sax.set_facecolor('#f8f8fc'); sax.imshow(sh_l); sax.axis('off')
        if sh_v is not None:
            sax = fig.add_axes([0.075, badge_y, badge_w, badge_h])
            sax.set_facecolor('#f8f8fc'); sax.imshow(sh_v); sax.axis('off')

        # Nombres
        rax.text(0.135, 0.62, lo.upper(), color=WHITE,
                 ha='left', va='center', **bebas(8))
        rax.text(0.135, 0.32, 'vs ' + vi.upper(), color=GRAY,
                 ha='left', va='center', fontsize=6.5)

        # ELO local / visitante
        for x, elo_val, col in [(0.280, elo_l, c_lo), (0.380, elo_v, c_vi)]:
            rax.text(x, 0.60, f'{elo_val:.0f}', color=col,
                     ha='center', va='center', fontsize=9, fontweight='bold')
            rax.text(x, 0.28, 'ELO', color=GRAY, ha='center', va='center', fontsize=6)

        # ELO+Poisson probs
        pred_ep = 'L' if ep_l >= ep_e and ep_l >= ep_v else ('E' if ep_e >= ep_v else 'V')
        pred_pp = 'L' if pp_l >= pp_e and pp_l >= pp_v else ('E' if pp_e >= pp_v else 'V')
        for x, val, col in [(0.460, ep_l, c_lo), (0.510, ep_e, '#888888'), (0.560, ep_v, c_vi)]:
            rax.text(x, 0.50, f'{val*100:.1f}', color=col if val == max(ep_l, ep_e, ep_v) else WHITE,
                     ha='center', va='center', fontsize=8.5,
                     fontweight='bold' if val == max(ep_l, ep_e, ep_v) else 'normal')
        pred_colors_ep = {'L': c_lo, 'E': '#bbbbbb', 'V': c_vi}
        rax.text(0.618, 0.50, pred_ep, color=pred_colors_ep[pred_ep],
                 ha='center', va='center', fontsize=11, fontweight='bold')

        # Poisson puro probs
        for x, val, col in [(0.690, pp_l, c_lo), (0.740, pp_e, '#888888'), (0.790, pp_v, c_vi)]:
            rax.text(x, 0.50, f'{val*100:.1f}', color=col if val == max(pp_l, pp_e, pp_v) else WHITE,
                     ha='center', va='center', fontsize=8.5,
                     fontweight='bold' if val == max(pp_l, pp_e, pp_v) else 'normal')
        pred_colors_pp = {'L': c_lo, 'E': '#bbbbbb', 'V': c_vi}
        rax.text(0.848, 0.50, pred_pp, color=pred_colors_pp[pred_pp],
                 ha='center', va='center', fontsize=11, fontweight='bold')

        # Δ%  (diferencia en predicción local entre modelos)
        delta_l = (ep_l - pp_l) * 100
        sign    = '+' if delta_l >= 0 else ''
        d_color = GREEN if delta_l > 1 else (PALETTE['negative'] if delta_l < -1 else GRAY)
        rax.text(0.940, 0.50, f'{sign}{delta_l:.1f}',
                 color=d_color, ha='center', va='center', fontsize=8, fontweight='bold')

    # ── LEYENDA INFERIOR ─────────────────────────────────────────────────────
    lax = fig.add_axes([0.01, FOOTER_H * 0.55, 0.98, FOOTER_H * 0.35])
    lax.axis('off')
    lax.text(0.0, 0.5,
             'L = Predicción: Victoria Local  ·  E = Empate  ·  V = Victoria Visitante  '
             '·  Δ% = diferencia en P(Local) entre ELO+Poisson y Poisson puro',
             color=PALETTE['text_secondary'], fontsize=6.8, ha='left', va='center',
             transform=lax.transAxes)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.50])
    fax.set_facecolor(PALETTE['bg_secondary'])
    fax.axis('off')
    fax.axhline(1, color=RED, lw=2.0)
    fax.text(0.015, 0.45, 'Fuente: FotMob · Histórico 2010–2026',
             color=GRAY, fontsize=9, ha='left', va='center', transform=fax.transAxes)
    fax.text(0.985, 0.45, 'MAU-STATISTICS',
             color=RED, ha='right', va='center', transform=fax.transAxes, **bebas(20))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {output_path}')


def main():
    print('Cargando modelo Poisson...')
    matches    = load_model_matches()
    model      = build_poisson_model(matches)
    mu_home    = model['mu_home']
    mu_away    = model['mu_away']
    print(f'  {len(matches)} partidos  |  μ_home={mu_home:.3f}  μ_away={mu_away:.3f}')

    print('Cargando ELO ratings...')
    elo_raw    = load_elo_ratings()   # dict equipo → float

    # Calcular promedio ELO de los 18 equipos actuales (base de referencia)
    current_18 = [
        'Chivas', 'Cruz Azul', 'Toluca', 'América', 'Tigres', 'Monterrey',
        'Pumas', 'Santos Laguna', 'Pachuca', 'Atlas', 'León', 'Necaxa',
        'Tijuana', 'Querétaro', 'FC Juárez', 'Mazatlán', 'San Luis', 'Puebla',
    ]
    elo_avg = np.mean([elo_raw.get(t, ELO_BASE) for t in current_18])
    print(f'  ELO promedio liga (18 equipos): {elo_avg:.1f}')

    print('\nGenerando predicciones jornada 13...')
    print(f'{"Partido":35s} | {"ELO+P LOC":>9} {"ELO+P EMP":>9} {"ELO+P VIS":>9} | '
          f'{"PP LOC":>8} {"PP EMP":>8} {"PP VIS":>8}')
    print('-' * 90)

    partidos_data = []
    for (lo, vi) in JORNADA_13:
        lo_n = norm(lo); vi_n = norm(vi)

        elo_l = elo_raw.get(lo_n, ELO_BASE)
        elo_v = elo_raw.get(vi_n, ELO_BASE)

        # Factor defensa del visitante (como en modelo Poisson)
        def_away_vi = model['defe'].get(vi_n, {}).get('away', 1.0)
        def_home_lo = model['defe'].get(lo_n, {}).get('home', 1.0)

        # ELO + Poisson
        ep_l, ep_e, ep_v, lam_el, lam_ev, _, _ = predict_elo_poisson(
            mu_home, mu_away, elo_l, elo_v, def_away_vi, def_home_lo)

        # Poisson puro
        pp_l, pp_e, pp_v, lam_pl, lam_pv, _, _ = predict_poisson(model, lo_n, vi_n)

        print(f'{lo_n[:16]:16s} vs {vi_n[:16]:16s} | '
              f'{ep_l*100:9.1f} {ep_e*100:9.1f} {ep_v*100:9.1f} | '
              f'{pp_l*100:8.1f} {pp_e*100:8.1f} {pp_v*100:8.1f}')

        partidos_data.append({
            'local': lo_n, 'visitante': vi_n,
            'elo_local': elo_l, 'elo_visit': elo_v,
            'ep_local': ep_l, 'ep_empate': ep_e, 'ep_visit': ep_v,
            'pp_local': pp_l, 'pp_empate': pp_e, 'pp_visit': pp_v,
        })

    out = OUT_DIR / 'predicciones_j13_comparativo.png'
    render(partidos_data, out)


if __name__ == '__main__':
    main()
