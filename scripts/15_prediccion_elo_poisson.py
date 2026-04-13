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
from config_visual import PALETTE, PALETAS, PALETA_ACTIVA, get_paleta, bebas, hex_rgba, hex_rgb, make_h_gradient

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
HIST_DIR   = BASE / 'data/raw/historico'
ELO_CSV    = BASE / 'data/processed/elo_historico.csv'
IMG_TEAMS  = BASE / 'data/raw/images/teams'
OUT_DIR    = BASE / 'output/charts'
# Fuente centralizada desde config_visual (incluye assets/fonts/ como candidato)
try:
    from config_visual import BEBAS_TTF, BEBAS_AVAILABLE
except ImportError:
    BEBAS_TTF = None
    BEBAS_AVAILABLE = False

IMG_TEAMS.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
_pal  = get_paleta()
BG    = _pal['bg_primary']
WHITE = _pal['text_primary']
GRAY  = _pal['text_secondary']
RED   = _pal['accent']
GREEN = _pal['cell_high']

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

DC_RHO = -0.13  # parámetro Dixon-Coles (estándar académico para fútbol)

def dixon_coles_correction(home_goals, away_goals, lambda_home, lambda_away, rho):
    """
    Factor de corrección para marcadores bajos (Dixon & Coles 1997).
    Corrige la subestimación de 0-0, 1-0, 0-1, 1-1 en Poisson independiente.
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_home * lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0

def _compute_probs(lam_l, lam_v, max_goals=5, rho=DC_RHO):
    n = max_goals + 1
    mat = np.zeros((n, n))
    for gl in range(n):
        for gv in range(n):
            dc = dixon_coles_correction(gl, gv, lam_l, lam_v, rho)
            mat[gl, gv] = poisson.pmf(gl, lam_l) * poisson.pmf(gv, lam_v) * dc
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

def get_shield(team_name: str, size: int = 64) -> np.ndarray | None:
    name_key = {
        'FC Juárez': 'FC Juarez', 'Mazatlán': 'Mazatlan FC',
        'San Luis': 'Atletico de San Luis', 'Querétaro': 'Queretaro FC',
        'América': 'CF America',
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
def render(partidos_data, output_path):
    N = len(partidos_data)
    FIG_W = 10.0
    FIG_H = 12.0   # fixed height

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Background gradient
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = np.array([0x0a,0x0e,0x12])/255*(1-t) + np.array([0x13,0x1a,0x24])/255*t
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bgax.axis('off')

    HEADER_H = 1.90 / FIG_H
    FOOTER_H = 0.90 / FIG_H
    ROW_H    = (1.0 - HEADER_H - FOOTER_H) / N
    AR       = FIG_H / FIG_W

    # ── HEADER ────────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(_pal['bg_secondary'])
    hax.axis('off')
    hax.axhline(0, color=RED, lw=2.5)
    hax.text(0.50, 0.90, 'JORNADA 13 · ELO+POISSON VS POISSON PURO',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(22))
    hax.text(0.50, 0.58, '¿A quién beneficia más el historial de 15 años?  ·  Clausura 2026',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=9.5)
    hax.text(0.50, 0.34,
             'ELO+Poisson: pondera fuerza histórica (15 años, ~5,250 partidos) · '
             'Poisson puro: forma reciente (4 últimos torneos)',
             color=GRAY, ha='center', va='top',
             transform=hax.transAxes, fontsize=7.5)

    # ── COLUMN HEADERS ───────────────────────────────────────────────────────
    # Match area: 0.000 – 0.278  (local badge | vs | visit badge | ELO info)
    # ELO+Poisson: 0.278 – 0.528
    # Poisson puro: 0.528 – 0.762
    # Δ: 0.762 – 0.840
    MATCH_X   = 0.000
    EP_X      = 0.278
    PP_X      = 0.528
    DELTA_X   = 0.762
    DELTA_W   = 0.078
    COL_W3    = (PP_X - EP_X) / 3

    col_hdr_y = 1.0 - HEADER_H - ROW_H * 0.22
    chax = fig.add_axes([0, col_hdr_y, 1, ROW_H * 0.22])
    chax.set_facecolor(_pal['bg_primary']); chax.axis('off')

    # Group labels
    chax.text(EP_X + (PP_X - EP_X)/2, 0.72, 'ELO + POISSON',
              color='#A8B8C8', ha='center', va='center', fontsize=8.0,
              fontweight='bold', transform=chax.transAxes)
    chax.text(PP_X + (DELTA_X - PP_X)/2, 0.72, 'POISSON PURO',
              color=GRAY, ha='center', va='center', fontsize=8.0,
              fontweight='bold', transform=chax.transAxes)

    # Sub-column labels
    EP_COLS = [(EP_X + i*COL_W3 + COL_W3/2, lbl)
               for i, lbl in enumerate(['LOC %', 'EMP %', 'VIS %'])]
    PP_COLS = [(PP_X + i*COL_W3 + COL_W3/2, lbl)
               for i, lbl in enumerate(['LOC %', 'EMP %', 'VIS %'])]

    for x, lbl in EP_COLS + PP_COLS:
        chax.text(x, 0.22, lbl, color=GRAY,
                  ha='center', va='center', fontsize=6.5, transform=chax.transAxes)
    chax.text(DELTA_X + DELTA_W/2, 0.45, 'Δ LOC',
              color=GRAY, ha='center', va='center',
              fontsize=6.5, transform=chax.transAxes)

    # ── MATCH ROWS ───────────────────────────────────────────────────────────
    SHIELD_S = 64

    for i, pd_data in enumerate(partidos_data):
        row_y = FOOTER_H + (N - 1 - i) * ROW_H
        lo = pd_data['local']; vi = pd_data['visitante']
        ep_l = pd_data['ep_local']; ep_e = pd_data['ep_empate']; ep_v = pd_data['ep_visit']
        pp_l = pd_data['pp_local']; pp_e = pd_data['pp_empate']; pp_v = pd_data['pp_visit']
        elo_l = pd_data['elo_local']; elo_v = pd_data['elo_visit']

        c_lo = TEAM_COLORS.get(lo, '#888888')
        c_vi = TEAM_COLORS.get(vi, '#888888')

        rax = fig.add_axes([0, row_y, 1, ROW_H])
        rax.set_facecolor(_pal['bg_secondary'] if i % 2 == 0 else _pal['bg_primary'])
        rax.set_xlim(0, 1); rax.set_ylim(0, 1)
        rax.axis('off')
        rax.axhline(1, color=BG, lw=0.6)

        # Badge layout in match column:
        # local area: 0.006 – 0.133  |  "vs" at 0.139  |  visit area: 0.148 – 0.278
        bh     = ROW_H * 0.50          # badge height (normalized fig)
        bw     = bh * AR               # square in output
        badge_y_abs = row_y + (ROW_H - bh) / 2

        LO_AREA_MID = 0.069            # center of local area
        VI_AREA_MID = 0.210            # center of visit area
        bx_lo = LO_AREA_MID - bw / 2
        bx_vi = VI_AREA_MID - bw / 2

        sh_l = get_shield(lo, SHIELD_S)
        if sh_l is not None:
            sax = fig.add_axes([bx_lo, badge_y_abs, bw, bh])
            sax.set_facecolor('#f8f8fc'); sax.imshow(sh_l); sax.axis('off')

        sh_v = get_shield(vi, SHIELD_S)
        if sh_v is not None:
            sax = fig.add_axes([bx_vi, badge_y_abs, bw, bh])
            sax.set_facecolor('#f8f8fc'); sax.imshow(sh_v); sax.axis('off')

        # "vs" between the two badges
        rax.text(0.139, 0.52, 'vs',
                 color=GRAY, ha='center', va='center', fontsize=6.5)

        # Team names below each badge
        rax.text(LO_AREA_MID, 0.16, lo.upper(),
                 color=WHITE, ha='center', va='center', fontsize=6.5, fontweight='bold')
        rax.text(VI_AREA_MID, 0.16, vi.upper(),
                 color=WHITE, ha='center', va='center', fontsize=6.5, fontweight='bold')

        # ELO values below names
        rax.text(LO_AREA_MID, 0.06, f'ELO {elo_l:.0f}',
                 color=c_lo, ha='center', va='center', fontsize=5.5)
        rax.text(VI_AREA_MID, 0.06, f'ELO {elo_v:.0f}',
                 color=c_vi, ha='center', va='center', fontsize=5.5)

        # ── ELO+Poisson values ─────────────────────────────────────────────
        ep_max = max(ep_l, ep_e, ep_v)
        for col_i, val in enumerate([ep_l, ep_e, ep_v]):
            x = EP_X + col_i * COL_W3 + COL_W3/2
            is_max = (val == ep_max)
            col_for_val = [c_lo, GRAY, c_vi][col_i]
            rax.text(x, 0.50, f'{val*100:.1f}',
                     color=col_for_val if is_max else GRAY,
                     ha='center', va='center', fontsize=10.0,
                     fontweight='bold' if is_max else 'normal')

        # ── Poisson pure values ────────────────────────────────────────────
        pp_max = max(pp_l, pp_e, pp_v)
        for col_i, val in enumerate([pp_l, pp_e, pp_v]):
            x = PP_X + col_i * COL_W3 + COL_W3/2
            is_max = (val == pp_max)
            col_for_val = [c_lo, GRAY, c_vi][col_i]
            rax.text(x, 0.50, f'{val*100:.1f}',
                     color=col_for_val if is_max else GRAY,
                     ha='center', va='center', fontsize=10.0,
                     fontweight='bold' if is_max else 'normal')

        # ── Delta ─────────────────────────────────────────────────────────
        delta = (ep_l - pp_l) * 100
        sign  = '+' if delta >= 0 else ''
        d_col = GREEN if delta > 1.5 else (_pal['accent'] if delta < -1.5 else GRAY)
        rax.text(DELTA_X + DELTA_W/2, 0.50, f'{sign}{delta:.1f}',
                 color=d_col, ha='center', va='center', fontsize=9.5, fontweight='bold')

    # ── LEGEND ────────────────────────────────────────────────────────────────
    lax = fig.add_axes([0.01, FOOTER_H * 0.62, 0.98, FOOTER_H * 0.30])
    lax.axis('off')
    lax.text(0.0, 0.5,
             'LOC = victoria local · EMP = empate · VIS = victoria visitante · '
             'Δ LOC = diferencia en P(local) entre modelos  '
             '(+verde: ELO beneficia al local · −rojo: ELO lo penaliza)',
             color=GRAY, fontsize=7.0, ha='left', va='center',
             transform=lax.transAxes)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.52])
    fax.set_facecolor(_pal['bg_secondary']); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.0)
    fax.text(0.015, 0.45, 'Modelo: ELO + Poisson-Dixon-Coles · Fuente: FotMob · Histórico 2010–2026',
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
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument('--paleta', default=None, choices=list(PALETAS.keys()))
    _args = _parser.parse_args()
    if _args.paleta:
        _p = get_paleta(_args.paleta)
        BG    = _p['bg_primary']
        WHITE = _p['text_primary']
        GRAY  = _p['text_secondary']
        RED   = _p['accent']
        GREEN = _p['cell_high']
        _pal  = _p
    main()
