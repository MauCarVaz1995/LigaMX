#!/usr/bin/env python3
"""
18_prediccion_selecciones.py
Modelo ELO para selecciones + predicción México vs Portugal con Poisson.

Salidas:
  output/charts/selecciones_prediccion.png   — heatmap México vs Portugal
  output/charts/selecciones_ranking_elo.png  — Top 20 ranking ELO
  output/charts/selecciones_ultimos5.png     — últimos 5 partidos MÉX & POR

Uso:
  python 18_prediccion_selecciones.py                 # genera las 3
  python 18_prediccion_selecciones.py --chart ultimos5
  python 18_prediccion_selecciones.py --chart heatmap
  python 18_prediccion_selecciones.py --chart ranking
"""

import argparse, sys, warnings, random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnnotationBbox
from scipy.stats import poisson

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETAS, PALETA_ACTIVA, get_paleta, bebas, hex_rgba, hex_rgb, get_escudo

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
CSV_PATH  = BASE / 'data/raw/internacional/results.csv'
OUT_DIR   = BASE / 'output/charts'
# Fuente centralizada desde config_visual (incluye assets/fonts/ como candidato)
try:
    from config_visual import BEBAS_TTF, BEBAS_AVAILABLE
except ImportError:
    BEBAS_TTF = None
    BEBAS_AVAILABLE = False

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PALETA  — lee de config_visual.PALETA_ACTIVA (actualmente 'rojo_fuego')
# ─────────────────────────────────────────────────────────────────────────────
PAL   = get_paleta()
BG    = PAL['bg_primary']
BG2   = PAL['bg_secondary']
WHITE = PAL['text_primary']
GRAY  = PAL['text_secondary']
RED   = PAL['accent']
ACC2  = PAL['accent2']
CHIGH = PAL['cell_high']
CMID  = PAL['cell_mid']
CLOW  = PAL['cell_low']
BRAND = PAL['brand_color']
GOLD  = '#FFD700'

# ─────────────────────────────────────────────────────────────────────────────
# LOCALIZACIÓN / BANDERAS
# ─────────────────────────────────────────────────────────────────────────────
TEAM_ES = {
    'Mexico':'México',          'Portugal':'Portugal',
    'Brazil':'Brasil',          'France':'Francia',
    'Spain':'España',           'England':'Inglaterra',
    'Argentina':'Argentina',    'Germany':'Alemania',
    'Italy':'Italia',           'Belgium':'Bélgica',
    'Netherlands':'Países Bajos','Croatia':'Croacia',
    'Morocco':'Marruecos',      'South Korea':'Corea del Sur',
    'Japan':'Japón',            'Switzerland':'Suiza',
    'Denmark':'Dinamarca',      'Colombia':'Colombia',
    'Uruguay':'Uruguay',        'Chile':'Chile',
    'Bolivia':'Bolivia',        'Paraguay':'Paraguay',
    'Panama':'Panamá',          'Iceland':'Islandia',
    'United States':'EE.UU.',   'Republic of Ireland':'Irlanda',
    'Armenia':'Armenia',        'Hungary':'Hungría',
    'Czech Republic':'Rep. Checa','Slovakia':'Eslovaquia',
    'Poland':'Polonia',         'Serbia':'Serbia',
    'Ukraine':'Ucrania',        'Turkey':'Turquía',
    'Senegal':'Senegal',        'Canada':'Canadá',
    'Ecuador':'Ecuador',        'Wales':'Gales',
    'Scotland':'Escocia',       'Austria':'Austria',
    'Sweden':'Suecia',          'Norway':'Noruega',
    'Finland':'Finlandia',      'Russia':'Rusia',
    'Greece':'Grecia',          'Romania':'Rumanía',
    'Iran':'Irán',              'Australia':'Australia',
    'Saudi Arabia':'Arabia Saudí','Ghana':'Ghana',
    'Nigeria':'Nigeria',        'Egypt':'Egipto',
    'Costa Rica':'Costa Rica',  'Honduras':'Honduras',
    'Jamaica':'Jamaica',        'Venezuela':'Venezuela',
    'Peru':'Perú',              'China PR':'China',
    'Ivory Coast':'Costa de Marfil','Cuba':'Cuba',
    'El Salvador':'El Salvador','Guatemala':'Guatemala',
    'Luxembourg':'Luxemburgo',  'Albania':'Albania',
    'Slovenia':'Eslovenia',     'Israel':'Israel',
    'Georgia':'Georgia',        'Kosovo':'Kosovo',
}


MONTHS_ES = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',
             7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}

def fmt_date(d):
    return f"{d.day} {MONTHS_ES[d.month]} {d.year}"

def team_es(name):  return TEAM_ES.get(name, name)

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS ELO  (metodología eloratings.net)
# ─────────────────────────────────────────────────────────────────────────────
ELO_BASE = 1500
HOME_ADV = 100
SCALE    = 400

# K dinámico por tipo de torneo
def k_base(tournament: str) -> int:
    """Devuelve K base según la metodología eloratings.net."""
    t    = str(tournament).strip().lower()
    qual = 'qualifier' in t or 'qualifying' in t or 'qualification' in t

    # K=60 — Copa del Mundo (rondas finales, no clasificatorias)
    if ('world cup' in t or 'fifa world cup' in t) and not qual:
        return 60

    # K=35 — Torneos continentales principales (sin UEFA Nations League)
    main_35 = [
        'concacaf gold cup', 'copa america', 'afc asian cup',
        'africa cup of nations', 'confederations cup', 'ofc nations cup',
        'concacaf nations league',
    ]
    is_euro_main = (('euro ' in t or t.startswith('euro') or
                      'european championship' in t) and not qual)
    if is_euro_main:
        return 35
    if any(x in t for x in main_35) and not qual and 'uefa nations' not in t:
        return 35

    # K=25 — Clasificatorias y UEFA Nations League
    if qual or 'uefa nations league' in t:
        return 25

    # K=20 — Amistosos y cualquier torneo no mapeado
    return 20


# Multiplicador por diferencia de goles
def goal_mult(gf: int, ga: int) -> float:
    """Factor de ajuste por margen de victoria (eloratings.net)."""
    diff = abs(gf - ga)
    if diff <= 1:   return 1.0
    elif diff == 2: return 1.5
    else:           return 1.75 + (diff - 3) * 0.04

EXTRA_MEXICO = [
    ('2024-10-12', 'Mexico', 'Panama',   1, 0, 'Friendly', True),
    ('2024-10-15', 'Mexico', 'Paraguay', 1, 2, 'Friendly', True),
    ('2024-10-19', 'Mexico', 'Uruguay',  0, 0, 'Friendly', True),
    ('2025-01-18', 'Mexico', 'Iceland',  4, 0, 'Friendly', True),
    ('2025-01-22', 'Mexico', 'Bolivia',  1, 0, 'Friendly', True),
]

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────────────────────
def load_csv():
    return pd.read_csv(CSV_PATH, parse_dates=['date'])

def print_stats(df):
    print(f'\n{"─"*60}')
    print(f'  Fecha más reciente : {df["date"].max().date()}')
    print(f'  Total partidos     : {len(df):,}')
    mex = df[(df['home_team']=='Mexico') | (df['away_team']=='Mexico')]
    por = df[(df['home_team']=='Portugal') | (df['away_team']=='Portugal')]
    print(f'  Partidos México    : {len(mex):,}')
    print(f'  Partidos Portugal  : {len(por):,}')
    print(f'{"─"*60}')

def print_hist_mex_por(df):
    mask = (
        ((df['home_team']=='Mexico')   & (df['away_team']=='Portugal')) |
        ((df['home_team']=='Portugal') & (df['away_team']=='Mexico'))
    )
    h = df[mask].copy().sort_values('date')
    print(f'\n  Historial México vs Portugal ({len(h)} partidos):')
    if h.empty:
        print('  — No hay enfrentamientos.'); return h
    for _, r in h.iterrows():
        print(f'  {r["date"].date()}  {r["home_team"]:20s} {r["home_score"]}-{r["away_score"]}  '
              f'{r["away_team"]:20s}  [{r["tournament"]}]')
    return h

# ─────────────────────────────────────────────────────────────────────────────
# MODELO ELO
# ─────────────────────────────────────────────────────────────────────────────
def expected_score(elo_a, elo_b, home_adv=0):
    return 1 / (1 + 10 ** ((elo_b - (elo_a + home_adv)) / SCALE))

def result_score(gl, gv):
    if gl > gv: return 1.0
    elif gl < gv: return 0.0
    return 0.5

def calc_elo(df):
    """
    Calcula ELO histórico con metodología eloratings.net:
      - K dinámico por tipo de torneo (20/25/35/60)
      - Multiplicador por diferencia de goles
      - Bono de localía +100 solo en el cálculo de We (no al ELO almacenado)
      - Sin regresión anual
      - Todas las selecciones arrancan en 1500
    """
    elos = defaultdict(lambda: ELO_BASE)
    for r in df.sort_values('date').itertuples(index=False):
        home, away = r.home_team, r.away_team
        gl, gv    = int(r.home_score), int(r.away_score)
        neutral   = str(r.neutral).upper() == 'TRUE'
        adv       = 0 if neutral else HOME_ADV
        ea        = expected_score(elos[home], elos[away], home_adv=adv)
        sa        = result_score(gl, gv)
        k_eff     = k_base(r.tournament) * goal_mult(gl, gv)
        elos[home] += k_eff * (sa - ea)
        elos[away] += k_eff * ((1 - sa) - (1 - ea))
    return dict(elos)

def apply_extra_matches(elos, extra):
    """Aplica partidos recientes con el mismo sistema K dinámico."""
    for date_str, home, away, gl, gv, torneo, neutral in extra:
        adv   = 0 if neutral else HOME_ADV
        ea    = expected_score(elos.get(home, ELO_BASE), elos.get(away, ELO_BASE), home_adv=adv)
        sa    = result_score(gl, gv)
        k_eff = k_base(torneo) * goal_mult(gl, gv)
        elos[home] = elos.get(home, ELO_BASE) + k_eff * (sa - ea)
        elos[away] = elos.get(away, ELO_BASE) + k_eff * ((1 - sa) - (1 - ea))
    return elos

# ─────────────────────────────────────────────────────────────────────────────
# POISSON + DIXON-COLES
# ─────────────────────────────────────────────────────────────────────────────
DC_RHO = -0.13  # parámetro Dixon-Coles (estándar académico para fútbol)

def dixon_coles_correction(i, j, lambda_home, lambda_away, rho=-0.13):
    if i == 0 and j == 0:
        return 1 - lambda_home * lambda_away * rho
    elif i == 0 and j == 1:
        return 1 + lambda_home * rho
    elif i == 1 and j == 0:
        return 1 + lambda_away * rho
    elif i == 1 and j == 1:
        return 1 - rho
    else:
        return 1.0

def poisson_probs(lam_h, lam_a, max_goals=5):
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            dc = dixon_coles_correction(i, j, lam_h, lam_a, DC_RHO)
            probs[i, j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) * dc
    probs /= probs.sum()
    return probs, np.sum(np.tril(probs,-1)), np.sum(np.diag(probs)), np.sum(np.triu(probs,1))

def elo_to_lambda(elo_local, elo_away, elo_mean, avg_goals, loc_factor=1.0):
    return avg_goals*(elo_local/elo_mean)*loc_factor, avg_goals*(elo_away/elo_mean)

# ─────────────────────────────────────────────────────────────────────────────
# ÚLTIMOS N PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────
def last_n(df, team, n=5, extra=None):
    mask = (df['home_team']==team) | (df['away_team']==team)
    rows = df[mask].copy().sort_values('date').tail(n)
    result = []
    for _, r in rows.iterrows():
        ih = (r['home_team']==team)
        opp = r['away_team'] if ih else r['home_team']
        gf  = int(r['home_score'] if ih else r['away_score'])
        ga  = int(r['away_score'] if ih else r['home_score'])
        result.append({'date':r['date'].date(),'opponent':opp,'gf':gf,'ga':ga,
                       'ha':'L' if ih else 'V',
                       'res':'G' if gf>ga else ('E' if gf==ga else 'P'),
                       'tournament':r['tournament']})
    if extra and team=='Mexico':
        for ds, home, away, gl, gv, torneo, _ in extra:
            ih = (home==team)
            opp = away if ih else home
            gf,ga = (gl,gv) if ih else (gv,gl)
            result.append({'date':datetime.strptime(ds,'%Y-%m-%d').date(),
                           'opponent':opp,'gf':gf,'ga':ga,
                           'ha':'L' if ih else 'V',
                           'res':'G' if gf>ga else ('E' if gf==ga else 'P'),
                           'tournament':torneo})
        result = sorted(result, key=lambda x: x['date'])[-n:]
    return result

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def _bg_gradient(fig):
    bg_rgb, bg2_rgb = hex_rgb(BG), hex_rgb(BG2)
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = np.array(bg_rgb)/255*(1-t) + np.array(bg2_rgb)/255*t
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.set_zorder(-100)
    bgax.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bgax.axis('off')

def _footer(fig, source_text, h=0.055, fact=None):
    if fact:
        fct = fig.add_axes([0, h, 1, 0.030])
        fct.set_facecolor(BG); fct.axis('off')
        fct.text(0.5, 0.5, fact, color=GRAY, fontsize=8.5,
                 ha='center', va='center', style='italic',
                 transform=fct.transAxes)
    fax = fig.add_axes([0, 0, 1, h*0.55])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.5)
    fax.text(0.015, 0.50, source_text, color=GRAY, fontsize=9,
             ha='left', va='center', transform=fax.transAxes)
    shadow = [mpe.withStroke(linewidth=3, foreground='#000000')]
    fax.text(0.985, 0.50, 'MAU-STATISTICS', color=BRAND, ha='right', va='center',
             transform=fax.transAxes, path_effects=shadow, **bebas(20))

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN 1: HEATMAP PREDICCIÓN (5×5)
# ─────────────────────────────────────────────────────────────────────────────
def render_heatmap(probs, p_home, p_draw, p_away, lam_h, lam_a,
                   elo_mex, elo_por, out_path, fact=''):
    MAX_G = 5; N = MAX_G + 1
    FIG_W, FIG_H = 14.0, 16.5

    HEADER_H = 0.220
    FOOTER_H = 0.055
    FACT_H   = 0.042
    PROB_H   = 0.125
    COLHDR_H = 0.050
    ROWHDR_W = 0.082
    R_MARGIN = 0.008

    GRID_Y = FOOTER_H + FACT_H + PROB_H + 0.005
    GRID_H = 1.0 - HEADER_H - GRID_Y - COLHDR_H
    CELL_H = GRID_H / N
    CELL_W = (1.0 - ROWHDR_W - R_MARGIN) / N

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _bg_gradient(fig)

    hax = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=4.0)
    hax.text(0.50, 0.98, '¿QUIÉN GANA HOY?',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(46))
    hax.text(0.50, 0.60,
             'México vs Portugal  ·  Amistoso 2026',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=20)

    for col_x, team_key, name_es, elo, lam, border_c in [
        (0.07, 'Mexico',   'MÉXICO',   elo_mex, lam_h, CHIGH),
        (0.57, 'Portugal', 'PORTUGAL', elo_por, lam_a, ACC2),
    ]:
        pax = fig.add_axes([col_x, 1-HEADER_H+HEADER_H*0.06, 0.35, HEADER_H*0.44])
        pax.set_facecolor(BG); pax.axis('off')
        for sp in pax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(border_c); sp.set_linewidth(1.8)
        fi = get_escudo(team_key, size=(58, 38))
        if fi is not None:
            pax.add_artist(AnnotationBbox(fi, (0.40, 0.80),
                           frameon=False, xycoords='axes fraction',
                           box_alignment=(1.0, 0.5)))
            pax.text(0.42, 0.80, name_es, color=border_c, ha='left', va='center',
                     fontsize=18, fontweight='bold', transform=pax.transAxes)
        else:
            pax.text(0.50, 0.80, name_es, color=border_c, ha='center', va='center',
                     fontsize=18, fontweight='bold', transform=pax.transAxes)
        pax.text(0.50, 0.40, f'ELO: {elo:.0f}',
                 color=WHITE, ha='center', va='center',
                 fontsize=15, transform=pax.transAxes)
        pax.text(0.50, 0.07, f'λ esperado = {lam:.3f} goles',
                 color=GRAY, ha='center', va='center',
                 fontsize=11, transform=pax.transAxes)

    vax = fig.add_axes([0.40, 1-HEADER_H+HEADER_H*0.06, 0.20, HEADER_H*0.44])
    vax.set_facecolor(BG2); vax.axis('off')
    vax.text(0.50, 0.55, 'VS', color='#FF0000', ha='center', va='center',
             fontsize=29, fontweight='bold', transform=vax.transAxes)
    vax.text(0.50, 0.20, 'Poisson · ELO', color=GRAY, ha='center', va='center',
             fontsize=8.5, transform=vax.transAxes)

    chdr_y = GRID_Y + GRID_H
    lbl_ax = fig.add_axes([ROWHDR_W, chdr_y+COLHDR_H*0.55, N*CELL_W, COLHDR_H*0.42])
    lbl_ax.set_facecolor(BG2); lbl_ax.axis('off')
    lbl_ax.text(0.5, 0.5, 'GOLES  PORTUGAL',
                color=ACC2, ha='center', va='center',
                fontsize=11, fontweight='bold', transform=lbl_ax.transAxes)
    for j in range(N):
        nax = fig.add_axes([ROWHDR_W+j*CELL_W, chdr_y, CELL_W, COLHDR_H*0.55])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.5, 0.5, str(j), color=WHITE, ha='center', va='center',
                 fontsize=14, fontweight='bold', transform=nax.transAxes)

    rmx = fig.add_axes([0, GRID_Y, ROWHDR_W*0.35, GRID_H])
    rmx.set_facecolor(BG2); rmx.axis('off')
    rmx.text(0.5, 0.5, 'GOLES  MÉXICO',
             color=CHIGH, ha='center', va='center', fontsize=10, fontweight='bold',
             rotation=90, transform=rmx.transAxes)
    for i in range(N):
        nax = fig.add_axes([ROWHDR_W*0.35, GRID_Y+i*CELL_H, ROWHDR_W*0.65, CELL_H])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.65, 0.5, str(i), color=WHITE, ha='center', va='center',
                 fontsize=14, fontweight='bold', transform=nax.transAxes)

    def cell_style(p):
        if p > 0.090:   return CHIGH,   WHITE,       True,  20   # rojo brillante → blanco
        elif p > 0.045: return CMID,    WHITE,       True,  17   # rojo medio → blanco
        elif p > 0.015: return CLOW,    '#FF6B6B',   True,  14   # oscuro → rojo claro
        elif p > 0.004: return BG2,     WHITE,       False, 13   # oscuro → blanco
        else:           return BG,      GRAY,        False, 11   # casi negro → gris (valor mínimo)

    for i in range(N):
        for j in range(N):
            p  = probs[i, j]
            fc, tc, bold, fs = cell_style(p)
            cax = fig.add_axes([ROWHDR_W+j*CELL_W, GRID_Y+i*CELL_H, CELL_W, CELL_H])
            cax.set_xlim(0,1); cax.set_ylim(0,1)
            cax.add_patch(mpatches.Rectangle(
                (0,0),1,1, facecolor=fc, edgecolor='none', zorder=0,
                transform=cax.transAxes, clip_on=False))
            cax.axis('off')
            if i == j:
                for sp in cax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(GOLD); sp.set_linewidth(2.0)
            if p >= 0.003:
                txt = f'{p*100:.1f}' if p >= 0.010 else f'{p*100:.2f}'
                cax.text(0.5, 0.5, txt, color=tc, ha='center', va='center',
                         fontsize=fs, fontweight='bold' if bold else 'normal',
                         transform=cax.transAxes, zorder=5)

    # ── Área de resultados: 3 bloques iguales ───────────────────────────────
    prob_y  = FOOTER_H + FACT_H + 0.006
    BLK_GAP = 0.014
    BLK_W   = (1.0 - 4 * BLK_GAP) / 3
    max_p   = max(p_home, p_draw, p_away)

    result_blocks = [
        ('Mexico',   'GANA MÉXICO',   p_home),
        (None,       'EMPATE',         p_draw),
        ('Portugal', 'GANA PORTUGAL',  p_away),
    ]

    for k, (team_key, label, pval) in enumerate(result_blocks):
        bx     = BLK_GAP + k * (BLK_W + BLK_GAP)
        is_max = (pval >= max_p - 1e-9)
        pct_c  = RED   if is_max else GRAY
        lbl_c  = WHITE if is_max else GRAY
        pct_fs = 50    if is_max else 32

        seg = fig.add_axes([bx, prob_y, BLK_W, PROB_H])
        seg.set_facecolor('#1c0000' if is_max else BG2)
        seg.axis('off'); seg.set_xlim(0, 1); seg.set_ylim(0, 1)
        if is_max:
            for sp in seg.spines.values():
                sp.set_visible(True); sp.set_edgecolor(RED); sp.set_linewidth(2.5)

        # Bandera + etiqueta (grupo centrado en la mitad superior)
        if team_key:
            fi_blk = get_escudo(team_key, size=(36, 24))
            if fi_blk is not None:
                seg.add_artist(AnnotationBbox(fi_blk, (0.37, 0.80),
                               frameon=False, xycoords='axes fraction',
                               box_alignment=(1.0, 0.5)))
                seg.text(0.40, 0.80, label, color=lbl_c, ha='left', va='center',
                         fontsize=12 if is_max else 10, fontweight='bold',
                         transform=seg.transAxes)
            else:
                seg.text(0.50, 0.80, label, color=lbl_c, ha='center', va='center',
                         fontsize=12 if is_max else 10, fontweight='bold',
                         transform=seg.transAxes)
        else:
            seg.text(0.50, 0.80, label, color=lbl_c, ha='center', va='center',
                     fontsize=12 if is_max else 10, fontweight='bold',
                     transform=seg.transAxes)

        # Porcentaje grande centrado en mitad inferior
        seg.text(0.50, 0.36, f'{pval*100:.1f}%', color=pct_c, ha='center', va='center',
                 fontsize=pct_fs, fontweight='bold', transform=seg.transAxes)

    _footer(fig, 'Modelo: ELO + Poisson-Dixon-Coles · Fuente: martj42/international_results',
            h=FOOTER_H, fact=fact)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {out_path}')

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN 2: RANKING ELO TOP 20
# ─────────────────────────────────────────────────────────────────────────────
def render_ranking(elos, elo_mex, elo_por, out_path, fact=''):
    top20 = sorted(elos.items(), key=lambda x: x[1], reverse=True)[:20]
    n     = len(top20)
    FIG_W, FIG_H = 12.0, 16.0

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _bg_gradient(fig)

    HEADER_H  = 0.200    # espacio para 3 líneas + leyenda
    FOOTER_H  = 0.055
    FACT_H    = 0.030
    CONTENT_Y = FOOTER_H + FACT_H
    CONTENT_H = 1 - HEADER_H - FOOTER_H - FACT_H
    ROW_H     = CONTENT_H / n

    # ── Header: título + 3 líneas + leyenda de K ─────────────────────────────
    hax = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=4.0)

    hax.text(0.50, 0.97, 'RANKING ELO · TOP 20 MUNDIAL',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(36))
    hax.text(0.50, 0.70,
             'Metodología eloratings.net · K dinámico: Amistoso=20 · Clasificatorio=25 · Continental=35 · Mundial=60',
             color='#dddddd', ha='center', va='top', transform=hax.transAxes, fontsize=8.5)
    hax.text(0.50, 0.55,
             'Margen de goles · Ventaja de localía · Partidos desde 1872 hasta 2026 · ~49,000 partidos',
             color='#dddddd', ha='center', va='top', transform=hax.transAxes, fontsize=8.5)

    # Leyenda de K con cuadros de color
    K_LEGEND = [
        ('#FFD700', 'K=20  Amistoso'),
        ('#FFA040', 'K=25  Clasificatorio'),
        ('#FF7070', 'K=35  Continental'),
        ('#AA1122', 'K=60  Mundial'),
    ]
    for idx_l, ((lc, lbl), lx) in enumerate(zip(K_LEGEND, [0.04, 0.28, 0.54, 0.76])):
        hax.add_patch(mpatches.Rectangle(
            (lx, 0.22), 0.018, 0.16, facecolor=lc, edgecolor='none',
            transform=hax.transAxes, clip_on=False, zorder=5))
        hax.text(lx + 0.026, 0.30, lbl, color='#dddddd',
                 ha='left', va='center', fontsize=8.5, transform=hax.transAxes)

    # ── Filas — data coords = PUNTOS desde borde izquierdo ──────────────────
    elo_max = top20[0][1]
    elo_min = min(e for _, e in top20) - 30
    FEATURED = {'Mexico', 'Portugal'}

    # Dimensiones de cada fila en PUNTOS (1 pt = 1/72 in)
    ROW_PTS_W = FIG_W * 72          # 12in × 72 = 864 pt (ancho)
    ROW_PTS_H = ROW_H * FIG_H * 72  # alto por fila en puntos

    # Tabla invisible — 3 columnas de ancho fijo en pts:
    #   col0 = 40pt  (número,  ha='right')
    #   col1 = 50pt  (bandera, centro en 40+25=65pt)
    #   col2 = 180pt (nombre,  inicio en 40+50=90pt)
    NUM_RIGHT    = 40        # borde derecho col0
    FLAG_CX      = 65        # centro col1
    NAME_X       = 90        # inicio col2
    BAR_X        = 320       # inicio track de barra
    BAR_W        = ROW_PTS_W - BAR_X - 90   # ancho dinámico
    ELO_X        = ROW_PTS_W - 12            # ELO right-aligned
    GOLD_STRIP_W = 5

    for i, (team, elo) in enumerate(top20):
        row_y  = CONTENT_Y + (n-1-i) * ROW_H
        is_ft  = team in FEATURED
        row_bg = '#180808' if team=='Mexico' else ('#0e0814' if team=='Portugal' else
                  (BG2 if i%2==0 else BG))

        ax = fig.add_axes([0, row_y, 1, ROW_H])
        ax.set_facecolor(row_bg)
        ax.set_xlim(0, ROW_PTS_W)   # data coords = puntos
        ax.set_ylim(0, ROW_PTS_H)
        ax.axis('off')

        MID = ROW_PTS_H / 2   # centro vertical en puntos

        if is_ft:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(GOLD); sp.set_linewidth(1.5)
            ax.add_patch(mpatches.Rectangle(
                (0, ROW_PTS_H * 0.05), GOLD_STRIP_W, ROW_PTS_H * 0.90,
                facecolor=GOLD, linewidth=0))

        # ── col0: [Número] right-aligned en NUM_RIGHT ────────────────────────
        pos_color = GOLD if i < 3 else WHITE
        ax.text(NUM_RIGHT, MID, f'{i+1}',
                color=GOLD if is_ft else pos_color,
                fontsize=12, fontweight='bold', va='center', ha='right')

        # ── col1: [Bandera 40×27] centrada en FLAG_CX ────────────────────────
        flag_im = get_escudo(team, size=(40, 27))
        if flag_im is not None:
            ax.add_artist(AnnotationBbox(flag_im, (FLAG_CX, MID),
                          frameon=False, xycoords='data',
                          box_alignment=(0.5, 0.5)))

        # ── col2: [Nombre] a partir de NAME_X ────────────────────────────────
        name_color = CHIGH if team=='Mexico' else (ACC2 if team=='Portugal' else WHITE)
        ax.text(NAME_X, MID, team_es(team), color=name_color,
                fontsize=12, fontweight='bold' if is_ft else 'normal',
                va='center', ha='left')

        # ── [Barra ELO] degradado simulado con 3 capas ───────────────────────
        bar_val = max(0, (elo - elo_min) / (elo_max - elo_min + 1))
        bar_h   = ROW_PTS_H * 0.60       # 60% del alto de la fila
        bar_y   = MID - bar_h / 2
        bw      = BAR_W * bar_val

        # Track de fondo
        ax.add_patch(mpatches.FancyBboxPatch(
            (BAR_X, bar_y), BAR_W, bar_h,
            boxstyle='round,pad=1.5',
            facecolor=BG2, linewidth=0, zorder=1))

        if bw > 3:
            bar_sh = '#8B7000' if is_ft else '#9B0C23'
            bar_fc = GOLD    if is_ft else '#C8102E'
            bar_hi = '#FFE066' if is_ft else '#E8384F'

            # Capa 1 — sombra en #9B0C23 (borde redondeado)
            ax.add_patch(mpatches.FancyBboxPatch(
                (BAR_X, bar_y), bw, bar_h,
                boxstyle='round,pad=1.5',
                facecolor=bar_sh, linewidth=0, zorder=2))
            # Capa 2 — color principal #C8102E
            ax.add_patch(mpatches.Rectangle(
                (BAR_X, bar_y), bw, bar_h,
                facecolor=bar_fc, linewidth=0, zorder=3))
            # Capa 3 — highlight #E8384F en el 30% superior
            ax.add_patch(mpatches.Rectangle(
                (BAR_X, bar_y + bar_h * 0.70), bw, bar_h * 0.30,
                facecolor=bar_hi, linewidth=0, zorder=4, alpha=0.55))

        # ── [ELO] right-aligned en ELO_X ──────────────────────────────────────
        ax.text(ELO_X, MID, f'{elo:.0f}',
                color=GOLD if is_ft else WHITE,
                fontsize=12, fontweight='bold' if is_ft else 'normal',
                va='center', ha='right')
        ax.axhline(0, color=BG, lw=0.6)

    _footer(fig, 'Fuente: martj42/international_results · Partidos desde 1872',
            h=FOOTER_H, fact=fact)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {out_path}')

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN 3: ÚLTIMOS 5 PARTIDOS — TARJETAS REDISEÑADAS
# ─────────────────────────────────────────────────────────────────────────────
def render_ultimos5(last_mex, last_por, out_path, fact=''):
    """
    Dos columnas de tarjetas con paleta aleatoria.
    Cada tarjeta:
      • Fondo BG2 redondeado (FancyBboxPatch)
      • Borde vertical fino 5px izquierdo: acento lleno=ganó, semitransparente=empató, gris=perdió
      • Marcador Bebas Neue grande centrado
      • Rival: bandera 48×32 + nombre
      • Torneo: texto pequeño en color acento de la paleta
      • Fecha: gris claro (text_secondary)  |  Local/Visita: color acento (siempre)
    """
    # ── Paleta aleatoria ──────────────────────────────────────────────────────
    pal_key = random.choice(list(PALETAS.keys()))
    print(f'  🎨  Paleta seleccionada para ultimos5: {pal_key}')
    pal = PALETAS[pal_key]
    _BG   = pal['bg_primary']
    _BG2  = pal['bg_secondary']
    _ACC  = pal['accent']
    _ACC2 = pal['accent2']
    _WHITE = pal['text_primary']
    _GRAY  = pal['text_secondary']
    _BRAND = pal['brand_color']

    FIG_W, FIG_H = 14.0, 12.5
    N = 5

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=_BG)

    # Gradiente de fondo (usa colores locales)
    import numpy as np
    from matplotlib.image import AxesImage
    grad = np.zeros((256, 1, 4))
    r1, g1, b1 = hex_rgb(_BG)
    r2, g2, b2 = hex_rgb(_BG2)
    for i in range(256):
        t = i / 255
        grad[i, 0] = [r1/255*(1-t)+r2/255*t, g1/255*(1-t)+g2/255*t,
                      b1/255*(1-t)+b2/255*t, 1.0]
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg_ax.imshow(grad, aspect='auto', extent=[0,1,0,1],
                 origin='lower', transform=fig.transFigure, zorder=-10)
    bg_ax.axis('off')

    HEADER_H  = 0.172
    FOOTER_H  = 0.055
    FACT_H    = 0.030
    TITLE_H   = 0.055
    GAP_COL   = 0.006
    CONTENT_Y = FOOTER_H + FACT_H
    CONTENT_H = 1.0 - HEADER_H - FOOTER_H - FACT_H - TITLE_H - 0.008
    COL_W     = (1.0 - GAP_COL) / 2.0   # ~0.497

    # 5px accent strip width in data units (x ∈ [0,1], col_width = COL_W*FIG_W*150dpi px)
    STRIP_W = 5.0 / (COL_W * FIG_W * 150.0)   # ≈ 0.0048

    # ── Header (solo texto, sin banderas) ────────────────────────────────────
    hax = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    hax.set_facecolor(_BG2); hax.axis('off')
    hax.axhline(0, color=_ACC, lw=4.0)
    hax.text(0.50, 0.96, 'ASÍ LLEGAN AL PARTIDO',
             color=_WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(40))
    hax.text(0.50, 0.50,
             'México vs Portugal  ·  Últimos 5 partidos',
             color=_WHITE, ha='center', va='top', transform=hax.transAxes, fontsize=18)

    # ── Columnas ─────────────────────────────────────────────────────────────
    TEAMS = [
        ('Mexico',   last_mex, 0.0,             _ACC),
        ('Portugal', last_por, COL_W + GAP_COL, _ACC2),
    ]

    for team_key, matches, col_x, team_color in TEAMS:
        name_es_upper = team_es(team_key).upper()

        # Título de columna
        col_top_y = CONTENT_Y + CONTENT_H
        tax = fig.add_axes([col_x, col_top_y, COL_W, TITLE_H])
        tax.set_facecolor(_BG2); tax.axis('off')
        tax.set_xlim(0, 1); tax.set_ylim(0, 1)

        flag_title = get_escudo(team_key, size=(48, 32))
        if flag_title is not None:
            tax.add_artist(AnnotationBbox(flag_title, (0.36, 0.50),
                           frameon=False, xycoords='axes fraction',
                           box_alignment=(1.0, 0.5)))
        tax.text(0.40, 0.50, name_es_upper, color=team_color,
                 ha='left', va='center', fontsize=17, fontweight='bold',
                 transform=tax.transAxes)

        # Eje principal de la columna (coordenadas: x∈[0,1], y∈[0,N])
        ax = fig.add_axes([col_x, CONTENT_Y, COL_W, CONTENT_H])
        ax.set_facecolor(_BG); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, N)

        for idx, m in enumerate(reversed(matches)):
            win  = m['res'] == 'G'
            draw = m['res'] == 'E'

            # ── Tarjeta fondo redondeado ──────────────────────────────────────
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.016, idx + 0.060), 0.968, 0.878,
                boxstyle='round,pad=0.012',
                facecolor=_BG2, edgecolor='none', linewidth=0, zorder=1))

            # ── Borde vertical fino 5px izquierdo ────────────────────────────
            if win:
                strip_c, strip_a = _ACC, 1.0
            elif draw:
                strip_c, strip_a = _ACC, 0.45
            else:
                strip_c, strip_a = '#333333', 1.0
            ax.add_patch(mpatches.Rectangle(
                (0.016, idx + 0.095), STRIP_W, 0.808,
                facecolor=strip_c, alpha=strip_a,
                linewidth=0, zorder=2))

            # ── Marcador (Bebas Neue grande, centrado) ────────────────────────
            ax.text(0.500, idx + 0.640,
                    f'{m["gf"]}-{m["ga"]}',
                    color=_WHITE, ha='center', va='center',
                    fontsize=29, zorder=3, **bebas(29))

            # ── Bandera 48×32 + Rival (debajo del marcador, centrado) ─────────
            opp_flag_im = get_escudo(m['opponent'], size=(48, 32))
            opp_es = team_es(m['opponent'])
            if len(opp_es) > 15: opp_es = opp_es[:14] + '.'

            FLAG_W = 0.046
            if opp_flag_im is not None:
                flag_x = 0.500 - FLAG_W * 0.5 - 0.012
                ax.add_artist(AnnotationBbox(opp_flag_im, (flag_x, idx + 0.370),
                              frameon=False, xycoords='data',
                              box_alignment=(0.5, 0.5), zorder=4))
                text_x = 0.500 + FLAG_W * 0.5 + 0.006
                ax.text(text_x, idx + 0.370, opp_es,
                        color=_WHITE, ha='left', va='center',
                        fontsize=10.5, fontweight='bold', zorder=3)
            else:
                ax.text(0.500, idx + 0.370, opp_es,
                        color=_WHITE, ha='center', va='center',
                        fontsize=10.5, fontweight='bold', zorder=3)

            # ── Torneo (color acento de la paleta) ────────────────────────────
            ax.text(0.500, idx + 0.190, m['tournament'][:30],
                    color=_ACC, ha='center', va='center',
                    fontsize=8.5, zorder=3)

            # ── Fecha humana (arriba-derecha, gris) ───────────────────────────
            ax.text(0.970, idx + 0.840, fmt_date(m['date']),
                    color=_GRAY, ha='right', va='center',
                    fontsize=8.0, zorder=3)

            # ── Local/Visita (color acento siempre) ───────────────────────────
            ha_lbl = 'Local' if m['ha'] == 'L' else 'Visita'
            ax.text(0.970, idx + 0.690, ha_lbl,
                    color=_ACC, ha='right', va='center',
                    fontsize=7.0, fontweight='bold', zorder=3)

    # ── Divisor entre columnas ────────────────────────────────────────────────
    div = fig.add_axes([COL_W, CONTENT_Y, GAP_COL, CONTENT_H + TITLE_H])
    div.set_facecolor(_ACC); div.axis('off')

    _footer(fig,
            'Fuente: martj42/international_results + resultados recientes México',
            h=FOOTER_H, fact=fact)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'✓ Guardado: {out_path}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chart',
                        choices=['heatmap', 'ranking', 'ultimos5', 'all'],
                        default='all',
                        help='Qué gráfica(s) generar (default: all)')
    parser.add_argument('--print-only', action='store_true',
                        help='Solo muestra el Top 20 en texto, no genera imágenes')
    args = parser.parse_args()

    print('\n' + '═'*60)
    print('  MODELO ELO SELECCIONES + PREDICCIÓN MÉXICO vs PORTUGAL')
    print('═'*60)

    print('\n[1] Cargando dataset...')
    df = load_csv()
    print_stats(df)

    print('\n[2] Historial México vs Portugal:')
    h_mv = print_hist_mex_por(df)
    if h_mv is not None and len(h_mv) > 0:
        wins_mex = sum(1 for _, r in h_mv.iterrows()
                       if (r['home_team']=='Mexico' and r['home_score']>r['away_score']) or
                          (r['away_team']=='Mexico' and r['away_score']>r['home_score']))
        draws    = sum(1 for _, r in h_mv.iterrows() if r['home_score']==r['away_score'])
        wins_por = len(h_mv) - wins_mex - draws
    else:
        wins_mex, draws, wins_por = 0, 0, 0

    print('\n[3] Calculando ELO...')
    elos = calc_elo(df)
    print(f'    {len(elos)} selecciones procesadas.')

    print('\n[4] Aplicando partidos recientes de México...')
    elos = apply_extra_matches(elos, EXTRA_MEXICO)

    elo_mex  = elos.get('Mexico',   ELO_BASE)
    elo_por  = elos.get('Portugal', ELO_BASE)
    elo_mean = np.mean(list(elos.values()))

    top20     = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    rank_mex  = next((i+1 for i,(t,_) in enumerate(top20) if t=='Mexico'),  '?')
    rank_por  = next((i+1 for i,(t,_) in enumerate(top20) if t=='Portugal'),'?')
    elo_diff  = abs(elo_mex - elo_por)
    leader    = 'México' if elo_mex > elo_por else 'Portugal'

    print(f'\n[5] ELO actual:')
    print(f'    México   : {elo_mex:.1f}  (#{rank_mex} mundial)')
    print(f'    Portugal : {elo_por:.1f}  (#{rank_por} mundial)')

    print('\n[TOP 20 RANKING ELO — metodología eloratings.net]')
    print(f'  {"#":>2}  {"Selección":<22}  {"ELO":>6}')
    print(f'  {"─"*2}  {"─"*22}  {"─"*6}')
    for idx20, (t20, e20) in enumerate(sorted(elos.items(), key=lambda x: x[1], reverse=True)[:20], 1):
        mark = ' ◄ México' if t20 == 'Mexico' else (' ◄ Portugal' if t20 == 'Portugal' else '')
        print(f'  {idx20:>2}  {team_es(t20):<22}  {e20:>6.0f}{mark}')

    if args.print_only:
        print('\n' + '═'*60 + '\n')
        return

    mask       = df['home_score'].notna() & df['away_score'].notna()
    avg_goals  = (df.loc[mask,'home_score'].mean() + df.loc[mask,'away_score'].mean()) / 2
    lam_h, lam_a = elo_to_lambda(elo_mex, elo_por, elo_mean, avg_goals, 1.0)
    probs, p_home, p_draw, p_away = poisson_probs(lam_h, lam_a, max_goals=5)
    max_idx    = np.unravel_index(np.argmax(probs), probs.shape)
    best_p     = probs[max_idx] * 100

    print(f'\n[6] Predicción Poisson:')
    print(f'    P(MÉX gana)={p_home*100:.1f}%  P(Empate)={p_draw*100:.1f}%  P(POR)={p_away*100:.1f}%')
    print(f'    Más probable: México {max_idx[0]}-{max_idx[1]} Portugal ({best_p:.1f}%)')

    fact_ht = (f'Marcador más probable: México {max_idx[0]}-{max_idx[1]} Portugal ({best_p:.1f}%)  ·  '
               f'{leader} llega con {elo_diff:.0f} pts ELO de ventaja')
    fact_rk = (f'México #{rank_mex} ({elo_mex:.0f} pts)  ·  Portugal #{rank_por} ({elo_por:.0f} pts)  ·  '
               f'Diferencia: {elo_diff:.0f} puntos ELO')
    fact_u5 = (f'Historial MEX vs POR: {wins_mex}G {draws}E {wins_por}P '
               f'en {wins_mex+draws+wins_por} encuentros desde 1969')

    last_mex = last_n(df, 'Mexico',   5, extra=EXTRA_MEXICO)
    last_por = last_n(df, 'Portugal', 5)

    if args.chart in ('ultimos5', 'all'):
        print('\n[Gráfica] ultimos5...')
        render_ultimos5(last_mex, last_por,
                        OUT_DIR / 'selecciones_ultimos5.png', fact=fact_u5)

    if args.chart in ('heatmap', 'all'):
        print('\n[Gráfica] heatmap...')
        render_heatmap(probs, p_home, p_draw, p_away, lam_h, lam_a,
                       elo_mex, elo_por,
                       OUT_DIR / 'selecciones_prediccion.png', fact=fact_ht)

    if args.chart in ('ranking', 'all'):
        print('\n[Gráfica] ranking...')
        render_ranking(elos, elo_mex, elo_por,
                       OUT_DIR / 'selecciones_ranking_elo.png', fact=fact_rk)

    print('\n' + '═'*60 + '\n')


if __name__ == '__main__':
    main()
