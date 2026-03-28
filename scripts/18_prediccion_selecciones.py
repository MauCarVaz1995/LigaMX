#!/usr/bin/env python3
"""
18_prediccion_selecciones.py
Modelo ELO para selecciones + predicción México vs Portugal con Poisson.

Salidas:
  output/charts/selecciones_prediccion.png   — heatmap México vs Portugal
  output/charts/selecciones_ranking_elo.png  — Top 20 ranking ELO
  output/charts/selecciones_ultimos5.png     — últimos 5 partidos MÉX & POR
"""

import sys, warnings, math
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# Emoji fallback: trata de usar fuentes que soporten emojis
matplotlib.rcParams['font.sans-serif'] = [
    'DejaVu Sans', 'Noto Color Emoji', 'Segoe UI Emoji',
    'Apple Color Emoji', 'Symbola', 'sans-serif'
]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from scipy.stats import poisson

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETAS, PALETA_ACTIVA, get_paleta, bebas, hex_rgba, hex_rgb

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
CSV_PATH  = BASE / 'data/raw/internacional/results.csv'
OUT_DIR   = BASE / 'output/charts'
BEBAS_TTF = Path.home() / '.fonts/BebasNeue.ttf'

OUT_DIR.mkdir(parents=True, exist_ok=True)

if BEBAS_TTF.exists():
    fm.fontManager.addfont(str(BEBAS_TTF))

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
PAL    = get_paleta()
BG     = PAL['bg_primary']
BG2    = PAL['bg_secondary']
WHITE  = PAL['text_primary']
GRAY   = PAL['text_secondary']
RED    = PAL['accent']
ACC2   = PAL['accent2']
CHIGH  = PAL['cell_high']
CMID   = PAL['cell_mid']
CLOW   = PAL['cell_low']
BRAND  = PAL['brand_color']
GOLD   = '#FFD700'

# ─────────────────────────────────────────────────────────────────────────────
# NOMBRES EN ESPAÑOL Y BANDERAS
# ─────────────────────────────────────────────────────────────────────────────
TEAM_ES = {
    'Mexico': 'México',           'Portugal': 'Portugal',
    'Brazil': 'Brasil',           'France': 'Francia',
    'Spain': 'España',            'England': 'Inglaterra',
    'Argentina': 'Argentina',     'Germany': 'Alemania',
    'Italy': 'Italia',            'Belgium': 'Bélgica',
    'Netherlands': 'Países Bajos', 'Croatia': 'Croacia',
    'Morocco': 'Marruecos',       'South Korea': 'Corea del Sur',
    'Japan': 'Japón',             'Switzerland': 'Suiza',
    'Denmark': 'Dinamarca',       'Colombia': 'Colombia',
    'Uruguay': 'Uruguay',         'Chile': 'Chile',
    'Bolivia': 'Bolivia',         'Paraguay': 'Paraguay',
    'Panama': 'Panamá',           'Iceland': 'Islandia',
    'United States': 'EE.UU.',    'Republic of Ireland': 'Irlanda',
    'Armenia': 'Armenia',         'Hungary': 'Hungría',
    'Czech Republic': 'Rep. Checa', 'Slovakia': 'Eslovaquia',
    'Poland': 'Polonia',          'Serbia': 'Serbia',
    'Ukraine': 'Ucrania',         'Turkey': 'Turquía',
    'Senegal': 'Senegal',         'Canada': 'Canadá',
    'Ecuador': 'Ecuador',         'Wales': 'Gales',
    'Scotland': 'Escocia',        'Austria': 'Austria',
    'Sweden': 'Suecia',           'Norway': 'Noruega',
    'Finland': 'Finlandia',       'Russia': 'Rusia',
    'Greece': 'Grecia',           'Romania': 'Rumanía',
    'Iran': 'Irán',               'Australia': 'Australia',
    'Saudi Arabia': 'Arabia Saudí', 'Ghana': 'Ghana',
    'Nigeria': 'Nigeria',         'Egypt': 'Egipto',
    'Costa Rica': 'Costa Rica',   'Honduras': 'Honduras',
    'Jamaica': 'Jamaica',         'Venezuela': 'Venezuela',
    'Peru': 'Perú',               'China PR': 'China',
    'Ivory Coast': 'Costa de Marfil', 'Cuba': 'Cuba',
    'El Salvador': 'El Salvador', 'Guatemala': 'Guatemala',
    'Luxembourg': 'Luxemburgo',   'Albania': 'Albania',
    'Slovenia': 'Eslovenia',      'Israel': 'Israel',
    'Georgia': 'Georgia',
}

FLAGS = {
    'Mexico': '🇲🇽',       'Portugal': '🇵🇹',     'Brazil': '🇧🇷',
    'France': '🇫🇷',        'Spain': '🇪🇸',        'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    'Argentina': '🇦🇷',     'Germany': '🇩🇪',      'Italy': '🇮🇹',
    'Belgium': '🇧🇪',       'Netherlands': '🇳🇱',  'Croatia': '🇭🇷',
    'Morocco': '🇲🇦',       'South Korea': '🇰🇷',  'Japan': '🇯🇵',
    'Switzerland': '🇨🇭',   'Denmark': '🇩🇰',      'Colombia': '🇨🇴',
    'Uruguay': '🇺🇾',       'Chile': '🇨🇱',        'Bolivia': '🇧🇴',
    'Paraguay': '🇵🇾',      'Panama': '🇵🇦',       'Iceland': '🇮🇸',
    'United States': '🇺🇸', 'Republic of Ireland': '🇮🇪',
    'Armenia': '🇦🇲',       'Hungary': '🇭🇺',      'Czech Republic': '🇨🇿',
    'Slovakia': '🇸🇰',      'Poland': '🇵🇱',       'Serbia': '🇷🇸',
    'Ukraine': '🇺🇦',       'Turkey': '🇹🇷',       'Senegal': '🇸🇳',
    'Canada': '🇨🇦',        'Ecuador': '🇪🇨',      'Wales': '🏴󠁧󠁢󠁷󠁬󠁳󠁿',
    'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',    'Austria': '🇦🇹',      'Sweden': '🇸🇪',
    'Norway': '🇳🇴',        'Finland': '🇫🇮',      'Russia': '🇷🇺',
    'Greece': '🇬🇷',        'Romania': '🇷🇴',      'Iran': '🇮🇷',
    'Australia': '🇦🇺',     'Saudi Arabia': '🇸🇦',  'Ghana': '🇬🇭',
    'Nigeria': '🇳🇬',       'Egypt': '🇪🇬',        'Costa Rica': '🇨🇷',
    'Honduras': '🇭🇳',      'Jamaica': '🇯🇲',      'Venezuela': '🇻🇪',
    'Peru': '🇵🇪',          'China PR': '🇨🇳',     'Ivory Coast': '🇨🇮',
    'Cuba': '🇨🇺',          'El Salvador': '🇸🇻',  'Guatemala': '🇬🇹',
    'Luxembourg': '🇱🇺',    'Albania': '🇦🇱',      'Slovenia': '🇸🇮',
    'Israel': '🇮🇱',        'Georgia': '🇬🇪',
}

MONTHS_ES = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',
             7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}

def fmt_date(d):
    """date → '14 oct 2025'"""
    return f"{d.day} {MONTHS_ES[d.month]} {d.year}"

def team_es(name): return TEAM_ES.get(name, name)
def team_flag(name): return FLAGS.get(name, '🌐')

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS ELO
# ─────────────────────────────────────────────────────────────────────────────
ELO_BASE   = 1500
K          = 40
HOME_ADV   = 100
SCALE      = 400
REGRESSION = 0.10

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
        print('  — No hay enfrentamientos en el dataset.'); return
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
    elos = defaultdict(lambda: ELO_BASE)
    prev_year = None
    for r in df.sort_values('date').itertuples(index=False):
        yr = pd.Timestamp(r.date).year
        if prev_year is not None and yr != prev_year:
            for t in list(elos.keys()):
                elos[t] += REGRESSION * (ELO_BASE - elos[t])
        prev_year = yr
        home, away = r.home_team, r.away_team
        gl, gv = int(r.home_score), int(r.away_score)
        neutral = str(r.neutral).upper() == 'TRUE'
        adv = 0 if neutral else HOME_ADV
        ea  = expected_score(elos[home], elos[away], home_adv=adv)
        sa  = result_score(gl, gv)
        torneo = str(r.tournament).lower()
        k_mult = 1.25 if any(x in torneo for x in
            ['world cup', 'copa america', 'euro', 'gold cup',
             'nations league', 'olympic', 'confederation']) else 1.0
        elos[home] += K * k_mult * (sa - ea)
        elos[away] += K * k_mult * ((1 - sa) - (1 - ea))
    return dict(elos)

def apply_extra_matches(elos, extra):
    for date_str, home, away, gl, gv, torneo, neutral in extra:
        adv = 0 if neutral else HOME_ADV
        ea  = expected_score(elos.get(home, ELO_BASE), elos.get(away, ELO_BASE), home_adv=adv)
        sa  = result_score(gl, gv)
        k_mult = 1.25 if any(x in torneo.lower() for x in
            ['world cup', 'copa', 'euro', 'gold cup', 'nations league']) else 1.0
        elos[home] = elos.get(home, ELO_BASE) + K * k_mult * (sa - ea)
        elos[away] = elos.get(away, ELO_BASE) + K * k_mult * ((1 - sa) - (1 - ea))
    return elos

# ─────────────────────────────────────────────────────────────────────────────
# MODELO POISSON
# ─────────────────────────────────────────────────────────────────────────────
def poisson_probs(lam_h, lam_a, max_goals=5):
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    p_home = np.sum(np.tril(probs, -1))
    p_draw = np.sum(np.diag(probs))
    p_away = np.sum(np.triu(probs, 1))
    return probs, p_home, p_draw, p_away

def elo_to_lambda(elo_local, elo_away, elo_mean, avg_goals, loc_factor=1.0):
    lam_h = avg_goals * (elo_local / elo_mean) * loc_factor
    lam_a = avg_goals * (elo_away  / elo_mean)
    return lam_h, lam_a

# ─────────────────────────────────────────────────────────────────────────────
# ÚLTIMOS N PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────
def last_n(df, team, n=5, extra=None):
    mask = (df['home_team'] == team) | (df['away_team'] == team)
    rows = df[mask].copy().sort_values('date').tail(n)
    result = []
    for _, r in rows.iterrows():
        is_home = (r['home_team'] == team)
        opp  = r['away_team'] if is_home else r['home_team']
        gf   = int(r['home_score']) if is_home else int(r['away_score'])
        ga   = int(r['away_score']) if is_home else int(r['home_score'])
        result.append({
            'date': r['date'].date(), 'opponent': opp,
            'gf': gf, 'ga': ga,
            'ha': 'L' if is_home else 'V',
            'res': 'G' if gf > ga else ('E' if gf == ga else 'P'),
            'tournament': r['tournament']
        })
    if extra and team == 'Mexico':
        for date_str, home, away, gl, gv, torneo, neutral in extra:
            is_home = (home == team)
            opp  = away if is_home else home
            gf   = gl if is_home else gv
            ga   = gv if is_home else gl
            result.append({
                'date': datetime.strptime(date_str, '%Y-%m-%d').date(),
                'opponent': opp, 'gf': gf, 'ga': ga,
                'ha': 'L' if is_home else 'V',
                'res': 'G' if gf > ga else ('E' if gf == ga else 'P'),
                'tournament': torneo
            })
        result = sorted(result, key=lambda x: x['date'])[-n:]
    return result

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE DIBUJO
# ─────────────────────────────────────────────────────────────────────────────
def _bg_gradient(fig):
    """Aplica gradiente de fondo a toda la figura."""
    bg_rgb  = hex_rgb(BG)
    bg2_rgb = hex_rgb(BG2)
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = np.array(bg_rgb) / 255 * (1 - t) + np.array(bg2_rgb) / 255 * t
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.set_zorder(-100)
    bgax.imshow(grad, aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    bgax.axis('off')

def _footer(fig, source_text, h=0.050, fact=None):
    """Pie de página estándar."""
    if fact:
        fct = fig.add_axes([0, h, 1, 0.030])
        fct.set_facecolor(BG); fct.axis('off')
        fct.text(0.5, 0.5, fact,
                 color=GRAY, fontsize=8.5, ha='center', va='center',
                 style='italic', transform=fct.transAxes)
    fax = fig.add_axes([0, 0, 1, h * 0.55])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.5)
    fax.text(0.015, 0.50, source_text, color=GRAY, fontsize=9,
             ha='left', va='center', transform=fax.transAxes)
    shadow = [mpe.withStroke(linewidth=3, foreground='#000000')]
    fax.text(0.985, 0.50, 'MAU-STATISTICS',
             color=BRAND, ha='right', va='center',
             transform=fax.transAxes, path_effects=shadow, **bebas(20))

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN 1: HEATMAP PREDICCIÓN (5×5)
# ─────────────────────────────────────────────────────────────────────────────
def render_heatmap(probs, p_home, p_draw, p_away, lam_h, lam_a,
                   elo_mex, elo_por, out_path, fact=''):
    MAX_G    = 5       # 0-5 goles
    N        = MAX_G + 1
    FIG_W, FIG_H = 12.0, 13.0

    HEADER_H  = 0.215  # título + paneles de equipo
    FOOTER_H  = 0.055
    FACT_H    = 0.033
    PROB_H    = 0.110  # barra de probabilidades (más alta)
    COLHDR_H  = 0.055  # "GOLES PORTUGAL" + números de columna
    ROWHDR_W  = 0.095  # números de fila + "GOL MÉX"
    R_MARGIN  = 0.012

    GRID_Y = FOOTER_H + FACT_H + PROB_H + 0.005
    GRID_H = 1.0 - HEADER_H - GRID_Y - COLHDR_H
    CELL_H = GRID_H / N
    CELL_W = (1.0 - ROWHDR_W - R_MARGIN) / N

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _bg_gradient(fig)

    # ── HEADER ───────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=4.0)

    # Título principal (top 40%)
    hax.text(0.50, 0.98, '¿QUIÉN GANA HOY?',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(38))
    hax.text(0.50, 0.64,
             f'{team_flag("Mexico")} México vs Portugal {team_flag("Portugal")}  ·  Amistoso 2026',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=15)

    # Paneles de equipo (bottom 50% del header)
    for col_x, flag, name_es, elo, lam, border_c in [
        (0.07, team_flag('Mexico'),   'MÉXICO',   elo_mex, lam_h, CHIGH),
        (0.57, team_flag('Portugal'), 'PORTUGAL', elo_por, lam_a, ACC2),
    ]:
        pax = fig.add_axes([col_x, 1 - HEADER_H + HEADER_H * 0.06,
                            0.35,   HEADER_H * 0.44])
        pax.set_facecolor(BG); pax.axis('off')
        for sp in pax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(border_c); sp.set_linewidth(1.8)
        pax.text(0.50, 0.80, f'{flag}  {name_es}',
                 color=border_c, ha='center', va='center',
                 fontsize=15, fontweight='bold', transform=pax.transAxes)
        pax.text(0.50, 0.48, f'ELO: {elo:.0f}',
                 color=WHITE, ha='center', va='center',
                 fontsize=13, transform=pax.transAxes)
        pax.text(0.50, 0.18, f'λ esperado = {lam:.3f} goles',
                 color=GRAY, ha='center', va='center',
                 fontsize=9.5, transform=pax.transAxes)

    # VS
    vax = fig.add_axes([0.40, 1 - HEADER_H + HEADER_H * 0.06, 0.20, HEADER_H * 0.44])
    vax.set_facecolor(BG2); vax.axis('off')
    vax.text(0.50, 0.55, 'VS',
             color=RED, ha='center', va='center', fontsize=22, fontweight='bold',
             transform=vax.transAxes)
    vax.text(0.50, 0.20, 'Poisson · ELO',
             color=GRAY, ha='center', va='center',
             fontsize=8.5, transform=vax.transAxes)

    # ── COLUMN HEADER (GOLES PORTUGAL + números) ─────────────────────────────
    chdr_y = GRID_Y + GRID_H
    # Etiqueta "GOLES PORTUGAL"
    lbl_ax = fig.add_axes([ROWHDR_W, chdr_y + COLHDR_H * 0.55,
                           N * CELL_W, COLHDR_H * 0.42])
    lbl_ax.set_facecolor(BG2); lbl_ax.axis('off')
    lbl_ax.text(0.5, 0.5, f'GOLES  {team_flag("Portugal")}  PORTUGAL',
                color=ACC2, ha='center', va='center',
                fontsize=11, fontweight='bold', transform=lbl_ax.transAxes)
    # Números 0–5
    for j in range(N):
        nax = fig.add_axes([ROWHDR_W + j * CELL_W, chdr_y, CELL_W, COLHDR_H * 0.55])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.5, 0.5, str(j), color=WHITE, ha='center', va='center',
                 fontsize=12, fontweight='bold', transform=nax.transAxes)

    # ── ROW HEADER (GOLES MÉXICO + números) ──────────────────────────────────
    # Etiqueta "GOLES MÉXICO" — rotada
    rmx = fig.add_axes([0, GRID_Y, ROWHDR_W * 0.35, GRID_H])
    rmx.set_facecolor(BG2); rmx.axis('off')
    rmx.text(0.5, 0.5, f'GOLES  {team_flag("Mexico")}  MÉXICO',
             color=CHIGH, ha='center', va='center', fontsize=10, fontweight='bold',
             rotation=90, transform=rmx.transAxes)
    # Números 0–5
    for i in range(N):
        nax = fig.add_axes([ROWHDR_W * 0.35, GRID_Y + i * CELL_H,
                            ROWHDR_W * 0.65, CELL_H])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.65, 0.5, str(i), color=WHITE, ha='center', va='center',
                 fontsize=12, fontweight='bold', transform=nax.transAxes)

    # ── HEATMAP CELLS ─────────────────────────────────────────────────────────
    def cell_style(p):
        if p > 0.090:  return CHIGH, '#000000', True,  14
        elif p > 0.045: return CMID, WHITE,     True,  12
        elif p > 0.015: return CLOW, CHIGH,     True,  10
        elif p > 0.004: return BG2,  GRAY,      False,  9
        else:           return BG,   BG2,       False,  8

    for i in range(N):          # filas = goles México
        for j in range(N):      # cols  = goles Portugal
            p  = probs[i, j]
            fc, tc, bold, fs = cell_style(p)
            cx = ROWHDR_W + j * CELL_W
            cy = GRID_Y   + i * CELL_H

            cax = fig.add_axes([cx, cy, CELL_W, CELL_H])
            cax.set_xlim(0, 1); cax.set_ylim(0, 1)
            cax.add_patch(mpatches.Rectangle(
                (0, 0), 1, 1, facecolor=fc, edgecolor='none', zorder=0,
                transform=cax.transAxes, clip_on=False))
            cax.axis('off')

            # Borde dorado en empate
            if i == j:
                for sp in cax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(GOLD); sp.set_linewidth(2.0)

            if p >= 0.003:
                txt = f'{p*100:.1f}' if p >= 0.010 else f'{p*100:.2f}'
                cax.text(0.5, 0.5, txt,
                         color=tc, ha='center', va='center',
                         fontsize=fs, fontweight='bold' if bold else 'normal',
                         transform=cax.transAxes, zorder=5)

    # ── BARRA DE PROBABILIDADES (grande, con banderas) ───────────────────────
    prob_y = FOOTER_H + FACT_H + 0.005
    total  = p_home + p_draw + p_away + 1e-9
    w_h, w_d, w_a = p_home/total, p_draw/total, p_away/total

    segments = [
        (0,              w_h, CHIGH,   f'{team_flag("Mexico")} GANA MÉXICO',   p_home, '#000000'),
        (w_h,            w_d, ACC2,    'EMPATE',                               p_draw, '#000000'),
        (w_h + w_d,      w_a, RED,     f'GANA PORTUGAL {team_flag("Portugal")}', p_away, WHITE),
    ]
    GAP = 0.004
    for xstart, width, color, label, pval, tc in segments:
        seg = fig.add_axes([xstart + GAP, prob_y, width - 2*GAP, PROB_H])
        seg.set_facecolor(color); seg.axis('off')
        seg.set_xlim(0, 1); seg.set_ylim(0, 1)
        if width > 0.06:
            seg.text(0.5, 0.65, f'{pval*100:.1f}%',
                     color=tc, ha='center', va='center',
                     fontsize=17 if width > 0.20 else 13,
                     fontweight='bold', transform=seg.transAxes)
            seg.text(0.5, 0.22, label,
                     color=tc, ha='center', va='center',
                     fontsize=9 if width > 0.18 else 7.5,
                     transform=seg.transAxes)

    _footer(fig,
            'Modelo: ELO + Poisson · Fuente: martj42/international_results',
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
    FIG_W, FIG_H = 11.0, 13.0

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _bg_gradient(fig)

    HEADER_H = 0.130
    FOOTER_H = 0.055
    FACT_H   = 0.030
    CONTENT_Y = FOOTER_H + FACT_H
    CONTENT_H = 1 - HEADER_H - FOOTER_H - FACT_H
    ROW_H = CONTENT_H / n

    # Header
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=4.0)
    hax.text(0.50, 0.90, 'RANKING ELO — SELECCIONES NACIONALES',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(30))
    hax.text(0.50, 0.42, 'Top 20 · Modelo ELO histórico · K=40 · Regresión 10% anual · ~49,000 partidos',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=11)

    elo_max = top20[0][1]
    elo_min = min(e for _, e in top20) - 30
    FEATURED = {'Mexico', 'Portugal'}

    for i, (team, elo) in enumerate(top20):
        row_y  = CONTENT_Y + (n - 1 - i) * ROW_H
        is_ft  = team in FEATURED
        # Fila más clara para los destacados
        row_bg = '#1a2510' if team == 'Mexico' else ('#1a1030' if team == 'Portugal' else
                  (BG2 if i % 2 == 0 else BG))

        ax = fig.add_axes([0, row_y, 1, ROW_H])
        ax.set_facecolor(row_bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

        # Borde dorado para México y Portugal
        if is_ft:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(GOLD); sp.set_linewidth(1.5)
            # Acento lateral dorado
            ax.add_patch(mpatches.Rectangle(
                (0, 0.05), 0.006, 0.90, facecolor=GOLD, linewidth=0))

        # Posición
        pos_color = GOLD if i < 3 else (CMID if i < 8 else WHITE)
        ax.text(0.022, 0.50, f'{i+1:2d}',
                color=pos_color if not is_ft else GOLD,
                fontsize=12, fontweight='bold', va='center')

        # Bandera + nombre en español
        flag = team_flag(team)
        name = team_es(team)
        name_color = CHIGH if team == 'Mexico' else (ACC2 if team == 'Portugal' else
                     (GOLD if i < 3 else WHITE))
        ax.text(0.048, 0.55, flag,
                color=name_color, fontsize=13, va='center')
        ax.text(0.085, 0.55, name,
                color=name_color,
                fontsize=12, fontweight='bold' if is_ft else 'normal', va='center')

        # Barra — verde CMID para todos, destacados en GOLD
        bar_x  = 0.365
        bar_w  = 0.490
        bar_val = max(0, (elo - elo_min) / (elo_max - elo_min + 1))
        ax.add_patch(mpatches.Rectangle(
            (bar_x, 0.25), bar_w, 0.50, facecolor=BG2, linewidth=0))
        bar_col = GOLD if is_ft else CMID
        ax.add_patch(mpatches.Rectangle(
            (bar_x, 0.25), bar_w * bar_val, 0.50,
            facecolor=bar_col, linewidth=0, alpha=0.85))

        # ELO value
        ax.text(0.875, 0.55, f'{elo:.0f}',
                color=GOLD if is_ft else WHITE,
                fontsize=12, fontweight='bold' if is_ft else 'normal',
                va='center', ha='right')

        # Partidos
        ax.axhline(0, color=BG, lw=0.6)

    _footer(fig,
            'Fuente: martj42/international_results · Partidos desde 1872',
            h=FOOTER_H, fact=fact)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {out_path}')

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN 3: ÚLTIMOS 5 PARTIDOS (FORMA RECIENTE)
# ─────────────────────────────────────────────────────────────────────────────
def render_ultimos5(last_mex, last_por, out_path, fact=''):
    FIG_W, FIG_H = 13.0, 9.5
    N = 5

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _bg_gradient(fig)

    HEADER_H = 0.160
    FOOTER_H = 0.055
    FACT_H   = 0.030
    CONTENT_Y = FOOTER_H + FACT_H
    CONTENT_H = 1 - HEADER_H - FOOTER_H - FACT_H - 0.045  # 0.045 para título de columna
    ROW_H    = CONTENT_H / N
    COL_W    = 0.500

    # Header
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=4.0)
    hax.text(0.50, 0.90, 'ASÍ LLEGAN AL PARTIDO',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(33))
    hax.text(0.50, 0.44,
             f'{team_flag("Mexico")} México vs Portugal {team_flag("Portugal")}  ·  Últimos 5 partidos',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=14)

    # ── Columnas ─────────────────────────────────────────────────────────────
    TEAMS = [
        ('Mexico',   last_mex, 0.0,    CHIGH),
        ('Portugal', last_por, COL_W,  ACC2),
    ]

    for team_key, matches, col_x, team_color in TEAMS:
        name_es = team_es(team_key)
        flag    = team_flag(team_key)

        # Título de columna
        col_top_y = CONTENT_Y + CONTENT_H
        tax = fig.add_axes([col_x, col_top_y, COL_W, 0.040])
        tax.set_facecolor(BG2); tax.axis('off')
        tax.text(0.5, 0.5, f'{flag}  {name_es.upper()}',
                 color=team_color, ha='center', va='center',
                 fontsize=14, fontweight='bold', transform=tax.transAxes)

        for idx, m in enumerate(reversed(matches)):
            row_y  = CONTENT_Y + idx * ROW_H
            row_bg = BG2 if idx % 2 == 0 else BG

            rax = fig.add_axes([col_x, row_y, COL_W, ROW_H])
            rax.set_facecolor(row_bg); rax.axis('off')
            rax.set_xlim(0, 1); rax.set_ylim(0, 1)
            rax.axhline(0, color=BG, lw=0.8)

            # ── Resultado badge (grande) ──────────────────────────────────────
            res_color = CHIGH if m['res'] == 'G' else (GRAY if m['res'] == 'E' else RED)
            res_tc    = '#000000' if m['res'] == 'G' else WHITE
            rax.add_patch(mpatches.FancyBboxPatch(
                (0.018, 0.12), 0.105, 0.76,
                boxstyle='round,pad=0.005',
                facecolor=res_color, linewidth=0, zorder=3))
            rax.text(0.070, 0.52, m['res'],
                     color=res_tc, ha='center', va='center',
                     fontsize=14, fontweight='bold', zorder=4)

            # ── Marcador ──────────────────────────────────────────────────────
            score = f'{m["gf"]}-{m["ga"]}'
            rax.text(0.185, 0.60, score,
                     color=WHITE, fontsize=14, fontweight='bold', va='center')

            # ── H/A indicator ─────────────────────────────────────────────────
            ha_lbl  = 'LOCAL' if m['ha'] == 'L' else 'VISIT'
            ha_col  = team_color if m['ha'] == 'L' else GRAY
            rax.text(0.185, 0.24, ha_lbl,
                     color=ha_col, fontsize=7, va='center', fontweight='bold')

            # ── Bandera + rival ───────────────────────────────────────────────
            opp_flag  = team_flag(m['opponent'])
            opp_es    = team_es(m['opponent'])
            rax.text(0.305, 0.62, f'{opp_flag}  {opp_es}',
                     color=WHITE, fontsize=11, fontweight='bold', va='center')

            # ── Torneo ────────────────────────────────────────────────────────
            rax.text(0.308, 0.26, m['tournament'][:28],
                     color=GRAY, fontsize=7.5, va='center')

            # ── Fecha en formato humano ───────────────────────────────────────
            rax.text(0.970, 0.52, fmt_date(m['date']),
                     color=GRAY, fontsize=9, ha='right', va='center')

    # Divisor vertical
    div = fig.add_axes([COL_W - 0.003, CONTENT_Y, 0.006, CONTENT_H + 0.040])
    div.set_facecolor(RED); div.axis('off')

    _footer(fig,
            'Fuente: martj42/international_results + resultados recientes México',
            h=FOOTER_H, fact=fact)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {out_path}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '═' * 60)
    print('  MODELO ELO SELECCIONES + PREDICCIÓN MÉXICO vs PORTUGAL')
    print('═' * 60)

    # 1. Cargar CSV
    print('\n[1] Cargando dataset...')
    df = load_csv()
    print_stats(df)

    # 2. Historial México vs Portugal
    print('\n[2] Historial México vs Portugal:')
    h_mv = print_hist_mex_por(df)
    if h_mv is not None and len(h_mv) > 0:
        wins_mex = sum(1 for _, r in h_mv.iterrows()
                       if (r['home_team']=='Mexico' and r['home_score']>r['away_score']) or
                          (r['away_team']=='Mexico' and r['away_score']>r['home_score']))
        draws = sum(1 for _, r in h_mv.iterrows() if r['home_score']==r['away_score'])
        wins_por = len(h_mv) - wins_mex - draws
    else:
        wins_mex, draws, wins_por = 0, 0, 0

    # 3. Calcular ELO
    print('\n[3] Calculando ELO para todas las selecciones...')
    elos = calc_elo(df)
    print(f'    {len(elos)} selecciones procesadas.')

    # 4. Partidos recientes México
    print('\n[4] Aplicando partidos recientes de México...')
    elos = apply_extra_matches(elos, EXTRA_MEXICO)
    for fecha, home, away, gl, gv, torneo, _ in EXTRA_MEXICO:
        print(f'    {fecha}  {home} {gl}-{gv} {away}  [{torneo}]')

    # 5. ELOs finales
    elo_mex  = elos.get('Mexico',   ELO_BASE)
    elo_por  = elos.get('Portugal', ELO_BASE)
    elo_mean = np.mean(list(elos.values()))

    top20 = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    rank_mex = next((i+1 for i,(t,_) in enumerate(top20) if t=='Mexico'), '?')
    rank_por = next((i+1 for i,(t,_) in enumerate(top20) if t=='Portugal'), '?')

    print(f'\n[5] ELO actual:')
    print(f'    México   : {elo_mex:.1f}  (#{rank_mex} mundial)')
    print(f'    Portugal : {elo_por:.1f}  (#{rank_por} mundial)')
    print(f'    Promedio : {elo_mean:.1f}')

    # 6. Poisson
    mask = df['home_score'].notna() & df['away_score'].notna()
    avg_goals = (df.loc[mask,'home_score'].mean() + df.loc[mask,'away_score'].mean()) / 2
    lam_h, lam_a = elo_to_lambda(elo_mex, elo_por, elo_mean, avg_goals, 1.0)
    probs, p_home, p_draw, p_away = poisson_probs(lam_h, lam_a, max_goals=5)

    print(f'\n[6] Predicción Poisson:')
    print(f'    avg_goals={avg_goals:.3f}  lam_MÉX={lam_h:.3f}  lam_POR={lam_a:.3f}')
    print(f'    P(México gana) = {p_home*100:.1f}%')
    print(f'    P(Empate)      = {p_draw*100:.1f}%')
    print(f'    P(Portugal)    = {p_away*100:.1f}%')
    max_idx = np.unravel_index(np.argmax(probs), probs.shape)
    best_p  = probs[max_idx] * 100
    print(f'    Marcador más probable: México {max_idx[0]}-{max_idx[1]} Portugal ({best_p:.1f}%)')

    elo_diff = abs(elo_mex - elo_por)
    leader   = 'México' if elo_mex > elo_por else 'Portugal'

    # 7. Heatmap
    print('\n[7] Generando heatmap...')
    fact_ht = (f'Marcador más probable: México {max_idx[0]}-{max_idx[1]} Portugal ({best_p:.1f}%)  ·  '
               f'{leader} llega con {elo_diff:.0f} pts ELO de ventaja')
    render_heatmap(probs, p_home, p_draw, p_away, lam_h, lam_a,
                   elo_mex, elo_por,
                   OUT_DIR / 'selecciones_prediccion.png',
                   fact=fact_ht)

    # 8. Ranking
    print('\n[8] Generando ranking ELO Top 20...')
    fact_rk = (f'México #{rank_mex} ({elo_mex:.0f} pts) y Portugal #{rank_por} ({elo_por:.0f} pts)  ·  '
               f'Diferencia: {elo_diff:.0f} puntos ELO')
    render_ranking(elos, elo_mex, elo_por,
                   OUT_DIR / 'selecciones_ranking_elo.png',
                   fact=fact_rk)

    # 9. Forma reciente
    print('\n[9] Generando forma reciente...')
    last_mex = last_n(df, 'Mexico',   5, extra=EXTRA_MEXICO)
    last_por = last_n(df, 'Portugal', 5)
    hist_txt = f'Historial MEX vs POR: {wins_mex}G {draws}E {wins_por}P en {wins_mex+draws+wins_por} encuentros desde 1969'
    render_ultimos5(last_mex, last_por,
                    OUT_DIR / 'selecciones_ultimos5.png',
                    fact=hist_txt)

    print('\n' + '═' * 60)
    print('  Listo.')
    print('═' * 60 + '\n')


if __name__ == '__main__':
    main()
