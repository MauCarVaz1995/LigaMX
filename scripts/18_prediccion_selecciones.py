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

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS ELO
# ─────────────────────────────────────────────────────────────────────────────
ELO_BASE    = 1500
K           = 40
HOME_ADV    = 100
SCALE       = 400
REGRESSION  = 0.10   # 10% regresión a la media al inicio de cada año

# ─────────────────────────────────────────────────────────────────────────────
# PARTIDOS RECIENTES DE MÉXICO (no disponibles en el CSV público)
# ─────────────────────────────────────────────────────────────────────────────
EXTRA_MEXICO = [
    # (fecha, local, visitante, goles_local, goles_visit, torneo, neutral)
    ('2024-10-12', 'Mexico', 'Panama',    1, 0, 'Friendly', True),
    ('2024-10-15', 'Mexico', 'Paraguay',  1, 2, 'Friendly', True),
    ('2024-10-19', 'Mexico', 'Uruguay',   0, 0, 'Friendly', True),
    ('2025-01-18', 'Mexico', 'Iceland',   4, 0, 'Friendly', True),
    ('2025-01-22', 'Mexico', 'Bolivia',   1, 0, 'Friendly', True),
]

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
def load_csv():
    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ESTADÍSTICAS BÁSICAS
# ─────────────────────────────────────────────────────────────────────────────
def print_stats(df):
    print(f'\n{"─"*60}')
    print(f'  Fecha más reciente : {df["date"].max().date()}')
    print(f'  Total partidos     : {len(df):,}')
    mex = df[(df['home_team'] == 'Mexico') | (df['away_team'] == 'Mexico')]
    por = df[(df['home_team'] == 'Portugal') | (df['away_team'] == 'Portugal')]
    print(f'  Partidos México    : {len(mex):,}')
    print(f'  Partidos Portugal  : {len(por):,}')
    print(f'{"─"*60}')

# ─────────────────────────────────────────────────────────────────────────────
# HISTORIAL MÉXICO VS PORTUGAL
# ─────────────────────────────────────────────────────────────────────────────
def print_hist_mex_por(df):
    mask = (
        ((df['home_team'] == 'Mexico') & (df['away_team'] == 'Portugal')) |
        ((df['home_team'] == 'Portugal') & (df['away_team'] == 'Mexico'))
    )
    h = df[mask].copy().sort_values('date')
    print(f'\n  Historial México vs Portugal ({len(h)} partidos):')
    if h.empty:
        print('  — No hay enfrentamientos en el dataset.')
        return
    for _, r in h.iterrows():
        print(f'  {r["date"].date()}  {r["home_team"]:20s} {r["home_score"]}-{r["away_score"]}  {r["away_team"]:20s}  [{r["tournament"]}]')

# ─────────────────────────────────────────────────────────────────────────────
# MODELO ELO
# ─────────────────────────────────────────────────────────────────────────────
def expected_score(elo_a, elo_b, home_adv=0):
    """P(A gana) según ELO. home_adv se suma al ELO de A si juega en casa."""
    return 1 / (1 + 10 ** ((elo_b - (elo_a + home_adv)) / SCALE))

def result_score(gl, gv):
    """1=local gana, 0=visitante gana, 0.5=empate."""
    if gl > gv:   return 1.0
    elif gl < gv: return 0.0
    return 0.5

def calc_elo(df):
    elos   = defaultdict(lambda: ELO_BASE)
    prev_year = None

    rows = df.sort_values('date').itertuples(index=False)
    for r in rows:
        yr = r.date.year if hasattr(r.date, 'year') else pd.Timestamp(r.date).year

        # Regresión a la media cada año nuevo
        if prev_year is not None and yr != prev_year:
            for t in list(elos.keys()):
                elos[t] = elos[t] + REGRESSION * (ELO_BASE - elos[t])

        prev_year = yr

        home, away = r.home_team, r.away_team
        gl, gv = int(r.home_score), int(r.away_score)
        neutral = str(r.neutral).upper() == 'TRUE'

        adv = 0 if neutral else HOME_ADV
        ea  = expected_score(elos[home], elos[away], home_adv=adv)
        eb  = 1 - ea

        sa  = result_score(gl, gv)
        sb  = 1 - sa

        # K scaling: torneos más importantes → K×1.25 (WC, CONMEBOL, UEFA)
        torneo = str(r.tournament).lower()
        k_mult = 1.25 if any(x in torneo for x in
                              ['world cup', 'copa america', 'euro', 'gold cup',
                               'nations league', 'olympic', 'confederation']) else 1.0

        elos[home] += K * k_mult * (sa - ea)
        elos[away] += K * k_mult * (sb - eb)

    return dict(elos)

def apply_extra_matches(elos, extra):
    """Aplica los partidos extra de México al ELO ya calculado."""
    for date_str, home, away, gl, gv, torneo, neutral in extra:
        adv = 0 if neutral else HOME_ADV
        ea  = expected_score(elos.get(home, ELO_BASE), elos.get(away, ELO_BASE), home_adv=adv)
        eb  = 1 - ea
        sa  = result_score(gl, gv)
        sb  = 1 - sa
        k_mult = 1.25 if any(x in torneo.lower() for x in
                              ['world cup', 'copa', 'euro', 'gold cup',
                               'nations league', 'olympic']) else 1.0
        elos[home] = elos.get(home, ELO_BASE) + K * k_mult * (sa - ea)
        elos[away] = elos.get(away, ELO_BASE) + K * k_mult * (sb - eb)
    return elos

# ─────────────────────────────────────────────────────────────────────────────
# MODELO POISSON
# ─────────────────────────────────────────────────────────────────────────────
def poisson_probs(lam_h, lam_a, max_goals=8):
    """Matriz de probabilidades de marcador y P(local/empate/visitante)."""
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    p_home = np.sum(np.tril(probs, -1))
    p_draw = np.sum(np.diag(probs))
    p_away = np.sum(np.triu(probs, 1))
    return probs, p_home, p_draw, p_away

def elo_to_lambda(elo_local, elo_away, elo_mean, avg_goals, loc_factor=1.10):
    """
    lambda_local = avg_goals * (elo_local / elo_mean) * loc_factor
    lambda_away  = avg_goals * (elo_away  / elo_mean)
    """
    lam_h = avg_goals * (elo_local / elo_mean) * loc_factor
    lam_a = avg_goals * (elo_away  / elo_mean)
    return lam_h, lam_a

# ─────────────────────────────────────────────────────────────────────────────
# ÚLTIMOS N PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────
def last_n(df, team, n=5, extra=None):
    """Últimos N partidos de un equipo (incluyendo extra si aplica)."""
    mask = (df['home_team'] == team) | (df['away_team'] == team)
    rows = df[mask].copy().sort_values('date').tail(n)
    result = []
    for _, r in rows.iterrows():
        is_home = (r['home_team'] == team)
        opp  = r['away_team'] if is_home else r['home_team']
        gf   = int(r['home_score']) if is_home else int(r['away_score'])
        ga   = int(r['away_score']) if is_home else int(r['home_score'])
        HA   = 'L' if is_home else 'V'
        res  = 'G' if gf > ga else ('E' if gf == ga else 'P')
        result.append({
            'date': r['date'].date(),
            'opponent': opp, 'gf': gf, 'ga': ga,
            'ha': HA, 'res': res, 'tournament': r['tournament']
        })

    # Agregar extra si el equipo es Mexico
    if extra and team == 'Mexico':
        for date_str, home, away, gl, gv, torneo, neutral in extra:
            is_home = (home == team)
            opp  = away if is_home else home
            gf   = gl if is_home else gv
            ga   = gv if is_home else gl
            HA   = 'L' if is_home else 'V'
            res  = 'G' if gf > ga else ('E' if gf == ga else 'P')
            result.append({
                'date': datetime.strptime(date_str, '%Y-%m-%d').date(),
                'opponent': opp, 'gf': gf, 'ga': ga,
                'ha': HA, 'res': res, 'tournament': torneo
            })
        result = sorted(result, key=lambda x: x['date'])[-n:]

    return result

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN 1: HEATMAP MARCADORES
# ─────────────────────────────────────────────────────────────────────────────
def render_heatmap(probs, p_home, p_draw, p_away, lam_h, lam_a,
                   elo_mex, elo_por, out_path):
    max_g = probs.shape[0] - 1
    FIG_W, FIG_H = 13.0, 10.0

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # ── Gradient background ──────────────────────────────────────────────────
    bg_rgb  = hex_rgb(BG)
    bg2_rgb = hex_rgb(BG2)
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = (np.array(bg_rgb) / 255 * (1 - t) + np.array(bg2_rgb) / 255 * t)
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.set_zorder(-100)
    bgax.imshow(grad, aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    bgax.axis('off')

    # ── Layout ───────────────────────────────────────────────────────────────
    HEADER_H = 0.155
    FOOTER_H = 0.085
    L_MARGIN = 0.055
    R_MARGIN = 0.012
    GRID_Y   = FOOTER_H + 0.085
    GRID_H   = 1.0 - HEADER_H - FOOTER_H - 0.085

    CELL_W = (1 - L_MARGIN - R_MARGIN) / (max_g + 1)
    CELL_H = GRID_H / (max_g + 1)

    # ── Header ───────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=4.0)

    hax.text(0.50, 0.88, '¿QUIÉN GANA HOY?',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(38))
    hax.text(0.50, 0.52, 'México vs Portugal · Amistoso 2026',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=16)
    hax.text(0.50, 0.22,
             f'ELO México: {elo_mex:.0f}  ·  ELO Portugal: {elo_por:.0f}  ·  '
             f'λ MÉX={lam_h:.2f}  λ POR={lam_a:.2f}  ·  10,000 comb. Poisson',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=9.5)

    # ── Axis labels ──────────────────────────────────────────────────────────
    lax = fig.add_axes([0, GRID_Y, L_MARGIN, GRID_H])
    lax.set_facecolor(BG); lax.axis('off')
    lax.set_xlim(0, 1); lax.set_ylim(0, max_g + 1)
    for j in range(max_g + 1):
        lax.text(0.85, j + 0.5, str(j),
                 color=WHITE, ha='right', va='center', fontsize=10, fontweight='bold')
    lax.text(0.40, (max_g + 1) / 2, 'GOL. MÉXICO',
             color=BRAND, ha='center', va='center', fontsize=8.5,
             rotation=90, fontweight='bold')

    tax = fig.add_axes([L_MARGIN, GRID_Y + GRID_H, (1 - L_MARGIN - R_MARGIN), 0.040])
    tax.set_facecolor(BG); tax.axis('off')
    tax.set_xlim(0, max_g + 1); tax.set_ylim(0, 1)
    for j in range(max_g + 1):
        tax.text(j + 0.5, 0.3, str(j),
                 color=WHITE, ha='center', va='center', fontsize=10, fontweight='bold')
    tax.text((max_g + 1) / 2, 0.88, 'GOLES PORTUGAL',
             color=BRAND, ha='center', va='top', fontsize=8.5, fontweight='bold')

    # ── Heatmap grid ─────────────────────────────────────────────────────────
    def cell_style(p):
        if p > 0.08:   return CHIGH, '#000000', True,  13
        elif p > 0.04: return CMID,  WHITE,     True,  11
        elif p > 0.01: return CLOW,  CHIGH,     True,  9
        elif p > 0.003: return BG2,  GRAY,      False, 8
        else:           return BG,   BG2,       False, 7

    for i in range(max_g + 1):   # filas = goles México (locales/arriba)
        for j in range(max_g + 1):  # cols = goles Portugal
            p = probs[i, j]
            fc, tc, bold, fs = cell_style(p)
            cx = L_MARGIN + j * CELL_W
            cy = GRID_Y + i * CELL_H

            cax = fig.add_axes([cx, cy, CELL_W, CELL_H])
            cax.set_xlim(0, 1); cax.set_ylim(0, 1)
            cax.add_patch(mpatches.Rectangle(
                (0, 0), 1, 1, facecolor=fc, edgecolor='none', zorder=0,
                transform=cax.transAxes, clip_on=False))
            cax.axis('off')

            # Diagonal highlight (empate)
            if i == j:
                for sp in cax.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(ACC2)
                    sp.set_linewidth(1.5)

            if p >= 0.002:
                cax.text(0.5, 0.5, f'{p*100:.1f}',
                         color=tc, ha='center', va='center',
                         fontsize=fs, fontweight='bold' if bold else 'normal',
                         transform=cax.transAxes, zorder=5)

    # ── Probability summary bar ───────────────────────────────────────────────
    BAR_Y = FOOTER_H + 0.010
    BAR_H = 0.060
    bax = fig.add_axes([0.03, BAR_Y, 0.94, BAR_H])
    bax.set_facecolor(BG2); bax.axis('off')
    bax.set_xlim(0, 1); bax.set_ylim(0, 1)
    total = p_home + p_draw + p_away + 1e-9
    w_h = p_home / total
    w_d = p_draw / total
    w_a = p_away / total

    for xstart, width, color, label, pval in [
        (0,           w_h, CHIGH, 'GANA MÉXICO',   p_home),
        (w_h,         w_d, ACC2,  'EMPATE',        p_draw),
        (w_h + w_d,   w_a, RED,   'GANA PORTUGAL', p_away),
    ]:
        bax.add_patch(mpatches.Rectangle(
            (xstart + 0.003, 0.05), width - 0.006, 0.90,
            facecolor=color, edgecolor='none'))
        if width > 0.05:
            bax.text(xstart + width / 2, 0.52, f'{pval*100:.1f}%',
                     color='#000000' if color == CHIGH else WHITE,
                     ha='center', va='center', fontsize=13, fontweight='bold')
            bax.text(xstart + width / 2, 0.12, label,
                     color='#000000' if color == CHIGH else WHITE,
                     ha='center', va='center', fontsize=8)

    # ── Footer ───────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.45])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.5)
    fax.text(0.015, 0.50, 'Modelo: ELO + Poisson · Fuente: martj42/international_results',
             color=GRAY, fontsize=9.5, ha='left', va='center', transform=fax.transAxes)
    shadow = [mpe.withStroke(linewidth=3, foreground='#000000')]
    fax.text(0.985, 0.50, 'MAU-STATISTICS',
             color=BRAND, ha='right', va='center',
             transform=fax.transAxes, path_effects=shadow, **bebas(22))

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {out_path}')

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN 2: RANKING ELO TOP 20
# ─────────────────────────────────────────────────────────────────────────────
def render_ranking(elos, out_path):
    top20 = sorted(elos.items(), key=lambda x: x[1], reverse=True)[:20]
    n     = len(top20)
    FIG_W, FIG_H = 10.0, 12.0

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Gradient bg
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

    HEADER_H = 0.14
    FOOTER_H = 0.07
    CONTENT_Y = FOOTER_H
    CONTENT_H = 1 - HEADER_H - FOOTER_H
    ROW_H = CONTENT_H / n

    # Header
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=4.0)
    hax.text(0.50, 0.82, 'RANKING ELO — SELECCIONES NACIONALES',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(30))
    hax.text(0.50, 0.40, 'Top 20 · Modelo ELO histórico · K=40 · Regresión 10% anual',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=12)

    elo_max = top20[0][1]
    elo_min = min(e for _, e in top20) - 20

    HIGHLIGHT = {'Mexico', 'Portugal', 'Brazil', 'France', 'Argentina', 'Spain'}

    for i, (team, elo) in enumerate(top20):
        row_y  = CONTENT_Y + (n - 1 - i) * ROW_H
        row_bg = BG2 if i % 2 == 0 else BG
        is_hl  = team in HIGHLIGHT

        ax = fig.add_axes([0, row_y, 1, ROW_H])
        ax.set_facecolor(row_bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')

        # Acento lateral para los destacados
        if is_hl:
            color_hl = CHIGH if team == 'Mexico' else (ACC2 if team == 'Portugal' else BRAND)
            ax.add_patch(mpatches.Rectangle(
                (0, 0), 0.005, 1, facecolor=color_hl, linewidth=0))

        # Posición
        pos_color = CHIGH if i < 3 else (ACC2 if i < 8 else WHITE)
        ax.text(0.025, 0.50, f'{i+1:2d}', color=pos_color,
                fontsize=13, fontweight='bold', va='center')

        # Nombre
        ax.text(0.072, 0.50, team,
                color=CHIGH if team == 'Mexico' else (ACC2 if team == 'Portugal' else WHITE),
                fontsize=11, fontweight='bold' if is_hl else 'normal', va='center')

        # Barra
        bar_x   = 0.36
        bar_w   = 0.52
        bar_val = (elo - elo_min) / (elo_max - elo_min + 1)
        ax.add_patch(mpatches.Rectangle(
            (bar_x, 0.22), bar_w, 0.56,
            facecolor=BG2, linewidth=0))
        bar_col = CHIGH if team == 'Mexico' else (ACC2 if team == 'Portugal' else CMID)
        ax.add_patch(mpatches.Rectangle(
            (bar_x, 0.22), bar_w * bar_val, 0.56,
            facecolor=bar_col, linewidth=0))

        # ELO value
        ax.text(0.91, 0.50, f'{elo:.0f}',
                color=WHITE, fontsize=11, fontweight='bold', va='center', ha='right')

    # Footer
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.60])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.5)
    fax.text(0.015, 0.50, 'Fuente: martj42/international_results · Partidos desde 1872',
             color=GRAY, fontsize=9, ha='left', va='center', transform=fax.transAxes)
    shadow = [mpe.withStroke(linewidth=3, foreground='#000000')]
    fax.text(0.985, 0.50, 'MAU-STATISTICS',
             color=BRAND, ha='right', va='center',
             transform=fax.transAxes, path_effects=shadow, **bebas(20))

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {out_path}')

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN 3: ÚLTIMOS 5 PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────
def render_ultimos5(last_mex, last_por, out_path):
    FIG_W, FIG_H = 12.0, 8.0
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

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

    HEADER_H = 0.16
    FOOTER_H = 0.08
    CONTENT_Y = FOOTER_H
    CONTENT_H = 1 - HEADER_H - FOOTER_H
    N = 5

    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=4.0)
    hax.text(0.50, 0.85, 'FORMA RECIENTE',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(32))
    hax.text(0.50, 0.40, 'México · Portugal · Últimos 5 partidos',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=13)

    COL_W  = 0.5
    ROW_H  = CONTENT_H / N
    TEAMS  = [
        ('México', last_mex, 0.0,     CHIGH),
        ('Portugal', last_por, COL_W, ACC2),
    ]

    for team_label, matches, col_x, team_color in TEAMS:
        # Column title
        tax = fig.add_axes([col_x, CONTENT_Y + CONTENT_H, COL_W, 0.040])
        tax.set_facecolor(BG); tax.axis('off')
        tax.text(0.5, 0.5, team_label.upper(),
                 color=team_color, ha='center', va='center',
                 fontsize=14, fontweight='bold')

        for idx, m in enumerate(reversed(matches)):
            row_y  = CONTENT_Y + idx * ROW_H
            row_bg = BG2 if idx % 2 == 0 else BG

            rax = fig.add_axes([col_x, row_y, COL_W, ROW_H])
            rax.set_facecolor(row_bg); rax.axis('off')
            rax.set_xlim(0, 1); rax.set_ylim(0, 1)

            # Result badge
            res_color = CHIGH if m['res'] == 'G' else (GRAY if m['res'] == 'E' else RED)
            rax.add_patch(mpatches.FancyBboxPatch(
                (0.025, 0.18), 0.07, 0.64,
                boxstyle='round,pad=0.01',
                facecolor=res_color, linewidth=0))
            rax.text(0.06, 0.50, m['res'],
                     color='#000000' if m['res'] == 'G' else WHITE,
                     ha='center', va='center', fontsize=13, fontweight='bold')

            # Score
            rax.text(0.170, 0.55, f'{m["gf"]}-{m["ga"]}',
                     color=WHITE, fontsize=13, fontweight='bold', va='center')

            # H/A badge
            ha_color = team_color if m['ha'] == 'L' else GRAY
            rax.text(0.270, 0.35, f'({m["ha"]})',
                     color=ha_color, fontsize=8, va='center')

            # Opponent
            rax.text(0.320, 0.58, m['opponent'],
                     color=WHITE, fontsize=10, fontweight='bold', va='center')

            # Tournament
            rax.text(0.320, 0.25, m['tournament'][:30],
                     color=GRAY, fontsize=7.5, va='center')

            # Date
            rax.text(0.96, 0.50, str(m['date']),
                     color=GRAY, fontsize=8.5, ha='right', va='center')

    # Divider between columns
    div = fig.add_axes([COL_W - 0.002, CONTENT_Y, 0.004, CONTENT_H + 0.040])
    div.set_facecolor(RED); div.axis('off')

    # Footer
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.60])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.5)
    fax.text(0.015, 0.50, 'Fuente: martj42/international_results + resultados recientes México',
             color=GRAY, fontsize=9, ha='left', va='center', transform=fax.transAxes)
    shadow = [mpe.withStroke(linewidth=3, foreground='#000000')]
    fax.text(0.985, 0.50, 'MAU-STATISTICS',
             color=BRAND, ha='right', va='center',
             transform=fax.transAxes, path_effects=shadow, **bebas(20))

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
    print_hist_mex_por(df)

    # 3. Calcular ELO histórico
    print('\n[3] Calculando ELO para todas las selecciones...')
    elos = calc_elo(df)
    print(f'    {len(elos)} selecciones procesadas.')

    # 4. Aplicar partidos recientes de México
    print('\n[4] Aplicando partidos recientes de México...')
    elos = apply_extra_matches(elos, EXTRA_MEXICO)
    for fecha, home, away, gl, gv, torneo, neutral in EXTRA_MEXICO:
        res = f'{gl}-{gv}'
        print(f'    {fecha}  {home} {res} {away}  [{torneo}]')

    # 5. ELO actual México y Portugal
    elo_mex = elos.get('Mexico', ELO_BASE)
    elo_por = elos.get('Portugal', ELO_BASE)
    elo_mean = np.mean(list(elos.values()))

    print(f'\n[5] ELO actual:')
    print(f'    México    : {elo_mex:.1f}')
    print(f'    Portugal  : {elo_por:.1f}')
    print(f'    Promedio  : {elo_mean:.1f}')

    # 6. Poisson
    # Promedio de goles por partido (solo partidos con equipos con suficiente historia)
    mask = (df['home_score'].notna()) & (df['away_score'].notna())
    avg_goals = (df.loc[mask, 'home_score'].mean() + df.loc[mask, 'away_score'].mean()) / 2

    # Para amistoso = neutral, factor localía = 1.0
    LOC_FACTOR = 1.0  # partido neutral
    lam_h, lam_a = elo_to_lambda(elo_mex, elo_por, elo_mean, avg_goals, LOC_FACTOR)

    probs, p_home, p_draw, p_away = poisson_probs(lam_h, lam_a)

    print(f'\n[6] Predicción Poisson:')
    print(f'    avg_goals = {avg_goals:.3f}  lam_MÉX = {lam_h:.3f}  lam_POR = {lam_a:.3f}')
    print(f'    P(México gana) = {p_home*100:.1f}%')
    print(f'    P(Empate)      = {p_draw*100:.1f}%')
    print(f'    P(Portugal)    = {p_away*100:.1f}%')

    # Marcador más probable
    max_idx = np.unravel_index(np.argmax(probs), probs.shape)
    print(f'    Marcador más probable: México {max_idx[0]}-{max_idx[1]} Portugal  ({probs[max_idx]*100:.1f}%)')

    # 7. Heatmap
    print('\n[7] Generando heatmap...')
    render_heatmap(probs, p_home, p_draw, p_away, lam_h, lam_a,
                   elo_mex, elo_por,
                   OUT_DIR / 'selecciones_prediccion.png')

    # 8. Ranking Top 20
    print('\n[8] Generando ranking ELO Top 20...')
    render_ranking(elos, OUT_DIR / 'selecciones_ranking_elo.png')

    # 9. Últimos 5 partidos
    print('\n[9] Generando forma reciente...')
    last_mex = last_n(df, 'Mexico', 5, extra=EXTRA_MEXICO)
    last_por = last_n(df, 'Portugal', 5)
    render_ultimos5(last_mex, last_por, OUT_DIR / 'selecciones_ultimos5.png')

    print('\n' + '═' * 60)
    print('  Listo.')
    print('═' * 60 + '\n')


if __name__ == '__main__':
    main()
