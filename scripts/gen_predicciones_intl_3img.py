#!/usr/bin/env python3
"""
gen_predicciones_intl_3img.py — v2
Genera 3 imágenes por partido con todos los fixes aplicados:
  - Heatmap  : 1080×1080px (7.2×7.2 in @ 150dpi)
  - Ranking  : 1080×1350px (7.2×9.0 in @ 150dpi)
  - Últimos5 : 1080×1350px (7.2×9.0 in @ 150dpi)
  - Tipografía: Bebas Neue 64/48/36/32pt según elemento
  - Colormap: gradiente real con paleta activa
  - Paleta completa aplicada en todos los elementos

Uso:
  python gen_predicciones_intl_3img.py              # genera todos
  python gen_predicciones_intl_3img.py --only 0    # solo primer partido
"""

import sys, json, argparse, importlib, random, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox
from scipy.stats import poisson

warnings.filterwarnings('ignore')

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from config_visual import PALETAS, get_paleta, bebas, hex_rgb, get_escudo

BEBAS_TTF = Path.home() / '.fonts/BebasNeue.ttf'
if BEBAS_TTF.exists():
    try:
        fm.fontManager.addfont(str(BEBAS_TTF))
    except Exception:
        BEBAS_TTF = None

CSV_PATH = BASE / 'data/raw/internacional/results.csv'
ELO_JSON = BASE / 'data/processed/elos_selecciones_20260329.json'


# ── Partidos a generar ────────────────────────────────────────────────────────
MATCHES = [
    dict(team1='Mexico',    team2='Belgium',       fecha='2026-04-01',
         tournament='Friendly', label='México vs Bélgica  ·  Amistoso · 1 abr 2026'),
    dict(team1='Mexico',    team2='Serbia',         fecha='2026-04-06',
         tournament='Friendly', label='México vs Serbia  ·  Amistoso · 6 abr 2026'),
    dict(team1='Portugal',  team2='United States',  fecha='2026-03-31',
         tournament='Friendly', label='Portugal vs EE.UU.  ·  Amistoso · 31 mar 2026'),
    dict(team1='England',   team2='Japan',          fecha='2026-03-31',
         tournament='Friendly', label='Inglaterra vs Japón  ·  Amistoso · 31 mar 2026'),
    dict(team1='Spain',     team2='Egypt',          fecha='2026-03-31',
         tournament='Friendly', label='España vs Egipto  ·  Amistoso · 31 mar 2026'),
    dict(team1='Argentina', team2='Qatar',          fecha='2026-03-31',
         tournament='Friendly', label='Argentina vs Qatar  ·  Amistoso · 31 mar 2026'),
    dict(team1='Brazil',    team2='Croatia',        fecha='2026-04-01',
         tournament='Friendly', label='Brasil vs Croacia  ·  Amistoso · 1 abr 2026'),
]

# ── Localización ──────────────────────────────────────────────────────────────
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
    'Poland':'Polonia',         'Serbia':'Serbia',
    'Ukraine':'Ucrania',        'Turkey':'Turquía',
    'United States':'EE.UU.',   'USA':'EE.UU.',
    'Egypt':'Egipto',           'Qatar':'Qatar',
    'Ecuador':'Ecuador',        'Canada':'Canadá',
    'Austria':'Austria',        'Sweden':'Suecia',
    'Norway':'Noruega',         'Russia':'Rusia',
    'Greece':'Grecia',          'Romania':'Rumanía',
    'Iran':'Irán',              'Australia':'Australia',
    'Ghana':'Ghana',            'Nigeria':'Nigeria',
    'Costa Rica':'Costa Rica',  'Honduras':'Honduras',
    'Bosnia and Herzegovina':'Bosnia y Herz.',
    'Czech Republic':'Rep. Checa',
    'Slovakia':'Eslovaquia',    'Finland':'Finlandia',
    'Scotland':'Escocia',       'Wales':'Gales',
    'Saudi Arabia':'Arabia Saudí',
    'Ivory Coast':"Costa de Marfil",
}

MONTHS_ES = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',
             7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}

def team_es(name): return TEAM_ES.get(name, name)
def fmt_date(d):   return f"{d.day} {MONTHS_ES[d.month]} {d.year}"

# ── ELO ───────────────────────────────────────────────────────────────────────
ELO_BASE = 1500

def k_base(tournament):
    t = str(tournament).strip().lower()
    qual = 'qualifier' in t or 'qualifying' in t or 'qualification' in t
    if ('world cup' in t or 'fifa world cup' in t) and not qual: return 60
    main_35 = ['concacaf gold cup','copa america','afc asian cup',
               'africa cup of nations','confederations cup','concacaf nations league']
    is_euro_main = (('euro ' in t or t.startswith('euro') or
                     'european championship' in t) and not qual)
    if is_euro_main: return 35
    if any(x in t for x in main_35) and not qual and 'uefa nations' not in t: return 35
    if qual or 'uefa nations league' in t: return 25
    return 20

def poisson_probs(lam_h, lam_a, max_goals=5):
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            probs[i,j] = poisson.pmf(i,lam_h)*poisson.pmf(j,lam_a)
    return probs, np.sum(np.tril(probs,-1)), np.sum(np.diag(probs)), np.sum(np.triu(probs,1))

def elo_to_lambda(elo1, elo2, elo_mean, avg_goals, loc_factor=1.0):
    return avg_goals*(elo1/elo_mean)*loc_factor, avg_goals*(elo2/elo_mean)

def last_n(df, team, n=5):
    mask = (df['home_team']==team)|(df['away_team']==team)
    rows = df[mask].dropna(subset=['home_score','away_score']).sort_values('date').tail(n)
    result = []
    for _,r in rows.iterrows():
        ih  = (r['home_team']==team)
        opp = r['away_team'] if ih else r['home_team']
        gf  = int(r['home_score'] if ih else r['away_score'])
        ga  = int(r['away_score'] if ih else r['home_score'])
        result.append({'date':r['date'].date(),'opponent':opp,'gf':gf,'ga':ga,
                       'ha':'L' if ih else 'V',
                       'res':'G' if gf>ga else ('E' if gf==ga else 'P'),
                       'tournament':r['tournament']})
    return result

# ── Helpers visualización ─────────────────────────────────────────────────────
def _bg_gradient(fig, BG, BG2):
    bg_rgb, bg2_rgb = hex_rgb(BG), hex_rgb(BG2)
    grad = np.zeros((200,2,3))
    for ii in range(200):
        t = ii/199
        grad[ii] = np.array(bg_rgb)/255*(1-t)+np.array(bg2_rgb)/255*t
    bgax = fig.add_axes([0,0,1,1])
    bgax.set_zorder(-100)
    bgax.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bgax.axis('off')

def _footer(fig, source_text, BG, BG2, RED, GRAY, BRAND, h=0.042, fact=None):
    if fact:
        fct = fig.add_axes([0, h, 1, 0.030])
        fct.set_facecolor(BG); fct.axis('off')
        fct.text(0.5, 0.5, fact, color=GRAY, fontsize=11, alpha=0.7,
                 ha='center', va='center', style='italic', transform=fct.transAxes)
    fax = fig.add_axes([0, 0, 1, h*0.60])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.5)
    fax.text(0.015, 0.50, source_text, color=GRAY, fontsize=8,
             ha='left', va='center', transform=fax.transAxes)
    shadow = [mpe.withStroke(linewidth=3, foreground='#000000')]
    fax.text(0.985, 0.50, 'MAU-STATISTICS', color=BRAND, ha='right', va='center',
             transform=fax.transAxes, path_effects=shadow, **bebas(16))

def _make_cmap(BG, BG2, CMID, CHIGH):
    """Colormap paleta: negro → BG → BG2 → CMID → CHIGH"""
    return LinearSegmentedColormap.from_list(
        'pal_custom', ['#000000', BG, BG2, CMID, CHIGH], N=256)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGEN 1: HEATMAP POISSON  1080×1080px  (v3 — todos los fixes)
# ─────────────────────────────────────────────────────────────────────────────
def _bootstrap_ci(lam_h, lam_a, n_sim=10_000, n_boot=1_000):
    """Bootstrap Monte Carlo IC 95% para P(local), P(empate), P(visitante)."""
    rng = np.random.default_rng(42)
    hg  = rng.poisson(lam_h, n_sim)
    ag  = rng.poisson(lam_a, n_sim)
    wins, draws, loses = hg > ag, hg == ag, hg < ag
    bh, bd, ba = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_sim, n_sim)
        bh.append(wins[idx].mean());  bd.append(draws[idx].mean())
        ba.append(loses[idx].mean())
    ci_h = (np.percentile(bh, 2.5), np.percentile(bh, 97.5))
    ci_d = (np.percentile(bd, 2.5), np.percentile(bd, 97.5))
    ci_a = (np.percentile(ba, 2.5), np.percentile(ba, 97.5))
    return ci_h, ci_d, ci_a


def render_heatmap(probs, p_home, p_draw, p_away, lam_h, lam_a,
                   elo1, elo2, team1, team2, label, out_path, pal):
    # ── Paleta ────────────────────────────────────────────────────────────────
    BG    = pal['bg_primary']
    BG2   = pal['bg_secondary']
    WHITE = pal['text_primary']
    GRAY  = pal['text_secondary']
    RED   = pal['accent']
    CHIGH = pal['cell_high']
    BRAND = pal['brand_color']

    t1_es = team_es(team1).upper()
    t2_es = team_es(team2).upper()

    ci_h, ci_d, ci_a = _bootstrap_ci(lam_h, lam_a)
    N = 6

    # Colormap: pal['fondo'] → pal['acento']
    cmap     = LinearSegmentedColormap.from_list('pal_hm', [BG, RED], N=256)
    prob_max = float(probs.max())
    max_idx  = np.unravel_index(np.argmax(probs), probs.shape)
    score_str = f'{max_idx[0]}-{max_idx[1]}'

    # ── Layout 7.2×7.2 @ 150dpi = 1080×1080px ────────────────────────────────
    FIG_W = FIG_H = 7.2
    HEADER_H  = 0.150   # max 15%
    FOOTER_H  = 0.042
    FACT_H    = 0.025
    PROB_H    = 0.145
    COL_HDR_H = 0.028
    ROWHDR_W  = 0.090
    R_MARGIN  = 0.010
    GAP       = 0.006

    GRID_Y    = FOOTER_H + FACT_H + PROB_H + GAP    # 0.218
    HEADER_Y  = 1.0 - HEADER_H                       # 0.850
    COL_HDR_Y = HEADER_Y - GAP - COL_HDR_H           # 0.816
    GRID_H    = COL_HDR_Y - GAP - GRID_Y             # 0.592
    CELL_H    = GRID_H / N
    CELL_W    = (1.0 - ROWHDR_W - R_MARGIN) / N

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _bg_gradient(fig, BG, BG2)

    # ── Header compacto (max 15%) ──────────────────────────────────────────────
    hax = fig.add_axes([0, HEADER_Y, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=CHIGH, lw=2.0)

    # Título 36pt
    hax.text(0.50, 0.98, '¿QUIÉN GANA?',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(36))
    # Score secundario (marcador más probable) 32pt
    hax.text(0.50, 0.64, score_str,
             color=CHIGH, ha='center', va='top', transform=hax.transAxes, **bebas(32))

    # Team line — una sola línea: [flag] NOMBRE  ELO  λ | VS | NOMBRE [flag]  ELO  λ
    fi1 = get_escudo(team1, size=(40, 27))
    if fi1 is not None:
        hax.add_artist(AnnotationBbox(fi1, (0.04, 0.22),
                       frameon=False, xycoords='axes fraction', box_alignment=(0.5, 0.5)))
    hax.text(0.08, 0.22, f'{t1_es}  ELO: {elo1:.0f}  λ={lam_h:.2f}',
             color=CHIGH, ha='left', va='center', fontsize=11,
             fontweight='bold', transform=hax.transAxes)

    hax.text(0.50, 0.22, 'VS', color=CHIGH, ha='center', va='center',
             transform=hax.transAxes, **bebas(20))

    hax.text(0.92, 0.22, f'{t2_es}  ELO: {elo2:.0f}  λ={lam_a:.2f}',
             color=CHIGH, ha='right', va='center', fontsize=11,
             fontweight='bold', transform=hax.transAxes)
    fi2 = get_escudo(team2, size=(40, 27))
    if fi2 is not None:
        hax.add_artist(AnnotationBbox(fi2, (0.96, 0.22),
                       frameon=False, xycoords='axes fraction', box_alignment=(0.5, 0.5)))

    # Subtítulo 11pt
    hax.text(0.50, 0.03, label,
             color=GRAY, ha='center', va='bottom', fontsize=11, transform=hax.transAxes)

    # ── Cabecera de columnas: "GOLES t2" + números 0-5 ───────────────────────
    lbl_ax = fig.add_axes([ROWHDR_W, COL_HDR_Y, N*CELL_W, COL_HDR_H])
    lbl_ax.set_facecolor(BG2); lbl_ax.axis('off')
    lbl_ax.text(0.5, 0.78, f'GOLES  {t2_es}',
                color=RED, ha='center', va='center', fontsize=9,
                fontweight='bold', transform=lbl_ax.transAxes)
    for j in range(N):
        lbl_ax.text((j + 0.5) / N, 0.25, str(j),
                    color=WHITE, ha='center', va='center',
                    fontsize=10, fontweight='bold', transform=lbl_ax.transAxes)

    # ── Cabecera de filas: "GOLES t1" rotado + números ───────────────────────
    rmx = fig.add_axes([0, GRID_Y, ROWHDR_W * 0.40, GRID_H])
    rmx.set_facecolor(BG2); rmx.axis('off')
    rmx.text(0.5, 0.5, f'GOLES  {t1_es}',
             color=RED, ha='center', va='center', fontsize=7, fontweight='bold',
             rotation=90, transform=rmx.transAxes)
    for i in range(N):
        nax = fig.add_axes([ROWHDR_W*0.40, GRID_Y+i*CELL_H, ROWHDR_W*0.60, CELL_H])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.5, 0.5, str(i), color=WHITE, ha='center', va='center',
                 fontsize=11, fontweight='bold', transform=nax.transAxes)

    # ── Celdas simples Rectangle + colormap BG→accent ─────────────────────────
    grid_ax = fig.add_axes([ROWHDR_W, GRID_Y, N*CELL_W, GRID_H])
    grid_ax.set_facecolor(BG)
    grid_ax.set_xlim(0, N); grid_ax.set_ylim(0, N)
    grid_ax.axis('off')

    for i in range(N):
        for j in range(N):
            prob   = probs[i, j]
            p_norm = (prob / prob_max) if prob_max > 0 else 0.0
            color  = cmap(p_norm)

            # Relleno simple, sin borde
            grid_ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1, facecolor=color, edgecolor='none', linewidth=0, zorder=1))

            # Diagonal: borde blanco delgado 1px (escenarios de empate)
            if i == j:
                grid_ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor='none', edgecolor=WHITE,
                    linewidth=1, zorder=3))

            # Texto
            if prob >= 0.003:
                txt = f'{prob*100:.1f}' if prob >= 0.010 else f'{prob*100:.2f}'
                txt_color = WHITE if p_norm > 0.45 else GRAY
                fs = max(13, int(9 + p_norm * 6))
                grid_ax.text(j+0.5, i+0.5, txt, color=txt_color,
                             ha='center', va='center', fontsize=fs, fontweight='bold',
                             zorder=5)

    # ── Bloques de resultado ───────────────────────────────────────────────────
    prob_y  = FOOTER_H + FACT_H + GAP * 0.5
    BLK_GAP = 0.012
    BLK_W   = (1.0 - 4*BLK_GAP) / 3
    max_p   = max(p_home, p_draw, p_away)
    cis     = [ci_h, ci_d, ci_a]
    result_blocks = [
        (team1, f'GANA {t1_es}', p_home),
        (None,  'EMPATE',        p_draw),
        (team2, f'GANA {t2_es}', p_away),
    ]
    for k, ((tk, blk_label, pval), ci) in enumerate(zip(result_blocks, cis)):
        bx     = BLK_GAP + k*(BLK_W+BLK_GAP)
        is_max = (pval >= max_p - 1e-9)
        pct_c  = CHIGH if is_max else WHITE
        pct_fs = 56 if is_max else 42
        lbl_c  = WHITE

        seg = fig.add_axes([bx, prob_y, BLK_W, PROB_H])
        seg.set_facecolor(BG2); seg.axis('off')
        seg.set_xlim(0, 1); seg.set_ylim(0, 1)
        if is_max:
            for sp in seg.spines.values():
                sp.set_visible(True); sp.set_edgecolor(CHIGH); sp.set_linewidth(2.5)

        if tk:
            fi_blk = get_escudo(tk, size=(28, 19))
            if fi_blk is not None:
                seg.add_artist(AnnotationBbox(fi_blk, (0.35, 0.90),
                               frameon=False, xycoords='axes fraction', box_alignment=(1.0, 0.5)))
                seg.text(0.38, 0.90, blk_label, color=lbl_c, ha='left', va='center',
                         fontsize=14, fontweight='bold', transform=seg.transAxes)
            else:
                seg.text(0.50, 0.90, blk_label, color=lbl_c, ha='center', va='center',
                         fontsize=14, fontweight='bold', transform=seg.transAxes)
        else:
            seg.text(0.50, 0.90, blk_label, color=lbl_c, ha='center', va='center',
                     fontsize=14, fontweight='bold', transform=seg.transAxes)

        seg.text(0.50, 0.52, f'{pval*100:.1f}%', color=pct_c, ha='center', va='center',
                 transform=seg.transAxes, zorder=5, **bebas(pct_fs))

        ci_txt = f'IC 95%: [{ci[0]*100:.1f}% — {ci[1]*100:.1f}%]'
        seg.text(0.50, 0.12, ci_txt, color=GRAY, ha='center', va='center',
                 fontsize=9, alpha=0.6, transform=seg.transAxes)

    # ── Footer con nota diagonal ───────────────────────────────────────────────
    best_p = probs[max_idx] * 100
    _footer(fig, 'Modelo: ELO + Poisson · martj42/international_results',
            BG, BG2, RED, GRAY, BRAND, h=FOOTER_H,
            fact=(f'Diagonal = escenarios de empate  ·  '
                  f'Marcador más probable: {team_es(team1)} {max_idx[0]}-{max_idx[1]}'
                  f' {team_es(team2)} ({best_p:.1f}%)'))

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  ✓ heatmap   → {out_path.name}')
    return max_idx, best_p


# ─────────────────────────────────────────────────────────────────────────────
# IMAGEN 2: RANKING ELO  1080×1350px
# ─────────────────────────────────────────────────────────────────────────────
def render_ranking(elos, team1, team2, out_path, pal, fact=''):
    BG=pal['bg_primary']; BG2=pal['bg_secondary']; WHITE=pal['text_primary']
    GRAY=pal['text_secondary']; RED=pal['accent']; ACC2=pal['accent2']
    CHIGH=pal['cell_high']; BRAND=pal['brand_color']

    sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    all_ranks   = {t:(i+1) for i,(t,_) in enumerate(sorted_elos)}

    display_set = list(sorted_elos[:20])
    extras = []
    for tm in (team1, team2):
        if tm not in {t for t,_ in display_set}:
            extras.append((tm, elos.get(tm, ELO_BASE)))
    display_set_full = display_set + extras
    n = len(display_set_full)

    # 1080×1350 = 7.2×9.0 in @ 150dpi
    FIG_W, FIG_H = 7.2, 9.0

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _bg_gradient(fig, BG, BG2)

    HEADER_H  = 0.185
    FOOTER_H  = 0.042
    FACT_H    = 0.030
    CONTENT_Y = FOOTER_H + FACT_H
    CONTENT_H = 1 - HEADER_H - FOOTER_H - FACT_H
    ROW_H     = CONTENT_H / n

    # Header — FIX 5: bebas 48pt
    hax = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=3.5)
    hax.text(0.50, 0.97, 'RANKING ELO · TOP 20 MUNDIAL',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(48))
    hax.text(0.50, 0.64,
             'Metodología eloratings.net · K: Amistoso=20 · Clasif.=25 · Continental=35 · Mundial=60',
             color='#dddddd', ha='center', va='top', transform=hax.transAxes, fontsize=7.5)
    hax.text(0.50, 0.50,
             f'Destacados: {team_es(team1).upper()} vs {team_es(team2).upper()}',
             color=CHIGH, ha='center', va='top', transform=hax.transAxes, fontsize=8.5)

    # Leyenda K
    K_LEGEND = [('#FFD700','K=20 Amistoso'),('#FFA040','K=25 Clasif.'),
                ('#FF7070','K=35 Cont.'),('#AA1122','K=60 Mundial')]
    for (lc,lbl), lx in zip(K_LEGEND, [0.03,0.27,0.53,0.75]):
        hax.add_patch(mpatches.Rectangle(
            (lx, 0.18), 0.018, 0.18, facecolor=lc, edgecolor='none',
            transform=hax.transAxes, clip_on=False, zorder=5))
        hax.text(lx+0.025, 0.27, lbl, color='#dddddd',
                 ha='left', va='center', fontsize=7.5, transform=hax.transAxes)

    elo_max = display_set_full[0][1]
    elo_min = min(e for _,e in display_set_full) - 30

    ROW_PTS_W  = FIG_W * 72        # 518.4 pt
    ROW_PTS_H  = ROW_H * FIG_H * 72
    NUM_RIGHT  = 30;  FLAG_CX = 48;  NAME_X = 65
    BAR_X      = 235; BAR_W   = ROW_PTS_W - BAR_X - 65;  ELO_X = ROW_PTS_W - 8
    GOLD_STRIP = 4

    for i, (team, elo) in enumerate(display_set_full):
        rank_num = all_ranks.get(team, '?')
        row_y    = CONTENT_Y + (n-1-i) * ROW_H
        is_t1    = (team == team1)
        is_t2    = (team == team2)
        is_ft    = is_t1 or is_t2

        # Separador si equipo extra fuera del top 20
        if i == 20:
            sep = fig.add_axes([0.05, row_y+ROW_H*1.02, 0.90, ROW_H*0.25])
            sep.set_facecolor(BG); sep.axis('off')
            sep.text(0.5, 0.5, '· · · · ·', color=GRAY,
                     ha='center', va='center', fontsize=8, transform=sep.transAxes)

        if is_t1:
            row_bg = '#0a0d1a'; strip_c = CHIGH; name_c = CHIGH
            bar_fc = CHIGH; bar_sh = '#004433'; bar_hi = '#80ffdd'
        elif is_t2:
            row_bg = '#0d0a1a'; strip_c = ACC2;  name_c = ACC2
            bar_fc = ACC2;  bar_sh = '#3a0055'; bar_hi = '#cc88ff'
        else:
            row_bg = BG2 if i%2==0 else BG
            strip_c = None; name_c = WHITE
            bar_fc = '#C8102E'; bar_sh = '#9B0C23'; bar_hi = '#E8384F'

        ax = fig.add_axes([0, row_y, 1, ROW_H])
        ax.set_facecolor(row_bg)
        ax.set_xlim(0, ROW_PTS_W); ax.set_ylim(0, ROW_PTS_H); ax.axis('off')
        MID = ROW_PTS_H/2

        if is_ft:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(strip_c); sp.set_linewidth(1.2)
            ax.add_patch(mpatches.Rectangle(
                (0, ROW_PTS_H*0.08), GOLD_STRIP, ROW_PTS_H*0.84,
                facecolor=strip_c, linewidth=0))

        ax.text(NUM_RIGHT, MID, f'{rank_num}',
                color=strip_c if is_ft else (ACC2 if i<3 else WHITE),
                fontsize=8, fontweight='bold', va='center', ha='right')

        flag_im = get_escudo(team, size=(32,22))
        if flag_im is not None:
            ax.add_artist(AnnotationBbox(flag_im, (FLAG_CX, MID),
                          frameon=False, xycoords='data', box_alignment=(0.5,0.5)))

        ax.text(NAME_X, MID, team_es(team),
                color=name_c, fontsize=8,
                fontweight='bold' if is_ft else 'normal', va='center', ha='left')

        bar_val = max(0, (elo-elo_min)/(elo_max-elo_min+1))
        bar_h   = ROW_PTS_H*0.56; bar_y = MID-bar_h/2; bw = BAR_W*bar_val

        ax.add_patch(mpatches.FancyBboxPatch(
            (BAR_X, bar_y), BAR_W, bar_h, boxstyle='round,pad=1',
            facecolor=BG2, linewidth=0, zorder=1))
        if bw>2:
            ax.add_patch(mpatches.FancyBboxPatch(
                (BAR_X, bar_y), bw, bar_h, boxstyle='round,pad=1',
                facecolor=bar_sh, linewidth=0, zorder=2))
            ax.add_patch(mpatches.Rectangle(
                (BAR_X, bar_y), bw, bar_h, facecolor=bar_fc, linewidth=0, zorder=3))
            ax.add_patch(mpatches.Rectangle(
                (BAR_X, bar_y+bar_h*0.70), bw, bar_h*0.30,
                facecolor=bar_hi, linewidth=0, zorder=4, alpha=0.50))

        ax.text(ELO_X, MID, f'{elo:.0f}',
                color=strip_c if is_ft else WHITE,
                fontsize=8, fontweight='bold' if is_ft else 'normal',
                va='center', ha='right')
        ax.axhline(0, color=BG, lw=0.5)

    rank1 = all_ranks.get(team1,'?'); elo1v = elos.get(team1, ELO_BASE)
    rank2 = all_ranks.get(team2,'?'); elo2v = elos.get(team2, ELO_BASE)
    _footer(fig, 'Fuente: martj42/international_results · ~49,000 partidos desde 1872',
            BG, BG2, RED, GRAY, BRAND, h=FOOTER_H,
            fact=fact or f'{team_es(team1)} #{rank1} ({elo1v:.0f} pts)  ·  '
                         f'{team_es(team2)} #{rank2} ({elo2v:.0f} pts)')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  ✓ ranking   → {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# IMAGEN 3: ÚLTIMOS 5 PARTIDOS  1080×1350px
# ─────────────────────────────────────────────────────────────────────────────
def render_ultimos5(last_t1, last_t2, team1, team2, out_path, pal, fact=''):
    BG=pal['bg_primary']; BG2=pal['bg_secondary']; WHITE=pal['text_primary']
    GRAY=pal['text_secondary']; ACC=pal['accent']; ACC2=pal['accent2']
    BRAND=pal['brand_color']

    # 1080×1350 = 7.2×9.0 in @ 150dpi
    FIG_W, FIG_H = 7.2, 9.0
    N = 5

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    r1,g1,b1 = hex_rgb(BG); r2,g2,b2 = hex_rgb(BG2)
    grad = np.zeros((256,1,4))
    for i in range(256):
        t = i/255
        grad[i,0] = [r1/255*(1-t)+r2/255*t, g1/255*(1-t)+g2/255*t,
                     b1/255*(1-t)+b2/255*t, 1.0]
    bg_ax = fig.add_axes([0,0,1,1], zorder=-10)
    bg_ax.imshow(grad, aspect='auto', extent=[0,1,0,1],
                 origin='lower', transform=fig.transFigure, zorder=-10)
    bg_ax.axis('off')

    HEADER_H = 0.148
    FOOTER_H = 0.042
    FACT_H   = 0.028
    TITLE_H  = 0.055
    GAP_COL  = 0.006
    CONTENT_Y = FOOTER_H + FACT_H
    CONTENT_H = 1.0 - HEADER_H - FOOTER_H - FACT_H - TITLE_H - 0.006
    COL_W     = (1.0 - GAP_COL) / 2.0

    STRIP_W = 5.0 / (COL_W * FIG_W * 150.0)

    # FIX 5: Header bebas 48pt, subtítulo 20pt
    hax = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=ACC, lw=3.5)
    hax.text(0.50, 0.97, 'ASÍ LLEGAN AL PARTIDO',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(48))
    hax.text(0.50, 0.50,
             f'{team_es(team1)} vs {team_es(team2)}  ·  Últimos 5 partidos',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, fontsize=20)

    TEAMS_DATA = [
        (team1, last_t1, 0.0,           ACC),
        (team2, last_t2, COL_W+GAP_COL, ACC2),
    ]

    for tm, matches, col_x, team_color in TEAMS_DATA:
        name_es_upper = team_es(tm).upper()
        col_top_y = CONTENT_Y + CONTENT_H

        # FIX 5: team header bebas 32pt
        tax = fig.add_axes([col_x, col_top_y, COL_W, TITLE_H])
        tax.set_facecolor(BG2); tax.axis('off')
        tax.set_xlim(0,1); tax.set_ylim(0,1)
        flag_title = get_escudo(tm, size=(48,32))
        if flag_title is not None:
            tax.add_artist(AnnotationBbox(flag_title, (0.35,0.50),
                           frameon=False, xycoords='axes fraction', box_alignment=(1.0,0.5)))
        tax.text(0.38, 0.50, name_es_upper, color=team_color,
                 ha='left', va='center', transform=tax.transAxes, **bebas(32))

        ax = fig.add_axes([col_x, CONTENT_Y, COL_W, CONTENT_H])
        ax.set_facecolor(BG); ax.axis('off')
        ax.set_xlim(0,1); ax.set_ylim(0,N)

        padded = list(reversed(matches))
        while len(padded) < N:
            padded.append(None)

        for idx in range(N):
            m = padded[idx]
            if m is None:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.016, idx+0.060), 0.968, 0.878,
                    boxstyle='round,pad=0.012',
                    facecolor=BG2, edgecolor='none', linewidth=0, zorder=1, alpha=0.4))
                ax.text(0.500, idx+0.500, '— sin datos —',
                        color=GRAY, ha='center', va='center', fontsize=9, zorder=3)
                continue

            win  = m['res']=='G'; draw = m['res']=='E'

            ax.add_patch(mpatches.FancyBboxPatch(
                (0.016, idx+0.060), 0.968, 0.878,
                boxstyle='round,pad=0.012',
                facecolor=BG2, edgecolor='none', linewidth=0, zorder=1))

            if win:    strip_c, strip_a = team_color, 1.0
            elif draw: strip_c, strip_a = team_color, 0.45
            else:      strip_c, strip_a = '#333333',  1.0
            ax.add_patch(mpatches.Rectangle(
                (0.016, idx+0.095), STRIP_W, 0.808,
                facecolor=strip_c, alpha=strip_a, linewidth=0, zorder=2))

            ax.text(0.500, idx+0.640, f'{m["gf"]}-{m["ga"]}',
                    color=WHITE, ha='center', va='center',
                    fontsize=29, zorder=3, **bebas(29))

            opp_flag = get_escudo(m['opponent'], size=(44,29))
            opp_es   = team_es(m['opponent'])
            if len(opp_es)>15: opp_es = opp_es[:14]+'.'
            FLAG_W = 0.044
            if opp_flag is not None:
                flag_x = 0.500 - FLAG_W*0.5 - 0.010
                ax.add_artist(AnnotationBbox(opp_flag, (flag_x, idx+0.375),
                              frameon=False, xycoords='data',
                              box_alignment=(0.5,0.5), zorder=4))
                ax.text(0.500+FLAG_W*0.5+0.006, idx+0.375, opp_es,
                        color=WHITE, ha='left', va='center',
                        fontsize=10, fontweight='bold', zorder=3)
            else:
                ax.text(0.500, idx+0.375, opp_es,
                        color=WHITE, ha='center', va='center',
                        fontsize=10, fontweight='bold', zorder=3)

            ax.text(0.500, idx+0.198, m['tournament'][:30],
                    color=ACC, ha='center', va='center', fontsize=8, zorder=3)
            ax.text(0.970, idx+0.845, fmt_date(m['date']),
                    color=GRAY, ha='right', va='center', fontsize=7.5, zorder=3)
            ha_lbl = 'Local' if m['ha']=='L' else 'Visita'
            ax.text(0.970, idx+0.695, ha_lbl,
                    color=ACC, ha='right', va='center', fontsize=7.0, fontweight='bold', zorder=3)

    div = fig.add_axes([COL_W, CONTENT_Y, GAP_COL, CONTENT_H+TITLE_H])
    div.set_facecolor(ACC); div.axis('off')

    _footer(fig, 'Fuente: martj42/international_results + resultados recientes',
            BG, BG2, ACC, GRAY, BRAND, h=FOOTER_H, fact=fact)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  ✓ últimos5  → {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=int, default=-1)
    args = parser.parse_args()

    with open(ELO_JSON) as f:
        elos = json.load(f)

    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    mask = df['home_score'].notna() & df['away_score'].notna()
    avg_goals = (df.loc[mask,'home_score'].mean()+df.loc[mask,'away_score'].mean())/2
    elo_mean  = np.mean(list(elos.values()))

    tracker  = importlib.import_module('04_predicciones_tracker')
    pal_keys = list(PALETAS.keys())
    last_pal = None

    matches_to_run = MATCHES if args.only==-1 else [MATCHES[args.only]]

    for m in matches_to_run:
        team1, team2 = m['team1'], m['team2']
        fecha, label = m['fecha'], m['label']

        print(f'\n{"═"*60}')
        print(f'  {label}')
        print(f'{"═"*60}')

        available = [k for k in pal_keys if k != last_pal]
        pal_key   = random.choice(available)
        last_pal  = pal_key
        pal       = PALETAS[pal_key]

        # FIX 6: imprimir colores exactos de la paleta
        print(f'  🎨  Paleta: {pal_key}')
        print(f'      bg_primary   = {pal["bg_primary"]}')
        print(f'      bg_secondary = {pal["bg_secondary"]}')
        print(f'      cell_high    = {pal["cell_high"]}')
        print(f'      cell_mid     = {pal["cell_mid"]}')
        print(f'      accent       = {pal["accent"]}')
        print(f'      accent2      = {pal["accent2"]}')
        print(f'      brand_color  = {pal["brand_color"]}')

        elo1 = elos.get(team1, ELO_BASE)
        elo2 = elos.get(team2, ELO_BASE)

        slug1 = team1.replace(' ','_'); slug2 = team2.replace(' ','_')
        out_dir = BASE/'output/charts/predicciones'/f'{slug1}_vs_{slug2}'
        out_dir.mkdir(parents=True, exist_ok=True)

        lam_h, lam_a = elo_to_lambda(elo1, elo2, elo_mean, avg_goals, 1.0)
        probs, p_home, p_draw, p_away = poisson_probs(lam_h, lam_a, max_goals=5)
        max_idx = np.unravel_index(np.argmax(probs), probs.shape)
        best_p  = probs[max_idx]*100

        last_t1 = last_n(df, team1, 5)
        last_t2 = last_n(df, team2, 5)

        sorted_r = sorted(elos.items(), key=lambda x: x[1], reverse=True)
        rank1_num = next((i+1 for i,(t,_) in enumerate(sorted_r) if t==team1), '?')
        rank2_num = next((i+1 for i,(t,_) in enumerate(sorted_r) if t==team2), '?')

        fact_rk = (f'{team_es(team1)} #{rank1_num} ({elo1:.0f} pts)  ·  '
                   f'{team_es(team2)} #{rank2_num} ({elo2:.0f} pts)')
        leader  = team_es(team1) if elo1>elo2 else team_es(team2)
        fact_u5 = (f'{leader} llega con mayor ELO ({max(elo1,elo2):.0f} pts)  ·  '
                   f'Diferencia: {abs(elo1-elo2):.0f} pts')

        render_ranking(elos, team1, team2,
                       out_dir/f'{slug1}_vs_{slug2}_ranking.png', pal, fact=fact_rk)
        render_heatmap(probs, p_home, p_draw, p_away, lam_h, lam_a,
                       elo1, elo2, team1, team2, label,
                       out_dir/f'{slug1}_vs_{slug2}_heatmap.png', pal)
        render_ultimos5(last_t1, last_t2, team1, team2,
                        out_dir/f'{slug1}_vs_{slug2}_ultimos5.png', pal, fact=fact_u5)

        tracker.registrar_prediccion(
            equipo_local=team1, equipo_visitante=team2,
            elo_local=elo1, elo_visitante=elo2,
            prob_local=p_home, prob_empate=p_draw, prob_visitante=p_away,
            marcador_probable=f'{max_idx[0]}-{max_idx[1]}',
            lambda_local=lam_h, lambda_visitante=lam_a,
            fecha_partido=fecha, paleta=pal_key,
        )
        print(f'  P(local)={p_home*100:.1f}%  P(empate)={p_draw*100:.1f}%  P(visita)={p_away*100:.1f}%')
        print(f'  Más probable: {team_es(team1)} {max_idx[0]}-{max_idx[1]} {team_es(team2)} ({best_p:.1f}%)')

    print('\nListo.')


if __name__ == '__main__':
    main()
