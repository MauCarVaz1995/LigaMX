#!/usr/bin/env python3
"""
19_predicciones_hoy.py
Genera imágenes de predicción 1080×1080 para partidos del día.
Paleta aleatoria por partido (sin repetir consecutivas).

Uso:
  python 19_predicciones_hoy.py               # genera todos
  python 19_predicciones_hoy.py --match COL   # solo partidos que contengan 'COL'
"""

import argparse, importlib, random, warnings, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnnotationBbox
from scipy.stats import poisson
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETAS, bebas, hex_rgb, get_escudo

# ─────────────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE / 'output/charts/predicciones_hoy'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fuente centralizada desde config_visual (incluye assets/fonts/ como candidato)
try:
    from config_visual import BEBAS_TTF, BEBAS_AVAILABLE
except ImportError:
    BEBAS_TTF = None
    BEBAS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# NOMBRES EN ESPAÑOL
# ─────────────────────────────────────────────────────────────────────────────
TEAM_ES = {
    'Colombia': 'Colombia',          'France': 'Francia',
    'Aruba': 'Aruba',                'Liechtenstein': 'Liechtenstein',
    'Barbados': 'Barbados',          'Saint Vincent and the Grenadines': 'San Vicente',
    'Virgin Islands': 'Islas Vírgenes', 'Anguilla': 'Anguila',
    'Dominican Republic': 'Rep. Dominicana', 'Cuba': 'Cuba',
    'Cayman Islands': 'Islas Caimán', 'Bahamas': 'Bahamas',
    'Cape Verde': 'Cabo Verde',      'Finland': 'Finlandia',
    'Mexico': 'México',              'Portugal': 'Portugal',
    'Brazil': 'Brasil',              'Spain': 'España',
    'England': 'Inglaterra',         'Germany': 'Alemania',
    'Argentina': 'Argentina',        'Italy': 'Italia',
    'Netherlands': 'Países Bajos',   'Croatia': 'Croacia',
    'Japan': 'Japón',                'Switzerland': 'Suiza',
    'Denmark': 'Dinamarca',          'Uruguay': 'Uruguay',
    'Ecuador': 'Ecuador',            'Senegal': 'Senegal',
    'Morocco': 'Marruecos',          'Belgium': 'Bélgica',
}

def team_es(name): return TEAM_ES.get(name, name)

# ─────────────────────────────────────────────────────────────────────────────
# PARTIDOS DEL DÍA  (ELOs fijos, amistosos en cancha neutral)
# ─────────────────────────────────────────────────────────────────────────────
MATCHES = [
    dict(team1='Colombia',                              team2='France',
         elo1=2027,  elo2=2080,
         file='prediccion_COL_FRA.png',
         label='Colombia vs Francia  ·  Amistoso · 29 mar 2026',
         fecha='2026-03-29'),
    dict(team1='Aruba',                                 team2='Liechtenstein',
         elo1=1222,  elo2=1032,
         file='prediccion_ARU_LIE.png',
         label='Aruba vs Liechtenstein  ·  Amistoso · 29 mar 2026',
         fecha='2026-03-29'),
    dict(team1='Barbados',                              team2='Saint Vincent and the Grenadines',
         elo1=1199,  elo2=1403,
         file='prediccion_BAR_SVI.png',
         label='Barbados vs San Vicente  ·  Amistoso · 29 mar 2026',
         fecha='2026-03-29'),
    dict(team1='Virgin Islands',                        team2='Anguilla',
         elo1=1100,  elo2=976,
         file='prediccion_VIR_ANG.png',
         label='Islas Vírgenes vs Anguila  ·  Amistoso · 29 mar 2026',
         fecha='2026-03-29'),
    dict(team1='Dominican Republic',                    team2='Cuba',
         elo1=1494,  elo2=1430,
         file='prediccion_DOM_CUB.png',
         label='Rep. Dominicana vs Cuba  ·  Amistoso · 29 mar 2026',
         fecha='2026-03-29'),
    dict(team1='Cayman Islands',                        team2='Bahamas',
         elo1=1115,  elo2=1086,
         file='prediccion_CAY_BAH.png',
         label='Islas Caimán vs Bahamas  ·  Amistoso · 29 mar 2026',
         fecha='2026-03-29'),
    dict(team1='Cape Verde',                            team2='Finland',
         elo1=1638,  elo2=1576,
         file='prediccion_CPV_FIN.png',
         label='Cabo Verde vs Finlandia  ·  Amistoso · 29 mar 2026',
         fecha='2026-03-29'),
]

# ─────────────────────────────────────────────────────────────────────────────
# MODELO POISSON
# ─────────────────────────────────────────────────────────────────────────────
SCALE    = 400
AVG_GOALS_GLOBAL = 1.35   # promedio histórico internacional por equipo por partido

def poisson_probs(lam_h, lam_a, max_goals=5):
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            probs[i, j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    return probs, np.sum(np.tril(probs,-1)), np.sum(np.diag(probs)), np.sum(np.triu(probs,1))

def elo_to_lambda(elo1, elo2, all_elos):
    """Neutral: loc_factor=1.0 para ambos equipos."""
    elo_mean = np.mean(all_elos)
    lam1 = AVG_GOALS_GLOBAL * (elo1 / elo_mean)
    lam2 = AVG_GOALS_GLOBAL * (elo2 / elo_mean)
    return lam1, lam2

# ELO de todos los equipos de hoy para calcular la media contextual
ALL_ELOS_TODAY = [m['elo1'] for m in MATCHES] + [m['elo2'] for m in MATCHES]

# ─────────────────────────────────────────────────────────────────────────────
# RENDER — 1080×1080 (7.2×7.2 in @ 150 dpi)
# ─────────────────────────────────────────────────────────────────────────────
GOLD = '#FFD700'

def render_prediccion(m, pal):
    team1  = m['team1'];  team2  = m['team2']
    elo1   = m['elo1'];   elo2   = m['elo2']
    label  = m['label'];  out_path = OUT_DIR / m['file']

    lam1, lam2 = elo_to_lambda(elo1, elo2, ALL_ELOS_TODAY)
    probs, p_home, p_draw, p_away = poisson_probs(lam1, lam2, max_goals=5)
    max_idx = np.unravel_index(np.argmax(probs), probs.shape)
    best_p  = probs[max_idx] * 100

    # Colores de la paleta
    BG    = pal['bg_primary'];   BG2   = pal['bg_secondary']
    WHITE = pal['text_primary']; GRAY  = pal['text_secondary']
    RED   = pal['accent'];       ACC2  = pal['accent2']
    CHIGH = pal['cell_high'];    CMID  = pal['cell_mid']
    CLOW  = pal['cell_low'];     BRAND = pal['brand_color']

    t1_es = team_es(team1).upper()
    t2_es = team_es(team2).upper()

    # ── Figura 1080×1080 ──────────────────────────────────────────────────────
    FIG_W = FIG_H = 7.2   # 1080px / 150dpi

    HEADER_H = 0.230
    FOOTER_H = 0.058
    FACT_H   = 0.046
    PROB_H   = 0.138
    COLHDR_H = 0.055
    ROWHDR_W = 0.088
    R_MARGIN = 0.010

    GRID_Y = FOOTER_H + FACT_H + PROB_H + 0.006
    GRID_H = 1.0 - HEADER_H - GRID_Y - COLHDR_H
    N = 6  # goles 0..5
    CELL_H = GRID_H / N
    CELL_W = (1.0 - ROWHDR_W - R_MARGIN) / N

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Gradiente de fondo
    bg_rgb, bg2_rgb = hex_rgb(BG), hex_rgb(BG2)
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = np.array(bg_rgb)/255*(1-t) + np.array(bg2_rgb)/255*t
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.set_zorder(-100)
    bgax.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bgax.axis('off')

    # ── Header ────────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=3.0)
    hax.text(0.50, 0.98, '¿QUIÉN GANA HOY?',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(30))
    hax.text(0.50, 0.62, label,
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=9.5)

    # ── Paneles de equipo ─────────────────────────────────────────────────────
    for col_x, team_key, name_es, elo, lam, border_c in [
        (0.04, team1, t1_es, elo1, lam1, CHIGH),
        (0.56, team2, t2_es, elo2, lam2, ACC2),
    ]:
        pax = fig.add_axes([col_x, 1-HEADER_H + HEADER_H*0.06, 0.38, HEADER_H*0.44])
        pax.set_facecolor(BG); pax.axis('off')
        for sp in pax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(border_c); sp.set_linewidth(1.5)
        fi = get_escudo(team_key, size=(48, 32))
        if fi is not None:
            pax.add_artist(AnnotationBbox(fi, (0.37, 0.80),
                           frameon=False, xycoords='axes fraction',
                           box_alignment=(1.0, 0.5)))
            pax.text(0.40, 0.80, name_es, color=border_c, ha='left', va='center',
                     fontsize=10, fontweight='bold', transform=pax.transAxes)
        else:
            pax.text(0.50, 0.80, name_es, color=border_c, ha='center', va='center',
                     fontsize=10, fontweight='bold', transform=pax.transAxes)
        pax.text(0.50, 0.40, f'ELO: {elo:.0f}',
                 color=WHITE, ha='center', va='center', fontsize=8.5, transform=pax.transAxes)
        pax.text(0.50, 0.10, f'λ esperado = {lam:.3f} goles',
                 color=GRAY, ha='center', va='center', fontsize=7, transform=pax.transAxes)

    # ── VS central ────────────────────────────────────────────────────────────
    vax = fig.add_axes([0.40, 1-HEADER_H + HEADER_H*0.06, 0.18, HEADER_H*0.44])
    vax.set_facecolor(BG2); vax.axis('off')
    vax.text(0.50, 0.55, 'VS', color=RED, ha='center', va='center',
             fontsize=22, fontweight='bold', transform=vax.transAxes)
    vax.text(0.50, 0.18, 'Poisson · ELO', color=GRAY, ha='center', va='center',
             fontsize=6.5, transform=vax.transAxes)

    # ── Cabecera de columnas (goles equipo 2) ─────────────────────────────────
    chdr_y = GRID_Y + GRID_H
    lbl_ax = fig.add_axes([ROWHDR_W, chdr_y + COLHDR_H*0.55, N*CELL_W, COLHDR_H*0.42])
    lbl_ax.set_facecolor(BG2); lbl_ax.axis('off')
    lbl_ax.text(0.5, 0.5, f'GOLES  {t2_es}',
                color=ACC2, ha='center', va='center',
                fontsize=7, fontweight='bold', transform=lbl_ax.transAxes)
    for j in range(N):
        nax = fig.add_axes([ROWHDR_W + j*CELL_W, chdr_y, CELL_W, COLHDR_H*0.55])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.5, 0.5, str(j), color=WHITE, ha='center', va='center',
                 fontsize=9, fontweight='bold', transform=nax.transAxes)

    # ── Cabecera de filas (goles equipo 1) ────────────────────────────────────
    rmx = fig.add_axes([0, GRID_Y, ROWHDR_W*0.35, GRID_H])
    rmx.set_facecolor(BG2); rmx.axis('off')
    rmx.text(0.5, 0.5, f'GOLES  {t1_es}',
             color=CHIGH, ha='center', va='center', fontsize=6, fontweight='bold',
             rotation=90, transform=rmx.transAxes)
    for i in range(N):
        nax = fig.add_axes([ROWHDR_W*0.35, GRID_Y + i*CELL_H, ROWHDR_W*0.65, CELL_H])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.65, 0.5, str(i), color=WHITE, ha='center', va='center',
                 fontsize=9, fontweight='bold', transform=nax.transAxes)

    # ── Celdas del heatmap ────────────────────────────────────────────────────
    def cell_style(p):
        if p > 0.090:   return CHIGH, WHITE,  True,  11
        elif p > 0.045: return CMID,  WHITE,  True,  9
        elif p > 0.015: return CLOW,  CHIGH,  True,  8
        elif p > 0.004: return BG2,   WHITE,  False, 7
        else:           return BG,    GRAY,   False, 6

    for i in range(N):
        for j in range(N):
            p  = probs[i, j]
            fc, tc, bold, fs = cell_style(p)
            cax = fig.add_axes([ROWHDR_W + j*CELL_W, GRID_Y + i*CELL_H, CELL_W, CELL_H])
            cax.set_xlim(0,1); cax.set_ylim(0,1)
            cax.add_patch(mpatches.Rectangle((0,0), 1, 1, facecolor=fc, edgecolor='none',
                          zorder=0, transform=cax.transAxes, clip_on=False))
            cax.axis('off')
            if i == j:
                for sp in cax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(GOLD); sp.set_linewidth(1.5)
            if p >= 0.003:
                txt = f'{p*100:.1f}' if p >= 0.010 else f'{p*100:.2f}'
                cax.text(0.5, 0.5, txt, color=tc, ha='center', va='center',
                         fontsize=fs, fontweight='bold' if bold else 'normal',
                         transform=cax.transAxes, zorder=5)

    # ── Bloques de resultado ──────────────────────────────────────────────────
    prob_y  = FOOTER_H + FACT_H + 0.006
    BLK_GAP = 0.012
    BLK_W   = (1.0 - 4*BLK_GAP) / 3
    max_p   = max(p_home, p_draw, p_away)

    result_blocks = [
        (team1, f'GANA {t1_es}', p_home),
        (None,  'EMPATE',         p_draw),
        (team2, f'GANA {t2_es}',  p_away),
    ]

    for k, (team_key, blk_label, pval) in enumerate(result_blocks):
        bx     = BLK_GAP + k * (BLK_W + BLK_GAP)
        is_max = (pval >= max_p - 1e-9)
        pct_c  = RED   if is_max else GRAY
        lbl_c  = WHITE if is_max else GRAY
        pct_fs = 30    if is_max else 21

        seg = fig.add_axes([bx, prob_y, BLK_W, PROB_H])
        seg.set_facecolor(BG2)
        seg.axis('off'); seg.set_xlim(0,1); seg.set_ylim(0,1)
        if is_max:
            for sp in seg.spines.values():
                sp.set_visible(True); sp.set_edgecolor(RED); sp.set_linewidth(2.0)

        if team_key:
            fi_blk = get_escudo(team_key, size=(28, 19))
            if fi_blk is not None:
                seg.add_artist(AnnotationBbox(fi_blk, (0.36, 0.84),
                               frameon=False, xycoords='axes fraction',
                               box_alignment=(1.0, 0.5)))
                seg.text(0.39, 0.84, blk_label, color=lbl_c, ha='left', va='center',
                         fontsize=6.5, fontweight='bold', transform=seg.transAxes)
            else:
                seg.text(0.50, 0.84, blk_label, color=lbl_c, ha='center', va='center',
                         fontsize=6.5, fontweight='bold', transform=seg.transAxes)
        else:
            seg.text(0.50, 0.84, blk_label, color=lbl_c, ha='center', va='center',
                     fontsize=6.5, fontweight='bold', transform=seg.transAxes)

        seg.text(0.50, 0.38, f'{pval*100:.1f}%', color=pct_c, ha='center', va='center',
                 fontsize=pct_fs, fontweight='bold', transform=seg.transAxes)

    # ── Footer ────────────────────────────────────────────────────────────────
    fact_txt = (f'Marcador más probable: {team_es(team1)} {max_idx[0]}-{max_idx[1]}'
                f' {team_es(team2)} ({best_p:.1f}%)')

    fct = fig.add_axes([0, FOOTER_H, 1, FACT_H])
    fct.set_facecolor(BG); fct.axis('off')
    fct.text(0.5, 0.5, fact_txt, color=GRAY, fontsize=6.5,
             ha='center', va='center', style='italic', transform=fct.transAxes)

    fax = fig.add_axes([0, 0, 1, FOOTER_H*0.55])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.axhline(1, color=RED, lw=2.0)
    fax.text(0.015, 0.50, 'Modelo: ELO + Poisson · eloratings.net methodology',
             color=GRAY, fontsize=6, ha='left', va='center', transform=fax.transAxes)
    shadow = [mpe.withStroke(linewidth=2, foreground='#000000')]
    fax.text(0.985, 0.50, 'MAU-STATISTICS', color=BRAND, ha='right', va='center',
             transform=fax.transAxes, path_effects=shadow, **bebas(13))

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  ✓ {out_path.name}')
    return p_home, p_draw, p_away, max_idx, best_p


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match', default='', help='Filtro de partido (ej: COL)')
    args = parser.parse_args()

    matches = MATCHES
    if args.match:
        matches = [m for m in MATCHES if args.match.upper() in m['file'].upper()
                   or args.match.upper() in m['label'].upper()]

    print(f'\n{"═"*58}')
    print(f'  PREDICCIONES DEL DÍA — {len(matches)} partido(s)')
    print(f'{"═"*58}\n')

    # Cargar tracker (nombre empieza con dígito → importlib)
    tracker = importlib.import_module('04_predicciones_tracker')

    pal_keys   = list(PALETAS.keys())
    last_pal   = None

    for m in matches:
        # Paleta aleatoria, sin repetir consecutiva
        available = [k for k in pal_keys if k != last_pal]
        pal_key   = random.choice(available)
        last_pal  = pal_key
        pal       = PALETAS[pal_key]
        print(f'🎨  Paleta: {pal_key}')
        print(f'⚽  {m["label"]}')

        ph, pd, pa, midx, bp = render_prediccion(m, pal)
        winner = team_es(m['team1']) if ph > pa else (team_es(m['team2']) if pa > ph else 'Empate')
        print(f'    P(local)={ph*100:.1f}%  P(empate)={pd*100:.1f}%  P(visita)={pa*100:.1f}%')
        print(f'    Más probable: {team_es(m["team1"])} {midx[0]}-{midx[1]} {team_es(m["team2"])} ({bp:.1f}%)\n')

        # Registrar predicción en el tracker
        lam1, lam2 = elo_to_lambda(m['elo1'], m['elo2'], ALL_ELOS_TODAY)
        tracker.registrar_prediccion(
            equipo_local      = m['team1'],
            equipo_visitante  = m['team2'],
            elo_local         = m['elo1'],
            elo_visitante     = m['elo2'],
            prob_local        = ph,
            prob_empate       = pd,
            prob_visitante    = pa,
            marcador_probable = f"{midx[0]}-{midx[1]}",
            lambda_local      = lam1,
            lambda_visitante  = lam2,
            fecha_partido     = m['fecha'],
            paleta            = pal_key,
        )

    print('Listo.')


if __name__ == '__main__':
    main()
