#!/usr/bin/env python3
"""
gen_predicciones_ccl.py
Predicciones CONCACAF Champions Cup 2025-26.

Modelo: ELO calculado desde resultados CCL 2025-26 + Poisson + Dixon-Coles (rho=-0.13)
Imágenes: 1080x1080px, estructura idéntica a Liga MX, paleta medianoche_neon

Logos: data/raw/logos/ccl/{equipo}.png (descargados por scrape_ccl_logos.py)
Salida: output/charts/predicciones/CCL_2025-26/

Uso:
  python gen_predicciones_ccl.py                         # partidos pendientes
  python gen_predicciones_ccl.py --stage semifinals      # solo semis
  python gen_predicciones_ccl.py --match America         # filtrar por nombre
  python gen_predicciones_ccl.py --leg2                  # solo 2da vuelta (contexto agregado)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image as _PIL
from scipy.stats import poisson

warnings.filterwarnings('ignore')

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from config_visual import PALETAS, PALETTE, bebas, hex_rgb

try:
    from config_visual import BEBAS_TTF, BEBAS_AVAILABLE
except ImportError:
    BEBAS_TTF = None
    BEBAS_AVAILABLE = False

CCL_JSON = BASE / 'data/raw/fotmob/ccl'
LOGO_DIR = BASE / 'data/raw/logos/ccl'
OUT_DIR  = BASE / 'output/charts/predicciones/CCL_2025-26'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── ELO desde resultados CCL ─────────────────────────────────────────────────
K        = 35     # K alto — torneo continental corto
HOME_ADV = 75     # ventaja local moderada (partidos en sede neutral frecuentes)
SCALE    = 400
BASE_ELO = 1500


def elo_expected(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / SCALE))


def goal_mult(gh, ga):
    diff = abs(gh - ga)
    return 1.0 if diff == 0 else 1.0 + 0.5 * np.log(diff + 1)


def load_ccl_fixtures() -> list[dict]:
    files = sorted(CCL_JSON.glob('ccl_fixtures_*.json'))
    if not files:
        raise FileNotFoundError(f'No se encontró ccl_fixtures_*.json en {CCL_JSON}')
    return json.loads(files[-1].read_text())['matches']


def calc_elo_ccl(matches: list[dict]) -> dict[str, float]:
    """Calcula ELO de todos los equipos CCL desde cero con los resultados disponibles."""
    elos: dict[str, float] = {}

    for m in sorted(matches, key=lambda x: x['date']):
        if not m.get('finished'):
            continue
        home = m['home']
        away = m['away']
        gh   = int(m['home_score'])
        ga   = int(m['away_score'])

        elo_h = elos.get(home, BASE_ELO)
        elo_a = elos.get(away, BASE_ELO)

        exp_h = elo_expected(elo_h + HOME_ADV, elo_a)
        res_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        mult  = goal_mult(gh, ga)

        elos[home] = elo_h + K * mult * (res_h       - exp_h)
        elos[away] = elo_a + K * mult * ((1 - res_h) - (1 - exp_h))

    return elos


# ─── Dixon-Coles ──────────────────────────────────────────────────────────────
DC_RHO = -0.13


def _dc_tau(i, j, lh, la, rho=DC_RHO):
    if   i == 0 and j == 0: return 1 - lh * la * rho
    elif i == 0 and j == 1: return 1 + lh * rho
    elif i == 1 and j == 0: return 1 + la * rho
    elif i == 1 and j == 1: return 1 - rho
    return 1.0


def poisson_probs(lam_h, lam_a, max_goals=5):
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = (poisson.pmf(i, lam_h) *
                           poisson.pmf(j, lam_a) *
                           _dc_tau(i, j, lam_h, lam_a))
    probs /= probs.sum()
    p_home = float(np.sum(np.tril(probs, -1)))
    p_draw = float(np.sum(np.diag(probs)))
    p_away = float(np.sum(np.triu(probs, 1)))
    return probs, p_home, p_draw, p_away


MU_CCL = 1.25   # media de goles CCL (torneo competitivo, menos goles que Liga MX)


def lambdas(elo_h, elo_a):
    """ELO diferencial → lambdas para Poisson."""
    exp_h = elo_expected(elo_h + HOME_ADV, elo_a)
    lam_h = MU_CCL * (exp_h / 0.5)
    lam_a = MU_CCL * ((1 - exp_h) / 0.5)
    return max(0.1, lam_h), max(0.1, lam_a)


def bootstrap_ci(elo_h, elo_a, n=1000):
    lh, la = lambdas(elo_h, elo_a)
    results = {'home': [], 'draw': [], 'away': []}
    rng = np.random.default_rng(42)
    for _ in range(n):
        gh = rng.poisson(lh)
        ga = rng.poisson(la)
        if gh > ga:   results['home'].append(1)
        elif gh == ga: results['draw'].append(1)
        else:          results['away'].append(1)
    n_sims = n
    return {
        'home': (
            np.percentile([r / n_sims for r in [len(results['home'])]]*100, 2.5),
            np.percentile([r / n_sims for r in [len(results['home'])]]*100, 97.5),
        ),
        'ph': len(results['home']) / n_sims,
        'pd': len(results['draw']) / n_sims,
        'pa': len(results['away']) / n_sims,
    }


# ─── Logos ────────────────────────────────────────────────────────────────────
def get_logo(team: str, zoom: float = 1.0):
    path = LOGO_DIR / f'{team}.png'
    if not path.exists():
        return None
    try:
        img = _PIL.open(path).convert('RGBA')
        return OffsetImage(np.array(img), zoom=zoom)
    except Exception:
        return None


# ─── Nombres de display ───────────────────────────────────────────────────────
DISPLAY = {
    'América':          'AMÉRICA',
    'Cruz Azul':        'CRUZ AZUL',
    'Tigres':           'TIGRES',
    'Toluca':           'TOLUCA',
    'Monterrey':        'MONTERREY',
    'Nashville SC':     'NASHVILLE',
    'LAFC':             'LAFC',
    'LA Galaxy':        'LA GALAXY',
    'Seattle Sounders': 'SEATTLE',
    'Cincinnati':       'CINCINNATI',
    'Inter Miami CF':   'INTER MIAMI',
    'Philadelphia':     'PHILA.',
    'Vancouver':        'VANCOUVER',
    'San Diego FC':     'SAN DIEGO',
    'LD Alajuelense':   'ALAJUELENSE',
    'C.S. Cartaginés':  'CARTAGINÉS',
    'Atlético Ottawa':  'A. OTTAWA',
    'Real Espana':      'REAL ESPAÑA',
}


def display_name(name: str) -> str:
    return DISPLAY.get(name, name.upper())


# ─── Imagen predicción ────────────────────────────────────────────────────────
PALETA_CCL = 'medianoche_neon'   # paleta fija CCL (identificación visual)


def generar_imagen(match: dict, elos: dict, out_path: Path,
                   agg_home: int = None, agg_away: int = None):
    """
    Genera imagen 1080×1080 de predicción CCL.
    agg_home / agg_away: marcador agregado de la ida (para 2da vuelta).
    """
    _p  = PALETAS[PALETA_CCL]
    # Mezclar: PALETAS tiene los colores temáticos, PALETTE tiene los neutrales
    pal = {
        'bg_main':        _p['bg_primary'],
        'bg_secondary':   _p['bg_secondary'],
        'bg_card':        PALETTE['bg_card'],
        'text_primary':   _p['text_primary'],
        'text_secondary': _p['text_secondary'],
        'accent':         _p['accent'],
        'divider':        PALETTE['divider'],
        'heatmap_low':    _p['cell_low'],
        'heatmap_high':   _p['cell_high'],
    }

    home = match['home']
    away = match['away']
    date = match['date']
    stage_label = {
        'play_in':      'Primera Ronda',
        'quarterfinals': 'Cuartos de Final',
        'semifinals':   'Semifinales',
        'final':        'Final',
    }.get(match.get('stage', ''), 'CONCACAF Champions Cup')

    elo_h = elos.get(home, BASE_ELO)
    elo_a = elos.get(away, BASE_ELO)
    lh, la = lambdas(elo_h, elo_a)
    probs, ph, pd, pa = poisson_probs(lh, la)

    # Marcador más probable
    idx = np.unravel_index(np.argmax(probs), probs.shape)
    score_str = f'{idx[0]}-{idx[1]}'

    # Monte Carlo CI
    ci = bootstrap_ci(elo_h, elo_a, n=1000)

    # ── Canvas ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 7.2), dpi=150)
    fig.patch.set_facecolor(pal['bg_main'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1080); ax.set_ylim(0, 1080)
    ax.set_facecolor(pal['bg_main'])
    ax.axis('off')

    W, H = 1080, 1080

    # ── Banda de título ──────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, H - 110), W, 110, boxstyle='square,pad=0',
        facecolor=pal['bg_secondary'], edgecolor='none'))
    ax.axhline(H - 110, color=pal['accent'], linewidth=2.5)

    ax.text(W / 2, H - 40, 'CONCACAF CHAMPIONS CUP 2025-26',
            ha='center', va='center', color=pal['text_primary'],
            **bebas(20))
    ax.text(W / 2, H - 80, f'{stage_label}  ·  {date}',
            ha='center', va='center', color=pal['text_secondary'],
            **bebas(14))

    # ── Logos y nombres ──────────────────────────────────────────────────────
    logo_y = H - 230
    for team, x_center in [(home, W * 0.27), (away, W * 0.73)]:
        logo = get_logo(team, zoom=0.55)
        if logo:
            ab = AnnotationBbox(logo, (x_center, logo_y + 20),
                                frameon=False, xycoords='data')
            ax.add_artist(ab)

    ax.text(W * 0.27, H - 330, display_name(home),
            ha='center', va='center', color=pal['text_primary'], **bebas(28))
    ax.text(W * 0.73, H - 330, display_name(away),
            ha='center', va='center', color=pal['text_primary'], **bebas(28))

    # ELOs
    ax.text(W * 0.27, H - 360, f'ELO {elo_h:.0f}',
            ha='center', va='center', color=pal['text_secondary'], **bebas(13))
    ax.text(W * 0.73, H - 360, f'ELO {elo_a:.0f}',
            ha='center', va='center', color=pal['text_secondary'], **bebas(13))

    # VS
    ax.text(W * 0.5, H - 250, 'VS',
            ha='center', va='center', color=pal['accent'], **bebas(36))

    # Agregado si aplica (2da vuelta)
    if agg_home is not None and agg_away is not None:
        agg_txt = f'Agregado: {agg_home}–{agg_away}'
        ax.text(W / 2, H - 385, agg_txt,
                ha='center', va='center', color='#FFD700', **bebas(15))

    # ── Probabilidades ───────────────────────────────────────────────────────
    sep = H - 420
    ax.axhline(sep, color=pal['divider'], linewidth=1)

    cats = [
        (home[:10], ph, W * 0.22),
        ('Empate',  pd, W * 0.50),
        (away[:10], pa, W * 0.78),
    ]
    max_p = max(ph, pd, pa)
    for label, prob, xc in cats:
        size = 44 if prob == max_p else 34
        color = pal['accent'] if prob == max_p else pal['text_primary']
        ax.text(xc, sep - 55, f'{prob*100:.1f}%',
                ha='center', va='center', color=color, **bebas(size))
        ax.text(xc, sep - 90, label.upper(),
                ha='center', va='center', color=pal['text_secondary'], **bebas(13))

    # ── Heatmap ──────────────────────────────────────────────────────────────
    hm_top  = sep - 115
    hm_size = 340
    hm_left = (W - hm_size) / 2
    hm_bot  = hm_top - hm_size

    c1 = pal.get('heatmap_low',  pal['bg_card'])
    c2 = pal.get('heatmap_high', pal['accent'])
    cmap = LinearSegmentedColormap.from_list('ccl', [c1, c2])

    max_goals = probs.shape[0]
    cell_w = hm_size / max_goals
    cell_h = hm_size / max_goals

    for i in range(max_goals):           # goles local (fila)
        for j in range(max_goals):       # goles visitante (col)
            p_cell = float(probs[i, j])
            color  = cmap(p_cell / (probs.max() + 1e-9))
            rx = hm_left + j * cell_w
            ry = hm_bot  + (max_goals - 1 - i) * cell_h
            ax.add_patch(mpatches.Rectangle(
                (rx, ry), cell_w - 1, cell_h - 1,
                facecolor=color, edgecolor='none'))
            if p_cell > 0.04:
                ax.text(rx + cell_w / 2, ry + cell_h / 2,
                        f'{p_cell*100:.0f}%',
                        ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold')

    # Etiquetas de eje heatmap
    for j in range(max_goals):
        ax.text(hm_left + j * cell_w + cell_w / 2, hm_bot - 14,
                str(j), ha='center', va='center',
                color=pal['text_secondary'], fontsize=9)
    for i in range(max_goals):
        ax.text(hm_left - 14, hm_bot + (max_goals - 1 - i) * cell_h + cell_h / 2,
                str(i), ha='center', va='center',
                color=pal['text_secondary'], fontsize=9)

    ax.text(W / 2, hm_bot - 35, display_name(away),
            ha='center', va='center', color=pal['text_secondary'], **bebas(11))
    ax.text(hm_left - 35, hm_bot + hm_size / 2, display_name(home),
            ha='center', va='center', color=pal['text_secondary'],
            rotation=90, **bebas(11))

    # ── Marcador probable + IC ───────────────────────────────────────────────
    ic_y = hm_bot - 65
    ax.text(W / 2, ic_y,
            f'Marcador probable: {score_str}',
            ha='center', va='center', color=pal['accent'], **bebas(18))

    ax.text(W / 2, ic_y - 30,
            f'IC 95%  Local {ci["ph"]*100:.0f}%  '
            f'Empate {ci["pd"]*100:.0f}%  '
            f'Visit. {ci["pa"]*100:.0f}%',
            ha='center', va='center', color=pal['text_secondary'], **bebas(11))

    # ── Footer ───────────────────────────────────────────────────────────────
    footer_y = 30
    ax.axhline(footer_y + 18, color=pal['accent'], linewidth=1.5)
    ax.text(W - 16, footer_y,
            'MAU-STATISTICS',
            ha='right', va='center', color=pal['accent'], **bebas(14))
    ax.text(16, footer_y,
            'Modelo: ELO + Poisson-Dixon-Coles · FotMob',
            ha='left', va='center', color=pal['text_secondary'], **bebas(11))

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=pal['bg_main'], edgecolor='none')
    plt.close(fig)
    print(f'  [ok] {out_path.name}')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',  default='', help='Filtrar por etapa (quarterfinals, semifinals, final)')
    parser.add_argument('--match',  default='', help='Filtrar por nombre de equipo')
    parser.add_argument('--leg2',   action='store_true', help='Mostrar marcador agregado en 2da vuelta')
    parser.add_argument('--force',  action='store_true', help='Re-generar aunque ya exista')
    args = parser.parse_args()

    matches = load_ccl_fixtures()
    elos    = calc_elo_ccl(matches)

    print('ELO CCL 2025-26 calculado:')
    for team, elo in sorted(elos.items(), key=lambda x: -x[1]):
        print(f'  {elo:6.1f}  {team}')
    print()

    # Calcular agregados por cruce — incluye 1a pierna aunque la 2a esté pendiente
    from collections import defaultdict
    pair_legs: dict[tuple, list] = defaultdict(list)
    for m in sorted(matches, key=lambda x: x['date']):
        key = tuple(sorted([m['home'], m['away']]))
        pair_legs[key].append(m)

    aggregates: dict[tuple, dict] = {}
    for pair, legs in pair_legs.items():
        # Solo nos interesa si hay exactamente 2 piernas y la 1a está terminada
        finished = [l for l in legs if l.get('finished')]
        pending  = [l for l in legs if not l.get('finished')]
        if not finished or not pending:
            continue   # cruce de una sola pierna o ya terminado → no aplica
        leg1 = finished[-1]    # más reciente terminado = ida
        leg2 = pending[0]      # próximo pendiente = vuelta
        h2 = leg2['home']
        a2 = leg2['away']
        # Goles acumulados del equipo local de la vuelta
        if leg1['home'] == h2:
            agg_h2 = int(leg1['home_score'])
            agg_a2 = int(leg1['away_score'])
        else:
            agg_h2 = int(leg1['away_score'])
            agg_a2 = int(leg1['home_score'])
        aggregates[pair] = {
            'home_leg2': h2, 'away_leg2': a2,
            'agg_home': agg_h2, 'agg_away': agg_a2,
        }

    # Filtrar partidos pendientes
    pending = [m for m in matches if not m.get('finished')]
    if args.stage:
        pending = [m for m in pending if m.get('stage', '') == args.stage]
    if args.match:
        kw = args.match.lower()
        pending = [m for m in pending if kw in m['home'].lower() or kw in m['away'].lower()]

    if not pending:
        print('Sin partidos pendientes con los filtros dados.')
        return

    print(f'Generando {len(pending)} imagen(es)...')
    for m in pending:
        fname = f"pred_{m['home'].replace(' ', '_')}_{m['away'].replace(' ', '_')}_{m['date']}.png"
        out_path = OUT_DIR / fname

        if out_path.exists() and not args.force:
            print(f'  [skip] {fname}')
            continue

        pair = tuple(sorted([m['home'], m['away']]))
        agg = aggregates.get(pair)
        agg_h = agg['agg_home'] if agg and args.leg2 else None
        agg_a = agg['agg_away'] if agg and args.leg2 else None

        generar_imagen(m, elos, out_path, agg_home=agg_h, agg_away=agg_a)

    print(f'\nListo → {OUT_DIR}')


if __name__ == '__main__':
    main()
