#!/usr/bin/env python3
"""
14_simulacion_montecarlo.py
Simulación de 10,000 escenarios del Clausura 2026.
Lee tabla actual + partidos restantes (jornadas 13-17) y simula con modelo Poisson.
Genera heatmap de probabilidades de posición final por equipo.
Salida: output/charts/montecarlo_clausura2026.png  (150 DPI)
"""

import json, glob, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from scipy.stats import poisson
from PIL import Image, ImageDraw
import urllib.request

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETTE, bebas, hex_rgba, hex_rgb

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

N_SIM = 10_000

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
BG    = PALETTE['bg_main']
WHITE = PALETTE['text_primary']
GRAY  = PALETTE['text_secondary']
RED   = PALETTE['accent']

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
# DATOS: TABLA ACTUAL + PARTIDOS RESTANTES
# ─────────────────────────────────────────────────────────────────────────────
def load_current_data():
    """Lee clausura_2026.json: tabla actual + partidos pendientes."""
    fpath = HIST_DIR / 'historico_clausura_2026.json'
    data  = json.load(open(fpath, encoding='utf-8'))

    # Tabla actual (después de J12)
    tabla = {}
    for row in data.get('tabla', []):
        equipo = norm(row['equipo'])
        tabla[equipo] = {
            'pts': int(row.get('pts', 0)),
            'pj':  int(row.get('pj',  0)),
            'g':   int(row.get('g',   0)),
            'e':   int(row.get('e',   0)),
            'p':   int(row.get('p',   0)),
            'gf':  int(row.get('gf',  0)),
            'gc':  int(row.get('gc',  0)),
        }

    # Partidos pendientes (terminado=False)
    pendientes = []
    for p in data.get('partidos', []):
        if p.get('terminado'):
            continue
        lo = norm(p.get('local', ''))
        vi = norm(p.get('visitante', ''))
        pendientes.append({'local': lo, 'visitante': vi})

    return tabla, pendientes

# ─────────────────────────────────────────────────────────────────────────────
# MODELO DE POISSON
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

def build_model(matches):
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

def get_lambda(model, local, visitante):
    mu_h = model['mu_home']; mu_a = model['mu_away']
    att  = model['att'];     defe = model['defe']
    lam_l = att.get(local,     {}).get('home', 1.0) * defe.get(visitante, {}).get('away', 1.0) * mu_h
    lam_v = att.get(visitante, {}).get('away', 1.0) * defe.get(local,     {}).get('home', 1.0) * mu_a
    return max(lam_l, 0.2), max(lam_v, 0.2)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def simulate_one(model, tabla_base, pendientes, rng):
    """Un escenario: simula todos los partidos pendientes y devuelve ranking final."""
    pts = {t: v['pts'] for t, v in tabla_base.items()}
    gf  = {t: v['gf']  for t, v in tabla_base.items()}
    gc  = {t: v['gc']  for t, v in tabla_base.items()}

    for p in pendientes:
        lo = p['local']; vi = p['visitante']
        if lo not in pts: pts[lo] = gf[lo] = gc[lo] = 0
        if vi not in pts: pts[vi] = gf[vi] = gc[vi] = 0

        lam_l, lam_v = get_lambda(model, lo, vi)
        gl = rng.poisson(lam_l)
        gv = rng.poisson(lam_v)

        gf[lo] += gl; gc[lo] += gv
        gf[vi] += gv; gc[vi] += gl

        if gl > gv:        pts[lo] += 3
        elif gl == gv:     pts[lo] += 1; pts[vi] += 1
        else:              pts[vi] += 3

    # Ordenar: pts desc, luego diferencia de goles desc, luego goles a favor desc
    teams = sorted(pts.keys(), key=lambda t: (pts[t], gf[t] - gc[t], gf[t]), reverse=True)
    return {t: i + 1 for i, t in enumerate(teams)}

# ─────────────────────────────────────────────────────────────────────────────
# ESCUDO
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_TEAM = 'https://images.fotmob.com/image_resources/logo/teamlogo/{}.png'

def get_shield(team_name: str, size: int = 48) -> np.ndarray | None:
    name_key = {
        'América': 'CF America', 'FC Juárez': 'FC Juarez',
        'Mazatlán': 'Mazatlan FC', 'San Luis': 'Atletico de San Luis',
        'Querétaro': 'Queretaro FC', 'León': 'León',
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
def render(teams_sorted, prob_matrix, tabla_base, output_path):
    """
    teams_sorted: lista de equipos ordenados por P(top4)
    prob_matrix:  dict[team][pos] = probabilidad (0-1)
    """
    N_TEAMS = len(teams_sorted)
    N_POS   = N_TEAMS

    # Márgenes y tamaños (en pulgadas)
    LEFT_W   = 1.0    # espacio para escudo + nombre
    CELL_W   = 0.46
    HEADER_H = 1.20
    FOOTER_H = 0.55
    CELL_H   = 0.42

    FIG_W = LEFT_W + N_POS * CELL_W + 0.15
    FIG_H = HEADER_H + N_TEAMS * CELL_H + FOOTER_H

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Gradiente de fondo
    grad = np.zeros((200, 2, 3))
    for i in range(200):
        t = i / 199
        grad[i] = np.array([0x0a, 0x0e, 0x12]) / 255 * (1-t) + np.array([0x13, 0x1a, 0x24]) / 255 * t
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(grad, aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    bg.axis('off')

    # Mapa de colores: verde→amarillo→rojo para 100→0%
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'prob', ['#0d1117', '#1a3a1a', '#2ea043', '#f5a623', '#f85149'], N=256)

    # Convertir a figura-normalized coords
    def fig_coord(row_i, col_j):
        """Celda (row=0 es top team): retorna (x0, y0, w, h) en fig coords."""
        x0 = (LEFT_W + col_j * CELL_W) / FIG_W
        y0 = (FOOTER_H + (N_TEAMS - 1 - row_i) * CELL_H) / FIG_H
        w  = CELL_W / FIG_W
        h  = CELL_H / FIG_H
        return x0, y0, w, h

    # ── HEADER ────────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H / FIG_H, 1, HEADER_H / FIG_H])
    hax.set_facecolor(PALETTE['bg_secondary'])
    hax.axis('off')
    hax.axhline(0, color=RED, lw=2.5)

    hax.text(0.50, 0.88, '¿CÓMO TERMINA EL CLAUSURA 2026?',
             color=WHITE, ha='center', va='top', transform=hax.transAxes, **bebas(22))
    hax.text(0.50, 0.56,
             f'Simulación de {N_SIM:,} escenarios · Jornadas 13–17 · 5 jornadas · 45 partidos',
             color=GRAY, ha='center', va='top', transform=hax.transAxes, fontsize=9)
    hax.text(0.50, 0.32,
             'Cada celda = % de simulaciones donde ese equipo terminó en esa posición. '
             'Equipos ordenados por P(Top 4) decreciente.',
             color=PALETTE['text_secondary'], ha='center', va='top',
             transform=hax.transAxes, fontsize=7.2)

    # ── CABECERA DE COLUMNAS (posiciones) ─────────────────────────────────────
    col_header_y = (FOOTER_H + N_TEAMS * CELL_H) / FIG_H
    col_header_h = (HEADER_H * 0.20) / FIG_H

    # Separador liguilla / liguilla (top 4 / 5-8 / 9-18)
    LIGUILLA_DIRECT = 4
    REPECHAJE       = 8

    for j in range(N_POS):
        x0, _, w, _ = fig_coord(0, j)
        col_ax = fig.add_axes([x0, col_header_y - col_header_h * 0.95,
                               w, col_header_h * 0.95])
        pos = j + 1
        if pos <= LIGUILLA_DIRECT:
            bg_col = '#1a3a1a'
        elif pos <= REPECHAJE:
            bg_col = '#2a2a10'
        else:
            bg_col = PALETTE['bg_card']
        col_ax.set_facecolor(bg_col)
        col_ax.axis('off')
        col_ax.text(0.5, 0.5, f'#{pos}', color=WHITE,
                    ha='center', va='center',
                    fontsize=7.5, fontweight='bold' if pos <= LIGUILLA_DIRECT else 'normal')

    # ── CELDAS ────────────────────────────────────────────────────────────────
    for i, team in enumerate(teams_sorted):
        prob_row = prob_matrix[team]  # dict pos → prob

        # Columna izquierda: escudo
        row_y_abs  = FOOTER_H + (N_TEAMS - 1 - i) * CELL_H
        shield_h   = CELL_H * 0.80
        shield_w   = shield_h * (FIG_H / FIG_W)
        shield_x   = 0.010 / FIG_W
        shield_y   = (row_y_abs + CELL_H * 0.10) / FIG_H

        shield = get_shield(team, 48)
        if shield is not None:
            sax = fig.add_axes([shield_x, shield_y, shield_w / FIG_W * FIG_W, shield_h / FIG_H])
            sax.set_facecolor('#f8f8fc')
            sax.imshow(shield)
            sax.axis('off')

        # Nombre equipo
        name_ax = fig.add_axes([shield_x + shield_w / FIG_W * FIG_W + 0.005,
                                 (row_y_abs + CELL_H * 0.2) / FIG_H,
                                 (LEFT_W * 0.45) / FIG_W, CELL_H * 0.6 / FIG_H])
        name_ax.axis('off')
        short = team[:6].upper() if len(team) > 8 else team.upper()
        name_ax.text(0, 0.5, short, color=WHITE, ha='left', va='center',
                     fontsize=7.0, fontweight='bold')

        # Fondo de fila alterno
        for j in range(N_POS):
            x0, y0, w, h = fig_coord(i, j)
            cell_ax = fig.add_axes([x0, y0, w, h])

            pos = j + 1
            prob = prob_row.get(pos, 0.0)

            # Color: verde para alta prob, oscuro para baja
            color_val = cmap(min(prob * 4.0, 1.0))  # escala para que 25%+ sea verde intenso
            cell_ax.set_facecolor(color_val)

            # Borde suave
            for spine in cell_ax.spines.values():
                spine.set_edgecolor(PALETTE['divider'])
                spine.set_linewidth(0.5)

            cell_ax.tick_params(left=False, bottom=False,
                                labelleft=False, labelbottom=False)

            if prob >= 0.005:
                txt_color = WHITE if prob >= 0.08 else GRAY
                cell_ax.text(0.5, 0.5, f'{prob*100:.1f}',
                             color=txt_color, ha='center', va='center',
                             fontsize=6.8, fontweight='bold' if prob >= 0.15 else 'normal')

    # ── LEYENDA ────────────────────────────────────────────────────────────────
    leg_ax = fig.add_axes([0.015, FOOTER_H * 0.55 / FIG_H, 0.35, FOOTER_H * 0.30 / FIG_H])
    leg_ax.axis('off')
    for k, (lbl, col) in enumerate([
        ('Top 4 — Liguilla directa', '#1a3a1a'),
        ('5-8 — Repechaje', '#2a2a10'),
    ]):
        rx = 0.02 + k * 0.50
        leg_ax.add_patch(plt.Rectangle((rx, 0.1), 0.06, 0.8,
                                        facecolor=col, transform=leg_ax.transAxes))
        leg_ax.text(rx + 0.09, 0.5, lbl, color=GRAY, va='center',
                    fontsize=6.5, transform=leg_ax.transAxes)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H * 0.50 / FIG_H])
    fax.set_facecolor(PALETTE['bg_secondary'])
    fax.axis('off')
    fax.axhline(1, color=RED, lw=2.0)

    fax.text(0.015, 0.45, 'Fuente: FotMob · Clausura 2026',
             color=GRAY, fontsize=9, ha='left', va='center', transform=fax.transAxes)
    fax.text(0.985, 0.45, 'MAU-STATISTICS',
             color=RED, ha='right', va='center', transform=fax.transAxes, **bebas(20))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'✓ Guardado: {output_path}')


def main():
    print('Cargando datos del Clausura 2026...')
    tabla_base, pendientes = load_current_data()
    print(f'  {len(tabla_base)} equipos en tabla, {len(pendientes)} partidos pendientes')

    print('Construyendo modelo Poisson...')
    matches = load_model_matches()
    model   = build_model(matches)
    print(f'  {len(matches)} partidos de entrenamiento')

    print(f'Simulando {N_SIM:,} escenarios...')
    rng = np.random.default_rng(42)
    # posiciones[equipo][pos] = conteo
    posiciones = {t: defaultdict(int) for t in tabla_base}
    for sim_i in range(N_SIM):
        pos_map = simulate_one(model, tabla_base, pendientes, rng)
        for team, pos in pos_map.items():
            if team in posiciones:
                posiciones[team][pos] += 1

    # Convertir a probabilidades
    prob_matrix = {}
    for team in posiciones:
        prob_matrix[team] = {pos: cnt / N_SIM for pos, cnt in posiciones[team].items()}

    # Ordenar por P(top 4)
    def p_top4(team):
        return sum(prob_matrix[team].get(p, 0) for p in range(1, 5))

    teams_sorted = sorted(prob_matrix.keys(), key=p_top4, reverse=True)

    print('\nTop 5 equipos por P(Liguilla directa):')
    for t in teams_sorted[:5]:
        print(f'  {t:20s}: {p_top4(t)*100:.1f}%')

    out = OUT_DIR / 'montecarlo_clausura2026.png'
    render(teams_sorted, prob_matrix, tabla_base, out)


if __name__ == '__main__':
    main()
