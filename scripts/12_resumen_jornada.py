#!/usr/bin/env python3
"""
12_resumen_jornada.py
Genera imagen resumen "PROBABILIDADES DE LA JORNADA N" con todos los partidos
en una sola infografía.

Uso:
  python 12_resumen_jornada.py 13
"""

import sys
import json
import urllib.request
import io
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFilter
from scipy.stats import poisson

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
HIST_DIR  = BASE / 'data/raw/historico'
IMG_DIR   = BASE / 'data/raw/images/teams'
OUT_BASE  = BASE / 'output/charts'
BEBAS_TTF = Path.home() / '.fonts/BebasNeue.ttf'

IMG_DIR.mkdir(parents=True, exist_ok=True)

_bebas_path = str(BEBAS_TTF) if BEBAS_TTF.exists() else None
if _bebas_path:
    fm.fontManager.addfont(_bebas_path)

def bebas(size):
    if _bebas_path:
        return FontProperties(fname=_bebas_path, size=size)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
WHITE     = '#e6edf3'
GRAY      = '#8b949e'
RED_BRAND = '#D5001C'
GREEN_WIN = '#00C87A'
YELLOW_DW = '#F5A623'

TEAM_COLORS = {
    'América':       '#FFCC00',
    'Atlas':         '#7A1C2E',
    'Chivas':        '#CC2229',
    'Cruz Azul':     '#3065AC',
    'FC Juárez':     '#C8102E',
    'León':          '#006B3F',
    'Mazatlán':      '#5B2D8E',
    'Monterrey':     '#003399',
    'Necaxa':        '#CC0000',
    'Pachuca':       '#2D6CC0',
    'Puebla':        '#4B2C8C',
    'Pumas':         '#C9A227',
    'Querétaro':     '#003087',
    'San Luis':      '#D4002B',
    'Santos Laguna': '#B0C03C',
    'Tigres':        '#FFD700',
    'Tijuana':       '#CC0000',
    'Toluca':        '#BB1E21',
}

TEAM_IDS = {
    'América':       6576,
    'Atlas':         6577,
    'Chivas':        7807,
    'Cruz Azul':     6578,
    'FC Juárez':     649424,
    'León':          1841,
    'Mazatlán':      1170234,
    'Monterrey':     7849,
    'Necaxa':        1842,
    'Pachuca':       7848,
    'Puebla':        7847,
    'Pumas':         1946,
    'Querétaro':     1943,
    'San Luis':      6358,
    'Santos Laguna': 7857,
    'Tigres':        8561,
    'Tijuana':       162418,
    'Toluca':        6618,
}

_ALIAS = {
    'america': 'América', 'cf america': 'América', 'cf américa': 'América',
    'cf. america': 'América', 'club america': 'América', 'america fc': 'América',
    'atlas': 'Atlas', 'atlas fc': 'Atlas',
    'guadalajara': 'Chivas', 'chivas': 'Chivas', 'cd guadalajara': 'Chivas',
    'cruz azul': 'Cruz Azul', 'deportivo cruz azul': 'Cruz Azul',
    'fc juárez': 'FC Juárez', 'fc juarez': 'FC Juárez', 'juárez': 'FC Juárez',
    'juarez': 'FC Juárez', 'bravos': 'FC Juárez', 'bravos de juárez': 'FC Juárez',
    'club leon': 'León', 'leon': 'León', 'club léon': 'León', 'léon': 'León',
    'mazatlan fc': 'Mazatlán', 'mazatlán fc': 'Mazatlán', 'mazatlan': 'Mazatlán',
    'mazatlán': 'Mazatlán',
    'monterrey': 'Monterrey', 'cf monterrey': 'Monterrey', 'rayados': 'Monterrey',
    'necaxa': 'Necaxa', 'club necaxa': 'Necaxa',
    'pachuca': 'Pachuca', 'cf pachuca': 'Pachuca', 'tuzos': 'Pachuca',
    'puebla': 'Puebla', 'club puebla': 'Puebla', 'puebla fc': 'Puebla',
    'pumas': 'Pumas', 'unam': 'Pumas', 'pumas unam': 'Pumas', 'club universidad': 'Pumas',
    'querétaro': 'Querétaro', 'queretaro': 'Querétaro', 'querétaro fc': 'Querétaro',
    'queretaro fc': 'Querétaro', 'gallos blancos': 'Querétaro',
    'atletico de san luis': 'San Luis', 'atlético de san luis': 'San Luis',
    'atletico san luis': 'San Luis', 'san luis': 'San Luis', 'atlético san luis': 'San Luis',
    'santos': 'Santos Laguna', 'santos laguna': 'Santos Laguna',
    'tigres': 'Tigres', 'tigres uanl': 'Tigres',
    'tijuana': 'Tijuana', 'xolos': 'Tijuana', 'xoloitzcuintles': 'Tijuana',
    'club tijuana': 'Tijuana',
    'toluca': 'Toluca', 'deportivo toluca': 'Toluca',
}

def norm(name: str) -> str:
    return _ALIAS.get(name.lower().strip(), name.strip())

TORNEO_WEIGHTS = {
    '2025/2026 - Clausura': 4,
    '2025/2026 - Apertura': 3,
    '2024/2025 - Clausura': 2,
    '2024/2025 - Apertura': 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# MODELO (copiado de 11_modelo_prediccion.py)
# ─────────────────────────────────────────────────────────────────────────────
def _torneo_key(filename: str) -> str:
    name = Path(filename).stem.replace('historico_', '')
    name = name.replace('-', '/', 1)
    parts = name.split('_-_', 1)
    if len(parts) == 2:
        year   = parts[0].replace('_', '/')
        torneo = parts[1].replace('_', ' ').title()
        return f'{year} - {torneo}'
    return name

def load_model_matches():
    rows = []
    for fpath in sorted(HIST_DIR.glob('*.json')):
        tkey = _torneo_key(str(fpath))
        w = TORNEO_WEIGHTS.get(tkey)
        if not w:
            continue
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        for p in data.get('partidos', []):
            if p.get('goles_local') is None or p.get('goles_visit') is None:
                continue
            if not p.get('terminado'):
                continue
            rows.append({
                'local':     norm(p.get('local', '')),
                'visitante': norm(p.get('visitante', '')),
                'gl':        int(p['goles_local']),
                'gv':        int(p['goles_visit']),
                'peso':      w,
            })
    return rows

def build_model(matches):
    from collections import defaultdict
    sum_gl = sum_gv = sum_w = 0.0
    for m in matches:
        sum_gl += m['gl'] * m['peso']
        sum_gv += m['gv'] * m['peso']
        sum_w  += m['peso']
    mu_home = sum_gl / sum_w
    mu_away = sum_gv / sum_w

    att_h = defaultdict(lambda: [0.0, 0.0])
    def_h = defaultdict(lambda: [0.0, 0.0])
    att_a = defaultdict(lambda: [0.0, 0.0])
    def_a = defaultdict(lambda: [0.0, 0.0])

    for m in matches:
        w = m['peso']
        l, v = m['local'], m['visitante']
        att_h[l][0] += m['gl'] * w;  att_h[l][1] += w
        def_h[l][0] += m['gv'] * w;  def_h[l][1] += w
        att_a[v][0] += m['gv'] * w;  att_a[v][1] += w
        def_a[v][0] += m['gl'] * w;  def_a[v][1] += w

    def ratio(d, team, mu):
        if d[team][1] == 0:
            return 1.0
        return (d[team][0] / d[team][1]) / mu

    model = {'mu_home': mu_home, 'mu_away': mu_away,
             'att_h': att_h, 'def_h': def_h,
             'att_a': att_a, 'def_a': def_a}
    return model

def predict(model, local, visitante, max_goals=5):
    mh, ma = model['mu_home'], model['mu_away']
    lam_l = (model['att_h'][local][0]/model['att_h'][local][1] if model['att_h'][local][1] else mh) * \
            (model['def_a'][visitante][0]/model['def_a'][visitante][1] if model['def_a'][visitante][1] else ma) / mh * mh
    lam_v = (model['att_a'][visitante][0]/model['att_a'][visitante][1] if model['att_a'][visitante][1] else ma) * \
            (model['def_h'][local][0]/model['def_h'][local][1] if model['def_h'][local][1] else mh) / ma * ma

    n = max_goals + 1
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matriz[i][j] = poisson.pmf(i, lam_l) * poisson.pmf(j, lam_v)
    p_local  = float(np.sum(np.tril(matriz, -1)))
    p_empate = float(np.trace(matriz))
    p_visit  = float(np.sum(np.triu(matriz, 1)))
    return lam_l, lam_v, matriz, p_local, p_empate, p_visit

# ─────────────────────────────────────────────────────────────────────────────
# ESCUDOS
# ─────────────────────────────────────────────────────────────────────────────
def get_shield(canonical_name, size=72):
    tid = TEAM_IDS.get(canonical_name)
    if tid is None:
        return _blank_shield(size)
    cache = IMG_DIR / f'{tid}.png'
    if not cache.exists():
        url = f'https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as r:
                cache.write_bytes(r.read())
        except Exception:
            return _blank_shield(size)
    try:
        img = Image.open(cache).convert('RGBA')
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return _blank_shield(size)

def _blank_shield(size):
    img = Image.new('RGBA', (size, size), (40, 40, 40, 200))
    return np.array(img)

# ─────────────────────────────────────────────────────────────────────────────
# RENDER RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
def render_jornada(jornada_num, partidos, model, out_path):
    """
    partidos: lista de dicts con 'local' y 'visitante' (nombres canónicos)
    """
    n = len(partidos)

    # Dimensiones: 10 wide, 1.4 por partido + cabecera + footer
    FIG_W   = 10.0
    ROW_H   = 1.55
    HEAD_H  = 1.10
    FOOT_H  = 0.55
    FIG_H   = HEAD_H + n * ROW_H + FOOT_H

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)

    # ── HEADER ───────────────────────────────────────────────────────────────
    head_y  = (FIG_H - HEAD_H) / FIG_H
    head_ax = fig.add_axes([0, head_y, 1, HEAD_H / FIG_H])
    head_ax.set_facecolor(DARK_BG)
    head_ax.axis('off')

    # Título principal
    bp = bebas(36)
    kw = {'fontproperties': bp} if bp else {'fontsize': 36, 'fontweight': 'bold'}
    head_ax.text(0.5, 0.78, f'JORNADA {jornada_num}',
                 ha='center', va='top', color=WHITE, transform=head_ax.transAxes,
                 **kw)
    bp2 = bebas(18)
    kw2 = {'fontproperties': bp2} if bp2 else {'fontsize': 18, 'fontweight': 'bold'}
    head_ax.text(0.5, 0.44, 'PROBABILIDADES DE MARCADOR · LIGA MX · CLAUSURA 2026',
                 ha='center', va='top', color=GRAY, transform=head_ax.transAxes,
                 **kw2)
    # Línea divisoria
    head_ax.axhline(0.0, color=RED_BRAND, linewidth=2, xmin=0.04, xmax=0.96)

    # Precalcular predicciones
    resultados = []
    for p in partidos:
        local_c  = norm(p['local'])
        visit_c  = norm(p['visitante'])
        lam_l, lam_v, _, p_l, p_e, p_v = predict(model, local_c, visit_c)
        resultados.append({
            'local':    local_c,
            'visitante': visit_c,
            'p_local':  p_l,
            'p_empate': p_e,
            'p_visit':  p_v,
            'lam_l':    lam_l,
            'lam_v':    lam_v,
        })

    # Precargar escudos
    shields = {}
    all_teams = set()
    for r in resultados:
        all_teams.add(r['local'])
        all_teams.add(r['visitante'])
    for t in all_teams:
        shields[t] = get_shield(t, size=72)

    # ── FILAS DE PARTIDOS ─────────────────────────────────────────────────────
    for i, r in enumerate(resultados):
        # y en figura coords (de arriba a abajo)
        row_top  = (FIG_H - HEAD_H - i * ROW_H) / FIG_H
        row_h_fc = ROW_H / FIG_H
        row_y    = row_top - row_h_fc   # esquina inferior del axes

        # Fondo alternado sutil
        bg_color = '#111920' if i % 2 == 0 else '#0d1117'
        row_ax = fig.add_axes([0.02, row_y + 0.005, 0.96, row_h_fc - 0.008])
        row_ax.set_facecolor(bg_color)
        row_ax.set_xlim(0, 1)
        row_ax.set_ylim(0, 1)
        row_ax.axis('off')

        # Borde izquierdo de color del equipo local
        local_color = TEAM_COLORS.get(r['local'], '#888888')
        visit_color = TEAM_COLORS.get(r['visitante'], '#888888')
        row_ax.axvline(0.0, color=local_color, linewidth=4)

        # ── Escudo local (izquierda) ──────────────────────────────────────────
        sh_l = shields[r['local']]
        sh_ax_l = fig.add_axes([0.04, row_y + row_h_fc * 0.12, 0.10, row_h_fc * 0.76])
        sh_ax_l.imshow(sh_l, aspect='equal')
        sh_ax_l.axis('off')

        # ── Nombre equipo local ───────────────────────────────────────────────
        bp3 = bebas(13)
        kw3 = {'fontproperties': bp3} if bp3 else {'fontsize': 13, 'fontweight': 'bold'}
        row_ax.text(0.185, 0.72, r['local'].upper(),
                    ha='center', va='center', color=WHITE,
                    transform=row_ax.transAxes, **kw3)

        # ── Barras de probabilidades (centro) ────────────────────────────────
        BAR_LEFT  = 0.26
        BAR_RIGHT = 0.74
        BAR_W     = BAR_RIGHT - BAR_LEFT
        BAR_Y     = 0.50
        BAR_H     = 0.22

        # Fondo gris
        bar_bg = mpatches.FancyBboxPatch(
            (BAR_LEFT, BAR_Y - BAR_H / 2), BAR_W, BAR_H,
            boxstyle='round,pad=0.005', linewidth=0,
            facecolor='#1c2128', transform=row_ax.transAxes, zorder=2
        )
        row_ax.add_patch(bar_bg)

        # Segmentos: local | empate | visitante
        segs = [
            (r['p_local'],  local_color,  'LOCAL'),
            (r['p_empate'], '#555e6a',     'EMPATE'),
            (r['p_visit'],  visit_color,   'VISIT'),
        ]
        x_cursor = BAR_LEFT
        seg_bounds = []
        for pct, color, _ in segs:
            w = BAR_W * pct
            rect = mpatches.FancyBboxPatch(
                (x_cursor, BAR_Y - BAR_H / 2), w, BAR_H,
                boxstyle='square,pad=0', linewidth=0,
                facecolor=color, alpha=0.9,
                transform=row_ax.transAxes, zorder=3
            )
            row_ax.add_patch(rect)
            seg_bounds.append((x_cursor, x_cursor + w, pct))
            x_cursor += w

        # Porcentajes sobre las barras
        labels_data = [
            (seg_bounds[0], r['p_local'],  local_color,  r['local']),
            (seg_bounds[1], r['p_empate'], '#aaaaaa',    'Empate'),
            (seg_bounds[2], r['p_visit'],  visit_color,  r['visitante']),
        ]
        for (x0, x1, _), pct, color, label in labels_data:
            cx = (x0 + x1) / 2
            bp_sm = bebas(14)
            kw_sm = {'fontproperties': bp_sm} if bp_sm else {'fontsize': 14, 'fontweight': 'bold'}
            row_ax.text(cx, BAR_Y + BAR_H / 2 + 0.15, f'{pct*100:.1f}%',
                        ha='center', va='bottom', color=WHITE,
                        transform=row_ax.transAxes, **kw_sm)
            # Etiqueta pequeña debajo
            bp_xs = bebas(8)
            kw_xs = {'fontproperties': bp_xs} if bp_xs else {'fontsize': 8}
            row_ax.text(cx, BAR_Y - BAR_H / 2 - 0.12, label.upper(),
                        ha='center', va='top', color=GRAY,
                        transform=row_ax.transAxes, **kw_xs)

        # λ valores debajo de todo
        bp_lam = bebas(9)
        kw_lam = {'fontproperties': bp_lam} if bp_lam else {'fontsize': 9}
        row_ax.text(0.50, 0.04,
                    f'λ local={r["lam_l"]:.2f}   λ visit={r["lam_v"]:.2f}   · Modelo Poisson 4 torneos',
                    ha='center', va='bottom', color='#4a5568',
                    transform=row_ax.transAxes, **kw_lam)

        # ── Escudo visitante (derecha) ────────────────────────────────────────
        sh_v = shields[r['visitante']]
        sh_ax_v = fig.add_axes([0.86, row_y + row_h_fc * 0.12, 0.10, row_h_fc * 0.76])
        sh_ax_v.imshow(sh_v, aspect='equal')
        sh_ax_v.axis('off')

        # ── Nombre equipo visitante ───────────────────────────────────────────
        row_ax.text(0.815, 0.72, r['visitante'].upper(),
                    ha='center', va='center', color=WHITE,
                    transform=row_ax.transAxes, **kw3)

        # Borde derecho visitante
        row_ax.axvline(1.0, color=visit_color, linewidth=4)

        # Línea separadora inferior
        row_ax.axhline(0.0, color='#21262d', linewidth=1)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    foot_ax = fig.add_axes([0, 0, 1, FOOT_H / FIG_H])
    foot_ax.set_facecolor(DARK_BG)
    foot_ax.axis('off')

    # MAU-STATISTICS
    bp_mau = bebas(24)
    kw_mau = {'fontproperties': bp_mau} if bp_mau else {'fontsize': 24, 'fontweight': 'bold'}
    # Sombra
    foot_ax.text(0.985 + 0.002, 0.55 - 0.002, 'MAU-STATISTICS',
                 ha='right', va='center', color='#000000', alpha=0.6,
                 transform=foot_ax.transAxes, **kw_mau)
    foot_ax.text(0.985, 0.55, 'MAU-STATISTICS',
                 ha='right', va='center', color=RED_BRAND,
                 transform=foot_ax.transAxes, **kw_mau)

    bp_src = bebas(11)
    kw_src = {'fontproperties': bp_src} if bp_src else {'fontsize': 11}
    foot_ax.text(0.015, 0.55, 'Fuente: FotMob  ·  Modelo Poisson ponderado',
                 ha='left', va='center', color=GRAY,
                 transform=foot_ax.transAxes, **kw_src)

    foot_ax.axhline(1.0, color=RED_BRAND, linewidth=1.5, xmin=0.02, xmax=0.98)

    plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f'✓ Guardado: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    jornada_num = int(sys.argv[1]) if len(sys.argv) > 1 else 13

    # Cargar partidos de la jornada desde el archivo Clausura 2026
    clausura_file = HIST_DIR / 'historico_2025-2026_-_clausura.json'
    with open(clausura_file, encoding='utf-8') as f:
        data = json.load(f)

    partidos_raw = [p for p in data.get('partidos', [])
                    if str(p.get('jornada')) == str(jornada_num)]

    if not partidos_raw:
        print(f'⚠ No se encontraron partidos para la jornada {jornada_num}')
        sys.exit(1)

    partidos = [{'local': p['local'], 'visitante': p['visitante']}
                for p in partidos_raw]

    print(f'Jornada {jornada_num}: {len(partidos)} partidos')
    for p in partidos:
        print(f'  {p["local"]} vs {p["visitante"]}')

    # Construir modelo
    print('\nCargando modelo Poisson...')
    matches = load_model_matches()
    print(f'  {len(matches)} partidos ponderados')
    model = build_model(matches)
    print(f'  μ_home={model["mu_home"]:.3f}  μ_away={model["mu_away"]:.3f}')

    # Output
    out_dir = OUT_BASE / f'jornada{jornada_num}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'resumen_jornada{jornada_num}.png'

    print('\nGenerando imagen resumen...')
    render_jornada(jornada_num, partidos, model, out_path)


if __name__ == '__main__':
    main()
