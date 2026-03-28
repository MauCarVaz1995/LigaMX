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
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFilter
from scipy.stats import poisson

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETAS, PALETA_ACTIVA, get_paleta

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
_pal      = get_paleta()
DARK_BG   = _pal['bg_primary']
WHITE     = _pal['text_primary']
GRAY      = _pal['text_secondary']
RED_BRAND = _pal['accent']
GREEN_WIN = _pal['cell_high']
YELLOW_DW = _pal['accent2']

TEAM_COLORS = {
    'América':       '#FFD700',
    'Atlas':         '#B22222',
    'Chivas':        '#CD1F2D',
    'Cruz Azul':     '#0047AB',
    'FC Juárez':     '#4CAF50',
    'León':          '#2D8C3C',
    'Mazatlán':      '#5B2C8F',
    'Monterrey':     '#003DA5',
    'Necaxa':        '#D62828',
    'Pachuca':       '#1E3A5F',
    'Puebla':        '#2563EB',
    'Pumas':         '#1C2C5B',
    'Querétaro':     '#1565C0',
    'San Luis':      '#D52B1E',
    'Santos Laguna': '#2E8B57',
    'Tigres':        '#F5A623',
    'Tijuana':       '#C62828',
    'Toluca':        '#D5001C',
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
    # Normalizar: el truncamiento en max_goals hace que la suma < 1
    total = float(matriz.sum())
    if total > 0:
        matriz /= total
    p_local  = float(np.sum(np.tril(matriz, -1)))
    p_empate = float(np.trace(matriz))
    p_visit  = float(np.sum(np.triu(matriz, 1)))
    # Marcador más probable
    idx = np.unravel_index(np.argmax(matriz), matriz.shape)
    best_gl, best_gv = int(idx[0]), int(idx[1])
    best_prob = float(matriz[idx])
    return lam_l, lam_v, matriz, p_local, p_empate, p_visit, best_gl, best_gv, best_prob

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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _text_color(hex_bg):
    """Negro para fondos claros (amarillo/verde), blanco para oscuros."""
    r = int(hex_bg[1:3], 16)
    g = int(hex_bg[3:5], 16)
    b = int(hex_bg[5:7], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return '#111111' if lum > 160 else '#ffffff'

def _hex_to_rgba(hex_color, alpha=1.0):
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return np.array([r, g, b, alpha])

def get_shield_framed(canonical_name, size=88, border=5):
    """Escudo con borde redondeado del color del equipo, fondo oscuro interior."""
    inner = size - border * 2
    shield = get_shield(canonical_name, size=inner)
    team_hex = TEAM_COLORS.get(canonical_name, '#444444')
    r, g, b = int(team_hex[1:3],16), int(team_hex[3:5],16), int(team_hex[5:7],16)

    # Canvas transparente
    result = Image.new('RGBA', (size, size), (0, 0, 0, 0))

    # Fondo exterior de color equipo (rounded)
    bg = Image.new('RGBA', (size, size), (r, g, b, 210))
    mask_o = Image.new('L', (size, size), 0)
    draw_o = ImageDraw.Draw(mask_o)
    rad = size // 5
    try:
        draw_o.rounded_rectangle([0, 0, size-1, size-1], radius=rad, fill=255)
    except AttributeError:
        draw_o.rectangle([0, 0, size-1, size-1], fill=255)
    bg.putalpha(mask_o)
    result.paste(bg, (0, 0), bg)

    # Fondo interior blanco (los escudos de FotMob están diseñados para fondo claro)
    inner_bg = Image.new('RGBA', (inner, inner), (248, 248, 252, 255))
    mask_i = Image.new('L', (inner, inner), 0)
    draw_i = ImageDraw.Draw(mask_i)
    rad_i = inner // 5
    try:
        draw_i.rounded_rectangle([0, 0, inner-1, inner-1], radius=rad_i, fill=255)
    except AttributeError:
        draw_i.rectangle([0, 0, inner-1, inner-1], fill=255)
    inner_bg.putalpha(mask_i)
    result.paste(inner_bg, (border, border), inner_bg)

    # Pegar escudo encima
    shield_img = Image.fromarray(shield).convert('RGBA').resize((inner, inner), Image.LANCZOS)
    result.paste(shield_img, (border, border), shield_img)
    return np.array(result)

def _draw_gradient_separator(fig, x, y, w, h, color_l, color_r, alpha=0.85):
    """Dibuja una tira horizontal con gradiente de color_l → color_r."""
    grad = np.zeros((1, 200, 4))
    rgba_l = _hex_to_rgba(color_l, alpha)
    rgba_r = _hex_to_rgba(color_r, alpha)
    for k in range(200):
        t = k / 199
        grad[0, k] = (1 - t) * rgba_l + t * rgba_r
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(grad, aspect='auto', extent=[0, 1, 0, 1])
    ax.axis('off')

# ─────────────────────────────────────────────────────────────────────────────
# RENDER RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
def render_jornada(jornada_num, partidos, model, out_path):
    n = len(partidos)

    FIG_W  = 10.0
    ROW_H  = 1.50
    HEAD_H = 1.10
    FOOT_H = 0.55
    FIG_H  = HEAD_H + n * ROW_H + FOOT_H

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)

    # ── HEADER ───────────────────────────────────────────────────────────────
    head_y  = (FIG_H - HEAD_H) / FIG_H
    head_ax = fig.add_axes([0, head_y, 1, HEAD_H / FIG_H])
    head_ax.set_facecolor(DARK_BG)
    head_ax.axis('off')

    bp_t = bebas(38)
    kw_t = {'fontproperties': bp_t} if bp_t else {'fontsize': 38, 'fontweight': 'bold'}
    head_ax.text(0.5, 0.80, f'JORNADA {jornada_num}',
                 ha='center', va='top', color=WHITE,
                 transform=head_ax.transAxes, **kw_t)
    bp_s = bebas(17)
    kw_s = {'fontproperties': bp_s} if bp_s else {'fontsize': 17, 'fontweight': 'bold'}
    head_ax.text(0.5, 0.44, 'PROBABILIDADES DE MARCADOR · LIGA MX · CLAUSURA 2026',
                 ha='center', va='top', color=GRAY,
                 transform=head_ax.transAxes, **kw_s)
    head_ax.axhline(0.0, color=RED_BRAND, linewidth=2.5, xmin=0.03, xmax=0.97)

    # Precalcular predicciones
    resultados = []
    for p in partidos:
        local_c = norm(p['local'])
        visit_c = norm(p['visitante'])
        _, _, _, p_l, p_e, p_v, best_gl, best_gv, best_prob = predict(model, local_c, visit_c)
        resultados.append({
            'local':     local_c,
            'visitante': visit_c,
            'p_local':   p_l,
            'p_empate':  p_e,
            'p_visit':   p_v,
            'best_gl':   best_gl,
            'best_gv':   best_gv,
            'best_prob': best_prob,
        })

    # Precargar escudos con borde
    shields = {}
    all_teams = set(t for r in resultados for t in (r['local'], r['visitante']))
    for t in all_teams:
        shields[t] = get_shield_framed(t, size=88, border=5)

    # ── FILAS ────────────────────────────────────────────────────────────────
    for i, r in enumerate(resultados):
        row_top  = (FIG_H - HEAD_H - i * ROW_H) / FIG_H
        row_h_fc = ROW_H / FIG_H
        row_y    = row_top - row_h_fc

        lc = TEAM_COLORS.get(r['local'], '#888888')
        vc = TEAM_COLORS.get(r['visitante'], '#888888')

        # Fondo de fila alternado
        bg = '#111920' if i % 2 == 0 else '#0d1117'
        row_ax = fig.add_axes([0.02, row_y + 0.004, 0.96, row_h_fc - 0.010])
        row_ax.set_facecolor(bg)
        row_ax.set_xlim(0, 1)
        row_ax.set_ylim(0, 1)
        row_ax.axis('off')

        # Wash sutil de color del equipo en los extremos de la fila
        row_ax.add_patch(mpatches.Rectangle(
            (0, 0), 0.23, 1, transform=row_ax.transAxes,
            facecolor=lc, alpha=0.07, linewidth=0, zorder=0))
        row_ax.add_patch(mpatches.Rectangle(
            (0.77, 0), 0.23, 1, transform=row_ax.transAxes,
            facecolor=vc, alpha=0.07, linewidth=0, zorder=0))

        # Bordes laterales — 2px, color equipo local (izq) y visitante (der)
        row_ax.axvline(0.002, color=lc, linewidth=2, zorder=5)
        row_ax.axvline(0.998, color=vc, linewidth=2, zorder=5)

        # ── Escudo local ──────────────────────────────────────────────────────
        SH_W = 0.105; SH_H_FC = 0.82
        sh_x_l = 0.035
        sh_y_l = row_y + row_h_fc * ((1 - SH_H_FC) / 2)
        sh_h_l = row_h_fc * SH_H_FC
        sh_ax_l = fig.add_axes([sh_x_l, sh_y_l, SH_W, sh_h_l])
        sh_ax_l.imshow(shields[r['local']], aspect='equal')
        sh_ax_l.axis('off')

        # Nombre local
        bp_nm = bebas(11)
        kw_nm = {'fontproperties': bp_nm} if bp_nm else {'fontsize': 11, 'fontweight': 'bold'}
        row_ax.text(0.185, 0.18, r['local'].upper(),
                    ha='center', va='center', color=WHITE,
                    transform=row_ax.transAxes, **kw_nm)

        # ── Barras redondeadas ────────────────────────────────────────────────
        BAR_L = 0.265; BAR_R = 0.735
        BAR_W = BAR_R - BAR_L
        BAR_CY = 0.60
        BAR_H  = 0.32

        # Determinar favorito (max prob entre las tres)
        probs  = [r['p_local'], r['p_empate'], r['p_visit']]
        fav_idx = int(np.argmax(probs))

        # Clip path redondeado para toda la barra
        clip = mpatches.FancyBboxPatch(
            (BAR_L, BAR_CY - BAR_H / 2), BAR_W, BAR_H,
            boxstyle='round,pad=0.003',
            transform=row_ax.transAxes,
            linewidth=0, facecolor='none', zorder=3
        )
        row_ax.add_patch(clip)

        segs = [
            (r['p_local'],  lc),
            (r['p_empate'], '#666666'),
            (r['p_visit'],  vc),
        ]
        x_cur = BAR_L
        seg_info = []
        for idx_s, (pct, color) in enumerate(segs):
            seg_w = BAR_W * pct
            rect = mpatches.Rectangle(
                (x_cur, BAR_CY - BAR_H / 2), seg_w, BAR_H,
                transform=row_ax.transAxes,
                linewidth=0, facecolor=color, alpha=0.95, zorder=3
            )
            rect.set_clip_path(clip)
            row_ax.add_patch(rect)
            seg_info.append((x_cur + seg_w / 2, pct, color, seg_w, idx_s))
            x_cur += seg_w

        # Borde fino sobre la barra (da profundidad)
        row_ax.add_patch(mpatches.FancyBboxPatch(
            (BAR_L, BAR_CY - BAR_H / 2), BAR_W, BAR_H,
            boxstyle='round,pad=0.003',
            transform=row_ax.transAxes,
            linewidth=0.8, edgecolor='#ffffff22', facecolor='none', zorder=5
        ))

        # Porcentajes DENTRO de las barras
        for cx, pct, color_seg, seg_w, idx_s in seg_info:
            if seg_w < 0.042:
                continue
            txt_col = _text_color(color_seg)
            is_fav  = (idx_s == fav_idx)
            # Favorito: fuente mayor y sin transparencia
            size_pct = 16 if is_fav else 12
            bp_p = bebas(size_pct)
            kw_p = {'fontproperties': bp_p} if bp_p else {'fontsize': size_pct, 'fontweight': 'bold'}
            row_ax.text(cx, BAR_CY, f'{pct*100:.1f}%',
                        ha='center', va='center', color=txt_col,
                        transform=row_ax.transAxes, zorder=6, **kw_p)

        # Marcador más probable
        bp_ms = bebas(10)
        kw_ms = {'fontproperties': bp_ms} if bp_ms else {'fontsize': 10}
        row_ax.text(
            (BAR_L + BAR_R) / 2, BAR_CY - BAR_H / 2 - 0.09,
            f'Marcador más probable:  {r["best_gl"]}-{r["best_gv"]}  ({r["best_prob"]*100:.1f}%)',
            ha='center', va='top', color='#6e7681',
            transform=row_ax.transAxes, **kw_ms
        )

        # ── Escudo visitante ──────────────────────────────────────────────────
        sh_x_v = 0.860
        sh_ax_v = fig.add_axes([sh_x_v, sh_y_l, SH_W, sh_h_l])
        sh_ax_v.imshow(shields[r['visitante']], aspect='equal')
        sh_ax_v.axis('off')

        # Nombre visitante
        row_ax.text(0.815, 0.18, r['visitante'].upper(),
                    ha='center', va='center', color=WHITE,
                    transform=row_ax.transAxes, **kw_nm)

        # ── Separador con gradiente ───────────────────────────────────────────
        sep_h = 0.0025
        _draw_gradient_separator(
            fig, 0.02, row_y + 0.002, 0.96, sep_h, lc, vc, alpha=0.65
        )

    # ── FOOTER ───────────────────────────────────────────────────────────────
    foot_ax = fig.add_axes([0, 0, 1, FOOT_H / FIG_H])
    foot_ax.set_facecolor(DARK_BG)
    foot_ax.axis('off')

    bp_mau = bebas(24)
    kw_mau = {'fontproperties': bp_mau} if bp_mau else {'fontsize': 24, 'fontweight': 'bold'}
    foot_ax.text(0.985 + 0.002, 0.55 - 0.002, 'MAU-STATISTICS',
                 ha='right', va='center', color='#000000', alpha=0.6,
                 transform=foot_ax.transAxes, **kw_mau)
    foot_ax.text(0.985, 0.55, 'MAU-STATISTICS',
                 ha='right', va='center', color=RED_BRAND,
                 transform=foot_ax.transAxes, **kw_mau)

    bp_src = bebas(11)
    kw_src = {'fontproperties': bp_src} if bp_src else {'fontsize': 11}
    foot_ax.text(0.015, 0.55, 'Fuente: FotMob  ·  Modelo Poisson ponderado (últimos 4 torneos)',
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
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument('--paleta', default=None, choices=list(PALETAS.keys()))
    _args, _rest = _parser.parse_known_args()
    if _args.paleta:
        _p = get_paleta(_args.paleta)
        DARK_BG   = _p['bg_primary']
        WHITE     = _p['text_primary']
        GRAY      = _p['text_secondary']
        RED_BRAND = _p['accent']
        GREEN_WIN = _p['cell_high']
        YELLOW_DW = _p['accent2']
        _pal      = _p
    main()
