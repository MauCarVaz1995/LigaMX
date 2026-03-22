#!/usr/bin/env python3
"""
08_comparativo_1v1.py
Infografía comparativa 1v1 – Liga MX Clausura 2026
Uso: python 08_comparativo_1v1.py <id_jugador1> <id_jugador2>
Ejemplo: python 08_comparativo_1v1.py 361377 215428
"""

import sys
import json
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFilter
import urllib.request

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
CSV_PATH  = BASE / 'data/processed/jugadores_clausura2026.csv'
STATS_DIR = BASE / 'data/raw/stats_detalladas'
IMG_PLAY  = BASE / 'data/raw/images/players'
IMG_TEAMS = BASE / 'data/raw/images/teams'
OUT_DIR   = BASE / 'output/charts'
BEBAS_TTF = Path.home() / '.fonts/BebasNeue.ttf'

for d in (IMG_PLAY, IMG_TEAMS, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Bebas Neue ───────────────────────────────────────────────────────────────
_bebas = None
if BEBAS_TTF.exists():
    fm.fontManager.addfont(str(BEBAS_TTF))
    _bebas = FontProperties(fname=str(BEBAS_TTF))

def bebas(size):
    if _bebas:
        p = FontProperties(fname=str(BEBAS_TTF))
        p.set_size(size)
        return {'fontproperties': p}
    return {'fontsize': size, 'fontweight': 'bold'}

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
WHITE     = '#e6edf3'
GRAY      = '#8b949e'
RED_BRAND = '#D5001C'
GOLD      = '#FFD700'

# Colores por equipo_id (tono principal)
TEAM_COLORS = {
    6618:    '#D5001C',  # Toluca
    7807:    '#CE1141',  # Chivas
    8561:    '#FF7900',  # Tigres
    6576:    '#FFD700',  # América
    6578:    '#0038A8',  # Cruz Azul
    7849:    '#003F8A',  # Monterrey
    1946:    '#002B5B',  # Pumas
    7857:    '#006341',  # Santos Laguna
    7848:    '#004D8E',  # Pachuca
    7847:    '#00479D',  # Puebla
    1841:    '#006633',  # León
    1842:    '#CC0000',  # Necaxa
    162418:  '#1A1A1A',  # Tijuana
    6577:    '#CC2200',  # Atlas
    1943:    '#0033A0',  # Querétaro
    649424:  '#C8102E',  # FC Juárez
    1170234: '#D47900',  # Mazatlán
    6358:    '#006633',  # San Luis
}

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS POR POSICIÓN
# ─────────────────────────────────────────────────────────────────────────────
_POS_CAT = {
    'GK':  'Portero',
    'CB':  'Defensa',  'LB': 'Defensa',  'RB': 'Defensa',
    'LWB': 'Defensa',  'RWB': 'Defensa',
    'CDM': 'Mediocampista', 'CM':  'Mediocampista',
    'CAM': 'Mediocampista', 'LM':  'Mediocampista', 'RM': 'Mediocampista',
    'LW':  'Delantero', 'RW': 'Delantero', 'ST': 'Delantero',
}

METRICS_BY_POS = {
    'Delantero': [
        ('goles_p90',                    'Goles P90'),
        ('xG_p90',                       'xG P90'),
        ('tiros_p90',                    'Tiros P90'),
        ('tiros_a_puerta_p90',           'Tiros a Puerta P90'),
        ('asistencias_p90',              'Asistencias P90'),
        ('xA_p90',                       'xA P90'),
        ('chances_creadas_p90',          'Chances Creadas P90'),
        ('pases_precisos_p90',           'Pases Precisos P90'),
        ('duelos_tierra_ganados_p90',    'Duelos Ganados P90'),
    ],
    'Mediocampista': [
        ('pases_precisos_p90',           'Pases Precisos P90'),
        ('pases_largos_p90',             'Pases Progresivos P90'),
        ('asistencias_p90',              'Asistencias P90'),
        ('xA_p90',                       'xA P90'),
        ('chances_creadas_p90',          'Chances Creadas P90'),
        ('recuperaciones_campo_rival_p90','Recuperaciones P90'),
        ('duelos_tierra_ganados_p90',    'Duelos Ganados P90'),
        ('entradas_p90',                 'Entradas P90'),
        ('intercepciones_p90',           'Intercepciones P90'),
    ],
    'Defensa': [
        ('intercepciones_p90',           'Intercepciones P90'),
        ('duelos_tierra_ganados_p90',    'Duelos Ganados P90'),
        ('recuperaciones_campo_rival_p90','Recuperaciones P90'),
        ('pases_precisos_p90',           'Pases Precisos P90'),
        ('pases_largos_p90',             'Pases Largos P90'),
        ('despejes_p90',                 'Despejes P90'),
        ('entradas_p90',                 'Entradas P90'),
        ('faltas_cometidas_p90',         'Faltas P90'),
        ('tiros_bloqueados_p90',         'Tiros Bloqueados P90'),
    ],
    'Portero': [
        ('paradas_p90',                  'Paradas P90'),
        ('porcentaje_paradas_p90',       '% Paradas P90'),
        ('porterias_cero_p90',           'Porterías Cero P90'),
        ('goles_recibidos_p90',          'Goles Concedidos P90'),
        ('goles_evitados_p90',           'Goles Evitados P90'),
        ('pases_precisos_p90',           'Pases Precisos P90'),
    ],
}

def classify(pos_str):
    if pd.isna(pos_str):
        return 'Delantero'
    primary = str(pos_str).split(',')[0].strip()
    return _POS_CAT.get(primary, 'Mediocampista')

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df['categoria'] = df['posicion'].apply(classify)
    return df

def load_stats_detalladas(player_id: int, equipo_id: int) -> dict:
    """Busca el jugador en los JSON de stats_detalladas de su equipo."""
    pattern = STATS_DIR / f'{equipo_id}_*.json'
    files = glob.glob(str(pattern))
    for fpath in files:
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        for j in data.get('jugadores', []):
            if j.get('id') == player_id:
                return j
    return {}

def get_player_data(player_id: int, df: pd.DataFrame):
    row = df[df['id'] == player_id]
    if row.empty:
        raise ValueError(f'Jugador ID {player_id} no encontrado en el CSV.')
    row = row.iloc[0]
    stats = load_stats_detalladas(int(player_id), int(row['equipo_id']))
    # Extraer tarjetas de stats detalladas
    disc = stats.get('grupos', {}).get('discipline', {}).get('stats', {})
    yellow = int(disc.get('yellow_cards', {}).get('value', 0) or 0)
    red    = int(disc.get('red_cards',   {}).get('value', 0) or 0)
    return row, yellow, red

# ─────────────────────────────────────────────────────────────────────────────
# PERCENTILES (dentro de toda la liga, misma categoría)
# ─────────────────────────────────────────────────────────────────────────────
def compute_percentiles(df: pd.DataFrame, cat: str, cols: list) -> pd.DataFrame:
    """Devuelve percentiles 0-100 por columna para la categoría dada."""
    sub = df[(df['categoria'] == cat) & (df['minutos_stats'] >= 200)].copy()
    pct = pd.DataFrame(index=sub.index)
    for c in cols:
        if c in sub.columns:
            pct[c] = sub[c].fillna(0).rank(pct=True) * 100
        else:
            pct[c] = 0.0
    pct['id'] = sub['id'].values
    return pct

# ─────────────────────────────────────────────────────────────────────────────
# IMÁGENES
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_PLAYER = 'https://images.fotmob.com/image_resources/playerimages/{}.png'
FOTMOB_TEAM   = 'https://images.fotmob.com/image_resources/logo/teamlogo/{}.png'
_HEADERS = {'User-Agent': 'Mozilla/5.0'}

def _fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=6) as r:
            dest.write_bytes(r.read())
        return True
    except Exception:
        return False

def get_player_img(pid):
    dest = IMG_PLAY / f'{pid}.png'
    if not _fetch(FOTMOB_PLAYER.format(pid), dest):
        return None
    try:
        return Image.open(dest).convert('RGBA')
    except Exception:
        return None

def get_team_img(tid):
    dest = IMG_TEAMS / f'{tid}.png'
    if not _fetch(FOTMOB_TEAM.format(tid), dest):
        return None
    try:
        return Image.open(dest).convert('RGBA')
    except Exception:
        return None

def circular_crop(img: Image.Image, size=100) -> np.ndarray:
    img = img.convert('RGBA').resize((size, size), Image.LANCZOS)
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size-1, size-1), fill=255)
    out = Image.new('RGBA', (size, size), (0,0,0,0))
    out.paste(img, mask=mask)
    return np.array(out)

def placeholder_player(size=100, color=(55,62,75)) -> np.ndarray:
    img = Image.new('RGBA', (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((0,0,size-1,size-1), fill=(*color,255))
    draw.ellipse((size//4, size//8, size*3//4, size//2), fill=(95,102,115,200))
    draw.ellipse((size//8, size*9//16, size*7//8, size-1), fill=(95,102,115,200))
    return np.array(img)

def hex_to_rgba_dark(hex_color: str, alpha=0.18) -> tuple:
    r, g, b = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255 for i in (0,2,4))
    return (r, g, b, alpha)

# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
FIG_W  = 14
FIG_H  = 11
DPI    = 150

# Zonas (coordenadas figura)
HEADER_H  = 0.24   # encabezado con fotos y datos generales
FOOTER_H  = 0.065
STATS_Y0  = FOOTER_H
STATS_H   = 1.0 - HEADER_H - FOOTER_H  # zona de barras

# Columnas
LEFT_PHOTO_X   = 0.03
CENTER_X_L     = 0.38   # borde izquierdo de la columna central (etiquetas)
CENTER_X_R     = 0.62   # borde derecho de la columna central
BAR_MAX_W      = 0.34   # max ancho de barra en coordenadas figura
RIGHT_PHOTO_X  = 0.93


def draw_card_icons(ax, yellow, red, x_start, y, direction='left'):
    """Dibuja íconos de tarjetas como rectángulos pequeños."""
    card_w, card_h = 0.022, 0.038
    gap = 0.007
    items = []
    if yellow > 0:
        items.append(('#F5C518', str(yellow)))
    if red > 0:
        items.append(('#D5001C', str(red)))

    for color, count in items:
        if direction == 'left':
            rect = mpatches.FancyBboxPatch(
                (x_start, y - card_h/2), card_w, card_h,
                boxstyle='round,pad=0.002', color=color,
                transform=ax.transAxes, zorder=5
            )
            ax.add_patch(rect)
            ax.text(x_start + card_w/2, y, count,
                    color='white', fontsize=7, fontweight='bold',
                    ha='center', va='center', transform=ax.transAxes, zorder=6)
            x_start += card_w + gap
        else:
            x_start -= card_w
            rect = mpatches.FancyBboxPatch(
                (x_start, y - card_h/2), card_w, card_h,
                boxstyle='round,pad=0.002', color=color,
                transform=ax.transAxes, zorder=5
            )
            ax.add_patch(rect)
            ax.text(x_start + card_w/2, y, count,
                    color='white', fontsize=7, fontweight='bold',
                    ha='center', va='center', transform=ax.transAxes, zorder=6)
            x_start -= gap


def rating_color(r):
    if r is None or pd.isna(r):
        return GRAY
    if r >= 7.5:
        return '#00C87A'
    if r >= 7.0:
        return '#F5A623'
    return '#D5001C'


def render_1v1(p1_id: int, p2_id: int):
    df   = load_csv()
    r1, y1, rc1 = get_player_data(p1_id, df)
    r2, y2, rc2 = get_player_data(p2_id, df)

    cat1 = r1['categoria']
    cat2 = r2['categoria']
    # Usar la categoría del primero como base; si difieren, mostrar métricas comunes
    cat  = cat1 if cat1 == cat2 else 'Mediocampista'
    metrics = METRICS_BY_POS.get(cat, METRICS_BY_POS['Delantero'])
    cols = [m[0] for m in metrics]

    # Percentiles
    pct_df = compute_percentiles(df, cat, cols)
    def get_pct(pid, col):
        row = pct_df[pct_df['id'] == pid]
        if row.empty or col not in pct_df.columns:
            return 0.0
        return float(row.iloc[0][col])

    c1 = TEAM_COLORS.get(int(r1['equipo_id']), '#444444')
    c2 = TEAM_COLORS.get(int(r2['equipo_id']), '#444444')

    # ── FIGURA ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)

    # ── FONDO CON TONO DE EQUIPO ─────────────────────────────────────────
    bg_ax = fig.add_axes([0, 0, 1, 1])
    bg_ax.set_xlim(0, 1); bg_ax.set_ylim(0, 1); bg_ax.axis('off')
    bg_ax.add_patch(mpatches.Rectangle((0, 0), 0.5, 1,
                    color=hex_to_rgba_dark(c1, 0.22), transform=bg_ax.transAxes))
    bg_ax.add_patch(mpatches.Rectangle((0.5, 0), 0.5, 1,
                    color=hex_to_rgba_dark(c2, 0.22), transform=bg_ax.transAxes))
    # Línea central
    bg_ax.plot([0.5, 0.5], [FOOTER_H, 1.0], color='#2d333b', lw=1.5,
               transform=bg_ax.transAxes)

    # ── HEADER ───────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_xlim(0, 1); hax.set_ylim(0, 1); hax.axis('off')
    hax.set_facecolor('#0d1117')

    # Fondo header tono de equipo
    hax.add_patch(mpatches.Rectangle((0, 0), 0.5, 1,
                  color=hex_to_rgba_dark(c1, 0.30), transform=hax.transAxes))
    hax.add_patch(mpatches.Rectangle((0.5, 0), 0.5, 1,
                  color=hex_to_rgba_dark(c2, 0.30), transform=hax.transAxes))

    # Línea superior de color de equipo
    hax.add_patch(mpatches.Rectangle((0, 0.96), 0.5, 0.04,
                  color=c1, transform=hax.transAxes))
    hax.add_patch(mpatches.Rectangle((0.5, 0.96), 0.5, 0.04,
                  color=c2, transform=hax.transAxes))

    # Título central
    hax.text(0.5, 0.90, 'VS',
             color=WHITE, **bebas(28),
             va='center', ha='center', transform=hax.transAxes, zorder=5)
    hax.text(0.5, 0.75, f'Liga MX · Clausura 2026',
             color=GRAY, fontsize=8,
             va='center', ha='center', transform=hax.transAxes)

    # ── FOTO JUGADOR 1 (izquierda) ───────────────────────────────────────
    pimg1 = get_player_img(p1_id)
    arr1  = circular_crop(pimg1, 90) if pimg1 else placeholder_player(90, color=(60,20,20))
    photo_ax1 = fig.add_axes([0.04, 1 - HEADER_H + 0.01, 0.09, HEADER_H * 0.80])
    photo_ax1.imshow(arr1); photo_ax1.axis('off')

    # ── FOTO JUGADOR 2 (derecha) ──────────────────────────────────────────
    pimg2 = get_player_img(p2_id)
    arr2  = circular_crop(pimg2, 90) if pimg2 else placeholder_player(90, color=(20,20,60))
    photo_ax2 = fig.add_axes([0.87, 1 - HEADER_H + 0.01, 0.09, HEADER_H * 0.80])
    photo_ax2.imshow(arr2); photo_ax2.axis('off')

    # ── ESCUDO EQUIPO 1 ──────────────────────────────────────────────────
    timg1 = get_team_img(int(r1['equipo_id']))
    if timg1:
        t1arr = np.array(timg1.convert('RGBA').resize((45,45), Image.LANCZOS))
        tax1 = fig.add_axes([0.145, 1 - HEADER_H + HEADER_H*0.52, 0.04, HEADER_H*0.38])
        tax1.imshow(t1arr); tax1.axis('off')

    # ── ESCUDO EQUIPO 2 ──────────────────────────────────────────────────
    timg2 = get_team_img(int(r2['equipo_id']))
    if timg2:
        t2arr = np.array(timg2.convert('RGBA').resize((45,45), Image.LANCZOS))
        tax2 = fig.add_axes([0.815, 1 - HEADER_H + HEADER_H*0.52, 0.04, HEADER_H*0.38])
        tax2.imshow(t2arr); tax2.axis('off')

    # ── TEXTOS JUGADOR 1 ─────────────────────────────────────────────────
    hax.text(0.15, 0.65, r1['nombre'].upper(),
             color=WHITE, **bebas(20),
             va='center', ha='left', transform=hax.transAxes)
    hax.text(0.15, 0.48, f"{r1['equipo']}  ·  {r1['posicion']}  ·  {int(r1['edad'])} años",
             color=GRAY, fontsize=8.5,
             va='center', ha='left', transform=hax.transAxes)
    hax.text(0.15, 0.35,
             f"{int(r1.get('minutos_stats',0) or 0)} min  ·  {int(r1.get('partidos_stats',0) or 0)} partidos",
             color=GRAY, fontsize=8,
             va='center', ha='left', transform=hax.transAxes)

    # Rating jugador 1
    rat1 = r1.get('rating')
    rat1_str = f'{float(rat1):.2f}' if pd.notna(rat1) else 'N/A'
    hax.add_patch(mpatches.FancyBboxPatch(
        (0.15, 0.10), 0.075, 0.16,
        boxstyle='round,pad=0.01', color=rating_color(rat1 if pd.notna(rat1) else None),
        transform=hax.transAxes, zorder=4))
    hax.text(0.1875, 0.18, rat1_str,
             color='#0d1117', **bebas(15),
             va='center', ha='center', transform=hax.transAxes, zorder=5)
    hax.text(0.1875, 0.10, 'Rating',
             color='#0d1117', fontsize=6, fontweight='bold',
             va='bottom', ha='center', transform=hax.transAxes, zorder=5)

    # Tarjetas jugador 1
    draw_card_icons(hax, y1, rc1, x_start=0.24, y=0.18, direction='left')

    # ── TEXTOS JUGADOR 2 ─────────────────────────────────────────────────
    hax.text(0.85, 0.65, r2['nombre'].upper(),
             color=WHITE, **bebas(20),
             va='center', ha='right', transform=hax.transAxes)
    hax.text(0.85, 0.48, f"{r2['equipo']}  ·  {r2['posicion']}  ·  {int(r2['edad'])} años",
             color=GRAY, fontsize=8.5,
             va='center', ha='right', transform=hax.transAxes)
    hax.text(0.85, 0.35,
             f"{int(r2.get('minutos_stats',0) or 0)} min  ·  {int(r2.get('partidos_stats',0) or 0)} partidos",
             color=GRAY, fontsize=8,
             va='center', ha='right', transform=hax.transAxes)

    # Rating jugador 2
    rat2 = r2.get('rating')
    rat2_str = f'{float(rat2):.2f}' if pd.notna(rat2) else 'N/A'
    hax.add_patch(mpatches.FancyBboxPatch(
        (0.775, 0.10), 0.075, 0.16,
        boxstyle='round,pad=0.01', color=rating_color(rat2 if pd.notna(rat2) else None),
        transform=hax.transAxes, zorder=4))
    hax.text(0.8125, 0.18, rat2_str,
             color='#0d1117', **bebas(15),
             va='center', ha='center', transform=hax.transAxes, zorder=5)
    hax.text(0.8125, 0.10, 'Rating',
             color='#0d1117', fontsize=6, fontweight='bold',
             va='bottom', ha='center', transform=hax.transAxes, zorder=5)

    # Tarjetas jugador 2 (de derecha a izquierda)
    draw_card_icons(hax, y2, rc2, x_start=0.76, y=0.18, direction='right')

    # ── ZONA DE BARRAS ───────────────────────────────────────────────────
    n = len(metrics)
    row_h = STATS_H / n
    stats_top = 1.0 - HEADER_H

    for i, (col, label) in enumerate(metrics):
        row_y  = stats_top - (i + 1) * row_h
        row_bg = '#161b22' if i % 2 == 0 else DARK_BG

        # Fondo de fila
        rb = fig.add_axes([0, row_y, 1, row_h])
        rb.set_facecolor(row_bg); rb.axis('off')
        rb.axhline(0, color='#21262d', lw=0.6)

        # Etiqueta central
        lax = fig.add_axes([CENTER_X_L, row_y, CENTER_X_R - CENTER_X_L, row_h])
        lax.set_facecolor(row_bg); lax.axis('off')
        lax.text(0.5, 0.5, label.upper(),
                 color=GRAY, fontsize=7.5, fontweight='bold',
                 va='center', ha='center', transform=lax.transAxes)

        # Valores raw y percentiles
        v1  = float(r1.get(col, 0) or 0)
        v2  = float(r2.get(col, 0) or 0)
        p1_ = get_pct(p1_id, col)
        p2_ = get_pct(p2_id, col)

        wins1 = p1_ >= p2_
        wins2 = not wins1

        # Colores de barra: ganador = color equipo saturado, perdedor = más apagado
        def bar_color(team_color, wins):
            r_, g_, b_ = tuple(int(team_color.lstrip('#')[k:k+2], 16)/255 for k in (0,2,4))
            if wins:
                return team_color
            # Apagar el color
            return mc.to_hex((r_*0.45, g_*0.45, b_*0.45))

        bc1 = bar_color(c1, wins1)
        bc2 = bar_color(c2, wins2)

        # ── BARRA JUGADOR 1 (va hacia la izquierda desde CENTER_X_L) ────
        bar_len1 = p1_ / 100 * BAR_MAX_W
        bax1 = fig.add_axes([CENTER_X_L - bar_len1, row_y + row_h * 0.20,
                             bar_len1 + 0.001, row_h * 0.60])
        bax1.set_facecolor(row_bg); bax1.axis('off')
        # Fondo gris completo
        bg1 = fig.add_axes([CENTER_X_L - BAR_MAX_W, row_y + row_h * 0.20,
                            BAR_MAX_W, row_h * 0.60])
        bg1.set_facecolor('#21262d'); bg1.axis('off')
        # Barra real
        bax1_real = fig.add_axes([CENTER_X_L - bar_len1, row_y + row_h * 0.20,
                                  bar_len1, row_h * 0.60])
        bax1_real.set_facecolor(bc1); bax1_real.axis('off')
        # Ganador: añadir marcador ★
        if wins1:
            fig.text(CENTER_X_L - bar_len1 - 0.015, row_y + row_h * 0.50,
                     '◀', color=c1, fontsize=8, va='center', ha='center')

        # Valor jugador 1 (al borde izquierdo de la barra)
        val_str1 = f'{v1:.2f}' if v1 != int(v1) else f'{int(v1)}'
        fig.text(CENTER_X_L - BAR_MAX_W - 0.005, row_y + row_h * 0.50,
                 val_str1,
                 color=WHITE if wins1 else GRAY,
                 fontsize=8.5, fontweight='bold' if wins1 else 'normal',
                 va='center', ha='right')

        # ── BARRA JUGADOR 2 (va hacia la derecha desde CENTER_X_R) ────
        bar_len2 = p2_ / 100 * BAR_MAX_W
        # Fondo gris completo
        bg2 = fig.add_axes([CENTER_X_R, row_y + row_h * 0.20,
                            BAR_MAX_W, row_h * 0.60])
        bg2.set_facecolor('#21262d'); bg2.axis('off')
        # Barra real
        bax2_real = fig.add_axes([CENTER_X_R, row_y + row_h * 0.20,
                                  bar_len2, row_h * 0.60])
        bax2_real.set_facecolor(bc2); bax2_real.axis('off')
        # Ganador marcador
        if wins2:
            fig.text(CENTER_X_R + bar_len2 + 0.015, row_y + row_h * 0.50,
                     '▶', color=c2, fontsize=8, va='center', ha='center')

        # Valor jugador 2
        val_str2 = f'{v2:.2f}' if v2 != int(v2) else f'{int(v2)}'
        fig.text(CENTER_X_R + BAR_MAX_W + 0.005, row_y + row_h * 0.50,
                 val_str2,
                 color=WHITE if wins2 else GRAY,
                 fontsize=8.5, fontweight='bold' if wins2 else 'normal',
                 va='center', ha='left')

    # ── CONTEO DE VICTORIAS ──────────────────────────────────────────────
    wins_p1 = sum(1 for col, _ in metrics
                  if get_pct(p1_id, col) >= get_pct(p2_id, col))
    wins_p2 = n - wins_p1

    w_ax = fig.add_axes([0, stats_top - n * row_h - 0.001, 1, 0.001])
    w_ax.axis('off')

    # Mostrar conteo en la zona baja del header (sobre las barras)
    hax.text(0.32, 0.22, f'{wins_p1} victorias',
             color=c1, **bebas(13),
             va='center', ha='right', transform=hax.transAxes)
    hax.text(0.68, 0.22, f'{wins_p2} victorias',
             color=c2, **bebas(13),
             va='center', ha='left', transform=hax.transAxes)

    # ── FOOTER ───────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H])
    fax.set_facecolor('#080c10'); fax.axis('off')
    fax.axhline(1, color=RED_BRAND, lw=2.0)

    fax.text(0.012, 0.44, 'Fuente: FotMob',
             color=GRAY, fontsize=9, va='center', ha='left',
             transform=fax.transAxes)

    # MAU-STATISTICS con sombra
    mau_kw = dict(va='center', ha='right', transform=fax.transAxes, **bebas(20))
    fax.text(0.989, 0.36, 'MAU-STATISTICS', color='#000000', alpha=0.65, **mau_kw)
    fax.text(0.987, 0.44, 'MAU-STATISTICS', color=RED_BRAND, **mau_kw)

    # ── GUARDAR ──────────────────────────────────────────────────────────
    nombre1 = str(r1['nombre']).lower().replace(' ', '_').replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n').replace('ü','u')
    nombre2 = str(r2['nombre']).lower().replace(' ', '_').replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n').replace('ü','u')
    out = OUT_DIR / f'comparativo_{nombre1}_vs_{nombre2}.png'

    plt.savefig(out, dpi=DPI, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f'✓ Guardado: {out}')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) == 3:
        id1, id2 = int(sys.argv[1]), int(sys.argv[2])
    else:
        # Ejemplo por defecto: Paulinho vs Ángel Sepúlveda
        id1, id2 = 361377, 215428
    render_1v1(id1, id2)
