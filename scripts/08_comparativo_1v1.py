#!/usr/bin/env python3
"""
08_comparativo_1v1.py  –  Comparativo 1v1 jugadores Liga MX
Uso: python 08_comparativo_1v1.py [id_jugador1] [id_jugador2]
     python 08_comparativo_1v1.py 361377 215428   # Paulinho vs Ángel Sepúlveda
"""

import sys, warnings, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw
import urllib.request

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETTE, bebas as _bebas_cv, hex_rgba, hex_rgb as _hex_rgb_cv

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
CSV_PATH  = BASE / 'data/processed/jugadores_clausura2026.csv'
IMG_PLAY  = BASE / 'data/raw/images/players'
IMG_TEAMS = BASE / 'data/raw/images/teams'
OUT_DIR   = BASE / 'output/charts'
BEBAS_TTF = Path.home() / '.fonts/BebasNeue.ttf'

for d in (IMG_PLAY, IMG_TEAMS, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

if BEBAS_TTF.exists():
    fm.fontManager.addfont(str(BEBAS_TTF))

def bebas(size):
    return _bebas_cv(size)

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = PALETTE['bg_main']
WHITE     = PALETTE['text_primary']
GRAY      = PALETTE['text_secondary']
GRAY_DIM  = PALETTE['bar_loser']
GRAY_BORDER = PALETTE['bar_loser_border']
RED_BRAND = PALETTE['accent']
BAR_TRACK = PALETTE['bar_track']

TEAM_COLORS = {
    6618:    '#D5001C',
    7807:    '#CD1F2D',
    8561:    '#F5A623',
    6576:    '#FFD700',
    6578:    '#0047AB',
    7849:    '#003DA5',
    1946:    '#C8A84B',
    7857:    '#2E8B57',
    7848:    '#A8B8C8',
    7847:    '#2563EB',
    1841:    '#2D8C3C',
    1842:    '#D62828',
    162418:  '#C62828',
    6577:    '#B22222',
    1943:    '#1A7FCB',
    649424:  '#4CAF50',
    1170234: '#9B59B6',
    6358:    '#D52B1E',
}

METRICS = [
    ('goles_p90',                 'GOLES P90'),
    ('xG_p90',                    'xG P90'),
    ('tiros_p90',                 'TIROS P90'),
    ('tiros_a_puerta_p90',        'TIROS A PUERTA P90'),
    ('asistencias_p90',           'ASISTENCIAS P90'),
    ('xA_p90',                    'xA P90'),
    ('chances_creadas_p90',       'CHANCES CREADAS P90'),
    ('pases_precisos_p90',        'PASES PRECISOS P90'),
    ('duelos_tierra_ganados_p90', 'DUELOS GANADOS P90'),
    ('regates_exitosos_p90',      'REGATES EXITOSOS P90'),
]
N_METRICS = len(METRICS)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 8.0, 12.0
DPI          = 150
AR           = FIG_H / FIG_W   # 1.5

FOOTER_H  = 0.063
WINCTR_H  = 0.072
ROW_H     = 0.050   # más compacto
METRICS_H = ROW_H * N_METRICS
RATING_H  = 0.105
HEADER_H  = 0.210

FOOTER_Y  = 0.000
WINCTR_Y  = FOOTER_Y + FOOTER_H
METRICS_Y = WINCTR_Y + WINCTR_H
RATING_Y  = METRICS_Y + METRICS_H
HEADER_Y  = RATING_Y + RATING_H

CENTER_X  = 0.500
LABEL_W   = 0.170
LABEL_X   = CENTER_X - LABEL_W / 2
VAL_W     = 0.085
BAR_MAX_W = LABEL_X - VAL_W
BAR_L_X   = VAL_W
BAR2_X    = LABEL_X + LABEL_W
BAR_H_ROW = 0.58   # barras más altas dentro de la fila más compacta
BAR_PADY  = (1 - BAR_H_ROW) / 2

PHOTO_H_FIG = 0.110
PHOTO_W_FIG = PHOTO_H_FIG * AR

# ─────────────────────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────────────────────
def _classify(pos_str):
    POS = {'GK':'Portero','CB':'Defensa','LB':'Defensa','RB':'Defensa',
           'LWB':'Defensa','RWB':'Defensa','CDM':'Mediocampista',
           'CM':'Mediocampista','CAM':'Mediocampista','LM':'Mediocampista',
           'RM':'Mediocampista','LW':'Delantero','RW':'Delantero','ST':'Delantero'}
    if pd.isna(pos_str): return 'Delantero'
    return POS.get(str(pos_str).split(',')[0].strip(), 'Mediocampista')

def load_csv():
    df = pd.read_csv(CSV_PATH)
    df['categoria'] = df['posicion'].apply(_classify)
    return df

def get_player_row(pid, df):
    row = df[df['id'] == pid]
    if row.empty: raise ValueError(f'Jugador {pid} no encontrado.')
    return row.iloc[0]

def get_metric_value(row, col):
    v = row.get(col, 0)
    if v is None or (isinstance(v, float) and np.isnan(v)): return 0.0
    return float(v)

# ─────────────────────────────────────────────────────────────────────────────
# IMÁGENES
# ─────────────────────────────────────────────────────────────────────────────
_HDRS = {'User-Agent': 'Mozilla/5.0'}

def _fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 500: return True
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=_HDRS), timeout=8) as r:
            dest.write_bytes(r.read())
        return True
    except Exception:
        return False

def get_player_img(pid):
    dest = IMG_PLAY / f'{pid}.png'
    if not _fetch(f'https://images.fotmob.com/image_resources/playerimages/{pid}.png', dest):
        return None
    try: return Image.open(dest).convert('RGBA')
    except Exception: return None

def get_team_img(tid):
    dest = IMG_TEAMS / f'{tid}.png'
    if not _fetch(f'https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png', dest):
        return None
    try: return Image.open(dest).convert('RGBA')
    except Exception: return None

def _hex_rgb_tuple(hex_color):
    return _hex_rgb_cv(hex_color)

def circular_crop(img: Image.Image, px: int, border_color: tuple) -> np.ndarray:
    """Recorta en círculo con borde de 3px en color del equipo."""
    img = img.convert('RGBA').resize((px, px), Image.LANCZOS)
    out = Image.new('RGBA', (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    # Borde exterior en color de equipo
    draw.ellipse((0, 0, px-1, px-1), fill=(*border_color, 255))
    # Máscara interior (3px de borde)
    bw = 3
    mask = Image.new('L', (px, px), 0)
    ImageDraw.Draw(mask).ellipse((bw, bw, px-bw-1, px-bw-1), fill=255)
    out.paste(img, mask=mask)
    return np.array(out)

def placeholder_circle(px=120, color=(60,60,80)) -> np.ndarray:
    img = Image.new('RGBA', (px, px), (0,0,0,0))
    ImageDraw.Draw(img).ellipse((0,0,px-1,px-1), fill=(*color, 255))
    return np.array(img)

def make_rating_circle(rating_val, team_color_hex: str, px=160) -> np.ndarray:
    """Círculo con rating en el color del equipo."""
    rgb = _hex_rgb_tuple(team_color_hex)
    # Oscurecer ligeramente para fondo del círculo
    dark_rgb = tuple(max(0, c - 30) for c in rgb)

    if rating_val is None or (isinstance(rating_val, float) and np.isnan(rating_val)):
        text = 'N/A'
    else:
        text = f'{float(rating_val):.2f}'

    img  = Image.new('RGBA', (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Sombra
    draw.ellipse((4, 4, px-1, px-1), fill=(0, 0, 0, 130))
    # Anillo exterior del color del equipo
    draw.ellipse((0, 0, px-5, px-5), fill=(*rgb, 255))
    # Interior más oscuro
    inner = 8
    draw.ellipse((inner, inner, px-5-inner, px-5-inner), fill=(*dark_rgb, 255))

    try:
        from PIL import ImageFont
        fnt_big   = ImageFont.truetype(str(BEBAS_TTF), px // 3)
        fnt_small = ImageFont.truetype(str(BEBAS_TTF), px // 8)
    except Exception:
        fnt_big = fnt_small = None

    cx = cy = (px - 5) // 2
    if fnt_big:
        bb = draw.textbbox((0,0), text, font=fnt_big)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        draw.text((cx - tw//2, cy - th//2 - 3), text, fill=(255,255,255,255), font=fnt_big)
        lbl = 'RATING FBM'
        bb2 = draw.textbbox((0,0), lbl, font=fnt_small)
        draw.text((cx - (bb2[2]-bb2[0])//2, cy + th//2 + 3), lbl,
                  fill=(220,220,220,200), font=fnt_small)
    else:
        draw.text((cx-12, cy-10), text, fill=(255,255,255,255))
    return np.array(img)

def _hex_rgba(hex_color, alpha=1.0):
    r,g,b = tuple(int(hex_color.lstrip('#')[i:i+2],16)/255 for i in (0,2,4))
    return (r,g,b,alpha)

def _add_img_axis(fig, arr, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(arr, aspect='equal')
    ax.axis('off')
    return ax

def _slugify(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.lower().replace(' ', '_')

# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render_1v1(pid1: int, pid2: int):
    df = load_csv()
    r1 = get_player_row(pid1, df)
    r2 = get_player_row(pid2, df)

    c1     = TEAM_COLORS.get(int(r1['equipo_id']), '#4a5568')
    c2     = TEAM_COLORS.get(int(r2['equipo_id']), '#4a5568')
    c1_rgb = _hex_rgb_tuple(c1)
    c2_rgb = _hex_rgb_tuple(c2)

    # ── Figura + gradiente ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)

    grad = np.zeros((200, 2, 3))
    for i in range(200):
        t = i/199
        grad[i] = np.array([0x0a,0x0e,0x12])/255 * (1-t) + np.array([0x13,0x1a,0x24])/255 * t
    bg = fig.add_axes([0,0,1,1])
    bg.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bg.set_xlim(0,1); bg.set_ylim(0,1); bg.axis('off')
    bg.add_patch(mpatches.Rectangle((0,0),0.5,1, facecolor=_hex_rgba(c1,0.10), linewidth=0))
    bg.add_patch(mpatches.Rectangle((0.5,0),0.5,1, facecolor=_hex_rgba(c2,0.10), linewidth=0))
    bg.plot([0.5,0.5],[FOOTER_Y+FOOTER_H, 1.0], color='#2d333b', lw=1.0)

    # ── Header ────────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, HEADER_Y, 1, HEADER_H])
    hax.set_facecolor('#0d1117'); hax.axis('off')
    hax.set_xlim(0,1); hax.set_ylim(0,1)
    hax.add_patch(mpatches.Rectangle((0,0),0.5,1, facecolor=_hex_rgba(c1,0.22), linewidth=0))
    hax.add_patch(mpatches.Rectangle((0.5,0),0.5,1, facecolor=_hex_rgba(c2,0.22), linewidth=0))
    hax.add_patch(mpatches.Rectangle((0,0.94),0.5,0.06, facecolor=c1, linewidth=0))
    hax.add_patch(mpatches.Rectangle((0.5,0.94),0.5,0.06, facecolor=c2, linewidth=0))

    hax.text(0.500, 0.63, 'VS', color=WHITE, ha='center', va='center',
             transform=hax.transAxes, **bebas(36))
    hax.text(0.500, 0.31, 'LIGA MX · CLAUSURA 2026',
             color=GRAY, ha='center', va='center', fontsize=8.5,
             transform=hax.transAxes)

    # Fotos circulares con borde del color del equipo
    pimg1 = get_player_img(pid1)
    pimg2 = get_player_img(pid2)
    arr1 = circular_crop(pimg1, 130, c1_rgb) if pimg1 else placeholder_circle(130, (60,20,20))
    arr2 = circular_crop(pimg2, 130, c2_rgb) if pimg2 else placeholder_circle(130, (20,20,60))

    ph_y = HEADER_Y + HEADER_H * 0.13
    _add_img_axis(fig, arr1, 0.025, ph_y, PHOTO_W_FIG, PHOTO_H_FIG)
    _add_img_axis(fig, arr2, 1.0 - 0.025 - PHOTO_W_FIG, ph_y, PHOTO_W_FIG, PHOTO_H_FIG)

    # Escudos
    tid1 = int(r1['equipo_id']); tid2 = int(r2['equipo_id'])
    timg1 = get_team_img(tid1);  timg2 = get_team_img(tid2)
    sh_h  = 0.052; sh_w = sh_h * AR
    shield_y = HEADER_Y + HEADER_H * 0.44
    text_x1  = 0.025 + PHOTO_W_FIG + 0.015

    if timg1:
        ax_ = fig.add_axes([text_x1, shield_y, sh_w, sh_h])
        ax_.set_facecolor('#f8f8fc')
        ax_.imshow(np.array(timg1.resize((80,80), Image.LANCZOS)), aspect='equal')
        ax_.axis('off')

    text_x2 = 1.0 - 0.025 - PHOTO_W_FIG - 0.015 - sh_w
    if timg2:
        ax_ = fig.add_axes([text_x2, shield_y, sh_w, sh_h])
        ax_.set_facecolor('#f8f8fc')
        ax_.imshow(np.array(timg2.resize((80,80), Image.LANCZOS)), aspect='equal')
        ax_.axis('off')

    name2_x = 1.0 - 0.025 - PHOTO_W_FIG - 0.015
    hax.text(text_x1, 0.82, r1['nombre'].upper(), color=WHITE, ha='left', va='center',
             transform=hax.transAxes, **bebas(21))
    hax.text(name2_x, 0.82, r2['nombre'].upper(), color=WHITE, ha='right', va='center',
             transform=hax.transAxes, **bebas(21))
    hax.text(text_x1 + sh_w + 0.008, 0.47, r1['equipo'].upper(),
             color=c1, ha='left', va='center', transform=hax.transAxes, **bebas(13))
    hax.text(text_x2 - 0.008, 0.47, r2['equipo'].upper(),
             color=c2, ha='right', va='center', transform=hax.transAxes, **bebas(13))
    hax.text(text_x1, 0.24,
             f"{r1.get('posicion','—')}  ·  {int(r1.get('minutos_stats',0) or 0)} MIN",
             color=GRAY, ha='left', va='center', fontsize=8.5, transform=hax.transAxes)
    hax.text(name2_x, 0.24,
             f"{r2.get('posicion','—')}  ·  {int(r2.get('minutos_stats',0) or 0)} MIN",
             color=GRAY, ha='right', va='center', fontsize=8.5, transform=hax.transAxes)

    # ── Rating circles — color del equipo ────────────────────────────────────
    rat1 = r1.get('rating'); rat2 = r2.get('rating')
    rc_h  = 0.100; rc_w = rc_h * AR
    rc_y  = RATING_Y + (RATING_H - rc_h) / 2

    rc1_arr = make_rating_circle(rat1 if pd.notna(rat1) else None, c1, 160)
    rc2_arr = make_rating_circle(rat2 if pd.notna(rat2) else None, c2, 160)
    _add_img_axis(fig, rc1_arr, CENTER_X - rc_w - 0.042, rc_y, rc_w, rc_h)
    _add_img_axis(fig, rc2_arr, CENTER_X + 0.042, rc_y, rc_w, rc_h)

    # ── Métricas ──────────────────────────────────────────────────────────────
    mx = fig.add_axes([0, METRICS_Y, 1, METRICS_H])
    mx.set_xlim(0, 1); mx.set_ylim(0, N_METRICS)
    mx.set_facecolor(DARK_BG); mx.axis('off')

    wins1 = wins2 = 0

    for i, (col, label) in enumerate(METRICS):
        row_bot = N_METRICS - 1 - i
        bg_row  = '#0f151e' if i % 2 == 0 else DARK_BG
        mx.add_patch(Rectangle((0, row_bot), 1, 1, facecolor=bg_row, zorder=1))
        mx.axhline(row_bot + 1, color='#1e2530', lw=0.6, zorder=2)

        v1 = get_metric_value(r1, col)
        v2 = get_metric_value(r2, col)
        max_v = max(v1, v2, 1e-9)
        w1 = v1 / max_v; w2 = v2 / max_v

        if v1 > v2:   wins1 += 1
        elif v2 > v1: wins2 += 1
        else:         wins1 += 1; wins2 += 1

        won1 = v1 >= v2
        bar_b = row_bot + BAR_PADY

        # Track izquierdo
        mx.add_patch(Rectangle((BAR_L_X, bar_b), BAR_MAX_W, BAR_H_ROW,
                                facecolor=BAR_TRACK, zorder=3))
        # Barra izquierda
        if w1 > 0.001:
            mx.add_patch(Rectangle(
                (LABEL_X - BAR_MAX_W * w1, bar_b), BAR_MAX_W * w1, BAR_H_ROW,
                facecolor=c1 if won1 else GRAY_DIM,
                edgecolor=(c1 if won1 else GRAY_BORDER),
                linewidth=(0 if won1 else 1.0),
                zorder=4))

        # Track derecho
        mx.add_patch(Rectangle((BAR2_X, bar_b), BAR_MAX_W, BAR_H_ROW,
                                facecolor=BAR_TRACK, zorder=3))
        # Barra derecha
        if w2 > 0.001:
            mx.add_patch(Rectangle(
                (BAR2_X, bar_b), BAR_MAX_W * w2, BAR_H_ROW,
                facecolor=c2 if not won1 else GRAY_DIM,
                edgecolor=(c2 if not won1 else GRAY_BORDER),
                linewidth=(0 if not won1 else 1.0),
                zorder=4))

        # Etiqueta
        mx.text(CENTER_X, row_bot + 0.50, label,
                color=GRAY, fontsize=6.5, fontweight='bold',
                ha='center', va='center', zorder=5)

        # Valores
        val1 = f'{v1:.2f}' if v1 != int(v1) else str(int(v1))
        val2 = f'{v2:.2f}' if v2 != int(v2) else str(int(v2))
        mx.text(BAR_L_X - 0.008, row_bot + 0.50, val1,
                color=WHITE if won1 else GRAY, fontsize=8,
                fontweight='bold' if won1 else 'normal',
                ha='right', va='center', zorder=5)
        mx.text(1.0 - VAL_W + 0.008, row_bot + 0.50, val2,
                color=WHITE if not won1 else GRAY, fontsize=8,
                fontweight='bold' if not won1 else 'normal',
                ha='left', va='center', zorder=5)

    # ── Banner victorias — barras proporcionales ──────────────────────────────
    wax = fig.add_axes([0, WINCTR_Y, 1, WINCTR_H])
    wax.set_facecolor('#060a0f'); wax.axis('off')
    wax.set_xlim(0, 1); wax.set_ylim(0, 1)
    wax.axhline(1.0, color='#21262d', lw=0.8)

    total_wins = wins1 + wins2 if (wins1 + wins2) > 0 else 1
    frac1 = wins1 / total_wins   # fracción del total
    frac2 = wins2 / total_wins

    BAR_TOP = 0.72; BAR_BOT = 0.18; BAR_H_WIN = BAR_TOP - BAR_BOT
    GAP = 0.015   # gap central

    # Barra izquierda (crece desde el centro hacia la izquierda)
    wax.add_patch(mpatches.Rectangle(
        (0.0, BAR_BOT), 0.5 - GAP/2, BAR_H_WIN,
        facecolor='#1a1f28', linewidth=0, zorder=1))
    wax.add_patch(mpatches.Rectangle(
        (0.5 - GAP/2 - (0.5-GAP/2)*frac1, BAR_BOT),
        (0.5 - GAP/2) * frac1, BAR_H_WIN,
        facecolor=c1, linewidth=0, zorder=2, alpha=0.90))

    # Barra derecha (crece desde el centro hacia la derecha)
    wax.add_patch(mpatches.Rectangle(
        (0.5 + GAP/2, BAR_BOT), 0.5 - GAP/2, BAR_H_WIN,
        facecolor='#1a1f28', linewidth=0, zorder=1))
    wax.add_patch(mpatches.Rectangle(
        (0.5 + GAP/2, BAR_BOT), (0.5-GAP/2)*frac2, BAR_H_WIN,
        facecolor=c2, linewidth=0, zorder=2, alpha=0.90))

    ap1 = r1['nombre'].split()[-1].upper()
    ap2 = r2['nombre'].split()[-1].upper()
    mid_y = (BAR_TOP + BAR_BOT) / 2

    wax.text(0.5 - GAP/2 - 0.008, mid_y,
             f'{ap1}  {wins1} MÉTRICAS',
             color=WHITE, ha='right', va='center',
             transform=wax.transAxes, **bebas(13))
    wax.text(0.5 + GAP/2 + 0.008, mid_y,
             f'{wins2} MÉTRICAS  {ap2}',
             color=WHITE, ha='left', va='center',
             transform=wax.transAxes, **bebas(13))

    # ── Footer ────────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, FOOTER_Y, 1, FOOTER_H])
    fax.set_facecolor('#080c10'); fax.axis('off')
    fax.set_xlim(0,1); fax.set_ylim(0,1)
    fax.axhline(1.0, color=RED_BRAND, lw=2.0)
    fax.text(0.014, 0.46, 'Fuente: FotMob',
             color=GRAY, fontsize=9, ha='left', va='center', transform=fax.transAxes)
    kw = dict(ha='right', va='center', transform=fax.transAxes, **bebas(22))
    fax.text(0.991, 0.38, 'MAU-STATISTICS', color='#000000', alpha=0.55, **kw)
    fax.text(0.989, 0.50, 'MAU-STATISTICS', color=RED_BRAND, **kw)

    # ── Guardar ───────────────────────────────────────────────────────────────
    out = OUT_DIR / f'comparativo_{_slugify(str(r1["nombre"]))}_vs_{_slugify(str(r2["nombre"]))}.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f'✓ Guardado: {out}')
    return out

if __name__ == '__main__':
    pid1 = int(sys.argv[1]) if len(sys.argv) > 1 else 361377
    pid2 = int(sys.argv[2]) if len(sys.argv) > 2 else 215428
    render_1v1(pid1, pid2)
