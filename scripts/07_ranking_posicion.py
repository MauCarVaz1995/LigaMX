#!/usr/bin/env python3
"""
07_ranking_posicion.py
Top 10 por posición – Liga MX Clausura 2026
Genera: output/charts/ranking_{posicion}.png  (150 DPI)
"""

import os
import io
import urllib.request
import urllib.error
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFilter

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
CSV_PATH   = BASE / 'data/processed/jugadores_clausura2026.csv'
IMG_PLAY   = BASE / 'data/raw/images/players'
IMG_TEAMS  = BASE / 'data/raw/images/teams'
OUT_DIR    = BASE / 'output/charts'
BEBAS_TTF  = Path.home() / '.fonts/BebasNeue.ttf'

for d in (IMG_PLAY, IMG_TEAMS, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Registrar Bebas Neue si existe ──────────────────────────────────────────
_bebas_prop = None
if BEBAS_TTF.exists():
    fm.fontManager.addfont(str(BEBAS_TTF))
    _bebas_prop = FontProperties(fname=str(BEBAS_TTF))

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = '#0d1117'
BG_ROW_A = '#161b22'
BG_ROW_B = '#0d1117'
RED_BRAND   = '#D5001C'
GREEN_BRAND = '#00D4AA'
WHITE = '#e6edf3'
GRAY  = '#8b949e'
GOLD  = '#FFD700'

# Colores del círculo de score por ranking
def score_circle_color(rank_i: int) -> str:
    if rank_i < 3:
        return '#00C87A'   # verde
    if rank_i < 7:
        return '#F5A623'   # amarillo/naranja
    return '#E05A1B'       # naranja oscuro

# ─────────────────────────────────────────────────────────────────────────────
# MAPEO DE POSICIÓN FotMob → categoría
# ─────────────────────────────────────────────────────────────────────────────
_POS = {
    'GK':  'Portero',
    'CB':  'Defensa',  'LB': 'Defensa',  'RB': 'Defensa',
    'LWB': 'Defensa',  'RWB': 'Defensa',
    'CDM': 'Mediocampista', 'CM': 'Mediocampista',
    'CAM': 'Mediocampista', 'LM': 'Mediocampista', 'RM': 'Mediocampista',
    'LW':  'Delantero', 'RW': 'Delantero', 'ST': 'Delantero',
}

def classify(pos_str):
    if pd.isna(pos_str):
        return None
    primary = str(pos_str).split(',')[0].strip()
    return _POS.get(primary, None)

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS POR POSICIÓN
# ─────────────────────────────────────────────────────────────────────────────
METRICS = {
    'Delantero': {
        'cols':   ['goles_p90', 'xG_p90', 'tiros_p90', 'asistencias_p90', 'xA_p90'],
        'labels': ['Goles\nP90', 'xG\nP90', 'Tiros\nP90', 'Asist\nP90', 'xA\nP90'],
        'invert': [],
    },
    'Mediocampista': {
        'cols':   ['pases_precisos_p90', 'pases_largos_p90', 'asistencias_p90',
                   'recuperaciones_campo_rival_p90', 'duelos_tierra_ganados_p90'],
        'labels': ['Pases\nPrec P90', 'Pases\nProg P90', 'Asist\nP90',
                   'Recup\nP90', 'Duelos\nP90'],
        'invert': [],
    },
    'Defensa': {
        'cols':   ['intercepciones_p90', 'duelos_tierra_ganados_p90',
                   'recuperaciones_campo_rival_p90', 'pases_precisos_p90', 'despejes_p90'],
        'labels': ['Intercep\nP90', 'Duelos\nP90', 'Recup\nP90',
                   'Pases\nPrec P90', 'Despejes\nP90'],
        'invert': [],
    },
    'Portero': {
        'cols':   ['paradas_p90', 'porterias_cero_p90', 'goles_recibidos_p90'],
        'labels': ['Paradas\nP90', 'Porterías\nCero P90', 'Goles\nConced P90'],
        'invert': ['goles_recibidos_p90'],
    },
}

MIN_MINUTOS = 300
JORNADAS    = '1-12'     # rango confirmado por partidos_stats máx=12 en datos

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE IMAGEN
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_PLAYER = 'https://images.fotmob.com/image_resources/playerimages/{}.png'
FOTMOB_TEAM   = 'https://images.fotmob.com/image_resources/logo/teamlogo/{}.png'
_HEADERS = {'User-Agent': 'Mozilla/5.0'}

def _fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=6) as r:
            data = r.read()
        dest.write_bytes(data)
        return True
    except Exception:
        return False

def get_player_img(player_id):
    dest = IMG_PLAY / f'{player_id}.png'
    if not _fetch(FOTMOB_PLAYER.format(player_id), dest):
        return None
    try:
        return Image.open(dest).convert('RGBA')
    except Exception:
        return None

def get_team_img(team_id):
    dest = IMG_TEAMS / f'{team_id}.png'
    if not _fetch(FOTMOB_TEAM.format(team_id), dest):
        return None
    try:
        return Image.open(dest).convert('RGBA')
    except Exception:
        return None

def circular_crop(img: Image.Image, size: int = 80) -> np.ndarray:
    img = img.convert('RGBA').resize((size, size), Image.LANCZOS)
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    out = Image.new('RGBA', (size, size), (13, 17, 23, 0))
    out.paste(img, mask=mask)
    return np.array(out)

def placeholder_circle(size: int = 80) -> np.ndarray:
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((0, 0, size - 1, size - 1), fill=(55, 62, 75, 255))
    draw.ellipse((20, 8, 60, 42), fill=(95, 102, 115, 200))
    draw.ellipse((12, 46, 68, 80), fill=(95, 102, 115, 200))
    return np.array(img)

# ─────────────────────────────────────────────────────────────────────────────
# DEGRADADO RED→GREEN
# ─────────────────────────────────────────────────────────────────────────────
_cmap = mc.LinearSegmentedColormap.from_list(
    'rg', ['#D5001C', '#FFB700', '#00D4AA'])

def pct_color(pct: float) -> str:
    return mc.to_hex(_cmap(np.clip(pct, 0, 100) / 100))

# ─────────────────────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df['categoria'] = df['posicion'].apply(classify)
    return df[df['minutos_stats'] >= MIN_MINUTOS].copy()

def build_ranking(df: pd.DataFrame, categoria: str, top_n: int = 10) -> pd.DataFrame:
    cfg  = METRICS[categoria]
    cols = cfg['cols']
    inv  = cfg['invert']
    sub  = df[df['categoria'] == categoria].copy()
    for c in cols:
        if c not in sub.columns:
            sub[c] = np.nan
    pct_df = pd.DataFrame(index=sub.index)
    for c in cols:
        ranked = sub[c].fillna(0).rank(pct=True) * 100
        pct_df[c + '_pct'] = 100 - ranked if c in inv else ranked
    sub['score'] = pct_df.mean(axis=1)
    for c in cols:
        sub[c + '_pct'] = pct_df[c + '_pct']
    return sub.nlargest(top_n, 'score').reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
FIG_W    = 14
FIG_H    = 11
DPI      = 150

HEADER_H = 0.10
LABELS_H = 0.045
FOOTER_H = 0.06
ROWS_N   = 10
ROW_H    = (1.0 - HEADER_H - LABELS_H - FOOTER_H) / ROWS_N

COL_RANK        = (0.01,  0.030)
COL_PHOTO       = (0.04,  0.065)
COL_NAME        = (0.115, 0.195)
COL_BADGE       = (0.315, 0.050)
COL_BARS_START  = 0.375
COL_SCORE_W     = 0.055   # ancho columna score (círculo)
COL_BARS_TOTAL  = 1.0 - COL_BARS_START - COL_SCORE_W - 0.005   # hasta 0.945

# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render_ranking(df_top: pd.DataFrame, categoria: str, output_path: Path):
    cfg    = METRICS[categoria]
    cols   = cfg['cols']
    labels = cfg['labels']
    n_met  = len(cols)
    bar_w  = COL_BARS_TOTAL / n_met

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)

    # ── HEADER ──────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor('#161b22')
    hax.set_xlim(0, 1); hax.set_ylim(0, 1); hax.axis('off')
    hax.add_patch(mpatches.Rectangle((0, 0), 0.005, 1,
                  color=RED_BRAND, transform=hax.transAxes))
    hax.text(0.016, 0.64, f'TOP 10  ·  {categoria.upper()}S',
             color=WHITE, fontsize=22, fontweight='bold',
             va='center', ha='left', transform=hax.transAxes)
    hax.text(0.016, 0.24,
             f'Liga MX  ·  Clausura 2026  ·  Jornadas {JORNADAS}  ·  Mínimo {MIN_MINUTOS} min',
             color=GRAY, fontsize=9.5, va='center', ha='left',
             transform=hax.transAxes)
    hax.text(0.982, 0.5, 'Clausura\n2026',
             color=RED_BRAND, fontsize=11, fontweight='bold',
             va='center', ha='right', transform=hax.transAxes)

    # ── ETIQUETAS MÉTRICAS ───────────────────────────────────────────────
    label_y = 1 - HEADER_H - LABELS_H
    for j, lbl in enumerate(labels):
        bx = COL_BARS_START + j * bar_w
        lax = fig.add_axes([bx, label_y, bar_w, LABELS_H])
        lax.set_facecolor(DARK_BG); lax.axis('off')
        lax.text(0.5, 0.5, lbl, color=GRAY, fontsize=7.5, fontweight='bold',
                 va='center', ha='center', transform=lax.transAxes,
                 linespacing=1.3)

    sc_label_x = COL_BARS_START + COL_BARS_TOTAL + 0.005
    sl = fig.add_axes([sc_label_x, label_y, COL_SCORE_W, LABELS_H])
    sl.set_facecolor(DARK_BG); sl.axis('off')
    sl.text(0.5, 0.5, 'Score\nTotal', color=GRAY, fontsize=7.5,
            fontweight='bold', va='center', ha='center', transform=sl.transAxes)

    # ── FILAS DE JUGADORES ────────────────────────────────────────────────
    for i, row in df_top.iterrows():
        row_y  = 1 - HEADER_H - LABELS_H - (i + 1) * ROW_H
        row_bg = BG_ROW_A if i % 2 == 0 else BG_ROW_B

        # Fondo + separador
        bgax = fig.add_axes([0, row_y, 1, ROW_H])
        bgax.set_facecolor(row_bg); bgax.axis('off')
        bgax.axhline(0, color='#21262d', lw=0.8)

        # ── Nº ranking ──────────────────────────────────────────────────
        rax = fig.add_axes([COL_RANK[0], row_y, COL_RANK[1], ROW_H])
        rax.set_facecolor(row_bg); rax.axis('off')
        r_color = GOLD if i == 0 else (WHITE if i < 3 else GRAY)
        r_size  = 14 if i == 0 else (12 if i < 3 else 10)
        rax.text(0.5, 0.5, str(i + 1), color=r_color, fontsize=r_size,
                 fontweight='bold', va='center', ha='center',
                 transform=rax.transAxes)

        # ── Foto circular ────────────────────────────────────────────────
        pimg = get_player_img(row['id'])
        arr  = circular_crop(pimg, 80) if pimg else placeholder_circle(80)
        pax  = fig.add_axes([COL_PHOTO[0], row_y + ROW_H * 0.05,
                             COL_PHOTO[1], ROW_H * 0.90])
        pax.imshow(arr); pax.axis('off')

        # ── Nombre + equipo ──────────────────────────────────────────────
        nax = fig.add_axes([COL_NAME[0], row_y, COL_NAME[1], ROW_H])
        nax.set_facecolor(row_bg); nax.axis('off')
        nax.text(0.0, 0.65, row['nombre'], color=WHITE, fontsize=9.5,
                 fontweight='bold', va='center', ha='left',
                 transform=nax.transAxes, clip_on=True)
        edad_str = f"  ·  {int(row['edad'])} años" if pd.notna(row.get('edad')) else ''
        nax.text(0.0, 0.26, str(row['equipo']) + edad_str,
                 color=GRAY, fontsize=8, va='center', ha='left',
                 transform=nax.transAxes, clip_on=True)

        # ── Escudo ───────────────────────────────────────────────────────
        timg = get_team_img(row['equipo_id'])
        if timg is not None:
            t_arr = np.array(timg.convert('RGBA').resize((50, 50), Image.LANCZOS))
            bax   = fig.add_axes([COL_BADGE[0], row_y + ROW_H * 0.1,
                                  COL_BADGE[1], ROW_H * 0.80])
            bax.imshow(t_arr); bax.axis('off')

        # ── Barras de métricas ───────────────────────────────────────────
        for j, col in enumerate(cols):
            pct_col = col + '_pct'
            pct_val = float(row.get(pct_col, 0) or 0)
            raw_val = row.get(col, np.nan)
            val_str = f'{raw_val:.2f}' if pd.notna(raw_val) else '—'

            bx = COL_BARS_START + j * bar_w
            # Dejar margen derecho interno para que el número no se corte
            bar_ax = fig.add_axes([bx + 0.004, row_y + ROW_H * 0.27,
                                   bar_w - 0.010, ROW_H * 0.48])
            bar_ax.set_facecolor(row_bg)
            bar_ax.set_xlim(0, 115)   # ← margen derecho de 15 pt extra
            bar_ax.set_ylim(0, 1)
            bar_ax.axis('off')

            bar_ax.barh(0.5, 100, height=0.58, color='#21262d', align='center')
            bar_color = pct_color(pct_val)
            bar_ax.barh(0.5, pct_val, height=0.58, color=bar_color, align='center')

            # Número: dentro si pct_val alto, fuera si bajo
            if pct_val >= 70:
                bar_ax.text(pct_val - 2, 0.5, val_str,
                            color='#0d1117', fontsize=6.5,
                            va='center', ha='right')
            else:
                bar_ax.text(pct_val + 3, 0.5, val_str,
                            color=WHITE, fontsize=6.5,
                            va='center', ha='left')

        # ── Score — CÍRCULO ──────────────────────────────────────────────
        score      = float(row.get('score', 0) or 0)
        circ_color = score_circle_color(i)
        sc_x       = COL_BARS_START + COL_BARS_TOTAL + 0.005

        # Axes para el círculo (usamos imshow de una imagen PIL para control total)
        circ_size = 60
        circ_img  = Image.new('RGBA', (circ_size, circ_size), (0, 0, 0, 0))
        cir_draw  = ImageDraw.Draw(circ_img)
        # Sombra (círculo negro ligeramente desplazado)
        cir_draw.ellipse((3, 3, circ_size - 1, circ_size - 1),
                         fill=(0, 0, 0, 130))
        # Círculo principal
        r, g, b = tuple(int(circ_color.lstrip('#')[k:k+2], 16) for k in (0, 2, 4))
        cir_draw.ellipse((0, 0, circ_size - 4, circ_size - 4),
                         fill=(r, g, b, 255))

        sc_ax = fig.add_axes([sc_x, row_y + ROW_H * 0.08,
                              COL_SCORE_W, ROW_H * 0.84])
        sc_ax.imshow(np.array(circ_img)); sc_ax.axis('off')
        # Texto encima
        sc_ax.text(0.45, 0.46, f'{score:.0f}',
                   color='#0d1117', fontsize=11, fontweight='bold',
                   va='center', ha='center', transform=sc_ax.transAxes)

    # ── FOOTER ────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, 0, 1, FOOTER_H])
    fax.set_facecolor('#080c10'); fax.axis('off')
    # Línea roja separadora arriba del footer
    fax.axhline(1, color=RED_BRAND, lw=2.0)

    fax.text(0.012, 0.44, 'Fuente: FotMob',
             color=GRAY, fontsize=9, va='center', ha='left',
             transform=fax.transAxes)

    # MAU-STATISTICS — sombra negra + texto rojo en Bebas Neue
    mau_x, mau_y = 0.987, 0.44
    shadow_kwargs = dict(fontsize=20, fontweight='bold',
                         va='center', ha='right', transform=fax.transAxes)
    if _bebas_prop:
        shadow_kwargs['fontproperties'] = _bebas_prop
    # Sombra (texto desplazado en negro semitransparente)
    fax.text(mau_x + 0.002, mau_y - 0.06, 'MAU-STATISTICS',
             color='#000000', alpha=0.7, **shadow_kwargs)
    fax.text(mau_x, mau_y, 'MAU-STATISTICS',
             color=RED_BRAND, **shadow_kwargs)

    plt.savefig(output_path, dpi=DPI, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f'  ✓ Guardado: {output_path}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(posiciones=None):
    print('Cargando datos...')
    df = load_data()
    if posiciones is None:
        posiciones = ['Delantero', 'Mediocampista', 'Defensa', 'Portero']
    for cat in posiciones:
        print(f'\n[{cat}]')
        top = build_ranking(df, cat)
        print(f'  {len(top)} jugadores')
        if top.empty:
            print('  ⚠ Sin datos, omitiendo.'); continue
        out = OUT_DIR / f'ranking_{cat.lower()}.png'
        render_ranking(top, cat, out)


if __name__ == '__main__':
    import sys
    cats = sys.argv[1:] if len(sys.argv) > 1 else None
    main(posiciones=cats)
