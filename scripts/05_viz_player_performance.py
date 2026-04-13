#!/usr/bin/env python3
"""
05_viz_player_performance.py
Infografías de rendimiento de partido desde JSON FotMob.

Genera:
  {CODE}_ratings_{team}.png  — ficha técnica por equipo (1080×1500 portrait)
  {CODE}_team_stats.png      — estadísticas comparativas (1080×1080)

Uso:
  python scripts/05_viz_player_performance.py
  python scripts/05_viz_player_performance.py --json <ruta> --code <CÓDIGO>
"""

import argparse, json, re, sys, urllib.request, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETAS, get_paleta, bebas, hex_rgb, get_escudo

# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
OUT_DIR   = BASE / 'output/charts/partidos'
PHOTO_DIR = BASE / 'data/raw/fotmob/playerimages'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHOTO_DIR.mkdir(parents=True, exist_ok=True)

# Fuente centralizada desde config_visual (incluye assets/fonts/ como candidato)
try:
    from config_visual import BEBAS_TTF, BEBAS_AVAILABLE
except ImportError:
    BEBAS_TTF = None
    BEBAS_AVAILABLE = False

# Paleta fija — cambiar cuando el usuario lo indique
PALETA_ACTIVA = 'rojo_fuego'

POS_MAP    = {0: 'POR', 1: 'DEF', 2: 'MED', 3: 'DEL'}
POS_COLORS = {'POR': '#FF8C00', 'DEF': '#4A90D9', 'MED': '#4CAF50', 'DEL': '#E8384F'}
GOLD       = '#FFD700'
TEAM_ES    = {'Mexico': 'México', 'Portugal': 'Portugal'}
SEP_C      = '#2A2A2A'


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _parse_num(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.match(r'[\d.]+', val.strip())
        return float(m.group()) if m else 0.0
    return 0.0


def _fmt(val):
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float)) and float(val) == int(float(val)):
        return str(int(val))
    return str(val)


def _ap_str(v):
    """accurate_passes: dict {'value':x,'total':y} → 'x/y'"""
    if isinstance(v, dict):
        return f"{v.get('value','?')}/{v.get('total','?')}"
    return str(v) if v is not None else '—'


def _pos_stats(stats: dict, pos_id: int) -> tuple:
    """2 stats clave (label1, val1, label2, val2) según posición."""
    if pos_id == 0:
        return 'Saves', stats.get('saves', '—'), \
               'GC',    stats.get('goals_conceded', '—')
    elif pos_id == 1:
        return 'Duelos', stats.get('duel_won', '—'), \
               'Pases',  _ap_str(stats.get('accurate_passes'))
    elif pos_id == 2:
        return 'Pases',  _ap_str(stats.get('accurate_passes')), \
               'Recup.', stats.get('recoveries', '—')
    else:
        return 'Box',   stats.get('touches_opp_box', '—'), \
               'F.rec.', stats.get('was_fouled', '—')


def _pos_metrics(stats: dict, pos_id: int) -> list:
    """4 métricas (label, val) para FIGURA DEL PARTIDO."""
    mins = stats.get('minutes_played', 90)
    if pos_id == 0:
        return [
            ('Rating',  str(stats.get('rating_title', '—'))),
            ('Minutos', str(mins)),
            ('Saves',   str(stats.get('saves', '—'))),
            ('GC',      str(stats.get('goals_conceded', '—'))),
        ]
    elif pos_id == 1:
        return [
            ('Rating',    str(stats.get('rating_title', '—'))),
            ('Minutos',   str(mins)),
            ('Duelos G.', str(stats.get('duel_won', '—'))),
            ('Pases',     _ap_str(stats.get('accurate_passes'))),
        ]
    elif pos_id == 2:
        return [
            ('Rating',  str(stats.get('rating_title', '—'))),
            ('Minutos', str(mins)),
            ('Pases',   _ap_str(stats.get('accurate_passes'))),
            ('Recup.',  str(stats.get('recoveries', '—'))),
        ]
    else:
        return [
            ('Rating',  str(stats.get('rating_title', '—'))),
            ('Minutos', str(mins)),
            ('Box',     str(stats.get('touches_opp_box', '—'))),
            ('F. rec.', str(stats.get('was_fouled', '—'))),
        ]


def _get_starters(players: list) -> list:
    s = [p for p in players if p.get('is_starter')]
    s.sort(key=lambda p: {0: 0, 1: 1, 2: 2, 3: 3}.get(p.get('usual_position', 2), 2))
    return s


def _get_player_photo(player_id, size: int = 120) -> 'OffsetImage | None':
    """Descarga foto del jugador desde FotMob. Falla silenciosamente.
    Siempre recorta a size×size con PIL (center-crop) antes de insertar."""
    if not player_id:
        return None
    cache = PHOTO_DIR / f'{player_id}.png'
    if not cache.exists():
        try:
            url = f'https://images.fotmob.com/image_resources/playerimages/{player_id}.png'
            urllib.request.urlretrieve(url, str(cache))
        except Exception:
            return None
    try:
        from PIL import Image as PILImage
        resample = PILImage.Resampling.LANCZOS if hasattr(PILImage, 'Resampling') else PILImage.LANCZOS
        img = PILImage.open(cache).convert('RGBA')
        # Center-crop al cuadrado más pequeño, luego resize a size×size exacto
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top  = (h - min_dim) // 2
        img  = img.crop((left, top, left + min_dim, top + min_dim))
        img  = img.resize((size, size), resample)
        return OffsetImage(np.array(img), zoom=1.0)
    except Exception:
        return None


def _shade(hex_color: str, factor: float) -> str:
    """Oscurece (factor<1) o aclara (factor>1) un color hex."""
    r, g, b = hex_rgb(hex_color)
    return f'#{min(255,int(r*factor)):02X}{min(255,int(g*factor)):02X}{min(255,int(b*factor)):02X}'


# ─────────────────────────────────────────────────────────────────────────────
# FONDOS
# ─────────────────────────────────────────────────────────────────────────────
def _radial_bg(fig, bg: str, bg2: str, res: int = 256):
    """Gradiente radial: centro=bg2 (ligeramente más claro), esquinas=bg."""
    c_ctr = np.array(hex_rgb(bg2)) / 255
    c_cor = np.array(hex_rgb(bg))  / 255
    y_, x_ = np.mgrid[0:res, 0:res]
    cx = cy = res / 2
    t  = np.clip(np.sqrt((x_-cx)**2 + (y_-cy)**2) / (np.sqrt(2)*cx), 0, 1)
    g  = c_ctr*(1-t[..., np.newaxis]) + c_cor*t[..., np.newaxis]
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_zorder(-100); ax.axis('off')
    ax.imshow(g, aspect='auto', extent=[0,1,0,1], origin='lower')


def _diag_bg(fig, bg: str, bg2: str, res: int = 256):
    """Gradiente diagonal: bg arriba-izq, bg2 abajo-der."""
    c_tl = np.array(hex_rgb(bg))  / 255
    c_br = np.array(hex_rgb(bg2)) / 255
    y_, x_ = np.mgrid[0:res, 0:res]
    t  = (x_/(res-1) + (1.0 - y_/(res-1))) / 2.0
    g  = c_tl*(1-t[..., np.newaxis]) + c_br*t[..., np.newaxis]
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_zorder(-100); ax.axis('off')
    ax.imshow(g, aspect='auto', extent=[0,1,0,1], origin='lower')


# ─────────────────────────────────────────────────────────────────────────────
# IMAGEN 1 — FICHA TÉCNICA  (1080×1500 px)
# ─────────────────────────────────────────────────────────────────────────────
def render_ratings(data: dict, team_key: str, players: list,
                   code: str, pal_key: str):
    pal   = get_paleta(pal_key)
    BG    = pal['bg_primary']
    BG2   = pal['bg_secondary']
    ACC   = pal['accent']
    ACC2  = pal['accent2']
    WHITE = pal['text_primary']
    GRAY  = pal['text_secondary']
    HIGH  = pal['cell_high']
    BRAND = pal['brand_color']

    is_home   = (team_key == 'home')
    team_name = data['home_team'] if is_home else data['away_team']
    team_e    = TEAM_ES.get(team_name, team_name)
    formation = data['formation_home'] if is_home else data['formation_away']
    avg_r     = data['avg_rating_home'] if is_home else data['avg_rating_away']
    score     = data['score_str']
    rival     = data['away_team'] if is_home else data['home_team']
    rival_e   = TEAM_ES.get(rival, rival)
    file_slug = team_name.lower().replace(' ', '_')

    starters = _get_starters(players)
    if not starters:
        print(f'  [warn] sin titulares: {team_name}'); return

    ratings = [p.get('rating') or 0.0 for p in starters]
    max_r   = max(ratings) if ratings else 0.0
    top_idx = ratings.index(max_r)
    top_p   = starters[top_idx]

    # ── Layout constants ──────────────────────────────────────────────────
    FIG_W, FIG_H = 7.2, 10.0   # 1080×1500 px @150dpi

    N         = len(starters)
    ROW_H     = 0.048
    ROW_GAP   = 0.006
    FOOTER_H  = 0.022
    ROWS_Y    = FOOTER_H + 0.008
    ROWS_SPAN = N * ROW_H + (N - 1) * ROW_GAP

    FIGURA_Y  = ROWS_Y + ROWS_SPAN + 0.012
    FIGURA_H  = 0.200

    HEADER_Y  = FIGURA_Y + FIGURA_H + 0.010
    HEADER_H  = 0.130

    # Column layout inside each row (axes fraction)
    SHIRT_X   = 0.022
    FLAG_CX   = 0.064
    NAME_X    = 0.104
    POS_X     = 0.355
    MINS_X    = 0.410
    BAR_LEFT  = 0.438
    BAR_W     = 0.310
    BAR_H_FR  = 0.55   # fraction of row height used by bar
    RATING_X  = 0.775
    STAT_X    = 0.815

    # ── Figura ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _radial_bg(fig, BG, BG2)

    # ── HEADER ───────────────────────────────────────────────────────────
    hax = fig.add_axes([0, HEADER_Y, 1, HEADER_H])
    hax.set_xlim(0, 1); hax.set_ylim(0, 1); hax.axis('off')

    # Fondo degradado
    for fy, fc in [(0.00, BG), (0.38, BG2), (0.72, _shade(BG2, 1.20))]:
        hax.add_patch(mpatches.Rectangle((0, fy), 1, 0.35,
                      facecolor=fc, linewidth=0, transform=hax.transAxes))
    # Separador inferior
    hax.add_patch(mpatches.Rectangle((0, 0), 1, 0.025,
                  facecolor=ACC, linewidth=0, transform=hax.transAxes))

    # Bandera — alineada con Línea 1
    flag = get_escudo(team_name, size=(96, 64))
    if flag:
        hax.add_artist(AnnotationBbox(flag, (0.082, 0.76),
                       frameon=False, xycoords='axes fraction',
                       box_alignment=(0.5, 0.5)))

    shadow = [mpe.withStroke(linewidth=2, foreground='#000000')]

    # Línea 1: Nombre del equipo (Bebas Neue 54pt)
    hax.text(0.19, 0.82, team_e.upper(), color=WHITE, ha='left', va='center',
             transform=hax.transAxes, **bebas(54))

    # Línea 2: Formación + Rating prom (14pt, gris #888888) — mínimo 28px bajo Línea 1
    hax.text(0.19, 0.42, f'{formation}     Rating prom: {avg_r}',
             color='#888888', ha='left', va='center', fontsize=14,
             fontweight='bold', transform=hax.transAxes)

    # Línea 3: Rival + competición + fecha (12pt, gris)
    hax.text(0.19, 0.22, f'{rival_e}  ·  {data["league"]}  ·  {data["date"]}',
             color=GRAY, ha='left', va='center', fontsize=12,
             transform=hax.transAxes)

    # Score — derecha, centrado verticalmente en Línea 1
    hax.text(0.975, 0.82, score, color=ACC, ha='right', va='center',
             transform=hax.transAxes, path_effects=shadow, **bebas(54))

    # ── FIGURA DEL PARTIDO ────────────────────────────────────────────────
    fax = fig.add_axes([0, FIGURA_Y, 1, FIGURA_H])
    fax.set_xlim(0, 1); fax.set_ylim(0, 1); fax.axis('off')

    figura_bg = _shade(BG2, 3.0)
    fax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
                  facecolor=figura_bg, linewidth=0, transform=fax.transAxes))
    # Borde superior (acento HIGH)
    fax.add_patch(mpatches.Rectangle((0, 0.960), 1, 0.040,
                  facecolor=HIGH, linewidth=0, transform=fax.transAxes))
    # Borde inferior separador
    fax.add_patch(mpatches.Rectangle((0, 0), 1, 0.018,
                  facecolor=ACC, linewidth=0, transform=fax.transAxes))

    # Subtítulo pequeño en la franja superior
    fax.text(0.50, 0.940, 'FIGURA DEL PARTIDO', color=figura_bg,
             ha='center', va='top', fontsize=7.5, fontweight='bold',
             transform=fax.transAxes)

    # Foto del jugador — zona izquierda, max 120×120px (PIL center-crop)
    PHOTO_SIZE = 120
    photo = _get_player_photo(top_p.get('id'), size=PHOTO_SIZE)
    if photo:
        fax.add_artist(AnnotationBbox(photo, (0.10, 0.48),
                       frameon=False, xycoords='axes fraction',
                       box_alignment=(0.5, 0.5), zorder=3))
        # right_x0: borde derecho de la foto + margen de 18px
        photo_half = (PHOTO_SIZE / 2) / (FIG_W * 150)
        right_x0   = 0.10 + photo_half + (18 / (FIG_W * 150))
    else:
        right_x0 = 0.025

    # Nombre del jugador (Bebas 42pt mínimo) — siempre delante de la foto
    fax.text(right_x0, 0.84, top_p['name'].upper(), color=HIGH,
             ha='left', va='center', transform=fax.transAxes,
             zorder=10, **bebas(42))

    # 2×2 grid de métricas — zona derecha, sin solapar foto
    metrics = _pos_metrics(top_p.get('stats') or {}, top_p.get('usual_position', 1))
    right_w = 1.0 - right_x0 - 0.015
    col_w   = right_w / 2

    # y_val = centro del valor, y_lbl = centro de la etiqueta
    metric_positions = [
        (right_x0 + col_w * 0.5, 0.60, 0.46),  # top-left
        (right_x0 + col_w * 1.5, 0.60, 0.46),  # top-right
        (right_x0 + col_w * 0.5, 0.30, 0.16),  # bottom-left
        (right_x0 + col_w * 1.5, 0.30, 0.16),  # bottom-right
    ]
    for (mx, vy, ly), (lbl, val) in zip(metric_positions, metrics):
        fax.text(mx, vy, str(val), color=WHITE, ha='center', va='center',
                 fontsize=36, fontweight='bold', transform=fax.transAxes,
                 **bebas(36))
        fax.text(mx, ly, lbl, color=GRAY, ha='center', va='center',
                 fontsize=13, transform=fax.transAxes)

    # ── FILAS DE JUGADORES ────────────────────────────────────────────────
    for i, player in enumerate(starters):
        row_y  = ROWS_Y + (N - 1 - i) * (ROW_H + ROW_GAP)
        rating = player.get('rating') or 0.0
        stats  = player.get('stats') or {}
        mins   = stats.get('minutes_played', 90) or 90
        pos_id = player.get('usual_position', 2)
        pos    = POS_MAP[pos_id]
        country= player.get('country', '')
        name   = player.get('name', '')
        shirt  = str(player.get('shirt_number', ''))
        is_top = (i == top_idx)

        pos_c  = POS_COLORS[pos]
        name_c = GOLD if is_top else WHITE
        row_bg = _shade(BG2, 1.18) if is_top else (BG if i % 2 == 0 else BG2)

        rax = fig.add_axes([0, row_y, 1, ROW_H])
        rax.set_facecolor('none')
        rax.set_xlim(0, 1); rax.set_ylim(0, 1); rax.axis('off')

        # Fondo con bordes redondeados
        rax.add_patch(mpatches.FancyBboxPatch(
            (0.004, 0.06), 0.992, 0.88,
            boxstyle='round,pad=0.01', facecolor=row_bg,
            linewidth=0, transform=rax.transAxes, clip_on=False, zorder=1))

        # Borde izquierdo dorado para jugador top
        if is_top:
            rax.add_patch(mpatches.Rectangle(
                (0.004, 0.06), 0.007, 0.88,
                facecolor=GOLD, linewidth=0,
                transform=rax.transAxes, clip_on=False, zorder=2))

        # Separador inferior
        rax.add_patch(mpatches.Rectangle(
            (0, 0), 1, 0.030, facecolor=SEP_C, linewidth=0,
            transform=rax.transAxes, clip_on=False, zorder=0))

        # Número de camiseta
        rax.text(SHIRT_X, 0.52, shirt, color=GRAY, ha='center', va='center',
                 fontsize=7, transform=rax.transAxes, zorder=3)

        # Bandera del país
        flag_p = get_escudo(country, size=(28, 19))
        if flag_p:
            rax.add_artist(AnnotationBbox(flag_p, (FLAG_CX, 0.52),
                           frameon=False, xycoords='axes fraction',
                           box_alignment=(0.5, 0.5), zorder=3))

        # Nombre
        rax.text(NAME_X, 0.52, name, color=name_c, ha='left', va='center',
                 fontsize=8.2, fontweight='bold' if is_top else 'normal',
                 transform=rax.transAxes, zorder=3)

        # Posición
        rax.text(POS_X, 0.52, pos, color=pos_c, ha='center', va='center',
                 fontsize=7.5, fontweight='bold', transform=rax.transAxes, zorder=3)

        # Minutos
        rax.text(MINS_X, 0.52, f"{mins}'", color=GRAY, ha='center', va='center',
                 fontsize=6.8, transform=rax.transAxes, zorder=3)

        # ── Barra de rating: sub-axes con barh, xlim=(5.0, 10.0) ──
        bar_y_fig = row_y + ROW_H * (1.0 - BAR_H_FR) / 2.0
        bar_h_fig = ROW_H * BAR_H_FR

        bax = fig.add_axes([BAR_LEFT, bar_y_fig, BAR_W, bar_h_fig])
        bax.set_facecolor('#1A1A1A')
        bax.set_xlim(5.0, 10.0)
        bax.set_ylim(0, 1)
        bax.axis('off')
        for sp in bax.spines.values():
            sp.set_visible(False)

        if rating > 5.0:
            bar_len = rating - 5.0
            # Capa principal: color primario (accent) con contorno brillante
            bax.barh(0.5, width=bar_len, height=0.75, left=5.0,
                     color=ACC, edgecolor=ACC2, linewidth=1.5, zorder=2)
            # Capa brillo: 35% del ancho total en accent2
            bax.barh(0.5, width=bar_len * 0.35, height=0.75,
                     left=5.0 + bar_len * 0.65,
                     color=ACC2, alpha=0.45, edgecolor='none', zorder=3)

        # Rating (posición fija, sobre rax con transAxes)
        rating_txt = f'{rating:.1f}' + (' ★' if is_top else '')
        rax.text(RATING_X, 0.52, rating_txt, color=name_c, ha='center', va='center',
                 fontsize=8.8, fontweight='bold' if is_top else 'normal',
                 transform=rax.transAxes, zorder=3)

        # Stats clave — 2 líneas separadas, 11pt mínimo
        l1, v1, l2, v2 = _pos_stats(stats, pos_id)
        rax.text(STAT_X, 0.72, f'{l1}: {v1}', color=GRAY, ha='left', va='center',
                 fontsize=11, transform=rax.transAxes, zorder=3)
        rax.text(STAT_X, 0.28, f'{l2}: {v2}', color=GRAY, ha='left', va='center',
                 fontsize=11, transform=rax.transAxes, zorder=3)

    # ── FOOTER ───────────────────────────────────────────────────────────
    ftax = fig.add_axes([0, 0, 1, FOOTER_H * 0.55])
    ftax.set_facecolor(BG2); ftax.axis('off')
    ftax.add_patch(mpatches.Rectangle((0, 0.88), 1, 0.12,
                   facecolor=ACC, linewidth=0, transform=ftax.transAxes))
    ftax.text(0.015, 0.50, 'Fuente: FotMob  ·  Ratings al finalizar el partido',
              color=GRAY, fontsize=6, ha='left', va='center',
              transform=ftax.transAxes)
    ftax.text(0.985, 0.50, 'MAU-STATISTICS', color=BRAND, ha='right', va='center',
              transform=ftax.transAxes,
              path_effects=[mpe.withStroke(linewidth=2, foreground='#000000')],
              **bebas(13))

    out_path = OUT_DIR / f'{code}_ratings_{file_slug}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1,
                facecolor=BG)
    plt.close(fig)
    print(f'  ✓ {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# IMAGEN 2 — TEAM STATS  (1080×1080 px)
# ─────────────────────────────────────────────────────────────────────────────
STATS_CFG = [
    ('total_shots',    'Tiros totales'),
    ('shotsontarget',  'Tiros a puerta'),
    ('passes',         'Pases'),
    ('accurate_passes','Pases precisos'),
    ('fouls',          'Faltas'),
    ('corners',        'Corners'),
    ('duel_won',       'Duelos ganados'),
    ('interceptions',  'Intercepciones'),
    ('offsides',       'Fueras de juego'),
    ('blocked_shots',  'Tiros bloqueados'),
]


def render_team_stats(data: dict, code: str, pal_key: str):
    pal   = get_paleta(pal_key)
    BG    = pal['bg_primary']
    BG2   = pal['bg_secondary']
    ACC   = pal['accent']
    ACC2  = pal['accent2']
    WHITE = pal['text_primary']
    GRAY  = pal['text_secondary']
    HIGH  = pal['cell_high']
    BRAND = pal['brand_color']

    home_name = data['home_team'];  home_e = TEAM_ES.get(home_name, home_name)
    away_name = data['away_team'];  away_e = TEAM_ES.get(away_name, away_name)
    score     = data['score_str']
    ts        = data.get('team_stats', {})

    raw_poss = ts.get('ballpossesion', [50, 50])
    h_poss   = float(raw_poss[0]) if isinstance(raw_poss, list) else 50.0
    a_poss   = float(raw_poss[1]) if isinstance(raw_poss, list) else 50.0

    rows = []
    for key, label in STATS_CFG:
        raw = ts.get(key)
        if raw is None:
            continue
        if isinstance(raw, list) and len(raw) == 2:
            h_n = _parse_num(raw[0]); a_n = _parse_num(raw[1])
            h_s = _fmt(raw[0]);       a_s = _fmt(raw[1])
        else:
            h_n = a_n = 0.0; h_s = a_s = '—'
        rows.append((label, h_s, a_s, h_n, a_n))

    # ── Figura 1080×1080 ─────────────────────────────────────────────────
    FIG_W = FIG_H = 7.2
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    _diag_bg(fig, BG, BG2)

    HEADER_H = 0.190
    FOOTER_H = 0.045
    POSS_H   = 0.155
    POSS_Y   = 1.0 - HEADER_H - POSS_H
    BODY_Y   = FOOTER_H + 0.006
    BODY_H   = POSS_Y - BODY_Y - 0.008
    N_ROWS   = len(rows)
    ROW_H    = BODY_H / N_ROWS if N_ROWS else 1

    # ── Header ───────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_xlim(0, 1); hax.set_ylim(0, 1); hax.axis('off')

    for fy, fc in [(0.00, BG), (0.38, BG2), (0.72, _shade(BG2, 1.20))]:
        hax.add_patch(mpatches.Rectangle((0, fy), 1, 0.35,
                      facecolor=fc, linewidth=0, transform=hax.transAxes))
    hax.add_patch(mpatches.Rectangle((0, 0), 1, 0.025,
                  facecolor=ACC, linewidth=0, transform=hax.transAxes))

    for tname, te, bar_c, xfl, xnm, halign in [
        (home_name, home_e, HIGH,  0.09, 0.17, 'left'),
        (away_name, away_e, ACC2,  0.91, 0.83, 'right'),
    ]:
        flag = get_escudo(tname, size=(64, 43))
        if flag:
            hax.add_artist(AnnotationBbox(flag, (xfl, 0.66),
                           frameon=False, xycoords='axes fraction',
                           box_alignment=(0.5, 0.5)))
        hax.text(xnm, 0.90, te.upper(), color=bar_c, ha=halign, va='top',
                 transform=hax.transAxes, **bebas(28))

    shadow = [mpe.withStroke(linewidth=3, foreground='#000000')]
    hax.text(0.50, 0.74, 'VS', color=GRAY, ha='center', va='center',
             fontsize=12, fontweight='bold', transform=hax.transAxes)
    hax.text(0.50, 0.46, score, color=WHITE, ha='center', va='center',
             transform=hax.transAxes, path_effects=shadow, **bebas(44))
    hax.text(0.50, 0.16, f'{data["league"]}  ·  {data["date"]}',
             color=GRAY, ha='center', va='center', fontsize=8,
             transform=hax.transAxes)

    # ── Posesión — donut al doble, radio exterior 0.85 ───────────────────
    dax = fig.add_axes([0.26, POSS_Y + 0.008, 0.48, POSS_H - 0.012])
    dax.set_xlim(-1.15, 1.15); dax.set_ylim(-1.15, 1.15)
    dax.set_aspect('equal'); dax.axis('off')
    dax.set_facecolor('none')

    h_ang = 360.0 * (h_poss / 100.0)
    dax.add_patch(mpatches.Wedge((0, 0), 0.85, 90, 90 + h_ang,
                  facecolor='#C8102E', width=0.42, linewidth=0))
    dax.add_patch(mpatches.Wedge((0, 0), 0.85, 90 + h_ang, 450,
                  facecolor=ACC2, width=0.42, linewidth=0))
    dax.text(0, 0, 'Posesión', color=GRAY, ha='center', va='center',
             fontsize=6.5, fontweight='bold')

    for xfrac, pct, color, te in [
        (0.055, h_poss, '#C8102E', home_e),
        (0.945, a_poss, ACC2,      away_e),
    ]:
        lax = fig.add_axes([xfrac - 0.115, POSS_Y, 0.230, POSS_H])
        lax.set_facecolor('none'); lax.axis('off')
        lax.text(0.5, 0.68, f'{pct:.0f}%', color=color, ha='center', va='center',
                 fontsize=22, fontweight='bold', transform=lax.transAxes,
                 **bebas(24))
        lax.text(0.5, 0.28, te, color=GRAY, ha='center', va='center',
                 fontsize=7.5, transform=lax.transAxes)

    # ── Filas de stats ────────────────────────────────────────────────────
    CTR_W    = 0.240
    CTR_X    = 0.380
    BAR_MAX  = 0.300
    HOME_END = CTR_X
    AWAY_STA = CTR_X + CTR_W
    BAR_H_FR = 0.75
    MIN_BAR  = 0.005   # mínimo en coordenadas de bar_axes

    for i, (label, h_s, a_s, h_n, a_n) in enumerate(rows):
        row_y  = BODY_Y + (N_ROWS - 1 - i) * ROW_H
        row_bg = BG if i % 2 == 0 else _shade(BG, 1.18)

        stat_max = max(h_n, a_n)
        h_bw = max(BAR_MAX * (h_n / stat_max), MIN_BAR) if stat_max > 0 else MIN_BAR
        a_bw = max(BAR_MAX * (a_n / stat_max), MIN_BAR) if stat_max > 0 else MIN_BAR

        rax = fig.add_axes([0, row_y, 1, ROW_H])
        rax.set_facecolor(row_bg)
        rax.set_xlim(0, 1); rax.set_ylim(0, 1); rax.axis('off')

        rax.add_patch(mpatches.Rectangle((0, 0), 1, 0.025,
                      facecolor=SEP_C, linewidth=0,
                      transform=rax.transAxes, clip_on=False))

        # ── Sub-axes para barras ──
        bar_y_fig = row_y + ROW_H * (1.0 - BAR_H_FR) / 2.0
        bar_h_fig = ROW_H * BAR_H_FR

        # Home bar — eje invertido para que la barra crezca hacia la izquierda
        hbax = fig.add_axes([HOME_END - BAR_MAX, bar_y_fig, BAR_MAX, bar_h_fig])
        hbax.set_facecolor('#1A1A1A')
        hbax.set_xlim(BAR_MAX, 0)   # invertido: 0 en derecha (centro de figura)
        hbax.set_ylim(0, 1)
        hbax.axis('off')

        hbax.barh(0.5, width=h_bw, height=0.75, left=0,
                  color=ACC, edgecolor=ACC2, linewidth=1.5, zorder=2)
        hbax.barh(0.5, width=h_bw * 0.35, height=0.75, left=h_bw * 0.65,
                  color=ACC2, alpha=0.45, edgecolor='none', zorder=3)

        # Away bar — izquierda a derecha desde centro
        abax = fig.add_axes([AWAY_STA, bar_y_fig, BAR_MAX, bar_h_fig])
        abax.set_facecolor('#1A1A1A')
        abax.set_xlim(0, BAR_MAX)
        abax.set_ylim(0, 1)
        abax.axis('off')

        abax.barh(0.5, width=a_bw, height=0.75, left=0,
                  color=ACC2, edgecolor=_shade(ACC2, 1.35), linewidth=1.5, zorder=2)
        abax.barh(0.5, width=a_bw * 0.35, height=0.75, left=a_bw * 0.65,
                  color=_shade(ACC2, 1.35), alpha=0.45, edgecolor='none', zorder=3)

        # Etiqueta central
        rax.text(CTR_X + CTR_W / 2, 0.50, label, color=GRAY,
                 ha='center', va='center', fontsize=13, fontweight='bold',
                 transform=rax.transAxes)

        # Valores numéricos — 110px mínimo reservado; 13pt si >8 chars
        h_fs = 13 if len(str(h_s)) > 8 else 16
        a_fs = 13 if len(str(a_s)) > 8 else 16
        rax.text(HOME_END - BAR_MAX - 0.010, 0.50, h_s, color=WHITE,
                 ha='right', va='center', fontsize=h_fs, fontweight='bold',
                 transform=rax.transAxes)
        rax.text(AWAY_STA + BAR_MAX + 0.010, 0.50, a_s, color=ACC2,
                 ha='left', va='center', fontsize=a_fs, fontweight='bold',
                 transform=rax.transAxes)

    # ── Footer ───────────────────────────────────────────────────────────
    ftax = fig.add_axes([0, 0, 1, FOOTER_H * 0.55])
    ftax.set_facecolor(BG2); ftax.axis('off')
    ftax.add_patch(mpatches.Rectangle((0, 0.88), 1, 0.12,
                   facecolor=ACC, linewidth=0, transform=ftax.transAxes))
    ftax.text(0.015, 0.50, 'Fuente: FotMob',
              color=GRAY, fontsize=6, ha='left', va='center',
              transform=ftax.transAxes)
    ftax.text(0.985, 0.50, 'MAU-STATISTICS', color=BRAND, ha='right', va='center',
              transform=ftax.transAxes,
              path_effects=[mpe.withStroke(linewidth=2, foreground='#000000')],
              **bebas(13))

    out_path = OUT_DIR / f'{code}_team_stats.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1,
                facecolor=BG)
    plt.close(fig)
    print(f'  ✓ {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default=str(
        BASE / 'data/raw/fotmob/Mexico_Portugal_2026-03-29.json'))
    parser.add_argument('--code', default='MEX_POR')
    args = parser.parse_args()

    p = Path(args.json)
    if not p.exists():
        print(f'ERROR: {p}', file=sys.stderr); sys.exit(1)

    with open(p, encoding='utf-8') as f:
        data = json.load(f)

    print(f'\n{"═"*56}')
    print(f'  {data["home_team"]} vs {data["away_team"]}  —  {data["score_str"]}')
    print(f'{"═"*56}')
    print(f'  Paleta activa: {PALETA_ACTIVA}\n')

    print('Generando ratings local...')
    render_ratings(data, 'home', data['players_home'], args.code, PALETA_ACTIVA)

    print('Generando ratings visitante...')
    render_ratings(data, 'away', data['players_away'], args.code, PALETA_ACTIVA)

    print('Generando team stats...')
    render_team_stats(data, args.code, PALETA_ACTIVA)

    print('\nListo.')


if __name__ == '__main__':
    main()
