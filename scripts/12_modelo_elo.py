#!/usr/bin/env python3
"""
12_modelo_elo.py  –  Sistema de rating ELO para Liga MX
Procesa los 32 torneos históricos (2010/11 → 2025/26) y genera:
  1. data/processed/elo_historico.csv   – historial completo de ELO
  2. output/charts/elo_evolucion.png    – líneas de evolución (18 equipos actuales)
  3. output/charts/elo_ranking.png      – tabla de ranking ELO actual
"""

import json, glob, math, warnings, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from PIL import Image
import urllib.request

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETTE, PALETAS, PALETA_ACTIVA, get_paleta, bebas, hex_rgba, hex_rgb, darken, make_h_gradient

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
HIST_DIR    = BASE / 'data/raw/historico'
OUT_CSV     = BASE / 'data/processed/elo_historico.csv'
OUT_DIR     = BASE / 'output/charts'
IMG_TEAMS   = BASE / 'data/raw/images/teams'
# Fuente centralizada desde config_visual (incluye assets/fonts/ como candidato)
try:
    from config_visual import BEBAS_TTF, BEBAS_AVAILABLE
except ImportError:
    BEBAS_TTF = None
    BEBAS_AVAILABLE = False

OUT_DIR.mkdir(parents=True, exist_ok=True)


def bebas(size):
    if BEBAS_TTF and BEBAS_TTF.exists():
        return {'fontproperties': FontProperties(fname=str(BEBAS_TTF), size=size)}
    return {'fontsize': size, 'fontweight': 'bold'}

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS ELO
# ─────────────────────────────────────────────────────────────────────────────
ELO_BASE    = 1500
K           = 32
HOME_ADV    = 100
SCALE       = 400
REGRESSION  = 0.30   # 30% regresión a la media al inicio de cada torneo

# ─────────────────────────────────────────────────────────────────────────────
# EQUIPOS ACTUALES (Clausura 2026) — colores y IDs FotMob
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_TEAMS = {
    'América':       {'id': 6576,    'color': '#FFD700'},
    'Atlas':         {'id': 6577,    'color': '#B22222'},
    'Chivas':        {'id': 7807,    'color': '#CD1F2D'},
    'Cruz Azul':     {'id': 6578,    'color': '#0047AB'},
    'FC Juárez':     {'id': 649424,  'color': '#4CAF50'},
    'León':          {'id': 1841,    'color': '#2D8C3C'},
    'Mazatlán':      {'id': 1170234, 'color': '#9B59B6'},
    'Monterrey':     {'id': 7849,    'color': '#003DA5'},
    'Necaxa':        {'id': 1842,    'color': '#D62828'},
    'Pachuca':       {'id': 7848,    'color': '#A8B8C8'},
    'Puebla':        {'id': 7847,    'color': '#2563EB'},
    'Pumas':         {'id': 1946,    'color': '#C8A84B'},
    'Querétaro':     {'id': 1943,    'color': '#1A7FCB'},
    'San Luis':      {'id': 6358,    'color': '#D52B1E'},
    'Santos Laguna': {'id': 7857,    'color': '#2E8B57'},
    'Tigres':        {'id': 8561,    'color': '#F5A623'},
    'Tijuana':       {'id': 162418,  'color': '#C62828'},
    'Toluca':        {'id': 6618,    'color': '#D5001C'},
}

# Normalización de nombres históricos → nombre canónico
NAME_MAP = {
    'CF America':         'América',
    'Atletico de San Luis': 'San Luis',
    'Queretaro FC':       'Querétaro',
    'FC Juarez':          'FC Juárez',
    'Mazatlan FC':        'Mazatlán',
    'Atletico Morelia':   'Morelia',
    'Jaguares Chiapas':   'Chiapas',
    'Leones Negros':      'Leones Negros',
    'Lobos de la BUAP':   'Lobos BUAP',
    'Mineros de Zacatecas': 'Mineros',
}

def normalize(name: str) -> str:
    return NAME_MAP.get(name, name)

# ─────────────────────────────────────────────────────────────────────────────
# MULTIPLICADOR POR MARGEN DE GOLES
# Basado en metodología 538/Club Elo: penaliza goleadas pero con rendimientos
# decrecientes para evitar que un 5-0 domine demasiado el cálculo.
# ─────────────────────────────────────────────────────────────────────────────
def goal_margin_multiplier(goles_local: int, goles_visit: int) -> float:
    diff = abs(goles_local - goles_visit)
    if diff == 0:
        return 1.0
    # log natural + 1 para rendimientos decrecientes
    return 1.0 + math.log(diff + 1) * 0.5

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: CARGAR Y ORDENAR TODOS LOS PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────
def load_all_matches() -> list[dict]:
    files = sorted(glob.glob(str(HIST_DIR / 'historico_20*.json')))
    matches = []
    for f in files:
        data = json.load(open(f, encoding='utf-8'))
        torneo = data.get('torneo', Path(f).stem)
        for p in data.get('partidos', []):
            if not p.get('terminado'):
                continue
            if p.get('goles_local') is None or p.get('goles_visit') is None:
                continue
            fecha_str = p.get('fecha', '')
            try:
                fecha = datetime.fromisoformat(fecha_str.replace('Z', '+00:00'))
            except Exception:
                continue
            matches.append({
                'fecha':       fecha,
                'torneo':      torneo,
                'local':       normalize(p['local']),
                'visitante':   normalize(p['visitante']),
                'goles_local': int(p['goles_local']),
                'goles_visit': int(p['goles_visit']),
            })

    matches.sort(key=lambda x: x['fecha'])
    print(f'✓ {len(matches)} partidos cargados ({matches[0]["fecha"].date()} → {matches[-1]["fecha"].date()})')
    return matches

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: CALCULAR ELO
# ─────────────────────────────────────────────────────────────────────────────
def calc_elo(matches: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    elo: dict[str, float] = {}          # estado actual
    history: list[dict]   = []          # historial para el CSV
    snapshots: list[dict] = []          # un snapshot por partido (para la gráfica)
    prev_torneo = None

    for m in matches:
        local     = m['local']
        visitante = m['visitante']
        torneo    = m['torneo']

        # Inicializar equipos nuevos
        if local not in elo:
            elo[local] = ELO_BASE
        if visitante not in elo:
            elo[visitante] = ELO_BASE

        # ── Regresión a la media al inicio de cada torneo nuevo ───────────────
        if torneo != prev_torneo:
            if prev_torneo is not None:
                for team in list(elo.keys()):
                    elo[team] = elo[team] + REGRESSION * (ELO_BASE - elo[team])
            prev_torneo = torneo

        # ── Calcular resultado esperado ───────────────────────────────────────
        elo_l = elo[local]
        elo_v = elo[visitante]
        # Local tiene ventaja de HOME_ADV puntos
        expected_l = 1 / (1 + 10 ** ((elo_v - elo_l - HOME_ADV) / SCALE))
        expected_v = 1 - expected_l

        # ── Resultado real ────────────────────────────────────────────────────
        gl, gv = m['goles_local'], m['goles_visit']
        if gl > gv:
            score_l, score_v = 1.0, 0.0
        elif gl < gv:
            score_l, score_v = 0.0, 1.0
        else:
            score_l, score_v = 0.5, 0.5

        # ── Multiplicador por margen ──────────────────────────────────────────
        mult = goal_margin_multiplier(gl, gv)

        # ── Actualizar ELO ────────────────────────────────────────────────────
        delta_l = K * mult * (score_l - expected_l)
        elo[local]     = round(elo[local]     + delta_l, 2)
        elo[visitante] = round(elo[visitante] - delta_l, 2)

        # ── Guardar en historial ──────────────────────────────────────────────
        fecha_date = m['fecha'].date()
        history.append({'fecha': fecha_date, 'equipo': local,     'elo': elo[local],     'torneo': torneo})
        history.append({'fecha': fecha_date, 'equipo': visitante, 'elo': elo[visitante], 'torneo': torneo})

        # Snapshot de todos los equipos actuales después de este partido
        snap = {'fecha': fecha_date, 'torneo': torneo}
        for t in CURRENT_TEAMS:
            snap[t] = round(elo.get(t, ELO_BASE), 1)
        snapshots.append(snap)

    return elo, history, snapshots

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE IMAGEN
# ─────────────────────────────────────────────────────────────────────────────
_HDRS = {'User-Agent': 'Mozilla/5.0'}

def get_shield(tid: int, size: int = 32) -> np.ndarray | None:
    dest = IMG_TEAMS / f'{tid}.png'
    url  = f'https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png'
    if not dest.exists():
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=_HDRS), timeout=6) as r:
                dest.write_bytes(r.read())
        except Exception:
            return None
    try:
        img = Image.open(dest).convert('RGBA').resize((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: VISUALIZACIÓN 1 — Evolución del ELO
# ─────────────────────────────────────────────────────────────────────────────
_pal      = get_paleta()
DARK_BG   = _pal['bg_primary']
WHITE     = _pal['text_primary']
GRAY      = _pal['text_secondary']
RED_BRAND = _pal['accent']

def plot_evolucion(snapshots: list[dict], final_elo: dict[str, float]):
    df = pd.DataFrame(snapshots)
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Resample semanal + suavizado rolling 8
    df = df.set_index('fecha')
    team_cols = list(CURRENT_TEAMS.keys())
    df_weekly = df[team_cols].resample('W').last().ffill()
    df_smooth = df_weekly.rolling(window=8, center=True, min_periods=1).mean().reset_index()
    df_weekly  = df_weekly.reset_index()

    ranking = sorted(final_elo.items(), key=lambda x: -x[1])
    top5 = [t for t, _ in ranking if t in CURRENT_TEAMS][:5]

    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)

    # Gradiente fondo
    grad = np.zeros((200, 2, 3))
    for i in range(200):
        t = i/199
        grad[i] = np.array([0x0a,0x0e,0x12])/255*(1-t) + np.array([0x13,0x1a,0x24])/255*t
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bg.axis('off')

    ax = fig.add_axes([0.055, 0.11, 0.80, 0.75])
    ax.set_facecolor('none')
    ax.yaxis.grid(True, color=DARK_BG, lw=0.6, zorder=0, alpha=0.7)
    ax.set_axisbelow(True); ax.xaxis.grid(False)

    # Línea 1500 — blanca punteada alpha=0.3
    ax.axhline(ELO_BASE, color='white', lw=1.2, ls='--', zorder=2, alpha=0.30)
    ax.text(df_smooth['fecha'].iloc[0], ELO_BASE + 8, '── 1500',
            color='#ffffff', alpha=0.35, fontsize=7.5, va='bottom', ha='left')

    # Bandas año sutiles
    for yr in range(2011, 2027, 2):
        ax.axvspan(pd.Timestamp(f'{yr}-01-01'), pd.Timestamp(f'{yr}-12-31'),
                   alpha=0.025, color='white', linewidth=0, zorder=1)

    fecha_ini = df_smooth['fecha'].iloc[0]
    fecha_fin = df_smooth['fecha'].iloc[-1]

    # ── Anotaciones históricas ────────────────────────────────────────────────
    annotations = [
        (pd.Timestamp('2020-03-15'), 'COVID-19\nLiga suspendida', '#5a6470'),
    ]
    for ann_date, ann_text, ann_color in annotations:
        ax.axvline(ann_date, color=ann_color, lw=0.9, ls=':', zorder=2, alpha=0.7)
        ax.text(ann_date - pd.Timedelta(days=5), 1272, ann_text,
                color=ann_color, fontsize=7, rotation=90,
                ha='right', va='bottom', alpha=0.85)

    # Pico histórico de Toluca (equipo #1)
    toluca_series = df_smooth['Toluca'].dropna()
    peak_idx  = toluca_series.idxmax()
    peak_date = df_smooth.loc[peak_idx, 'fecha']
    peak_val  = toluca_series.max()
    ax.plot(peak_date, peak_val, 'o',
            color=CURRENT_TEAMS['Toluca']['color'], ms=7, zorder=6,
            markeredgecolor='white', markeredgewidth=0.8)
    ax.annotate(f'Máx Toluca\n{peak_val:.0f}',
                xy=(peak_date, peak_val),
                xytext=(peak_date + pd.Timedelta(days=120), peak_val + 30),
                color=CURRENT_TEAMS['Toluca']['color'], fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color=CURRENT_TEAMS['Toluca']['color'],
                                lw=0.8),
                bbox=dict(boxstyle='round,pad=0.2', facecolor=DARK_BG,
                          edgecolor='none', alpha=0.7))

    # ── Curvas top 5 + fill ───────────────────────────────────────────────────
    raw_last_vals = {}
    for rank, team in enumerate(top5):
        color  = CURRENT_TEAMS[team]['color']
        series = df_smooth[team].values
        raw_last_vals[team] = float(df_smooth[team].dropna().iloc[-1])

        ax.fill_between(df_smooth['fecha'], series, ELO_BASE,
                        where=(series >= ELO_BASE), alpha=0.09,
                        color=color, zorder=3, interpolate=True)
        ax.fill_between(df_smooth['fecha'], series, ELO_BASE,
                        where=(series < ELO_BASE), alpha=0.04,
                        color=color, zorder=3, interpolate=True)
        ax.plot(df_smooth['fecha'], series,
                color=color, lw=2.5, alpha=0.95, zorder=5,
                solid_capstyle='round', solid_joinstyle='round')

    # ── Etiquetas con separación vertical mínima 22 ELO units ────────────────
    xlim_end = fecha_fin + pd.Timedelta(days=240)
    label_data = sorted(
        [(rank, team, raw_last_vals[team]) for rank, team in enumerate(top5)],
        key=lambda x: -x[2])

    MIN_SEP = 22
    adjusted = []
    for i, (rank, team, raw_y) in enumerate(label_data):
        y = raw_y
        if i > 0 and adjusted[-1][2] - y < MIN_SEP:
            y = adjusted[-1][2] - MIN_SEP
        adjusted.append((rank, team, y))

    for rank, team, label_y in adjusted:
        color = CURRENT_TEAMS[team]['color']
        elo_v = raw_last_vals[team]
        # Conector de la línea real al label
        ax.annotate('',
                    xy=(fecha_fin, elo_v),
                    xytext=(fecha_fin + pd.Timedelta(days=10), label_y),
                    arrowprops=dict(arrowstyle='-', color=color,
                                   lw=0.7, alpha=0.6),
                    annotation_clip=False)
        ax.annotate(
            f'  #{rank+1} {team}  {elo_v:.0f}  ',
            xy=(fecha_fin + pd.Timedelta(days=10), label_y),
            xytext=(fecha_fin + pd.Timedelta(days=15), label_y),
            color='white', fontsize=8.5, fontweight='bold',
            va='center', ha='left', annotation_clip=False,
            bbox=dict(boxstyle='round,pad=0.28',
                      facecolor=color, edgecolor='none', alpha=0.92))

    # Ejes
    ax.set_xlim(fecha_ini, xlim_end)
    ax.set_ylim(1260, 1800)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', which='both', bottom=False, length=0,
                   pad=6, colors=GRAY, labelsize=9)
    ax.tick_params(axis='y', colors=GRAY, labelsize=9, length=0, pad=4)
    ax.yaxis.set_major_locator(plt.MultipleLocator(100))
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY); spine.set_linewidth(0.6)

    # Títulos con gancho
    fig.text(0.055, 0.940, '15 AÑOS DE PODER EN LIGA MX',
             color=WHITE, ha='left', va='bottom', **bebas(32))
    fig.text(0.055, 0.910, 'Evolución del rating ELO · Top 5 equipos · 2010 – 2026  ·  5,250 partidos',
             color=GRAY, ha='left', va='bottom', fontsize=10)

    # Footer
    fig.text(0.055, 0.022, 'Fuente: FotMob · Histórico 2010-2026',
             color=GRAY, fontsize=10, ha='left', va='bottom')
    kw = dict(ha='right', va='bottom', **bebas(20))
    fig.text(0.958, 0.014, 'MAU-STATISTICS', color='#000000', alpha=0.50, **kw)
    fig.text(0.956, 0.024, 'MAU-STATISTICS', color=RED_BRAND, **kw)

    out = OUT_DIR / 'elo_evolucion.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'✓ Guardado: {out}')

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4: VISUALIZACIÓN 2 — Ranking ELO actual (tabla)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ranking(final_elo: dict[str, float]):
    ranking = [(t, final_elo.get(t, ELO_BASE)) for t in CURRENT_TEAMS]
    ranking.sort(key=lambda x: -x[1])
    n = len(ranking)

    FIG_W, FIG_H = 7.5, 11.5
    CONTENT_Y = 0.042
    CONTENT_H = 0.864
    ROW_H_N   = CONTENT_H / n
    AR_FIG    = FIG_H / FIG_W

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)

    # Gradiente fondo
    grad = np.zeros((200, 2, 3))
    for i in range(200):
        t = i/199
        grad[i] = np.array([0x0a,0x0e,0x12])/255*(1-t) + np.array([0x13,0x1a,0x24])/255*t
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bg.axis('off')

    # Título + subtítulo + línea explicativa (bien separados)
    fig.text(0.50, 0.990, '¿QUIÉN DOMINA LA LIGA MX?',
             color=WHITE, ha='center', va='top', **bebas(30))
    fig.text(0.50, 0.954, 'Rating ELO acumulado · 15 años · 5,250 partidos · Base 1500',
             color=GRAY, ha='center', va='top', fontsize=9)
    fig.text(0.50, 0.934,
             'El rating ELO mide la fuerza histórica de cada equipo. '
             'Se actualiza partido a partido considerando rival, localía y margen de goles.',
             color=GRAY, ha='center', va='top', fontsize=7.2)
    fig.text(0.50, 0.920,
             'Base: 1500 = promedio de la liga.',
             color=GRAY, ha='center', va='top', fontsize=7.2)

    # Eje principal
    ax = fig.add_axes([0.02, CONTENT_Y, 0.96, CONTENT_H])
    ax.set_facecolor('none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis('off')

    ELO_BAR_MIN = 1300
    elos      = [e for _, e in ranking]
    elo_max   = max(elos)
    elo_range = max(elo_max - ELO_BAR_MIN, 1)

    # Columnas
    X_POS     = 0.012   # número de posición
    X_BADGE   = 0.068   # escudo (55px)
    X_NAME    = 0.188   # nombre
    BAR_X     = 0.420
    BAR_MAX_W = 0.365
    X_ELO     = BAR_X + BAR_MAX_W + 0.022
    X_DELTA   = 0.985

    BADGE_SIZE = 55

    for i, (team, elo_v) in enumerate(ranking):
        row_bot = n - 1 - i
        color   = CURRENT_TEAMS[team]['color']
        is_top3 = i < 3

        # Fondo de fila — Top 3 iluminado
        if is_top3:
            row_bg = '#1a1f2e'
        elif i % 2 == 0:
            row_bg = _pal['bg_secondary']
        else:
            row_bg = DARK_BG

        ax.add_patch(mpatches.Rectangle(
            (0, row_bot), 1, 1, facecolor=row_bg, linewidth=0, zorder=1))

        # Separador sutil
        ax.axhline(row_bot + 1, color=DARK_BG, lw=0.6, zorder=2)

        # Acento lateral Top 3
        if is_top3:
            ax.add_patch(mpatches.Rectangle(
                (0, row_bot), 0.004, 1,
                facecolor=color, linewidth=0, zorder=3, alpha=0.85))

        # ── Posición ─────────────────────────────────────────────────────────
        pos_color = RED_BRAND if is_top3 else (WHITE if i < 8 else GRAY)
        ax.text(X_POS, row_bot + 0.50, f'#{i+1}',
                color=pos_color,
                fontsize=8.5 if is_top3 else 7.5,
                fontweight='bold' if is_top3 else 'normal',
                ha='left', va='center', zorder=4)

        # ── Escudo 55×55 ─────────────────────────────────────────────────────
        tid  = CURRENT_TEAMS[team]['id']
        sarr = get_shield(tid, BADGE_SIZE)
        if sarr is not None:
            badge_h_fig = ROW_H_N * 0.74
            badge_w_fig = badge_h_fig * AR_FIG
            badge_y_fig = CONTENT_Y + (n-1-i)*ROW_H_N + ROW_H_N*0.13
            sh_ax = fig.add_axes([X_BADGE, badge_y_fig, badge_w_fig, badge_h_fig])
            sh_ax.set_facecolor('#f8f8fc')
            sh_ax.imshow(sarr, aspect='equal')
            sh_ax.axis('off')

        # ── Nombre ───────────────────────────────────────────────────────────
        ax.text(X_NAME, row_bot + 0.52, team.upper(),
                color=WHITE if is_top3 else '#d0d7de',
                ha='left', va='center', zorder=4,
                **bebas(11 if is_top3 else 10))

        # ── Track + Barra con gradiente horizontal ────────────────────────────
        bar_h = 0.50
        bar_y = row_bot + (1 - bar_h) / 2
        bar_w = BAR_MAX_W * max(elo_v - ELO_BAR_MIN, 0) / elo_range

        # Track oscuro
        ax.add_patch(mpatches.Rectangle(
            (BAR_X, bar_y), BAR_MAX_W, bar_h,
            facecolor='#101520', linewidth=0, zorder=3))

        # Barra con gradiente via imshow
        if bar_w > 0.005:
            grad_img = make_h_gradient(color, width=256, dark_factor=0.50)
            ax.imshow(
                grad_img,
                extent=[BAR_X, BAR_X + bar_w, bar_y, bar_y + bar_h],
                aspect='auto', zorder=4, origin='lower',
                interpolation='bilinear')

        # ── ELO (negrita, color equipo para top 5) ────────────────────────────
        ax.text(X_ELO, row_bot + 0.55,
                f'{elo_v:.0f}',
                color=color if i < 5 else WHITE,
                fontsize=10.5 if is_top3 else 9.5,
                fontweight='bold' if i < 5 else 'normal',
                ha='left', va='center', zorder=4)

        # ── Delta Δ1500 ───────────────────────────────────────────────────────
        delta = elo_v - ELO_BASE
        sign  = '+' if delta >= 0 else ''
        ax.text(X_DELTA, row_bot + 0.55,
                f'{sign}{delta:.0f}',
                color=_pal['cell_high'] if delta >= 0 else _pal['accent'],
                fontsize=8.5, ha='right', va='center', zorder=4)

    # Footer
    fig.text(0.04, 0.013, 'Fuente: FotMob · Histórico 2010-2026',
             color=GRAY, fontsize=10, ha='left', va='bottom')
    kw = dict(ha='right', va='bottom', **bebas(20))
    fig.text(0.970, 0.005, 'MAU-STATISTICS', color='#000000', alpha=0.5, **kw)
    fig.text(0.968, 0.015, 'MAU-STATISTICS', color=RED_BRAND, **kw)

    out = OUT_DIR / 'elo_ranking.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'✓ Guardado: {out}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument('--paleta', default=None, choices=list(PALETAS.keys()))
    _args = _parser.parse_args()
    if _args.paleta:
        _p = get_paleta(_args.paleta)
        DARK_BG   = _p['bg_primary']
        WHITE     = _p['text_primary']
        GRAY      = _p['text_secondary']
        RED_BRAND = _p['accent']
        _pal      = _p

    # 1. Cargar partidos
    matches = load_all_matches()

    # 2. Calcular ELO
    print('Calculando ELO...')
    final_elo, history, snapshots = calc_elo(matches)

    # 3. Guardar CSV historial
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(OUT_CSV, index=False)
    print(f'✓ Historial guardado: {OUT_CSV}  ({len(df_hist):,} filas)')

    # 4. Graficar evolución
    print('Generando gráfica de evolución...')
    plot_evolucion(snapshots, final_elo)

    # 5. Graficar ranking
    print('Generando tabla de ranking...')
    plot_ranking(final_elo)

    # 6. Imprimir Top 5 y Bottom 5
    ranking_current = [(t, final_elo.get(t, ELO_BASE))
                       for t in CURRENT_TEAMS]
    ranking_current.sort(key=lambda x: -x[1])

    print('\n' + '='*45)
    print('  RANKING ELO — LIGA MX · Clausura 2026')
    print('='*45)
    print(f'  {"#":<4} {"Equipo":<20} {"ELO":>6}  {"Δ1500":>6}')
    print('  ' + '-'*41)
    for i, (team, elo_v) in enumerate(ranking_current[:5]):
        delta = elo_v - ELO_BASE
        sign  = '+' if delta >= 0 else ''
        print(f'  {i+1:<4} {team:<20} {elo_v:>6.0f}  {sign}{delta:>5.0f}')
    print('  ...')
    for i, (team, elo_v) in enumerate(ranking_current[-5:]):
        pos   = len(ranking_current) - 4 + i
        delta = elo_v - ELO_BASE
        sign  = '+' if delta >= 0 else ''
        print(f'  {pos:<4} {team:<20} {elo_v:>6.0f}  {sign}{delta:>5.0f}')
    print('='*45)
