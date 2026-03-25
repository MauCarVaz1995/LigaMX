#!/usr/bin/env python3
"""
12_modelo_elo.py  –  Sistema de rating ELO para Liga MX
Procesa los 32 torneos históricos (2010/11 → 2025/26) y genera:
  1. data/processed/elo_historico.csv   – historial completo de ELO
  2. output/charts/elo_evolucion.png    – líneas de evolución (18 equipos actuales)
  3. output/charts/elo_ranking.png      – tabla de ranking ELO actual
"""

import json, glob, math, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from PIL import Image
import urllib.request

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
HIST_DIR    = BASE / 'data/raw/historico'
OUT_CSV     = BASE / 'data/processed/elo_historico.csv'
OUT_DIR     = BASE / 'output/charts'
IMG_TEAMS   = BASE / 'data/raw/images/teams'
BEBAS_TTF   = Path.home() / '.fonts/BebasNeue.ttf'

OUT_DIR.mkdir(parents=True, exist_ok=True)

if BEBAS_TTF.exists():
    fm.fontManager.addfont(str(BEBAS_TTF))

def bebas(size):
    if BEBAS_TTF.exists():
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
DARK_BG   = '#0d1117'
WHITE     = '#e6edf3'
GRAY      = '#8b949e'
RED_BRAND = '#D5001C'

def plot_evolucion(snapshots: list[dict], final_elo: dict[str, float]):
    df = pd.DataFrame(snapshots)
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Resample semanal para suavizar la curva
    df = df.set_index('fecha')
    team_cols = list(CURRENT_TEAMS.keys())
    df_weekly = df[team_cols].resample('W').last().ffill()
    df_weekly = df_weekly.reset_index()

    # Top 5 por ELO final — resaltados
    ranking = sorted(final_elo.items(), key=lambda x: -x[1])
    top5    = [t for t, _ in ranking if t in CURRENT_TEAMS][:5]

    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    ax  = fig.add_axes([0.06, 0.10, 0.88, 0.78])
    ax.set_facecolor(DARK_BG)

    # Fondo con bandas sutiles por era (cada 4 torneos ≈ 2 años)
    for yr in range(2011, 2027, 2):
        ax.axvspan(pd.Timestamp(f'{yr}-01-01'), pd.Timestamp(f'{yr}-12-31'),
                   alpha=0.04, color='white', linewidth=0)

    # Línea base ELO
    ax.axhline(ELO_BASE, color='#2d333b', lw=1.0, ls='--', zorder=1)

    # Dibujar todas las líneas
    for team in team_cols:
        color  = CURRENT_TEAMS[team]['color']
        is_top = team in top5
        lw     = 2.4 if is_top else 0.9
        alpha  = 1.0 if is_top else 0.35
        zorder = 4  if is_top else 2

        ax.plot(df_weekly['fecha'], df_weekly[team],
                color=color, lw=lw, alpha=alpha, zorder=zorder)

        # Etiqueta al final de la línea (solo top5)
        if is_top:
            last_val = df_weekly[team].dropna().iloc[-1]
            ax.text(df_weekly['fecha'].iloc[-1] + pd.Timedelta(days=18),
                    last_val, team,
                    color=color, fontsize=9, fontweight='bold',
                    va='center', ha='left')

    # Formato de ejes
    ax.set_xlim(df_weekly['fecha'].min(), df_weekly['fecha'].max() + pd.Timedelta(days=100))
    ax.set_ylim(1300, 1750)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.spines[:].set_color('#2d333b')
    ax.yaxis.set_tick_params(colors=GRAY)
    ax.xaxis.set_tick_params(colors=GRAY)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d333b')

    # Título
    fig.text(0.06, 0.935, 'EVOLUCIÓN ELO — LIGA MX',
             color=WHITE, ha='left', va='bottom', **bebas(26))
    fig.text(0.06, 0.908, '32 torneos · 2010/11 – 2025/26 · Top 5 resaltados',
             color=GRAY, ha='left', va='bottom', fontsize=10)

    # Leyenda top 5 (esquina superior derecha)
    legend_x = 0.947
    for i, team in enumerate(top5):
        color = CURRENT_TEAMS[team]['color']
        elo_v = final_elo.get(team, ELO_BASE)
        fig.text(legend_x, 0.88 - i*0.046,
                 f'#{i+1}  {team}',
                 color=color, ha='right', va='center',
                 fontsize=9, fontweight='bold')

    # Footer
    fig.text(0.06, 0.020, 'Fuente: FotMob · Histórico 2010-2026',
             color=GRAY, fontsize=8.5, ha='left', va='bottom')
    kw = dict(ha='right', va='bottom', **bebas(18))
    fig.text(0.953, 0.013, 'MAU-STATISTICS', color='#000000', alpha=0.5,  **kw)
    fig.text(0.951, 0.022, 'MAU-STATISTICS', color=RED_BRAND, **kw)

    out = OUT_DIR / 'elo_evolucion.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'✓ Guardado: {out}')

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4: VISUALIZACIÓN 2 — Ranking ELO actual (tabla)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ranking(final_elo: dict[str, float]):
    # Solo los 18 equipos actuales, ordenados
    ranking = [(t, final_elo.get(t, ELO_BASE))
               for t in CURRENT_TEAMS]
    ranking.sort(key=lambda x: -x[1])

    n = len(ranking)
    FIG_W, FIG_H = 7, 10
    ROW_H = 1 / (n + 3)   # +3 para header + footer

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)

    # Fondo gradiente
    grad = np.zeros((200, 2, 3))
    bot = np.array([0x0a,0x0e,0x12])/255
    top = np.array([0x13,0x1a,0x24])/255
    for i in range(200):
        t = i/199
        grad[i] = bot*(1-t) + top*t
    bg = fig.add_axes([0,0,1,1])
    bg.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bg.axis('off')

    # Título
    fig.text(0.50, 0.965, 'RANKING ELO — LIGA MX',
             color=WHITE, ha='center', va='top', **bebas(28))
    fig.text(0.50, 0.940, 'Clausura 2026 · Rating al día de hoy',
             color=GRAY, ha='center', va='top', fontsize=9.5)

    # Eje principal para la tabla
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])
    ax.set_facecolor('none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis('off')

    # ELO máximo y mínimo para normalizar las barras
    elos = [e for _, e in ranking]
    elo_max = max(elos)
    elo_min = min(elos)
    elo_range = elo_max - elo_min if elo_max != elo_min else 1

    BAR_MAX_W  = 0.46
    BAR_X      = 0.40
    BAR_H_FRAC = 0.60
    BADGE_W    = 0.055
    BADGE_X    = 0.025

    for i, (team, elo_v) in enumerate(ranking):
        row_y = n - 1 - i   # fila 0 = rank 1 (arriba)
        color = CURRENT_TEAMS[team]['color']

        # Fondo alterno
        bg_row = '#0f151e' if i % 2 == 0 else DARK_BG
        ax.add_patch(mpatches.Rectangle(
            (0, row_y), 1, 1,
            facecolor=bg_row, linewidth=0, zorder=1))

        # Separador top
        ax.axhline(row_y + 1, color='#1e2530', lw=0.5, zorder=2)

        # Posición
        pos_color = RED_BRAND if i < 3 else (WHITE if i < 8 else GRAY)
        ax.text(0.018, row_y + 0.50, f'#{i+1}',
                color=pos_color, fontsize=9,
                fontweight='bold' if i < 3 else 'normal',
                ha='left', va='center', zorder=3)

        # Escudo
        tid  = CURRENT_TEAMS[team]['id']
        sarr = get_shield(tid, 36)
        if sarr is not None:
            sh_ax = fig.add_axes([
                BADGE_X,
                0.04 + (n - 1 - i) * (0.88/n) + (0.88/n)*0.15,
                BADGE_W * (FIG_H/FIG_W),
                (0.88/n) * 0.70
            ])
            sh_ax.set_facecolor('#f8f8fc')
            sh_ax.imshow(sarr, aspect='equal')
            sh_ax.axis('off')

        # Nombre del equipo
        ax.text(0.135, row_y + 0.52, team.upper(),
                color=WHITE, fontsize=9, fontweight='bold',
                ha='left', va='center', zorder=3, **bebas(11))

        # Barra proporcional
        bar_w = BAR_MAX_W * (elo_v - elo_min) / elo_range
        bar_h = BAR_H_FRAC
        bar_y = row_y + (1 - bar_h) / 2
        # Track
        ax.add_patch(mpatches.Rectangle(
            (BAR_X, bar_y), BAR_MAX_W, bar_h,
            facecolor='#1a1f28', linewidth=0, zorder=3))
        # Barra coloreada
        if bar_w > 0.002:
            ax.add_patch(mpatches.Rectangle(
                (BAR_X, bar_y), bar_w, bar_h,
                facecolor=color, linewidth=0, zorder=4, alpha=0.90))

        # Valor ELO
        ax.text(BAR_X + BAR_MAX_W + 0.016, row_y + 0.50,
                f'{elo_v:.0f}',
                color=color if i < 5 else WHITE,
                fontsize=9.5,
                fontweight='bold' if i < 5 else 'normal',
                ha='left', va='center', zorder=3)

        # Delta desde 1500
        delta = elo_v - ELO_BASE
        sign  = '+' if delta >= 0 else ''
        ax.text(0.97, row_y + 0.50,
                f'{sign}{delta:.0f}',
                color='#2ea043' if delta >= 0 else '#f85149',
                fontsize=8, ha='right', va='center', zorder=3)

    # Footer
    fig.text(0.04, 0.010, 'Fuente: FotMob · Histórico 2010-2026',
             color=GRAY, fontsize=8, ha='left', va='bottom')
    kw = dict(ha='right', va='bottom', **bebas(18))
    fig.text(0.969, 0.004, 'MAU-STATISTICS', color='#000000', alpha=0.5, **kw)
    fig.text(0.967, 0.012, 'MAU-STATISTICS', color=RED_BRAND, **kw)

    out = OUT_DIR / 'elo_ranking.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'✓ Guardado: {out}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
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
