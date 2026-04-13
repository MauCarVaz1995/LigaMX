#!/usr/bin/env python3
"""
11_modelo_prediccion.py
Modelo de Poisson para predicción de marcadores – Liga MX
Uso: python 11_modelo_prediccion.py "Pachuca" "Toluca"
"""

import sys
import json
import glob
import math
import warnings
import urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw
from scipy.stats import poisson

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETAS, PALETA_ACTIVA, get_paleta

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

# ── Bebas Neue ───────────────────────────────────────────────────────────────
_bebas_path = str(BEBAS_TTF) if BEBAS_TTF.exists() else None
if _bebas_path:
    try:
        fm.fontManager.addfont(_bebas_path)
    except Exception:
        _bebas_path = None

def bebas_fp(size):
    if _bebas_path:
        p = FontProperties(fname=_bebas_path, size=size)
        return {'fontproperties': p}
    return {'fontsize': size, 'fontweight': 'bold'}

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
_pal      = get_paleta()
DARK_BG   = _pal['bg_primary']
WHITE     = _pal['text_primary']
GRAY      = _pal['text_secondary']
RED_BRAND = _pal['accent']
GOLD      = _pal['accent2']

TEAM_COLORS = {
    'América': '#FFD700',      'CF America': '#FFD700',
    'Chivas': '#CE1141',       'Guadalajara': '#CE1141',
    'Cruz Azul': '#0038A8',
    'Tigres': '#FF7900',       'Tigres UANL': '#FF7900',
    'Monterrey': '#003F8A',    'CF Monterrey': '#003F8A',
    'Pumas': '#002B5B',        'Pumas UNAM': '#002B5B',
    'Toluca': '#D5001C',
    'Santos Laguna': '#006341','Santos': '#006341',
    'Pachuca': '#004D8E',
    'Atlas': '#CC2200',
    'León': '#006633',
    'Necaxa': '#CC0000',
    'Tijuana': '#1A1A1A',
    'Querétaro': '#0033A0',    'Queretaro FC': '#0033A0',
    'FC Juárez': '#C8102E',    'Juárez': '#C8102E', 'Juárez FC': '#C8102E',
    'Mazatlán': '#D47900',     'Mazatlán FC': '#D47900',
    'San Luis': '#006633',     'Atletico de San Luis': '#006633',
    'Puebla': '#00479D',
    'Atlante': '#001489',
    'Veracruz': '#003087',
    'Morelia': '#6600CC',
    'Lobos BUAP': '#003087',
    'Dorados': '#FFD700',
    'Chiapas': '#005B20',
    'Jaguares': '#005B20',
    'Estudiantes': '#002366',
}

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE NOMBRES DE EQUIPOS
# ─────────────────────────────────────────────────────────────────────────────
# Mapea variantes históricas al nombre canónico
_ALIAS = {
    'cf america': 'América',         'america': 'América',
    'cf américa': 'América',         'américa': 'América',
    'chivas': 'Chivas',              'guadalajara': 'Chivas',
    'cruz azul': 'Cruz Azul',
    'tigres': 'Tigres',              'tigres uanl': 'Tigres',
    'monterrey': 'Monterrey',        'cf monterrey': 'Monterrey', 'rayados': 'Monterrey',
    'pumas': 'Pumas',                'pumas unam': 'Pumas',       'unam': 'Pumas',
    'toluca': 'Toluca',
    'santos laguna': 'Santos Laguna','santos': 'Santos Laguna',
    'pachuca': 'Pachuca',            'cf pachuca': 'Pachuca',
    'atlas': 'Atlas',
    'león': 'León',                  'leon': 'León',
    'necaxa': 'Necaxa',
    'tijuana': 'Tijuana',            'xolos': 'Tijuana',
    'querétaro': 'Querétaro',        'queretaro': 'Querétaro',    'queretaro fc': 'Querétaro',
    'fc juárez': 'FC Juárez',        'fc juarez': 'FC Juárez',    'juárez': 'FC Juárez', 'juarez': 'FC Juárez',
    'mazatlán': 'Mazatlán',          'mazatlan': 'Mazatlán',      'mazatlán fc': 'Mazatlán', 'mazatlan fc': 'Mazatlán',
    'atletico de san luis': 'San Luis','atlético de san luis': 'San Luis','san luis': 'San Luis',
    'puebla': 'Puebla',
    'atlante': 'Atlante',
    'veracruz': 'Veracruz',          'tiburones rojos': 'Veracruz',
    'morelia': 'Morelia',            'monarcas morelia': 'Morelia',
    'lobos buap': 'Lobos BUAP',
    'dorados': 'Dorados',            'dorados de sinaloa': 'Dorados',
    'chiapas': 'Chiapas',            'chiapas fc': 'Chiapas', 'jaguares': 'Chiapas',
    'estudiantes tecos': 'Estudiantes','tecos': 'Estudiantes',
    'indios de ciudad juárez': 'FC Juárez',
}

def norm(name: str) -> str:
    if not name:
        return name
    key = str(name).lower().strip()
    return _ALIAS.get(key, name.strip())

# ─────────────────────────────────────────────────────────────────────────────
# PESOS POR TORNEO
# ─────────────────────────────────────────────────────────────────────────────
# Los 4 últimos torneos con pesos decrecientes
TORNEO_WEIGHTS = {
    '2025/2026 - Clausura': 4,   # Clausura 2026
    '2025/2026 - Apertura': 3,   # Apertura 2025
    '2024/2025 - Clausura': 2,   # Clausura 2025
    '2024/2025 - Apertura': 1,   # Apertura 2024
}
MODEL_TORNEOS = list(TORNEO_WEIGHTS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
def _torneo_key(filename: str) -> str:
    """Convierte nombre de archivo → clave de torneo."""
    name = Path(filename).stem.replace('historico_', '')
    # "2025-2026_-_clausura" → "2025/2026_-_clausura"
    name = name.replace('-', '/', 1)
    # Queda "2025/2026_-_clausura"
    parts = name.split('_-_', 1)
    if len(parts) == 2:
        year   = parts[0].replace('_', '/')
        torneo = parts[1].replace('_', ' ').title()
        return f'{year} - {torneo}'
    return name


def load_all_matches() -> list[dict]:
    """Carga todos los partidos históricos con su clave de torneo."""
    rows = []
    for fpath in sorted(glob.glob(str(HIST_DIR / '*.json'))):
        tkey = _torneo_key(fpath)
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        for p in data.get('partidos', []):
            if p.get('goles_local') is None or p.get('goles_visit') is None:
                continue
            if not p.get('terminado'):
                continue
            rows.append({
                'torneo':    tkey,
                'fecha':     p.get('fecha', ''),
                'local':     norm(p.get('local', '')),
                'visitante': norm(p.get('visitante', '')),
                'gl':        int(p['goles_local']),
                'gv':        int(p['goles_visit']),
            })
    return rows


def load_model_matches() -> list[dict]:
    """Carga solo los 4 torneos del modelo con su peso."""
    all_m = load_all_matches()
    result = []
    for m in all_m:
        w = TORNEO_WEIGHTS.get(m['torneo'])
        if w:
            result.append({**m, 'peso': w})
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MODELO DE POISSON
# ─────────────────────────────────────────────────────────────────────────────
def build_poisson_model(matches: list[dict]) -> dict:
    """
    Calcula parámetros del modelo de Poisson con ponderación por torneo.
    Retorna: {mu_home, mu_away, attack, defense}
    """
    # Acumuladores ponderados
    sum_gl = sum_gv = sum_w = 0.0
    home_scored   = defaultdict(float)  # goles marcados como local (ponderado)
    home_conceded = defaultdict(float)  # goles recibidos como local
    away_scored   = defaultdict(float)  # goles marcados como visitante
    away_conceded = defaultdict(float)  # goles recibidos como visitante
    home_games    = defaultdict(float)  # partidos como local (ponderado)
    away_games    = defaultdict(float)

    for m in matches:
        w = m['peso']
        local, visit = m['local'], m['visitante']
        gl, gv = m['gl'], m['gv']

        sum_gl += gl * w
        sum_gv += gv * w
        sum_w  += w

        home_scored[local]    += gl * w
        home_conceded[local]  += gv * w
        away_scored[visit]    += gv * w
        away_conceded[visit]  += gl * w
        home_games[local]     += w
        away_games[visit]     += w

    mu_home = sum_gl / sum_w  # promedio ponderado goles local
    mu_away = sum_gv / sum_w  # promedio ponderado goles visitante

    # Fuerza de ataque y defensa
    teams = set(home_scored) | set(away_scored)
    attack  = {}
    defense = {}
    for t in teams:
        hg = home_games.get(t, 0)
        ag = away_games.get(t, 0)
        attack[t]  = {
            'home': (home_scored[t]  / hg / mu_home) if hg > 0 else 1.0,
            'away': (away_scored[t]  / ag / mu_away) if ag > 0 else 1.0,
        }
        defense[t] = {
            'home': (home_conceded[t] / hg / mu_away) if hg > 0 else 1.0,
            'away': (away_conceded[t] / ag / mu_home) if ag > 0 else 1.0,
        }

    return {
        'mu_home':  mu_home,
        'mu_away':  mu_away,
        'attack':   attack,
        'defense':  defense,
        'teams':    sorted(teams),
    }


DC_RHO = -0.13  # parámetro Dixon-Coles (estándar académico para fútbol)

def dixon_coles_correction(home_goals, away_goals, lambda_home, lambda_away, rho):
    """
    Factor de corrección para marcadores bajos (Dixon & Coles 1997).
    Corrige la subestimación de 0-0, 1-0, 0-1, 1-1 en Poisson independiente.
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_home * lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0

def predict(model: dict, local: str, visitante: str, max_goals: int = 5, rho: float = DC_RHO):
    """
    Calcula lambdas y matriz de probabilidades hasta max_goals×max_goals.
    Aplica corrección Dixon-Coles para marcadores bajos.
    Retorna (lambda_l, lambda_v, matriz, p_local, p_empate, p_visitante)
    """
    local_n = norm(local)
    visit_n = norm(visitante)

    att  = model['attack']
    defe = model['defense']
    mu_h = model['mu_home']
    mu_a = model['mu_away']

    att_h  = att.get(local_n,  {}).get('home',  1.0)
    def_h  = defe.get(local_n, {}).get('home',  1.0)
    att_a  = att.get(visit_n,  {}).get('away',  1.0)
    def_a  = defe.get(visit_n, {}).get('away',  1.0)

    lambda_l = att_h * def_a * mu_h
    lambda_v = att_a * def_h * mu_a

    # Distribuciones de Poisson con corrección Dixon-Coles
    n = max_goals + 1
    matriz = np.zeros((n, n))
    for gl in range(n):
        for gv in range(n):
            dc = dixon_coles_correction(gl, gv, lambda_l, lambda_v, rho)
            matriz[gl, gv] = poisson.pmf(gl, lambda_l) * poisson.pmf(gv, lambda_v) * dc

    # Renormalizar (DC altera ligeramente la suma total)
    total = matriz.sum()
    if total > 0:
        matriz /= total

    p_local     = float(np.tril(matriz, -1).sum())  # local > visitante
    p_empate    = float(np.trace(matriz))            # iguales
    p_visitante = float(np.triu(matriz, 1).sum())   # visitante > local

    return lambda_l, lambda_v, matriz, p_local, p_empate, p_visitante

# ─────────────────────────────────────────────────────────────────────────────
# IMÁGENES
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_TEAM = 'https://images.fotmob.com/image_resources/logo/teamlogo/{}.png'
_HEADERS    = {'User-Agent': 'Mozilla/5.0'}

# Mapeo nombre → equipo_id FotMob (para descargar escudo)
TEAM_IDS = {
    'América':       6576,  'Chivas': 7807,    'Cruz Azul': 6578,
    'Tigres':        8561,  'Monterrey': 7849, 'Pumas':     1946,
    'Toluca':        6618,  'Santos Laguna': 7857, 'Pachuca': 7848,
    'Atlas':         6577,  'León':  1841,     'Necaxa':    1842,
    'Tijuana':       162418,'Querétaro': 1943, 'FC Juárez': 649424,
    'Mazatlán':      1170234,'San Luis': 6358, 'Puebla':    7847,
}

def get_team_shield(team_name: str):
    """Descarga y retorna escudo como PIL Image, o None."""
    tid = TEAM_IDS.get(norm(team_name))
    if not tid:
        return None
    dest = IMG_TEAMS / f'{tid}.png'
    if not (dest.exists() and dest.stat().st_size > 0):
        try:
            req = urllib.request.Request(FOTMOB_TEAM.format(tid), headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=6) as r:
                dest.write_bytes(r.read())
        except Exception:
            return None
    try:
        return Image.open(dest).convert('RGBA')
    except Exception:
        return None


def shield_array(team_name: str, size: int = 60) -> np.ndarray | None:
    img = get_team_shield(team_name)
    if img is None:
        return None
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def add_shield_ax(fig, arr, x, y, w, h):
    """Inserta escudo como axes sin bordes en coordenadas de figura."""
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(arr)
    ax.axis('off')
    return ax

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 8, 10
DPI = 150

def team_color(name: str) -> str:
    n = norm(name)
    return TEAM_COLORS.get(n, TEAM_COLORS.get(name, '#444455'))

def draw_circle_score(fig, cx, cy, r_fig, color, pct_text, label_text,
                      shield_arr=None, shield_size=0.055):
    """
    Dibuja un círculo grande con porcentaje grande dentro.
    cx, cy, r_fig: coordenadas de figura (0-1).
    """
    # Conversión a coordenadas de axes (todo en figura)
    circle_ax = fig.add_axes([cx - r_fig, cy - r_fig, r_fig*2, r_fig*2])
    circle_ax.set_xlim(-1, 1); circle_ax.set_ylim(-1, 1)
    circle_ax.set_aspect('equal')
    circle_ax.axis('off')
    circle_ax.set_facecolor('none')

    # Sombra del círculo
    shadow = plt.Circle((0.04, -0.04), 0.92, color='black', alpha=0.4, zorder=1)
    circle_ax.add_patch(shadow)
    # Círculo principal
    circ = plt.Circle((0, 0), 0.90, color=color, zorder=2)
    circle_ax.add_patch(circ)
    # Borde fino
    border = plt.Circle((0, 0), 0.90, fill=False, edgecolor='white',
                        linewidth=1.5, alpha=0.4, zorder=3)
    circle_ax.add_patch(border)

    # Porcentaje (texto grande)
    circle_ax.text(0, 0.08, pct_text,
                   color='white', va='center', ha='center',
                   zorder=5, **bebas_fp(32))

    # Label debajo del porcentaje
    circle_ax.text(0, -0.45, label_text,
                   color='white', fontsize=8, alpha=0.9,
                   va='center', ha='center', zorder=5)

    # Escudo encima (fuera del círculo, en coordenadas figura)
    if shield_arr is not None:
        sh_x = cx - shield_size / 2
        sh_y = cy + r_fig - 0.005
        sh_ax = fig.add_axes([sh_x, sh_y, shield_size, shield_size * (FIG_W/FIG_H)])
        sh_ax.imshow(shield_arr)
        sh_ax.axis('off')


def render_prediction(local: str, visitante: str,
                      lambda_l: float, lambda_v: float,
                      matriz: np.ndarray,
                      p_local: float, p_empate: float, p_visit: float,
                      output_path: Path):

    n = matriz.shape[0]  # 6 (0-5)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)

    # ── LAYOUT en coordenadas de figura (0=abajo, 1=arriba) ──────────────
    FOOTER_H  = 0.065
    FOOTER_Y  = 0.0
    TITLE_H   = 0.10
    TITLE_Y   = 0.905
    CIRCLES_H = 0.165
    CIRCLES_Y = 0.730
    HEATMAP_H = 0.600
    HEATMAP_Y = FOOTER_H + 0.025   # 0.09
    # Verificar que no se encimen:
    # heatmap top = 0.09 + 0.600 = 0.690  < circles_y 0.730  ✓
    # circles top = 0.730 + 0.165 = 0.895 < title_y 0.905    ✓

    # ── FONDO CON GRADIENTE SUTIL ─────────────────────────────────────────
    bg_ax = fig.add_axes([0, 0, 1, 1])
    bg_ax.set_facecolor(DARK_BG); bg_ax.axis('off')
    # Tono de color de cada equipo en su mitad
    c1 = team_color(local)
    c2 = team_color(visitante)

    def hex_rgba(h, a=0.12):
        r,g,b = tuple(int(h.lstrip('#')[i:i+2],16)/255 for i in (0,2,4))
        return (r,g,b,a)

    bg_ax.add_patch(mpatches.Rectangle((0, 0), 0.5, 1,
                    color=hex_rgba(c1, 0.15), transform=bg_ax.transAxes))
    bg_ax.add_patch(mpatches.Rectangle((0.5, 0), 0.5, 1,
                    color=hex_rgba(c2, 0.15), transform=bg_ax.transAxes))

    # ── TÍTULO ────────────────────────────────────────────────────────────
    title_ax = fig.add_axes([0.03, TITLE_Y, 0.94, TITLE_H])
    title_ax.set_facecolor('#161b22'); title_ax.axis('off')
    title_ax.add_patch(mpatches.Rectangle((0,0), 0.004, 1,
                       color=RED_BRAND, transform=title_ax.transAxes))

    title_ax.text(0.016, 0.70, 'PROBABILIDADES DE MARCADOR',
                  color=WHITE, va='center', ha='left',
                  transform=title_ax.transAxes, **bebas_fp(18))
    title_ax.text(0.016, 0.22,
                  f'{norm(local)} vs {norm(visitante)}  ·  Clausura 2026',
                  color=GRAY, fontsize=8, va='center', ha='left',
                  transform=title_ax.transAxes)

    # ── CÍRCULOS WIN/DRAW/LOSS ─────────────────────────────────────────────
    shield_local = shield_array(local,     60)
    shield_visit = shield_array(visitante, 60)

    r = 0.080   # radio en coordenadas de figura
    c_y = CIRCLES_Y + CIRCLES_H * 0.38   # centro vertical de los círculos

    # Local (izquierda)
    draw_circle_score(fig,
        cx=0.22, cy=c_y, r_fig=r,
        color=c1,
        pct_text=f'{p_local*100:.1f}%',
        label_text='Victoria Local',
        shield_arr=shield_local, shield_size=0.065)

    # Empate (centro)
    draw_circle_score(fig,
        cx=0.50, cy=c_y, r_fig=r,
        color='#3a3f4b',
        pct_text=f'{p_empate*100:.1f}%',
        label_text='Empate',
        shield_arr=None)

    # Visitante (derecha)
    draw_circle_score(fig,
        cx=0.78, cy=c_y, r_fig=r,
        color=c2,
        pct_text=f'{p_visit*100:.1f}%',
        label_text='Victoria Visitante',
        shield_arr=shield_visit, shield_size=0.065)

    # Línea separadora debajo de los círculos
    sep_ax = fig.add_axes([0.05, CIRCLES_Y - 0.005, 0.90, 0.003])
    sep_ax.set_facecolor('#21262d'); sep_ax.axis('off')

    # ── HEATMAP ───────────────────────────────────────────────────────────
    # Coordenadas del heatmap (dejando espacio para etiquetas de ejes)
    HM_LEFT   = 0.14
    HM_BOTTOM = HEATMAP_Y + 0.04
    HM_WIDTH  = 0.72
    HM_HEIGHT = HEATMAP_H - 0.06

    hm_ax = fig.add_axes([HM_LEFT, HM_BOTTOM, HM_WIDTH, HM_HEIGHT])

    # Colormap: negro oscuro → rojo intenso
    cmap_hm = mc.LinearSegmentedColormap.from_list(
        'heatred', ['#0d1117', '#3d0005', '#8b0010', '#D5001C', '#ff3344'])

    # imshow: filas = goles local (0 abajo → 5 arriba) → invertir
    im = hm_ax.imshow(
        np.flipud(matriz),
        cmap=cmap_hm,
        aspect='auto',
        vmin=0, vmax=matriz.max() * 1.05
    )

    # Grid lines
    for x in np.arange(-0.5, n, 1):
        hm_ax.axvline(x, color='#21262d', lw=0.8)
    for y in np.arange(-0.5, n, 1):
        hm_ax.axhline(y, color='#21262d', lw=0.8)

    # Texto de porcentaje en cada celda
    for gl in range(n):
        for gv in range(n):
            pct = matriz[gl, gv] * 100
            row = (n - 1) - gl   # invertido
            txt_color = 'white' if pct < matriz.max() * 100 * 0.6 else '#0d1117'
            fs = 9.5 if pct >= 1.0 else 8
            fw = 'bold' if pct >= 3.0 else 'normal'
            hm_ax.text(gv, row, f'{pct:.1f}%',
                       ha='center', va='center',
                       color=txt_color, fontsize=fs, fontweight=fw)

    # Configuración de ejes
    hm_ax.set_xticks(range(n))
    hm_ax.set_yticks(range(n))
    hm_ax.set_xticklabels([str(i) for i in range(n)],
                           color=WHITE, fontsize=9)
    hm_ax.set_yticklabels([str(i) for i in reversed(range(n))],
                           color=WHITE, fontsize=9)
    hm_ax.tick_params(colors=WHITE, length=0)
    for spine in hm_ax.spines.values():
        spine.set_edgecolor('#21262d')

    # Etiquetas de eje con nombre de equipo
    hm_ax.set_xlabel(f'Goles  {norm(visitante).upper()}  (visitante)',
                     color=WHITE, fontsize=10, labelpad=8, **bebas_fp(10))
    hm_ax.set_ylabel(f'Goles  {norm(local).upper()}  (local)',
                     color=WHITE, fontsize=10, labelpad=8, **bebas_fp(10))

    # Escudos en los ejes (solo si disponibles)
    shield_s = 0.046   # tamaño en coords figura

    # Escudo visitante → debajo del eje X (izquierda del label)
    sv = shield_array(visitante, 50)
    if sv is not None:
        sv_ax = fig.add_axes([
            HM_LEFT - shield_s - 0.005,
            HM_BOTTOM - 0.075,
            shield_s, shield_s * (FIG_W/FIG_H)
        ])
        sv_ax.imshow(sv); sv_ax.axis('off')

    # Escudo local → a la izquierda del eje Y
    sl = shield_array(local, 50)
    if sl is not None:
        sl_ax = fig.add_axes([
            HM_LEFT - 0.10,
            HM_BOTTOM + HM_HEIGHT/2 - (shield_s * (FIG_W/FIG_H))/2,
            shield_s, shield_s * (FIG_W/FIG_H)
        ])
        sl_ax.imshow(sl); sl_ax.axis('off')

    # ── FOOTER ────────────────────────────────────────────────────────────
    fax = fig.add_axes([0, FOOTER_Y, 1, FOOTER_H])
    fax.set_facecolor('#080c10'); fax.axis('off')
    fax.set_xlim(0, 1); fax.set_ylim(0, 1)

    # Línea roja separadora
    fax.axhline(1.0, color=RED_BRAND, lw=2.0, xmin=0, xmax=1)

    # Fuente: FotMob
    fax.text(0.015, 0.42, 'Modelo: ELO + Poisson + Dixon-Coles  |  Fuente: FotMob',
             color=GRAY, fontsize=9, va='center', ha='left',
             transform=fax.transAxes)

    # MAU-STATISTICS — sombra negra + texto rojo en Bebas Neue 24pt
    mau_kw = {**bebas_fp(24), 'va': 'center', 'ha': 'right',
              'transform': fax.transAxes}
    # Sombra (desplazada 2px = ~0.003 en coords figura a 150DPI)
    fax.text(0.990, 0.35, 'MAU-STATISTICS',
             color='#000000', alpha=0.75, **mau_kw)
    fax.text(0.988, 0.44, 'MAU-STATISTICS',
             color=RED_BRAND, **mau_kw)

    # ── GUARDAR ──────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f'✓ Guardado: {output_path}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    local     = sys.argv[1] if len(sys.argv) > 1 else 'Pachuca'
    visitante = sys.argv[2] if len(sys.argv) > 2 else 'Toluca'

    local_n = norm(local)
    visit_n = norm(visitante)

    print(f'Modelo Poisson: {local_n} vs {visit_n}')
    print(f'Torneos: {", ".join(MODEL_TORNEOS)}')
    print()

    print('Cargando datos históricos...')
    matches = load_model_matches()
    print(f'  {len(matches)} partidos en los últimos 4 torneos (ponderados)')

    print('Construyendo modelo...')
    model = build_poisson_model(matches)
    print(f'  μ_home={model["mu_home"]:.3f}  μ_away={model["mu_away"]:.3f}')
    print(f'  Equipos en el modelo: {len(model["teams"])}')

    if local_n not in model['attack']:
        print(f'  ⚠ {local_n!r} no encontrado. Equipos disponibles:')
        for t in model['teams']:
            print(f'    - {t}')
    if visit_n not in model['attack']:
        print(f'  ⚠ {visit_n!r} no encontrado.')

    print(f'\nCalculando predicción...')
    lambda_l, lambda_v, matriz, p_local, p_empate, p_visit = \
        predict(model, local_n, visit_n, max_goals=5)

    print(f'  λ local    = {lambda_l:.3f}')
    print(f'  λ visitante = {lambda_v:.3f}')
    print(f'  P(Victoria {local_n})  = {p_local*100:.1f}%')
    print(f'  P(Empate)               = {p_empate*100:.1f}%')
    print(f'  P(Victoria {visit_n}) = {p_visit*100:.1f}%')

    print('\nGenerando visualización...')
    local_slug = local_n.lower().replace(' ', '_').replace('/', '-')
    visit_slug = visit_n.lower().replace(' ', '_').replace('/', '-')
    out = OUT_DIR / f'prediccion_{local_slug}_vs_{visit_slug}.png'
    render_prediction(
        local_n, visit_n,
        lambda_l, lambda_v,
        matriz, p_local, p_empate, p_visit,
        out
    )


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
        GOLD      = _p['accent2']
        _pal      = _p
    main()
