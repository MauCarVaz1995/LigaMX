#!/usr/bin/env python3
"""
generar_prediccion.py — Script CANÓNICO de predicciones MAU-STATISTICS
=======================================================================
UN solo script para todas las competiciones. Mismo diseño visual siempre.

Competiciones soportadas:
  ligamx   → output/charts/predicciones/LigaMX_Clausura_2026/J{N}/
  ccl      → output/charts/predicciones/CCL_2025-26/{stage}/
  intl     → output/charts/predicciones/Internacional/{YYYY-MM-DD}/

Uso:
  python generar_prediccion.py --competition ligamx              # partidos de hoy en Liga MX
  python generar_prediccion.py --competition ligamx --date 2026-04-19
  python generar_prediccion.py --competition ccl --leg2          # cuartos/semis vuelta
  python generar_prediccion.py --competition intl --date 2026-04-15
  python generar_prediccion.py --all                             # todo lo de hoy
  python generar_prediccion.py --competition ligamx --force      # re-generar aunque existan

Este script REEMPLAZA a:
  - 19_predicciones_hoy.py       (deprecated)
  - gen_predicciones_ligamx_*.py (deprecated)
  - gen_predicciones_ccl.py      (deprecated)
"""

import argparse
import json
import random
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

MX_TZ = timezone(timedelta(hours=-6))

def utc_str_to_mx_date(utc_str: str) -> str:
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        return dt.astimezone(MX_TZ).strftime("%Y-%m-%d")
    except Exception:
        return utc_str[:10]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import poisson
from PIL import Image as _PIL

warnings.filterwarnings('ignore')

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from config_visual import PALETAS, PALETTE, bebas, hex_rgb

try:
    from config_visual import BEBAS_TTF, BEBAS_AVAILABLE
except ImportError:
    BEBAS_TTF = None
    BEBAS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────
ELO_CSV       = BASE / 'data/processed/elo_historico.csv'
HIST_DIR      = BASE / 'data/raw/historico'
INTL_CSV      = BASE / 'data/raw/internacional/results.csv'
LOGOS_LIGAMX  = BASE / 'data/raw/logos/ligamx'
LOGOS_CCL     = BASE / 'data/raw/logos/ccl'
FLAGS_DIR     = BASE / 'data/raw/flags'
CCL_JSON_DIR  = BASE / 'data/raw/fotmob/ccl'
OUT_BASE      = BASE / 'output/charts/predicciones'

TODAY = date.today().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# PALETAS POR COMPETICIÓN
# ─────────────────────────────────────────────────────────────────────────────
PALETAS_LIGAMX = ['medianoche_neon', 'oceano_esmeralda', 'rojo_fuego']
PALETA_CCL     = 'medianoche_neon'
PALETA_INTL    = 'oceano_esmeralda'

# ─────────────────────────────────────────────────────────────────────────────
# COLORES DE EQUIPOS (para paneles con borde de color del equipo)
# ─────────────────────────────────────────────────────────────────────────────
TEAM_COLORS = {
    # Liga MX
    'Monterrey': '#003DA5', 'San Luis':   '#D52B1E',
    'Querétaro': '#1A7FCB', 'Toluca':     '#D5001C',
    'Cruz Azul': '#0047AB', 'Pachuca':    '#A8B8C8',
    'León':      '#2D8C3C', 'Atlas':      '#B22222',
    'Santos':    '#2E8B57', 'América':    '#FFD700',
    'Chivas':    '#CE1141', 'Pumas':      '#C8A84B',
    'Tigres':    '#F5A623', 'Necaxa':     '#D62828',
    'Tijuana':   '#C62828', 'FC Juárez':  '#4CAF50',
    'Mazatlán':  '#9B59B6', 'Puebla':     '#2563EB',
    # CCL — MLS
    'Nashville SC':     '#C8102E', 'LAFC':             '#C39E6D',
    'LA Galaxy':        '#00245D', 'Seattle Sounders': '#5D9732',
    'Cincinnati':       '#F05323', 'Inter Miami CF':   '#F7B5CD',
    'Philadelphia':     '#B54E22', 'Vancouver':        '#00245D',
    'San Diego FC':     '#002855', 'Atlético Ottawa':  '#000000',
    # Selecciones comunes
    'México':    '#006847', 'Mexico':     '#006847',
    'Portugal':  '#006600', 'Argentina':  '#74ACDF',
    'Brasil':    '#009C3B', 'España':     '#AA151B',
    'Francia':   '#0055A4', 'Alemania':   '#000000',
    'Inglaterra':'#CF091F', 'Colombia':   '#FCD116',
    'Uruguay':   '#5EB6E4', 'Marruecos':  '#C1272D',
}

TEAM_DISPLAY = {
    # Liga MX
    'Monterrey': 'MONTERREY', 'San Luis':  'SAN LUIS',
    'Querétaro': 'QUERÉTARO', 'Toluca':    'TOLUCA',
    'Cruz Azul': 'CRUZ AZUL', 'Pachuca':   'PACHUCA',
    'León':      'LEÓN',      'Atlas':     'ATLAS',
    'Santos':    'SANTOS',    'América':   'AMÉRICA',
    'Santos Laguna': 'SANTOS', 'Atletico de San Luis': 'SAN LUIS',
    'Chivas':    'CHIVAS',    'Pumas':     'PUMAS',
    'Tigres':    'TIGRES',    'Necaxa':    'NECAXA',
    'Tijuana':   'TIJUANA',   'FC Juárez': 'FC JUÁREZ',
    'Mazatlán':  'MAZATLÁN',  'Puebla':    'PUEBLA',
    # CCL — MLS
    'Nashville SC':     'NASHVILLE', 'LAFC':             'LAFC',
    'LA Galaxy':        'LA GALAXY', 'Seattle Sounders': 'SEATTLE',
    'Cincinnati':       'CINCY',     'Inter Miami CF':   'MIAMI',
    'Philadelphia':     'PHILA.',    'Vancouver':        'VANCOUVER',
    'San Diego FC':     'SAN DIEGO', 'Atlético Ottawa':  'OTTAWA',
    'LD Alajuelense':   'ALAJUELENSE',
    # Selecciones
    'Mexico': 'MÉXICO',
}

LIGAMX_NAME_FIX = {
    'CF America': 'América', 'Atletico de San Luis': 'San Luis',
    'Queretaro FC': 'Querétaro', 'FC Juarez': 'FC Juárez',
    'Mazatlan FC': 'Mazatlán', 'Santos Laguna': 'Santos',
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGOS
# ─────────────────────────────────────────────────────────────────────────────
FOTMOB_IDS_LIGAMX = {
    'Monterrey': 7849, 'San Luis': 6358, 'Querétaro': 1943,
    'Toluca': 6618, 'Cruz Azul': 6578, 'Pachuca': 7848,
    'León': 1841, 'Atlas': 6577, 'Santos': 7857,
    'América': 6576, 'Chivas': 7807, 'Pumas': 1946,
    'Tigres': 8561, 'Necaxa': 1842, 'Tijuana': 162418,
    'Mazatlán': 1170234, 'FC Juárez': 649424, 'Puebla': 7847,
}

FLAG_CODES = {
    'México': 'mx', 'Mexico': 'mx', 'Argentina': 'ar', 'Brasil': 'br',
    'Portugal': 'pt', 'España': 'es', 'Francia': 'fr', 'Alemania': 'de',
    'Inglaterra': 'gb-eng', 'Colombia': 'co', 'Uruguay': 'uy',
    'Marruecos': 'ma', 'Bélgica': 'be', 'Croacia': 'hr',
    'Países Bajos': 'nl', 'Japón': 'jp', 'Suiza': 'ch',
    'Dinamarca': 'dk', 'Ecuador': 'ec', 'Senegal': 'sn',
    'Chile': 'cl', 'Perú': 'pe', 'Bolivia': 'bo', 'Paraguay': 'py',
    'Costa Rica': 'cr', 'Honduras': 'hn', 'Panamá': 'pa',
    'El Salvador': 'sv', 'Guatemala': 'gt', 'Jamaica': 'jm',
    'Italia': 'it', 'Turquía': 'tr', 'Polonia': 'pl',
    'Suecia': 'se', 'Noruega': 'no', 'Bielorrusia': 'by',
    'Ucrania': 'ua', 'República Checa': 'cz', 'Kosovo': 'xk',
    'Luxemburgo': 'lu', 'Gibraltar': 'gi',
    # Raw FotMob names
    'Brazil': 'br', 'Spain': 'es', 'France': 'fr', 'Germany': 'de',
    'England': 'gb-eng', 'Netherlands': 'nl', 'Japan': 'jp',
    'Switzerland': 'ch', 'Denmark': 'dk', 'Morocco': 'ma',
    'Belgium': 'be', 'Croatia': 'hr', 'Sweden': 'se',
    'Italy': 'it', 'Turkey': 'tr', 'Poland': 'pl',
}


def load_logo(team: str, logo_dir: Path) -> np.ndarray | None:
    """Carga logo desde disco. Retorna array RGBA o None."""
    path = logo_dir / f'{team}.png'
    if not path.exists():
        # Intento fallback: FotMob CDN para Liga MX
        tid = FOTMOB_IDS_LIGAMX.get(team)
        if tid:
            import urllib.request
            try:
                url = f'https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png'
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as r:
                    path.write_bytes(r.read())
            except Exception:
                pass
    if path.exists():
        try:
            return np.array(_PIL.open(path).convert('RGBA'))
        except Exception:
            pass
    return None


def get_flag(team: str) -> np.ndarray | None:
    code = FLAG_CODES.get(team)
    if not code:
        return None
    path = FLAGS_DIR / f'{code}.png'
    if path.exists():
        try:
            return np.array(_PIL.open(path).convert('RGBA'))
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────────────────────
DC_RHO = -0.13
GOLD   = '#FFD700'


def dixon_coles(gh, ga, lh, la, rho=DC_RHO):
    if   gh == 0 and ga == 0: return 1 - lh * la * rho
    elif gh == 0 and ga == 1: return 1 + lh * rho
    elif gh == 1 and ga == 0: return 1 + la * rho
    elif gh == 1 and ga == 1: return 1 - rho
    return 1.0


def poisson_probs(lam_h, lam_a, max_goals=5):
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = (poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) *
                           dixon_coles(i, j, lam_h, lam_a))
    probs /= probs.sum()
    ph = float(np.tril(probs, -1).sum())
    pd = float(np.diag(probs).sum())
    pa = float(np.triu(probs, 1).sum())
    return probs, ph, pd, pa


def elo_to_lambda(elo_h, elo_a, mean_elo, avg_goals, home_adv=100):
    return (avg_goals * (elo_h + home_adv) / mean_elo,
            avg_goals * elo_a / mean_elo)


def bootstrap_ci(lam_h, lam_a, n=1000, ci=0.95):
    rng = np.random.default_rng(42)
    gh  = rng.poisson(lam_h, n)
    ga  = rng.poisson(lam_a, n)
    out = np.where(gh > ga, 0, np.where(gh < ga, 2, 1))
    alpha = (1 - ci) / 2
    bh, bd, ba = [], [], []
    for _ in range(500):
        s = rng.choice(out, size=n, replace=True)
        c = np.bincount(s, minlength=3)
        bh.append(c[0] / n); bd.append(c[1] / n); ba.append(c[2] / n)
    def bnds(arr):
        return (np.percentile(arr, alpha * 100) * 100,
                np.percentile(arr, (1 - alpha) * 100) * 100)
    return {'home': bnds(bh), 'draw': bnds(bd), 'away': bnds(ba)}


def fmt_date(d: str) -> str:
    mo = {'01':'ene','02':'feb','03':'mar','04':'abr','05':'may','06':'jun',
          '07':'jul','08':'ago','09':'sep','10':'oct','11':'nov','12':'dic'}
    return f'{int(d[8:])} {mo.get(d[5:7], d[5:7])} {d[:4]}'


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — función canónica (NO MODIFICAR)
# ─────────────────────────────────────────────────────────────────────────────
def render(match: dict, pal_key: str, elo_h: float, elo_a: float,
           mean_elo: float, avg_goals: float, home_adv: float,
           out_file: Path, logo_dir: Path,
           agg_label: str = None) -> dict:
    """
    Genera imagen 1080×1080 de predicción.

    match dict requiere: home, away, fecha, torneo
    Retorna dict con probabilidades calculadas.
    """
    pal   = PALETAS[pal_key]
    BG    = pal['bg_primary'];   BG2   = pal['bg_secondary']
    WHITE = pal['text_primary']; GRAY  = pal['text_secondary']
    RED   = pal['accent'];       ACC2  = pal['accent2']
    CHIGH = pal['cell_high'];    CMID  = pal['cell_mid']
    CLOW  = pal['cell_low']

    home, away = match['home'], match['away']
    t1 = TEAM_DISPLAY.get(home, home.upper())
    t2 = TEAM_DISPLAY.get(away, away.upper())
    c1 = TEAM_COLORS.get(home, CHIGH)
    c2 = TEAM_COLORS.get(away, ACC2)

    lam_h, lam_a = elo_to_lambda(elo_h, elo_a, mean_elo, avg_goals, home_adv)
    probs, ph, pd, pa = poisson_probs(lam_h, lam_a)
    _t = ph + pd + pa
    ph, pd, pa = ph / _t, pd / _t, 1.0 - ph / _t - pd / _t
    max_idx = np.unravel_index(np.argmax(probs), probs.shape)
    best_p  = float(probs[max_idx]) * 100
    ci_dict = bootstrap_ci(lam_h, lam_a)

    cmap = LinearSegmentedColormap.from_list('pred', [BG, BG2, CMID, CHIGH], N=256)
    p_max = float(probs.max())

    # ── Layout constants ─────────────────────────────────────────────────────
    FIG_W = FIG_H = 7.2
    HEADER_H = 0.245
    FOOTER_H = 0.055
    FACT_H   = 0.052
    PROB_H   = 0.155
    COLHDR_H = 0.055
    ROWHDR_W = 0.088
    R_MARGIN = 0.010
    GRID_Y   = FOOTER_H + FACT_H + PROB_H + 0.006
    GRID_H   = 1.0 - HEADER_H - GRID_Y - COLHDR_H
    N        = 6
    CELL_H   = GRID_H / N
    CELL_W   = (1.0 - ROWHDR_W - R_MARGIN) / N

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Fondo degradado
    bg_rgb, bg2_rgb = hex_rgb(BG), hex_rgb(BG2)
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = np.array(bg_rgb)/255*(1-t) + np.array(bg2_rgb)/255*t
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.set_zorder(-100)
    bgax.imshow(grad, aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    bgax.axis('off')

    # ── Header ───────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1 - HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=3.0)
    hax.text(0.50, 0.97, '¿QUIÉN GANA HOY?',
             color=WHITE, ha='center', va='top',
             transform=hax.transAxes, **bebas(30))
    sub = match.get('torneo', '')
    if agg_label:
        sub += f'  ·  {agg_label}'
    hax.text(0.50, 0.68,
             f'{sub}  ·  {fmt_date(match["fecha"])}',
             color=GRAY, ha='center', va='top',
             transform=hax.transAxes, fontsize=8.5)

    # ── Paneles de equipo ─────────────────────────────────────────────────────
    PANEL_Y = 1 - HEADER_H + HEADER_H * 0.06
    PANEL_H = HEADER_H * 0.40
    LOGO_SZ = 0.067

    for col_x, team_key, name_es, elo, lam, border_c, side in [
        (0.04, home, t1, elo_h, lam_h, c1, 'home'),
        (0.56, away, t2, elo_a, lam_a, c2, 'away'),
    ]:
        pax = fig.add_axes([col_x, PANEL_Y, 0.38, PANEL_H])
        pax.set_facecolor(BG); pax.axis('off')
        for sp in pax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(border_c); sp.set_linewidth(1.5)

        logo_cy = PANEL_Y + PANEL_H * 0.68
        logo_y  = logo_cy - LOGO_SZ / 2

        logo_arr = load_logo(team_key, logo_dir)
        if logo_arr is None and logo_dir != FLAGS_DIR:
            logo_arr = get_flag(team_key)

        if logo_arr is not None:
            try:
                logo_x = (col_x + 0.012) if side == 'home' else (col_x + 0.38 - LOGO_SZ - 0.012)
                lax = fig.add_axes([logo_x, logo_y, LOGO_SZ, LOGO_SZ])
                lax.imshow(logo_arr); lax.axis('off')
                pax.text(0.58, 0.72, name_es, color=border_c,
                         ha='center', va='center', fontsize=9.5,
                         fontweight='bold', transform=pax.transAxes)
            except Exception:
                pax.text(0.50, 0.72, name_es, color=border_c,
                         ha='center', va='center', fontsize=10,
                         fontweight='bold', transform=pax.transAxes)
        else:
            pax.text(0.50, 0.72, name_es, color=border_c,
                     ha='center', va='center', fontsize=10,
                     fontweight='bold', transform=pax.transAxes)

        loc_str = 'LOCAL +100 ELO' if side == 'home' else 'VISITANTE'
        pax.text(0.50, 0.32, f'ELO: {elo:.0f}',
                 color=WHITE, ha='center', va='center',
                 fontsize=8.5, transform=pax.transAxes)
        pax.text(0.50, 0.16, loc_str,
                 color=GRAY, ha='center', va='center',
                 fontsize=6.5, transform=pax.transAxes)
        pax.text(0.50, 0.03, f'λ = {lam:.3f} goles',
                 color=GRAY, ha='center', va='center',
                 fontsize=7, transform=pax.transAxes)

    # VS central
    vax = fig.add_axes([0.40, PANEL_Y, 0.18, PANEL_H])
    vax.set_facecolor(BG2); vax.axis('off')
    vax.text(0.50, 0.55, 'VS', color=RED, ha='center', va='center',
             fontsize=22, fontweight='bold', transform=vax.transAxes)
    vax.text(0.50, 0.18, 'Poisson DC · ELO', color=GRAY, ha='center',
             va='center', fontsize=6.5, transform=vax.transAxes)

    # ── Cabecera heatmap ──────────────────────────────────────────────────────
    chdr_y = GRID_Y + GRID_H
    lbl_ax = fig.add_axes([ROWHDR_W, chdr_y + COLHDR_H*0.55, N*CELL_W, COLHDR_H*0.42])
    lbl_ax.set_facecolor(BG2); lbl_ax.axis('off')
    lbl_ax.text(0.5, 0.5, f'GOLES  {t2}', color=c2,
                ha='center', va='center', fontsize=7,
                fontweight='bold', transform=lbl_ax.transAxes)
    for j in range(N):
        nax = fig.add_axes([ROWHDR_W + j*CELL_W, chdr_y, CELL_W, COLHDR_H*0.55])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.5, 0.5, str(j), color=WHITE, ha='center', va='center',
                 fontsize=9, fontweight='bold', transform=nax.transAxes)

    rmx = fig.add_axes([0, GRID_Y, ROWHDR_W*0.35, GRID_H])
    rmx.set_facecolor(BG2); rmx.axis('off')
    rmx.text(0.5, 0.5, f'GOLES  {t1}', color=c1,
             ha='center', va='center', fontsize=6, fontweight='bold',
             rotation=90, transform=rmx.transAxes)
    for i in range(N):
        nax = fig.add_axes([ROWHDR_W*0.35, GRID_Y + i*CELL_H, ROWHDR_W*0.65, CELL_H])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.65, 0.5, str(i), color=WHITE, ha='center', va='center',
                 fontsize=9, fontweight='bold', transform=nax.transAxes)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    for i in range(N):
        for j in range(N):
            p   = float(probs[i, j])
            t_n = p / p_max if p_max > 0 else 0
            fc  = cmap(t_n)
            brightness = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
            tc  = '#ffffff' if brightness < 0.55 else '#000000'
            cax = fig.add_axes([ROWHDR_W + j*CELL_W, GRID_Y + i*CELL_H, CELL_W, CELL_H])
            cax.set_xlim(0, 1); cax.set_ylim(0, 1)
            cax.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=fc,
                          edgecolor='none', zorder=0,
                          transform=cax.transAxes, clip_on=False))
            cax.axis('off')
            if i == j:
                for sp in cax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(GOLD); sp.set_linewidth(1.5)
            if p >= 0.003:
                txt = f'{p*100:.1f}' if p >= 0.010 else f'{p*100:.2f}'
                cax.text(0.5, 0.5, txt, color=tc, ha='center', va='center',
                         fontsize=11 if p > 0.09 else (9 if p > 0.045 else 7),
                         fontweight='bold' if p > 0.03 else 'normal',
                         transform=cax.transAxes, zorder=5)

    # ── Bloques de probabilidad con IC 95% ───────────────────────────────────
    prob_y  = FOOTER_H + FACT_H + 0.006
    BLK_GAP = 0.012
    BLK_W   = (1.0 - 4*BLK_GAP) / 3
    max_p   = max(ph, pd, pa)

    for k, (team_key, blk_label, pval, ci_bounds) in enumerate([
        (home, f'GANA {t1}', ph, ci_dict['home']),
        (None, 'EMPATE',     pd, ci_dict['draw']),
        (away, f'GANA {t2}', pa, ci_dict['away']),
    ]):
        bx     = BLK_GAP + k * (BLK_W + BLK_GAP)
        is_max = (pval >= max_p - 1e-9)
        blk_c  = TEAM_COLORS.get(team_key, RED) if is_max and team_key else (RED if is_max else GRAY)
        lbl_c  = WHITE if is_max else GRAY
        pct_fs = int(26 * 1.20) if is_max else 26

        seg = fig.add_axes([bx, prob_y, BLK_W, PROB_H])
        seg.set_facecolor(BG2); seg.axis('off')
        seg.set_xlim(0, 1); seg.set_ylim(0, 1)
        if is_max:
            for sp in seg.spines.values():
                sp.set_visible(True); sp.set_edgecolor(blk_c); sp.set_linewidth(2.0)

        seg.text(0.5, 0.78, blk_label, color=lbl_c, ha='center', va='center',
                 fontsize=7.5, fontweight='bold', transform=seg.transAxes)
        seg.text(0.5, 0.50, f'{pval*100:.1f}%', color=blk_c if is_max else GRAY,
                 ha='center', va='center', transform=seg.transAxes, **bebas(pct_fs))
        ci_lo, ci_hi = ci_bounds
        seg.text(0.5, 0.17,
                 f'IC 95%: [{ci_lo:.1f}% — {ci_hi:.1f}%]',
                 color=WHITE, alpha=0.6, ha='center', va='center',
                 fontsize=9, transform=seg.transAxes)

    # ── Marcador más probable ─────────────────────────────────────────────────
    fax = fig.add_axes([0, FOOTER_H, 1, FACT_H])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.text(0.5, 0.5,
             f'Marcador más probable:  {t1} {max_idx[0]}-{max_idx[1]} {t2}  ({best_p:.1f}%)',
             color=RED, ha='center', va='center',
             fontsize=13, fontweight='bold', transform=fax.transAxes)

    # ── Footer ────────────────────────────────────────────────────────────────
    ftax = fig.add_axes([0, 0, 1, FOOTER_H])
    ftax.set_facecolor(BG); ftax.axis('off')
    ftax.axhline(1.0, color=RED, lw=2.0, zorder=10)
    ftax.text(0.5, 0.5,
              'Modelo: ELO + Poisson + Dixon-Coles  |  @Miau_Stats_MX  |  IC 95% bootstrap n=1000',
              color=GRAY, ha='center', va='center',
              fontsize=6.5, transform=ftax.transAxes, zorder=10)
    ftax.text(0.98, 0.5, 'MAU-STATISTICS',
              color=RED, ha='right', va='center',
              fontsize=14, zorder=10, transform=ftax.transAxes, **bebas(14))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)

    winner = home if ph > pa else (away if pa > ph else 'Empate')
    return {'home': home, 'away': away, 'ph': ph, 'pd': pd, 'pa': pa,
            'score': f'{max_idx[0]}-{max_idx[1]}', 'best_p': best_p,
            'winner': winner, 'out_file': out_file}


# ─────────────────────────────────────────────────────────────────────────────
# CARGADORES DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_ligamx_matches(target_date: str) -> list[dict]:
    """Devuelve partidos de Liga MX en target_date que no estén terminados."""
    f = HIST_DIR / 'historico_clausura_2026.json'
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    out = []
    for p in d['partidos']:
        if utc_str_to_mx_date(p['fecha']) != target_date:
            continue
        if p.get('terminado'):
            continue
        home = LIGAMX_NAME_FIX.get(p['local'], p['local'])
        away = LIGAMX_NAME_FIX.get(p['visitante'], p['visitante'])
        out.append({
            'home':    home,
            'away':    away,
            'fecha':   utc_str_to_mx_date(p['fecha']),
            'torneo':  'Liga MX Clausura 2026',
            'jornada': p.get('jornada', '?'),
        })
    return out


def load_ligamx_elos() -> dict:
    df = pd.read_csv(ELO_CSV)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df.sort_values('fecha').groupby('equipo').last()['elo'].to_dict()


LIGAMX_ELO_ALIAS = {
    'Monterrey': 'Monterrey', 'San Luis':   'San Luis',
    'Querétaro': 'Querétaro', 'Toluca':     'Toluca',
    'Cruz Azul': 'Cruz Azul', 'Pachuca':    'Pachuca',
    'León':      'Leon',      'Atlas':      'Atlas',
    'Santos':    'Santos Laguna', 'América': 'América',
    'Chivas':    'Chivas',    'Pumas':      'Pumas',
    'Tigres':    'Tigres',    'Necaxa':     'Necaxa',
    'Tijuana':   'Tijuana',   'FC Juárez':  'FC Juárez',
    'Mazatlán':  'Mazatlán',  'Puebla':     'Puebla',
    'Atletico de San Luis': 'San Luis',
}


def load_ccl_fixtures() -> list[dict]:
    files = sorted(CCL_JSON_DIR.glob('ccl_fixtures_*.json'))
    if not files:
        return []
    return json.loads(files[-1].read_text())['matches']


# ELOs base para equipos MLS (estimados desde historial CCL)
MLS_BASE_ELOS = {
    'LAFC':             1580, 'Seattle Sounders': 1560,
    'Nashville SC':     1530, 'LA Galaxy':        1520,
    'Cincinnati':       1490, 'Inter Miami CF':   1470,
    'Philadelphia':     1460, 'Vancouver':        1450,
    'San Diego FC':     1440, 'Atlético Ottawa':  1400,
}


def calc_ccl_elos(matches: list[dict], ligamx_elos: dict) -> tuple[dict, float]:
    """
    Calcula ELOs CCL usando Liga MX ELO como base para equipos mexicanos
    y MLS_BASE_ELOS como base para equipos MLS.
    """
    K = 35; HOME_ADV = 75; SCALE = 400

    # Inicializar con bases conocidas
    elos: dict[str, float] = {}

    # Equipos Liga MX: usar ELO de Liga MX como base
    for ccl_name, elo_alias in [
        ('América', 'América'), ('Cruz Azul', 'Cruz Azul'),
        ('Tigres',  'Tigres'),  ('Toluca',    'Toluca'),
        ('Monterrey', 'Monterrey'),
    ]:
        elos[ccl_name] = ligamx_elos.get(elo_alias, 1500)

    # Equipos MLS: usar bases manuales
    for team, base in MLS_BASE_ELOS.items():
        elos[team] = base

    def exp(a, b): return 1 / (1 + 10 ** ((b - a) / SCALE))

    def goal_m(gh, ga):
        d = abs(gh - ga)
        return 1.0 if d == 0 else 1.0 + 0.5 * np.log(d + 1)

    for m in sorted(matches, key=lambda x: x['date']):
        if not m.get('finished'):
            continue
        h, a = m['home'], m['away']
        gh, ga = int(m['home_score']), int(m['away_score'])
        elo_h = elos.get(h, 1500)
        elo_a = elos.get(a, 1500)
        eh = exp(elo_h + HOME_ADV, elo_a)
        rh = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        mult = goal_m(gh, ga)
        elos[h] = elo_h + K * mult * (rh       - eh)
        elos[a] = elo_a + K * mult * ((1 - rh) - (1 - eh))

    mean_elo = float(np.mean(list(elos.values())))
    return elos, mean_elo


def load_intl_elos() -> dict:
    files = sorted(BASE.glob('data/processed/elos_selecciones_*.json'))
    if not files:
        return {}
    return json.loads(files[-1].read_text())

INTL_NAME_ES = {
    'Colombia': 'Colombia',  'France': 'Francia',   'Brazil': 'Brasil',
    'Spain': 'España',       'Germany': 'Alemania',  'England': 'Inglaterra',
    'Argentina': 'Argentina','Italy': 'Italia',      'Netherlands': 'Países Bajos',
    'Portugal': 'Portugal',  'Mexico': 'México',     'Uruguay': 'Uruguay',
    'Morocco': 'Marruecos',  'Belgium': 'Bélgica',   'Croatia': 'Croacia',
    'Japan': 'Japón',        'Switzerland': 'Suiza', 'Denmark': 'Dinamarca',
    'Ecuador': 'Ecuador',    'Senegal': 'Senegal',   'Chile': 'Chile',
    'Peru': 'Perú',          'Bolivia': 'Bolivia',   'Paraguay': 'Paraguay',
    'Costa Rica': 'Costa Rica', 'Honduras': 'Honduras', 'Panama': 'Panamá',
    'Sweden': 'Suecia',      'Turkey': 'Turquía',    'Poland': 'Polonia',
    'Kosovo': 'Kosovo',
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERADORES POR COMPETICIÓN
# ─────────────────────────────────────────────────────────────────────────────

def generar_ligamx(target_date: str, force: bool = False) -> list[dict]:
    matches = load_ligamx_matches(target_date)
    if not matches:
        print(f'  Liga MX: sin partidos el {target_date}')
        return []

    elos    = load_ligamx_elos()
    all_elo_vals = [elos.get(LIGAMX_ELO_ALIAS.get(m['home'], m['home']), 1500) for m in matches] + \
                   [elos.get(LIGAMX_ELO_ALIAS.get(m['away'], m['away']), 1500) for m in matches]
    mean_elo = float(np.mean(all_elo_vals)) if all_elo_vals else 1500.0

    # Paleta aleatoria sin repetir
    pal_pool = list(PALETAS_LIGAMX)
    last_pal = None
    results  = []

    jornada = matches[0].get('jornada', '?')
    out_dir = OUT_BASE / 'LigaMX_Clausura_2026' / f'J{jornada}'
    print(f'  Liga MX J{jornada}: {len(matches)} partido(s) → {out_dir}')

    for m in matches:
        fname    = f'pred_{m["home"].replace(" ","_")}_{m["away"].replace(" ","_")}.png'
        out_file = out_dir / fname

        if out_file.exists() and not force:
            print(f'    [skip] {fname}')
            continue

        elo_h = elos.get(LIGAMX_ELO_ALIAS.get(m['home'], m['home']), 1500)
        elo_a = elos.get(LIGAMX_ELO_ALIAS.get(m['away'], m['away']), 1500)

        available = [k for k in pal_pool if k != last_pal]
        pal_key   = random.choice(available)
        last_pal  = pal_key

        res = render(m, pal_key, elo_h, elo_a, mean_elo,
                     avg_goals=1.35, home_adv=100,
                     out_file=out_file, logo_dir=LOGOS_LIGAMX)
        print(f'    [ok] {fname}  {res["winner"]} ({res["ph"]*100:.0f}%/{res["pd"]*100:.0f}%/{res["pa"]*100:.0f}%)')
        results.append(res)

    return results


def generar_ccl(target_date: str = None, force: bool = False,
                show_leg2: bool = False) -> list[dict]:
    matches = load_ccl_fixtures()
    if not matches:
        print('  CCL: no se encontró ccl_fixtures_*.json')
        return []

    ligamx_elos = load_ligamx_elos()
    elos, mean_elo = calc_ccl_elos(matches, ligamx_elos)

    # Calcular agregados (1a pierna terminada → 2a pendiente)
    from collections import defaultdict
    pair_legs: dict[tuple, list] = defaultdict(list)
    for m in sorted(matches, key=lambda x: x['date']):
        pair_legs[tuple(sorted([m['home'], m['away']]))].append(m)

    aggregates = {}
    for pair, legs in pair_legs.items():
        finished = [l for l in legs if l.get('finished')]
        pending  = [l for l in legs if not l.get('finished')]
        if not finished or not pending:
            continue
        leg1, leg2 = finished[-1], pending[0]
        h2 = leg2['home']
        agg_h = int(leg1['home_score']) if leg1['home'] == h2 else int(leg1['away_score'])
        agg_a = int(leg1['away_score']) if leg1['home'] == h2 else int(leg1['home_score'])
        aggregates[pair] = (agg_h, agg_a)

    # Partidos pendientes (filtrar por fecha si se especifica)
    pending_matches = [m for m in matches if not m.get('finished')]
    if target_date:
        # Normalizar: fixture dates son "20260415", target puede ser "2026-04-15"
        td_compact = target_date.replace('-', '')
        pending_matches = [m for m in pending_matches if m['date'].replace('-', '') == td_compact]

    if not pending_matches:
        print(f'  CCL: sin partidos pendientes{" el " + target_date if target_date else ""}')
        return []

    results = []
    for m in pending_matches:
        stage   = m.get('stage', 'ccl')
        stage_label = {
            'play_in': 'Primera_Ronda', 'quarterfinals': 'Cuartos',
            'semifinals': 'Semis', 'final': 'Final',
        }.get(stage, 'CCL')
        torneo_label = {
            'play_in': 'CCL — Primera Ronda',
            'quarterfinals': 'CCL — Cuartos de Final',
            'semifinals': 'CCL — Semifinales',
            'final': 'CCL — Final',
        }.get(stage, 'CONCACAF Champions Cup')

        out_dir  = OUT_BASE / 'CCL_2025-26' / stage_label
        fname    = f'pred_{m["home"].replace(" ","_")}_{m["away"].replace(" ","_")}_{m["date"]}.png'
        out_file = out_dir / fname

        if out_file.exists() and not force:
            print(f'    [skip] {fname}')
            continue

        elo_h = elos.get(m['home'], 1500)
        elo_a = elos.get(m['away'], 1500)

        pair     = tuple(sorted([m['home'], m['away']]))
        agg      = aggregates.get(pair)
        agg_lbl  = f'Agregado: {agg[0]}–{agg[1]}' if agg and show_leg2 else None

        # fmt_date necesita YYYY-MM-DD; normalizar desde "20260415"
        raw_date = m['date'].replace('-', '')
        fmt_fecha = f'{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}'
        match_dict = {**m, 'torneo': torneo_label, 'fecha': fmt_fecha}
        res = render(match_dict, PALETA_CCL, elo_h, elo_a, mean_elo,
                     avg_goals=1.25, home_adv=75,
                     out_file=out_file, logo_dir=LOGOS_CCL,
                     agg_label=agg_lbl)
        print(f'    [ok] {fname}  {res["winner"]} ({res["ph"]*100:.0f}%/{res["pd"]*100:.0f}%/{res["pa"]*100:.0f}%)')
        results.append(res)

    return results


def generar_intl(target_date: str, force: bool = False) -> list[dict]:
    """Genera predicciones para partidos internacionales de la fecha."""
    import requests

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json',
        'Referer': 'https://www.fotmob.com/',
    }

    url = f'https://www.fotmob.com/api/data/matches?date={target_date.replace("-","")}'
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'  Intl: error FotMob {e}')
        return []

    torneos_relevantes = ['World Cup', 'Copa América', 'Nations League',
                          'Gold Cup', 'Champions', 'CONCACAF', 'UEFA']
    matches_intl = []
    for league in data.get('leagues', []):
        if league.get('ccode') != 'INT':
            continue
        league_name = league.get('name', '')
        if not any(t.lower() in league_name.lower() for t in torneos_relevantes):
            continue
        for m in league.get('matches', []):
            status = m.get('status', {})
            if status.get('finished') or not status.get('started', True) is False:
                if not status.get('finished'):
                    h = m.get('home', {}).get('name', '')
                    a = m.get('away', {}).get('name', '')
                    if h and a:
                        matches_intl.append({
                            'home':   INTL_NAME_ES.get(h, h),
                            'away':   INTL_NAME_ES.get(a, a),
                            'fecha':  target_date,
                            'torneo': league_name,
                        })

    if not matches_intl:
        print(f'  Intl: sin partidos relevantes el {target_date}')
        return []

    elos = load_intl_elos()
    all_vals = [elos.get(m['home'], 1500) for m in matches_intl] + \
               [elos.get(m['away'], 1500) for m in matches_intl]
    mean_elo = float(np.mean(all_vals)) if all_vals else 1600.0

    out_dir = OUT_BASE / 'Internacional' / target_date
    results = []
    last_pal = None

    for m in matches_intl:
        fname    = f'pred_{m["home"].replace(" ","_")}_{m["away"].replace(" ","_")}.png'
        out_file = out_dir / fname
        if out_file.exists() and not force:
            print(f'    [skip] {fname}')
            continue

        elo_h = elos.get(m['home'], 1500)
        elo_a = elos.get(m['away'], 1500)

        available = [k for k in PALETAS if k != last_pal and k not in PALETAS_LIGAMX]
        if not available:
            available = [k for k in PALETAS if k != last_pal]
        pal_key  = random.choice(available)
        last_pal = pal_key

        res = render(m, pal_key, elo_h, elo_a, mean_elo,
                     avg_goals=1.20, home_adv=100,
                     out_file=out_file, logo_dir=FLAGS_DIR)
        print(f'    [ok] {fname}  {res["winner"]}')
        results.append(res)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generador canónico de predicciones MAU-STATISTICS')
    parser.add_argument('--competition', choices=['ligamx', 'ccl', 'intl', 'all'],
                        default='all', help='Competición a procesar')
    parser.add_argument('--date',  default=TODAY, help='Fecha YYYY-MM-DD (default: hoy)')
    parser.add_argument('--force', action='store_true', help='Re-generar aunque existan')
    parser.add_argument('--leg2',  action='store_true', help='CCL: mostrar marcador agregado')
    args = parser.parse_args()

    print(f'\n{"═"*60}')
    print(f'  MAU-STATISTICS — Predicciones {args.date}')
    print(f'{"═"*60}')

    total = []

    if args.competition in ('ligamx', 'all'):
        print('\n── Liga MX ──')
        total += generar_ligamx(args.date, args.force)

    if args.competition in ('ccl', 'all'):
        print('\n── CCL ──')
        total += generar_ccl(args.date if args.competition == 'ccl' else None,
                             args.force, args.leg2)

    if args.competition in ('intl', 'all'):
        print('\n── Internacional ──')
        total += generar_intl(args.date, args.force)

    print(f'\n{"═"*60}')
    print(f'  Total imágenes generadas: {len(total)}')
    print(f'{"═"*60}\n')


if __name__ == '__main__':
    main()
