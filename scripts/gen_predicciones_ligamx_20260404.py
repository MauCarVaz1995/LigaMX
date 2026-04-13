#!/usr/bin/env python3
"""
gen_predicciones_ligamx_20260404.py  v2
Predicciones Liga MX Clausura 2026 — Jornada 2026-04-04/05
Fixes v2:
  - Paletas: solo medianoche_neon, oceano_esmeralda, rojo_fuego
  - Heatmap con gradiente real LinearSegmentedColormap
  - % favorito 20 pct más grande
  - Marcador mínimo 13pt, color accent
  - Subtítulo en línea separada sin encimar
  - IC 95% bootstrap (1000 sims Monte Carlo) debajo de cada probabilidad
Salida: output/charts/predicciones/LigaMX_Clausura_2026/
"""
import sys, random, warnings, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox
from scipy.stats import poisson

warnings.filterwarnings('ignore')

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from config_visual import PALETAS, bebas, hex_rgb

BEBAS_TTF = Path.home() / '.fonts/BebasNeue.ttf'
if BEBAS_TTF.exists():
    try:
        fm.fontManager.addfont(str(BEBAS_TTF))
    except Exception:
        BEBAS_TTF = None

ELO_CSV  = BASE / 'data/processed/elo_historico.csv'
LOGO_DIR = BASE / 'data/raw/logos/ligamx'
OUT_DIR  = BASE / 'output/charts/predicciones/LigaMX_Clausura_2026'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Escudos Liga MX ─────────────────────────────────────────────────────────
# Prioridad 1: cache local en data/raw/logos/ligamx/{equipo}.png (descargado de FotMob)
# Prioridad 2: descarga en vivo desde FotMob CDN
# Prioridad 3: None → solo texto, nunca rompe la imagen

TEAM_IDS_FOTMOB = {
    'Monterrey': 7849, 'San Luis':  6358, 'Querétaro': 1943,
    'Toluca':    6618, 'Cruz Azul': 6578, 'Pachuca':   7848,
    'León':      1841, 'Atlas':     6577, 'Santos':    7857,
    'América':   6576, 'Chivas':    7807, 'Pumas':     1946,
    'Tigres':    8561, 'Necaxa':    1842, 'Tijuana':   162418,
    'Mazatlán':  1170234, 'FC Juárez': 649424, 'Puebla': 7847,
}

def get_escudo_ligamx(team: str, size=(52, 52)):
    """
    Carga el escudo de un equipo de Liga MX como OffsetImage.
    Fuentes en orden: cache local → FotMob CDN → None.
    """
    import urllib.request
    from PIL import Image as _PIL
    from matplotlib.offsetbox import OffsetImage

    # 1) cache local
    cache = LOGO_DIR / f'{team}.png'

    # 2) si no existe en cache, intenta descargar de FotMob (_xh = extra high res)
    if not cache.exists():
        tid = TEAM_IDS_FOTMOB.get(team)
        if tid:
            url = f'https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png'
            try:
                req = urllib.request.Request(url,
                      headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as r:
                    cache.write_bytes(r.read())
            except Exception:
                return None
        else:
            return None

    # 3) abrir a resolución nativa — el caller controla el zoom
    try:
        img = _PIL.open(cache).convert('RGBA')
        return OffsetImage(np.array(img), zoom=1.0)
    except Exception:
        return None

# ─── Paletas permitidas para Liga MX ─────────────────────────────────────────
PALETAS_LIGAMX = ['medianoche_neon', 'oceano_esmeralda', 'rojo_fuego']

# ─── ELO Liga MX ─────────────────────────────────────────────────────────────
def load_elos():
    df = pd.read_csv(ELO_CSV)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df.sort_values('fecha').groupby('equipo').last()['elo'].to_dict()

FOTMOB_TO_ELO = {
    'Monterrey': 'Monterrey', 'San Luis':  'San Luis',
    'Querétaro': 'Querétaro', 'Toluca':    'Toluca',
    'Cruz Azul': 'Cruz Azul', 'Pachuca':   'Pachuca',
    'León':      'Leon',      'Atlas':     'Atlas',
    'Santos':    'Santos Laguna', 'América': 'América',
    'Chivas':    'Chivas',    'Pumas':     'Pumas',
    'Tigres':    'Tigres',    'Necaxa':    'Necaxa',
    'Tijuana':   'Tijuana',   'FC Juárez': 'FC Juárez',
    'Mazatlán':  'Mazatlán',  'Puebla':    'Puebla',
}

TEAM_DISPLAY = {
    'Monterrey': 'MONTERREY', 'San Luis':  'SAN LUIS',
    'Querétaro': 'QUERÉTARO', 'Toluca':    'TOLUCA',
    'Cruz Azul': 'CRUZ AZUL', 'Pachuca':   'PACHUCA',
    'León':      'LEÓN',      'Atlas':     'ATLAS',
    'Santos':    'SANTOS',    'América':   'AMÉRICA',
    'Chivas':    'CHIVAS',    'Pumas':     'PUMAS',
    'Tigres':    'TIGRES',    'Necaxa':    'NECAXA',
    'Tijuana':   'TIJUANA',   'FC Juárez': 'FC JUÁREZ',
    'Mazatlán':  'MAZATLÁN',  'Puebla':    'PUEBLA',
}

TEAM_COLORS = {
    'Monterrey': '#003DA5', 'San Luis':  '#D52B1E',
    'Querétaro': '#1A7FCB', 'Toluca':    '#D5001C',
    'Cruz Azul': '#0047AB', 'Pachuca':   '#A8B8C8',
    'León':      '#2D8C3C', 'Atlas':     '#B22222',
    'Santos':    '#2E8B57', 'América':   '#FFD700',
    'Chivas':    '#CE1141', 'Pumas':     '#C8A84B',
    'Tigres':    '#F5A623', 'Necaxa':    '#D62828',
    'Tijuana':   '#C62828', 'FC Juárez': '#4CAF50',
    'Mazatlán':  '#9B59B6', 'Puebla':    '#2563EB',
}

MATCHES = [
    dict(home='Monterrey', away='San Luis',  fecha='2026-04-04', hora='19:00 CDT',
         match_id=5101826, torneo='Liga MX Clausura'),
    dict(home='Querétaro', away='Toluca',    fecha='2026-04-04', hora='19:00 CDT',
         match_id=5101830, torneo='Liga MX Clausura'),
    dict(home='Cruz Azul', away='Pachuca',   fecha='2026-04-05', hora='21:05 CDT',
         match_id=5101828, torneo='Liga MX Clausura'),
    dict(home='León',      away='Atlas',     fecha='2026-04-05', hora='21:06 CDT',
         match_id=5101827, torneo='Liga MX Clausura'),
    dict(home='Santos',    away='América',   fecha='2026-04-05', hora='23:10 CDT',
         match_id=5101829, torneo='Liga MX Clausura'),
    dict(home='Chivas',    away='Pumas',     fecha='2026-04-05', hora='21:07 CDT',
         match_id=5101831, torneo='Liga MX Clausura'),
]

# ─── Modelo ───────────────────────────────────────────────────────────────────
AVG_GOALS    = 1.35
HOME_ADV_ELO = 100
GOLD         = '#FFD700'
DC_RHO       = -0.13   # parámetro de correlación Dixon-Coles (estándar académico)

def elo_to_lambda(elo_h, elo_a, all_elos):
    elo_mean = np.mean(all_elos)
    return (AVG_GOALS * (elo_h + HOME_ADV_ELO) / elo_mean,
            AVG_GOALS * elo_a / elo_mean)

def dixon_coles_correction(home_goals, away_goals, lambda_home, lambda_away, rho):
    """
    Factor de corrección para marcadores bajos (Dixon & Coles 1997).
    Corrige la subestimación de resultados 0-0, 1-0, 0-1, 1-1 en Poisson independiente.
    rho < 0 → aumenta 0-0 y 1-1, disminuye 1-0 y 0-1 levemente.
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

def poisson_probs(lam_h, lam_a, max_goals=5, rho=DC_RHO):
    """Matriz de probabilidades Poisson con corrección Dixon-Coles."""
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            dc = dixon_coles_correction(i, j, lam_h, lam_a, rho)
            probs[i, j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) * dc
    # Renormalizar: la corrección DC altera ligeramente la suma total
    probs /= probs.sum()
    return probs, float(np.tril(probs,-1).sum()), float(np.diag(probs).sum()), float(np.triu(probs,1).sum())

def bootstrap_ci(lam_h, lam_a, n_sim=1000, ci=0.95):
    """
    Simula n_sim partidos Poisson y calcula IC95% por bootstrap.
    Retorna dict con claves 'home','draw','away', cada una (lo, hi).
    """
    rng = np.random.default_rng(42)
    gh  = rng.poisson(lam_h, n_sim)
    ga  = rng.poisson(lam_a, n_sim)
    outcomes = np.where(gh > ga, 0, np.where(gh < ga, 2, 1))  # 0=local,1=emp,2=visita

    alpha = (1 - ci) / 2
    boot_h, boot_d, boot_a = [], [], []
    for _ in range(1000):
        sample = rng.choice(outcomes, size=n_sim, replace=True)
        counts = np.bincount(sample, minlength=3)
        boot_h.append(counts[0] / n_sim)
        boot_d.append(counts[1] / n_sim)
        boot_a.append(counts[2] / n_sim)

    def ci_bounds(arr):
        return (np.percentile(arr, alpha*100) * 100,
                np.percentile(arr, (1-alpha)*100) * 100)

    return {'home': ci_bounds(boot_h),
            'draw': ci_bounds(boot_d),
            'away': ci_bounds(boot_a)}

def fmt_date(d, hora):
    mo = {'01':'ene','02':'feb','03':'mar','04':'abr','05':'may','06':'jun',
          '07':'jul','08':'ago','09':'sep','10':'oct','11':'nov','12':'dic'}
    day, m = d[8:], d[5:7]
    return f'{int(day)} {mo.get(m,m)} {d[:4]}  ·  {hora}'

# ─── Render ───────────────────────────────────────────────────────────────────
def render(m, pal, elo_h, elo_a, all_elos, validate_only=False):
    home, away = m['home'], m['away']
    lam_h, lam_a = elo_to_lambda(elo_h, elo_a, all_elos)
    probs, p_home, p_draw, p_away = poisson_probs(lam_h, lam_a)
    max_idx  = np.unravel_index(np.argmax(probs), probs.shape)
    best_p   = float(probs[max_idx]) * 100
    ci_dict  = bootstrap_ci(lam_h, lam_a)

    # Normalizar para que sumen exactamente 1.0 (el render los muestra como %)
    _total = p_home + p_draw + p_away
    p_home = p_home / _total
    p_draw = p_draw / _total
    p_away = 1.0 - p_home - p_draw

    BG    = pal['bg_primary'];   BG2   = pal['bg_secondary']
    WHITE = pal['text_primary']; GRAY  = pal['text_secondary']
    RED   = pal['accent'];       ACC2  = pal['accent2']
    CHIGH = pal['cell_high'];    CMID  = pal['cell_mid']
    CLOW  = pal['cell_low']

    t1_es = TEAM_DISPLAY.get(home, home.upper())
    t2_es = TEAM_DISPLAY.get(away, away.upper())
    c1    = TEAM_COLORS.get(home, CHIGH)
    c2    = TEAM_COLORS.get(away, ACC2)

    # Colormap para el heatmap: negro→fondo secundario→cell_mid→cell_high
    cmap = LinearSegmentedColormap.from_list(
        'ligamx', [BG, BG2, CMID, CHIGH], N=256)
    p_max = float(probs.max())

    FIG_W = FIG_H = 7.2        # 1080×1080 @ 150 dpi
    HEADER_H = 0.245            # un poco más alto para dos líneas de subtítulo
    FOOTER_H = 0.055
    FACT_H   = 0.052
    PROB_H   = 0.155            # más alto para caber IC
    COLHDR_H = 0.055
    ROWHDR_W = 0.088
    R_MARGIN = 0.010
    GRID_Y   = FOOTER_H + FACT_H + PROB_H + 0.006
    GRID_H   = 1.0 - HEADER_H - GRID_Y - COLHDR_H
    N        = 6
    CELL_H   = GRID_H / N
    CELL_W   = (1.0 - ROWHDR_W - R_MARGIN) / N

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    # Gradiente de fondo
    bg_rgb, bg2_rgb = hex_rgb(BG), hex_rgb(BG2)
    grad = np.zeros((200, 2, 3))
    for ii in range(200):
        t = ii / 199
        grad[ii] = np.array(bg_rgb)/255*(1-t) + np.array(bg2_rgb)/255*t
    bgax = fig.add_axes([0, 0, 1, 1])
    bgax.set_zorder(-100)
    bgax.imshow(grad, aspect='auto', extent=[0,1,0,1], origin='lower')
    bgax.axis('off')

    # ── Header ────────────────────────────────────────────────────────────────
    hax = fig.add_axes([0, 1-HEADER_H, 1, HEADER_H])
    hax.set_facecolor(BG2); hax.axis('off')
    hax.axhline(0, color=RED, lw=3.0)
    # Título principal — arriba fijo
    hax.text(0.50, 0.97, '¿QUIÉN GANA HOY?',
             color=WHITE, ha='center', va='top',
             transform=hax.transAxes, **bebas(30))
    # Subtítulo — segunda línea separada, nunca sobre el título
    hax.text(0.50, 0.68,
             f'{m["torneo"]}  ·  {fmt_date(m["fecha"], m["hora"])}',
             color=GRAY, ha='center', va='top',
             transform=hax.transAxes, fontsize=8.5)

    # ── Paneles de equipo ─────────────────────────────────────────────────────
    # Layout: local  → [escudo izq | NOMBRE der]
    #         visita → [NOMBRE izq | escudo der]
    PANEL_Y = 1 - HEADER_H + HEADER_H * 0.06
    PANEL_H = HEADER_H * 0.40

    # Logo size en fracciones de figura (60px / 1080px ≈ 0.056)
    LOGO_SZ = 0.067   # ancho y alto del logo en figure fraction

    panels = [
        (0.04, home, t1_es, elo_h, lam_h, c1, 'home'),
        (0.56, away, t2_es, elo_a, lam_a, c2, 'away'),
    ]
    for col_x, team_key, name_es, elo, lam, border_c, side in panels:
        pax = fig.add_axes([col_x, PANEL_Y, 0.38, PANEL_H])
        pax.set_facecolor(BG); pax.axis('off')
        for sp in pax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(border_c); sp.set_linewidth(1.5)

        # Centro vertical del logo dentro del panel (figura fraction)
        logo_cy = PANEL_Y + PANEL_H * 0.68
        logo_y  = logo_cy - LOGO_SZ / 2   # esquina inferior del axes del logo

        cache = LOGO_DIR / f'{team_key}.png'
        logo_loaded = False
        if cache.exists():
            try:
                from PIL import Image as _PIL
                logo_arr = np.array(_PIL.open(cache).convert('RGBA'))
                if side == 'home':
                    # Escudo izquierda del panel: col_x + pequeño margen
                    logo_x = col_x + 0.012
                    nombre_x, nombre_ha = 0.58, 'center'
                else:
                    # Escudo derecha del panel: col_x + 0.38 - logo - margen
                    logo_x = col_x + 0.38 - LOGO_SZ - 0.012
                    nombre_x, nombre_ha = 0.42, 'center'
                lax = fig.add_axes([logo_x, logo_y, LOGO_SZ, LOGO_SZ])
                lax.imshow(logo_arr)
                lax.axis('off')
                logo_loaded = True
            except Exception:
                pass

        if logo_loaded:
            pax.text(nombre_x, 0.72, name_es, color=border_c,
                     ha=nombre_ha, va='center', fontsize=9.5,
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

    # ── Cabecera columnas ─────────────────────────────────────────────────────
    chdr_y = GRID_Y + GRID_H
    lbl_ax = fig.add_axes([ROWHDR_W, chdr_y + COLHDR_H*0.55, N*CELL_W, COLHDR_H*0.42])
    lbl_ax.set_facecolor(BG2); lbl_ax.axis('off')
    lbl_ax.text(0.5, 0.5, f'GOLES  {t2_es}', color=c2,
                ha='center', va='center', fontsize=7,
                fontweight='bold', transform=lbl_ax.transAxes)
    for j in range(N):
        nax = fig.add_axes([ROWHDR_W + j*CELL_W, chdr_y, CELL_W, COLHDR_H*0.55])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.5, 0.5, str(j), color=WHITE, ha='center', va='center',
                 fontsize=9, fontweight='bold', transform=nax.transAxes)

    # ── Cabecera filas ────────────────────────────────────────────────────────
    rmx = fig.add_axes([0, GRID_Y, ROWHDR_W*0.35, GRID_H])
    rmx.set_facecolor(BG2); rmx.axis('off')
    rmx.text(0.5, 0.5, f'GOLES  {t1_es}', color=c1,
             ha='center', va='center', fontsize=6, fontweight='bold',
             rotation=90, transform=rmx.transAxes)
    for i in range(N):
        nax = fig.add_axes([ROWHDR_W*0.35, GRID_Y + i*CELL_H, ROWHDR_W*0.65, CELL_H])
        nax.set_facecolor(BG2); nax.axis('off')
        nax.text(0.65, 0.5, str(i), color=WHITE, ha='center', va='center',
                 fontsize=9, fontweight='bold', transform=nax.transAxes)

    # ── Heatmap con gradiente real ────────────────────────────────────────────
    for i in range(N):
        for j in range(N):
            p   = float(probs[i, j])
            t_n = p / p_max if p_max > 0 else 0
            fc  = cmap(t_n)           # RGBA desde colormap
            # color de texto: blanco sobre celdas oscuras, negro sobre muy brillantes
            brightness = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
            tc = '#ffffff' if brightness < 0.55 else '#000000'
            bold = p > 0.03

            cax = fig.add_axes([ROWHDR_W + j*CELL_W,
                                GRID_Y   + i*CELL_H, CELL_W, CELL_H])
            cax.set_xlim(0,1); cax.set_ylim(0,1)
            cax.add_patch(mpatches.Rectangle((0,0), 1, 1, facecolor=fc,
                          edgecolor='none', zorder=0,
                          transform=cax.transAxes, clip_on=False))
            cax.axis('off')
            if i == j:  # diagonal = empates
                for sp in cax.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(GOLD)
                    sp.set_linewidth(1.5)
            if p >= 0.003:
                txt = f'{p*100:.1f}' if p >= 0.010 else f'{p*100:.2f}'
                cax.text(0.5, 0.5, txt, color=tc, ha='center', va='center',
                         fontsize=11 if p > 0.09 else (9 if p > 0.045 else 7),
                         fontweight='bold' if bold else 'normal',
                         transform=cax.transAxes, zorder=5)

    # ── Bloques de probabilidad con IC 95% ───────────────────────────────────
    prob_y  = FOOTER_H + FACT_H + 0.006
    BLK_GAP = 0.012
    BLK_W   = (1.0 - 4*BLK_GAP) / 3
    max_p   = max(p_home, p_draw, p_away)

    result_blocks = [
        (home, f'GANA {t1_es}', p_home, ci_dict['home']),
        (None, 'EMPATE',         p_draw, ci_dict['draw']),
        (away, f'GANA {t2_es}',  p_away, ci_dict['away']),
    ]

    BASE_PCT_FS = 26   # tamaño base para no-favorito
    FAV_PCT_FS  = int(BASE_PCT_FS * 1.20)  # 20% más grande para el favorito

    for k, (team_key, blk_label, pval, ci_bounds) in enumerate(result_blocks):
        bx     = BLK_GAP + k * (BLK_W + BLK_GAP)
        is_max = (pval >= max_p - 1e-9)
        blk_c  = TEAM_COLORS.get(team_key, RED) if is_max and team_key else (RED if is_max else GRAY)
        lbl_c  = WHITE if is_max else GRAY
        pct_fs = FAV_PCT_FS if is_max else BASE_PCT_FS

        seg = fig.add_axes([bx, prob_y, BLK_W, PROB_H])
        seg.set_facecolor(BG2)
        seg.axis('off'); seg.set_xlim(0,1); seg.set_ylim(0,1)
        if is_max:
            for sp in seg.spines.values():
                sp.set_visible(True); sp.set_edgecolor(blk_c); sp.set_linewidth(2.0)

        # Etiqueta (sin escudo en bloques de resultado)
        seg.text(0.5, 0.78, blk_label, color=lbl_c, ha='center', va='center',
                 fontsize=7.5, fontweight='bold', transform=seg.transAxes)
        # Porcentaje — favorito más grande
        seg.text(0.5, 0.50, f'{pval*100:.1f}%', color=blk_c if is_max else GRAY,
                 ha='center', va='center', transform=seg.transAxes,
                 **bebas(pct_fs))
        # IC 95%
        ci_lo, ci_hi = ci_bounds
        seg.text(0.5, 0.17,
                 f'IC 95%: [{ci_lo:.1f}% — {ci_hi:.1f}%]',
                 color=WHITE, alpha=0.6, ha='center', va='center',
                 fontsize=9, transform=seg.transAxes)

    # ── Marcador más probable — mínimo 13pt, color accent ─────────────────────
    fax = fig.add_axes([0, FOOTER_H, 1, FACT_H])
    fax.set_facecolor(BG2); fax.axis('off')
    fax.text(0.5, 0.5,
             f'Marcador más probable:  {t1_es} {max_idx[0]}-{max_idx[1]} {t2_es}  ({best_p:.1f}%)',
             color=RED, ha='center', va='center',
             fontsize=13, fontweight='bold', transform=fax.transAxes)

    # ── Footer ────────────────────────────────────────────────────────────────
    ftax = fig.add_axes([0, 0, 1, FOOTER_H])
    ftax.set_facecolor(BG); ftax.axis('off')
    ftax.axhline(1.0, color=RED, lw=2.0, zorder=10)
    ftax.text(0.5, 0.5, 'Modelo: ELO + Poisson + Dixon-Coles  |  @LigaMX_Stats  |  IC 95% bootstrap n=1000',
              color=GRAY, ha='center', va='center',
              fontsize=6.5, transform=ftax.transAxes, zorder=10)
    ftax.text(0.98, 0.5, 'MAU-STATISTICS',
              color=RED, alpha=1.0, ha='right', va='center',
              fontsize=14, zorder=10, transform=ftax.transAxes,
              **bebas(14))

    out_file = OUT_DIR / f'pred_{home.replace(" ","_")}_{away.replace(" ","_")}.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return out_file, p_home, p_draw, p_away, max_idx, best_p, ci_dict

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=int, default=None,
                        help='Índice 0-4 para generar solo ese partido (validación)')
    args = parser.parse_args()

    elos     = load_elos()
    all_elos = []
    for m in MATCHES:
        all_elos += [elos.get(FOTMOB_TO_ELO.get(m['home'], m['home']), 1500),
                     elos.get(FOTMOB_TO_ELO.get(m['away'], m['away']), 1500)]

    pal_keys = list(PALETAS_LIGAMX)
    last_pal = None
    results  = []

    targets = [MATCHES[args.only]] if args.only is not None else MATCHES

    print(f'\n{"═"*70}')
    print(f'  PREDICCIONES LIGA MX CLAUSURA 2026 — {len(targets)} partido(s)')
    print(f'{"═"*70}\n')

    for m in targets:
        elo_h = elos.get(FOTMOB_TO_ELO.get(m['home'], m['home']), 1500)
        elo_a = elos.get(FOTMOB_TO_ELO.get(m['away'], m['away']), 1500)

        available = [k for k in pal_keys if k != last_pal]
        pal_key   = random.choice(available)
        last_pal  = pal_key

        print(f'🎨  Paleta: {pal_key}')
        print(f'⚽  {m["home"]} (ELO {elo_h:.0f})  vs  {m["away"]} (ELO {elo_a:.0f})')

        out_file, ph, pd_, pa, midx, bp, ci = render(
            m, PALETAS[pal_key], elo_h, elo_a, all_elos)

        winner = m['home'] if ph > pa else (m['away'] if pa > ph else 'Empate')
        print(f'    L={ph*100:.1f}%  IC[{ci["home"][0]:.1f}%–{ci["home"][1]:.1f}%]')
        print(f'    E={pd_*100:.1f}%  IC[{ci["draw"][0]:.1f}%–{ci["draw"][1]:.1f}%]')
        print(f'    V={pa*100:.1f}%  IC[{ci["away"][0]:.1f}%–{ci["away"][1]:.1f}%]')
        print(f'    Marcador: {midx[0]}-{midx[1]} ({bp:.1f}%)  → {winner}')
        print(f'    → {out_file.name}\n')
        results.append((m, elo_h, elo_a, ph, pd_, pa, midx, bp, winner))

    print(f'{"═"*95}')
    print(f'{"LOCAL":<14} {"VISITANTE":<14} {"ELO-L":>6} {"ELO-V":>6} '
          f'{"L%":>7} {"E%":>7} {"V%":>7} {"SCORE":>6}  PREDICCIÓN')
    print(f'{"─"*95}')
    for m, eh, ea, ph, pd_, pa, midx, bp, winner in results:
        print(f'{m["home"]:<14} {m["away"]:<14} {eh:>6.0f} {ea:>6.0f} '
              f'{ph*100:>6.1f}% {pd_*100:>6.1f}% {pa*100:>6.1f}% '
              f'{midx[0]}-{midx[1]}    {winner}')
    print(f'{"═"*95}')
    print(f'\n✓ Imágenes en: {OUT_DIR}')

if __name__ == '__main__':
    main()
