#!/usr/bin/env python3
"""
scripts/dashboard/app.py
MauStats MX — Dashboard Plotly Dash (upgraded)
5 tabs: Jornada 13 · Ranking ELO · Simulación · Jugadores · Comparativo
"""

import json, glob, warnings, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from flask import send_from_directory

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent.parent
HIST_DIR    = BASE / 'data/raw/historico'
ELO_CSV     = BASE / 'data/processed/elo_historico.csv'
JUG_CSV     = BASE / 'data/processed/jugadores_clausura2026.csv'
TEAMS_IMG   = BASE / 'data/raw/images/teams'
PLAYERS_IMG = BASE / 'data/raw/images/players'

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
BG_MAIN  = '#0d1117'
BG_SEC   = '#161b22'
BG_CARD  = '#0f151e'
BORDER   = '#30363d'
WHITE    = '#ffffff'
GRAY     = '#8b949e'
RED      = '#D5001C'
GREEN    = '#2ea043'
GOLD     = '#FFD700'

# Team name → hex color
TEAM_COLORS = {
    'Chivas':              '#CD1F2D',
    'Cruz Azul':           '#0047AB',
    'Toluca':              '#D5001C',
    'América':             '#FFD700',
    'Tigres':              '#F5A623',
    'Monterrey':           '#003DA5',
    'Pumas':               '#C8A84B',
    'Santos Laguna':       '#2E8B57',
    'Pachuca':             '#A8B8C8',
    'Atlas':               '#B22222',
    'León':                '#2D8C3C',
    'Necaxa':              '#D62828',
    'Tijuana':             '#C62828',
    'Querétaro':           '#1A7FCB',
    'FC Juárez':           '#4CAF50',
    'Mazatlán':            '#9B59B6',
    'San Luis':            '#D52B1E',
    'Puebla':              '#2563EB',
}

# Team canonical name → equipo_id (for shield images)
TEAM_IDS = {
    'Chivas':        7807,
    'Cruz Azul':     6578,
    'Toluca':        6618,
    'Pumas':         1946,
    'Pachuca':       7848,
    'Atlas':         6577,
    'Tigres':        8561,
    'América':       6576,
    'Monterrey':     7849,
    'FC Juárez':     649424,
    'Necaxa':        1842,
    'León':          1841,
    'Tijuana':       162418,
    'Puebla':        7847,
    'San Luis':      6358,
    'Mazatlán':      1170234,
    'Querétaro':     1943,
    'Santos Laguna': 7857,
}

TORNEO_WEIGHTS = {
    '2025/2026 - Clausura': 4,
    '2025/2026 - Apertura': 3,
    '2024/2025 - Clausura': 2,
    '2024/2025 - Apertura': 1,
}

_ALIAS = {
    'cf america': 'América',        'america': 'América',
    'chivas': 'Chivas',             'guadalajara': 'Chivas',
    'cruz azul': 'Cruz Azul',
    'tigres': 'Tigres',             'tigres uanl': 'Tigres',
    'monterrey': 'Monterrey',       'cf monterrey': 'Monterrey',
    'pumas': 'Pumas',               'pumas unam': 'Pumas',
    'toluca': 'Toluca',
    'santos laguna': 'Santos Laguna',
    'pachuca': 'Pachuca',
    'atlas': 'Atlas',
    'león': 'León',                 'leon': 'León',
    'necaxa': 'Necaxa',
    'tijuana': 'Tijuana',
    'querétaro': 'Querétaro',       'queretaro': 'Querétaro',
    'queretaro fc': 'Querétaro',
    'fc juárez': 'FC Juárez',       'fc juarez': 'FC Juárez',
    'mazatlán': 'Mazatlán',         'mazatlan fc': 'Mazatlán',
    'mazatlan': 'Mazatlán',
    'atletico de san luis': 'San Luis', 'san luis': 'San Luis',
    'puebla': 'Puebla',
}

def norm(n):
    return _ALIAS.get(str(n).lower().strip(), str(n).strip())

def shield_url(team):
    tid = TEAM_IDS.get(team)
    return f'/images/teams/{tid}.png' if tid else None

def player_img_url(player_id):
    if (PLAYERS_IMG / f'{int(player_id)}.png').exists():
        return f'/images/players/{int(player_id)}.png'
    return None

# ─────────────────────────────────────────────────────────────────────────────
# VERDE QUETZAL COLORSCALE
# ─────────────────────────────────────────────────────────────────────────────
VERDE_QUETZAL = [
    [0.00, BG_MAIN],
    [0.04, '#1B5E20'],
    [0.15, '#2E7D32'],
    [0.40, '#00C853'],
    [1.00, '#00FF88'],
]

PLOTLY_BASE = dict(
    paper_bgcolor=BG_MAIN,
    plot_bgcolor=BG_SEC,
    font=dict(color=WHITE, family='Arial'),
)

# ─────────────────────────────────────────────────────────────────────────────
# RADAR COLUMNS PER POSITION
# ─────────────────────────────────────────────────────────────────────────────
RADAR_COLS = {
    'POR': ['paradas_p90', 'porcentaje_paradas_p90', 'goles_evitados_p90',
            'goles_recibidos_p90', 'porterias_cero_p90'],
    'DEF': ['intercepciones_p90', 'entradas_p90', 'despejes_p90',
            'duelos_tierra_ganados_p90', 'pases_precisos_p90'],
    'MED': ['pases_precisos_p90', 'pases_largos_p90', 'recuperaciones_campo_rival_p90',
            'chances_creadas_p90', 'asistencias_p90'],
    'DEL': ['goles_p90', 'xG_p90', 'tiros_p90', 'tiros_a_puerta_p90', 'grandes_chances_p90'],
}

STAT_META = {
    'goles_p90':                   ('⚽', 'Goles / 90'),
    'xG_p90':                      ('📊', 'xG / 90'),
    'tiros_p90':                   ('🎯', 'Tiros / 90'),
    'tiros_a_puerta_p90':          ('🎯', 'Tiros puerta / 90'),
    'asistencias_p90':             ('🔑', 'Asistencias / 90'),
    'pases_precisos_p90':          ('🎽', 'Pases prec / 90'),
    'chances_creadas_p90':         ('💡', 'Chances / 90'),
    'intercepciones_p90':          ('🛡️', 'Intercepciones / 90'),
    'entradas_p90':                ('🦵', 'Entradas / 90'),
    'duelos_tierra_ganados_p90':   ('💪', 'Duelos gan / 90'),
    'recuperaciones_campo_rival_p90': ('🔄', 'Recup rival / 90'),
    'paradas_p90':                 ('🧤', 'Paradas / 90'),
    'porcentaje_paradas_p90':      ('📈', '% Paradas'),
    'goles_evitados_p90':          ('🚫', 'Goles evitados / 90'),
    'porterias_cero_p90':          ('🏆', 'Porterías cero / 90'),
    'rating':                      ('⭐', 'Rating SofaScore'),
    'minutos_stats':               ('⏱️', 'Minutos jugados'),
    'partidos_stats':              ('📅', 'Partidos'),
    'edad':                        ('🎂', 'Edad'),
    'valor_mercado_eur':           ('💶', 'Valor de mercado'),
}

def pos_group(pos):
    p = str(pos).upper()
    if any(x in p for x in ['GK', 'POR', 'GOALKEEPER']):
        return 'POR'
    if any(x in p for x in ['CB', 'LB', 'RB', 'WB', 'DEF', 'BACK', 'CENTRAL']):
        return 'DEF'
    if any(x in p for x in ['CM', 'DM', 'AM', 'MID', 'MF', 'MIDFIELDER']):
        return 'MED'
    return 'DEL'

# ─────────────────────────────────────────────────────────────────────────────
# MODELO POISSON
# ─────────────────────────────────────────────────────────────────────────────
def load_poisson_model():
    goals_for  = defaultdict(float)
    goals_ag   = defaultdict(float)
    matches_h  = defaultdict(float)
    matches_a  = defaultdict(float)

    for fpath in sorted(glob.glob(str(HIST_DIR / '*.json'))):
        stem  = Path(fpath).stem.replace('historico_', '')
        parts = stem.split('_-_', 1)
        if len(parts) != 2:
            continue
        year   = parts[0].replace('_', '/')
        torneo = parts[1].replace('_', ' ').title()
        tkey   = f'{year} - {torneo}'
        w = TORNEO_WEIGHTS.get(tkey)
        if not w:
            continue

        data = json.load(open(fpath, encoding='utf-8'))
        for p in data.get('partidos', []):
            if not p.get('terminado'):
                continue
            lo = norm(p['local'])
            vi = norm(p['visitante'])
            gl = int(p.get('goles_local', 0) or 0)
            gv = int(p.get('goles_visit', 0) or 0)
            goals_for[lo] += w * gl;  goals_ag[lo]  += w * gv;  matches_h[lo] += w
            goals_for[vi] += w * gv;  goals_ag[vi]  += w * gl;  matches_a[vi] += w

    teams = set(goals_for.keys())
    tot_gf = sum(goals_for.values())
    tot_m  = sum(matches_h.values()) + sum(matches_a.values())
    mu     = tot_gf / tot_m if tot_m else 1.35

    att, defe = {}, {}
    for t in teams:
        m = matches_h.get(t, 0) + matches_a.get(t, 0)
        att[t]  = (goals_for[t] / m) / mu if m > 0 else 1.0
        defe[t] = (goals_ag[t]  / m) / mu if m > 0 else 1.0

    return att, defe, mu, 1.15

def predict_match(local, visitante, att, defe, mu, home_adv, max_g=6):
    lo, vi = norm(local), norm(visitante)
    lam_l  = att.get(lo, 1.0) * defe.get(vi, 1.0) * mu * home_adv
    lam_v  = att.get(vi, 1.0) * defe.get(lo, 1.0) * mu

    probs = np.array([[poisson.pmf(g1, lam_l) * poisson.pmf(g2, lam_v)
                       for g2 in range(max_g + 1)]
                      for g1 in range(max_g + 1)])

    p_local = float(np.sum(np.tril(probs, -1)))
    p_emp   = float(np.trace(probs))
    p_visit = float(np.sum(np.triu(probs, 1)))
    return p_local, p_emp, p_visit, probs, lam_l, lam_v

# ─────────────────────────────────────────────────────────────────────────────
# MONTECARLO
# ─────────────────────────────────────────────────────────────────────────────
def run_montecarlo(att, defe, mu, home_adv, n_sim=5000):
    fpath = HIST_DIR / 'historico_clausura_2026.json'
    data  = json.load(open(fpath, encoding='utf-8'))

    tabla = {norm(r['equipo']): {'pts': int(r.get('pts', 0)),
                                  'gf':  int(r.get('gf',  0)),
                                  'gc':  int(r.get('gc',  0))}
             for r in data.get('tabla', [])}

    pendientes = [(norm(p['local']), norm(p['visitante']))
                  for p in data.get('partidos', [])
                  if not p.get('terminado')]

    teams   = sorted(tabla.keys())
    n       = len(teams)
    pos_cnt = np.zeros((n, n), dtype=np.int32)

    for _ in range(n_sim):
        pts = {t: tabla[t]['pts'] for t in teams}
        gf  = {t: tabla[t]['gf']  for t in teams}
        gc  = {t: tabla[t]['gc']  for t in teams}

        for lo, vi in pendientes:
            if lo not in pts or vi not in pts:
                continue
            gl = np.random.poisson(att.get(lo,1.0)*defe.get(vi,1.0)*mu*home_adv)
            gv = np.random.poisson(att.get(vi,1.0)*defe.get(lo,1.0)*mu)
            gf[lo] += gl; gc[lo] += gv
            gf[vi] += gv; gc[vi] += gl
            if gl > gv:   pts[lo] += 3
            elif gl == gv: pts[lo] += 1; pts[vi] += 1
            else:          pts[vi] += 3

        ranking = sorted(teams, key=lambda t: (pts[t], gf[t]-gc[t], gf[t]), reverse=True)
        for pos, t in enumerate(ranking):
            pos_cnt[teams.index(t), pos] += 1

    return teams, pos_cnt / n_sim

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def load_elo():
    df = pd.read_csv(ELO_CSV, parse_dates=['fecha'])
    df['equipo'] = df['equipo'].apply(norm)
    return df

def load_players():
    df = pd.read_csv(JUG_CSV)
    df.columns = df.columns.str.strip('\ufeff')
    df['equipo'] = df['equipo'].apply(norm)
    df = df[df['minutos_stats'] >= 200].copy()
    return df

def load_j13():
    data = json.load(open(HIST_DIR / 'historico_clausura_2026.json', encoding='utf-8'))
    return [(norm(p['local']), norm(p['visitante']))
            for p in data.get('partidos', [])
            if p.get('jornada') == '13']

# ─────────────────────────────────────────────────────────────────────────────
# PRE-CARGA
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando modelo Poisson...")
ATT, DEFE, MU, HOME_ADV = load_poisson_model()

print("Simulación Monte Carlo (5,000 escenarios)...")
MC_TEAMS, MC_PROBS = run_montecarlo(ATT, DEFE, MU, HOME_ADV, n_sim=5000)

print("ELO + Jugadores...")
ELO_DF = load_elo()
JUG_DF = load_players()
J13    = load_j13()
print("Listo.\n")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _rgba(hex_c, a):
    h = hex_c.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

def fig_score_heatmap(probs, local, visitante):
    max_g = probs.shape[0] - 1
    z     = (probs * 100).round(1)
    text  = [[f'{z[i,j]:.1f}%' for j in range(max_g+1)] for i in range(max_g+1)]

    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(max_g+1)), y=list(range(max_g+1)),
        text=text, texttemplate='%{text}', textfont=dict(size=9),
        colorscale=VERDE_QUETZAL, showscale=True,
        colorbar=dict(tickfont=dict(color=GRAY, size=10),
                      title=dict(text='%', font=dict(color=GRAY))),
        zmin=0, zmax=30,
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text='Distribución de marcadores', font=dict(size=13, color=GRAY), x=0.5),
        xaxis=dict(title=dict(text=f'Goles {visitante}', font=dict(color=GRAY, size=11)),
                   tickfont=dict(color=WHITE), gridcolor='#1e2530'),
        yaxis=dict(title=dict(text=f'Goles {local}', font=dict(color=GRAY, size=11)),
                   tickfont=dict(color=WHITE), gridcolor='#1e2530'),
        height=360,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig

def fig_elo_ranking():
    latest = ELO_DF.sort_values('fecha').groupby('equipo').last().reset_index()
    latest = latest.sort_values('elo', ascending=True)
    colors = [TEAM_COLORS.get(t, RED) for t in latest['equipo']]

    fig = go.Figure(go.Bar(
        x=latest['elo'], y=latest['equipo'],
        orientation='h', marker_color=colors,
        text=latest['elo'].round(0).astype(int),
        textposition='outside', textfont=dict(size=11, color=WHITE),
        marker_line_width=0,
    ))

    # Add team shields as layout images
    layout_imgs = []
    for _, row in latest.iterrows():
        tid = TEAM_IDS.get(row['equipo'])
        if tid and (TEAMS_IMG / f'{tid}.png').exists():
            layout_imgs.append(dict(
                source=f'/images/teams/{tid}.png',
                xref='paper', yref='y',
                x=-0.01, y=row['equipo'],
                sizex=0.04, sizey=0.75,
                xanchor='right', yanchor='middle',
                layer='above',
            ))

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text='Ranking ELO — Clausura 2026',
                   font=dict(size=15, color=WHITE), x=0.5),
        xaxis=dict(title='Rating ELO', tickfont=dict(color=GRAY),
                   gridcolor='#1e2530', range=[1300, None]),
        yaxis=dict(tickfont=dict(size=11, color=WHITE)),
        height=620,
        margin=dict(l=160, r=80, t=50, b=40),
        images=layout_imgs,
    )
    return fig

def fig_elo_history(teams):
    fig = go.Figure()
    for t in teams:
        df_t = ELO_DF[ELO_DF['equipo'] == t].sort_values('fecha')
        if df_t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_t['fecha'], y=df_t['elo'],
            mode='lines', name=t,
            line=dict(color=TEAM_COLORS.get(t, GRAY), width=2.5, shape='spline',
                      smoothing=0.8),
        ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text='Evolución ELO Histórica', font=dict(size=13, color=GRAY), x=0.5),
        xaxis=dict(tickfont=dict(color=GRAY), gridcolor='#1e2530'),
        yaxis=dict(tickfont=dict(color=GRAY), gridcolor='#1e2530', title='ELO',
                   range=[1300, None]),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER, font=dict(color=WHITE, size=10)),
        height=340,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig

def fig_montecarlo():
    teams, probs = MC_TEAMS, MC_PROBS
    n = len(teams)

    expected = np.sum(probs * np.arange(1, n+1), axis=1)
    order    = np.argsort(expected)
    teams_s  = [teams[i] for i in order]
    probs_s  = probs[order, :]

    z    = (probs_s * 100).round(1)
    text = [[f'{z[i,j]:.0f}%' if z[i,j] >= 3 else '' for j in range(n)]
            for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(i+1) for i in range(n)], y=teams_s,
        text=text, texttemplate='%{text}',
        textfont=dict(size=10, color=WHITE),
        colorscale=VERDE_QUETZAL, showscale=True,
        colorbar=dict(title=dict(text='%', font=dict(color=GRAY)),
                      tickfont=dict(color=GRAY)),
        zmin=0, zmax=70,
        hovertemplate='<b>%{y}</b><br>Posición %{x}: %{z:.1f}%<extra></extra>',
    ))

    # Highlight TOP4 region
    fig.add_shape(type='rect', x0=-0.5, x1=3.5, y0=-0.5, y1=n-0.5,
                  fillcolor='rgba(255,215,0,0.04)', line=dict(width=0), layer='below')
    # Highlight TOP8 region
    fig.add_shape(type='rect', x0=3.5, x1=7.5, y0=-0.5, y1=n-0.5,
                  fillcolor='rgba(46,160,67,0.04)', line=dict(width=0), layer='below')
    # Divider lines
    fig.add_shape(type='line', x0=3.5, x1=3.5, y0=-0.5, y1=n-0.5,
                  line=dict(color=GOLD, width=2, dash='dot'))
    fig.add_shape(type='line', x0=7.5, x1=7.5, y0=-0.5, y1=n-0.5,
                  line=dict(color=GREEN, width=2, dash='dot'))
    # Labels
    fig.add_annotation(x=1.5, y=n-0.5, text='▲ TOP 4', showarrow=False,
                       font=dict(color=GOLD, size=11, family='Arial Black'), yshift=14)
    fig.add_annotation(x=5.5, y=n-0.5, text='▲ TOP 8', showarrow=False,
                       font=dict(color=GREEN, size=11, family='Arial Black'), yshift=14)

    # Bold text for probabilities >20% — overlay scatter
    bold_x, bold_y, bold_txt = [], [], []
    for i, t in enumerate(teams_s):
        for j in range(n):
            if z[i, j] >= 20:
                bold_x.append(str(j+1))
                bold_y.append(t)
                bold_txt.append(f'<b>{z[i,j]:.0f}%</b>')

    if bold_x:
        fig.add_trace(go.Scatter(
            x=bold_x, y=bold_y, mode='text',
            text=bold_txt,
            textfont=dict(size=13, color=WHITE, family='Arial Black'),
            hoverinfo='skip', showlegend=False,
        ))

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text='Simulación Monte Carlo — Clausura 2026 (5,000 escenarios)',
                   font=dict(size=15, color=WHITE), x=0.5),
        xaxis=dict(title='Posición final', tickfont=dict(color=WHITE),
                   side='top', gridcolor='#1e2530'),
        yaxis=dict(tickfont=dict(size=12, color=WHITE), autorange='reversed',
                   gridcolor='#1e2530'),
        height=700,
        margin=dict(l=150, r=40, t=90, b=40),
    )
    return fig

def fig_radar(player_row):
    pg   = pos_group(player_row.get('posicion', 'DEL'))
    cols = [c for c in RADAR_COLS.get(pg, RADAR_COLS['DEL']) if c in JUG_DF.columns]
    if not cols:
        return go.Figure()

    same_pos = JUG_DF[JUG_DF['posicion'].apply(pos_group) == pg]
    values, labels = [], []
    for c in cols:
        v   = float(player_row.get(c, 0) or 0)
        col_data = same_pos[c].dropna()
        pct = float((col_data <= v).mean() * 100) if len(col_data) > 0 else 50.0
        values.append(pct)
        labels.append(c.replace('_p90','').replace('_',' ').upper())

    team_c = TEAM_COLORS.get(player_row.get('equipo',''), RED)
    fill_c = _rgba(team_c, 0.3)
    vc = values + [values[0]]
    lc = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vc, theta=lc, fill='toself',
        fillcolor=fill_c,
        line=dict(color=team_c, width=2.5),
        name=player_row.get('nombre',''),
        hovertemplate='<b>%{theta}</b><br>Percentil: %{r:.0f}<extra></extra>',
    ))
    # Dotted guide circles
    for v in [25, 50, 75]:
        fig.add_trace(go.Scatterpolar(
            r=[v]*len(lc), theta=lc,
            mode='lines',
            line=dict(color=BORDER, width=0.5, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=BG_SEC,
            radialaxis=dict(visible=True, range=[0,100],
                            tickvals=[25,50,75],
                            tickfont=dict(color=GRAY, size=8),
                            gridcolor=BORDER, linecolor=BORDER,
                            tickcolor=BORDER),
            angularaxis=dict(tickfont=dict(color=WHITE, size=10),
                             gridcolor=BORDER, linecolor=BORDER),
        ),
        paper_bgcolor=BG_MAIN,
        font=dict(color=WHITE),
        title=dict(
            text=f"<b>{player_row.get('nombre','')}</b> — Percentiles vs {pg}",
            font=dict(size=13, color=WHITE), x=0.5),
        showlegend=False,
        height=420,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig

def fig_comparativo(row1, row2):
    pg1 = pos_group(row1.get('posicion','DEL'))
    pg2 = pos_group(row2.get('posicion','DEL'))
    # Common metric set
    cols = [c for c in RADAR_COLS.get(pg1, RADAR_COLS['DEL'])
            if c in JUG_DF.columns and c in RADAR_COLS.get(pg2, RADAR_COLS['DEL'])]
    if not cols:
        cols_all = set(RADAR_COLS['DEL'])
        for v in RADAR_COLS.values():
            cols_all &= set(v)
        cols = [c for c in cols_all if c in JUG_DF.columns]
    if not cols:
        return go.Figure()

    # Normalize by percentile in union of their position groups
    all_players = JUG_DF[JUG_DF['posicion'].apply(pos_group).isin([pg1, pg2])]

    labels, v1_pct, v2_pct, v1_raw, v2_raw = [], [], [], [], []
    for c in cols:
        col_d = all_players[c].dropna()
        vr1   = float(row1.get(c, 0) or 0)
        vr2   = float(row2.get(c, 0) or 0)
        p1    = float((col_d <= vr1).mean() * 100) if len(col_d) else 50.0
        p2    = float((col_d <= vr2).mean() * 100) if len(col_d) else 50.0
        labels.append(c.replace('_p90','').replace('_',' ').upper())
        v1_pct.append(p1); v2_pct.append(p2)
        v1_raw.append(vr1); v2_raw.append(vr2)

    c1 = TEAM_COLORS.get(row1.get('equipo',''), RED)
    c2 = TEAM_COLORS.get(row2.get('equipo',''), '#4CAF50')
    n1 = row1.get('nombre','J1')
    n2 = row2.get('nombre','J2')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[-v for v in v1_pct],
        orientation='h', name=n1,
        marker_color=c1,
        text=[f'{v:.1f}' for v in v1_raw],
        textposition='inside', textfont=dict(color=WHITE, size=10),
        hovertemplate=f'<b>{n1}</b><br>%{{y}}: %{{customdata:.1f}} (pct %{{x:.0f}})<extra></extra>',
        customdata=v1_raw,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=v2_pct,
        orientation='h', name=n2,
        marker_color=c2,
        text=[f'{v:.1f}' for v in v2_raw],
        textposition='inside', textfont=dict(color=WHITE, size=10),
        hovertemplate=f'<b>{n2}</b><br>%{{y}}: %{{customdata:.1f}} (pct %{{x:.0f}})<extra></extra>',
        customdata=v2_raw,
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        barmode='overlay',
        title=dict(text=f'<b>{n1}</b> vs <b>{n2}</b> — Comparativo por percentil',
                   font=dict(size=14, color=WHITE), x=0.5),
        xaxis=dict(tickvals=[-100,-75,-50,-25,0,25,50,75,100],
                   ticktext=['100','75','50','25','0','25','50','75','100'],
                   tickfont=dict(color=GRAY), gridcolor='#1e2530',
                   zeroline=True, zerolinecolor=BORDER, zerolinewidth=2,
                   range=[-110, 110]),
        yaxis=dict(tickfont=dict(size=11, color=WHITE)),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER,
                    font=dict(color=WHITE), x=0.5, xanchor='center',
                    orientation='h', y=-0.1),
        height=480,
        margin=dict(l=150, r=30, t=60, b=60),
    )
    # Center line annotation
    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE HEAVY FIGURES
# ─────────────────────────────────────────────────────────────────────────────
FIG_ELO_RANK = fig_elo_ranking()
FIG_MC       = fig_montecarlo()

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title='MauStats MX',
)

# Image static routes
@app.server.route('/images/teams/<filename>')
def serve_team_img(filename):
    return send_from_directory(str(TEAMS_IMG), filename)

@app.server.route('/images/players/<filename>')
def serve_player_img(filename):
    return send_from_directory(str(PLAYERS_IMG), filename)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def make_header():
    return html.Div([
        html.Div([
            html.Span('MAU',    className='mau-logo-text mau-logo-red'),
            html.Span('STATS',  className='mau-logo-text mau-logo-white',
                      style={'marginLeft':'4px'}),
            html.Span('_MX',    className='mau-logo-text mau-logo-gray',
                      style={'marginLeft':'4px'}),
            html.Div('LIGA MX · CLAUSURA 2026', className='mau-header-sub'),
        ], style={'display':'flex', 'alignItems':'baseline', 'gap':'0'}),
        html.Div('ESTADÍSTICAS', className='mau-header-badge'),
    ], className='mau-header')


def make_dd_match():
    opts = [
        {'label': html.Span([
            html.Img(src=shield_url(lo) or '', height='20px',
                     style={'marginRight':'6px', 'verticalAlign':'middle'}),
            html.Span(f'{lo}', style={'color': TEAM_COLORS.get(lo, WHITE)}),
            html.Span(' vs ', style={'color':GRAY, 'margin':'0 6px'}),
            html.Img(src=shield_url(vi) or '', height='20px',
                     style={'marginRight':'6px', 'verticalAlign':'middle'}),
            html.Span(f'{vi}', style={'color': TEAM_COLORS.get(vi, WHITE)}),
        ]),
         'value': f'{lo}||{vi}'}
        for lo, vi in J13
    ]
    return dcc.Dropdown(
        id='dd-match', options=opts,
        value=opts[0]['value'] if opts else None,
        clearable=False, optionHeight=36,
        style={'backgroundColor': BG_CARD, 'color': WHITE,
               'border': f'1px solid {BORDER}', 'borderRadius':'8px'},
    )


def prob_circle_el(pct, label, color, team_name='', shield_src=None):
    return html.Div([
        (html.Img(src=shield_src, className='team-shield-top') if shield_src
         else html.Div(style={'height':'56px'})),
        html.Div([
            html.Div(f'{pct:.1f}%', className='prob-value',
                     style={'color': color}),
            html.Div(label, className='prob-label'),
        ], className='prob-circle',
           style={'borderColor': color,
                  'boxShadow': f'0 0 20px {_rgba(color, 0.3)} inset',
                  'backgroundColor': _rgba(color, 0.06)}),
        html.Div(team_name, className='prob-team-name',
                 style={'color': color}),
    ], className='prob-wrapper')


# ── TAB 1: JORNADA 13 ─────────────────────────────────────────────────────────
TAB_J13 = dbc.Tab(label='JORNADA 13', tab_id='tab-j13', children=
    dbc.Container([
        dbc.Row(dbc.Col(make_dd_match(), md=10, lg=8), justify='center',
                className='mb-4 mt-3'),

        dbc.Row([
            # Probability circles
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div('PROBABILIDADES', className='section-label text-center'),
                html.Div(id='prob-circles',
                         style={'display':'flex', 'justifyContent':'space-around',
                                'alignItems':'center', 'padding':'8px 0',
                                'flexWrap':'wrap', 'gap':'12px'}),
                html.Div(id='lambda-info',
                         className='text-center text-muted mt-2',
                         style={'fontSize':'11px'}),
            ])), md=5),

            # Score heatmap
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(id='graph-heatmap', config={'displayModeBar': False}),
            ])), md=7),
        ], className='g-3'),
    ], fluid=True, className='p-3')
)

# ── TAB 2: RANKING ELO ────────────────────────────────────────────────────────
TAB_ELO = dbc.Tab(label='RANKING ELO', tab_id='tab-elo', children=
    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=FIG_ELO_RANK, config={'displayModeBar': False})
            )), md=5),

            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.Div('COMPARAR EVOLUCIÓN', className='section-label'),
                    dcc.Dropdown(
                        id='dd-elo-teams',
                        options=[{'label': t, 'value': t}
                                 for t in sorted(ELO_DF['equipo'].unique())],
                        value=['América', 'Chivas', 'Cruz Azul'],
                        multi=True, optionHeight=30,
                        style={'backgroundColor': BG_CARD, 'color': WHITE,
                               'border': f'1px solid {BORDER}', 'borderRadius':'8px'},
                    ),
                ]), className='mb-3'),
                dbc.Card(dbc.CardBody(
                    dcc.Graph(id='graph-elo-hist', config={'displayModeBar': False})
                )),
            ], md=7),
        ], className='g-3 mt-3'),
    ], fluid=True, className='p-3')
)

# ── TAB 3: SIMULACIÓN ─────────────────────────────────────────────────────────
TAB_MC = dbc.Tab(label='SIMULACIÓN', tab_id='tab-mc', children=
    dbc.Container([
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(figure=FIG_MC, config={'displayModeBar': False})
        ))), className='mt-3'),
        dbc.Row(dbc.Col(
            html.P([
                html.Span('5,000 escenarios · ', className='text-muted'),
                html.Span('─── TOP 4 ', style={'color': GOLD, 'fontSize':'12px'}),
                html.Span('Liguilla directa · ', className='text-muted'),
                html.Span('─── TOP 8 ', style={'color': GREEN, 'fontSize':'12px'}),
                html.Span('Repechaje · ', className='text-muted'),
                html.Span('Modelo Poisson ponderado por torneo', className='text-muted'),
            ], style={'textAlign':'center', 'fontSize':'11px', 'marginTop':'8px'}),
        )),
    ], fluid=True, className='p-3')
)

# ── TAB 4: JUGADORES ─────────────────────────────────────────────────────────
TAB_PLAYERS = dbc.Tab(label='JUGADORES', tab_id='tab-players', children=
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div('EQUIPO', className='section-label'),
                dcc.Dropdown(
                    id='dd-team-players',
                    options=[{'label': t, 'value': t}
                             for t in sorted(JUG_DF['equipo'].dropna().unique())],
                    value=sorted(JUG_DF['equipo'].dropna().unique())[0],
                    clearable=False,
                    style={'backgroundColor': BG_CARD, 'color': WHITE,
                           'border': f'1px solid {BORDER}', 'borderRadius':'8px'},
                ),
            ], md=4),
            dbc.Col([
                html.Div('JUGADOR (≥200 min)', className='section-label'),
                dcc.Dropdown(
                    id='dd-player', options=[], value=None,
                    clearable=False, optionHeight=30,
                    style={'backgroundColor': BG_CARD, 'color': WHITE,
                           'border': f'1px solid {BORDER}', 'borderRadius':'8px'},
                ),
            ], md=4),
        ], className='g-3 mt-3'),

        dbc.Row([
            # Left: photo + stats cards
            dbc.Col([
                dbc.Card(dbc.CardBody(
                    html.Div(id='player-profile', style={'textAlign':'center'})
                ), className='mb-3'),
                dbc.Card(dbc.CardBody(
                    html.Div(id='player-stats-cards')
                )),
            ], md=4),
            # Right: radar
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(id='graph-radar', config={'displayModeBar': False})
            )), md=8),
        ], className='g-3 mt-3'),
    ], fluid=True, className='p-3')
)

# ── TAB 5: COMPARATIVO ───────────────────────────────────────────────────────
TAB_COMP = dbc.Tab(label='COMPARATIVO', tab_id='tab-comp', children=
    dbc.Container([
        dbc.Row([
            # Jugador 1
            dbc.Col([
                html.Div('JUGADOR 1', className='section-label'),
                dcc.Dropdown(
                    id='comp-team-1',
                    options=[{'label': t, 'value': t}
                             for t in sorted(JUG_DF['equipo'].dropna().unique())],
                    value=sorted(JUG_DF['equipo'].dropna().unique())[0],
                    clearable=False,
                    style={'backgroundColor': BG_CARD, 'color': WHITE,
                           'border': f'1px solid {BORDER}', 'borderRadius':'8px',
                           'marginBottom':'8px'},
                ),
                dcc.Dropdown(
                    id='comp-player-1', options=[], value=None, clearable=False,
                    style={'backgroundColor': BG_CARD, 'color': WHITE,
                           'border': f'1px solid {BORDER}', 'borderRadius':'8px'},
                ),
            ], md=4),

            dbc.Col(html.Div([
                html.Div('VS', className='vs-badge')
            ], className='vs-divider'), md=2,
                style={'display':'flex', 'alignItems':'flex-end', 'justifyContent':'center',
                       'paddingBottom':'4px'}),

            # Jugador 2
            dbc.Col([
                html.Div('JUGADOR 2', className='section-label'),
                dcc.Dropdown(
                    id='comp-team-2',
                    options=[{'label': t, 'value': t}
                             for t in sorted(JUG_DF['equipo'].dropna().unique())],
                    value=sorted(JUG_DF['equipo'].dropna().unique())[1],
                    clearable=False,
                    style={'backgroundColor': BG_CARD, 'color': WHITE,
                           'border': f'1px solid {BORDER}', 'borderRadius':'8px',
                           'marginBottom':'8px'},
                ),
                dcc.Dropdown(
                    id='comp-player-2', options=[], value=None, clearable=False,
                    style={'backgroundColor': BG_CARD, 'color': WHITE,
                           'border': f'1px solid {BORDER}', 'borderRadius':'8px'},
                ),
            ], md=4),
        ], className='g-3 mt-3', justify='center'),

        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id='graph-comp', config={'displayModeBar': False})
        ))), className='mt-3'),

        dbc.Row(dbc.Col(
            html.Div(id='comp-shields',
                     style={'display':'flex', 'justifyContent':'center',
                            'gap':'40px', 'marginTop':'12px'}),
        )),
    ], fluid=True, className='p-3')
)

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
app.layout = html.Div([
    make_header(),
    dbc.Tabs(
        [TAB_J13, TAB_ELO, TAB_MC, TAB_PLAYERS, TAB_COMP],
        id='main-tabs', active_tab='tab-j13',
        style={'backgroundColor': '#0a0e14',
               'borderBottom': f'1px solid {BORDER}'},
    ),
], style={'backgroundColor': BG_MAIN, 'minHeight': '100vh'})

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output('prob-circles',  'children'),
    Output('graph-heatmap', 'figure'),
    Output('lambda-info',   'children'),
    Input('dd-match', 'value'),
)
def update_j13(match_val):
    if not match_val:
        return [], go.Figure(), ''
    local, visitante = match_val.split('||')
    p_l, p_e, p_v, probs, lam_l, lam_v = predict_match(
        local, visitante, ATT, DEFE, MU, HOME_ADV)

    c_lo = TEAM_COLORS.get(local,    RED)
    c_vi = TEAM_COLORS.get(visitante, '#aaaaaa')

    circles = [
        prob_circle_el(p_l*100, 'LOCAL', c_lo, local,    shield_url(local)),
        prob_circle_el(p_e*100, 'EMPATE', GRAY, 'Empate', None),
        prob_circle_el(p_v*100, 'VISITANTE', c_vi, visitante, shield_url(visitante)),
    ]
    info = f'λ {local} = {lam_l:.2f}  ·  λ {visitante} = {lam_v:.2f}'
    return circles, fig_score_heatmap(probs, local, visitante), info


@app.callback(
    Output('graph-elo-hist', 'figure'),
    Input('dd-elo-teams', 'value'),
)
def update_elo_hist(teams):
    return fig_elo_history(teams or [])


@app.callback(
    Output('dd-player', 'options'),
    Output('dd-player', 'value'),
    Input('dd-team-players', 'value'),
)
def update_player_dropdown(team):
    if not team:
        return [], None
    players = (JUG_DF[JUG_DF['equipo'] == team]['nombre']
               .dropna().sort_values().tolist())
    opts = [{'label': p, 'value': p} for p in players]
    return opts, (opts[0]['value'] if opts else None)


@app.callback(
    Output('graph-radar',       'figure'),
    Output('player-profile',    'children'),
    Output('player-stats-cards', 'children'),
    Input('dd-player',      'value'),
    Input('dd-team-players','value'),
)
def update_player(player, team):
    if not player or not team:
        return go.Figure(), html.Div(), html.Div()

    rows = JUG_DF[(JUG_DF['equipo'] == team) & (JUG_DF['nombre'] == player)]
    if rows.empty:
        return go.Figure(), html.Div(), html.Div()
    row = rows.iloc[0]

    # Profile card
    pid      = row.get('id')
    img_url  = player_img_url(pid) if pd.notna(pid) else None
    team_c   = TEAM_COLORS.get(team, RED)
    tid      = TEAM_IDS.get(team)
    shld_url = f'/images/teams/{tid}.png' if tid else None

    profile = html.Div([
        html.Div([
            html.Img(src=img_url, className='player-photo',
                     style={'borderColor': team_c}) if img_url
            else html.Div('👤', className='player-photo-fallback',
                          style={'borderColor': team_c}),
            html.Img(src=shld_url, height='28px',
                     style={'position':'absolute', 'bottom':'0', 'right':'0',
                            'filter':'drop-shadow(0 1px 3px rgba(0,0,0,0.7))'})
            if shld_url else html.Div(),
        ], className='player-photo-wrap'),
        html.Div(player, className='player-name-header mt-2'),
        html.Div(team, className='player-team-header', style={'color': team_c}),
        html.Span(str(row.get('posicion','')), className='badge-jornada mt-1',
                  style={'display':'inline-block'}),
    ])

    # Stats cards
    stat_fields = [
        ('minutos_stats', 'rating', 'goles', 'asistencias'),
        ('goles_p90', 'xG_p90', 'tiros_p90', 'asistencias_p90', 'pases_precisos_p90'),
    ]
    all_fields = ['minutos_stats','rating','goles','asistencias',
                  'goles_p90','xG_p90','tiros_p90','asistencias_p90','pases_precisos_p90']

    cards = []
    for field in all_fields:
        if field not in JUG_DF.columns:
            continue
        meta = STAT_META.get(field, ('📋', field.replace('_',' ').title()))
        val  = row.get(field)
        if pd.isna(val):
            continue
        disp = f'{float(val):.2f}' if field.endswith('_p90') else (
               f'{float(val):.1f}' if field in ('rating',) else
               f'{int(val):,}' if field in ('minutos_stats','valor_mercado_eur') else
               str(int(val) if float(val) == int(float(val)) else round(float(val),2)))
        cards.append(
            html.Div([
                html.Div(meta[0], className='stat-icon'),
                html.Div([
                    html.Div(meta[1], className='stat-label'),
                    html.Div(disp, className='stat-value', style={'color': team_c}),
                ])
            ], className='stat-card mb-2')
        )

    return fig_radar(row), profile, html.Div(cards)


@app.callback(
    Output('comp-player-1', 'options'),
    Output('comp-player-1', 'value'),
    Input('comp-team-1', 'value'),
)
def upd_comp_p1(team):
    if not team:
        return [], None
    players = JUG_DF[JUG_DF['equipo']==team]['nombre'].dropna().sort_values().tolist()
    opts = [{'label': p, 'value': p} for p in players]
    return opts, (opts[0]['value'] if opts else None)


@app.callback(
    Output('comp-player-2', 'options'),
    Output('comp-player-2', 'value'),
    Input('comp-team-2', 'value'),
)
def upd_comp_p2(team):
    if not team:
        return [], None
    players = JUG_DF[JUG_DF['equipo']==team]['nombre'].dropna().sort_values().tolist()
    opts = [{'label': p, 'value': p} for p in players]
    return opts, (opts[0]['value'] if opts else None)


@app.callback(
    Output('graph-comp',   'figure'),
    Output('comp-shields', 'children'),
    Input('comp-player-1', 'value'),
    Input('comp-team-1',   'value'),
    Input('comp-player-2', 'value'),
    Input('comp-team-2',   'value'),
)
def update_comp(p1, t1, p2, t2):
    if not all([p1, t1, p2, t2]):
        return go.Figure(), html.Div()

    r1 = JUG_DF[(JUG_DF['equipo']==t1) & (JUG_DF['nombre']==p1)]
    r2 = JUG_DF[(JUG_DF['equipo']==t2) & (JUG_DF['nombre']==p2)]
    if r1.empty or r2.empty:
        return go.Figure(), html.Div()

    fig = fig_comparativo(r1.iloc[0], r2.iloc[0])

    c1, c2 = TEAM_COLORS.get(t1, RED), TEAM_COLORS.get(t2, '#4CAF50')
    id1 = TEAM_IDS.get(t1)
    id2 = TEAM_IDS.get(t2)

    shields = html.Div([
        html.Div([
            html.Img(src=f'/images/teams/{id1}.png', height='48px') if id1 else html.Div(),
            html.Div(p1, style={'color': c1, 'fontWeight':'700', 'marginTop':'4px',
                                'fontSize':'13px'}),
        ], style={'textAlign':'center'}),
        html.Div(html.Span('vs', style={'color': GRAY, 'fontSize':'18px',
                                         'fontWeight':'700'}),
                 style={'display':'flex', 'alignItems':'center'}),
        html.Div([
            html.Img(src=f'/images/teams/{id2}.png', height='48px') if id2 else html.Div(),
            html.Div(p2, style={'color': c2, 'fontWeight':'700', 'marginTop':'4px',
                                'fontSize':'13px'}),
        ], style={'textAlign':'center'}),
    ], style={'display':'flex', 'gap':'40px', 'justifyContent':'center',
              'alignItems':'center'})

    return fig, shields


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
