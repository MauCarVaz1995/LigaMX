#!/usr/bin/env python3
"""
scripts/dashboard/app.py
Dashboard interactivo MauStats_MX — Plotly Dash
Pestañas: Jornada 13 | Ranking ELO | Simulación Montecarlo | Jugadores
"""

import json, glob, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent.parent
HIST_DIR = BASE / 'data/raw/historico'
ELO_CSV  = BASE / 'data/processed/elo_historico.csv'
JUG_CSV  = BASE / 'data/processed/jugadores_clausura2026.csv'

# ─────────────────────────────────────────────────────────────────────────────
# PALETA & CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
BG_MAIN   = '#0d1117'
BG_SEC    = '#161b22'
BG_CARD   = '#0f151e'
BORDER    = '#30363d'
WHITE     = '#ffffff'
GRAY      = '#8b949e'
RED       = '#D5001C'
GREEN     = '#2ea043'
NEGATIVE  = '#f85149'

TEAM_COLORS = {
    'Chivas':         '#CD1F2D',
    'Cruz Azul':      '#0047AB',
    'Toluca':         '#D5001C',
    'América':        '#FFD700',
    'Tigres':         '#F5A623',
    'Monterrey':      '#003DA5',
    'Pumas':          '#C8A84B',
    'Santos Laguna':  '#2E8B57',
    'Pachuca':        '#A8B8C8',
    'Atlas':          '#B22222',
    'León':           '#2D8C3C',
    'Necaxa':         '#D62828',
    'Tijuana':        '#C62828',
    'Querétaro':      '#1A7FCB',
    'FC Juárez':      '#4CAF50',
    'Mazatlán':       '#9B59B6',
    'San Luis':       '#D52B1E',
    'Puebla':         '#2563EB',
}

TORNEO_WEIGHTS = {
    '2025/2026 - Clausura': 4,
    '2025/2026 - Apertura': 3,
    '2024/2025 - Clausura': 2,
    '2024/2025 - Apertura': 1,
}

_ALIAS = {
    'cf america': 'América',       'america': 'América',
    'chivas': 'Chivas',            'guadalajara': 'Chivas',
    'cruz azul': 'Cruz Azul',
    'tigres': 'Tigres',            'tigres uanl': 'Tigres',
    'monterrey': 'Monterrey',      'cf monterrey': 'Monterrey',
    'pumas': 'Pumas',              'pumas unam': 'Pumas',
    'toluca': 'Toluca',
    'santos laguna': 'Santos Laguna',
    'pachuca': 'Pachuca',
    'atlas': 'Atlas',
    'león': 'León',                'leon': 'León',
    'necaxa': 'Necaxa',
    'tijuana': 'Tijuana',
    'querétaro': 'Querétaro',      'queretaro': 'Querétaro',
    'queretaro fc': 'Querétaro',
    'fc juárez': 'FC Juárez',      'fc juarez': 'FC Juárez',
    'mazatlán': 'Mazatlán',        'mazatlan fc': 'Mazatlán',
    'mazatlan': 'Mazatlán',
    'atletico de san luis': 'San Luis', 'san luis': 'San Luis',
    'puebla': 'Puebla',
}

def norm(n):
    return _ALIAS.get(str(n).lower().strip(), n.strip())

# ─────────────────────────────────────────────────────────────────────────────
# MODELO POISSON
# ─────────────────────────────────────────────────────────────────────────────
def load_poisson_model():
    """Carga partidos históricos ponderados y devuelve parámetros de ataque/defensa."""
    goals_for   = defaultdict(float)
    goals_ag    = defaultdict(float)
    matches_h   = defaultdict(float)
    matches_a   = defaultdict(float)
    total_w     = 0.0

    for fpath in sorted(glob.glob(str(HIST_DIR / '*.json'))):
        stem = Path(fpath).stem.replace('historico_', '')
        parts = stem.split('_-_', 1)
        if len(parts) != 2:
            continue
        year  = parts[0].replace('_', '/')
        torneo = parts[1].replace('_', ' ').title()
        tkey  = f'{year} - {torneo}'
        w = TORNEO_WEIGHTS.get(tkey)
        if not w:
            continue

        data = json.load(open(fpath, encoding='utf-8'))
        for p in data.get('partidos', []):
            if not p.get('terminado'):
                continue
            lo = norm(p['local'])
            vi = norm(p['visitante'])
            gl = int(p.get('goles_local',  0) or 0)
            gv = int(p.get('goles_visit',  0) or 0)
            goals_for[lo]  += w * gl
            goals_ag[lo]   += w * gv
            matches_h[lo]  += w
            goals_for[vi]  += w * gv
            goals_ag[vi]   += w * gl
            matches_a[vi]  += w
            total_w += w

    teams = set(goals_for.keys())
    tot_gf = sum(goals_for.values())
    tot_m  = sum(matches_h.values()) + sum(matches_a.values())
    mu     = tot_gf / tot_m if tot_m else 1.35

    att, defe = {}, {}
    for t in teams:
        m = matches_h.get(t, 0) + matches_a.get(t, 0)
        if m > 0:
            att[t]  = (goals_for[t] / m) / mu
            defe[t] = (goals_ag[t]  / m) / mu
        else:
            att[t]  = 1.0
            defe[t] = 1.0

    HOME_ADV = 1.15
    return att, defe, mu, HOME_ADV

def predict_match(local, visitante, att, defe, mu, home_adv, max_g=6):
    """Devuelve (p_local, p_empate, p_visitante, heatmap NxN, lam_l, lam_v)."""
    lo  = norm(local)
    vi  = norm(visitante)
    lam_l = att.get(lo, 1.0) * defe.get(vi, 1.0) * mu * home_adv
    lam_v = att.get(vi, 1.0) * defe.get(lo, 1.0) * mu

    probs = np.zeros((max_g + 1, max_g + 1))
    for g1 in range(max_g + 1):
        for g2 in range(max_g + 1):
            probs[g1, g2] = poisson.pmf(g1, lam_l) * poisson.pmf(g2, lam_v)

    p_local = float(np.sum(np.tril(probs, -1)))
    p_emp   = float(np.trace(probs))
    p_visit = float(np.sum(np.triu(probs, 1)))
    return p_local, p_emp, p_visit, probs, lam_l, lam_v

# ─────────────────────────────────────────────────────────────────────────────
# MONTECARLO (pre-computado al arrancar)
# ─────────────────────────────────────────────────────────────────────────────
def run_montecarlo(att, defe, mu, home_adv, n_sim=5000):
    fpath = HIST_DIR / 'historico_clausura_2026.json'
    data  = json.load(open(fpath, encoding='utf-8'))

    tabla = {}
    for row in data.get('tabla', []):
        equipo = norm(row['equipo'])
        tabla[equipo] = {
            'pts': int(row.get('pts', 0)),
            'gf':  int(row.get('gf',  0)),
            'gc':  int(row.get('gc',  0)),
        }

    pendientes = []
    for p in data.get('partidos', []):
        if not p.get('terminado'):
            pendientes.append((norm(p['local']), norm(p['visitante'])))

    teams = sorted(tabla.keys())
    n_teams = len(teams)
    pos_count = np.zeros((n_teams, n_teams), dtype=np.int32)

    for _ in range(n_sim):
        pts  = {t: tabla[t]['pts'] for t in teams}
        gf   = {t: tabla[t]['gf']  for t in teams}
        gc   = {t: tabla[t]['gc']  for t in teams}

        for lo, vi in pendientes:
            if lo not in pts or vi not in pts:
                continue
            lam_l = att.get(lo, 1.0) * defe.get(vi, 1.0) * mu * home_adv
            lam_v = att.get(vi, 1.0) * defe.get(lo, 1.0) * mu
            gl = np.random.poisson(lam_l)
            gv = np.random.poisson(lam_v)
            gf[lo] += gl; gc[lo] += gv
            gf[vi] += gv; gc[vi] += gl
            if gl > gv:
                pts[lo] += 3
            elif gl == gv:
                pts[lo] += 1; pts[vi] += 1
            else:
                pts[vi] += 3

        ranking = sorted(teams,
                         key=lambda t: (pts[t], gf[t] - gc[t], gf[t]),
                         reverse=True)
        for pos, t in enumerate(ranking):
            ti = teams.index(t)
            pos_count[ti, pos] += 1

    probs = pos_count / n_sim
    return teams, probs

# ─────────────────────────────────────────────────────────────────────────────
# DATOS JORNADA 13
# ─────────────────────────────────────────────────────────────────────────────
def load_j13_matches():
    fpath = HIST_DIR / 'historico_clausura_2026.json'
    data  = json.load(open(fpath, encoding='utf-8'))
    return [
        (norm(p['local']), norm(p['visitante']))
        for p in data.get('partidos', [])
        if p.get('jornada') == '13'
    ]

# ─────────────────────────────────────────────────────────────────────────────
# DATOS ELO
# ─────────────────────────────────────────────────────────────────────────────
def load_elo():
    df = pd.read_csv(ELO_CSV, parse_dates=['fecha'])
    df['equipo'] = df['equipo'].apply(norm)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# DATOS JUGADORES
# ─────────────────────────────────────────────────────────────────────────────
def load_players():
    df = pd.read_csv(JUG_CSV)
    df.columns = df.columns.str.strip('\ufeff')
    df['equipo'] = df['equipo'].apply(norm)
    return df

RADAR_COLS = {
    'POR': ['paradas_p90', 'porcentaje_paradas_p90', 'goles_evitados_p90',
            'goles_recibidos_p90', 'porterias_cero_p90'],
    'DEF': ['intercepciones_p90', 'entradas_p90', 'despejes_p90',
            'duelos_tierra_ganados_p90', 'faltas_cometidas_p90'],
    'MED': ['pases_precisos_p90', 'pases_largos_p90', 'recuperaciones_campo_rival_p90',
            'chances_creadas_p90', 'asistencias_p90'],
    'DEL': ['goles_p90', 'xG_p90', 'tiros_p90', 'tiros_a_puerta_p90', 'grandes_chances_p90'],
}

def pos_group(pos):
    p = str(pos).upper()
    if 'GK' in p or 'POR' in p:
        return 'POR'
    if any(x in p for x in ['CB', 'LB', 'RB', 'WB', 'DEF']):
        return 'DEF'
    if any(x in p for x in ['CM', 'DM', 'AM', 'MID', 'MF']):
        return 'MED'
    return 'DEL'

# ─────────────────────────────────────────────────────────────────────────────
# PRE-CARGA AL ARRANCAR
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando modelo Poisson...")
ATT, DEFE, MU, HOME_ADV = load_poisson_model()

print("Corriendo simulación Monte Carlo (5,000 escenarios)...")
MC_TEAMS, MC_PROBS = run_montecarlo(ATT, DEFE, MU, HOME_ADV, n_sim=5000)

print("Cargando datos ELO y jugadores...")
ELO_DF  = load_elo()
JUG_DF  = load_players()
J13     = load_j13_matches()

print("Dashboard listo.")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_MAIN,
    plot_bgcolor=BG_SEC,
    font=dict(color=WHITE, family='Arial'),
    colorway=[RED, '#4CAF50', '#2196F3', '#FF9800', '#9C27B0',
              '#00BCD4', '#FFEB3B', '#F44336', '#8BC34A', '#E91E63'],
)
_DEF_MARGIN = dict(l=10, r=10, t=40, b=10)

def _layout(**overrides):
    """Merge PLOTLY_LAYOUT with per-figure overrides (margin-safe)."""
    base = dict(**PLOTLY_LAYOUT)
    base['margin'] = overrides.pop('margin', _DEF_MARGIN)
    base.update(overrides)
    return base

def card(children, **kwargs):
    return dbc.Card(children,
                    style={'backgroundColor': BG_CARD,
                           'border': f'1px solid {BORDER}',
                           'borderRadius': '8px'},
                    **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────────────────────────────────────

def fig_prob_bars(p_local, p_emp, p_visit, local, visitante):
    lo_c  = TEAM_COLORS.get(local,    RED)
    vi_c  = TEAM_COLORS.get(visitante, '#aaaaaa')
    cats  = [local, 'Empate', visitante]
    vals  = [round(p_local*100, 1), round(p_emp*100, 1), round(p_visit*100, 1)]
    cols  = [lo_c, GRAY, vi_c]
    fig = go.Figure(go.Bar(
        x=cats, y=vals,
        marker_color=cols,
        text=[f'{v}%' for v in vals],
        textposition='outside',
        textfont=dict(size=18, color=WHITE),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Probabilidades de resultado', font=dict(size=14, color=GRAY), x=0.5),
        yaxis=dict(range=[0, 110], showgrid=False, showticklabels=False,
                   zeroline=False),
        xaxis=dict(tickfont=dict(size=13)),
        showlegend=False,
        height=280,
    )
    return fig

def fig_score_heatmap(probs, local, visitante):
    max_g = probs.shape[0] - 1
    z = (probs * 100).round(1)
    text = [[f'{z[i,j]:.1f}%' for j in range(max_g + 1)]
            for i in range(max_g + 1)]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(max_g + 1)),
        y=list(range(max_g + 1)),
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=9),
        colorscale=[[0, BG_SEC], [0.3, '#1B3A4B'], [0.7, '#0E6655'], [1.0, '#00FF88']],
        showscale=True,
        colorbar=dict(tickfont=dict(color=GRAY), title=dict(text='%', font=dict(color=GRAY))),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Distribución de marcadores', font=dict(size=14, color=GRAY), x=0.5),
        xaxis=dict(title=dict(text=f'Goles {visitante}', font=dict(color=GRAY)),
                   tickfont=dict(color=WHITE)),
        yaxis=dict(title=dict(text=f'Goles {local}', font=dict(color=GRAY)),
                   tickfont=dict(color=WHITE)),
        height=380,
    )
    return fig

def fig_elo_ranking(df_elo, torneo_filter=None):
    latest = df_elo.sort_values('fecha').groupby('equipo').last().reset_index()
    if torneo_filter:
        latest = latest[latest['torneo'] == torneo_filter]
    latest = latest.sort_values('elo', ascending=True)
    colors = [TEAM_COLORS.get(t, RED) for t in latest['equipo']]

    fig = go.Figure(go.Bar(
        x=latest['elo'],
        y=latest['equipo'],
        orientation='h',
        marker_color=colors,
        text=latest['elo'].round(0).astype(int),
        textposition='outside',
        textfont=dict(size=11, color=WHITE),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Ranking ELO Actual', font=dict(size=16, color=WHITE), x=0.5),
        xaxis=dict(title='Rating ELO', tickfont=dict(color=GRAY),
                   gridcolor='#1e2530'),
        yaxis=dict(tickfont=dict(size=11)),
        height=600,
        margin=dict(l=140, r=80, t=50, b=40),
    )
    return fig

def fig_elo_history(df_elo, teams):
    fig = go.Figure()
    for t in teams:
        df_t = df_elo[df_elo['equipo'] == t].sort_values('fecha')
        if df_t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_t['fecha'], y=df_t['elo'],
            mode='lines',
            name=t,
            line=dict(color=TEAM_COLORS.get(t, GRAY), width=2),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Evolución ELO Histórica', font=dict(size=14, color=GRAY), x=0.5),
        xaxis=dict(tickfont=dict(color=GRAY), gridcolor='#1e2530'),
        yaxis=dict(tickfont=dict(color=GRAY), gridcolor='#1e2530', title='ELO'),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER, font=dict(color=WHITE)),
        height=350,
    )
    return fig

def fig_montecarlo(teams, probs):
    n = len(teams)
    # Sort teams by expected position
    expected = np.sum(probs * np.arange(1, n + 1), axis=1)
    order    = np.argsort(expected)
    teams_s  = [teams[i] for i in order]
    probs_s  = probs[order, :]

    z    = (probs_s * 100).round(1)
    text = [[f'{z[i,j]:.0f}%' if z[i,j] >= 5 else '' for j in range(n)]
            for i in range(n)]

    colorscale = [
        [0.00, BG_MAIN],
        [0.05, '#1B5E20'],
        [0.15, '#2E7D32'],
        [0.40, '#00C853'],
        [1.00, '#00FF88'],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(i+1) for i in range(n)],
        y=teams_s,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=9),
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=dict(text='%', font=dict(color=GRAY)),
                      tickfont=dict(color=GRAY)),
        zmin=0, zmax=80,
    ))
    fig.add_shape(type='line', x0=3.5, x1=3.5, y0=-0.5, y1=n-0.5,
                  line=dict(color='#FFD700', width=2, dash='dot'))
    fig.add_shape(type='line', x0=7.5, x1=7.5, y0=-0.5, y1=n-0.5,
                  line=dict(color='#4CAF50', width=2, dash='dot'))
    fig.add_annotation(x=1.5, y=n-0.3, text='TOP 4', showarrow=False,
                       font=dict(color='#FFD700', size=10))
    fig.add_annotation(x=5.5, y=n-0.3, text='TOP 8', showarrow=False,
                       font=dict(color='#4CAF50', size=10))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Simulación Monte Carlo — Clausura 2026 (5,000 escenarios)',
                   font=dict(size=16, color=WHITE), x=0.5),
        xaxis=dict(title='Posición Final', tickfont=dict(color=WHITE),
                   side='top'),
        yaxis=dict(tickfont=dict(size=11, color=WHITE), autorange='reversed'),
        height=700,
        margin=dict(l=140, r=40, t=80, b=40),
    )
    return fig

def fig_radar(player_row, jug_df):
    pg = pos_group(player_row['posicion'])
    cols = RADAR_COLS.get(pg, RADAR_COLS['DEL'])
    # Filter to existing columns
    cols = [c for c in cols if c in jug_df.columns]
    if not cols:
        return go.Figure()

    # Percentile within position group
    same_pos = jug_df[jug_df['posicion'].apply(pos_group) == pg]
    values, labels = [], []
    for c in cols:
        col_data = same_pos[c].dropna()
        v = player_row.get(c, 0) or 0
        pct = float((col_data <= v).mean()) * 100 if len(col_data) > 0 else 50.0
        values.append(pct)
        labels.append(c.replace('_p90', '').replace('_', ' ').upper())

    # Close the polygon
    values_c = values + [values[0]]
    labels_c = labels + [labels[0]]

    team_c = TEAM_COLORS.get(player_row.get('equipo', ''), RED)
    fig = go.Figure(go.Scatterpolar(
        r=values_c,
        theta=labels_c,
        fill='toself',
        fillcolor=f'rgba({int(team_c[1:3],16)},{int(team_c[3:5],16)},{int(team_c[5:7],16)},0.3)',
        line=dict(color=team_c, width=2),
        name=player_row.get('nombre', ''),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG_SEC,
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(color=GRAY, size=8),
                            gridcolor=BORDER),
            angularaxis=dict(tickfont=dict(color=WHITE, size=10),
                             gridcolor=BORDER),
        ),
        paper_bgcolor=BG_MAIN,
        font=dict(color=WHITE),
        title=dict(
            text=f"{player_row.get('nombre','')}  —  {player_row.get('equipo','')}  ({pg})",
            font=dict(size=14, color=WHITE), x=0.5),
        showlegend=False,
        height=450,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE FIGURAS PESADAS
# ─────────────────────────────────────────────────────────────────────────────
FIG_MC = fig_montecarlo(MC_TEAMS, MC_PROBS)
FIG_ELO_RANK = fig_elo_ranking(ELO_DF)

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title='MauStats MX',
)

# ── Opciones ──────────────────────────────────────────────────────────────────
j13_options = [
    {'label': f'{lo}  vs  {vi}', 'value': f'{lo}||{vi}'}
    for lo, vi in J13
]

teams_sorted = sorted(JUG_DF['equipo'].dropna().unique())
team_options = [{'label': t, 'value': t} for t in teams_sorted]

elo_teams = sorted(ELO_DF['equipo'].unique())
elo_team_opts = [{'label': t, 'value': t} for t in elo_teams]

# ── HEADER ────────────────────────────────────────────────────────────────────
header = dbc.Navbar(
    dbc.Container([
        html.Span('MAU', style={'color': RED, 'fontWeight': 'bold',
                                'fontSize': '22px', 'letterSpacing': '3px'}),
        html.Span('STATS', style={'color': WHITE, 'fontWeight': 'bold',
                                  'fontSize': '22px', 'letterSpacing': '3px'}),
        html.Span('_MX', style={'color': GRAY, 'fontSize': '18px',
                                'letterSpacing': '2px'}),
        html.Span(' · Liga MX Clausura 2026',
                  style={'color': GRAY, 'fontSize': '13px', 'marginLeft': '12px'}),
    ], fluid=True),
    color=BG_CARD,
    dark=True,
    style={'borderBottom': f'2px solid {RED}'},
)

# ── TABS ──────────────────────────────────────────────────────────────────────
TAB_STYLE       = {'backgroundColor': BG_CARD, 'color': GRAY,
                   'border': f'1px solid {BORDER}', 'padding': '8px 20px'}
TAB_STYLE_SEL   = {'backgroundColor': BG_MAIN, 'color': WHITE,
                   'borderBottom': f'2px solid {RED}', 'padding': '8px 20px'}

tab_j13 = dbc.Tab(
    label='JORNADA 13',
    tab_id='tab-j13',
    label_style=TAB_STYLE,
    active_label_style=TAB_STYLE_SEL,
    children=dbc.Container([
        dbc.Row(dbc.Col(
            dcc.Dropdown(
                id='dd-match',
                options=j13_options,
                value=j13_options[0]['value'] if j13_options else None,
                clearable=False,
                style={'backgroundColor': BG_CARD, 'color': WHITE},
            ), width=8), className='mb-3 mt-3', justify='center'),

        dbc.Row([
            dbc.Col(card(dcc.Graph(id='graph-probs', config={'displayModeBar': False})),
                    md=5),
            dbc.Col(card(dcc.Graph(id='graph-heatmap', config={'displayModeBar': False})),
                    md=7),
        ], className='g-3'),

        dbc.Row(dbc.Col(
            html.Div(id='lev-summary', className='text-center mt-3',
                     style={'color': GRAY, 'fontSize': '13px'})
        )),
    ], fluid=True, className='p-3'),
)

tab_elo = dbc.Tab(
    label='RANKING ELO',
    tab_id='tab-elo',
    label_style=TAB_STYLE,
    active_label_style=TAB_STYLE_SEL,
    children=dbc.Container([
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=FIG_ELO_RANK,
                                   config={'displayModeBar': False})),
                    md=5),
            dbc.Col([
                card([
                    html.P('Comparar evolución histórica',
                           style={'color': GRAY, 'margin': '10px 10px 4px',
                                  'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='dd-elo-teams',
                        options=elo_team_opts,
                        value=['América', 'Chivas', 'Cruz Azul'],
                        multi=True,
                        style={'backgroundColor': BG_CARD, 'color': WHITE},
                    ),
                ], className='mb-3 p-2'),
                card(dcc.Graph(id='graph-elo-hist',
                               config={'displayModeBar': False})),
            ], md=7),
        ], className='g-3 mt-3'),
    ], fluid=True, className='p-3'),
)

tab_mc = dbc.Tab(
    label='SIMULACIÓN',
    tab_id='tab-mc',
    label_style=TAB_STYLE,
    active_label_style=TAB_STYLE_SEL,
    children=dbc.Container([
        dbc.Row(dbc.Col(
            card(dcc.Graph(figure=FIG_MC, id='graph-mc',
                           config={'displayModeBar': False})),
        ), className='mt-3'),
        dbc.Row(dbc.Col(
            html.P('Basado en 5,000 simulaciones Poisson con pesos históricos. '
                   'Líneas: TOP 4 (Liguilla directa) y TOP 8 (Repechaje).',
                   style={'color': GRAY, 'fontSize': '12px', 'textAlign': 'center',
                          'marginTop': '8px'}),
        )),
    ], fluid=True, className='p-3'),
)

tab_players = dbc.Tab(
    label='JUGADORES',
    tab_id='tab-players',
    label_style=TAB_STYLE,
    active_label_style=TAB_STYLE_SEL,
    children=dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label('Equipo', style={'color': GRAY, 'fontSize': '12px'}),
                dcc.Dropdown(
                    id='dd-team-players',
                    options=team_options,
                    value=teams_sorted[0] if teams_sorted else None,
                    clearable=False,
                    style={'backgroundColor': BG_CARD, 'color': WHITE},
                ),
            ], md=4),
            dbc.Col([
                html.Label('Jugador', style={'color': GRAY, 'fontSize': '12px'}),
                dcc.Dropdown(
                    id='dd-player',
                    options=[],
                    value=None,
                    clearable=False,
                    style={'backgroundColor': BG_CARD, 'color': WHITE},
                ),
            ], md=4),
        ], className='g-3 mt-3'),

        dbc.Row(dbc.Col(
            card(dcc.Graph(id='graph-radar', config={'displayModeBar': False}))
        ), className='mt-3'),

        dbc.Row(dbc.Col(
            html.Div(id='player-stats-table')
        ), className='mt-3'),
    ], fluid=True, className='p-3'),
)

# ── LAYOUT PRINCIPAL ──────────────────────────────────────────────────────────
app.layout = html.Div([
    header,
    dbc.Tabs(
        [tab_j13, tab_elo, tab_mc, tab_players],
        id='main-tabs',
        active_tab='tab-j13',
        style={'backgroundColor': BG_CARD,
               'borderBottom': f'1px solid {BORDER}',
               'paddingLeft': '16px'},
    ),
], style={'backgroundColor': BG_MAIN, 'minHeight': '100vh'})

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output('graph-probs',   'figure'),
    Output('graph-heatmap', 'figure'),
    Output('lev-summary',   'children'),
    Input('dd-match', 'value'),
)
def update_j13(match_val):
    if not match_val:
        return go.Figure(), go.Figure(), ''
    local, visitante = match_val.split('||')
    p_l, p_e, p_v, probs, lam_l, lam_v = predict_match(
        local, visitante, ATT, DEFE, MU, HOME_ADV)
    fig_p = fig_prob_bars(p_l, p_e, p_v, local, visitante)
    fig_h = fig_score_heatmap(probs, local, visitante)
    summary = (f'λ {local} = {lam_l:.2f} goles esperados  |  '
               f'λ {visitante} = {lam_v:.2f} goles esperados')
    return fig_p, fig_h, summary


@app.callback(
    Output('graph-elo-hist', 'figure'),
    Input('dd-elo-teams', 'value'),
)
def update_elo_hist(teams):
    if not teams:
        return go.Figure()
    return fig_elo_history(ELO_DF, teams)


@app.callback(
    Output('dd-player', 'options'),
    Output('dd-player', 'value'),
    Input('dd-team-players', 'value'),
)
def update_player_dropdown(team):
    if not team:
        return [], None
    players = JUG_DF[JUG_DF['equipo'] == team]['nombre'].dropna().sort_values()
    opts = [{'label': p, 'value': p} for p in players]
    return opts, (opts[0]['value'] if opts else None)


@app.callback(
    Output('graph-radar',       'figure'),
    Output('player-stats-table', 'children'),
    Input('dd-player', 'value'),
    Input('dd-team-players', 'value'),
)
def update_radar(player, team):
    if not player or not team:
        return go.Figure(), ''
    row = JUG_DF[(JUG_DF['equipo'] == team) & (JUG_DF['nombre'] == player)]
    if row.empty:
        return go.Figure(), ''
    row = row.iloc[0]
    fig = fig_radar(row, JUG_DF)

    # Mini stats table
    show_cols = ['posicion', 'edad', 'partidos_stats', 'minutos_stats',
                 'goles', 'asistencias', 'rating']
    show_cols = [c for c in show_cols if c in JUG_DF.columns]
    rows = []
    for c in show_cols:
        val = row.get(c, '—')
        if pd.isna(val):
            val = '—'
        rows.append(html.Tr([
            html.Td(c.replace('_', ' ').title(),
                    style={'color': GRAY, 'padding': '4px 12px', 'fontSize': '12px'}),
            html.Td(str(val),
                    style={'color': WHITE, 'padding': '4px 12px', 'fontSize': '12px',
                           'fontWeight': 'bold'}),
        ]))
    table = html.Table(rows, style={'margin': 'auto'})
    return fig, table


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
