#!/usr/bin/env python3
"""
scripts/dashboard/app.py
MauStats MX — Dashboard Plotly Dash
6 tabs: HOME · Jornada 13 · Ranking ELO · Simulación · Jugadores · Comparativo
"""

import json, glob, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from flask import send_from_directory

import dash
from dash import dcc, html, Input, Output
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
BG_MAIN = '#0d1117'
BG_SEC  = '#161b22'
BG_CARD = '#0f151e'
BORDER  = '#30363d'
DIVIDER = '#21262d'
WHITE   = '#ffffff'
GRAY    = '#8b949e'
RED     = '#D5001C'
GREEN   = '#2ea043'
GOLD    = '#FFD700'
DANGER  = '#f85149'

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

TEAM_IDS = {
    'Chivas': 7807, 'Cruz Azul': 6578, 'Toluca': 6618, 'Pumas': 1946,
    'Pachuca': 7848, 'Atlas': 6577, 'Tigres': 8561, 'América': 6576,
    'Monterrey': 7849, 'FC Juárez': 649424, 'Necaxa': 1842, 'León': 1841,
    'Tijuana': 162418, 'Puebla': 7847, 'San Luis': 6358,
    'Mazatlán': 1170234, 'Querétaro': 1943, 'Santos Laguna': 7857,
}

TORNEO_WEIGHTS = {
    '2025/2026 - Clausura': 4, '2025/2026 - Apertura': 3,
    '2024/2025 - Clausura': 2, '2024/2025 - Apertura': 1,
}

_ALIAS = {
    'cf america': 'América',     'america': 'América',
    'chivas': 'Chivas',          'guadalajara': 'Chivas',
    'cruz azul': 'Cruz Azul',
    'tigres': 'Tigres',          'tigres uanl': 'Tigres',
    'monterrey': 'Monterrey',    'cf monterrey': 'Monterrey',
    'pumas': 'Pumas',            'pumas unam': 'Pumas',
    'toluca': 'Toluca',
    'santos laguna': 'Santos Laguna',
    'pachuca': 'Pachuca',
    'atlas': 'Atlas',
    'león': 'León',              'leon': 'León',
    'necaxa': 'Necaxa',
    'tijuana': 'Tijuana',
    'querétaro': 'Querétaro',    'queretaro': 'Querétaro', 'queretaro fc': 'Querétaro',
    'fc juárez': 'FC Juárez',    'fc juarez': 'FC Juárez',
    'mazatlán': 'Mazatlán',      'mazatlan fc': 'Mazatlán', 'mazatlan': 'Mazatlán',
    'atletico de san luis': 'San Luis', 'san luis': 'San Luis',
    'puebla': 'Puebla',
}

def norm(n): return _ALIAS.get(str(n).lower().strip(), str(n).strip())
def shield(team): tid = TEAM_IDS.get(team); return f'/images/teams/{tid}.png' if tid else ''
def player_img(pid):
    return f'/images/players/{int(pid)}.png' if pid and (PLAYERS_IMG / f'{int(pid)}.png').exists() else None

def _rgba(h, a):
    h = h.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor=BG_MAIN,
    plot_bgcolor=BG_SEC,
    font=dict(color=WHITE, family='Arial'),
    template='plotly_dark',
)

VERDE_QUETZAL = [
    [0.00, BG_MAIN], [0.04, '#1B5E20'], [0.15, '#2E7D32'],
    [0.40, '#00C853'], [1.00, '#00FF88'],
]

# ─────────────────────────────────────────────────────────────────────────────
# PLAYER STATS METADATA
# ─────────────────────────────────────────────────────────────────────────────
RADAR_COLS = {
    'POR': ['paradas_p90','porcentaje_paradas_p90','goles_evitados_p90',
            'goles_recibidos_p90','porterias_cero_p90'],
    'DEF': ['intercepciones_p90','entradas_p90','despejes_p90',
            'duelos_tierra_ganados_p90','pases_precisos_p90'],
    'MED': ['pases_precisos_p90','pases_largos_p90','recuperaciones_campo_rival_p90',
            'chances_creadas_p90','asistencias_p90'],
    'DEL': ['goles_p90','xG_p90','tiros_p90','tiros_a_puerta_p90','grandes_chances_p90'],
}

STAT_META = {
    'minutos_stats':                  ('⏱️', 'Minutos'),
    'partidos_stats':                 ('📅', 'Partidos'),
    'goles':                          ('⚽', 'Goles'),
    'asistencias':                    ('🔑', 'Asistencias'),
    'rating':                         ('⭐', 'Rating'),
    'goles_p90':                      ('⚽', 'Goles / 90'),
    'xG_p90':                         ('📊', 'xG / 90'),
    'tiros_p90':                      ('🎯', 'Tiros / 90'),
    'asistencias_p90':                ('🔑', 'Asis / 90'),
    'pases_precisos_p90':             ('🎽', 'Pases / 90'),
    'chances_creadas_p90':            ('💡', 'Chances / 90'),
    'intercepciones_p90':             ('🛡️', 'Intercep / 90'),
    'entradas_p90':                   ('🦵', 'Entradas / 90'),
    'duelos_tierra_ganados_p90':      ('💪', 'Duelos / 90'),
    'recuperaciones_campo_rival_p90': ('🔄', 'Recuper / 90'),
    'paradas_p90':                    ('🧤', 'Paradas / 90'),
    'porcentaje_paradas_p90':         ('📈', '% Paradas'),
    'goles_evitados_p90':             ('🚫', 'Evitados / 90'),
    'porterias_cero_p90':             ('🏆', 'Porterías 0 / 90'),
}

def pos_group(pos):
    p = str(pos).upper()
    if any(x in p for x in ['GK','POR','GOALKEEPER']): return 'POR'
    if any(x in p for x in ['CB','LB','RB','WB','DEF','BACK','CENTRAL']): return 'DEF'
    if any(x in p for x in ['CM','DM','AM','MID','MF','MIDFIELDER']): return 'MED'
    return 'DEL'

# ─────────────────────────────────────────────────────────────────────────────
# MODELO POISSON
# ─────────────────────────────────────────────────────────────────────────────
def load_poisson_model():
    gf = defaultdict(float); ga = defaultdict(float)
    mh = defaultdict(float); ma = defaultdict(float)
    for fpath in sorted(glob.glob(str(HIST_DIR / '*.json'))):
        stem  = Path(fpath).stem.replace('historico_','')
        parts = stem.split('_-_', 1)
        if len(parts) != 2: continue
        tkey = f"{parts[0].replace('_','/')} - {parts[1].replace('_',' ').title()}"
        w = TORNEO_WEIGHTS.get(tkey)
        if not w: continue
        for p in json.load(open(fpath, encoding='utf-8')).get('partidos',[]):
            if not p.get('terminado'): continue
            lo, vi = norm(p['local']), norm(p['visitante'])
            gl, gv = int(p.get('goles_local',0) or 0), int(p.get('goles_visit',0) or 0)
            gf[lo]+=w*gl; ga[lo]+=w*gv; mh[lo]+=w
            gf[vi]+=w*gv; ga[vi]+=w*gl; ma[vi]+=w
    mu  = sum(gf.values()) / (sum(mh.values())+sum(ma.values()) or 1)
    att = {}; defe = {}
    for t in gf:
        m = mh.get(t,0)+ma.get(t,0)
        att[t]  = (gf[t]/m)/mu if m else 1.0
        defe[t] = (ga[t]/m)/mu if m else 1.0
    return att, defe, mu, 1.15

def predict_match(local, visitante, att, defe, mu, home_adv, max_g=6):
    lo,vi = norm(local), norm(visitante)
    ll = att.get(lo,1.)*defe.get(vi,1.)*mu*home_adv
    lv = att.get(vi,1.)*defe.get(lo,1.)*mu
    probs = np.array([[poisson.pmf(i,ll)*poisson.pmf(j,lv)
                       for j in range(max_g+1)] for i in range(max_g+1)])
    return (float(np.sum(np.tril(probs,-1))), float(np.trace(probs)),
            float(np.sum(np.triu(probs,1))), probs, ll, lv)

# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def run_montecarlo(att, defe, mu, home_adv, n_sim=5000):
    data   = json.load(open(HIST_DIR/'historico_clausura_2026.json', encoding='utf-8'))
    tabla  = {norm(r['equipo']): {'pts':int(r.get('pts',0)),'gf':int(r.get('gf',0)),
                                   'gc':int(r.get('gc',0))} for r in data.get('tabla',[])}
    pend   = [(norm(p['local']),norm(p['visitante'])) for p in data.get('partidos',[])
              if not p.get('terminado')]
    teams  = sorted(tabla.keys()); n = len(teams)
    pcnt   = np.zeros((n,n), dtype=np.int32)
    for _ in range(n_sim):
        pts = {t:tabla[t]['pts'] for t in teams}
        gf  = {t:tabla[t]['gf']  for t in teams}
        gc  = {t:tabla[t]['gc']  for t in teams}
        for lo,vi in pend:
            if lo not in pts or vi not in pts: continue
            gl = np.random.poisson(att.get(lo,1.)*defe.get(vi,1.)*mu*home_adv)
            gv = np.random.poisson(att.get(vi,1.)*defe.get(lo,1.)*mu)
            gf[lo]+=gl; gc[lo]+=gv; gf[vi]+=gv; gc[vi]+=gl
            if gl>gv: pts[lo]+=3
            elif gl==gv: pts[lo]+=1; pts[vi]+=1
            else: pts[vi]+=3
        for pos,t in enumerate(sorted(teams,key=lambda t:(pts[t],gf[t]-gc[t],gf[t]),reverse=True)):
            pcnt[teams.index(t),pos]+=1
    return teams, pcnt/n_sim

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
    return df[df['minutos_stats'] >= 200].copy()

def load_j13():
    data = json.load(open(HIST_DIR/'historico_clausura_2026.json', encoding='utf-8'))
    return [(norm(p['local']),norm(p['visitante']))
            for p in data.get('partidos',[]) if p.get('jornada')=='13']

def load_historico():
    return json.load(open(HIST_DIR/'historico_clausura_2026.json', encoding='utf-8'))

# ─────────────────────────────────────────────────────────────────────────────
# PRE-CARGA
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando modelo Poisson...")
ATT, DEFE, MU, HOME_ADV = load_poisson_model()
print("Monte Carlo (5,000 escenarios)...")
MC_TEAMS, MC_PROBS = run_montecarlo(ATT, DEFE, MU, HOME_ADV, n_sim=5000)
print("ELO + Jugadores...")
ELO_DF = load_elo(); JUG_DF = load_players()
J13    = load_j13(); HIST   = load_historico()
print("Listo.\n")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def empty_dark_fig(h=400):
    return go.Figure(layout=dict(
        paper_bgcolor=BG_MAIN, plot_bgcolor=BG_SEC,
        height=h, margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    ))

def fig_score_heatmap(probs, local, visitante):
    max_g = probs.shape[0]-1
    z     = (probs*100).round(1)
    text  = [[f'{z[i,j]:.1f}%' for j in range(max_g+1)] for i in range(max_g+1)]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(max_g+1)), y=list(range(max_g+1)),
        text=text, texttemplate='%{text}', textfont=dict(size=9),
        colorscale=VERDE_QUETZAL, showscale=True, zmin=0, zmax=30,
        colorbar=dict(tickfont=dict(color=GRAY,size=9),
                      title=dict(text='%', font=dict(color=GRAY))),
        hovertemplate='<b>%{y}–%{x}</b>: %{z:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text='Distribución de marcadores',
                   font=dict(size=13,color=GRAY), x=0.5),
        xaxis=dict(title=dict(text=f'Goles {visitante}', font=dict(color=GRAY,size=11)),
                   tickfont=dict(color=WHITE), gridcolor='#1e2530'),
        yaxis=dict(title=dict(text=f'Goles {local}', font=dict(color=GRAY,size=11)),
                   tickfont=dict(color=WHITE), gridcolor='#1e2530'),
        height=365, margin=dict(l=60,r=20,t=50,b=55),
    )
    return fig

def fig_elo_ranking():
    latest = ELO_DF.sort_values('fecha').groupby('equipo').last().reset_index()
    latest = latest.sort_values('elo', ascending=True)
    colors = [TEAM_COLORS.get(t, RED) for t in latest['equipo']]
    imgs   = []
    for _,row in latest.iterrows():
        tid = TEAM_IDS.get(row['equipo'])
        if tid and (TEAMS_IMG/f'{tid}.png').exists():
            imgs.append(dict(source=f'/images/teams/{tid}.png',
                             xref='paper', yref='y',
                             x=-0.01, y=row['equipo'],
                             sizex=0.04, sizey=0.75,
                             xanchor='right', yanchor='middle', layer='above'))
    fig = go.Figure(go.Bar(
        x=latest['elo'], y=latest['equipo'], orientation='h',
        marker_color=colors, marker_line_width=0,
        text=latest['elo'].round(0).astype(int),
        textposition='outside', textfont=dict(size=11,color=WHITE),
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text='Ranking ELO — Clausura 2026',
                   font=dict(size=14,color=WHITE), x=0.5),
        xaxis=dict(title='Rating ELO', tickfont=dict(color=GRAY),
                   gridcolor='#1e2530', range=[1300,None]),
        yaxis=dict(tickfont=dict(size=11,color=WHITE)),
        height=620, margin=dict(l=160,r=80,t=50,b=40),
        images=imgs,
    )
    return fig

def fig_elo_history(teams):
    fig = go.Figure()
    for t in (teams or []):
        df_t = ELO_DF[ELO_DF['equipo']==t].sort_values('fecha')
        if df_t.empty: continue
        fig.add_trace(go.Scatter(
            x=df_t['fecha'], y=df_t['elo'], mode='lines', name=t,
            line=dict(color=TEAM_COLORS.get(t,GRAY), width=2.5,
                      shape='spline', smoothing=0.8),
        ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text='Evolución ELO Histórica',
                   font=dict(size=13,color=GRAY), x=0.5),
        xaxis=dict(tickfont=dict(color=GRAY), gridcolor='#1e2530'),
        yaxis=dict(tickfont=dict(color=GRAY), gridcolor='#1e2530',
                   title='ELO', range=[1300,None]),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER,
                    font=dict(color=WHITE,size=10)),
        height=350, margin=dict(l=60,r=20,t=50,b=40),
    )
    return fig

def fig_montecarlo():
    teams, probs = MC_TEAMS, MC_PROBS
    n     = len(teams)
    order = np.argsort(np.sum(probs * np.arange(1,n+1), axis=1))
    ts    = [teams[i] for i in order]
    ps    = probs[order,:]
    z     = (ps*100).round(1)
    text  = [[f'{z[i,j]:.0f}%' if z[i,j]>=3 else '' for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(i+1) for i in range(n)], y=ts,
        text=text, texttemplate='%{text}',
        textfont=dict(size=10, color=WHITE),
        colorscale=VERDE_QUETZAL, showscale=True, zmin=0, zmax=70,
        colorbar=dict(title=dict(text='%', font=dict(color=GRAY)),
                      tickfont=dict(color=GRAY)),
        hovertemplate='<b>%{y}</b><br>Pos %{x}: %{z:.1f}%<extra></extra>',
    ))
    # Zone highlights
    fig.add_shape(type='rect', x0=-0.5,x1=3.5, y0=-0.5,y1=n-0.5,
                  fillcolor='rgba(255,215,0,0.05)', line=dict(width=0), layer='below')
    fig.add_shape(type='rect', x0=3.5,x1=7.5, y0=-0.5,y1=n-0.5,
                  fillcolor='rgba(46,160,67,0.05)', line=dict(width=0), layer='below')
    # Divider lines
    fig.add_shape(type='line', x0=3.5,x1=3.5, y0=-0.5,y1=n-0.5,
                  line=dict(color=GOLD, width=2, dash='dot'))
    fig.add_shape(type='line', x0=7.5,x1=7.5, y0=-0.5,y1=n-0.5,
                  line=dict(color=GREEN, width=2, dash='dot'))
    # Zone labels (below chart, using annotation with yref paper)
    fig.add_annotation(x=1.5, y=-0.55, yref='paper', text='◆ LIGUILLA',
                       showarrow=False, font=dict(color=GOLD,size=10,family='Arial Black'))
    fig.add_annotation(x=5.5, y=-0.55, yref='paper', text='◆ REPECHAJE',
                       showarrow=False, font=dict(color=GREEN,size=10,family='Arial Black'))
    # Bold overlay for >20%
    bx,by,bt = [],[],[]
    for i,t in enumerate(ts):
        for j in range(n):
            if z[i,j] >= 20:
                bx.append(str(j+1)); by.append(t)
                bt.append(f'<b>{z[i,j]:.0f}%</b>')
    if bx:
        fig.add_trace(go.Scatter(x=bx, y=by, mode='text', text=bt,
                                  textfont=dict(size=14,color=WHITE,family='Arial Black'),
                                  hoverinfo='skip', showlegend=False))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text='Simulación Monte Carlo — Clausura 2026',
            font=dict(size=15,color=WHITE), x=0.5, y=0.98,
        ),
        xaxis=dict(
            # Axis title as annotation to avoid overlap with chart title
            title=dict(text='', standoff=0),
            tickfont=dict(color=WHITE,size=11),
            side='top', gridcolor='#1e2530',
        ),
        yaxis=dict(tickfont=dict(size=12,color=WHITE), autorange='reversed',
                   gridcolor='#1e2530'),
        height=720,
        margin=dict(l=150,r=40,t=70,b=60),
        annotations=list(fig.layout.annotations) + [
            dict(x=0.5, y=-0.06, xref='paper', yref='paper',
                 text='Posición final →',
                 showarrow=False,
                 font=dict(color=GRAY,size=11), xanchor='center'),
        ],
    )
    return fig

def fig_radar(row):
    pg   = pos_group(row.get('posicion','DEL'))
    cols = [c for c in RADAR_COLS.get(pg,RADAR_COLS['DEL']) if c in JUG_DF.columns]
    if not cols: return empty_dark_fig(420)
    same = JUG_DF[JUG_DF['posicion'].apply(pos_group)==pg]
    vals, lbls = [], []
    for c in cols:
        v   = float(row.get(c,0) or 0)
        cd  = same[c].dropna()
        pct = float((cd<=v).mean()*100) if len(cd)>0 else 50.
        vals.append(pct)
        lbls.append(c.replace('_p90','').replace('_',' ').upper())
    tc   = TEAM_COLORS.get(row.get('equipo',''), RED)
    vc   = vals+[vals[0]]; lc = lbls+[lbls[0]]
    fig  = go.Figure()
    # Guide circles at 25/50/75
    for v in [25,50,75]:
        fig.add_trace(go.Scatterpolar(
            r=[v]*len(lc), theta=lc, mode='lines',
            line=dict(color='#2d333b',width=0.8,dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))
    # Main trace — FILLED with team color
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=lc, fill='toself',
        fillcolor=_rgba(tc,0.45),
        line=dict(color=tc,width=2.5),
        name=row.get('nombre',''),
        hovertemplate='<b>%{theta}</b><br>Percentil: <b>%{r:.0f}</b><extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor=BG_MAIN,
        plot_bgcolor=BG_MAIN,
        template='plotly_dark',
        polar=dict(
            bgcolor=BG_SEC,
            radialaxis=dict(
                visible=True, range=[0,100],
                tickvals=[25,50,75], ticktext=['25','50','75'],
                tickfont=dict(color=GRAY,size=8),
                gridcolor='#2d333b', linecolor='#2d333b',
                tickcolor='#2d333b',
            ),
            angularaxis=dict(
                tickfont=dict(color=WHITE,size=10),
                gridcolor='#2d333b', linecolor='#2d333b',
            ),
        ),
        font=dict(color=WHITE),
        title=dict(
            text=f"<b>{row.get('nombre','')}</b> — Percentiles (grupo {pg})",
            font=dict(size=13,color=WHITE), x=0.5,
        ),
        showlegend=False,
        height=440,
        margin=dict(l=70,r=70,t=60,b=60),
    )
    return fig

def fig_comparativo(r1, r2):
    pg1 = pos_group(r1.get('posicion','DEL'))
    pg2 = pos_group(r2.get('posicion','DEL'))
    cols = [c for c in RADAR_COLS.get(pg1,RADAR_COLS['DEL'])
            if c in JUG_DF.columns and c in RADAR_COLS.get(pg2,RADAR_COLS['DEL'])]
    if not cols:
        cols = [c for c in RADAR_COLS['DEL'] if c in JUG_DF.columns]
    if not cols: return empty_dark_fig(480)
    pool = JUG_DF[JUG_DF['posicion'].apply(pos_group).isin([pg1,pg2])]
    lbls,p1s,p2s,v1s,v2s = [],[],[],[],[]
    for c in cols:
        cd = pool[c].dropna()
        v1 = float(r1.get(c,0) or 0); v2 = float(r2.get(c,0) or 0)
        p1 = float((cd<=v1).mean()*100) if len(cd) else 50.
        p2 = float((cd<=v2).mean()*100) if len(cd) else 50.
        lbls.append(c.replace('_p90','').replace('_',' ').upper())
        p1s.append(p1); p2s.append(p2); v1s.append(v1); v2s.append(v2)
    c1 = TEAM_COLORS.get(r1.get('equipo',''),RED)
    c2 = TEAM_COLORS.get(r2.get('equipo',''),'#4CAF50')
    n1,n2 = r1.get('nombre','J1'), r2.get('nombre','J2')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=lbls, x=[-v for v in p1s], orientation='h', name=n1,
        marker_color=c1, marker_line_width=0,
        text=[f'{v:.2f}' for v in v1s], textposition='inside',
        textfont=dict(color=WHITE,size=10),
        hovertemplate=f'<b>{n1}</b><br>%{{y}}: %{{customdata:.2f}}<extra></extra>',
        customdata=v1s,
    ))
    fig.add_trace(go.Bar(
        y=lbls, x=p2s, orientation='h', name=n2,
        marker_color=c2, marker_line_width=0,
        text=[f'{v:.2f}' for v in v2s], textposition='inside',
        textfont=dict(color=WHITE,size=10),
        hovertemplate=f'<b>{n2}</b><br>%{{y}}: %{{customdata:.2f}}<extra></extra>',
        customdata=v2s,
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        barmode='overlay',
        title=dict(text=f'<b>{n1}</b> vs <b>{n2}</b> — por percentil',
                   font=dict(size=14,color=WHITE), x=0.5),
        xaxis=dict(
            tickvals=[-100,-75,-50,-25,0,25,50,75,100],
            ticktext=['100','75','50','25','0','25','50','75','100'],
            tickfont=dict(color=GRAY), gridcolor='#1e2530',
            zeroline=True, zerolinecolor=BORDER, zerolinewidth=2,
            range=[-110,110],
        ),
        yaxis=dict(tickfont=dict(size=11,color=WHITE)),
        legend=dict(bgcolor=BG_CARD,bordercolor=BORDER,font=dict(color=WHITE),
                    x=0.5,xanchor='center',orientation='h',y=-0.12),
        height=480, margin=dict(l=160,r=30,t=60,b=70),
    )
    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HOME TAB BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _zone_class(pos):
    if pos <= 4:  return 'zone-top4'
    if pos <= 8:  return 'zone-top8'
    if pos >= 14: return 'zone-danger'
    return 'zone-none'

def build_standings():
    tabla = HIST.get('tabla', [])
    rows  = []
    for r in tabla:
        pos   = int(r.get('pos',0))
        team  = norm(r['equipo'])
        tid   = TEAM_IDS.get(team)
        tc    = TEAM_COLORS.get(team, GRAY)
        pts   = int(r.get('pts',0))
        pj    = int(r.get('pj',0))
        g,e,p = int(r.get('g',0)), int(r.get('e',0)), int(r.get('p',0))
        gf,gc = int(r.get('gf',0)), int(r.get('gc',0))
        dg    = gf - gc

        rows.append(html.Tr([
            html.Td(str(pos), style={'color': GOLD if pos<=4 else (GREEN if pos<=8 else (DANGER if pos>=14 else GRAY)),
                                      'fontWeight':'700','textAlign':'center'}),
            html.Td([
                html.Img(src=f'/images/teams/{tid}.png', height='18px',
                         style={'marginRight':'6px','verticalAlign':'middle',
                                'filter':'drop-shadow(0 1px 3px rgba(0,0,0,0.6))'}) if tid else None,
                html.Span(team, style={'color':tc,'fontWeight':'600','fontSize':'12px'}),
            ]),
            html.Td(str(pts), style={'fontWeight':'800','color':WHITE,'textAlign':'center','fontSize':'13px'}),
            html.Td(str(pj),  style={'color':GRAY,'textAlign':'center'}),
            html.Td(str(g),   style={'color':GREEN,'textAlign':'center','fontWeight':'600'}),
            html.Td(str(e),   style={'color':GRAY,'textAlign':'center'}),
            html.Td(str(p),   style={'color':DANGER,'textAlign':'center'}),
            html.Td(
                f'+{dg}' if dg>0 else str(dg),
                style={'color': GREEN if dg>0 else (DANGER if dg<0 else GRAY),
                       'textAlign':'center','fontWeight':'600'}
            ),
        ], className=_zone_class(pos),
           style={'backgroundColor': BG_CARD if pos%2==0 else BG_SEC}))

    return html.Div([
        html.Div('TABLA DE POSICIONES', className='section-label section-label-red'),
        html.Div([
            html.Div([
                html.Span('◆', style={'color':GOLD,'marginRight':'4px','fontSize':'8px'}),
                html.Span('Liguilla directa', style={'color':GOLD,'fontSize':'10px','marginRight':'12px'}),
                html.Span('◆', style={'color':GREEN,'marginRight':'4px','fontSize':'8px'}),
                html.Span('Repechaje', style={'color':GREEN,'fontSize':'10px','marginRight':'12px'}),
                html.Span('◆', style={'color':DANGER,'marginRight':'4px','fontSize':'8px'}),
                html.Span('Descenso', style={'color':DANGER,'fontSize':'10px'}),
            ], style={'marginBottom':'8px','display':'flex','alignItems':'center','flexWrap':'wrap','gap':'2px'}),
        ]),
        html.Table([
            html.Thead(html.Tr([
                html.Th('#'), html.Th('EQUIPO'),
                html.Th('PTS'), html.Th('PJ'),
                html.Th('G'), html.Th('E'), html.Th('P'), html.Th('DG'),
            ])),
            html.Tbody(rows),
        ], className='standings-table'),
    ])

def build_last_results():
    j12 = [p for p in HIST.get('partidos',[])
           if p.get('jornada')=='12' and p.get('terminado')]
    cards = []
    for p in j12:
        lo  = norm(p['local']); vi = norm(p['visitante'])
        gl  = int(p.get('goles_local',0) or 0)
        gv  = int(p.get('goles_visit',0) or 0)
        lo_c= TEAM_COLORS.get(lo,GRAY); vi_c=TEAM_COLORS.get(vi,GRAY)
        tid_lo = TEAM_IDS.get(lo); tid_vi = TEAM_IDS.get(vi)
        winner = 'local' if gl>gv else ('visit' if gv>gl else 'draw')
        cards.append(html.Div([
            html.Div([
                html.Img(src=f'/images/teams/{tid_lo}.png', className='result-shield') if tid_lo else None,
                html.Span(lo, style={'color': lo_c if winner=='local' else GRAY,
                                      'fontWeight':'700' if winner=='local' else '400'}),
            ], className='result-team'),
            html.Div(f'{gl} – {gv}', className='result-score',
                     style={'color': WHITE}),
            html.Div([
                html.Span(vi, style={'color': vi_c if winner=='visit' else GRAY,
                                      'fontWeight':'700' if winner=='visit' else '400'}),
                html.Img(src=f'/images/teams/{tid_vi}.png', className='result-shield') if tid_vi else None,
            ], className='result-team', style={'flexDirection':'row-reverse'}),
        ], className='result-card'))
    return html.Div([
        html.Div('ÚLTIMOS RESULTADOS — J12', className='section-label'),
        html.Div(cards),
    ])

def build_next_matches():
    j13 = [(norm(p['local']),norm(p['visitante']))
           for p in HIST.get('partidos',[]) if p.get('jornada')=='13']
    cards = []
    for lo,vi in j13:
        lo_c = TEAM_COLORS.get(lo,GRAY); vi_c=TEAM_COLORS.get(vi,GRAY)
        tid_lo = TEAM_IDS.get(lo); tid_vi = TEAM_IDS.get(vi)
        cards.append(html.Div([
            html.Div([
                html.Img(src=f'/images/teams/{tid_lo}.png', className='result-shield') if tid_lo else None,
                html.Span(lo, style={'color':lo_c,'fontWeight':'600'}),
            ], className='result-team'),
            html.Div('vs', className='match-vs-badge'),
            html.Div([
                html.Span(vi, style={'color':vi_c,'fontWeight':'600'}),
                html.Img(src=f'/images/teams/{tid_vi}.png', className='result-shield') if tid_vi else None,
            ], className='result-team', style={'flexDirection':'row-reverse'}),
        ], className='match-preview'))
    return html.Div([
        html.Div('PRÓXIMA JORNADA — J13', className='section-label'),
        html.Div(cards),
    ])

def build_elo_top5():
    latest = (ELO_DF.sort_values('fecha').groupby('equipo').last()
              .reset_index().sort_values('elo', ascending=False).head(5))
    rank_cls = ['elo-rank-1','elo-rank-2','elo-rank-3','','']
    cards = []
    for i,(_,row) in enumerate(latest.iterrows()):
        t   = row['equipo']
        tc  = TEAM_COLORS.get(t,RED)
        tid = TEAM_IDS.get(t)
        cards.append(html.Div([
            html.Div(str(i+1), className=f'elo-rank {rank_cls[i]}'),
            html.Img(src=f'/images/teams/{tid}.png', height='36px',
                     style={'filter':'drop-shadow(0 1px 4px rgba(0,0,0,0.6))'}) if tid else None,
            html.Div([
                html.Div(t, className='elo-team-name'),
                html.Div(f'{int(row["elo"])}', className='elo-value',
                         style={'color':tc}),
            ]),
        ], className='elo-mini-card'))
    return html.Div([
        html.Div('ELO TOP 5', className='section-label section-label-red'),
        html.Div(cards, style={'display':'flex','gap':'8px','flexWrap':'wrap'}),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE HEAVY FIGURES
# ─────────────────────────────────────────────────────────────────────────────
FIG_ELO_RANK = fig_elo_ranking()
FIG_MC       = fig_montecarlo()

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title='MauStats MX',
)

@app.server.route('/images/teams/<fn>')
def serve_team(fn): return send_from_directory(str(TEAMS_IMG), fn)

@app.server.route('/images/players/<fn>')
def serve_player(fn): return send_from_directory(str(PLAYERS_IMG), fn)

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
NAVBAR = html.Nav([
    # Left: Logo
    html.Div([
        html.Span('MAU',   className='mau-logo-part logo-mau'),
        html.Span('STATS', className='mau-logo-part logo-stats'),
        html.Span('_MX',   className='mau-logo-part logo-mx'),
    ], className='mau-logo'),

    # Center: breadcrumb
    html.Div([
        html.Span('Liga MX', style={'letterSpacing':'2px'}),
        html.Span(className='mau-nav-dot'),
        html.Span('Clausura 2026'),
    ], className='mau-nav-center'),

    # Right: badges
    html.Div([
        html.Span('J13 Próxima', className='mau-jornada-badge'),
        html.Span('ESTADÍSTICAS', className='mau-badge'),
    ], className='mau-nav-right'),
], className='mau-navbar')

# ─────────────────────────────────────────────────────────────────────────────
# TAB HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_TS  = {'padding':'12px 18px','fontSize':'11px','fontWeight':'700',
        'letterSpacing':'2px','textTransform':'uppercase',
        'border':'none','borderBottom':'2px solid transparent',
        'borderRadius':'0','backgroundColor':'transparent','color':GRAY}
_TSA = {**_TS, 'color':WHITE,'borderBottomColor':RED}

def dd_style():
    return {'backgroundColor':BG_CARD,'color':WHITE,
            'border':f'1px solid {BORDER}','borderRadius':'8px'}

def make_dd_match():
    opts = []
    for lo,vi in J13:
        tid_lo = TEAM_IDS.get(lo); tid_vi = TEAM_IDS.get(vi)
        lbl = html.Span([
            html.Img(src=f'/images/teams/{tid_lo}.png', height='18px',
                     style={'marginRight':'5px','verticalAlign':'middle'}) if tid_lo else None,
            html.Span(lo, style={'color':TEAM_COLORS.get(lo,WHITE),'fontWeight':'600'}),
            html.Span(' vs ', style={'color':GRAY,'margin':'0 5px'}),
            html.Img(src=f'/images/teams/{tid_vi}.png', height='18px',
                     style={'marginRight':'5px','verticalAlign':'middle'}) if tid_vi else None,
            html.Span(vi, style={'color':TEAM_COLORS.get(vi,WHITE),'fontWeight':'600'}),
        ])
        opts.append({'label':lbl,'value':f'{lo}||{vi}'})
    return dcc.Dropdown(id='dd-match', options=opts,
                        value=opts[0]['value'] if opts else None,
                        clearable=False, optionHeight=38, style=dd_style())

def prob_circle(pct, lbl, color, team='', src=None):
    return html.Div([
        html.Img(src=src, className='team-shield-top') if src
        else html.Div(style={'height':'56px'}),
        html.Div([
            html.Div(f'{pct:.1f}%', className='prob-value', style={'color':color}),
            html.Div(lbl, className='prob-label'),
        ], className='prob-circle',
           style={'borderColor':color,
                  'boxShadow':f'0 0 22px {_rgba(color,0.25)} inset',
                  'backgroundColor':_rgba(color,0.06)}),
        html.Div(team, className='prob-team-name', style={'color':color}),
    ], className='prob-wrapper')

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT TABS
# ─────────────────────────────────────────────────────────────────────────────

# ── HOME ──────────────────────────────────────────────────────────────────────
TAB_HOME = dbc.Tab(label='HOME', tab_id='tab-home',
    label_style=_TS, active_label_style=_TSA,
    children=dbc.Container([
        dbc.Row([
            # Left: standings
            dbc.Col(dbc.Card(dbc.CardBody(build_standings())), md=5),
            # Right: results + upcoming + elo
            dbc.Col([
                dbc.Card(dbc.CardBody(build_last_results()), className='mb-3'),
                dbc.Card(dbc.CardBody(build_next_matches()), className='mb-3'),
                dbc.Card(dbc.CardBody(build_elo_top5())),
            ], md=7),
        ], className='g-3 mt-3'),
    ], fluid=True, className='p-3'),
)

# ── JORNADA 13 ────────────────────────────────────────────────────────────────
TAB_J13 = dbc.Tab(label='JORNADA 13', tab_id='tab-j13',
    label_style=_TS, active_label_style=_TSA,
    children=dbc.Container([
        dbc.Row(dbc.Col(make_dd_match(), md=10, lg=8), justify='center',
                className='mb-4 mt-3'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div('PROBABILIDADES', className='section-label'),
                html.Div(id='prob-circles',
                         style={'display':'flex','justifyContent':'space-around',
                                'alignItems':'center','padding':'8px 0',
                                'flexWrap':'wrap','gap':'16px'}),
                html.Div(id='lambda-info', className='text-center mt-2',
                         style={'color':GRAY,'fontSize':'11px'}),
            ])), md=5),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(id='graph-heatmap', config={'displayModeBar':False})
            )), md=7),
        ], className='g-3'),
    ], fluid=True, className='p-3'),
)

# ── ELO ────────────────────────────────────────────────────────────────────────
TAB_ELO = dbc.Tab(label='RANKING ELO', tab_id='tab-elo',
    label_style=_TS, active_label_style=_TSA,
    children=dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=FIG_ELO_RANK, config={'displayModeBar':False})
            )), md=5),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.Div('COMPARAR EVOLUCIÓN', className='section-label'),
                    dcc.Dropdown(
                        id='dd-elo-teams',
                        options=[{'label':t,'value':t} for t in sorted(ELO_DF['equipo'].unique())],
                        value=['América','Chivas','Cruz Azul'],
                        multi=True, optionHeight=30, style=dd_style(),
                    ),
                ]), className='mb-3'),
                dbc.Card(dbc.CardBody(
                    dcc.Graph(id='graph-elo-hist', config={'displayModeBar':False})
                )),
            ], md=7),
        ], className='g-3 mt-3'),
    ], fluid=True, className='p-3'),
)

# ── SIMULACIÓN ────────────────────────────────────────────────────────────────
TAB_MC = dbc.Tab(label='SIMULACIÓN', tab_id='tab-mc',
    label_style=_TS, active_label_style=_TSA,
    children=dbc.Container([
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(figure=FIG_MC, config={'displayModeBar':False})
        ))), className='mt-3'),
        dbc.Row(dbc.Col(html.P([
            html.Span('5,000 escenarios · ', className='text-muted-sm'),
            html.Span('─── TOP 4 ', style={'color':GOLD,'fontSize':'12px'}),
            html.Span('Liguilla directa · ', className='text-muted-sm'),
            html.Span('─── TOP 8 ', style={'color':GREEN,'fontSize':'12px'}),
            html.Span('Repechaje · Modelo Poisson ponderado', className='text-muted-sm'),
        ], style={'textAlign':'center','marginTop':'8px'}))),
    ], fluid=True, className='p-3'),
)

# ── JUGADORES ─────────────────────────────────────────────────────────────────
TAB_PLAYERS = dbc.Tab(label='JUGADORES', tab_id='tab-players',
    label_style=_TS, active_label_style=_TSA,
    children=dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div('EQUIPO', className='section-label'),
                dcc.Dropdown(
                    id='dd-team-players',
                    options=[{'label':t,'value':t} for t in sorted(JUG_DF['equipo'].dropna().unique())],
                    value=sorted(JUG_DF['equipo'].dropna().unique())[0],
                    clearable=False, style=dd_style(),
                ),
            ], md=4),
            dbc.Col([
                html.Div('JUGADOR  (≥200 min)', className='section-label'),
                dcc.Dropdown(id='dd-player', options=[], value=None,
                             clearable=False, optionHeight=30, style=dd_style()),
            ], md=4),
        ], className='g-3 mt-3'),

        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody(
                    html.Div(id='player-profile', style={'textAlign':'center'})
                ), className='mb-3'),
                dbc.Card(dbc.CardBody(html.Div(id='player-stats-cards'))),
            ], md=4),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(id='graph-radar', figure=empty_dark_fig(440),
                          config={'displayModeBar':False})
            )), md=8),
        ], className='g-3 mt-3'),
    ], fluid=True, className='p-3'),
)

# ── COMPARATIVO ───────────────────────────────────────────────────────────────
_all_teams = sorted(JUG_DF['equipo'].dropna().unique())
TAB_COMP = dbc.Tab(label='COMPARATIVO', tab_id='tab-comp',
    label_style=_TS, active_label_style=_TSA,
    children=dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div('JUGADOR 1', className='section-label'),
                dcc.Dropdown(id='comp-team-1',
                             options=[{'label':t,'value':t} for t in _all_teams],
                             value=_all_teams[0], clearable=False,
                             style={**dd_style(),'marginBottom':'8px'}),
                dcc.Dropdown(id='comp-player-1', options=[], value=None,
                             clearable=False, style=dd_style()),
            ], md=4),
            dbc.Col(html.Div(html.Div('VS',className='vs-badge'),className='vs-divider'),
                    md=2, style={'display':'flex','alignItems':'flex-end','justifyContent':'center'}),
            dbc.Col([
                html.Div('JUGADOR 2', className='section-label'),
                dcc.Dropdown(id='comp-team-2',
                             options=[{'label':t,'value':t} for t in _all_teams],
                             value=_all_teams[1], clearable=False,
                             style={**dd_style(),'marginBottom':'8px'}),
                dcc.Dropdown(id='comp-player-2', options=[], value=None,
                             clearable=False, style=dd_style()),
            ], md=4),
        ], className='g-3 mt-3', justify='center'),

        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id='graph-comp', figure=empty_dark_fig(480),
                      config={'displayModeBar':False})
        ))), className='mt-3'),

        dbc.Row(dbc.Col(
            html.Div(id='comp-shields',
                     style={'display':'flex','justifyContent':'center',
                            'gap':'40px','marginTop':'12px'}),
        )),
    ], fluid=True, className='p-3'),
)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    NAVBAR,
    dbc.Tabs([TAB_HOME, TAB_J13, TAB_ELO, TAB_MC, TAB_PLAYERS, TAB_COMP],
             id='main-tabs', active_tab='tab-home',
             style={'backgroundColor':'#0a0e14',
                    'borderBottom':f'1px solid {BORDER}'}),
], style={'backgroundColor':BG_MAIN,'minHeight':'100vh'})

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output('prob-circles','children'),
    Output('graph-heatmap','figure'),
    Output('lambda-info','children'),
    Input('dd-match','value'),
)
def cb_j13(match_val):
    if not match_val: return [], empty_dark_fig(365), ''
    lo,vi = match_val.split('||')
    pl,pe,pv,probs,ll,lv = predict_match(lo,vi,ATT,DEFE,MU,HOME_ADV)
    clo = TEAM_COLORS.get(lo,RED); cvi=TEAM_COLORS.get(vi,'#aaa')
    circles = [
        prob_circle(pl*100,'LOCAL',    clo, lo, shield(lo) or None),
        prob_circle(pe*100,'EMPATE',   GRAY,'Empate',None),
        prob_circle(pv*100,'VISITANTE',cvi, vi, shield(vi) or None),
    ]
    info = f'λ {lo} = {ll:.2f} goles esperados  ·  λ {vi} = {lv:.2f} goles esperados'
    return circles, fig_score_heatmap(probs,lo,vi), info


@app.callback(Output('graph-elo-hist','figure'), Input('dd-elo-teams','value'))
def cb_elo_hist(teams): return fig_elo_history(teams)


@app.callback(
    Output('dd-player','options'), Output('dd-player','value'),
    Input('dd-team-players','value'),
)
def cb_player_dd(team):
    if not team: return [],None
    players = JUG_DF[JUG_DF['equipo']==team]['nombre'].dropna().sort_values().tolist()
    opts    = [{'label':p,'value':p} for p in players]
    return opts, (opts[0]['value'] if opts else None)


@app.callback(
    Output('graph-radar','figure'),
    Output('player-profile','children'),
    Output('player-stats-cards','children'),
    Input('dd-player','value'),
    Input('dd-team-players','value'),
)
def cb_player(player, team):
    if not player or not team: return empty_dark_fig(440), html.Div(), html.Div()
    rows = JUG_DF[(JUG_DF['equipo']==team)&(JUG_DF['nombre']==player)]
    if rows.empty: return empty_dark_fig(440), html.Div(), html.Div()
    row = rows.iloc[0]

    # Profile
    pid   = row.get('id')
    iurl  = player_img(pid) if pd.notna(pid) else None
    tc    = TEAM_COLORS.get(team, RED)
    tid   = TEAM_IDS.get(team)
    profile = html.Div([
        html.Div([
            html.Img(src=iurl, className='player-photo',
                     style={'borderColor':tc}) if iurl
            else html.Div('👤', className='player-photo-fallback',
                          style={'borderColor':tc}),
            html.Img(src=f'/images/teams/{tid}.png', height='26px',
                     style={'position':'absolute','bottom':'0','right':'0',
                            'filter':'drop-shadow(0 1px 3px rgba(0,0,0,0.8))'}) if tid else None,
        ], className='player-photo-wrap'),
        html.Div(player,       className='player-name-header mt-2'),
        html.Div(team,         className='player-team-header', style={'color':tc}),
        html.Span(str(row.get('posicion','')), className='badge-jornada mt-1',
                  style={'display':'inline-block','marginTop':'6px'}),
    ])

    # Stats cards — show key stats always visible
    key_stats = ['minutos_stats','partidos_stats','goles','asistencias','rating',
                 'goles_p90','xG_p90','tiros_p90','asistencias_p90','pases_precisos_p90']
    cards = []
    for f in key_stats:
        if f not in JUG_DF.columns: continue
        val = row.get(f)
        if pd.isna(val): continue
        meta = STAT_META.get(f,('📋', f.replace('_',' ').title()))
        disp = (f'{float(val):.2f}' if '_p90' in f else
                f'{float(val):.1f}' if f=='rating' else
                f'{int(float(val)):,}' if f in ('minutos_stats',) else
                str(int(float(val))) if float(val)==int(float(val)) else
                f'{float(val):.2f}')
        cards.append(html.Div([
            html.Div(meta[0], className='stat-icon'),
            html.Div([
                html.Div(meta[1], className='stat-label'),
                html.Div(disp,   className='stat-value', style={'color':tc}),
            ]),
        ], className='stat-card mb-2'))

    return fig_radar(row), profile, html.Div(cards)


@app.callback(
    Output('comp-player-1','options'), Output('comp-player-1','value'),
    Input('comp-team-1','value'),
)
def cb_cp1(t):
    if not t: return [],None
    ps = JUG_DF[JUG_DF['equipo']==t]['nombre'].dropna().sort_values().tolist()
    opts=[{'label':p,'value':p} for p in ps]
    return opts,(opts[0]['value'] if opts else None)

@app.callback(
    Output('comp-player-2','options'), Output('comp-player-2','value'),
    Input('comp-team-2','value'),
)
def cb_cp2(t):
    if not t: return [],None
    ps = JUG_DF[JUG_DF['equipo']==t]['nombre'].dropna().sort_values().tolist()
    opts=[{'label':p,'value':p} for p in ps]
    return opts,(opts[0]['value'] if opts else None)


@app.callback(
    Output('graph-comp','figure'),
    Output('comp-shields','children'),
    Input('comp-player-1','value'), Input('comp-team-1','value'),
    Input('comp-player-2','value'), Input('comp-team-2','value'),
)
def cb_comp(p1,t1,p2,t2):
    if not all([p1,t1,p2,t2]): return empty_dark_fig(480), html.Div()
    r1 = JUG_DF[(JUG_DF['equipo']==t1)&(JUG_DF['nombre']==p1)]
    r2 = JUG_DF[(JUG_DF['equipo']==t2)&(JUG_DF['nombre']==p2)]
    if r1.empty or r2.empty: return empty_dark_fig(480), html.Div()
    c1=TEAM_COLORS.get(t1,RED); c2=TEAM_COLORS.get(t2,'#4CAF50')
    id1=TEAM_IDS.get(t1); id2=TEAM_IDS.get(t2)
    shields = html.Div([
        html.Div([
            html.Img(src=f'/images/teams/{id1}.png', height='48px') if id1 else None,
            html.Div(p1,style={'color':c1,'fontWeight':'700','marginTop':'4px','fontSize':'12px'}),
        ], style={'textAlign':'center'}),
        html.Div(html.Span('vs',style={'color':GRAY,'fontSize':'16px','fontWeight':'700'}),
                 style={'display':'flex','alignItems':'center'}),
        html.Div([
            html.Img(src=f'/images/teams/{id2}.png', height='48px') if id2 else None,
            html.Div(p2,style={'color':c2,'fontWeight':'700','marginTop':'4px','fontSize':'12px'}),
        ], style={'textAlign':'center'}),
    ],style={'display':'flex','gap':'40px','justifyContent':'center','alignItems':'center'})
    return fig_comparativo(r1.iloc[0], r2.iloc[0]), shields


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
