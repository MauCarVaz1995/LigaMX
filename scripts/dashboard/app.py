#!/usr/bin/env python3
"""
MauStats MX — Dark Pro Dashboard v2
Sidebar layout · Poisson + Monte Carlo · ELO · Player analysis
"""
import json, glob, warnings, hashlib
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
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG    = '#0b0f14'
BG2   = '#101620'
BG3   = '#0f151e'
CARD  = 'rgba(15,21,30,0.95)'
BORDER= '#1e2a3a'
WHITE = '#e6edf3'
MUTED = '#6e7681'
GRAY  = '#8b949e'
RED   = '#D5001C'
GREEN = '#2ea043'
GOLD  = '#FFD700'
DANGER= '#f85149'

TEAM_COLORS = {
    'Chivas':      '#CD1F2D',
    'Cruz Azul':   '#1E6AE1',
    'Toluca':      '#D5001C',
    'América':     '#FFD700',
    'Tigres':      '#F5A623',
    'Monterrey':   '#003DA5',
    'Pumas':       '#C8A84B',
    'Santos Laguna':'#2E8B57',
    'Pachuca':     '#A0B8C8',
    'Atlas':       '#B22222',
    'León':        '#2D8C3C',
    'Necaxa':      '#D62828',
    'Tijuana':     '#C62828',
    'Querétaro':   '#1A7FCB',
    'FC Juárez':   '#4CAF50',
    'Mazatlán':    '#9B59B6',
    'San Luis':    '#D52B1E',
    'Puebla':      '#2563EB',
}

TEAM_IDS = {
    'Chivas':7807,'Cruz Azul':6578,'Toluca':6618,'Pumas':1946,
    'Pachuca':7848,'Atlas':6577,'Tigres':8561,'América':6576,
    'Monterrey':7849,'FC Juárez':649424,'Necaxa':1842,'León':1841,
    'Tijuana':162418,'Puebla':7847,'San Luis':6358,
    'Mazatlán':1170234,'Querétaro':1943,'Santos Laguna':7857,
}

# ─────────────────────────────────────────────────────────────────────────────
# TORNEO WEIGHTS  (most recent = highest weight)
# ─────────────────────────────────────────────────────────────────────────────
TORNEO_W = {
    '2025/2026 - Clausura': 4,
    '2025/2026 - Apertura': 3,
    '2024/2025 - Clausura': 2,
    '2024/2025 - Apertura': 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# TEAM NAME ALIASES
# ─────────────────────────────────────────────────────────────────────────────
_AL = {
    'cf america':'América','america':'América','chivas':'Chivas','guadalajara':'Chivas',
    'cruz azul':'Cruz Azul','tigres':'Tigres','tigres uanl':'Tigres',
    'monterrey':'Monterrey','cf monterrey':'Monterrey','pumas':'Pumas','pumas unam':'Pumas',
    'toluca':'Toluca','santos laguna':'Santos Laguna','pachuca':'Pachuca','atlas':'Atlas',
    'león':'León','leon':'León','necaxa':'Necaxa','tijuana':'Tijuana',
    'querétaro':'Querétaro','queretaro':'Querétaro','queretaro fc':'Querétaro',
    'fc juárez':'FC Juárez','fc juarez':'FC Juárez','fc juarez fc':'FC Juárez',
    'mazatlán':'Mazatlán','mazatlan fc':'Mazatlán','mazatlan':'Mazatlán',
    'atletico de san luis':'San Luis','san luis':'San Luis','puebla':'Puebla',
}
def norm(n): return _AL.get(str(n).lower().strip(), str(n).strip())
def shld(t): tid = TEAM_IDS.get(t); return f'/images/teams/{tid}.png' if tid else ''
def pimg(pid):
    if not pid or pd.isna(pid): return None
    p = PLAYERS_IMG / f'{int(pid)}.png'
    return f'/images/players/{int(pid)}.png' if p.exists() else None

def rgba(h, a):
    h = h.lstrip('#')
    r, g, b = int(h[:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

def safe_float(val, default=0.0):
    """Convert to float, returning default for None/NaN."""
    if val is None: return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY BASE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PBASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(color=WHITE, family='Roboto, Inter, Arial'),
    template='plotly_dark',
)
VERDE = [[0,BG],[0.04,'#1B5E20'],[0.15,'#2E7D32'],[0.40,'#00C853'],[1.0,'#00FF88']]

# ─────────────────────────────────────────────────────────────────────────────
# PLAYER POSITION → PITCH COORDINATES  (x=0..105, y=0..68)
# ─────────────────────────────────────────────────────────────────────────────
PITCH_XY = {
    'GK':(6,34),'CB':(19,34),'LCB':(18,22),'RCB':(18,46),
    'LB':(24,11),'RB':(24,57),'LWB':(30,8),'RWB':(30,60),
    'CDM':(38,34),'CM':(52,34),'LM':(48,12),'RM':(48,56),
    'CAM':(63,34),'LW':(72,11),'RW':(72,57),
    'SS':(79,34),'CF':(83,34),'ST':(88,34),'FW':(82,34),
}
def pos_xy(pos_str):
    primary = str(pos_str).split(',')[0].strip().upper()
    return PITCH_XY.get(primary, PITCH_XY['CM'])

def jitter_xy(pid, scale=2.2):
    h = int(hashlib.md5(str(pid).encode()).hexdigest()[:8], 16)
    return ((h>>0)&0xF)/15*scale - scale/2, ((h>>4)&0xF)/15*scale - scale/2

# ─────────────────────────────────────────────────────────────────────────────
# RADAR COLUMNS BY POSITION GROUP
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
    'minutos_stats':       ('⏱','Minutos'),
    'partidos_stats':      ('📅','Partidos'),
    'goles':               ('⚽','Goles'),
    'asistencias':         ('🔑','Asistencias'),
    'rating':              ('⭐','Rating'),
    'goles_p90':           ('⚽','Goles/90'),
    'xG_p90':              ('📊','xG/90'),
    'tiros_p90':           ('🎯','Tiros/90'),
    'asistencias_p90':     ('🔑','Asis/90'),
    'pases_precisos_p90':  ('🎽','Pases/90'),
    'chances_creadas_p90': ('💡','Chances/90'),
}

def pos_group(pos):
    p = str(pos).upper().split(',')[0].strip()
    if any(x in p for x in ['GK','POR']):           return 'POR'
    if any(x in p for x in ['CB','LB','RB','WB']):  return 'DEF'
    if any(x in p for x in ['CM','DM','AM','CDM','CAM','LM','RM']): return 'MED'
    return 'DEL'

# ─────────────────────────────────────────────────────────────────────────────
# POISSON MODEL  ← BUG FIXED: hyphens in year (2025-2026 → 2025/2026)
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    gf = defaultdict(float); ga = defaultdict(float)
    mh = defaultdict(float); ma = defaultdict(float)
    loaded = 0
    for fp in sorted(glob.glob(str(HIST_DIR / '*.json'))):
        stem  = Path(fp).stem.replace('historico_', '')
        parts = stem.split('_-_', 1)
        if len(parts) != 2:
            continue
        # FIX: replace HYPHENS first (2025-2026 → 2025/2026), then underscores
        year_part    = parts[0].replace('-', '/').replace('_', '/')
        torneo_part  = parts[1].replace('_', ' ').title()
        tkey = f"{year_part} - {torneo_part}"
        w = TORNEO_W.get(tkey)
        if not w:
            continue
        try:
            data = json.load(open(fp, encoding='utf-8'))
        except Exception:
            continue
        for p in data.get('partidos', []):
            if not p.get('terminado'):
                continue
            lo = norm(p['local']); vi = norm(p['visitante'])
            gl = int(p.get('goles_local', 0) or 0)
            gv = int(p.get('goles_visit', 0) or 0)
            gf[lo] += w*gl; ga[lo] += w*gv; mh[lo] += w
            gf[vi] += w*gv; ga[vi] += w*gl; ma[vi] += w
            loaded += 1
    total_m = sum(mh.values()) + sum(ma.values())
    mu = sum(gf.values()) / (total_m or 1)
    att = {}; defe = {}
    for t in gf:
        m = mh.get(t, 0) + ma.get(t, 0)
        att[t]  = (gf[t]/m) / mu if m else 1.0
        defe[t] = (ga[t]/m) / mu if m else 1.0
    print(f"  → Modelo cargado: {loaded} partidos, mu={mu:.3f}")
    return att, defe, mu, 1.15   # home advantage factor

def predict(lo, vi, att, defe, mu, ha, mg=6):
    ll = att.get(lo,1.) * defe.get(vi,1.) * mu * ha
    lv = att.get(vi,1.) * defe.get(lo,1.) * mu
    P  = np.array([[poisson.pmf(i,ll)*poisson.pmf(j,lv)
                    for j in range(mg+1)] for i in range(mg+1)])
    return (float(np.sum(np.tril(P,-1))), float(np.trace(P)),
            float(np.sum(np.triu(P,1))), P, ll, lv)

def montecarlo(att, defe, mu, ha, n=5000):
    d  = json.load(open(HIST_DIR/'historico_clausura_2026.json', encoding='utf-8'))
    tb = {norm(r['equipo']): {'pts':int(r.get('pts',0)),
                               'gf':int(r.get('gf',0)),
                               'gc':int(r.get('gc',0))}
          for r in d.get('tabla', [])}
    pend = [(norm(p['local']), norm(p['visitante']))
            for p in d.get('partidos', []) if not p.get('terminado')]
    ts = sorted(tb.keys()); N = len(ts)
    pc = np.zeros((N, N), dtype=np.int32)
    for _ in range(n):
        pts = {t: tb[t]['pts'] for t in ts}
        gf  = {t: tb[t]['gf']  for t in ts}
        gc  = {t: tb[t]['gc']  for t in ts}
        for lo, vi in pend:
            if lo not in pts or vi not in pts: continue
            ll = att.get(lo,1.) * defe.get(vi,1.) * mu * ha
            lv = att.get(vi,1.) * defe.get(lo,1.) * mu
            gl = np.random.poisson(max(ll, 0.01))
            gv = np.random.poisson(max(lv, 0.01))
            gf[lo]+=gl; gc[lo]+=gv; gf[vi]+=gv; gc[vi]+=gl
            if   gl > gv: pts[lo] += 3
            elif gl < gv: pts[vi] += 3
            else:         pts[lo] += 1; pts[vi] += 1
        for pos, t in enumerate(sorted(ts, key=lambda t:(pts[t], gf[t]-gc[t], gf[t]), reverse=True)):
            pc[ts.index(t), pos] += 1
    return ts, pc/n

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA AT STARTUP
# ─────────────────────────────────────────────────────────────────────────────
def load_elo():
    df = pd.read_csv(ELO_CSV, parse_dates=['fecha'])
    df['equipo'] = df['equipo'].apply(norm)
    return df

def load_jug():
    df = pd.read_csv(JUG_CSV)
    df.columns = df.columns.str.strip('\ufeff')
    df['equipo'] = df['equipo'].apply(norm)
    return df[df['minutos_stats'] >= 200].copy()

def load_hist():
    return json.load(open(HIST_DIR/'historico_clausura_2026.json', encoding='utf-8'))

def load_j13(hist):
    return [(norm(p['local']), norm(p['visitante']))
            for p in hist.get('partidos', []) if p.get('jornada') == '13']

print("Cargando modelo Poisson…"); ATT, DEFE, MU, HA = load_model()
print("Monte Carlo (5 000 sim)…");  MC_T, MC_P = montecarlo(ATT, DEFE, MU, HA)
print("ELO + jugadores…");          ELO = load_elo(); JUG = load_jug()
HIST = load_hist();                  J13 = load_j13(HIST)
print("Listo.\n")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def empty_fig(h=400):
    return go.Figure(layout=dict(
        paper_bgcolor=BG, plot_bgcolor=BG2, height=h,
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False)))

def fig_heatmap(probs, lo, vi):
    mg  = probs.shape[0] - 1
    z   = (probs * 100).round(1)
    txt = [[f'{z[i,j]:.1f}%' for j in range(mg+1)] for i in range(mg+1)]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(mg+1)), y=list(range(mg+1)),
        text=txt, texttemplate='%{text}', textfont=dict(size=9),
        colorscale=VERDE, showscale=True, zmin=0, zmax=30,
        colorbar=dict(tickfont=dict(color=MUTED,size=9),
                      title=dict(text='%',font=dict(color=MUTED))),
        hovertemplate='<b>%{y}–%{x}</b>: %{z:.1f}%<extra></extra>'))
    fig.update_layout(**PBASE,
        title=dict(text='Distribución de marcadores',font=dict(size=12,color=MUTED),x=0.5),
        xaxis=dict(title=dict(text=f'Goles {vi}',font=dict(color=MUTED,size=11)),
                   tickfont=dict(color=WHITE), gridcolor='#1e2530'),
        yaxis=dict(title=dict(text=f'Goles {lo}',font=dict(color=MUTED,size=11)),
                   tickfont=dict(color=WHITE), gridcolor='#1e2530'),
        height=340, margin=dict(l=55,r=20,t=45,b=50))
    return fig

def fig_elo_rank():
    lat  = ELO.sort_values('fecha').groupby('equipo').last().reset_index().sort_values('elo')
    cols = [TEAM_COLORS.get(t, RED) for t in lat['equipo']]
    imgs = [dict(source=f'/images/teams/{TEAM_IDS[t]}.png',
                 xref='paper', yref='y', x=-0.01, y=t,
                 sizex=0.04, sizey=0.75, xanchor='right', yanchor='middle', layer='above')
            for t in lat['equipo']
            if t in TEAM_IDS and (TEAMS_IMG/f'{TEAM_IDS[t]}.png').exists()]
    fig = go.Figure(go.Bar(
        x=lat['elo'], y=lat['equipo'], orientation='h',
        marker_color=cols, marker_line_width=0,
        text=lat['elo'].round(0).astype(int),
        textposition='outside', textfont=dict(size=11,color=WHITE)))
    fig.update_layout(**PBASE,
        title=dict(text='Ranking ELO',font=dict(size=14,color=WHITE),x=0.5),
        xaxis=dict(tickfont=dict(color=MUTED), gridcolor='#1e2530', range=[1300,None]),
        yaxis=dict(tickfont=dict(size=11,color=WHITE)),
        height=620, margin=dict(l=155,r=80,t=40,b=40), images=imgs)
    return fig

def fig_elo_hist(teams):
    fig = go.Figure()
    for t in (teams or []):
        d = ELO[ELO['equipo']==t].sort_values('fecha')
        if d.empty: continue
        fig.add_trace(go.Scatter(
            x=d['fecha'], y=d['elo'], mode='lines', name=t,
            line=dict(color=TEAM_COLORS.get(t,GRAY),width=2.5,shape='spline',smoothing=0.8)))
    fig.update_layout(**PBASE,
        title=dict(text='Evolución ELO',font=dict(size=13,color=MUTED),x=0.5),
        xaxis=dict(tickfont=dict(color=MUTED), gridcolor='#1e2530'),
        yaxis=dict(tickfont=dict(color=MUTED), gridcolor='#1e2530',
                   title='ELO', range=[1300,None]),
        legend=dict(bgcolor=BG3,bordercolor=BORDER,font=dict(color=WHITE,size=10)),
        height=330, margin=dict(l=55,r=20,t=45,b=40))
    return fig

def fig_mc():
    ts, ps = MC_T, MC_P; n = len(ts)
    order  = np.argsort(np.sum(ps * np.arange(1,n+1), axis=1))
    tso    = [ts[i] for i in order]; pso = ps[order,:]
    z   = (pso * 100).round(1)
    txt = [[f'{z[i,j]:.0f}%' if z[i,j] >= 3 else ''
            for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(i+1) for i in range(n)], y=tso,
        text=txt, texttemplate='%{text}', textfont=dict(size=10,color=WHITE),
        colorscale=VERDE, showscale=True, zmin=0, zmax=70,
        colorbar=dict(title=dict(text='%',font=dict(color=MUTED)),
                      tickfont=dict(color=MUTED)),
        hovertemplate='<b>%{y}</b><br>Pos %{x}: %{z:.1f}%<extra></extra>'))
    # Zone backgrounds
    fig.add_shape(type='rect',x0=-0.5,x1=3.5,y0=-0.5,y1=n-0.5,
                  fillcolor='rgba(255,215,0,0.05)',line=dict(width=0),layer='below')
    fig.add_shape(type='rect',x0=3.5,x1=7.5,y0=-0.5,y1=n-0.5,
                  fillcolor='rgba(46,160,67,0.05)',line=dict(width=0),layer='below')
    fig.add_shape(type='line',x0=3.5,x1=3.5,y0=-0.5,y1=n-0.5,
                  line=dict(color=GOLD,width=2,dash='dot'))
    fig.add_shape(type='line',x0=7.5,x1=7.5,y0=-0.5,y1=n-0.5,
                  line=dict(color=GREEN,width=2,dash='dot'))
    # Bold overlay ≥20%
    bx,by,bt = [],[],[]
    for i,t in enumerate(tso):
        for j in range(n):
            if z[i,j] >= 20:
                bx.append(str(j+1)); by.append(t); bt.append(f'<b>{z[i,j]:.0f}%</b>')
    if bx:
        fig.add_trace(go.Scatter(x=bx,y=by,mode='text',text=bt,
            textfont=dict(size=14,color=WHITE,family='Arial Black'),
            hoverinfo='skip',showlegend=False))
    # Annotations
    fig.add_annotation(x=1.5,y=-0.06,xref='x',yref='paper',
        text='◆ LIGUILLA',showarrow=False,
        font=dict(color=GOLD,size=10,family='Arial Black'))
    fig.add_annotation(x=5.5,y=-0.06,xref='x',yref='paper',
        text='◆ REPECHAJE',showarrow=False,
        font=dict(color=GREEN,size=10,family='Arial Black'))
    fig.update_layout(**PBASE,
        title=dict(text='Simulación Monte Carlo — Clausura 2026',
                   font=dict(size=15,color=WHITE),x=0.5,y=0.98),
        xaxis=dict(title='',tickfont=dict(color=WHITE,size=11),
                   side='top',gridcolor='#1e2530'),
        yaxis=dict(tickfont=dict(size=12,color=WHITE),
                   autorange='reversed',gridcolor='#1e2530'),
        height=710, margin=dict(l=150,r=40,t=65,b=55))
    return fig

def fig_pitch(team=None, highlight_player=None):
    LC = 'rgba(255,255,255,0.85)'; LW = 1.8
    shapes = [
        dict(type='rect',x0=0,y0=0,x1=105,y1=68,fillcolor='#1a5c1a',line_color=LC,line_width=LW),
        dict(type='rect',x0=0,y0=0,x1=52.5,y1=68,fillcolor='#1b6b1b',line_color='rgba(0,0,0,0)',line_width=0),
        dict(type='line',x0=52.5,y0=0,x1=52.5,y1=68,line_color=LC,line_width=LW),
        dict(type='circle',x0=43.35,y0=24.85,x1=61.65,y1=43.15,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='circle',x0=52.2,y0=33.7,x1=52.8,y1=34.3,fillcolor=LC,line_color=LC),
        dict(type='rect',x0=0,y0=13.85,x1=16.5,y1=54.15,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='rect',x0=88.5,y0=13.85,x1=105,y1=54.15,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='rect',x0=0,y0=24.84,x1=5.5,y1=43.16,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='rect',x0=99.5,y0=24.84,x1=105,y1=43.16,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='rect',x0=-2.4,y0=30.34,x1=0,y1=37.66,line_color=LC,line_width=LW,fillcolor='rgba(255,255,255,0.08)'),
        dict(type='rect',x0=105,y0=30.34,x1=107.4,y1=37.66,line_color=LC,line_width=LW,fillcolor='rgba(255,255,255,0.08)'),
        dict(type='circle',x0=10.7,y0=33.7,x1=11.3,y1=34.3,fillcolor=LC,line_color=LC),
        dict(type='circle',x0=93.7,y0=33.7,x1=94.3,y1=34.3,fillcolor=LC,line_color=LC),
        dict(type='circle',x0=1.85,y0=24.85,x1=20.15,y1=43.15,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='circle',x0=84.85,y0=24.85,x1=103.15,y1=43.15,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='circle',x0=-1,y0=-1,x1=1,y1=1,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='circle',x0=-1,y0=67,x1=1,y1=69,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='circle',x0=104,y0=-1,x1=106,y1=1,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='circle',x0=104,y0=67,x1=106,y1=69,line_color=LC,line_width=LW,fillcolor='rgba(0,0,0,0)'),
        dict(type='rect',x0=0,y0=0,x1=17.5,y1=68,fillcolor='rgba(0,0,0,0.04)',line_color='rgba(0,0,0,0)'),
        dict(type='rect',x0=35,y0=0,x1=52.5,y1=68,fillcolor='rgba(0,0,0,0.04)',line_color='rgba(0,0,0,0)'),
        dict(type='rect',x0=70,y0=0,x1=87.5,y1=68,fillcolor='rgba(0,0,0,0.04)',line_color='rgba(0,0,0,0)'),
    ]
    fig = go.Figure()
    fig.update_layout(shapes=shapes)

    if team:
        players = JUG[JUG['equipo'] == team]
        tc = TEAM_COLORS.get(team, RED)

        def rating_color(r):
            if pd.isna(r): return rgba(GRAY, 0.8)
            r = float(r)
            if r >= 7.5: return rgba('#00FF88', 0.9)
            if r >= 7.0: return rgba('#2ea043', 0.9)
            if r >= 6.5: return rgba('#F5A623', 0.9)
            return rgba('#f85149', 0.9)

        xs,ys,names,colors,sizes,hovers = [],[],[],[],[],[]
        for _, p in players.iterrows():
            bx, by = pos_xy(p['posicion'])
            jx, jy = jitter_xy(p.get('id', p['nombre']))
            xs.append(bx+jx); ys.append(by+jy)
            names.append(p['nombre'])
            r = p.get('rating'); colors.append(rating_color(r))
            mins = safe_float(p.get('minutos_stats', 0))
            sizes.append(8 + min(mins/90*0.7, 10))
            g    = safe_float(p.get('goles', 0))
            a    = safe_float(p.get('asistencias', 0))
            rat  = round(float(r),2) if pd.notna(r) else '—'
            pos_s = str(p.get('posicion','')).split(',')[0]
            hovers.append(f"<b>{p['nombre']}</b><br>{pos_s}<br>⭐ {rat} · ⚽ {int(g)} · 🔑 {int(a)}<br>⏱ {int(mins)} min")

        is_hl   = [p['nombre'] == highlight_player for _, p in players.iterrows()]
        hl_xs   = [x for x,h in zip(xs,is_hl) if h]
        hl_ys   = [y for y,h in zip(ys,is_hl) if h]
        hl_names= [n for n,h in zip(names,is_hl) if h]
        nl_xs   = [x for x,h in zip(xs,is_hl) if not h]
        nl_ys   = [y for y,h in zip(ys,is_hl) if not h]
        nl_col  = [c for c,h in zip(colors,is_hl) if not h]
        nl_sz   = [s for s,h in zip(sizes,is_hl) if not h]
        nl_hov  = [hv for hv,h in zip(hovers,is_hl) if not h]

        fig.add_trace(go.Scatter(
            x=nl_xs, y=nl_ys, mode='markers',
            marker=dict(color=nl_col,size=nl_sz,line=dict(color='rgba(0,0,0,0.5)',width=1),opacity=0.9),
            text=nl_hov, hovertemplate='%{text}<extra></extra>', showlegend=False))
        if hl_xs:
            fig.add_trace(go.Scatter(
                x=hl_xs, y=hl_ys, mode='markers+text',
                marker=dict(color=rgba(tc,1.0),size=18,symbol='star',
                            line=dict(color=WHITE,width=2)),
                text=hl_names, textposition='top center',
                textfont=dict(color=WHITE,size=10,family='Arial Black'),
                hovertemplate='<b>%{text}</b><extra></extra>', showlegend=False))
    else:
        fig.add_trace(go.Scatter(x=[52.5],y=[34],mode='text',
            text=['<b>Selecciona un equipo</b>'],
            textfont=dict(color='rgba(255,255,255,0.3)',size=14),
            hoverinfo='skip',showlegend=False))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        xaxis=dict(range=[-4,109],visible=False,fixedrange=True),
        yaxis=dict(range=[-2,70],visible=False,scaleanchor='x',scaleratio=1,fixedrange=True),
        height=420, margin=dict(l=0,r=0,t=0,b=0),
        showlegend=False, dragmode=False)
    return fig

def fig_radar(row):
    pg   = pos_group(row.get('posicion','DEL'))
    cols = [c for c in RADAR_COLS.get(pg, RADAR_COLS['DEL']) if c in JUG.columns]
    if not cols:
        # Fallback to any available p90 stats
        cols = [c for c in RADAR_COLS['DEL'] if c in JUG.columns]
    if not cols:
        return empty_fig(380)
    same = JUG[JUG['posicion'].apply(pos_group) == pg]
    vals, lbls = [], []
    for c in cols:
        v  = safe_float(row.get(c, 0))
        cd = same[c].dropna()
        vals.append(float((cd <= v).mean() * 100) if len(cd) else 50.)
        lbls.append(c.replace('_p90','').replace('_',' ').upper())
    tc  = TEAM_COLORS.get(row.get('equipo',''), RED)
    vc  = vals + [vals[0]]; lc = lbls + [lbls[0]]
    fig = go.Figure()
    for v in [25, 50, 75]:
        fig.add_trace(go.Scatterpolar(
            r=[v]*len(lc), theta=lc, mode='lines',
            line=dict(color='#2d333b',width=0.7,dash='dot'),
            showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=lc, fill='toself', fillcolor=rgba(tc,0.45),
        line=dict(color=tc,width=2.5), name=row.get('nombre',''),
        hovertemplate='<b>%{theta}</b><br>Pct: <b>%{r:.0f}</b><extra></extra>'))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, template='plotly_dark',
        polar=dict(
            bgcolor=BG2,
            radialaxis=dict(visible=True,range=[0,100],tickvals=[25,50,75],
                tickfont=dict(color=MUTED,size=8),gridcolor='#2d333b',linecolor='#2d333b'),
            angularaxis=dict(tickfont=dict(color=WHITE,size=10),
                gridcolor='#2d333b',linecolor='#2d333b')),
        font=dict(color=WHITE),
        title=dict(text=f"<b>{row.get('nombre','')}</b> — percentiles ({pg})",
                   font=dict(size=12,color=WHITE),x=0.5),
        showlegend=False, height=380, margin=dict(l=60,r=60,t=55,b=55))
    return fig

def fig_comp(r1, r2):
    pg1  = pos_group(r1.get('posicion','DEL'))
    pg2  = pos_group(r2.get('posicion','DEL'))
    cols = [c for c in RADAR_COLS.get(pg1, RADAR_COLS['DEL'])
            if c in JUG.columns and c in RADAR_COLS.get(pg2, RADAR_COLS['DEL'])]
    if not cols:
        cols = [c for c in RADAR_COLS['DEL'] if c in JUG.columns]
    if not cols:
        return empty_fig(460)
    pool = JUG[JUG['posicion'].apply(pos_group).isin([pg1, pg2])]
    lbls, p1, p2, v1, v2 = [], [], [], [], []
    for c in cols:
        cd  = pool[c].dropna()
        # FIX: use safe_float to handle NaN properly (NaN is truthy → "nan or 0" = nan)
        vv1 = safe_float(r1.get(c))
        vv2 = safe_float(r2.get(c))
        p1.append(float((cd <= vv1).mean() * 100) if len(cd) else 50.)
        p2.append(float((cd <= vv2).mean() * 100) if len(cd) else 50.)
        v1.append(vv1); v2.append(vv2)
        lbls.append(c.replace('_p90','').replace('_',' ').upper())
    c1 = TEAM_COLORS.get(r1.get('equipo',''), RED)
    c2 = TEAM_COLORS.get(r2.get('equipo',''), '#4CAF50')
    n1 = r1.get('nombre','J1'); n2 = r2.get('nombre','J2')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=lbls, x=[-x for x in p1], orientation='h', name=n1,
        marker_color=c1, marker_line_width=0,
        text=[f'{x:.2f}' for x in v1], textposition='inside',
        textfont=dict(color=WHITE,size=10),
        customdata=v1,
        hovertemplate=f'<b>{n1}</b><br>%{{y}}: %{{customdata:.2f}}<extra></extra>'))
    fig.add_trace(go.Bar(
        y=lbls, x=p2, orientation='h', name=n2,
        marker_color=c2, marker_line_width=0,
        text=[f'{x:.2f}' for x in v2], textposition='inside',
        textfont=dict(color=WHITE,size=10),
        customdata=v2,
        hovertemplate=f'<b>{n2}</b><br>%{{y}}: %{{customdata:.2f}}<extra></extra>'))
    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    fig.update_layout(**PBASE, barmode='overlay',
        title=dict(text=f'<b>{n1}</b>  vs  <b>{n2}</b> — por percentil',
                   font=dict(size=13,color=WHITE),x=0.5),
        xaxis=dict(tickvals=[-100,-75,-50,-25,0,25,50,75,100],
            ticktext=['100','75','50','25','0','25','50','75','100'],
            tickfont=dict(color=MUTED),gridcolor='#1e2530',
            zeroline=True,zerolinecolor=BORDER,zerolinewidth=2,range=[-110,110]),
        yaxis=dict(tickfont=dict(size=11,color=WHITE)),
        legend=dict(bgcolor=BG3,bordercolor=BORDER,font=dict(color=WHITE),
                    x=0.5,xanchor='center',orientation='h',y=-0.12),
        height=460, margin=dict(l=155,r=30,t=55,b=70))
    return fig

# Pre-compute heavy figures
FIG_ELO_RANK = fig_elo_rank()
FIG_MC       = fig_mc()

# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def _zone(pos):
    if pos <= 4:  return 'zone-t4'
    if pos <= 8:  return 'zone-t8'
    if pos >= 14: return 'zone-down'
    return 'zone-none'

def build_standings():
    rows = []
    for r in HIST.get('tabla', []):
        pos = int(r.get('pos', 0)); t = norm(r['equipo'])
        tid = TEAM_IDS.get(t); tc = TEAM_COLORS.get(t, GRAY)
        pts = int(r.get('pts',0)); pj = int(r.get('pj',0))
        g,e,p = int(r.get('g',0)),int(r.get('e',0)),int(r.get('p',0))
        gf,gc = int(r.get('gf',0)),int(r.get('gc',0)); dg = gf - gc
        pc = GOLD if pos<=4 else (GREEN if pos<=8 else (DANGER if pos>=14 else MUTED))
        rows.append(html.Tr([
            html.Td(str(pos),style={'color':pc,'fontWeight':'700','textAlign':'center'}),
            html.Td(html.Div([
                html.Img(src=f'/images/teams/{tid}.png',height='18px',
                         style={'marginRight':'6px','verticalAlign':'middle',
                                'filter':'drop-shadow(0 1px 3px rgba(0,0,0,0.6))'}) if tid else None,
                html.Span(t,style={'color':tc,'fontWeight':'600'}),
            ],className='team-cell')),
            html.Td(str(pts),style={'fontWeight':'800','color':WHITE,'textAlign':'center'}),
            html.Td(str(pj), style={'color':MUTED,'textAlign':'center'}),
            html.Td(str(g),  style={'color':GREEN,'textAlign':'center','fontWeight':'600'}),
            html.Td(str(e),  style={'color':MUTED,'textAlign':'center'}),
            html.Td(str(p),  style={'color':DANGER,'textAlign':'center'}),
            html.Td(f'+{dg}' if dg>0 else str(dg),
                style={'color':GREEN if dg>0 else (DANGER if dg<0 else MUTED),
                       'textAlign':'center','fontWeight':'600'}),
        ],className=_zone(pos),
          style={'backgroundColor':'rgba(15,21,30,0.8)' if pos%2==0 else 'rgba(22,27,34,0.5)'}))
    return html.Div([
        html.Div('Tabla de posiciones', className='section-ttl'),
        html.Div([
            html.Span('◆ TOP 4 ', style={'color':GOLD,'fontSize':'10px'}),
            html.Span('Liguilla  ',style={'color':MUTED,'fontSize':'10px'}),
            html.Span('◆ TOP 8 ', style={'color':GREEN,'fontSize':'10px'}),
            html.Span('Repechaje', style={'color':MUTED,'fontSize':'10px'}),
        ],style={'marginBottom':'10px'}),
        html.Div(html.Table([
            html.Thead(html.Tr([
                html.Th('#'),html.Th('EQUIPO'),html.Th('PTS'),html.Th('PJ'),
                html.Th('G'),html.Th('E'),html.Th('P'),html.Th('DG'),
            ])),
            html.Tbody(rows),
        ],className='standings-tbl'),className='standings-wrap'),
    ])

def build_results():
    j12   = [p for p in HIST.get('partidos',[]) if p.get('jornada')=='12' and p.get('terminado')]
    cards = []
    for p in j12:
        lo  = norm(p['local']); vi = norm(p['visitante'])
        gl  = int(p.get('goles_local',0) or 0); gv = int(p.get('goles_visit',0) or 0)
        tid_lo = TEAM_IDS.get(lo); tid_vi = TEAM_IDS.get(vi)
        w   = 'local' if gl>gv else ('visit' if gv>gl else 'draw')
        cards.append(html.Div([
            html.Div([
                html.Img(src=f'/images/teams/{tid_lo}.png',className='r-shield') if tid_lo else None,
                html.Span(lo,style={'color':TEAM_COLORS.get(lo,MUTED),'fontWeight':'700' if w=='local' else '400'}),
            ],className='r-team'),
            html.Div(f'{gl}–{gv}',className='r-score'),
            html.Div([
                html.Span(vi,style={'color':TEAM_COLORS.get(vi,MUTED),'fontWeight':'700' if w=='visit' else '400'}),
                html.Img(src=f'/images/teams/{tid_vi}.png',className='r-shield') if tid_vi else None,
            ],className='r-team',style={'flexDirection':'row-reverse'}),
        ],className='result-card'))
    return html.Div([
        html.Div('Últimos resultados — J12', className='section-ttl'),
        html.Div(cards),
    ])

def build_upcoming():
    cards = []
    for lo, vi in J13:
        tid_lo = TEAM_IDS.get(lo); tid_vi = TEAM_IDS.get(vi)
        cards.append(html.Div([
            html.Div([
                html.Img(src=f'/images/teams/{tid_lo}.png',className='r-shield') if tid_lo else None,
                html.Span(lo,style={'color':TEAM_COLORS.get(lo,MUTED),'fontWeight':'600'}),
            ],className='r-team'),
            html.Div('vs', className='vs-pill'),
            html.Div([
                html.Span(vi,style={'color':TEAM_COLORS.get(vi,MUTED),'fontWeight':'600'}),
                html.Img(src=f'/images/teams/{tid_vi}.png',className='r-shield') if tid_vi else None,
            ],className='r-team',style={'flexDirection':'row-reverse'}),
        ],className='match-preview'))
    return html.Div([
        html.Div('Próxima jornada — J13', className='section-ttl'),
        html.Div(cards),
    ])

def build_kpis():
    lat      = ELO.sort_values('fecha').groupby('equipo').last().reset_index()
    top_elo  = lat.nlargest(1,'elo').iloc[0]
    tbl      = HIST.get('tabla', [])
    pts_leader = norm(tbl[0]['equipo']) if tbl else '—'
    pts_val    = tbl[0].get('pts', 0) if tbl else 0
    tc_pts  = TEAM_COLORS.get(pts_leader, WHITE)
    tc_elo  = TEAM_COLORS.get(top_elo['equipo'], WHITE)
    return html.Div([
        html.Div([
            html.Img(src=f'/images/teams/{TEAM_IDS.get(pts_leader,0)}.png',
                     height='32px',style={'marginRight':'8px'}) if pts_leader in TEAM_IDS else None,
            html.Div([
                html.Div(f'{pts_val} pts', className='kpi-value', style={'color':tc_pts}),
                html.Div('Líder',          className='kpi-label'),
                html.Div(pts_leader,       className='kpi-sub'),
            ]),
        ],className='kpi-card'),
        html.Div([
            html.Img(src=f'/images/teams/{TEAM_IDS.get(top_elo["equipo"],0)}.png',
                     height='32px',style={'marginRight':'8px'}) if top_elo['equipo'] in TEAM_IDS else None,
            html.Div([
                html.Div(f'{int(top_elo["elo"])}', className='kpi-value', style={'color':tc_elo}),
                html.Div('ELO #1',                  className='kpi-label'),
                html.Div(top_elo['equipo'],          className='kpi-sub'),
            ]),
        ],className='kpi-card'),
        html.Div([
            html.Div('📅', className='kpi-icon'),
            html.Div([
                html.Div('J13',      className='kpi-value', style={'color':RED}),
                html.Div('Jornada',  className='kpi-label'),
                html.Div('9 partidos',className='kpi-sub'),
            ]),
        ],className='kpi-card'),
        html.Div([
            html.Div('🏟️', className='kpi-icon'),
            html.Div([
                html.Div('18',      className='kpi-value', style={'color':GOLD}),
                html.Div('Equipos', className='kpi-label'),
                html.Div('Liga MX', className='kpi-sub'),
            ]),
        ],className='kpi-card'),
    ],className='kpi-row')

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT — DARKLY theme + Bootstrap Icons
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    title='MauStats MX',
)

@app.server.route('/images/teams/<fn>')
def srv_team(fn): return send_from_directory(str(TEAMS_IMG), fn)

@app.server.route('/images/players/<fn>')
def srv_player(fn): return send_from_directory(str(PLAYERS_IMG), fn)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
NAV_PAGES = [
    ('/',          'bi bi-house-fill',        'Home'),
    ('/jornada',   'bi bi-lightning-charge-fill','Jornada 13'),
    ('/elo',       'bi bi-bar-chart-fill',    'Ranking ELO'),
    ('/sim',       'bi bi-dice-5-fill',       'Simulación'),
    ('/jugadores', 'bi bi-person-fill',       'Jugadores'),
    ('/comp',      'bi bi-arrows-expand',     'Comparativo'),
]

def make_sidebar(pathname='/'):
    lat      = ELO.sort_values('fecha').groupby('equipo').last().reset_index().nlargest(3,'elo')
    elo_rows = [html.Div([
        html.Span(str(i+1), className='elo-sidebar-rank',
                  style={'color': GOLD if i==0 else (GRAY if i==1 else '#CD7F32')}),
        html.Img(src=f'/images/teams/{TEAM_IDS.get(r["equipo"],0)}.png', height='16px')
            if r['equipo'] in TEAM_IDS else None,
        html.Span(r['equipo'],
                  style={'color':TEAM_COLORS.get(r['equipo'],GRAY),'fontSize':'10px','fontWeight':'600'}),
        html.Span(f'{int(r["elo"])}', className='elo-sidebar-val',
                  style={'color':TEAM_COLORS.get(r['equipo'],GRAY)}),
    ],className='elo-sidebar-row') for i,(_,r) in enumerate(lat.iterrows())]

    nav_items = [html.A([
        html.I(className=f'{icon} nav-icon'),
        html.Span(label),
    ], href=href,
       className='nav-item' + (' active' if (pathname==href or (pathname=='/' and href=='/')) else ''))
    for href, icon, label in NAV_PAGES]

    return html.Div([
        # Logo
        html.Div([
            html.Div([
                html.Span('MAU',   className='logo-mau'),
                html.Span('STATS', className='logo-stats'),
                html.Span('_MX',   className='logo-mx'),
            ],className='logo-wordmark'),
            html.Div('ESTADÍSTICAS DE FÚTBOL', className='logo-sub'),
        ],className='sidebar-logo'),

        # Navigation
        html.Nav([
            html.Div('MENÚ', className='nav-section-label'),
            *nav_items,
        ],className='sidebar-nav'),

        # Footer
        html.Div([
            html.Div([
                html.Img(src='/images/teams/7807.png', height='20px')
                    if (TEAMS_IMG/'7807.png').exists() else None,
                html.Div([
                    html.Div('Liga MX',      className='sidebar-liga-name'),
                    html.Div('Clausura 2026',className='sidebar-liga-sub'),
                ]),
            ],className='sidebar-liga-badge'),
            html.Div('ELO TOP 3',
                style={'fontSize':'9px','letterSpacing':'2px','color':'#30363d',
                       'fontWeight':'700','marginBottom':'6px','textTransform':'uppercase'}),
            html.Div(elo_rows, className='sidebar-elo-mini'),
        ],className='sidebar-footer'),
    ],className='sidebar')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
def page_header(title, subtitle='', badge=''):
    return html.Div([
        html.Div([
            html.Div(title,    className='page-title'),
            html.Div(subtitle, className='page-subtitle'),
        ]),
        html.Div(badge, className='page-badge') if badge else html.Div(),
    ],className='page-header')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUTS
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    return html.Div([
        page_header('Dashboard','Liga MX · Clausura 2026','J13 Próxima'),
        html.Div([
            build_kpis(),
            dbc.Row([
                dbc.Col(html.Div(build_standings(),className='g-card',style={'padding':'16px'}),md=5),
                dbc.Col([
                    html.Div(build_results(),  className='g-card gap-col',style={'padding':'16px'}),
                    html.Div(build_upcoming(), className='g-card',         style={'padding':'16px'}),
                ],md=7),
            ],className='g-3'),
        ],className='page-content'),
    ])

def page_jornada():
    opts = []
    for lo, vi in J13:
        tid_lo = TEAM_IDS.get(lo); tid_vi = TEAM_IDS.get(vi)
        lbl = html.Span([
            html.Img(src=f'/images/teams/{tid_lo}.png',height='18px',
                     style={'marginRight':'5px','verticalAlign':'middle'}) if tid_lo else None,
            html.Span(lo,style={'color':TEAM_COLORS.get(lo,WHITE),'fontWeight':'600'}),
            html.Span(' vs ',style={'color':MUTED,'margin':'0 5px'}),
            html.Img(src=f'/images/teams/{tid_vi}.png',height='18px',
                     style={'marginRight':'5px','verticalAlign':'middle'}) if tid_vi else None,
            html.Span(vi,style={'color':TEAM_COLORS.get(vi,WHITE),'fontWeight':'600'}),
        ])
        opts.append({'label':lbl,'value':f'{lo}||{vi}'})
    return html.Div([
        page_header('Jornada 13','Predicciones Poisson ponderado','Clausura 2026'),
        html.Div([
            dbc.Row(dbc.Col(dcc.Dropdown(
                id='dd-match', options=opts,
                value=opts[0]['value'] if opts else None,
                clearable=False, optionHeight=38,
                style={'backgroundColor':BG3,'border':f'1px solid {BORDER}','borderRadius':'7px'}),
                md=8,lg=7),justify='center',className='mb-4'),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div('PROBABILIDADES',className='g-card-title',style={'padding':'12px 16px 0'}),
                    html.Div(id='prob-circles',className='prob-row',style={'padding':'12px 16px'}),
                    html.Div(id='lambda-info', className='lambda-row',style={'padding':'0 16px 12px'}),
                ],className='g-card'),md=5),
                dbc.Col(html.Div([
                    dcc.Graph(id='graph-heatmap',figure=empty_fig(340),
                              config={'displayModeBar':False}),
                ],className='g-card'),md=7),
            ],className='g-3'),
        ],className='page-content'),
    ])

def page_elo():
    elo_teams = sorted(ELO['equipo'].unique())
    return html.Div([
        page_header('Ranking ELO','Evolución histórica por torneo','Modelo ELO'),
        html.Div([
            dbc.Row([
                dbc.Col(html.Div(
                    dcc.Graph(figure=FIG_ELO_RANK,config={'displayModeBar':False}),
                    className='g-card'),md=5),
                dbc.Col([
                    html.Div([
                        html.Div('Comparar equipos',className='g-card-title',
                                 style={'padding':'12px 16px 0'}),
                        html.Div(dcc.Dropdown(
                            id='dd-elo-teams',
                            options=[{'label':t,'value':t} for t in elo_teams],
                            value=['América','Chivas','Cruz Azul'],multi=True,optionHeight=30,
                            style={'backgroundColor':BG3,'border':f'1px solid {BORDER}','borderRadius':'7px'}),
                        style={'padding':'10px 16px 14px'}),
                    ],className='g-card gap-col'),
                    html.Div(dcc.Graph(id='graph-elo-hist',config={'displayModeBar':False}),
                             className='g-card'),
                ],md=7),
            ],className='g-3'),
        ],className='page-content'),
    ])

def page_sim():
    return html.Div([
        page_header('Simulación Monte Carlo','5,000 escenarios Clausura 2026','Modelo Poisson'),
        html.Div([
            html.Div(dcc.Graph(figure=FIG_MC,config={'displayModeBar':False}),
                     className='g-card gap-col'),
            html.Div([
                html.Span('◆ TOP 4 ',  style={'color':GOLD,'fontWeight':'700'}),
                html.Span('Clasificación directa a Liguilla  ',style={'color':MUTED}),
                html.Span('◆ TOP 8 ',  style={'color':GREEN,'fontWeight':'700'}),
                html.Span('Acceso por Repechaje  ',style={'color':MUTED}),
                html.Span('Modelo Poisson ponderado por torneo  ',style={'color':MUTED}),
                html.Span('(Clausura×4, Apertura×3, …)',style={'color':'#30363d'}),
            ],style={'textAlign':'center','fontSize':'11px'}),
        ],className='page-content'),
    ])

def page_jugadores():
    teams = sorted(JUG['equipo'].dropna().unique())
    return html.Div([
        page_header('Jugadores','Análisis por posición · pitch view','≥200 min'),
        html.Div([
            dbc.Row([
                # Left: pitch + radar
                dbc.Col([
                    html.Div('PITCH VIEW', className='section-ttl'),
                    html.Div(dcc.Graph(id='pitch-graph',figure=fig_pitch(),
                                       config={'displayModeBar':False,'scrollZoom':False}),
                             className='g-card'),
                    html.Div(dcc.Graph(id='graph-radar',figure=empty_fig(380),
                                       config={'displayModeBar':False}),
                             className='g-card',style={'marginTop':'12px'}),
                ],md=8),
                # Right: controls + stats panel
                dbc.Col([
                    html.Div([
                        html.Div('EQUIPO',  className='section-ttl'),
                        dcc.Dropdown(id='dd-team-jug',
                            options=[{'label':t,'value':t} for t in teams],
                            value=teams[0],clearable=False,
                            style={'backgroundColor':BG3,'border':f'1px solid {BORDER}','borderRadius':'7px'}),
                        html.Div('JUGADOR', className='section-ttl',style={'marginTop':'14px'}),
                        dcc.Dropdown(id='dd-jug',options=[],value=None,clearable=False,optionHeight=28,
                            style={'backgroundColor':BG3,'border':f'1px solid {BORDER}','borderRadius':'7px'}),
                    ],className='g-card',style={'padding':'14px','marginBottom':'12px'}),
                    html.Div(id='player-panel',className='g-card',style={'padding':'14px'}),
                ],md=4),
            ],className='g-3'),
        ],className='page-content'),
    ])

def page_comp():
    teams = sorted(JUG['equipo'].dropna().unique())
    return html.Div([
        page_header('Comparativo 1v1','Comparación por percentil por posición','Head-to-head'),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div('JUGADOR 1', className='section-ttl'),
                    dcc.Dropdown(id='comp-team-1',
                        options=[{'label':t,'value':t} for t in teams],
                        value=teams[0],clearable=False,
                        style={'backgroundColor':BG3,'border':f'1px solid {BORDER}',
                               'borderRadius':'7px','marginBottom':'8px'}),
                    dcc.Dropdown(id='comp-p1',options=[],value=None,clearable=False,
                        style={'backgroundColor':BG3,'border':f'1px solid {BORDER}','borderRadius':'7px'}),
                ],md=4),
                dbc.Col(html.Div(html.Div('VS',className='vs-badge'),className='vs-wrap'),
                    md=2,style={'display':'flex','alignItems':'flex-end','justifyContent':'center'}),
                dbc.Col([
                    html.Div('JUGADOR 2', className='section-ttl'),
                    dcc.Dropdown(id='comp-team-2',
                        options=[{'label':t,'value':t} for t in teams],
                        value=teams[1],clearable=False,
                        style={'backgroundColor':BG3,'border':f'1px solid {BORDER}',
                               'borderRadius':'7px','marginBottom':'8px'}),
                    dcc.Dropdown(id='comp-p2',options=[],value=None,clearable=False,
                        style={'backgroundColor':BG3,'border':f'1px solid {BORDER}','borderRadius':'7px'}),
                ],md=4),
            ],className='g-3',justify='center',style={'marginBottom':'16px'}),
            html.Div(dcc.Graph(id='graph-comp',figure=empty_fig(460),
                               config={'displayModeBar':False}),className='g-card'),
            html.Div(id='comp-footer',style={'marginTop':'12px','textAlign':'center'}),
        ],className='page-content'),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id='sidebar-wrap'),
    html.Div(html.Div(id='page-content'), className='main-content'),
],style={'backgroundColor':BG,'minHeight':'100vh'})

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — ROUTING
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('page-content','children'),
    Output('sidebar-wrap','children'),
    Input('url','pathname'),
)
def route(pathname):
    pathname = pathname or '/'
    sidebar  = make_sidebar(pathname)
    pages = {
        '/':          page_home,
        '/jornada':   page_jornada,
        '/elo':       page_elo,
        '/sim':       page_sim,
        '/jugadores': page_jugadores,
        '/comp':      page_comp,
    }
    fn = pages.get(pathname, page_home)
    return fn(), sidebar

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — JORNADA 13
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('prob-circles','children'),
    Output('graph-heatmap','figure'),
    Output('lambda-info','children'),
    Input('dd-match','value'),
)
def cb_j13(val):
    if not val: return [], empty_fig(340), ''
    lo, vi = val.split('||')
    pl, pe, pv, P, ll, lv = predict(lo, vi, ATT, DEFE, MU, HA)
    clo = TEAM_COLORS.get(lo, RED); cvi = TEAM_COLORS.get(vi, '#aaa')

    def circ(pct, lbl, color, team, src):
        return html.Div([
            html.Img(src=src,className='team-badge-top') if src else html.Div(style={'height':'50px'}),
            html.Div([
                html.Div(f'{pct:.1f}%', className='prob-pct', style={'color':color}),
                html.Div(lbl, className='prob-lbl'),
            ],className='prob-circle',
              style={'borderColor':color,
                     'boxShadow':f'0 0 20px {rgba(color,0.2)} inset',
                     'backgroundColor':rgba(color,0.06)}),
            html.Div(team, className='prob-name', style={'color':color}),
        ],className='prob-wrapper')

    circles = [
        circ(pl*100, 'LOCAL',  clo,  lo,      shld(lo) or None),
        circ(pe*100, 'EMPATE', MUTED,'Empate', None),
        circ(pv*100, 'VISITA', cvi,  vi,       shld(vi) or None),
    ]
    info = f'λ {lo} = {ll:.2f}   ·   λ {vi} = {lv:.2f}  goles esperados'
    return circles, fig_heatmap(P, lo, vi), info

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — ELO
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output('graph-elo-hist','figure'), Input('dd-elo-teams','value'))
def cb_elo(teams): return fig_elo_hist(teams)

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — JUGADORES
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('dd-jug','options'),
    Output('dd-jug','value'),
    Input('dd-team-jug','value'),
)
def cb_jug_dd(team):
    if not team: return [], None
    ps   = JUG[JUG['equipo']==team]['nombre'].dropna().sort_values().tolist()
    opts = [{'label':p,'value':p} for p in ps]
    return opts, (opts[0]['value'] if opts else None)

@app.callback(
    Output('pitch-graph','figure'),
    Output('graph-radar','figure'),
    Output('player-panel','children'),
    Input('dd-jug','value'),
    Input('dd-team-jug','value'),
)
def cb_player(player, team):
    if not team:
        return fig_pitch(), empty_fig(380), html.Div()
    pitch = fig_pitch(team, player)
    if not player:
        return pitch, empty_fig(380), html.Div(
            html.Div('Selecciona un jugador',
                     style={'color':MUTED,'textAlign':'center','padding':'20px'}))
    rows = JUG[(JUG['equipo']==team) & (JUG['nombre']==player)]
    if rows.empty:
        return pitch, empty_fig(380), html.Div()
    row  = rows.iloc[0]
    pid  = row.get('id'); iurl = pimg(pid)
    tc   = TEAM_COLORS.get(team, RED); tid = TEAM_IDS.get(team)
    pos_primary = str(row.get('posicion','')).split(',')[0]

    profile = html.Div([
        html.Div([
            html.Img(src=iurl,className='player-img',style={'borderColor':tc}) if iurl
            else html.Div('👤',className='player-img-fallback',style={'borderColor':tc}),
            html.Img(src=f'/images/teams/{tid}.png',height='24px',
                     style={'position':'absolute','bottom':'0','right':'0',
                            'filter':'drop-shadow(0 1px 3px rgba(0,0,0,0.9))'}) if tid else None,
        ],style={'position':'relative','width':'84px','height':'84px','margin':'0 auto'}),
        html.Div(player,       className='player-name-lrg'),
        html.Div(team,         className='player-team-lrg',style={'color':tc}),
        html.Span(pos_primary, className='pos-chip'),
    ],className='player-photo-block',style={'textAlign':'center','marginBottom':'14px'})

    key_fields = ['minutos_stats','partidos_stats','goles','asistencias','rating',
                  'goles_p90','xG_p90','tiros_p90','asistencias_p90','pases_precisos_p90']
    pills = []
    for f in key_fields:
        if f not in JUG.columns: continue
        val = row.get(f)
        if pd.isna(val): continue
        meta = STAT_META.get(f, ('📋', f.replace('_',' ')))
        vf   = float(val)
        disp = (f'{vf:.2f}'  if '_p90' in f else
                f'{vf:.1f}'  if f=='rating'   else
                f'{int(vf):,}' if f=='minutos_stats' else
                str(int(vf)) if vf==int(vf) else f'{vf:.2f}')
        pills.append(html.Div([
            html.Div(meta[0], className='stat-icon'),
            html.Div([
                html.Div(disp,    className='stat-v',style={'color':tc}),
                html.Div(meta[1], className='stat-l'),
            ],className='stat-txt'),
        ],className='stat-pill'))

    return pitch, fig_radar(row), html.Div([profile, html.Div(pills,className='stat-grid')])

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — COMPARATIVO
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('comp-p1','options'), Output('comp-p1','value'),
    Input('comp-team-1','value'))
def cb_ct1(t):
    if not t: return [], None
    ps = JUG[JUG['equipo']==t]['nombre'].dropna().sort_values().tolist()
    o  = [{'label':p,'value':p} for p in ps]
    return o, (o[0]['value'] if o else None)

@app.callback(
    Output('comp-p2','options'), Output('comp-p2','value'),
    Input('comp-team-2','value'))
def cb_ct2(t):
    if not t: return [], None
    ps = JUG[JUG['equipo']==t]['nombre'].dropna().sort_values().tolist()
    o  = [{'label':p,'value':p} for p in ps]
    return o, (o[0]['value'] if o else None)

@app.callback(
    Output('graph-comp','figure'),
    Output('comp-footer','children'),
    Input('comp-p1','value'), Input('comp-team-1','value'),
    Input('comp-p2','value'), Input('comp-team-2','value'),
)
def cb_comp(p1, t1, p2, t2):
    if not all([p1,t1,p2,t2]): return empty_fig(460), html.Div()
    r1 = JUG[(JUG['equipo']==t1) & (JUG['nombre']==p1)]
    r2 = JUG[(JUG['equipo']==t2) & (JUG['nombre']==p2)]
    if r1.empty or r2.empty: return empty_fig(460), html.Div()
    c1  = TEAM_COLORS.get(t1, RED); c2 = TEAM_COLORS.get(t2, '#4CAF50')
    id1 = TEAM_IDS.get(t1);         id2 = TEAM_IDS.get(t2)
    footer = html.Div([
        html.Div([
            html.Img(src=f'/images/teams/{id1}.png',height='44px') if id1 else None,
            html.Div(p1,style={'color':c1,'fontWeight':'700','fontSize':'12px','marginTop':'4px'}),
        ],style={'textAlign':'center'}),
        html.Span('vs',style={'color':MUTED,'fontSize':'16px','fontWeight':'700','alignSelf':'center'}),
        html.Div([
            html.Img(src=f'/images/teams/{id2}.png',height='44px') if id2 else None,
            html.Div(p2,style={'color':c2,'fontWeight':'700','fontSize':'12px','marginTop':'4px'}),
        ],style={'textAlign':'center'}),
    ],style={'display':'flex','gap':'32px','justifyContent':'center'})
    return fig_comp(r1.iloc[0], r2.iloc[0]), footer

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
