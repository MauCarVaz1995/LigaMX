"""
pages/home.py — Página HOME de MauStats MX
Llamar: layout(hist, jug, elo, team_ids, team_colors)
"""
import pandas as pd
from dash import html
import dash_bootstrap_components as dbc

# ── Paleta ────────────────────────────────────────────────────────────────────
WHITE  = '#e6edf3'
MUTED  = '#6e7681'
GRAY   = '#8b949e'
RED    = '#D5001C'
GREEN  = '#2ea043'
GOLD   = '#FFD700'
DANGER = '#f85149'
BG_CARD= '#161b22'
BORDER = '#30363d'

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
def _norm(n): return _AL.get(str(n).lower().strip(), str(n).strip())


# ── Card base style ───────────────────────────────────────────────────────────
CARD_STYLE = {
    'background': BG_CARD,
    'border': f'1px solid {BORDER}',
    'borderRadius': '12px',
    'padding': '16px 20px',
    'height': '100%',
}

def _shield(team_id, size=25):
    """Escudo como html.Img apuntando a /assets/teams/{id}.png"""
    if not team_id:
        return html.Span()
    return html.Img(
        src=f'/assets/teams/{int(team_id)}.png',
        style={
            'width': f'{size}px',
            'height': f'{size}px',
            'objectFit': 'contain',
            'verticalAlign': 'middle',
            'filter': 'drop-shadow(0 1px 4px rgba(0,0,0,0.7))',
            'flexShrink': '0',
        }
    )


# ── KPI CARDS ─────────────────────────────────────────────────────────────────
def _kpi_card(big_value, label, sub_left=None, sub_right=None, accent=WHITE):
    """
    Card KPI genérica:
      big_value  — número/texto grande (Bebas Neue 32pt)
      label      — etiqueta superior gris 12pt Roboto
      sub_left   — elemento html izquierdo (escudo / nombre)
      sub_right  — elemento html derecho (nombre / métrica)
      accent     — color del número grande
    """
    return html.Div([
        # Label superior
        html.Div(label, style={
            'fontSize': '11px', 'fontWeight': '500',
            'letterSpacing': '1.5px', 'textTransform': 'uppercase',
            'color': MUTED, 'marginBottom': '8px',
            'fontFamily': 'Roboto, sans-serif',
        }),
        # Número grande
        html.Div(big_value, style={
            'fontSize': '36px', 'lineHeight': '1',
            'fontFamily': "'Bebas Neue', 'Arial Black', sans-serif",
            'color': accent, 'marginBottom': '6px',
            'letterSpacing': '2px',
        }),
        # Sub-info
        html.Div([
            sub_left  if sub_left  else html.Span(),
            sub_right if sub_right else html.Span(),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'gap': '8px',
            'marginTop': '2px',
        }),
    ], style=CARD_STYLE)


def _build_kpis(hist, jug, elo, team_ids, team_colors):
    # ── Líder ──
    tabla = hist.get('tabla', [])
    if tabla:
        lider_raw = tabla[0]
        lider     = _norm(lider_raw['equipo'])
        lider_pts = int(lider_raw.get('pts', 0))
        lider_tid = lider_raw.get('equipo_id') or team_ids.get(lider)
        lider_col = team_colors.get(lider, WHITE)
    else:
        lider, lider_pts, lider_tid, lider_col = '—', 0, None, WHITE

    # ── ELO #1 ──
    elo_top  = elo.sort_values('fecha').groupby('equipo').last().reset_index().nlargest(1, 'elo').iloc[0]
    elo_name = elo_top['equipo']
    elo_val  = int(elo_top['elo'])
    elo_tid  = team_ids.get(elo_name)
    elo_col  = team_colors.get(elo_name, WHITE)

    # ── Goleador ──
    try:
        gol_row = jug.dropna(subset=['goles']).nlargest(1, 'goles').iloc[0]
        gol_name = gol_row['nombre']
        gol_num  = int(gol_row['goles'])
        gol_team = gol_row['equipo']
        gol_tid  = gol_row.get('equipo_id') or team_ids.get(gol_team)
    except (IndexError, KeyError):
        gol_name, gol_num, gol_tid = '—', 0, None

    # ── Jornada ──
    jornadas = [int(p.get('jornada','0')) for p in hist.get('partidos', []) if p.get('terminado')]
    last_j   = max(jornadas) if jornadas else 12
    next_j   = last_j + 1

    kpis = [
        # Líder
        dbc.Col(_kpi_card(
            big_value = f'{lider_pts} PTS',
            label     = '🏆  Líder de tabla',
            sub_left  = _shield(lider_tid, 24),
            sub_right = html.Span(lider, style={
                'color': lider_col, 'fontWeight': '600',
                'fontSize': '13px', 'fontFamily': 'Roboto, sans-serif',
            }),
            accent = lider_col,
        ), xs=6, sm=6, md=3),

        # ELO #1
        dbc.Col(_kpi_card(
            big_value = str(elo_val),
            label     = '📊  Mejor ELO',
            sub_left  = _shield(elo_tid, 24),
            sub_right = html.Span(elo_name, style={
                'color': elo_col, 'fontWeight': '600',
                'fontSize': '13px', 'fontFamily': 'Roboto, sans-serif',
            }),
            accent = elo_col,
        ), xs=6, sm=6, md=3),

        # Goleador
        dbc.Col(_kpi_card(
            big_value = f'{gol_num} GOL{"ES" if gol_num != 1 else ""}',
            label     = '⚽  Goleador',
            sub_left  = _shield(gol_tid, 24),
            sub_right = html.Span(gol_name, style={
                'color': WHITE, 'fontWeight': '600',
                'fontSize': '12px', 'fontFamily': 'Roboto, sans-serif',
                'whiteSpace': 'nowrap', 'overflow': 'hidden',
                'textOverflow': 'ellipsis', 'maxWidth': '100px',
            }),
            accent = GREEN,
        ), xs=6, sm=6, md=3),

        # Jornada
        dbc.Col(_kpi_card(
            big_value = f'J{next_j}',
            label     = '📅  Próxima jornada',
            sub_left  = html.Span('Clausura 2026', style={
                'color': MUTED, 'fontSize': '12px',
                'fontFamily': 'Roboto, sans-serif',
            }),
            accent = RED,
        ), xs=6, sm=6, md=3),
    ]

    return dbc.Row(kpis, className='g-3 mb-3')


# ── STANDINGS TABLE ────────────────────────────────────────────────────────────
def _build_standings(hist, team_ids, team_colors):
    tabla = hist.get('tabla', [])

    header = html.Thead(html.Tr([
        html.Th('#',    className='text-center', style={'width':'32px','color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}','paddingBottom':'8px'}),
        html.Th('',     style={'width':'28px','borderBottom':f'1px solid {BORDER}'}),
        html.Th('EQUIPO',style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}','paddingBottom':'8px'}),
        html.Th('PJ',   className='text-center', style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}'}),
        html.Th('G',    className='text-center', style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}'}),
        html.Th('E',    className='text-center', style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}'}),
        html.Th('P',    className='text-center', style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}'}),
        html.Th('GF',   className='text-center', style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}'}),
        html.Th('GC',   className='text-center', style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}'}),
        html.Th('DIF',  className='text-center', style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}'}),
        html.Th('PTS',  className='text-center', style={'color':MUTED,'fontSize':'10px','letterSpacing':'1px','borderBottom':f'1px solid {BORDER}'}),
    ]))

    rows = []
    for r in tabla:
        pos = int(r.get('pos', 0))
        t   = _norm(r['equipo'])
        tid = r.get('equipo_id') or team_ids.get(t)
        tc  = team_colors.get(t, GRAY)
        pts = int(r.get('pts', 0))
        pj  = int(r.get('pj',  0))
        g   = int(r.get('g',   0))
        e   = int(r.get('e',   0))
        p   = int(r.get('p',   0))
        gf  = int(r.get('gf',  0))
        gc  = int(r.get('gc',  0))
        dif = gf - gc

        # Zone colours
        if pos <= 4:
            row_bg  = 'rgba(46,160,67,0.10)'
            pos_col = GREEN
            left_border = f'3px solid {GREEN}'
        elif pos <= 8:
            row_bg  = 'rgba(212,167,44,0.10)'
            pos_col = GOLD
            left_border = f'3px solid {GOLD}'
        else:
            row_bg  = 'transparent'
            pos_col = MUTED
            left_border = '3px solid transparent'

        dif_col = GREEN if dif > 0 else (DANGER if dif < 0 else MUTED)
        dif_str = f'+{dif}' if dif > 0 else str(dif)

        td_c = {'textAlign':'center','padding':'7px 6px','fontSize':'12px',
                'fontFamily':'Roboto, sans-serif'}
        td_l = {**td_c, 'textAlign':'left'}

        rows.append(html.Tr([
            html.Td(str(pos), style={**td_c, 'color':pos_col,'fontWeight':'700',
                                     'paddingLeft':'10px','borderLeft':left_border}),
            html.Td(_shield(tid, 22), style={'padding':'4px 6px','verticalAlign':'middle'}),
            html.Td(html.Span(t, style={'color':tc,'fontWeight':'600'}), style=td_l),
            html.Td(str(pj), style={**td_c,'color':MUTED}),
            html.Td(str(g),  style={**td_c,'color':GREEN,'fontWeight':'600'}),
            html.Td(str(e),  style={**td_c,'color':MUTED}),
            html.Td(str(p),  style={**td_c,'color':DANGER}),
            html.Td(str(gf), style={**td_c,'color':WHITE}),
            html.Td(str(gc), style={**td_c,'color':WHITE}),
            html.Td(dif_str, style={**td_c,'color':dif_col,'fontWeight':'600'}),
            html.Td(str(pts),style={**td_c,'color':WHITE,'fontWeight':'800','fontSize':'13px'}),
        ], style={'backgroundColor':row_bg}))

    legend = html.Div([
        html.Span('●', style={'color':GREEN,'marginRight':'4px','fontSize':'10px'}),
        html.Span('Top 4 — Liguilla  ', style={'color':MUTED,'fontSize':'10px'}),
        html.Span('●', style={'color':GOLD,'marginRight':'4px','fontSize':'10px','marginLeft':'8px'}),
        html.Span('Top 8 — Repechaje', style={'color':MUTED,'fontSize':'10px'}),
    ], style={'marginBottom':'10px'})

    return html.Div([
        html.Div('Tabla de posiciones', style={
            'fontSize': '12px', 'fontWeight': '700', 'letterSpacing': '2px',
            'textTransform': 'uppercase', 'color': GRAY,
            'marginBottom': '14px', 'paddingBottom': '8px',
            'borderBottom': f'1px solid {BORDER}',
            'fontFamily': 'Roboto, sans-serif',
        }),
        legend,
        html.Div(
            dbc.Table(
                [header, html.Tbody(rows)],
                striped=False, bordered=False, hover=True,
                color='dark',
                style={
                    'marginBottom': '0',
                    'color': WHITE,
                    '--bs-table-hover-bg': 'rgba(255,255,255,0.04)',
                    '--bs-table-bg': 'transparent',
                    '--bs-table-color': WHITE,
                }
            ),
            style={'overflowX': 'auto'},
        ),
    ], style={**CARD_STYLE, 'padding': '20px'})


# ── RESULTS J12 ────────────────────────────────────────────────────────────────
def _build_results(hist, team_ids, team_colors):
    partidos = [p for p in hist.get('partidos', [])
                if p.get('jornada') == '12' and p.get('terminado')]

    cards = []
    for p in partidos:
        lo  = _norm(p['local']); vi = _norm(p['visitante'])
        gl  = int(p.get('goles_local',  0) or 0)
        gv  = int(p.get('goles_visit', 0) or 0)
        tid_lo = p.get('local_id')    or team_ids.get(lo)
        tid_vi = p.get('visitante_id') or team_ids.get(vi)

        if   gl > gv: winner = 'local'
        elif gv > gl: winner = 'visit'
        else:         winner = 'draw'

        lo_weight = '700' if winner == 'local' else '400'
        vi_weight = '700' if winner == 'visit' else '400'

        cards.append(html.Div([
            # Local
            html.Div([
                _shield(tid_lo, 22),
                html.Span(lo, style={
                    'color': team_colors.get(lo, MUTED),
                    'fontWeight': lo_weight, 'fontSize': '11px',
                    'fontFamily': 'Roboto, sans-serif',
                }),
            ], style={'display':'flex','alignItems':'center','gap':'7px',
                      'flex':'1','justifyContent':'flex-end'}),

            # Score
            html.Div(f'{gl} – {gv}', style={
                'fontSize': '16px', 'fontWeight': '900', 'color': WHITE,
                'fontFamily': 'Roboto, sans-serif',
                'letterSpacing': '2px', 'minWidth': '60px', 'textAlign': 'center',
                'padding': '0 8px',
            }),

            # Visitante
            html.Div([
                html.Span(vi, style={
                    'color': team_colors.get(vi, MUTED),
                    'fontWeight': vi_weight, 'fontSize': '11px',
                    'fontFamily': 'Roboto, sans-serif',
                }),
                _shield(tid_vi, 22),
            ], style={'display':'flex','alignItems':'center','gap':'7px','flex':'1'}),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
            'padding': '8px 10px',
            'background': 'rgba(255,255,255,0.02)',
            'border': f'1px solid {BORDER}',
            'borderRadius': '8px',
            'marginBottom': '6px',
            'transition': 'border-color 0.15s',
        }))

    return html.Div([
        html.Div('Últimos resultados — J12', style={
            'fontSize': '11px', 'fontWeight': '700', 'letterSpacing': '2px',
            'textTransform': 'uppercase', 'color': GRAY,
            'marginBottom': '12px', 'paddingBottom': '8px',
            'borderBottom': f'1px solid {BORDER}',
            'fontFamily': 'Roboto, sans-serif',
        }),
        html.Div(cards) if cards else html.Div('Sin resultados',
            style={'color':MUTED,'fontSize':'12px','textAlign':'center'}),
    ], style={**CARD_STYLE, 'marginBottom': '16px'})


# ── UPCOMING J13 ──────────────────────────────────────────────────────────────
def _build_upcoming(hist, team_ids, team_colors):
    partidos = [p for p in hist.get('partidos', [])
                if p.get('jornada') == '13' and not p.get('terminado')]

    cards = []
    for p in partidos:
        lo  = _norm(p['local']); vi = _norm(p['visitante'])
        tid_lo = p.get('local_id')    or team_ids.get(lo)
        tid_vi = p.get('visitante_id') or team_ids.get(vi)

        cards.append(html.Div([
            # Local
            html.Div([
                _shield(tid_lo, 22),
                html.Span(lo, style={
                    'color': team_colors.get(lo, MUTED),
                    'fontWeight': '600', 'fontSize': '11px',
                    'fontFamily': 'Roboto, sans-serif',
                }),
            ], style={'display':'flex','alignItems':'center','gap':'7px',
                      'flex':'1','justifyContent':'flex-end'}),

            # VS pill
            html.Div('VS', style={
                'fontSize': '9px', 'fontWeight': '700', 'letterSpacing': '2px',
                'color': MUTED,
                'background': 'rgba(255,255,255,0.05)',
                'border': f'1px solid {BORDER}',
                'borderRadius': '10px', 'padding': '2px 10px',
                'minWidth': '36px', 'textAlign': 'center',
                'fontFamily': 'Roboto, sans-serif',
            }),

            # Visitante
            html.Div([
                html.Span(vi, style={
                    'color': team_colors.get(vi, MUTED),
                    'fontWeight': '600', 'fontSize': '11px',
                    'fontFamily': 'Roboto, sans-serif',
                }),
                _shield(tid_vi, 22),
            ], style={'display':'flex','alignItems':'center','gap':'7px','flex':'1'}),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
            'padding': '8px 10px',
            'background': 'rgba(213,0,28,0.03)',
            'border': f'1px solid rgba(213,0,28,0.15)',
            'borderRadius': '8px',
            'marginBottom': '6px',
        }))

    return html.Div([
        html.Div('Próxima jornada — J13', style={
            'fontSize': '11px', 'fontWeight': '700', 'letterSpacing': '2px',
            'textTransform': 'uppercase', 'color': GRAY,
            'marginBottom': '12px', 'paddingBottom': '8px',
            'borderBottom': f'1px solid {BORDER}',
            'fontFamily': 'Roboto, sans-serif',
        }),
        html.Div(cards) if cards else html.Div('Sin partidos programados',
            style={'color':MUTED,'fontSize':'12px','textAlign':'center'}),
    ], style=CARD_STYLE)


# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
def layout(hist, jug, elo, team_ids, team_colors):
    """
    Retorna el layout completo de la página Home.
    Parámetros: datos ya cargados desde app.py
    """
    return html.Div([
        # ── Page header ──
        html.Div([
            html.Div([
                html.Div('Dashboard', style={
                    'fontSize': '22px', 'fontWeight': '400',
                    'letterSpacing': '3px', 'color': WHITE,
                    'textTransform': 'uppercase',
                    'fontFamily': "'Bebas Neue', 'Arial Black', sans-serif",
                }),
                html.Div('Liga MX · Clausura 2026', style={
                    'fontSize': '11px', 'color': MUTED, 'letterSpacing': '1px',
                    'marginTop': '2px', 'fontFamily': 'Roboto, sans-serif',
                }),
            ]),
            html.Div('J13 Próxima', style={
                'background': 'rgba(213,0,28,0.12)',
                'border': '1px solid rgba(213,0,28,0.3)',
                'color': RED, 'fontSize': '10px', 'fontWeight': '700',
                'letterSpacing': '1.5px', 'padding': '4px 12px',
                'borderRadius': '4px', 'textTransform': 'uppercase',
                'fontFamily': 'Roboto, sans-serif',
            }),
        ], style={
            'background': 'linear-gradient(90deg, rgba(213,0,28,0.06) 0%, transparent 100%)',
            'borderBottom': '1px solid #1a2030',
            'padding': '18px 28px 16px',
            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
        }),

        # ── Body ──
        html.Div([
            # KPI row
            _build_kpis(hist, jug, elo, team_ids, team_colors),

            # Main columns: standings (8) + results/upcoming (4)
            dbc.Row([
                # ── Columna izquierda: tabla ──
                dbc.Col(
                    _build_standings(hist, team_ids, team_colors),
                    md=8, style={'display':'flex','flexDirection':'column'},
                ),
                # ── Columna derecha: resultados + próxima ──
                dbc.Col([
                    _build_results(hist, team_ids, team_colors),
                    _build_upcoming(hist, team_ids, team_colors),
                ], md=4, style={'display':'flex','flexDirection':'column'}),
            ], className='g-3'),
        ], style={'padding': '20px 24px'}),
    ])
