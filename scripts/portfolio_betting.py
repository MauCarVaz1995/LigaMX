#!/usr/bin/env python3
"""
portfolio_betting.py — Motor de portafolio de apuestas (estilo quant)
=====================================================================
NO es un parlay. Es un portafolio de posiciones independientes con EV positivo,
como un quant fund que abre múltiples posiciones pequeñas basadas en edge real.

Filosofía:
  - Cada apuesta es una "posición" con edge estimado
  - Kelly fraccionado 25% con cap por posición (riesgo controlado)
  - Diversificación: múltiples mercados / partidos / ligas
  - La suma de EVs positivos converge en ganancia por la ley de grandes números
  - NO depende de que un partido específico resulte bien

Output:
  - Portafolio del día: posiciones, sizing, expected return total
  - Tracker histórico: bankroll, ROI acumulado, Sharpe betting
  - HTML para el email

Matemática:
  EV_i     = prob_i × cuota_i − 1
  Kelly_i  = EV_i / (cuota_i − 1) × 0.25  (fracción 25%)
  Bet_i    = min(Kelly_i × bankroll, MAX_POS × bankroll)
  Total    = Σ Bet_i  (cap: MAX_TOTAL × bankroll)
  E[P&L]   = Σ (EV_i × Bet_i)
  Var[P&L] = Σ (Bet_i² × prob_i × (1−prob_i))  (asumiendo independencia)

Uso:
  python scripts/portfolio_betting.py --bankroll 1000
  python scripts/portfolio_betting.py --bankroll 1000 --min-ev 0.05
  python scripts/portfolio_betting.py --history   # muestra historial
"""

import argparse
import json
import os
import sys
import warnings
import glob
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import poisson, norm

warnings.filterwarnings("ignore")

BASE        = Path(__file__).resolve().parent.parent
SCRIPTS     = BASE / "scripts"
REPORTS_DIR = BASE / "output/reports"
HIST_DIR    = BASE / "data/raw/historico"
PORTFOLIO_LOG = BASE / "data/processed/portfolio_log.csv"
BANKROLL_FILE = BASE / "data/processed/bankroll.json"

MX_TZ  = timezone(timedelta(hours=-6))
NOW_MX = datetime.now(MX_TZ)
TODAY  = NOW_MX.strftime("%Y-%m-%d")

# ── Parámetros del portafolio ──────────────────────────────────────────────
KELLY_FRAC  = 0.25     # Kelly fraccionado — reduce varianza
MAX_POS_PCT = 0.025    # máximo 2.5% del bankroll por posición
MAX_TOTAL_PCT = 0.10   # máximo 10% del bankroll total invertido por día
MIN_EV      = 0.04     # EV mínimo para incluir en portafolio (4%)
MIN_PROB    = 0.30     # prob mínima (evitar apuestas de muy baja prob)
MAX_PROB    = 0.92     # prob máxima (evitar papas seguras sin valor real)


# ─────────────────────────────────────────────────────────────────────────────
# Bankroll tracker
# ─────────────────────────────────────────────────────────────────────────────
def load_bankroll(default: float = 1000.0) -> dict:
    if BANKROLL_FILE.exists():
        try:
            return json.loads(BANKROLL_FILE.read_text())
        except Exception:
            pass
    return {"inicial": default, "actual": default, "fecha_inicio": TODAY}


def save_bankroll(data: dict):
    BANKROLL_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Cargar oportunidades del día
# ─────────────────────────────────────────────────────────────────────────────
def load_opportunities() -> list[dict]:
    """
    Carga todas las oportunidades EV+ del día desde:
    1. Odds API (cuotas reales Liga MX)
    2. Betting JSONs del bot (probs del modelo)
    3. Internacional (sin cuota real → watchlist)
    Retorna lista de dicts con toda la info para el portafolio.
    """
    opps = []

    # ── Liga MX: leer desde update_odds_ligamx ──
    try:
        sys.path.insert(0, str(SCRIPTS))
        from update_odds_ligamx import fetch_odds, BOOKMAKERS
        from send_bets_email import (
            load_model_probs, get_ligamx_today, fetch_ligamx_odds,
            build_ligamx_value_bets, dc_1x2, ev as _ev
        )

        partidos_hoy = get_ligamx_today()
        odds_dict    = fetch_ligamx_odds()
        probs_dict   = load_model_probs()

        value_bets   = build_ligamx_value_bets(partidos_hoy, odds_dict, probs_dict)

        for vb in value_bets:
            lk = vb["partido"].split(" vs ")[0].strip().lower()
            vk = vb["partido"].split(" vs ")[1].strip().lower()
            opps.append({
                "fecha":    TODAY,
                "liga":     "Liga MX",
                "partido":  vb["partido"],
                "hora":     vb.get("hora","?"),
                "mercado":  vb["mercado"],
                "prob":     vb["prob"],
                "cuota":    vb["cuota"],
                "bk":       vb.get("bk","?"),
                "ev":       vb["ev"],
                "tiene_cuota": True,
                "tipo":     "ligamx",
            })
    except Exception as e:
        print(f"  [warn] Liga MX opps: {e}")

    # ── Internacional: picks de alta confianza (watchlist, sin cuota) ──
    try:
        from send_bets_email import load_intl_picks
        intl = load_intl_picks()
        for pk in intl:
            opps.append({
                "fecha":    pk["fecha"],
                "liga":     pk["liga"],
                "partido":  f"{pk['local']} vs {pk['visita']}",
                "hora":     "?",
                "mercado":  pk["mercado"],
                "prob":     pk["prob"],
                "cuota":    None,  # sin cuota real
                "bk":       "—",
                "ev":       None,  # sin EV calculable
                "tiene_cuota": False,
                "tipo":     "intl",
            })
    except Exception as e:
        print(f"  [warn] Intl opps: {e}")

    return opps


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del portafolio
# ─────────────────────────────────────────────────────────────────────────────
def build_portfolio(opps: list[dict], bankroll: float) -> dict:
    """
    Construye el portafolio óptimo:
    - Solo posiciones con cuota real y EV >= MIN_EV
    - Kelly fraccionado con caps
    - Correlación: max 1 posición por partido-mercado-lado
    - Diversificación: no concentrar todo en un partido
    """
    # Filtrar elegibles para el portafolio activo
    elegibles = [o for o in opps
                 if o["tiene_cuota"]
                 and o["ev"] is not None
                 and o["ev"] >= MIN_EV
                 and MIN_PROB <= o["prob"] <= MAX_PROB]

    # Ordenar por EV descendente
    elegibles.sort(key=lambda x: -x["ev"])

    # Anti-correlación: máximo 2 posiciones por partido
    partido_count: dict[str, int] = {}
    posiciones = []

    for opp in elegibles:
        partido = opp["partido"]
        partido_count[partido] = partido_count.get(partido, 0)
        if partido_count[partido] >= 2:
            continue  # ya tenemos 2 posiciones en este partido

        # Sizing: Kelly fraccionado con cap
        b = opp["cuota"] - 1
        kelly_frac = (b * opp["prob"] - (1 - opp["prob"])) / b if b > 0 else 0
        kelly_frac = max(kelly_frac, 0)
        bet_pct   = min(kelly_frac * KELLY_FRAC, MAX_POS_PCT)
        bet_mxn   = round(bet_pct * bankroll, 1)

        if bet_mxn < 5.0:  # mínimo $5 MXN por apuesta
            continue

        posiciones.append({
            **opp,
            "kelly_full":  round(kelly_frac, 4),
            "kelly_frac":  round(kelly_frac * KELLY_FRAC, 4),
            "bet_pct":     round(bet_pct * 100, 2),
            "bet_mxn":     bet_mxn,
            "expected_return": round(opp["ev"] * bet_mxn, 2),
        })
        partido_count[partido] += 1

    # Cap total de exposición
    total_bet = sum(p["bet_mxn"] for p in posiciones)
    max_total = MAX_TOTAL_PCT * bankroll

    if total_bet > max_total:
        scale = max_total / total_bet
        for p in posiciones:
            p["bet_mxn"]         = round(p["bet_mxn"] * scale, 1)
            p["expected_return"] = round(p["ev"] * p["bet_mxn"], 2)
            p["bet_pct"]         = round(p["bet_pct"] * scale, 2)
        total_bet = max_total

    # Métricas del portafolio
    total_ev       = sum(p["expected_return"] for p in posiciones)
    total_bet      = sum(p["bet_mxn"] for p in posiciones)
    roi_esperado   = (total_ev / total_bet * 100) if total_bet > 0 else 0

    # Varianza del portafolio (asumiendo independencia — posiciones de ligas distintas)
    variance = sum(
        p["bet_mxn"]**2 * p["prob"] * (1 - p["prob"])
        for p in posiciones
    )
    std_dev  = variance ** 0.5
    sharpe   = total_ev / std_dev if std_dev > 0 else 0

    # Prob de que el portafolio sea rentable (aprox. normal por CLT)
    prob_profit = float(norm.cdf(total_ev / std_dev)) if std_dev > 0 else 0.5

    # Watchlist internacional (sin cuota real)
    watchlist = [o for o in opps
                 if not o["tiene_cuota"] and o["prob"] >= 0.65]
    watchlist.sort(key=lambda x: -x["prob"])

    # Guía "donde apostar": cuotas mínimas para cada mercado del portafolio
    # + mejor casa disponible en The Odds API accesible desde MX
    MX_ACCESSIBLE = ["onexbet", "betway", "betonlineag", "lowvig", "marathonbet",
                     "betus", "coolbet", "everygame", "betsson"]
    MIN_EV_GUIDE  = 0.04
    guia_casas = []
    for p in posiciones:
        cuota_min_caliente = round(1 / p["prob"] * (1 + MIN_EV_GUIDE), 2)
        # Buscar la mejor cuota accesible desde MX (no solo Pinnacle)
        best_mx = p["cuota"]  # ya tenemos la mejor cuota de todo The Odds API
        bk_mx   = p.get("bk","?")
        guia_casas.append({
            **{k: p[k] for k in ("partido","mercado","prob","cuota","ev","bk")},
            "cuota_min_caliente": cuota_min_caliente,
            "best_cuota_mx":      best_mx,
            "best_bk_mx":         bk_mx,
        })

    return {
        "fecha":         TODAY,
        "bankroll":      bankroll,
        "posiciones":    posiciones,
        "watchlist":     watchlist[:12],
        "guia_casas":    guia_casas,
        "total_invertido": round(total_bet, 1),
        "total_ev":      round(total_ev, 2),
        "roi_esperado":  round(roi_esperado, 2),
        "std_dev":       round(std_dev, 2),
        "sharpe":        round(sharpe, 3),
        "prob_profit":   round(prob_profit * 100, 1),
        "n_posiciones":  len(posiciones),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registro y tracker histórico
# ─────────────────────────────────────────────────────────────────────────────
def log_portfolio(portfolio: dict):
    """Guarda las posiciones del día en el log histórico."""
    rows = []
    for p in portfolio["posiciones"]:
        rows.append({
            "fecha":        TODAY,
            "liga":         p["liga"],
            "partido":      p["partido"],
            "hora":         p.get("hora","?"),
            "mercado":      p["mercado"],
            "prob_modelo":  p["prob"],
            "cuota":        p["cuota"],
            "bookmaker":    p["bk"],
            "ev":           p["ev"],
            "bet_mxn":      p["bet_mxn"],
            "expected_return": p["expected_return"],
            "resultado":    None,  # se llena después con bet_result
            "ganancia_real": None,
        })

    df_new = pd.DataFrame(rows)

    if PORTFOLIO_LOG.exists():
        df_old = pd.read_csv(PORTFOLIO_LOG)
        # No duplicar mismo partido+mercado+fecha
        df_out = pd.concat([df_old, df_new]).drop_duplicates(
            subset=["fecha","partido","mercado"], keep="last"
        ).reset_index(drop=True)
    else:
        df_out = df_new

    df_out.to_csv(PORTFOLIO_LOG, index=False)


def update_results(portfolio: dict) -> dict:
    """
    Actualiza resultados reales en portfolio_log.csv comparando contra
    resultados en historico_clausura_2026.json.
    Retorna dict con aciertos del día.
    """
    if not PORTFOLIO_LOG.exists():
        return {}

    df = pd.read_csv(PORTFOLIO_LOG)
    pending = df[df["resultado"].isna() & (df["fecha"] < TODAY)]
    if pending.empty:
        return {}

    # Cargar resultados del historico
    resultados: dict[tuple, tuple] = {}
    for hf in sorted(HIST_DIR.glob("historico_*.json"), reverse=True)[:2]:
        try:
            d = json.loads(hf.read_text())
            for p in d.get("partidos",[]):
                if not p.get("terminado"):
                    continue
                local  = p["local"].lower()
                visita = p.get("visitante","").lower()
                gl, gv = p.get("goles_local"), p.get("goles_visit")
                if gl is not None and gv is not None:
                    resultados[(local, visita)] = (int(gl), int(gv))
        except Exception:
            pass

    ganancia_total = 0.0
    for idx, row in pending.iterrows():
        local_r  = row["partido"].split(" vs ")[0].strip().lower()
        visita_r = row["partido"].split(" vs ")[1].strip().lower()
        res = resultados.get((local_r, visita_r))
        if res is None:
            continue
        gl, gv = res
        total_goles = gl + gv
        mercado = str(row["mercado"])
        # Evaluar resultado
        acierto = False
        if mercado.startswith("1 ("):         acierto = gl > gv
        elif mercado == "X (Empate)":          acierto = gl == gv
        elif mercado.startswith("2 ("):        acierto = gv > gl
        elif mercado == "Over 2.5 goles":      acierto = total_goles > 2.5
        elif mercado == "Under 2.5 goles":     acierto = total_goles < 2.5
        elif mercado == "Ambos anotan Sí":     acierto = gl > 0 and gv > 0
        elif mercado == "Tarjetas O4.5":       pass  # requiere match_events

        ganancia = round((row["cuota"] - 1) * row["bet_mxn"] if acierto else -row["bet_mxn"], 2)
        df.at[idx, "resultado"]    = "✅" if acierto else "❌"
        df.at[idx, "ganancia_real"] = ganancia
        ganancia_total += ganancia

    df.to_csv(PORTFOLIO_LOG, index=False)
    return {"ganancia_total": round(ganancia_total, 2)}


def get_performance_stats() -> dict:
    """Métricas históricas del portafolio."""
    if not PORTFOLIO_LOG.exists():
        return {}
    df = pd.read_csv(PORTFOLIO_LOG)
    resueltos = df.dropna(subset=["resultado"])
    if resueltos.empty:
        return {}

    total_apostado = resueltos["bet_mxn"].sum()
    ganancia       = resueltos["ganancia_real"].fillna(0).sum()
    n_total        = len(resueltos)
    n_aciertos     = (resueltos["resultado"] == "✅").sum()
    roi            = ganancia / total_apostado * 100 if total_apostado > 0 else 0

    # ROI por día (para Sharpe)
    daily = resueltos.groupby("fecha")["ganancia_real"].sum()
    sharpe = (daily.mean() / daily.std() * (365**0.5)) if daily.std() > 0 else 0

    # Drawdown
    bankroll_data = load_bankroll()
    bankroll_actual = bankroll_data["inicial"] + ganancia

    return {
        "n_apuestas":    n_total,
        "n_aciertos":    n_aciertos,
        "pct_aciertos":  round(n_aciertos / n_total * 100, 1) if n_total else 0,
        "total_apostado": round(total_apostado, 1),
        "ganancia_total": round(ganancia, 2),
        "roi_acumulado": round(roi, 2),
        "sharpe_anual":  round(sharpe, 3),
        "bankroll_actual": round(bankroll_actual, 2),
        "bankroll_inicial": bankroll_data["inicial"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML del email
# ─────────────────────────────────────────────────────────────────────────────
def build_portfolio_html(portfolio: dict, stats: dict) -> str:
    bankroll = portfolio["bankroll"]
    n_pos    = portfolio["n_posiciones"]
    total_inv= portfolio["total_invertido"]
    total_ev = portfolio["total_ev"]
    roi_esp  = portfolio["roi_esperado"]
    prob_profit = portfolio["prob_profit"]
    sharpe   = portfolio["sharpe"]

    # ── Stats históricos ──
    if stats:
        roi_hist  = stats.get("roi_acumulado", 0)
        n_hist    = stats.get("n_apuestas", 0)
        pct_hist  = stats.get("pct_aciertos", 0)
        gan_hist  = stats.get("ganancia_total", 0)
        bk_actual = stats.get("bankroll_actual", bankroll)
        roi_color = "#00C853" if roi_hist >= 0 else "#EF5350"
        hist_html = f"""
        <tr><td class="g">Apuestas resueltas</td><td><b>{n_hist}</b></td></tr>
        <tr><td class="g">Aciertos</td><td><b>{pct_hist:.1f}%</b></td></tr>
        <tr><td class="g">Ganancia acumulada</td>
            <td><b style="color:{roi_color}">${gan_hist:+.2f} MXN</b></td></tr>
        <tr><td class="g">ROI histórico</td>
            <td><b style="color:{roi_color}">{roi_hist:+.1f}%</b></td></tr>
        <tr><td class="g">Bankroll actual</td>
            <td><b>${bk_actual:.2f} MXN</b></td></tr>"""
    else:
        hist_html = '<tr><td colspan="2" class="g">Sin historial — primer día</td></tr>'

    # ── Posiciones del portafolio ──
    pos_html = ""
    for i, p in enumerate(portfolio["posiciones"], 1):
        ev_pct = p["ev"] * 100
        star = "⭐" if ev_pct >= 12 else ""
        ev_color = "#00C853" if ev_pct >= 8 else "#FFD740"
        pos_html += f"""
        <tr style="background:{'#1a2a1a' if i%2 else '#12161e'}">
          <td style="padding:5px 8px;color:#aaa;font-size:11px">{p.get('hora','?')}</td>
          <td style="padding:5px 8px;font-size:12px">
            <b>{p['partido']}</b>
            <div style="font-size:10px;color:#64B5F6">{p['liga']}</div>
          </td>
          <td style="padding:5px 8px;color:#ddd;font-size:11px">{p['mercado']}</td>
          <td style="padding:5px 8px;text-align:center;font-size:11px">
            {p['prob']:.0%}<br>
            <span style="color:#888;font-size:10px">@{p['cuota']:.2f}</span>
          </td>
          <td style="padding:5px 8px;text-align:center;color:{ev_color};font-weight:bold">
            {ev_pct:+.1f}%{star}
          </td>
          <td style="padding:5px 8px;text-align:center">
            <b style="color:#FFD740">${p['bet_mxn']:.0f}</b>
            <div style="font-size:10px;color:#888">{p['bet_pct']:.1f}%</div>
          </td>
          <td style="padding:5px 8px;text-align:center;color:#00C853">
            +${p['expected_return']:.1f}
          </td>
          <td style="padding:5px 8px;text-align:center;font-size:10px;color:#888">{p['bk']}</td>
        </tr>"""

    if not pos_html:
        pos_html = '<tr><td colspan="8" style="padding:12px;color:#888;text-align:center">Sin posiciones con cuota real disponibles hoy</td></tr>'

    # ── Watchlist internacional ──
    watch_html = ""
    liga_seen = set()
    for w in portfolio["watchlist"][:10]:
        if w["liga"] not in liga_seen:
            watch_html += f'<tr style="background:#0d1117"><td colspan="4" style="padding:3px 8px;color:#64B5F6;font-size:10px">{w["liga"].upper()}</td></tr>'
            liga_seen.add(w["liga"])
        watch_html += f"""
        <tr style="background:#111">
          <td style="padding:3px 8px;font-size:11px;color:#ddd">{w['fecha']}</td>
          <td style="padding:3px 8px;font-size:11px">{w['partido']}</td>
          <td style="padding:3px 8px;font-size:11px;color:#aaa">{w['mercado']}</td>
          <td style="padding:3px 8px;text-align:center;color:#00C853;font-size:11px">{w['prob']:.0%}</td>
        </tr>"""
    if not watch_html:
        watch_html = '<tr><td colspan="4" style="color:#888;padding:8px">Sin picks internacionales hoy</td></tr>'

    # ── Resumen del portafolio ──
    prob_color  = "#00C853" if prob_profit >= 60 else "#FFD740"
    roi_color2  = "#00C853" if roi_esp >= 0 else "#EF5350"

    return f"""
<div style="font-family:monospace;background:#0d1117;color:#e0e0e0;padding:12px;border-radius:8px">

  <!-- HEADER -->
  <div style="background:#0a0a1a;padding:12px 16px;border-radius:8px;
              margin-bottom:12px;border-left:4px solid #00C853">
    <div style="color:#00C853;font-size:16px;font-weight:bold">
      📊 PORTAFOLIO DE APUESTAS · {TODAY} {NOW_MX.strftime('%H:%M')} CDMX
    </div>
    <div style="color:#888;font-size:11px;margin-top:4px">
      Estrategia: múltiples posiciones EV+ independientes · Kelly fraccionado 25%
      · Riesgo controlado · Como un fondo quant
    </div>
  </div>

  <!-- RESUMEN DEL DÍA -->
  <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">

    <div style="background:#111827;padding:10px 14px;border-radius:6px;flex:1;min-width:130px;text-align:center">
      <div style="color:#888;font-size:10px">POSICIONES</div>
      <div style="font-size:22px;font-weight:bold;color:#FFD740">{n_pos}</div>
      <div style="color:#888;font-size:10px">apuestas hoy</div>
    </div>

    <div style="background:#111827;padding:10px 14px;border-radius:6px;flex:1;min-width:130px;text-align:center">
      <div style="color:#888;font-size:10px">INVERTIDO</div>
      <div style="font-size:22px;font-weight:bold;color:#64B5F6">${total_inv:.0f}</div>
      <div style="color:#888;font-size:10px">de ${bankroll:.0f} ({total_inv/bankroll*100:.1f}%)</div>
    </div>

    <div style="background:#111827;padding:10px 14px;border-radius:6px;flex:1;min-width:130px;text-align:center">
      <div style="color:#888;font-size:10px">RETORNO ESPERADO</div>
      <div style="font-size:22px;font-weight:bold;color:{roi_color2}">+${total_ev:.1f}</div>
      <div style="color:#888;font-size:10px">ROI {roi_esp:+.1f}%</div>
    </div>

    <div style="background:#111827;padding:10px 14px;border-radius:6px;flex:1;min-width:130px;text-align:center">
      <div style="color:#888;font-size:10px">PROB GANANCIA</div>
      <div style="font-size:22px;font-weight:bold;color:{prob_color}">{prob_profit:.0f}%</div>
      <div style="color:#888;font-size:10px">Sharpe {sharpe:.2f}</div>
    </div>

  </div>

  <!-- TABLA DE POSICIONES -->
  <div style="background:#111827;border-radius:6px;margin-bottom:12px;overflow:hidden">
    <div style="background:#1a1a2e;padding:8px 12px;color:#FFD740;font-weight:bold;font-size:13px">
      ⚡ POSICIONES ACTIVAS — apuestas con cuota real y EV ≥{MIN_EV:.0%}
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <tr style="background:#0d1117;color:#666;font-size:10px">
        <th style="padding:4px 8px">Hora</th>
        <th style="padding:4px 8px;text-align:left">Partido</th>
        <th style="padding:4px 8px;text-align:left">Mercado</th>
        <th style="padding:4px 8px">Prob/Cuota</th>
        <th style="padding:4px 8px">EV</th>
        <th style="padding:4px 8px">Apuesta</th>
        <th style="padding:4px 8px">E[Ret]</th>
        <th style="padding:4px 8px">Casa</th>
      </tr>
      {pos_html}
    </table>
    <div style="padding:6px 10px;background:#0a0a14;font-size:10px;color:#555">
      Kelly 25% · Cap 2.5% por posición · Cap 10% bankroll total diario
      · E[Ret] = retorno esperado si se repite muchas veces
    </div>
  </div>

  <!-- PERFORMANCE HISTÓRICO -->
  <div style="background:#111827;border-radius:6px;margin-bottom:12px;padding:12px">
    <div style="color:#90CAF9;font-weight:bold;margin-bottom:8px;font-size:13px">
      📈 Desempeño histórico del portafolio
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <colgroup><col style="width:60%"><col style="width:40%"></colgroup>
      {hist_html}
    </table>
  </div>

  <!-- WATCHLIST INTERNACIONAL -->
  <div style="background:#111827;border-radius:6px;margin-bottom:12px;overflow:hidden">
    <div style="background:#1a1a2e;padding:8px 12px;color:#90CAF9;font-weight:bold;font-size:13px">
      🌎 Watchlist internacional — revisar cuotas para agregar al portafolio
    </div>
    <table style="width:100%;border-collapse:collapse">
      <tr style="background:#0d1117;color:#666;font-size:10px">
        <th style="padding:3px 8px">Fecha</th>
        <th style="padding:3px 8px;text-align:left">Partido</th>
        <th style="padding:3px 8px;text-align:left">Mercado</th>
        <th style="padding:3px 8px">Prob</th>
      </tr>
      {watch_html}
    </table>
    <div style="padding:6px 10px;background:#0a0a14;font-size:10px;color:#555">
      Sin cuota real → buscar en Caliente/Betway/Pinnacle · si prob ≥ 1/cuota entonces hay valor
    </div>
  </div>

  <!-- NOTAS QUANT -->
  <div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:6px;padding:10px 12px;font-size:11px;color:#888">
    <b style="color:#64B5F6">¿Por qué este enfoque gana dinero a largo plazo?</b><br>
    Si cada apuesta tiene EV+5%, en 20 apuestas el retorno esperado es +${bankroll*0.025:.0f} MXN.
    La varianza se reduce con más posiciones independientes (diversificación).
    El Kelly fraccionado protege de ruina: nunca más del 10% del bankroll en un día.
    Con 100 apuestas, la probabilidad de ser rentable supera el 75% aun con modelo imperfecto.
  </div>

</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bankroll",  type=float, default=None,
                        help="Bankroll actual en MXN (default: carga desde bankroll.json)")
    parser.add_argument("--min-ev",   type=float, default=MIN_EV)
    parser.add_argument("--history",  action="store_true", help="Mostrar historial")
    parser.add_argument("--update-results", action="store_true",
                        help="Actualizar resultados reales de apuestas anteriores")
    parser.add_argument("--set-bankroll", type=float,
                        help="Configurar bankroll inicial")
    args = parser.parse_args()

    # Configurar bankroll inicial
    if args.set_bankroll:
        save_bankroll({"inicial": args.set_bankroll, "actual": args.set_bankroll,
                       "fecha_inicio": TODAY})
        print(f"  Bankroll configurado: ${args.set_bankroll:.2f} MXN")
        return

    bk_data  = load_bankroll(args.bankroll or 1000.0)
    bankroll = args.bankroll or bk_data.get("actual", bk_data["inicial"])

    # Actualizar resultados pendientes
    if args.update_results:
        res = update_results({})
        if res:
            print(f"  Resultados actualizados · ganancia: ${res.get('ganancia_total',0):+.2f} MXN")
        # Actualizar bankroll
        if PORTFOLIO_LOG.exists():
            df = pd.read_csv(PORTFOLIO_LOG)
            gan = df["ganancia_real"].fillna(0).sum()
            bk_data["actual"] = round(bk_data["inicial"] + gan, 2)
            save_bankroll(bk_data)
            print(f"  Bankroll actualizado: ${bk_data['actual']:.2f} MXN")

    # Historial
    stats = get_performance_stats()
    if args.history:
        print(f"\n── Historial portafolio ──")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    print(f"\n── portfolio_betting.py · {TODAY} ──")
    print(f"  Bankroll: ${bankroll:.2f} MXN")

    opps      = load_opportunities()
    portfolio = build_portfolio(opps, bankroll)

    print(f"\n  {'─'*50}")
    print(f"  PORTAFOLIO DEL DÍA — {portfolio['n_posiciones']} posiciones")
    print(f"  {'─'*50}")
    print(f"  {'Partido':<32} {'Mercado':<18} {'Prob':>5} {'Cuota':>6} {'EV':>7} {'Apuesta':>9} {'E[Ret]':>7}")
    print(f"  {'─'*50}")

    for p in portfolio["posiciones"]:
        print(f"  {p['partido'][:31]:<32} {p['mercado'][:17]:<18} "
              f"{p['prob']:>4.0%} {p['cuota']:>6.2f} {p['ev']:>+6.1%} "
              f"${p['bet_mxn']:>7.1f} +${p['expected_return']:>5.1f}")

    print(f"\n  Total invertido:  ${portfolio['total_invertido']:.1f} MXN "
          f"({portfolio['total_invertido']/bankroll*100:.1f}% bankroll)")
    print(f"  Retorno esperado: +${portfolio['total_ev']:.2f} MXN ({portfolio['roi_esperado']:+.1f}%)")
    print(f"  Prob de ganancia: {portfolio['prob_profit']:.0f}%")
    print(f"  Sharpe:           {portfolio['sharpe']:.3f}")

    if stats:
        print(f"\n  ── Historial ──")
        print(f"  ROI acumulado:  {stats.get('roi_acumulado',0):+.1f}%")
        print(f"  Aciertos:       {stats.get('pct_aciertos',0):.1f}% ({stats.get('n_aciertos',0)}/{stats.get('n_apuestas',0)})")
        print(f"  Bankroll actual:${stats.get('bankroll_actual', bankroll):.2f}")

    # Guardar en log
    log_portfolio(portfolio)

    # Guardar HTML
    html = build_portfolio_html(portfolio, stats)
    out  = REPORTS_DIR / f"portfolio_{TODAY}.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n  HTML: {out}")


if __name__ == "__main__":
    main()
