#!/usr/bin/env python3
"""
send_bets_email.py — Email horario de VALUE BETS
=================================================
Envía un correo conciso con todas las apuestas de valor para HOY:
  - Liga MX: 1X2 + Over/Under goles + tarjetas (con cuotas reales Pinnacle)
  - Ligas internacionales: picks de alta confianza (Brasileirao, Premier, etc.)
  - Parlays: 2-3 selecciones combinadas con EV positivo
  - EV calculado con Dixon-Coles sobre ELOs reales

Diseñado para correr cada hora vía GitHub Actions.
Solo llama a The Odds API si hay partidos de Liga MX en las próximas 4 horas
(para no desperdiciar requests del free tier 500/mes).

Uso:
  python scripts/send_bets_email.py
  python scripts/send_bets_email.py --dry-run      # muestra sin enviar
  python scripts/send_bets_email.py --force         # envía aunque no haya partidos próximos
"""

import argparse
import glob
import json
import os
import smtplib
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import poisson

warnings.filterwarnings("ignore")

BASE        = Path(__file__).resolve().parent.parent
SCRIPTS     = BASE / "scripts"
REPORTS_DIR = BASE / "output/reports"
HIST_DIR    = BASE / "data/raw/historico"
ODDS_CSV    = BASE / "data/processed/odds_ligamx.csv"
BETTING_LOG = BASE / "data/processed/betting_log.csv"

MX_TZ     = timezone(timedelta(hours=-6))
NOW_MX    = datetime.now(MX_TZ)
TODAY     = NOW_MX.strftime("%Y-%m-%d")
NOW_H     = NOW_MX.strftime("%H:%M")

TO_ADDR   = os.environ.get("GMAIL_TO",   "maucarvaz@gmail.com")
FROM_ADDR = os.environ.get("GMAIL_FROM", "maucarvaz@gmail.com")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

ODDS_API_KEY = ""
_env = BASE / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

DC_RHO = -0.22

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def utc_to_mx(utc_str: str) -> datetime:
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        return dt.astimezone(MX_TZ)
    except Exception:
        return NOW_MX

def dc_1x2(lam_l: float, lam_v: float, rho: float = DC_RHO):
    w1 = d = w2 = 0.0
    for i in range(10):
        for j in range(10):
            p = poisson.pmf(i, lam_l) * poisson.pmf(j, lam_v)
            if i == 0 and j == 0: p *= (1 - lam_l * lam_v * rho)
            elif i == 1 and j == 0: p *= (1 + lam_v * rho)
            elif i == 0 and j == 1: p *= (1 + lam_l * rho)
            elif i == 1 and j == 1: p *= (1 - rho)
            if i > j: w1 += p
            elif i == j: d += p
            else: w2 += p
    return round(w1, 4), round(d, 4), round(w2, 4)

def ev(prob, cuota):
    return round(prob * cuota - 1, 4)

def kelly(prob, cuota, bankroll=1000, frac=0.25, max_pct=0.03):
    b = cuota - 1
    k = (b * prob - (1 - prob)) / b
    if k <= 0: return 0.0
    return round(min(k * frac, max_pct) * bankroll, 0)

# ─────────────────────────────────────────────────────────────────────────────
# Verificar si hay partidos próximos de Liga MX
# ─────────────────────────────────────────────────────────────────────────────
def get_ligamx_today() -> list[dict]:
    """Retorna partidos de Liga MX de HOY (hora CDMX) no terminados."""
    partidos = []
    for hf in sorted(HIST_DIR.glob("historico_*.json"), reverse=True)[:2]:
        try:
            d = json.loads(hf.read_text())
            for p in d.get("partidos", []):
                if p.get("terminado"):
                    continue
                dt_mx = utc_to_mx(p["fecha"])
                if dt_mx.strftime("%Y-%m-%d") == TODAY:
                    p["_dt_mx"] = dt_mx
                    p["_hora_mx"] = dt_mx.strftime("%H:%M")
                    partidos.append(p)
        except Exception:
            pass
    return partidos


def hours_to_next_game(partidos: list) -> float:
    """Horas hasta el próximo partido no iniciado."""
    future = [p for p in partidos
              if p["_dt_mx"] > NOW_MX - timedelta(minutes=10)]
    if not future:
        return 99.0
    next_dt = min(p["_dt_mx"] for p in future)
    return (next_dt - NOW_MX).total_seconds() / 3600


# ─────────────────────────────────────────────────────────────────────────────
# Odds API
# ─────────────────────────────────────────────────────────────────────────────
BOOKMAKERS = ["pinnacle", "betsson", "onexbet", "marathonbet", "williamhill",
              "draftkings", "betonlineag", "fanduel", "betway"]
TEAM_MAP   = {
    "Club America": "América", "Cruz Azul": "Cruz Azul",
    "Chivas Guadalajara": "Guadalajara", "CF Monterrey": "Monterrey",
    "Tigres UANL": "Tigres", "Pumas UNAM": "Pumas", "Toluca": "Toluca",
    "Atlas": "Atlas", "Santos Laguna": "Santos Laguna", "Necaxa": "Necaxa",
    "Mazatlan FC": "Mazatlán", "FC Juarez": "FC Juárez",
    "Atletico San Luis": "San Luis", "Queretaro": "Querétaro",
    "Tijuana": "Tijuana", "Leon": "León", "Puebla": "Puebla",
    "Pachuca": "Pachuca",
}

def fetch_ligamx_odds() -> dict:
    """Retorna {partido_key: {bk_odd_1, bk_odd_X, bk_odd_2, over25, under25}}"""
    if not ODDS_API_KEY:
        return {}
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_mexico_ligamx/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us,eu,uk",
                    "markets": "h2h,totals", "oddsFormat": "decimal"},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [warn] Odds API: {e}")
        return {}

    result = {}
    for event in data:
        local  = TEAM_MAP.get(event.get("home_team",""), event.get("home_team",""))
        visita = TEAM_MAP.get(event.get("away_team",""), event.get("away_team",""))
        fecha_raw = event.get("commence_time","")
        dt_mx = utc_to_mx(fecha_raw)
        key = (local.lower(), visita.lower())
        entry = {"local": local, "visita": visita, "dt_mx": dt_mx,
                 "hora": dt_mx.strftime("%H:%M")}

        for bk in event.get("bookmakers",[]):
            bk_k = bk.get("key","")
            for market in bk.get("markets",[]):
                mk = market.get("key","")
                ocs = {o["name"]: o["price"] for o in market.get("outcomes",[])}
                if mk == "h2h":
                    for side, name in [("odd_1", event.get("home_team","")),
                                       ("odd_X", "Draw"),
                                       ("odd_2", event.get("away_team",""))]:
                        v = ocs.get(name)
                        if v and float(v) > 1.0:
                            curr = entry.get(f"best_{side}", 0)
                            if float(v) > curr:
                                entry[f"best_{side}"] = float(v)
                                entry[f"bk_{side}"]   = bk_k
                elif mk == "totals":
                    for o in market.get("outcomes",[]):
                        if o.get("point") == 2.5:
                            key2 = "best_over25" if o["name"]=="Over" else "best_under25"
                            bkey = "bk_over25" if o["name"]=="Over" else "bk_under25"
                            if float(o["price"]) > entry.get(key2, 0):
                                entry[key2] = float(o["price"])
                                entry[bkey] = bk_k
        result[key] = entry
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model probabilities desde betting JSONs
# ─────────────────────────────────────────────────────────────────────────────
def load_model_probs() -> dict:
    """Carga las prob del modelo desde los JSONs del betting bot (más frescos)."""
    probs = {}  # (local_lower, visita_lower, mercado) → prob

    for jf in sorted(glob.glob(str(REPORTS_DIR / "betting_*.json")), reverse=True)[:5]:
        if "intl" in jf:
            continue  # Liga MX primero
        try:
            j_data = json.loads(Path(jf).read_text())
            if not isinstance(j_data, list):
                continue
            for entry in j_data:
                fecha_j = entry.get("fecha","")[:10]
                if fecha_j != TODAY:
                    continue
                lk = entry.get("local","").lower()
                vk = entry.get("visita","").lower()
                btts = entry.get("btts",{}) or {}
                tarj = entry.get("tarjetas",{}) or {}

                # goles y btts
                for mercado, p in [
                    ("over_2.5", btts.get("over_2.5")),
                    ("under_2.5", btts.get("under_2.5")),
                    ("btts_si", btts.get("btts_si")),
                    ("tarjetas_over_3.5", tarj.get("over_3.5")),
                    ("tarjetas_over_4.5", tarj.get("over_4.5")),
                    ("tarjetas_over_5.5", tarj.get("over_5.5")),
                ]:
                    if p is not None:
                        probs.setdefault((lk, vk, mercado), float(p))

                # 1X2 con Dixon-Coles
                lam_l = btts.get("lambda_local")
                lam_v = btts.get("lambda_visita")
                if lam_l and lam_v:
                    w1, d, w2 = dc_1x2(lam_l, lam_v)
                    probs.setdefault((lk, vk, "1x2_local"), w1)
                    probs.setdefault((lk, vk, "1x2_draw"),  d)
                    probs.setdefault((lk, vk, "1x2_visita"),w2)
        except Exception:
            pass
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Picks internacionales
# ─────────────────────────────────────────────────────────────────────────────
def load_intl_picks() -> list[dict]:
    """Lee picks de alta confianza de ligas internacionales del día."""
    intl_file = REPORTS_DIR / f"betting_intl_{TODAY.replace('-','')}.json"
    if not intl_file.exists():
        # buscar el más reciente
        candidates = sorted(glob.glob(str(REPORTS_DIR / "betting_intl_*.json")), reverse=True)
        if not candidates:
            return []
        intl_file = Path(candidates[0])

    try:
        data = json.loads(intl_file.read_text())
        if not isinstance(data, list):
            return []
    except Exception:
        return []

    picks = []
    for entry in data:
        liga  = entry.get("liga", entry.get("torneo","?")).replace("_"," ").title()
        local = entry.get("local","")
        visita= entry.get("visita", entry.get("visitante",""))
        fecha = entry.get("fecha","")[:10]
        if fecha < TODAY:
            continue
        btts = entry.get("btts",{}) or {}
        # Threshold diferenciado: over_1.5 ≥72% (alta prob), over_2.5/btts ≥62%
        for prob_key, label, threshold in [
            ("over_1.5", "Over 1.5 goles",  0.72),
            ("over_2.5", "Over 2.5 goles",  0.62),
            ("btts_si",  "Ambos anotan",     0.62),
        ]:
            p = btts.get(prob_key)
            if p and isinstance(p, (int, float)) and float(p) >= threshold:
                picks.append({
                    "liga": liga, "local": local, "visita": visita,
                    "fecha": fecha, "mercado": label, "prob": float(p),
                })
    # deduplicar mismo partido/mercado
    seen = set()
    unique = []
    for pk in sorted(picks, key=lambda x: -x["prob"]):
        key = (pk["local"], pk["visita"], pk["mercado"])
        if key not in seen:
            seen.add(key)
            unique.append(pk)
    return unique[:15]  # máx 15 picks intl


# ─────────────────────────────────────────────────────────────────────────────
# Construir tabla de value bets Liga MX
# ─────────────────────────────────────────────────────────────────────────────
def build_ligamx_value_bets(partidos: list, odds: dict, probs: dict) -> list[dict]:
    rows = []
    for p in partidos:
        lk = p["local"].lower()
        vk = p.get("visitante", p.get("visita","")).lower()
        key = (lk, vk)
        o   = odds.get(key, {})
        hora = p.get("_hora_mx","?")
        local  = p["local"]
        visita = p.get("visitante", p.get("visita",""))

        mercados = [
            ("1x2_local",  f"1 ({local})",     o.get("best_odd_1"),  o.get("bk_odd_1")),
            ("1x2_draw",   "X (Empate)",        o.get("best_odd_X"),  o.get("bk_odd_X")),
            ("1x2_visita", f"2 ({visita})",     o.get("best_odd_2"),  o.get("bk_odd_2")),
            ("over_2.5",   "Over 2.5 goles",    o.get("best_over25"), o.get("bk_over25")),
            ("under_2.5",  "Under 2.5 goles",   o.get("best_under25"),o.get("bk_under25")),
            ("btts_si",    "Ambos anotan Sí",    None,                 None),
            ("tarjetas_over_4.5", "Tarjetas O4.5", None,              None),
        ]

        for mercado, label, cuota, bk in mercados:
            prob_m = probs.get((lk, vk, mercado))
            if prob_m is None or cuota is None:
                continue
            ev_val = ev(prob_m, cuota)
            if ev_val >= 0.04:
                kelly_val = kelly(prob_m, cuota)
                rows.append({
                    "hora": hora, "partido": f"{local} vs {visita}",
                    "mercado": label, "prob": prob_m,
                    "cuota": cuota, "bk": bk or "?",
                    "ev": ev_val, "kelly": kelly_val,
                })

    rows.sort(key=lambda x: -x["ev"])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Construir parlays 2 piernas desde Liga MX
# ─────────────────────────────────────────────────────────────────────────────
def build_parlays(value_bets: list[dict], bankroll=1000) -> list[dict]:
    from itertools import combinations
    parlays = []
    for a, b in combinations(value_bets[:6], 2):
        if a["partido"] == b["partido"]:
            continue
        prob_comb = a["prob"] * b["prob"]
        cuota_comb = a["cuota"] * b["cuota"]
        ev_comb   = ev(prob_comb, cuota_comb)
        if ev_comb >= 0.05:
            kelly_comb = kelly(prob_comb, cuota_comb)
            parlays.append({
                "picks": [a, b],
                "prob": round(prob_comb, 3),
                "cuota": round(cuota_comb, 2),
                "ev": ev_comb,
                "kelly": kelly_comb,
                "ganancia": round(kelly_comb * (cuota_comb - 1), 0),
            })
    parlays.sort(key=lambda x: -x["ev"])
    return parlays[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Construir HTML del email
# ─────────────────────────────────────────────────────────────────────────────
EV_COLOR  = {True: "#00C853", False: "#FF6F00"}  # verde / naranja

def _row_color(ev_val):
    if ev_val >= 0.15: return "#1B5E20"
    if ev_val >= 0.08: return "#2E7D32"
    return "#1a1a2e"

def build_html(value_bets: list, intl_picks: list, parlays: list,
               partidos_hoy: list) -> str:
    now_str = NOW_MX.strftime("%A %d %b · %H:%M CST")

    # ── Próximos partidos header ──
    juegos_html = ""
    for p in partidos_hoy:
        h = p.get("_hora_mx","?")
        l = p["local"]
        v = p.get("visitante", p.get("visita",""))
        diff = (p["_dt_mx"] - NOW_MX).total_seconds() / 3600
        tag = f"🔴 EN JUEGO" if -0.25 < diff < 2 else f"en {diff:.1f}h"
        juegos_html += f'<tr><td style="color:#aaa">{h} CST</td><td><b>{l}</b> vs <b>{v}</b></td><td style="color:#FF6F00">{tag}</td></tr>'

    if not juegos_html:
        juegos_html = '<tr><td colspan="3" style="color:#888">Sin partidos próximos de Liga MX</td></tr>'

    # ── Value bets Liga MX ──
    vb_html = ""
    if value_bets:
        for row in value_bets:
            bg = _row_color(row["ev"])
            star = "⭐" if row["ev"] >= 0.12 else ""
            vb_html += f"""
            <tr style="background:{bg}">
              <td style="padding:5px 8px; color:#ddd">{row['hora']}</td>
              <td style="padding:5px 8px"><b>{row['partido']}</b></td>
              <td style="padding:5px 8px; color:#aaa">{row['mercado']}</td>
              <td style="padding:5px 8px; text-align:center">{row['prob']:.0%}</td>
              <td style="padding:5px 8px; text-align:center">{row['cuota']:.2f}<br><span style="font-size:10px;color:#888">{row['bk']}</span></td>
              <td style="padding:5px 8px; text-align:center; color:#00C853; font-weight:bold">{row['ev']:+.1%} {star}</td>
              <td style="padding:5px 8px; text-align:center; color:#FFD740">${row['kelly']:.0f}</td>
            </tr>"""
    else:
        vb_html = '<tr><td colspan="7" style="color:#888;padding:10px">Sin value bets detectados para los partidos de hoy</td></tr>'

    # ── Internacionales ──
    intl_html = ""
    ligas_seen = set()
    for pick in intl_picks:
        liga = pick["liga"]
        if liga not in ligas_seen:
            intl_html += f'<tr style="background:#0d1117"><td colspan="4" style="padding:4px 8px; color:#64B5F6; font-size:11px">{liga}</td></tr>'
            ligas_seen.add(liga)
        star = "⭐" if pick["prob"] >= 0.72 else ""
        intl_html += f"""
        <tr style="background:#12161e">
          <td style="padding:4px 8px; color:#ddd">{pick['fecha']}</td>
          <td style="padding:4px 8px"><b>{pick['local']}</b> vs <b>{pick['visita']}</b></td>
          <td style="padding:4px 8px; color:#aaa">{pick['mercado']}</td>
          <td style="padding:4px 8px; text-align:center; color:#00C853">{pick['prob']:.0%} {star}</td>
        </tr>"""
    if not intl_html:
        intl_html = '<tr><td colspan="4" style="color:#888;padding:10px">Sin picks internacionales disponibles hoy</td></tr>'

    # ── Parlays ──
    parlay_html = ""
    for pl in parlays:
        picks_str = " + ".join(f"{r['partido']} ({r['mercado']})" for r in pl["picks"])
        parlay_html += f"""
        <tr style="background:#1a2a1a">
          <td style="padding:6px 8px; color:#ddd; font-size:12px">{picks_str}</td>
          <td style="padding:6px 8px; text-align:center">{pl['prob']:.0%}</td>
          <td style="padding:6px 8px; text-align:center; font-weight:bold">{pl['cuota']:.2f}</td>
          <td style="padding:6px 8px; text-align:center; color:#00C853; font-weight:bold">{pl['ev']:+.1%}</td>
          <td style="padding:6px 8px; text-align:center; color:#FFD740">${pl['kelly']:.0f} → +${pl['ganancia']:.0f}</td>
        </tr>"""
    if not parlay_html:
        parlay_html = '<tr><td colspan="5" style="color:#888;padding:10px">Sin parlays con EV positivo disponibles</td></tr>'

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
</head>
<body style="margin:0;padding:0;background:#0d1117;color:#e0e0e0;font-family:monospace">
<div style="max-width:680px;margin:0 auto;padding:12px">

  <!-- HEADER -->
  <div style="background:#0a0a1a;padding:12px 16px;border-radius:8px;
              margin-bottom:12px;border-left:4px solid #E53935">
    <div style="color:#E53935;font-size:18px;font-weight:bold">
      🎰 VALUE BETS · MAU-STATISTICS
    </div>
    <div style="color:#888;font-size:11px;margin-top:4px">
      {now_str} · Modelos: Dixon-Coles 1997 · ELO Liga MX
    </div>
  </div>

  <!-- PARTIDOS DE HOY -->
  <div style="background:#111827;padding:12px;border-radius:6px;margin-bottom:12px">
    <div style="color:#64B5F6;font-weight:bold;margin-bottom:8px">📅 Liga MX HOY</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      {juegos_html}
    </table>
  </div>

  <!-- VALUE BETS LIGA MX -->
  <div style="background:#111827;border-radius:6px;margin-bottom:12px;overflow:hidden">
    <div style="background:#1a1a2e;padding:10px 12px;color:#FFD740;font-weight:bold">
      ⚡ VALUE BETS Liga MX — cuotas reales (Pinnacle/FanDuel)
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <tr style="background:#0d1117;color:#888">
        <th style="padding:4px 8px;text-align:left">Hora</th>
        <th style="padding:4px 8px;text-align:left">Partido</th>
        <th style="padding:4px 8px;text-align:left">Mercado</th>
        <th style="padding:4px 8px">ProbM</th>
        <th style="padding:4px 8px">Cuota</th>
        <th style="padding:4px 8px">EV</th>
        <th style="padding:4px 8px">Kelly/$1k</th>
      </tr>
      {vb_html}
    </table>
    <div style="padding:6px 10px;font-size:10px;color:#555">
      EV = (Prob·Cuota − 1) · Kelly fraccionado 25% · Max 3% bankroll
      · ⭐ = EV alto ≥12%
    </div>
  </div>

  <!-- PARLAYS -->
  <div style="background:#111827;border-radius:6px;margin-bottom:12px;overflow:hidden">
    <div style="background:#1a2a1a;padding:10px 12px;color:#A5D6A7;font-weight:bold">
      🎯 PARLAYS combinados — EV positivo
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:11px">
      <tr style="background:#0d1117;color:#888">
        <th style="padding:4px 8px;text-align:left">Combinación</th>
        <th style="padding:4px 8px">Prob</th>
        <th style="padding:4px 8px">Cuota</th>
        <th style="padding:4px 8px">EV</th>
        <th style="padding:4px 8px">Kelly/$1k</th>
      </tr>
      {parlay_html}
    </table>
  </div>

  <!-- INTERNACIONALES -->
  <div style="background:#111827;border-radius:6px;margin-bottom:12px;overflow:hidden">
    <div style="background:#1a1a2e;padding:10px 12px;color:#90CAF9;font-weight:bold">
      🌎 Picks internacionales — prob ≥65% (Brasileirao · Premier · LaLiga · MLS · etc.)
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:11px">
      <tr style="background:#0d1117;color:#888">
        <th style="padding:4px 8px;text-align:left">Fecha</th>
        <th style="padding:4px 8px;text-align:left">Partido</th>
        <th style="padding:4px 8px;text-align:left">Mercado</th>
        <th style="padding:4px 8px">Prob</th>
      </tr>
      {intl_html}
    </table>
    <div style="padding:6px 10px;font-size:10px;color:#555">
      Picks sin cuota real — probabilidades del modelo ELO+Poisson · sin EV calculado
    </div>
  </div>

  <!-- FOOTER -->
  <div style="text-align:center;color:#555;font-size:10px;margin-top:8px">
    MAU-STATISTICS · email automático horario · {TODAY} {NOW_H}<br>
    Modelo: Dixon-Coles (1997) · ELO Liga MX · The Odds API (Pinnacle)<br>
    <span style="color:#333">⚠️ Solo informativo — apuesta con responsabilidad</span>
  </div>

</div>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true",
                        help="Enviar aunque no haya partidos próximos")
    parser.add_argument("--hours-ahead", type=float, default=4.0,
                        help="Solo llamar Odds API si hay partido en N horas (default 4)")
    args = parser.parse_args()

    app_pw = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not app_pw and not args.dry_run:
        print("  [ERROR] GMAIL_APP_PASSWORD no configurado")
        sys.exit(1)

    print(f"\n── send_bets_email.py · {TODAY} {NOW_H} ──")

    # Partidos de hoy (Liga MX)
    partidos_hoy = get_ligamx_today()
    hours_next   = hours_to_next_game(partidos_hoy)
    print(f"  Partidos hoy: {len(partidos_hoy)} · próximo en {hours_next:.1f}h")

    # Si no hay partidos en las próximas N horas y no es --force, salir
    if not args.force and hours_next > args.hours_ahead and not partidos_hoy:
        print(f"  Sin partidos próximos (>{args.hours_ahead}h) · no enviando email")
        return

    # Odds Liga MX (solo si hay partidos próximos — ahorra requests API)
    odds = {}
    if ODDS_API_KEY and (hours_next <= args.hours_ahead or args.force):
        print("  Descargando cuotas Liga MX...")
        odds = fetch_ligamx_odds()
        print(f"  {len(odds)} partidos con cuotas")
    else:
        print("  Sin Odds API key o partidos muy lejanos — sin cuotas")

    # Probabilidades del modelo
    probs = load_model_probs()
    print(f"  Probs modelo: {len(probs)} entradas")

    # Value bets Liga MX
    value_bets = build_ligamx_value_bets(partidos_hoy, odds, probs)
    print(f"  Value bets Liga MX: {len(value_bets)}")

    # Picks internacionales
    intl_picks = load_intl_picks()
    print(f"  Picks internacionales: {len(intl_picks)}")

    # Parlays
    parlays = build_parlays(value_bets)
    print(f"  Parlays con EV+: {len(parlays)}")

    # Construir email
    html = build_html(value_bets, intl_picks, parlays, partidos_hoy)

    n_value = len(value_bets)
    n_intl  = len(intl_picks)
    subject = (f"🎰 BETS {TODAY} {NOW_H} · "
               f"{n_value} value bets Liga MX · {n_intl} picks intl")

    if args.dry_run:
        out = BASE / "output/reports/bets_email_preview.html"
        out.write_text(html, encoding="utf-8")
        print(f"  [dry-run] Preview: {out}")
        print(f"  Asunto: {subject}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = FROM_ADDR
    msg["To"]      = TO_ADDR
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls()
            s.login(FROM_ADDR, app_pw)
            s.send_message(msg)
        print(f"  ✅ Email enviado → {TO_ADDR}")
        print(f"  Asunto: {subject}")
    except Exception as e:
        print(f"  ❌ Error enviando: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
