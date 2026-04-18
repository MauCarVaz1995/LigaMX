#!/usr/bin/env python3
"""
daily_betting_bot.py — Bot autónomo de análisis de value betting
================================================================
Corre cada día sin intervención manual. Analiza todos los partidos de Liga MX
del día/próximos 3 días, calcula EV para todos los mercados disponibles
y genera un reporte HTML + JSON.

Diseñado para correr desde GitHub Actions o cron. No requiere Claude.

Científico:
  - Corners: Poisson bivariado MLE (Dixon-Coles + Rue-Salvesen)
  - BTTS/Goles: Poisson independiente con corrección Dixon-Coles (ρ=-0.13)
  - Tarjetas: Poisson simple con card_rate MLE + factor árbitro/rivalidad
  - "Feeling": Liga MX Knowledge Base (rivalidades, altitud, fase, estadio)

Bankroll management: Kelly fraccionado 25%, máx 3% por apuesta

Outputs:
  output/reports/betting_{fecha}.json   — datos estructurados para otros scripts
  output/reports/betting_{fecha}.html   — reporte visual listo para email

Uso:
  python bots/daily_betting_bot.py                    # próximos 3 días
  python bots/daily_betting_bot.py --date 2026-04-20  # fecha específica
  python bots/daily_betting_bot.py --days 1           # solo hoy
  python bots/daily_betting_bot.py --retrain          # re-entrena modelos antes
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))

from modelo_corners  import predecir_corners
from modelo_tarjetas import predecir_tarjetas
from modelo_btts     import predecir_btts

HIST_DIR    = BASE / "data/raw/historico"
ELO_CSV     = BASE / "data/processed/elo_historico.csv"
REPORTS_DIR = BASE / "output/reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Líneas estándar de Liga MX (casas más comunes)
DEFAULT_LINES = {
    "corners": [8.5, 9.5, 10.5],
    "tarjetas": [3.5, 4.5, 5.5],
}

# EV mínimo para marcar como VALUE BET
EV_THRESHOLD = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Carga de fixtures pendientes del historico
# ─────────────────────────────────────────────────────────────────────────────

def get_fixtures(target_dates: list[str]) -> list[dict]:
    """
    Lee historico_clausura_2026.json y devuelve partidos pendientes
    en las fechas pedidas.
    """
    fixtures = []
    hist_file = HIST_DIR / "historico_clausura_2026.json"
    if not hist_file.exists():
        return []

    with open(hist_file) as f:
        d = json.load(f)

    for p in d["partidos"]:
        if p.get("terminado"):
            continue
        fecha = p["fecha"][:10]
        if fecha in target_dates:
            fixtures.append({
                "match_id":  p.get("id"),
                "fecha":     fecha,
                "jornada":   int(p.get("jornada", 0)),
                "local":     p["local"],
                "visita":    p["visitante"],
                "torneo":    "Liga MX Clausura 2026",
            })

    return fixtures


def get_latest_elo(equipo: str) -> float:
    try:
        df = pd.read_csv(ELO_CSV)
        rows = df[df["equipo"] == equipo]
        if len(rows):
            return float(rows.sort_values("fecha")["elo"].iloc[-1])
    except Exception:
        pass
    return 1500.0


# ─────────────────────────────────────────────────────────────────────────────
# Análisis por partido
# ─────────────────────────────────────────────────────────────────────────────

def analizar_partido(fix: dict) -> dict:
    """
    Corre los tres modelos sobre un partido y devuelve análisis completo.
    """
    local   = fix["local"]
    visita  = fix["visita"]
    jornada = fix["jornada"]

    result = {
        "match_id": fix.get("match_id"),
        "fecha":    fix["fecha"],
        "jornada":  jornada,
        "local":    local,
        "visita":   visita,
        "torneo":   fix.get("torneo", ""),
        "elo_local":  get_latest_elo(local),
        "elo_visita": get_latest_elo(visita),
        "corners": {},
        "tarjetas": {},
        "btts": {},
        "value_bets": [],
        "errors": [],
    }

    # ── Corners ──────────────────────────────────────────────────────────────
    try:
        c = predecir_corners(local, visita, jornada=jornada)
        for linea in DEFAULT_LINES["corners"]:
            over_key  = f"prob_over_{linea}"
            under_key = f"prob_under_{linea}"
            result["corners"][f"over_{linea}"]   = c.get(over_key)
            result["corners"][f"under_{linea}"]  = c.get(under_key)
        result["corners"]["lambda_total"]  = c["lambda_total"]
        result["corners"]["lambda_local"]  = c["lambda_local"]
        result["corners"]["lambda_visita"] = c["lambda_visita"]
        result["corners"]["notes"]         = c.get("notes", [])
    except Exception as e:
        result["errors"].append(f"corners: {e}")

    # ── Tarjetas ─────────────────────────────────────────────────────────────
    try:
        t = predecir_tarjetas(local, visita, jornada=jornada)
        for linea in DEFAULT_LINES["tarjetas"]:
            result["tarjetas"][f"over_{linea}"]  = t.get(f"prob_over_{linea}")
            result["tarjetas"][f"under_{linea}"] = t.get(f"prob_under_{linea}")
        result["tarjetas"]["lambda_total"] = t["lambda_total"]
        result["tarjetas"]["cr_local"]     = t["cr_local"]
        result["tarjetas"]["cr_visita"]    = t["cr_visita"]
        result["tarjetas"]["rival_bonus"]  = t["rival_bonus"]
    except Exception as e:
        result["errors"].append(f"tarjetas: {e}")

    # ── BTTS & Goles ─────────────────────────────────────────────────────────
    try:
        b = predecir_btts(local, visita,
                          elo_local=result["elo_local"],
                          elo_visita=result["elo_visita"])
        result["btts"] = {
            "btts_si":       b["p_btts_si"],
            "btts_no":       b["p_btts_no"],
            "over_1.5":      b["p_over_1.5"],
            "over_2.5":      b["p_over_2.5"],
            "under_2.5":     b["p_under_2.5"],
            "btts_over_2.5": b["p_btts_over_2.5"],
            "lambda_local":  b["lambda_local"],
            "lambda_visita": b["lambda_visita"],
        }
    except Exception as e:
        result["errors"].append(f"btts: {e}")

    return result


def calcular_ev_con_cuotas(analisis: dict, odds: dict) -> dict:
    """
    Toma las probabilidades del modelo y las cuotas de la casa y agrega EVs.
    odds = {"corners_over_9.5": 1.90, "corners_under_9.5": 1.90, ...}
    """
    bets = []

    for mercado, cuota in odds.items():
        prob = None
        label = mercado

        if mercado.startswith("corners_over_"):
            linea = float(mercado.split("_")[-1])
            prob  = analisis["corners"].get(f"over_{linea}")
        elif mercado.startswith("corners_under_"):
            linea = float(mercado.split("_")[-1])
            prob  = analisis["corners"].get(f"under_{linea}")
        elif mercado.startswith("tarjetas_over_"):
            linea = float(mercado.split("_")[-1])
            prob  = analisis["tarjetas"].get(f"over_{linea}")
        elif mercado.startswith("tarjetas_under_"):
            linea = float(mercado.split("_")[-1])
            prob  = analisis["tarjetas"].get(f"under_{linea}")
        elif mercado == "btts_si":
            prob  = analisis["btts"].get("btts_si")
        elif mercado == "btts_no":
            prob  = analisis["btts"].get("btts_no")
        elif mercado == "over_2.5":
            prob  = analisis["btts"].get("over_2.5")
        elif mercado == "under_2.5":
            prob  = analisis["btts"].get("under_2.5")

        if prob is None:
            continue

        ev = round(prob * cuota - 1, 4)
        bets.append({
            "mercado":  label,
            "prob":     round(prob, 3),
            "cuota":    cuota,
            "ev":       ev,
            "value":    ev >= EV_THRESHOLD,
            "tag":      "✅ VALUE" if ev >= EV_THRESHOLD else ("⚠️" if ev >= 0 else "❌"),
        })

    analisis["value_bets"].extend(bets)
    return analisis


# ─────────────────────────────────────────────────────────────────────────────
# Kelly fraccionado (bankroll management)
# ─────────────────────────────────────────────────────────────────────────────

def kelly_bet(prob: float, cuota: float, bankroll: float,
              fraccion: float = 0.25, max_pct: float = 0.03) -> float:
    """
    Kelly fraccionado.
    Referencia: Kelly (1956), aplicado a apuestas deportivas en
    Thorp (2008) "The Kelly Criterion in Blackjack, Sports Betting..."
    fraccion=0.25 → reduce varianza al 6.25% del Kelly óptimo.
    max_pct=0.03 → máximo 3% del bankroll por apuesta.
    """
    b  = cuota - 1
    p  = prob
    q  = 1 - p
    kf = (b * p - q) / b
    if kf <= 0:
        return 0.0
    bet = bankroll * kf * fraccion
    return round(min(bet, bankroll * max_pct), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Generación de reportes
# ─────────────────────────────────────────────────────────────────────────────

def generar_reporte_json(partidos: list[dict], fecha_str: str) -> Path:
    out = REPORTS_DIR / f"betting_{fecha_str}.json"
    out.write_text(json.dumps(partidos, ensure_ascii=False, indent=2))
    return out


def generar_reporte_html(partidos: list[dict], fecha_str: str) -> Path:
    """
    Genera HTML limpio listo para incluir en el email diario.
    """
    rows = []
    for p in partidos:
        local  = p["local"]
        visita = p["visita"]
        j      = p["jornada"]

        c   = p.get("corners", {})
        t   = p.get("tarjetas", {})
        b   = p.get("btts", {})
        vbs = [vb for vb in p.get("value_bets", []) if vb["value"]]
        notes = c.get("notes", [])

        # Tabla de probabilidades
        probs_html = f"""
        <table style="border-collapse:collapse; font-size:12px; width:100%;">
          <tr style="background:#1a1a2e; color:#e0e0e0;">
            <th style="padding:4px 8px; text-align:left;">Mercado</th>
            <th style="padding:4px 8px; text-align:center;">Prob</th>
            <th style="padding:4px 8px; text-align:center;">λ</th>
          </tr>
          <tr><td style="padding:3px 8px; color:#aaa;">Corners Over 8.5</td>
              <td style="text-align:center;">{c.get("over_8.5", 0):.0%}</td>
              <td style="text-align:center;" rowspan="3">{c.get("lambda_total","?")}</td></tr>
          <tr><td style="padding:3px 8px; color:#aaa;">Corners Over 9.5</td>
              <td style="text-align:center;">{c.get("over_9.5", 0):.0%}</td></tr>
          <tr><td style="padding:3px 8px; color:#aaa;">Corners Over 10.5</td>
              <td style="text-align:center;">{c.get("over_10.5", 0):.0%}</td></tr>
          <tr style="background:#111;"><td style="padding:3px 8px; color:#aaa;">Tarjetas Over 3.5</td>
              <td style="text-align:center;">{t.get("over_3.5", 0):.0%}</td>
              <td style="text-align:center;" rowspan="3">{t.get("lambda_total","?")}</td></tr>
          <tr style="background:#111;"><td style="padding:3px 8px; color:#aaa;">Tarjetas Over 4.5</td>
              <td style="text-align:center;">{t.get("over_4.5", 0):.0%}</td></tr>
          <tr style="background:#111;"><td style="padding:3px 8px; color:#aaa;">Tarjetas Over 5.5</td>
              <td style="text-align:center;">{t.get("over_5.5", 0):.0%}</td></tr>
          <tr><td style="padding:3px 8px; color:#aaa;">BTTS Sí</td>
              <td style="text-align:center;">{b.get("btts_si", 0):.0%}</td>
              <td style="text-align:center;">{b.get("lambda_local","?")}/{b.get("lambda_visita","?")}</td></tr>
          <tr><td style="padding:3px 8px; color:#aaa;">Over 2.5 goles</td>
              <td style="text-align:center;">{b.get("over_2.5", 0):.0%}</td>
              <td></td></tr>
        </table>"""

        notes_html = ""
        if notes:
            notes_html = "<br>".join(f"<span style='color:#FFA726;font-size:11px;'>📌 {n}</span>"
                                     for n in notes)
            notes_html = f"<div style='margin-top:4px;'>{notes_html}</div>"

        vb_html = ""
        if vbs:
            vb_rows = "".join(
                f"<tr style='background:#1B5E20;'>"
                f"<td style='padding:3px 8px;'>{vb['mercado']}</td>"
                f"<td style='text-align:center;'>{vb['prob']:.0%}</td>"
                f"<td style='text-align:center;'>×{vb['cuota']}</td>"
                f"<td style='text-align:center; color:#69F0AE;'>{vb['ev']:+.1%}</td>"
                f"</tr>"
                for vb in vbs
            )
            vb_html = f"""
            <div style='margin-top:8px;'>
              <div style='color:#69F0AE; font-weight:bold; font-size:12px;'>🎯 VALUE BETS DETECTADAS</div>
              <table style='border-collapse:collapse; font-size:12px; width:100%; color:#e0e0e0;'>
                <tr style='background:#0a3d1a; color:#81C784;'>
                  <th style='padding:3px 8px; text-align:left;'>Mercado</th>
                  <th style='padding:3px 8px;'>Prob</th>
                  <th style='padding:3px 8px;'>Cuota</th>
                  <th style='padding:3px 8px;'>EV</th>
                </tr>
                {vb_rows}
              </table>
            </div>"""

        rows.append(f"""
        <div style="background:#0d0d1e; border:1px solid #333; border-radius:8px;
                    padding:12px; margin-bottom:12px; color:#e0e0e0; font-family:monospace;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <span style="color:#E53935; font-size:15px; font-weight:bold;">
                {local}</span>
              <span style="color:#666; font-size:13px;"> vs </span>
              <span style="color:#1E88E5; font-size:15px; font-weight:bold;">
                {visita}</span>
            </div>
            <div style="color:#888; font-size:11px;">J{j} · {p["fecha"]}</div>
          </div>
          {notes_html}
          <div style="margin-top:8px;">{probs_html}</div>
          {vb_html}
        </div>""")

    has_value = any(vb["value"] for p in partidos for vb in p.get("value_bets", []))
    banner = ""
    if has_value:
        n_vbs = sum(1 for p in partidos for vb in p.get("value_bets", []) if vb["value"])
        banner = f"""<div style="background:#1B5E20; padding:10px; border-radius:6px;
                                 text-align:center; margin-bottom:16px;
                                 color:#69F0AE; font-size:14px; font-weight:bold;">
            🎯 {n_vbs} VALUE BET{'S' if n_vbs > 1 else ''} DETECTADA{'S' if n_vbs > 1 else ''} HOY
            &nbsp;|&nbsp; EV mínimo requerido: {EV_THRESHOLD:.0%}
        </div>"""

    html = f"""
    <div style="font-family:monospace; max-width:640px; margin:0 auto;">
      <div style="background:#0a0a1a; padding:12px 16px; border-radius:8px;
                  margin-bottom:16px; border-left:4px solid #E53935;">
        <div style="color:#E53935; font-size:16px; font-weight:bold;">
          🎰 ANÁLISIS BETTING — Liga MX · {fecha_str}
        </div>
        <div style="color:#888; font-size:11px; margin-top:4px;">
          Modelos: Poisson MLE (Dixon-Coles 1997) + Rue-Salvesen (2000) + Liga MX Knowledge Base
        </div>
      </div>
      {banner}
      {"".join(rows) if rows else '<p style="color:#888;">Sin partidos para esta fecha.</p>'}
    </div>"""

    out = REPORTS_DIR / f"betting_{fecha_str}.html"
    out.write_text(html)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bot autónomo de value betting Liga MX")
    parser.add_argument("--date",    help="Fecha YYYY-MM-DD (default: hoy)")
    parser.add_argument("--days",    type=int, default=3, help="Días a adelante (default 3)")
    parser.add_argument("--retrain", action="store_true", help="Re-entrenar modelos antes")
    parser.add_argument("--odds",    help="JSON con cuotas: {\"corners_over_9.5\":1.90,...}")
    args = parser.parse_args()

    if args.retrain:
        print("── Re-entrenando modelos ──")
        from modelo_corners  import CornersModel
        from modelo_tarjetas import TarjetasModel
        CornersModel().fit(verbose=True)
        TarjetasModel().fit(verbose=True)

    # Fechas a analizar
    start = date.fromisoformat(args.date) if args.date else date.today()
    target_dates = [(start + timedelta(days=i)).isoformat() for i in range(args.days)]

    print(f"\n── daily_betting_bot.py · fechas: {target_dates[0]} → {target_dates[-1]} ──")

    fixtures = get_fixtures(target_dates)
    print(f"  {len(fixtures)} partidos encontrados\n")

    if not fixtures:
        print("  Sin partidos pendientes. Saliendo.")
        return

    # Cuotas opcionales (para EV)
    odds = {}
    if args.odds:
        try:
            odds = json.loads(args.odds)
        except json.JSONDecodeError as e:
            print(f"  [warn] --odds JSON inválido: {e}")

    # Analizar partidos
    partidos = []
    for fix in fixtures:
        print(f"  Analizando: {fix['local']} vs {fix['visita']} ({fix['fecha']})", end=" ")
        try:
            an = analizar_partido(fix)
            if odds:
                an = calcular_ev_con_cuotas(an, odds)
            partidos.append(an)
            n_vbs = sum(1 for vb in an.get("value_bets", []) if vb.get("value"))
            print(f"✓ corners={an['corners'].get('lambda_total','?')} "
                  f"{'| 🎯 ' + str(n_vbs) + ' VALUE BETS' if n_vbs else ''}")
        except Exception as e:
            print(f"✗ {e}")

    # Reportes
    fecha_str = target_dates[0] if len(target_dates) == 1 else f"{target_dates[0]}_to_{target_dates[-1]}"
    json_out = generar_reporte_json(partidos, fecha_str)
    html_out = generar_reporte_html(partidos, fecha_str)

    print(f"\n  📊 JSON: {json_out}")
    print(f"  📄 HTML: {html_out}")

    # Resumen de value bets
    all_vbs = [vb for p in partidos for vb in p.get("value_bets", []) if vb.get("value")]
    if all_vbs:
        print(f"\n  {'═'*50}")
        print(f"  🎯 {len(all_vbs)} VALUE BETS DETECTADAS")
        print(f"  {'═'*50}")
        for vb in sorted(all_vbs, key=lambda x: x["ev"], reverse=True):
            print(f"    {vb['mercado']:<30} prob={vb['prob']:.0%} cuota={vb['cuota']} EV={vb['ev']:+.1%}")
    else:
        print("\n  Sin value bets con las cuotas proporcionadas.")
        print("  Usa --odds '{\"corners_over_9.5\":1.90,...}' para calcular EV.")


if __name__ == "__main__":
    main()
