#!/usr/bin/env python3
"""
audit_bot.py — Bot auditor del sistema MAU-STATISTICS
======================================================
Valida que todos los pipelines, modelos y bots estén funcionando correctamente.
Detecta problemas silenciosos que no fallan el CI pero producen datos incorrectos.

Qué audita:
  1. DATOS     — CSVs actualizados, sin gaps, sin valores imposibles
  2. MODELOS   — Brier scores dentro de rango, modelos no obsoletos
  3. PIPELINE  — Último run de GitHub Actions exitoso y reciente
  4. TRACKER   — Predicciones siendo registradas y resueltas
  5. BANKROLL  — Historial de apuestas (si existe), ROI, drawdown
  6. BOTS      — daily_betting_bot y retrain_bot corrieron recientemente

Output:
  output/reports/audit_{fecha}.json   — datos crudos
  output/reports/audit_{fecha}.html   — reporte visual para email

Uso:
  python bots/audit_bot.py            # auditoría completa
  python bots/audit_bot.py --section datos   # solo una sección
  python bots/audit_bot.py --verbose         # más detalle en stdout

Diseñado para correr al final del daily_pipeline.yml. Si hay problemas
críticos, sale con código 1 para que GitHub Actions marque el step como warning.
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE        = Path(__file__).resolve().parent.parent
SCRIPTS     = BASE / "scripts"
REPORTS_DIR = BASE / "output/reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds de alerta
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "max_days_since_match_events_update": 3,   # alerta si match_events no se actualiza en 3 días
    "max_days_since_elo_update":          2,
    "max_days_since_model_training":      14,  # re-entrenar cada 2 semanas mínimo
    "min_partidos_corners_model":         50,  # modelo no confiable con menos de 50
    "brier_corners_warning":              0.23, # peor que esto → warning
    "brier_corners_critical":             0.26, # peor que baseline → error
    "brier_tarjetas_warning":             0.25,
    "min_predictions_to_validate":        50,   # mínimo para reportar hit rate
    "max_drawdown_warning":               0.15, # 15% drawdown → warning
    "max_drawdown_critical":              0.25, # 25% → error
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def days_since(filepath: Path) -> int:
    """Días desde la última modificación del archivo."""
    if not filepath.exists():
        return 9999
    mtime = datetime.fromtimestamp(filepath.stat().st_mtime).date()
    return (date.today() - mtime).days


def status(ok: bool, warning: bool = False) -> str:
    if ok:      return "✅ OK"
    if warning: return "⚠️ WARN"
    return "❌ ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# Sección 1 — DATOS
# ─────────────────────────────────────────────────────────────────────────────

def audit_datos(verbose: bool = False) -> dict:
    results = {"section": "datos", "checks": [], "status": "ok"}

    def check(name: str, ok: bool, warn: bool, msg: str):
        level = "ok" if ok else ("warn" if warn else "error")
        if level == "error" and results["status"] != "error":
            results["status"] = "error"
        elif level == "warn" and results["status"] == "ok":
            results["status"] = "warn"
        results["checks"].append({"name": name, "level": level, "msg": msg})
        if verbose:
            print(f"  {status(ok, warn)} {name}: {msg}")

    # match_events.csv
    me = BASE / "data/processed/match_events.csv"
    if me.exists():
        df = pd.read_csv(me)
        n  = len(df)
        d  = days_since(me)
        pct_corners = df["corners_total"].notna().mean()
        pct_xg      = df["xg_local"].notna().mean()
        bad_corners = (df["corners_total"] > 40).sum()

        check("match_events.csv existe",   True,  False, f"{n} partidos")
        check("match_events actualizado",  d <= THRESHOLDS["max_days_since_match_events_update"],
              d <= THRESHOLDS["max_days_since_match_events_update"] + 2,
              f"hace {d} días")
        check("corners sin nulos",         pct_corners > 0.95, pct_corners > 0.85,
              f"{pct_corners:.0%} completo")
        check("xG sin nulos",              pct_xg > 0.80, pct_xg > 0.60,
              f"{pct_xg:.0%} completo")
        check("corners sin valores imposibles", bad_corners == 0, bad_corners < 3,
              f"{bad_corners} filas con >40 corners")
    else:
        check("match_events.csv existe", False, False, "archivo no encontrado")

    # elo_historico.csv
    elo_csv = BASE / "data/processed/elo_historico.csv"
    if elo_csv.exists():
        df_elo = pd.read_csv(elo_csv)
        d = days_since(elo_csv)
        n_equipos = df_elo["equipo"].nunique()
        # ELOs fuera de rango (Liga MX típicamente 1300-1900)
        last_elos = df_elo.groupby("equipo")["elo"].last()
        bad_elos  = ((last_elos < 1100) | (last_elos > 2100)).sum()
        check("elo_historico actualizado", d <= THRESHOLDS["max_days_since_elo_update"],
              d <= 5, f"hace {d} días, {n_equipos} equipos")
        check("ELOs en rango razonable", bad_elos == 0, bad_elos < 3,
              f"{bad_elos} equipos con ELO fuera de rango")
    else:
        check("elo_historico.csv existe", False, False, "archivo no encontrado")

    # predicciones_log.csv
    log_csv = BASE / "data/processed/predicciones_log.csv"
    if log_csv.exists():
        df_log = pd.read_csv(log_csv)
        n_total    = len(df_log)
        n_resolved = df_log["resultado_real"].notna().sum() if "resultado_real" in df_log.columns else 0
        d = days_since(log_csv)
        check("predicciones_log actualizado", d <= 3, d <= 7,
              f"{n_total} total, {n_resolved} resueltas, hace {d} días")
    else:
        check("predicciones_log existe", False, False, "sin tracker de predicciones")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sección 2 — MODELOS
# ─────────────────────────────────────────────────────────────────────────────

def audit_modelos(verbose: bool = False) -> dict:
    results = {"section": "modelos", "checks": [], "status": "ok"}

    def check(name: str, ok: bool, warn: bool, msg: str):
        level = "ok" if ok else ("warn" if warn else "error")
        if level == "error": results["status"] = "error"
        elif level == "warn" and results["status"] == "ok": results["status"] = "warn"
        results["checks"].append({"name": name, "level": level, "msg": msg})
        if verbose:
            print(f"  {status(ok, warn)} {name}: {msg}")

    # corners_model.json
    cm = BASE / "data/processed/corners_model.json"
    if cm.exists():
        m = json.loads(cm.read_text())
        n      = m.get("n_partidos", 0)
        d_train= days_since(cm)
        n_teams= len(m.get("teams", []))
        converged = m.get("converged", True)
        check("corners_model existe",      True, False, f"{n} partidos, {n_teams} equipos")
        check("corners_model tiene datos suficientes",
              n >= THRESHOLDS["min_partidos_corners_model"],
              n >= 30, f"{n} partidos entrenados")
        check("corners_model MLE convergió", converged, True,
              "✅ convergido" if converged else "MLE no convergió completamente")
        check("corners_model no obsoleto",
              d_train <= THRESHOLDS["max_days_since_model_training"],
              d_train <= 21, f"entrenado hace {d_train} días")
    else:
        check("corners_model.json existe", False, False, "ejecutar: python bots/retrain_bot.py --force")

    # retrain_log.json — Brier scores
    rl = BASE / "data/processed/retrain_log.json"
    if rl.exists():
        log = json.loads(rl.read_text())
        cm_brier = log.get("corners_metrics", {})
        tm_brier = log.get("tarjetas_metrics", {})
        baseline = 0.25

        b_c95 = cm_brier.get("brier_over_9.5")
        if b_c95 is not None:
            check("Brier corners Over 9.5",
                  b_c95 < THRESHOLDS["brier_corners_warning"],
                  b_c95 < THRESHOLDS["brier_corners_critical"],
                  f"{b_c95:.4f} (baseline={baseline})")

        b_t45 = tm_brier.get("brier_over_4.5")
        if b_t45 is not None:
            check("Brier tarjetas Over 4.5",
                  b_t45 < THRESHOLDS["brier_tarjetas_warning"],
                  b_t45 < baseline + 0.05,
                  f"{b_t45:.4f} (baseline={baseline})")

        mae = cm_brier.get("mae_total_corners")
        if mae is not None:
            check("MAE corners totales", mae < 3.0, mae < 4.0,
                  f"{mae:.2f} córners promedio")
    else:
        check("retrain_log existe", False, False,
              "ejecutar: python bots/retrain_bot.py --force")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sección 3 — BOTS (¿corrieron recientemente?)
# ─────────────────────────────────────────────────────────────────────────────

def audit_bots(verbose: bool = False) -> dict:
    results = {"section": "bots", "checks": [], "status": "ok"}

    def check(name: str, ok: bool, warn: bool, msg: str):
        level = "ok" if ok else ("warn" if warn else "error")
        if level == "error": results["status"] = "error"
        elif level == "warn" and results["status"] == "ok": results["status"] = "warn"
        results["checks"].append({"name": name, "level": level, "msg": msg})
        if verbose:
            print(f"  {status(ok, warn)} {name}: {msg}")

    # daily_betting_bot — ¿hay un reporte reciente?
    reports = sorted(REPORTS_DIR.glob("betting_*.json"))
    if reports:
        latest = reports[-1]
        d = days_since(latest)
        data = json.loads(latest.read_text())
        n_partidos = len(data)
        n_vbs = sum(1 for p in data for vb in p.get("value_bets", []) if vb.get("value"))
        check("daily_betting_bot corrió", d <= 2, d <= 3,
              f"hace {d} días — {n_partidos} partidos analizados, {n_vbs} value bets")
    else:
        check("daily_betting_bot corrió", False, False,
              "sin reportes de betting generados")

    # retrain_bot — ¿corrió alguna vez?
    rl = BASE / "data/processed/retrain_log.json"
    if rl.exists():
        log   = json.loads(rl.read_text())
        d_log = (date.today() - date.fromisoformat(log.get("last_train_date", "2000-01-01"))).days
        n_p   = log.get("n_partidos_at_train", 0)
        check("retrain_bot tiene log", True, False,
              f"último entrenamiento hace {d_log} días con {n_p} partidos")
        check("retrain_bot no obsoleto", d_log <= 14, d_log <= 21,
              f"hace {d_log} días" if d_log <= 14 else f"hace {d_log} días — re-entrenar")
    else:
        check("retrain_bot tiene log", True, True,
              "aún sin log — se generará en el próximo retrain_bot.py")

    # daily_pipeline summary
    summary_f = BASE / "logs/daily_summary.json"
    if summary_f.exists():
        d = days_since(summary_f)
        try:
            summary = json.loads(summary_f.read_text())
            pipeline_ok = summary.get("success", False)
            elapsed     = summary.get("elapsed_s", "?")
            check("daily_pipeline corrió", d <= 2, d <= 3,
                  f"hace {d} días, {'✅ OK' if pipeline_ok else '❌ FALLO'} ({elapsed}s)")
        except Exception:
            check("daily_pipeline log legible", False, True,
                  "JSON corrupto")
    else:
        check("daily_pipeline summary", False, False,
              "sin logs/daily_summary.json")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sección 4 — TRACKER DE PREDICCIONES Y PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def audit_tracker(verbose: bool = False) -> dict:
    results = {"section": "tracker", "checks": [], "status": "ok",
               "stats": {}}

    def check(name: str, ok: bool, warn: bool, msg: str):
        level = "ok" if ok else ("warn" if warn else "error")
        if level == "error": results["status"] = "error"
        elif level == "warn" and results["status"] == "ok": results["status"] = "warn"
        results["checks"].append({"name": name, "level": level, "msg": msg})
        if verbose:
            print(f"  {status(ok, warn)} {name}: {msg}")

    log_csv = BASE / "data/processed/predicciones_log.csv"
    if not log_csv.exists():
        check("predicciones_log existe", False, False, "sin tracker")
        return results

    df = pd.read_csv(log_csv)

    # Stats básicas
    n_total = len(df)
    results["stats"]["n_predicciones"] = n_total

    if "resultado_real" in df.columns:
        df_eval = df[df["resultado_real"].notna()]
        n_eval  = len(df_eval)
        results["stats"]["n_evaluadas"] = n_eval
        check("predicciones evaluadas", n_eval >= 20, n_eval >= 5,
              f"{n_eval}/{n_total} resueltas")

        if n_eval >= 5 and "acierto" in df_eval.columns:
            hit_rate = df_eval["acierto"].mean()
            results["stats"]["hit_rate_1x2"] = round(hit_rate, 3)
            check("hit rate 1X2 > 35%", hit_rate > 0.35, hit_rate > 0.28,
                  f"{hit_rate:.0%} ({n_eval} evaluadas) — baseline = 33%")

    # Predicciones recientes (últimos 7 días)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=7)
        n_recientes = (df["fecha"] >= cutoff).sum()
        check("predicciones recientes", n_recientes >= 3, n_recientes >= 1,
              f"{n_recientes} predicciones en los últimos 7 días")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sección 5 — BANKROLL (si existe el tracker de apuestas reales)
# ─────────────────────────────────────────────────────────────────────────────

def audit_bankroll(verbose: bool = False) -> dict:
    """Audita el historial de apuestas reales si existe."""
    results = {"section": "bankroll", "checks": [], "status": "ok", "stats": {}}

    def check(name: str, ok: bool, warn: bool, msg: str):
        level = "ok" if ok else ("warn" if warn else "error")
        if level == "error": results["status"] = "error"
        elif level == "warn" and results["status"] == "ok": results["status"] = "warn"
        results["checks"].append({"name": name, "level": level, "msg": msg})
        if verbose:
            print(f"  {status(ok, warn)} {name}: {msg}")

    bk_csv = BASE / "data/processed/bankroll_log.csv"
    if not bk_csv.exists():
        check("bankroll_log", True, False,
              "sin apuestas reales aún — correcto hasta validar 200 predicciones por mercado")
        return results

    df = pd.read_csv(bk_csv)
    required_cols = ["fecha", "mercado", "prob_modelo", "cuota", "resultado", "p_l"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        check("bankroll_log columnas", False, False,
              f"columnas faltantes: {missing}")
        return results

    n         = len(df)
    total_pl  = df["p_l"].sum()
    roi       = total_pl / df["cuota"].count() if n > 0 else 0

    # Drawdown máximo
    cumpl  = df["p_l"].cumsum()
    peak   = cumpl.cummax()
    dd     = ((cumpl - peak) / (peak.abs() + 1e-6)).min()

    results["stats"] = {
        "n_apuestas":  n,
        "total_pl":    round(total_pl, 2),
        "roi":         round(roi, 4),
        "max_drawdown": round(dd, 4),
    }

    check("ROI positivo", roi > 0, roi > -0.05, f"{roi:+.1%}")
    check("Drawdown controlado",
          abs(dd) < THRESHOLDS["max_drawdown_warning"],
          abs(dd) < THRESHOLDS["max_drawdown_critical"],
          f"max drawdown: {dd:.1%}")

    if verbose:
        print(f"  Apuestas: {n} | P&L: {total_pl:+.2f}u | ROI: {roi:+.1%}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Generación de reportes
# ─────────────────────────────────────────────────────────────────────────────

def generar_reporte(sections: list[dict]) -> tuple[Path, Path]:
    overall = "ok"
    for s in sections:
        if s["status"] == "error":
            overall = "error"
        elif s["status"] == "warn" and overall == "ok":
            overall = "warn"

    data = {
        "fecha":    TODAY,
        "overall":  overall,
        "sections": sections,
    }

    json_out = REPORTS_DIR / f"audit_{TODAY}.json"
    json_out.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # HTML
    STATUS_COLORS = {"ok": "#1B5E20", "warn": "#E65100", "error": "#B71C1C"}
    STATUS_ICONS  = {"ok": "✅", "warn": "⚠️", "error": "❌"}

    section_blocks = ""
    for s in sections:
        color   = STATUS_COLORS.get(s["status"], "#333")
        icon    = STATUS_ICONS.get(s["status"], "?")
        checks  = s.get("checks", [])
        stats   = s.get("stats", {})

        rows = "".join(
            "<tr style='background:{bg};'>"
            "<td style='padding:3px 10px;color:#eee;font-size:12px'>{icon} {name}</td>"
            "<td style='padding:3px 10px;color:#ccc;font-size:12px'>{msg}</td>"
            "</tr>".format(
                bg=STATUS_COLORS.get(c["level"], "#111"),
                icon=STATUS_ICONS.get(c["level"], "?"),
                name=c["name"], msg=c["msg"]
            )
            for c in checks
        )

        stats_html = ""
        if stats:
            stat_items = " &nbsp;|&nbsp; ".join(f"<b>{k}</b>: {v}" for k, v in stats.items())
            stats_html = f"<div style='color:#90CAF9;font-size:11px;margin-top:6px;'>{stat_items}</div>"

        section_blocks += f"""
        <div style='background:#111;border-left:3px solid {color};
                    padding:10px;margin-bottom:10px;border-radius:4px;'>
          <div style='color:{color};font-weight:bold;font-size:13px;margin-bottom:6px;'>
            {icon} {s["section"].upper()}</div>
          <table style='width:100%;border-collapse:collapse;'>{rows}</table>
          {stats_html}
        </div>"""

    overall_color = STATUS_COLORS.get(overall, "#333")
    overall_icon  = STATUS_ICONS.get(overall, "?")
    overall_label = {"ok": "SISTEMA OPERATIVO", "warn": "REVISIÓN NECESARIA",
                     "error": "ERROR CRÍTICO"}[overall]

    html = f"""
    <div style='font-family:monospace;max-width:640px;'>
      <div style='background:{overall_color};padding:12px 16px;border-radius:8px;
                  margin-bottom:12px;'>
        <div style='color:#fff;font-size:16px;font-weight:bold;'>
          {overall_icon} AUDIT BOT — {overall_label}</div>
        <div style='color:rgba(255,255,255,0.7);font-size:11px;margin-top:4px;'>
          MAU-STATISTICS · {TODAY} · {sum(len(s["checks"]) for s in sections)} checks
        </div>
      </div>
      {section_blocks}
    </div>"""

    html_out = REPORTS_DIR / f"audit_{TODAY}.html"
    html_out.write_text(html)
    return json_out, html_out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

SECTIONS_MAP = {
    "datos":    audit_datos,
    "modelos":  audit_modelos,
    "bots":     audit_bots,
    "tracker":  audit_tracker,
    "bankroll": audit_bankroll,
}

def main():
    parser = argparse.ArgumentParser(description="Audit bot — MAU-STATISTICS")
    parser.add_argument("--section", choices=list(SECTIONS_MAP.keys()),
                        help="Auditar solo una sección")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-html", action="store_true",
                        help="No generar HTML (solo JSON)")
    args = parser.parse_args()

    print(f"\n{'═'*55}")
    print(f"  audit_bot.py · {TODAY}")
    print(f"{'═'*55}\n")

    sections_to_run = ([args.section] if args.section
                       else list(SECTIONS_MAP.keys()))

    results = []
    for name in sections_to_run:
        if args.verbose:
            print(f"── {name.upper()} ──")
        r = SECTIONS_MAP[name](verbose=args.verbose)
        results.append(r)
        if not args.verbose:
            n_err  = sum(1 for c in r["checks"] if c["level"] == "error")
            n_warn = sum(1 for c in r["checks"] if c["level"] == "warn")
            n_ok   = sum(1 for c in r["checks"] if c["level"] == "ok")
            icon   = "❌" if n_err else ("⚠️" if n_warn else "✅")
            print(f"  {icon} {name:<12} {n_ok}✅ {n_warn}⚠️  {n_err}❌")

    json_out, html_out = generar_reporte(results)
    print(f"\n  📊 {json_out.name}")
    print(f"  📄 {html_out.name}")

    # Resumen de problemas
    all_errors = [c for s in results for c in s["checks"] if c["level"] == "error"]
    all_warns  = [c for s in results for c in s["checks"] if c["level"] == "warn"]

    if all_errors:
        print(f"\n  {'─'*50}")
        print(f"  ❌ {len(all_errors)} ERRORES CRÍTICOS:")
        for c in all_errors:
            print(f"    • {c['name']}: {c['msg']}")

    if all_warns:
        print(f"\n  ⚠️  {len(all_warns)} ADVERTENCIAS:")
        for c in all_warns:
            print(f"    • {c['name']}: {c['msg']}")

    if not all_errors and not all_warns:
        print(f"\n  ✅ SISTEMA OPERATIVO — todo dentro de parámetros")

    # Exit code para CI
    has_critical = any(s["status"] == "error" for s in results)
    sys.exit(1 if has_critical else 0)


if __name__ == "__main__":
    main()
