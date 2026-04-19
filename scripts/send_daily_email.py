#!/usr/bin/env python3
"""
send_daily_email.py — Resumen diario MAU-STATISTICS por correo
===============================================================
Envía a maucarvaz@gmail.com:
  - Resumen de stats del pipeline (ELO, partidos, tracker)
  - Imágenes generadas hoy adjuntas (predicciones + post-partido)

Requiere variable de entorno (o secret en GitHub Actions):
  GMAIL_APP_PASSWORD  — App Password de Gmail (no tu contraseña normal)

Cómo obtener el App Password:
  1. myaccount.google.com → Seguridad → Verificación en 2 pasos (activa)
  2. myaccount.google.com → Seguridad → Contraseñas de aplicaciones
  3. Nombre: "MAU-STATISTICS Bot" → Generar → copiar los 16 caracteres
  4. En GitHub: repo → Settings → Secrets → Actions → New: GMAIL_APP_PASSWORD

Uso local:
  GMAIL_APP_PASSWORD=xxxx python scripts/send_daily_email.py
  GMAIL_APP_PASSWORD=xxxx python scripts/send_daily_email.py --dry-run
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import date, datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
TODAY   = date.today().isoformat()         # "2026-04-12"
TO_ADDR   = os.environ.get("GMAIL_TO",   "") or "maucarvaz@gmail.com"
FROM_ADDR = os.environ.get("GMAIL_FROM", "") or "maucarvaz@gmail.com"
SMTP_HOST = os.environ.get("SMTP_HOST",  "") or "smtp.gmail.com"
SMTP_PORT = int(os.environ.get("SMTP_PORT", "") or "587")

PRED_DIR    = BASE / "output/charts/predicciones"
PARTY_DIR   = BASE / "output/charts/partidos"
LOGS_DIR    = BASE / "logs"
SUMMARY_F   = LOGS_DIR / "daily_summary.json"
REPORTS_DIR = BASE / "output/reports"


# ─────────────────────────────────────────────────────────────────────────────
# Colectar imágenes relevantes (no solo las de hoy)
# ─────────────────────────────────────────────────────────────────────────────
CHARTS   = BASE / "output/charts"
PRED_DIR = BASE / "output/charts/predicciones"


def _load_pending_matches() -> dict:
    """
    Carga partidos pendientes del historico activo.
    Retorna {liga: set_of_slugs} para filtrar qué predicciones son relevantes.
    """
    import json, re
    pending: dict[str, set] = {"ligamx": set(), "ccl": set()}
    today_s = TODAY

    # Liga MX
    hist = BASE / "data/raw/historico/historico_clausura_2026.json"
    if hist.exists():
        d = json.loads(hist.read_text())
        for p in d["partidos"]:
            if not p.get("terminado") and p.get("fecha", "")[:10] >= today_s:
                slug = re.sub(r"[^\w]", "_", f"{p['local']}_{p['visitante']}")
                pending["ligamx"].add(slug.lower())

    # CCL fixtures
    ccl_fixtures = sorted((BASE / "data/raw/fotmob/ccl").glob("ccl_fixtures_*.json")) if (BASE / "data/raw/fotmob/ccl").exists() else []
    if ccl_fixtures:
        d = json.loads(ccl_fixtures[-1].read_text())
        matches = d if isinstance(d, list) else d.get("matches", d.get("fixtures", []))
        for m in matches:
            fecha = str(m.get("date", "")).replace("-", "")[:8]
            fecha_iso = f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:8]}" if len(fecha) >= 8 else ""
            if fecha_iso >= today_s:
                h = re.sub(r"[^\w]", "_", m.get("home", m.get("homeTeam", {}).get("name", "")))
                a = re.sub(r"[^\w]", "_", m.get("away", m.get("awayTeam", {}).get("name", "")))
                pending["ccl"].add(f"{h}_{a}".lower())

    return pending


def _image_is_relevant(path: Path, pending: dict, today_s: str,
                        last_jornada: int) -> bool:
    """
    Decide si una imagen tiene valor ahora mismo.
    Reglas:
      pred_*.png           → solo si el partido aún no se jugó (fecha >= hoy)
      postpartido/*.png    → solo de la última jornada jugada (o la anterior)
      ELO/ranking*.png     → siempre relevantes
      Mexico_vs_X/*.png    → solo si la fecha en el nombre >= hoy-7
    """
    import re
    name  = path.name
    parts = path.parts

    # Predicciones Liga MX
    if "pred_" in name and "LigaMX" in str(path):
        # Extraer fecha del nombre si existe: pred_A_B_20260419.png
        m = re.search(r"(\d{8})", name)
        if m:
            fecha = m.group(1)
            return fecha >= today_s.replace("-", "")
        # Sin fecha en nombre: pertenece a una carpeta J{N}
        for part in parts:
            j_match = re.match(r"J(\d+)$", part)
            if j_match:
                j = int(j_match.group(1))
                return j >= last_jornada  # solo jornada actual o futura

    # Post-partido: solo las 2 últimas jornadas
    if "postpartido" in parts:
        for part in parts:
            j_match = re.match(r"J(\d+)$", part)
            if j_match:
                j = int(j_match.group(1))
                return j >= last_jornada - 1

    # CCL predicciones: solo partidos pendientes
    if "pred_" in name and "CCL" in str(path):
        m = re.search(r"(\d{8})", name)
        if m:
            return m.group(1) >= today_s.replace("-", "")
        return True  # sin fecha → incluir por defecto

    # Internacional carpetas Mexico_vs_X: solo últimos 7 días
    if "Mexico_vs_" in str(path):
        m = re.search(r"(\d{8})", name)
        if m:
            from datetime import datetime, timedelta
            try:
                img_date = datetime.strptime(m.group(1), "%Y%m%d").date()
                cutoff = date.today() - timedelta(days=7)
                return img_date >= cutoff
            except Exception:
                pass
        # Sin fecha en nombre: verificar mtime
        mtime = date.fromtimestamp(path.stat().st_mtime)
        return (date.today() - mtime).days <= 7

    # Internacional/FECHA/: solo últimos 7 días
    if "Internacional" in str(path):
        for part in parts:
            # carpeta con fecha YYYY-MM-DD
            import re as _re
            if _re.match(r"\d{4}-\d{2}-\d{2}", part):
                from datetime import datetime, timedelta
                try:
                    folder_date = date.fromisoformat(part)
                    return (date.today() - folder_date).days <= 7
                except Exception:
                    pass
        return True

    return True  # por defecto incluir


def _get_last_jornada() -> int:
    """Obtiene la última jornada jugada del historico activo."""
    import json
    hist = BASE / "data/raw/historico/historico_clausura_2026.json"
    if not hist.exists():
        return 1
    d = json.loads(hist.read_text())
    played = [int(p.get("jornada", 0)) for p in d["partidos"] if p.get("terminado")]
    return max(played) if played else 1


def collect_by_section() -> dict[str, list[Path]]:
    """
    Devuelve SOLO imágenes relevantes agrupadas por sección.
    Regla dura: nada del pasado que ya no aporta valor.
      "Liga MX"       → pred partidos PENDIENTES + postpartido última jornada
      "CCL"           → pred partidos CCL PENDIENTES
      "Internacional" → pred intl últimos 7 días + ELO selecciones
      "ELO & Stats"   → ELO, montecarlo, rankings (siempre relevantes)
    """
    today_s      = TODAY
    last_jornada = _get_last_jornada()
    pending      = _load_pending_matches()

    sections: dict[str, list[Path]] = {
        "Liga MX": [],
        "CCL": [],
        "Internacional": [],
        "ELO & Stats": [],
    }

    # ── Liga MX ───────────────────────────────────────────────────────────────
    liga_root = PRED_DIR / "LigaMX_Clausura_2026"
    if liga_root.exists():
        for jdir in sorted(liga_root.iterdir()):
            if not jdir.is_dir():
                continue
            for img in sorted(jdir.glob("pred_*.png")):
                if _image_is_relevant(img, pending, today_s, last_jornada):
                    sections["Liga MX"].append(img)
            pp_dir = jdir / "postpartido"
            if pp_dir.exists():
                for img in sorted(pp_dir.glob("*.png")):
                    if _image_is_relevant(img, pending, today_s, last_jornada):
                        sections["Liga MX"].append(img)

    # ── CCL ───────────────────────────────────────────────────────────────────
    ccl_root = PRED_DIR / "CCL_2025-26"
    if ccl_root.exists():
        for img in sorted(ccl_root.rglob("*.png")):
            if _image_is_relevant(img, pending, today_s, last_jornada):
                sections["CCL"].append(img)

    # ── Internacional ─────────────────────────────────────────────────────────
    intl_root = PRED_DIR / "Internacional"
    if intl_root.exists():
        for img in sorted(intl_root.rglob("*.png")):
            if _image_is_relevant(img, pending, today_s, last_jornada):
                sections["Internacional"].append(img)

    for folder in sorted(PRED_DIR.glob("Mexico_vs_*")):
        for img in sorted(folder.glob("*.png")):
            if _image_is_relevant(img, pending, today_s, last_jornada):
                sections["Internacional"].append(img)

    for name in ["selecciones_ranking_elo.png"]:
        p = CHARTS / name
        if p.exists():
            sections["Internacional"].append(p)

    # ── ELO & Stats ───────────────────────────────────────────────────────────
    for name in ["elo_ranking.png", "elo_evolucion.png",
                 "montecarlo_clausura2026.png",
                 "ranking_portero.png", "ranking_defensa.png",
                 "ranking_mediocampista.png", "ranking_delantero.png"]:
        p = CHARTS / name
        if p.exists():
            sections["ELO & Stats"].append(p)

    # Deduplicar
    for k in sections:
        seen: set = set()
        deduped = []
        for p in sections[k]:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        sections[k] = deduped

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Leer resumen del pipeline
# ─────────────────────────────────────────────────────────────────────────────
def load_summary() -> dict:
    if not SUMMARY_F.exists():
        return {}
    try:
        return json.loads(SUMMARY_F.read_text())
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Construir cuerpo HTML
# ─────────────────────────────────────────────────────────────────────────────
SECTION_COLORS = {
    "Liga MX":       "#e53935",
    "CCL":           "#1565C0",
    "Internacional": "#2E7D32",
    "ELO & Stats":   "#6A1B9A",
}

def load_betting_html() -> str:
    """Carga el reporte HTML de betting más reciente."""
    if not REPORTS_DIR.exists():
        return ""
    reports = sorted(REPORTS_DIR.glob("betting_*.html"))
    if not reports:
        return ""
    try:
        return reports[-1].read_text()
    except Exception:
        return ""


def load_audit_html() -> str:
    """Carga el reporte HTML del audit más reciente."""
    if not REPORTS_DIR.exists():
        return ""
    reports = sorted(REPORTS_DIR.glob("audit_*.html"))
    if not reports:
        return ""
    try:
        return reports[-1].read_text()
    except Exception:
        return ""


def load_discovery_html() -> str:
    """Carga el reporte de discovery más reciente."""
    latest = REPORTS_DIR / "discovery_latest.html"
    if latest.exists():
        try:
            return latest.read_text()
        except Exception:
            pass
    discovery_dir = REPORTS_DIR / "discovery"
    if discovery_dir.exists():
        reports = sorted(discovery_dir.glob("discovery_*.html"))
        if reports:
            try:
                return reports[-1].read_text()
            except Exception:
                pass
    return ""


def build_tracker_section() -> str:
    """Genera HTML con resumen del tracker de predicciones + resultados recientes."""
    import pandas as pd
    log_csv = BASE / "data/processed/predicciones_log.csv"
    if not log_csv.exists():
        return ""
    try:
        df = pd.read_csv(log_csv)
        n_total    = len(df)
        n_eval     = df["resultado_real"].notna().sum() if "resultado_real" in df.columns else 0
        n_correct  = (df["acierto"] == True).sum() if "acierto" in df.columns else 0
        pct        = f"{100*n_correct/n_eval:.1f}%" if n_eval > 0 else "N/A"
        color_pct  = "#44ff88" if n_eval > 0 and n_correct/n_eval >= 0.50 else "#ff9800"

        # Últimas 8 predicciones con resultado
        recientes = df[df["resultado_real"].notna()].tail(8)
        rows = ""
        for _, r in recientes.iterrows():
            acierto = r.get("acierto", None)
            icon = "✅" if acierto == True else ("❌" if acierto == False else "⏳")
            pred_col = "#aaa"
            partido = r.get("partido", r.get("local", "") + " vs " + r.get("visitante", ""))
            pred    = r.get("prediccion", r.get("ganador_predicho", "?"))
            real    = r.get("resultado_real", "?")
            fecha   = str(r.get("fecha", ""))[:10]
            rows += (
                f"<tr>"
                f"<td style='padding:4px 8px;color:#888;font-size:11px'>{fecha}</td>"
                f"<td style='padding:4px 8px;color:#ccc;font-size:11px'>{partido}</td>"
                f"<td style='padding:4px 8px;color:{pred_col};font-size:11px'>{pred}</td>"
                f"<td style='padding:4px 8px;color:#ccc;font-size:11px'>{real}</td>"
                f"<td style='padding:4px 8px;font-size:14px;text-align:center'>{icon}</td>"
                f"</tr>"
            )

        # Pendientes
        pendientes = df[df["resultado_real"].isna()]
        pend_rows = ""
        for _, r in pendientes.tail(5).iterrows():
            partido = r.get("partido", r.get("local", "") + " vs " + r.get("visitante", ""))
            pred    = r.get("prediccion", r.get("ganador_predicho", "?"))
            fecha   = str(r.get("fecha", ""))[:10]
            pend_rows += (
                f"<tr>"
                f"<td style='padding:3px 8px;color:#888;font-size:11px'>{fecha}</td>"
                f"<td style='padding:3px 8px;color:#ccc;font-size:11px'>{partido}</td>"
                f"<td style='padding:3px 8px;color:#ffaa00;font-size:11px'>{pred}</td>"
                f"<td style='padding:3px 8px;color:#555;font-size:11px'>⏳ pendiente</td>"
                f"</tr>"
            )

        return f"""
        <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
          <div style="color:#ffa726;font-size:15px;font-weight:bold;margin-bottom:10px">
            📈 Tracker de Predicciones</div>
          <div style="margin-bottom:12px">
            <span style="background:#2a2a2a;padding:6px 14px;border-radius:4px;
                         color:#ccc;font-size:13px;margin-right:8px">
              Total: <b style="color:#fff">{n_total}</b></span>
            <span style="background:#2a2a2a;padding:6px 14px;border-radius:4px;
                         color:#ccc;font-size:13px;margin-right:8px">
              Evaluadas: <b style="color:#fff">{n_eval}</b></span>
            <span style="background:#2a2a2a;padding:6px 14px;border-radius:4px;
                         color:#ccc;font-size:13px">
              Acierto: <b style="color:{color_pct}">{pct}</b></span>
          </div>
          {'<div style="color:#aaa;font-size:12px;margin-bottom:6px">Últimos resultados:</div>' if rows else ''}
          {'<table style="width:100%;border-collapse:collapse"><tr><th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Fecha</th><th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Partido</th><th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Pred</th><th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Real</th><th style="color:#555;font-size:11px;text-align:center;padding:4px 8px">✓</th></tr>' + rows + '</table>' if rows else ''}
          {'<div style="color:#aaa;font-size:12px;margin:10px 0 6px">Sin resultado aún:</div>' if pend_rows else ''}
          {'<table style="width:100%;border-collapse:collapse"><tr><th style="color:#555;font-size:11px;text-align:left;padding:3px 8px">Fecha</th><th style="color:#555;font-size:11px;text-align:left;padding:3px 8px">Partido</th><th style="color:#555;font-size:11px;text-align:left;padding:3px 8px">Pred</th><th style="color:#555;font-size:11px;text-align:left;padding:3px 8px">Estado</th></tr>' + pend_rows + '</table>' if pend_rows else ''}
        </div>"""
    except Exception as e:
        return f"<p style='color:#888'>Error tracker: {e}</p>"


def build_betting_analysis() -> str:
    """Genera HTML con análisis ML de los próximos partidos (corners + cards)."""
    try:
        import sys
        sys.path.insert(0, str(BASE / "scripts"))
        from modelo_ml import MLPredictor

        predictor = MLPredictor().load()

        # Leer fixtures próximos del betting_bot JSON
        reports = sorted((BASE / "output/reports").glob("betting_*.json"))
        if not reports:
            return ""
        data = json.loads(reports[-1].read_text())
        partidos = data.get("partidos", [])[:10]
        if not partidos:
            return ""

        rows = ""
        value_count = 0
        for p in partidos:
            local  = p.get("local", "")
            visita = p.get("visitante", "")
            fecha  = p.get("fecha", "")[:10]
            try:
                ml = predictor.predict(local, visita)
                c85  = ml.get("corners_over_8.5",  0)
                c95  = ml.get("corners_over_9.5",  0)
                t45  = ml.get("cards_over_4.5",    0)
                btts = ml.get("btts",              0)

                def pct_color(p_val):
                    if p_val >= 0.65: return "#44ff88"
                    if p_val >= 0.50: return "#ffaa00"
                    return "#ff6b6b"

                # Detectar si hay valor (>60% en cualquier mercado)
                has_value = any(x >= 0.62 for x in [c85, c95, t45])
                if has_value:
                    value_count += 1

                bg = "#1a2e1a" if has_value else "#1e1e1e"
                badge = " 🎯" if has_value else ""

                rows += f"""
                <tr style="background:{bg}">
                  <td style="padding:6px 8px;color:#888;font-size:11px">{fecha}</td>
                  <td style="padding:6px 8px;color:#fff;font-size:12px;font-weight:bold">{local[:10]} vs {visita[:10]}{badge}</td>
                  <td style="padding:6px 8px;color:{pct_color(c85)};font-size:12px;text-align:center">{c85:.0%}</td>
                  <td style="padding:6px 8px;color:{pct_color(c95)};font-size:12px;text-align:center">{c95:.0%}</td>
                  <td style="padding:6px 8px;color:{pct_color(t45)};font-size:12px;text-align:center">{t45:.0%}</td>
                  <td style="padding:6px 8px;color:{pct_color(btts)};font-size:12px;text-align:center">{btts:.0%}</td>
                </tr>"""
            except Exception:
                continue

        if not rows:
            return ""

        return f"""
        <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
          <div style="color:#00d4ff;font-size:15px;font-weight:bold;margin-bottom:4px">
            🤖 Análisis ML — Próximos partidos</div>
          <div style="color:#555;font-size:11px;margin-bottom:10px">
            🎯 = valor detectado (prob ≥ 62%) · LightGBM calibrado · 686 partidos</div>
          <table style="width:100%;border-collapse:collapse">
            <tr>
              <th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Fecha</th>
              <th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Partido</th>
              <th style="color:#00d4ff;font-size:11px;text-align:center;padding:4px 8px">C&gt;8.5</th>
              <th style="color:#00d4ff;font-size:11px;text-align:center;padding:4px 8px">C&gt;9.5</th>
              <th style="color:#ffaa00;font-size:11px;text-align:center;padding:4px 8px">T&gt;4.5</th>
              <th style="color:#aaa;font-size:11px;text-align:center;padding:4px 8px">BTTS</th>
            </tr>
            {rows}
          </table>
          <div style="color:#555;font-size:11px;margin-top:8px">
            Verde ≥65% · Naranja ≥50% · Rojo &lt;50% · C=Corners · T=Tarjetas</div>
        </div>"""
    except Exception as e:
        return f"<p style='color:#666;font-size:11px'>ML analysis: {e}</p>"


def build_html(summary: dict, sections: dict[str, list[Path]]) -> str:
    tracker_section  = build_tracker_section()
    ml_section       = build_betting_analysis()

    betting_html = load_betting_html()
    betting_section = ""
    if betting_html:
        betting_section = f"""
    <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
      <div style="color:#E53935;font-size:15px;font-weight:bold;margin-bottom:10px">
        🎰 Análisis Betting (Poisson) — próximos partidos</div>
      {betting_html}
    </div>"""

    audit_html = load_audit_html()
    audit_section = ""
    if audit_html:
        audit_section = f"""
    <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
      <div style="color:#90CAF9;font-size:15px;font-weight:bold;margin-bottom:10px">
        🔍 Estado del sistema — Audit Bot</div>
      {audit_html}
    </div>"""

    discovery_html = load_discovery_html()
    discovery_section = ""
    if discovery_html:
        discovery_section = f"""
    <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
      <div style="color:#a855f7;font-size:15px;font-weight:bold;margin-bottom:10px">
        🔬 Hallazgos automáticos — Discovery Bot</div>
      {discovery_html}
    </div>"""

    p = summary.get("pasos", {})
    nuevos_liga = p.get("paso1", {}).get("nuevos", 0)
    nuevos_intl = p.get("paso3", {}).get("nuevos", 0)
    equipos_elo = p.get("paso2", {}).get("equipos_afectados", 0)
    sels_elo    = p.get("paso4", {}).get("selecciones_afectadas", 0)
    tracker_upd = p.get("paso5", {}).get("actualizados", 0)
    aciertos    = p.get("paso5", {}).get("aciertos", 0)
    total_eval  = p.get("paso5", {}).get("total_evaluados", 0)
    pct_acierto = f"{100*aciertos/total_eval:.1f}%" if total_eval else "N/A"
    git_pushed  = "✅" if p.get("paso7", {}).get("pushed") else "⚠️ No"
    elapsed     = summary.get("elapsed_s", "?")
    pipeline_ok = "✅ OK" if summary.get("success") else "❌ FALLO"
    total_imgs  = sum(len(v) for v in sections.values())

    rows_stats = [
        ("Liga MX — resultados nuevos",       nuevos_liga),
        ("Internacional — resultados nuevos",  nuevos_intl),
        ("ELO Liga MX — equipos afectados",    equipos_elo),
        ("ELO Selecciones — actualizadas",     sels_elo),
        ("Tracker — predicciones resueltas",   tracker_upd),
        (f"Tracker — aciertos ({pct_acierto})", f"{aciertos}/{total_eval}"),
        ("Total imágenes adjuntas",            total_imgs),
        ("Git push",                           git_pushed),
        ("Pipeline",                           f"{pipeline_ok} ({elapsed}s)"),
    ]

    rows_html = "".join(
        f"<tr><td style='padding:5px 12px;color:#aaa;font-size:13px'>{k}</td>"
        f"<td style='padding:5px 12px;color:#fff;font-weight:bold;font-size:13px'>{v}</td></tr>"
        for k, v in rows_stats
    )

    # Secciones de imágenes
    sections_html = ""
    for section_name, imgs in sections.items():
        if not imgs:
            continue
        color = SECTION_COLORS.get(section_name, "#e53935")
        items = "".join(
            f"<li style='color:#ccc;font-size:12px;padding:2px 0'>{i.name}</li>"
            for i in imgs
        )
        sections_html += f"""
        <div style='margin-bottom:18px;border-left:3px solid {color};padding-left:12px'>
          <div style='color:{color};font-weight:bold;font-size:14px;
                      text-transform:uppercase;margin-bottom:6px'>
            {section_name} &nbsp;<span style='color:#555;font-size:12px;font-weight:normal'>
            ({len(imgs)} imagen{'es' if len(imgs)!=1 else ''})</span>
          </div>
          <ul style='margin:0;padding-left:16px'>{items}</ul>
        </div>"""

    if not sections_html:
        sections_html = "<p style='color:#888'>Sin imágenes.</p>"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="background:#121212;font-family:Arial,sans-serif;padding:24px">
  <div style="max-width:600px;margin:0 auto">

    <div style="background:#1e1e1e;border-top:4px solid #e53935;
                border-radius:6px;padding:20px 24px;margin-bottom:16px">
      <div style="color:#e53935;font-size:22px;font-weight:bold;letter-spacing:2px">
        MAU-STATISTICS</div>
      <div style="color:#888;font-size:13px;margin-top:4px">Resumen diario — {TODAY}</div>
    </div>

    <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
      <div style="color:#fff;font-size:15px;font-weight:bold;margin-bottom:10px">
        📊 Pipeline</div>
      <table style="width:100%;border-collapse:collapse">{rows_html}</table>
    </div>

    <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
      <div style="color:#fff;font-size:15px;font-weight:bold;margin-bottom:14px">
        🖼️ Imágenes adjuntas ({total_imgs})</div>
      {sections_html}
    </div>

    {tracker_section}

    {ml_section}

    {betting_section}

    {discovery_section}

    {audit_section}

    <div style="color:#444;font-size:11px;text-align:center;margin-top:12px">
      MAU-STATISTICS Bot · @Miau_Stats_MX · generado automáticamente
    </div>
  </div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Enviar
# ─────────────────────────────────────────────────────────────────────────────
def send_email(app_password: str, dry_run: bool = False) -> bool:
    summary  = load_summary()
    sections = collect_by_section()
    all_imgs = [img for imgs in sections.values() for img in imgs]
    html     = build_html(summary, sections)

    subject = f"MAU-STATISTICS · {TODAY} · {len(all_imgs)} imágenes"

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = FROM_ADDR
    msg["To"]      = TO_ADDR

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt)

    # Adjuntar imágenes en orden por sección
    attached = 0
    for img_path in all_imgs:
        try:
            data = img_path.read_bytes()
            mime_img = MIMEImage(data, name=img_path.name)
            mime_img.add_header(
                "Content-Disposition", "attachment",
                filename=img_path.name
            )
            msg.attach(mime_img)
            attached += 1
        except Exception as e:
            print(f"  [warn] No se pudo adjuntar {img_path.name}: {e}")

    for sec, imgs in sections.items():
        if imgs:
            print(f"  {sec}: {len(imgs)} imgs")
    print(f"  Asunto : {subject}")
    print(f"  Para   : {TO_ADDR}")
    print(f"  Adjuntos: {attached} imágenes")

    if dry_run:
        print("  [dry-run] Email NO enviado — modo prueba")
        return True

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.login(FROM_ADDR, app_password)
            s.sendmail(FROM_ADDR, TO_ADDR, msg.as_string())
        print(f"  ✅ Email enviado a {TO_ADDR}")
        return True
    except Exception as e:
        print(f"  ❌ Error enviando: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Construye el email pero no lo envía")
    args = parser.parse_args()

    pw = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not pw and not args.dry_run:
        print("ERROR: Variable GMAIL_APP_PASSWORD no encontrada.")
        print("  Agrega el App Password de Gmail como secret en GitHub Actions,")
        print("  o expórtala localmente: export GMAIL_APP_PASSWORD=xxxx")
        sys.exit(1)

    print(f"\n── Enviando resumen diario {TODAY} ──")
    ok = send_email(pw, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
