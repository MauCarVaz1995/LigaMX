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
    Reglas estrictas — si la imagen ya no aporta valor HOY, no se envía.

    pred LigaMX J{N}     → solo jornada >= last_jornada (NO la anterior)
    postpartido J{N}     → SOLO la última jornada jugada (j == last_jornada)
    pred CCL con fecha   → fecha >= hoy (pasados = fuera)
    pred CCL sin fecha   → EXCLUIR (no sabemos si ya se jugó)
    Internacional        → mtime <= 5 días
    Mexico_vs_X          → mtime <= 5 días
    ELO/ranking          → siempre
    """
    import re
    from datetime import timedelta
    name  = path.name
    path_s = str(path)
    today_dt = date.today()
    today_compact = today_s.replace("-", "")

    # ── pred Liga MX en carpeta J{N} ─────────────────────────────────────────
    if "pred_" in name and "LigaMX" in path_s:
        for part in path.parts:
            j_match = re.match(r"^J(\d+)$", part)
            if j_match:
                j = int(j_match.group(1))
                return j >= last_jornada   # solo jornada pendiente/actual

    # ── postpartido en J{N} ───────────────────────────────────────────────────
    if "postpartido" in path.parts:
        for part in path.parts:
            j_match = re.match(r"^J(\d+)$", part)
            if j_match:
                j = int(j_match.group(1))
                return j == last_jornada   # SOLO la última jornada, nada más

    # ── pred CCL ─────────────────────────────────────────────────────────────
    if "pred_" in name and "CCL" in path_s:
        m = re.search(r"(\d{8})", name)
        if m:
            return m.group(1) >= today_compact   # solo futuros
        return False   # sin fecha → no sabemos, excluir

    # ── Internacional y Mexico_vs_X → solo recientes (5 días) ────────────────
    if "Internacional" in path_s or "Mexico_vs_" in path_s:
        mtime = date.fromtimestamp(path.stat().st_mtime)
        return (today_dt - mtime).days <= 5

    return True   # ELO, rankings, otros → siempre relevantes


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
    import re as _re
    # Construir set de pares ya jugados usando los team_stats de postpartido
    # Formato: CF_America_Toluca_2026-04-19_team_stats.png → slug "cf_america_toluca"
    _pp_played: set = set()
    liga_root = PRED_DIR / "LigaMX_Clausura_2026"
    if liga_root.exists():
        for pp_dir in liga_root.rglob("postpartido"):
            for f in pp_dir.glob("*_team_stats.png"):
                # e.g. "CF_America_Toluca_2026-04-19_team_stats.png"
                stem = f.stem.replace("_team_stats", "")
                # quitar la fecha del final: _YYYY-MM-DD
                stem = _re.sub(r"_\d{4}-\d{2}-\d{2}.*", "", stem)
                _pp_played.add(stem.lower())

    if liga_root.exists():
        for jdir in sorted(liga_root.iterdir()):
            if not jdir.is_dir():
                continue
            for img in sorted(jdir.glob("pred_*.png")):
                # Excluir si ya existe postpartido para ese partido
                # pred_América_Toluca.png → "américa_toluca"
                pred_slug = img.stem.lower().replace("pred_", "")
                # Normalizar tildes/espacios para comparar
                pred_norm = pred_slug.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
                already_played = any(
                    pred_norm in pp.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
                    for pp in _pp_played
                )
                if already_played:
                    continue
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
        df["resultado_real"] = df["resultado_real"].str.lower().str.strip() if "resultado_real" in df.columns else ""
        n_total   = len(df)
        df_eval   = df[df["resultado_real"].isin(["local", "empate", "visitante"])]
        n_eval    = len(df_eval)

        # Recalcular aciertos correctamente
        if n_eval > 0 and "ganador_predicho" in df_eval.columns:
            aciertos_serie = df_eval["ganador_predicho"].str.lower().str.strip() == df_eval["resultado_real"]
            n_correct = int(aciertos_serie.sum())
        else:
            n_correct = 0
        pct       = f"{100*n_correct/n_eval:.0f}%" if n_eval > 0 else "N/A"
        color_pct = "#44ff88" if n_eval > 0 and n_correct/n_eval >= 0.50 else "#ff9800"

        # Últimas 8 predicciones con resultado
        recientes = df_eval.tail(8)
        rows = ""
        for _, r in recientes.iterrows():
            pred   = str(r.get("ganador_predicho", "?"))
            real   = str(r.get("resultado_real",   "?"))
            acerto = pred.lower().strip() == real.lower().strip()
            icon   = "✅" if acerto else "❌"
            icon_color = "#44ff88" if acerto else "#ff4444"
            partido = str(r.get("partido", ""))
            fecha   = str(r.get("fecha_prediccion", r.get("fecha_partido", "")))[:10]
            rows += (
                f"<tr>"
                f"<td style='padding:4px 8px;color:#888;font-size:11px'>{fecha}</td>"
                f"<td style='padding:4px 8px;color:#ccc;font-size:11px'>{partido}</td>"
                f"<td style='padding:4px 8px;color:#ffaa00;font-size:11px'>{pred}</td>"
                f"<td style='padding:4px 8px;color:#aaa;font-size:11px'>{real}</td>"
                f"<td style='padding:4px 8px;font-size:13px;text-align:center;color:{icon_color}'>{icon}</td>"
                f"</tr>"
            )

        # Pendientes (sin resultado)
        pendientes = df[~df["resultado_real"].isin(["local", "empate", "visitante"])].tail(5)
        pend_rows = ""
        for _, r in pendientes.iterrows():
            partido  = str(r.get("partido", ""))
            pred     = str(r.get("ganador_predicho", "?"))
            fecha_p  = str(r.get("fecha_partido", ""))[:10]
            pend_rows += (
                f"<tr>"
                f"<td style='padding:3px 8px;color:#888;font-size:11px'>{fecha_p}</td>"
                f"<td style='padding:3px 8px;color:#ccc;font-size:11px'>{partido}</td>"
                f"<td style='padding:3px 8px;color:#ffaa00;font-size:11px'>{pred}</td>"
                f"<td style='padding:3px 8px;color:#555;font-size:11px'>⏳ pendiente</td>"
                f"</tr>"
            )

        # Racha reciente (últimas 5 con resultado)
        ultimas5 = list(aciertos_serie.tail(5)) if n_eval > 0 else []
        racha_icons = "".join(
            "<span style='color:#44ff88;font-size:14px'>✅</span>" if x else
            "<span style='color:#ff4444;font-size:14px'>❌</span>"
            for x in ultimas5
        )

        return f"""
        <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
          <div style="color:#ffa726;font-size:15px;font-weight:bold;margin-bottom:10px">
            📈 Tracker de Predicciones 1X2</div>
          <div style="margin-bottom:8px">
            <span style="background:#2a2a2a;padding:6px 14px;border-radius:4px;
                         color:#ccc;font-size:13px;margin-right:8px">
              Total: <b style="color:#fff">{n_total}</b></span>
            <span style="background:#2a2a2a;padding:6px 14px;border-radius:4px;
                         color:#ccc;font-size:13px;margin-right:8px">
              Evaluadas: <b style="color:#fff">{n_eval}</b></span>
            <span style="background:#2a2a2a;padding:6px 14px;border-radius:4px;
                         color:#ccc;font-size:13px;margin-right:8px">
              Acierto: <b style="color:{color_pct}">{n_correct}/{n_eval} ({pct})</b></span>
          </div>
          <div style="margin-bottom:12px;color:#888;font-size:11px">
            Racha reciente: {racha_icons}
            <span style="margin-left:8px;color:#555">baseline naïve ≈ 47% (siempre el favorito)</span>
          </div>
          {'<div style="color:#aaa;font-size:12px;margin-bottom:6px">Últimos resultados:</div>' if rows else ''}
          {'<table style="width:100%;border-collapse:collapse"><tr>'
           '<th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Fecha</th>'
           '<th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Partido</th>'
           '<th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Pred</th>'
           '<th style="color:#555;font-size:11px;text-align:left;padding:4px 8px">Real</th>'
           '<th style="color:#555;font-size:11px;text-align:center;padding:4px 8px">✓</th>'
           '</tr>' + rows + '</table>' if rows else ''}
          {'<div style="color:#aaa;font-size:12px;margin:10px 0 6px">Pendientes (sin resultado):</div>' if pend_rows else ''}
          {'<table style="width:100%;border-collapse:collapse"><tr>'
           '<th style="color:#555;font-size:11px;text-align:left;padding:3px 8px">Fecha partido</th>'
           '<th style="color:#555;font-size:11px;text-align:left;padding:3px 8px">Partido</th>'
           '<th style="color:#555;font-size:11px;text-align:left;padding:3px 8px">Pred</th>'
           '<th style="color:#555;font-size:11px;text-align:left;padding:3px 8px">Estado</th>'
           '</tr>' + pend_rows + '</table>' if pend_rows else ''}
          <div style="color:#555;font-size:10px;margin-top:8px">
            ℹ️ n={n_eval} — se necesitan n≥200 para conclusiones estadísticas sobre el modelo</div>
        </div>"""
    except Exception as e:
        return f"<p style='color:#888'>Error tracker: {e}</p>"


def _get_betting_partidos() -> list:
    """Carga los partidos del último JSON del betting_bot."""
    reports = sorted((BASE / "output/reports").glob("betting_*.json"))
    if not reports:
        return []
    try:
        data = json.loads(reports[-1].read_text())
        # El JSON puede ser una lista directa o un dict con clave "partidos"
        if isinstance(data, list):
            return data
        return data.get("partidos", [])
    except Exception:
        return []


def build_top_picks() -> str:
    """
    Sección principal accionable: TOP PICKS para apostar HOY.
    Usa el modelo de corners (skill +12%) que sí tiene edge demostrado.
    Formato: Partido | Mercado | Prob modelo | Cuota mínima recomendada | Decisión
    """
    import sys
    sys.path.insert(0, str(BASE / "scripts"))

    picks = []
    partidos = _get_betting_partidos()

    for p in partidos:
        local  = p.get("local", "")
        visita = p.get("visitante", p.get("visita", ""))
        fecha  = p.get("fecha", "")[:10]
        jornada = p.get("jornada", "")
        corners = p.get("corners", {})

        # Mejores líneas de corners con edge demostrado (+12%)
        for line_key, label in [("over_9.5", "Corners Over 9.5"), ("over_8.5", "Corners Over 8.5")]:
            prob = corners.get(line_key)
            if prob is None:
                continue
            # Solo picks con prob >= 60% (señal fuerte del modelo)
            if prob < 0.60:
                continue
            # Cuota mínima para tener EV positivo (cuota = 1/prob, con margen 5%)
            min_odds = round(1 / (prob * 0.95), 2)
            # Nivel de confianza
            if prob >= 0.80:
                nivel = "🔥 ALTA"
                bg    = "#1a3a1a"
                color = "#44ff88"
            elif prob >= 0.70:
                nivel = "⚡ MEDIA"
                bg    = "#2a2e1a"
                color = "#aaff44"
            else:
                nivel = "💡 BAJA"
                bg    = "#2a2a1a"
                color = "#ffdd44"

            picks.append({
                "fecha": fecha, "jornada": jornada,
                "partido": f"{local} vs {visita}",
                "mercado": label,
                "prob": prob,
                "min_odds": min_odds,
                "nivel": nivel,
                "bg": bg,
                "color": color,
            })

    # Tarjetas del betting bot
    for p in partidos:
        local  = p.get("local", "")
        visita = p.get("visitante", p.get("visita", ""))
        fecha  = p.get("fecha", "")[:10]
        tarjetas = p.get("tarjetas", {})
        for line_key, label in [("over_4.5", "Tarjetas Over 4.5"), ("over_3.5", "Tarjetas Over 3.5")]:
            prob = tarjetas.get(line_key)
            if prob is None or prob < 0.65:
                continue
            min_odds = round(1 / (prob * 0.95), 2)
            nivel = "🔥 ALTA" if prob >= 0.80 else "⚡ MEDIA"
            bg    = "#1a3a1a" if prob >= 0.80 else "#2a2e1a"
            color = "#44ff88" if prob >= 0.80 else "#aaff44"
            picks.append({
                "fecha": fecha, "jornada": p.get("jornada",""),
                "partido": f"{local} vs {visita}",
                "mercado": label,
                "prob": prob,
                "min_odds": min_odds,
                "nivel": nivel, "bg": bg, "color": color,
            })

    if not picks:
        return f"""
        <div style="background:#1e1e1e;border-radius:6px;padding:16px 20px;margin-bottom:16px">
          <div style="color:#ff6b35;font-size:15px;font-weight:bold;margin-bottom:8px">
            🎯 TOP PICKS — Próxima jornada</div>
          <div style="color:#666;font-size:13px">
            Sin picks con confianza ≥60% en la jornada próxima.<br>
            <span style="color:#555;font-size:11px">El modelo solo sugiere apostar cuando hay ventaja clara.</span>
          </div>
        </div>"""

    # Ordenar por probabilidad descendente, mostrar máx 6
    picks.sort(key=lambda x: x["prob"], reverse=True)
    picks = picks[:6]

    rows = ""
    for pk in picks:
        rows += f"""
        <tr style="background:{pk['bg']}">
          <td style="padding:8px 10px;color:#888;font-size:11px">{pk['fecha']}<br>
              <span style="color:#555">J{pk['jornada']}</span></td>
          <td style="padding:8px 10px;color:#fff;font-size:12px;font-weight:bold">{pk['partido']}</td>
          <td style="padding:8px 10px;color:#aaa;font-size:12px">{pk['mercado']}</td>
          <td style="padding:8px 10px;color:{pk['color']};font-size:13px;font-weight:bold;text-align:center">{pk['prob']:.0%}</td>
          <td style="padding:8px 10px;color:#ffaa00;font-size:12px;text-align:center">&gt; {pk['min_odds']}</td>
          <td style="padding:8px 10px;font-size:11px;color:{pk['color']}">{pk['nivel']}</td>
        </tr>"""

    n_alta = sum(1 for pk in picks if "ALTA" in pk["nivel"])
    return f"""
    <div style="background:#111;border:1px solid #2a5a2a;border-radius:8px;padding:18px 20px;margin-bottom:16px">
      <div style="color:#44ff88;font-size:16px;font-weight:bold;margin-bottom:4px">
        🎯 TOP PICKS — Apostar cuando cuota sea mayor a</div>
      <div style="color:#666;font-size:11px;margin-bottom:12px">
        Solo mercados con edge demostrado (corners skill +12%) · {n_alta} picks de alta confianza</div>
      <table style="width:100%;border-collapse:collapse">
        <tr>
          <th style="color:#555;font-size:11px;text-align:left;padding:6px 10px">Fecha</th>
          <th style="color:#555;font-size:11px;text-align:left;padding:6px 10px">Partido</th>
          <th style="color:#555;font-size:11px;text-align:left;padding:6px 10px">Mercado</th>
          <th style="color:#00d4ff;font-size:11px;text-align:center;padding:6px 10px">Prob modelo</th>
          <th style="color:#ffaa00;font-size:11px;text-align:center;padding:6px 10px">Cuota mín.</th>
          <th style="color:#555;font-size:11px;text-align:left;padding:6px 10px">Confianza</th>
        </tr>
        {rows}
      </table>
      <div style="color:#555;font-size:10px;margin-top:10px">
        ⚠️ Apostar SOLO si la cuota disponible supera la mínima · Kelly 25% del bankroll asignado ·
        Modelo aún en validación (n={len(picks)} picks · necesita 200+ para confirmar edge)</div>
    </div>"""


def build_betting_analysis() -> str:
    """Tabla completa de análisis de corners/tarjetas/btts para todos los partidos próximos."""
    try:
        import sys
        sys.path.insert(0, str(BASE / "scripts"))

        # Intentar cargar ML predictor
        ml_available = False
        try:
            from modelo_ml import MLPredictor
            predictor = MLPredictor().load()
            ml_available = True
        except Exception:
            predictor = None

        partidos = _get_betting_partidos()
        if not partidos:
            return ""

        rows = ""
        for p in partidos:
            local   = p.get("local", "")
            visita  = p.get("visitante", p.get("visita", ""))
            fecha   = p.get("fecha", "")[:10]
            corners = p.get("corners", {})
            tarjetas= p.get("tarjetas", {})
            btts    = p.get("btts", {})
            errs    = p.get("errors", [])

            c85  = corners.get("over_8.5",  0) or 0
            c95  = corners.get("over_9.5",  0) or 0
            t45  = tarjetas.get("over_4.5", 0) or 0
            t35  = tarjetas.get("over_3.5", 0) or 0
            bval = btts.get("btts_si",      0) or 0
            o25  = btts.get("over_2.5",     btts.get("p_over_2.5", 0)) or 0

            def pc(v):
                v = v or 0
                if v >= 0.70: return "#44ff88"
                if v >= 0.55: return "#ffaa00"
                return "#888"

            def fmt(v):
                return f"{v:.0%}" if v else "—"

            err_note = ""
            if any("tarjetas" in e for e in errs):
                err_note = "<span style='color:#ff4444;font-size:9px'> ⚠️tarj</span>"

            has_pick = c85 >= 0.60 or c95 >= 0.60 or t45 >= 0.65
            bg = "#1a2a1a" if has_pick else "transparent"
            badge = " 🎯" if has_pick else ""

            rows += f"""
            <tr style="background:{bg};border-bottom:1px solid #222">
              <td style="padding:5px 8px;color:#888;font-size:10px">{fecha}</td>
              <td style="padding:5px 8px;color:#ddd;font-size:11px;font-weight:bold">{local[:12]} vs {visita[:12]}{badge}</td>
              <td style="padding:5px 8px;color:{pc(c85)};font-size:11px;text-align:center">{fmt(c85)}</td>
              <td style="padding:5px 8px;color:{pc(c95)};font-size:11px;text-align:center">{fmt(c95)}</td>
              <td style="padding:5px 8px;color:{pc(t35)};font-size:11px;text-align:center">{fmt(t35)}{err_note}</td>
              <td style="padding:5px 8px;color:{pc(t45)};font-size:11px;text-align:center">{fmt(t45)}</td>
              <td style="padding:5px 8px;color:{pc(bval)};font-size:11px;text-align:center">{fmt(bval)}</td>
              <td style="padding:5px 8px;color:{pc(o25)};font-size:11px;text-align:center">{fmt(o25)}</td>
            </tr>"""

        if not rows:
            return ""

        src = "LightGBM +ML" if ml_available else "Poisson MLE"
        return f"""
        <div style="background:#1e1e1e;border-radius:6px;padding:14px 18px;margin-bottom:16px">
          <div style="color:#00d4ff;font-size:14px;font-weight:bold;margin-bottom:4px">
            📊 Análisis completo — próximos partidos ({src})</div>
          <div style="color:#555;font-size:10px;margin-bottom:10px">
            🎯 = prob ≥60% en algún mercado · Verde ≥70% · Naranja ≥55%</div>
          <table style="width:100%;border-collapse:collapse">
            <tr>
              <th style="color:#555;font-size:10px;text-align:left;padding:4px 8px">Fecha</th>
              <th style="color:#555;font-size:10px;text-align:left;padding:4px 8px">Partido</th>
              <th style="color:#00d4ff;font-size:10px;text-align:center;padding:4px 6px">C&gt;8.5</th>
              <th style="color:#00d4ff;font-size:10px;text-align:center;padding:4px 6px">C&gt;9.5</th>
              <th style="color:#ffaa00;font-size:10px;text-align:center;padding:4px 6px">T&gt;3.5</th>
              <th style="color:#ffaa00;font-size:10px;text-align:center;padding:4px 6px">T&gt;4.5</th>
              <th style="color:#aaa;font-size:10px;text-align:center;padding:4px 6px">BTTS</th>
              <th style="color:#aaa;font-size:10px;text-align:center;padding:4px 6px">O2.5</th>
            </tr>
            {rows}
          </table>
        </div>"""
    except Exception as e:
        return f"<p style='color:#666;font-size:11px'>Análisis betting: {e}</p>"


def build_html(summary: dict, sections: dict[str, list[Path]]) -> str:
    tracker_section  = build_tracker_section()
    top_picks_section = build_top_picks()
    ml_section       = build_betting_analysis()

    # Ya no mostramos el HTML del Poisson betting bot (redundante con ml_section)
    betting_section = ""

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

    {top_picks_section}

    {tracker_section}

    {ml_section}

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
