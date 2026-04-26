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
import subprocess
import sys
from datetime import date, datetime, timedelta
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


def _img_age_days(path: Path) -> float:
    """
    Días reales de antigüedad de una imagen.
    En GitHub Actions el mtime = checkout time → usamos git log como fallback.
    """
    import time
    mtime_days = (time.time() - path.stat().st_mtime) / 86400
    # Si parece reciente (< 1 día), confiar en mtime
    if mtime_days < 1:
        return mtime_days
    # Si mtime parece ser el checkout (todos los archivos igual), usar git log
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", str(path)],
            capture_output=True, text=True, cwd=str(BASE), timeout=5
        )
        ct = result.stdout.strip()
        if ct:
            return (time.time() - int(ct)) / 86400
    except Exception:
        pass
    return mtime_days


def _mexico_has_match(past_days: int = 7, future_days: int = 7) -> bool:
    """
    True si México tiene partido registrado dentro de la ventana [hoy-past, hoy+future].
    Consulta data/raw/internacional/results.csv.
    """
    intl_csv = BASE / "data/raw/internacional/results.csv"
    if not intl_csv.exists():
        return False
    try:
        import pandas as pd
        today    = date.today()
        win_from = (today - timedelta(days=past_days)).isoformat()
        win_to   = (today + timedelta(days=future_days)).isoformat()
        df = pd.read_csv(intl_csv, usecols=["date", "home_team", "away_team"])
        mask = (
            ((df["home_team"] == "Mexico") | (df["away_team"] == "Mexico")) &
            (df["date"] >= win_from) &
            (df["date"] <= win_to)
        )
        return bool(mask.any())
    except Exception:
        return False


def _intl_match_date(country_a: str, country_b: str) -> str | None:
    """
    Busca la fecha más reciente de un partido internacional entre dos países.
    country_a/b en formato FotMob (ej. "Mexico", "Portugal").
    """
    intl_csv = BASE / "data/raw/internacional/results.csv"
    if not intl_csv.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(intl_csv, usecols=["date", "home_team", "away_team"])
        mask = (
            ((df["home_team"] == country_a) & (df["away_team"] == country_b)) |
            ((df["home_team"] == country_b) & (df["away_team"] == country_a))
        )
        rows = df[mask].sort_values("date", ascending=False)
        if rows.empty:
            return None
        return str(rows.iloc[0]["date"])[:10]
    except Exception:
        return None


# Mapa código FotMob → nombre en results.csv
_ISO_TO_COUNTRY = {
    "MEX": "Mexico", "ARG": "Argentina", "BRA": "Brazil",
    "USA": "United States", "COL": "Colombia", "CHI": "Chile",
    "URU": "Uruguay", "PER": "Peru",    "PAR": "Paraguay",
    "VEN": "Venezuela", "ECU": "Ecuador", "BOL": "Bolivia",
    "POR": "Portugal",  "ESP": "Spain",   "FRA": "France",
    "GER": "Germany",   "ITA": "Italy",   "ENG": "England",
    "NED": "Netherlands","BEL": "Belgium", "CRO": "Croatia",
    "SEN": "Senegal",   "GHA": "Ghana",   "MAR": "Morocco",
    "JPN": "Japan",     "KOR": "South Korea", "IRN": "Iran",
    "AUS": "Australia", "CAN": "Canada",  "COS": "Costa Rica",
    "HON": "Honduras",  "PAN": "Panama",  "JAM": "Jamaica",
    "HAI": "Haiti",     "SLV": "El Salvador", "GUA": "Guatemala",
}


def _regenerate_rankings_if_stale(max_age_days: int = 7):
    """
    Regenera los rankings de jugadores si tienen más de max_age_days.
    Llama a scripts/05_viz_player_performance.py silenciosamente.
    """
    ranking_files = [
        CHARTS / "ranking_portero.png",
        CHARTS / "ranking_defensa.png",
        CHARTS / "ranking_mediocampista.png",
        CHARTS / "ranking_delantero.png",
    ]
    needs_regen = any(
        not f.exists() or _img_age_days(f) > max_age_days
        for f in ranking_files
    )
    if not needs_regen:
        return

    print(f"  [rankings] Imágenes tienen >{max_age_days}d — regenerando...")
    script = BASE / "scripts" / "05_viz_player_performance.py"
    if not script.exists():
        print(f"  [rankings] WARN: {script} no encontrado")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE),
            env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("  [rankings] Regenerados OK")
        else:
            print(f"  [rankings] WARN: {result.stderr[:200]}")
    except Exception as e:
        print(f"  [rankings] WARN al regenerar: {e}")


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
        return _img_age_days(path) <= 5

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
                    # Solo team_stats (resumen) — no ratings individuales
                    # ratings = 2 imgs por partido × 9 = 18 imgs innecesarias en email
                    if "_ratings_" in img.name:
                        continue
                    if _image_is_relevant(img, pending, today_s, last_jornada):
                        sections["Liga MX"].append(img)

    # ── CCL ───────────────────────────────────────────────────────────────────
    ccl_root = PRED_DIR / "CCL_2025-26"
    if ccl_root.exists():
        for img in sorted(ccl_root.rglob("*.png")):
            if _image_is_relevant(img, pending, today_s, last_jornada):
                sections["CCL"].append(img)

    # ── Internacional ─────────────────────────────────────────────────────────
    import re as _re
    today_dt = date.today()

    # ¿México jugó en los últimos 7 días o tiene partido en los próximos 7?
    mex_recent  = _mexico_has_match(past_days=7,  future_days=0)
    mex_upcoming = _mexico_has_match(past_days=0, future_days=7)

    # Predicciones de selecciones en carpeta Internacional (< 5 días por git age)
    intl_root = PRED_DIR / "Internacional"
    if intl_root.exists():
        for img in sorted(intl_root.rglob("*.png")):
            if _image_is_relevant(img, pending, today_s, last_jornada):
                sections["Internacional"].append(img)

    # Carpetas Mexico_vs_X (< 5 días por git age)
    for folder in sorted(PRED_DIR.glob("Mexico_vs_*")):
        for img in sorted(folder.glob("*.png")):
            if _image_is_relevant(img, pending, today_s, last_jornada):
                sections["Internacional"].append(img)

    # selecciones_ranking_elo.png → siempre útil si < 30 días
    p = CHARTS / "selecciones_ranking_elo.png"
    if p.exists() and _img_age_days(p) <= 30:
        sections["Internacional"].append(p)

    # selecciones_prediccion.png → solo si México tiene partido próximo (7 días)
    p = CHARTS / "selecciones_prediccion.png"
    if p.exists() and mex_upcoming and _img_age_days(p) <= 7:
        sections["Internacional"].append(p)

    # selecciones_ultimos5.png → solo si México jugó en los últimos 7 días
    p = CHARTS / "selecciones_ultimos5.png"
    if p.exists() and mex_recent and _img_age_days(p) <= 7:
        sections["Internacional"].append(p)

    # Post-partido internacionales (MEX_POR etc.) en output/charts/partidos/
    # Solo imágenes con códigos de país ISO3 — verificar fecha real del partido en results.csv
    partidos_dir = BASE / "output/charts/partidos"
    if partidos_dir.exists():
        for img in sorted(partidos_dir.glob("*_team_stats.png")):
            # Patrón internacional: AAA_BBB_team_stats.png (sin fecha YYYY-MM-DD)
            m = _re.match(r"^([A-Z]{2,3})_([A-Z]{2,3})_team_stats\.png$", img.name)
            if not m:
                continue
            code_a, code_b = m.group(1), m.group(2)
            country_a = _ISO_TO_COUNTRY.get(code_a, code_a.capitalize())
            country_b = _ISO_TO_COUNTRY.get(code_b, code_b.capitalize())
            # Buscar fecha real del partido en results.csv
            match_date = _intl_match_date(country_a, country_b)
            if match_date:
                age = (today_dt - date.fromisoformat(match_date)).days
                if age <= 7:
                    sections["Internacional"].append(img)
            else:
                # Sin fecha en BD → usar git age como fallback (< 7 días)
                if _img_age_days(img) <= 7:
                    sections["Internacional"].append(img)

    # ── ELO & Stats ───────────────────────────────────────────────────────────
    # ELO rankings y evolución: generados por el pipeline diario → incluir si < 2 días
    for name in ["elo_ranking.png", "elo_evolucion.png"]:
        p = CHARTS / name
        if p.exists() and _img_age_days(p) <= 2:
            sections["ELO & Stats"].append(p)

    # Montecarlo: generado semanalmente → incluir si < 8 días
    p = CHARTS / "montecarlo_clausura2026.png"
    if p.exists() and _img_age_days(p) <= 8:
        sections["ELO & Stats"].append(p)

    # Rankings de jugadores: regenerados justo antes del email → incluir si < 2 días
    for name in ["ranking_portero.png", "ranking_defensa.png",
                 "ranking_mediocampista.png", "ranking_delantero.png"]:
        p = CHARTS / name
        if p.exists() and _img_age_days(p) <= 2:
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
            row_cls = "pick-row" if acerto else ""
            partido = str(r.get("partido", ""))
            fecha   = str(r.get("fecha_prediccion", r.get("fecha_partido", "")))[:10]
            rows += (
                f"<tr class='{row_cls}'>"
                f"<td class='gray'>{fecha}</td>"
                f"<td>{partido}</td>"
                f"<td class='orange'>{pred}</td>"
                f"<td>{real}</td>"
                f"<td style='text-align:center;font-size:13px'>{icon}</td>"
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
                f"<td class='gray'>{fecha_p}</td>"
                f"<td>{partido}</td>"
                f"<td class='orange'>{pred}</td>"
                f"<td class='gray'>⏳ pendiente</td>"
                f"</tr>"
            )

        # Racha reciente (últimas 5 con resultado)
        ultimas5 = list(aciertos_serie.tail(5)) if n_eval > 0 else []
        racha_icons = "".join(
            "<span style='font-size:14px'>✅</span>" if x else
            "<span style='font-size:14px'>❌</span>"
            for x in ultimas5
        )

        return f"""
        <div class="card">
          <div class="hdr">📈 Tracker de Predicciones 1X2</div>
          <div style="margin-bottom:8px">
            <span class="pill">Total: <b>{n_total}</b></span>
            <span class="pill">Evaluadas: <b>{n_eval}</b></span>
            <span class="pill">Acierto: <b class="{'green' if n_eval > 0 and n_correct/n_eval >= 0.50 else 'orange'}">{n_correct}/{n_eval} ({pct})</b></span>
          </div>
          <div style="margin-bottom:12px;color:#555;font-size:11px">
            Racha reciente: {racha_icons}
            <span style="margin-left:8px;color:#888">baseline naïve ≈ 47% (siempre el favorito)</span>
          </div>
          {'<div style="color:#555;font-size:12px;margin-bottom:6px">Últimos resultados:</div>' if rows else ''}
          {'<table><tr>'
           '<th>Fecha</th>'
           '<th>Partido</th>'
           '<th>Predicción</th>'
           '<th>Resultado real</th>'
           '<th style="text-align:center">✓</th>'
           '</tr>' + rows + '</table>' if rows else ''}
          {'<div style="color:#555;font-size:12px;margin:10px 0 6px">Pendientes (sin resultado):</div>' if pend_rows else ''}
          {'<table><tr>'
           '<th>Fecha partido</th>'
           '<th>Partido</th>'
           '<th>Predicción</th>'
           '<th>Estado</th>'
           '</tr>' + pend_rows + '</table>' if pend_rows else ''}
          <div style="color:#888;font-size:10px;margin-top:8px">
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
            elif prob >= 0.70:
                nivel = "⚡ MEDIA"
            else:
                nivel = "💡 BAJA"

            picks.append({
                "fecha": fecha, "jornada": jornada,
                "partido": f"{local} vs {visita}",
                "mercado": label,
                "prob": prob,
                "min_odds": min_odds,
                "nivel": nivel,
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
            picks.append({
                "fecha": fecha, "jornada": p.get("jornada",""),
                "partido": f"{local} vs {visita}",
                "mercado": label,
                "prob": prob,
                "min_odds": min_odds,
                "nivel": nivel,
            })

    if not picks:
        return f"""
        <div class="card">
          <div class="hdr">🎯 TOP PICKS — Próxima jornada</div>
          <div style="color:#666;font-size:13px">
            Sin picks con confianza ≥60% en la jornada próxima.<br>
            <span class="gray">El modelo solo sugiere apostar cuando hay ventaja clara.</span>
          </div>
        </div>"""

    # Ordenar por probabilidad descendente, mostrar máx 6
    picks.sort(key=lambda x: x["prob"], reverse=True)
    picks = picks[:6]

    rows = ""
    for pk in picks:
        row_cls = "pick-alta" if "ALTA" in pk["nivel"] else "pick-row"
        rows += f"""
        <tr class="{row_cls}">
          <td class="gray">{pk['fecha']}<br><small>J{pk['jornada']}</small></td>
          <td><b>{pk['partido']}</b></td>
          <td>{pk['mercado']}</td>
          <td style="text-align:center"><b class="green">{pk['prob']:.0%}</b></td>
          <td style="text-align:center"><b class="orange">&gt; {pk['min_odds']}</b></td>
          <td>{pk['nivel']}</td>
        </tr>"""

    n_alta = sum(1 for pk in picks if "ALTA" in pk["nivel"])
    return f"""
    <div class="card" style="border-left:4px solid #1a7a1a">
      <div class="hdr" style="border-color:#1a7a1a">🎯 TOP PICKS — Apostar cuando cuota sea mayor a la indicada</div>
      <div style="color:#555;font-size:11px;margin-bottom:12px">
        Solo mercados con edge demostrado (corners skill +12%) · {n_alta} picks de alta confianza</div>
      <table>
        <tr>
          <th>Fecha</th>
          <th>Partido</th>
          <th>Mercado</th>
          <th style="text-align:center">Prob modelo</th>
          <th style="text-align:center">Cuota mínima</th>
          <th>Confianza</th>
        </tr>
        {rows}
      </table>
      <div style="color:#888;font-size:10px;margin-top:10px">
        ⚠️ Apostar SOLO si la cuota disponible supera la mínima · Kelly 25% del bankroll asignado ·
        Modelo en validación (n={len(picks)} picks · necesita 200+ para confirmar edge)</div>
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

            err_note = ""
            if any("tarjetas" in e for e in errs):
                err_note = " <span style='color:#b00000;font-size:9px'>⚠️</span>"

            has_pick = c85 >= 0.60 or c95 >= 0.60 or t45 >= 0.65
            row_cls  = "pick-row" if has_pick else ""
            badge    = " 🎯" if has_pick else ""

            def cell(v):
                if not v:
                    return f"<td class='gray' style='text-align:center'>—</td>"
                cls = "green" if v >= 0.70 else ("orange" if v >= 0.55 else "gray")
                return f"<td class='{cls}' style='text-align:center'>{v:.0%}</td>"

            rows += f"""
            <tr class="{row_cls}">
              <td class="gray">{fecha}</td>
              <td><b>{local[:14]} vs {visita[:14]}</b>{badge}</td>
              {cell(c85)}{cell(c95)}{cell(t35)}{cell(t45)}{cell(bval)}{cell(o25)}
            </tr>"""

        if not rows:
            return ""

        src = "LightGBM" if ml_available else "Poisson"
        return f"""
        <div class="card">
          <div class="hdr">📊 Análisis de mercados — próximos partidos ({src})</div>
          <div style="color:#666;font-size:10px;margin-bottom:10px">
            🎯 = prob ≥60% en algún mercado · <span class="green">verde ≥70%</span> · <span class="orange">naranja ≥55%</span></div>
          <table>
            <tr>
              <th>Fecha</th>
              <th>Partido</th>
              <th style="text-align:center">Corners &gt;8.5</th>
              <th style="text-align:center">Corners &gt;9.5</th>
              <th style="text-align:center">Tarjetas &gt;3.5</th>
              <th style="text-align:center">Tarjetas &gt;4.5</th>
              <th style="text-align:center">BTTS (ambos anotan)</th>
              <th style="text-align:center">Over 2.5 goles</th>
            </tr>
            {rows}
          </table>
        </div>"""
    except Exception as e:
        return f"<p style='color:#666;font-size:11px'>Análisis betting: {e}</p>"


def build_parlay_section() -> str:
    """
    Tabla de parlays del día: combinaciones 2-3 selecciones de alta confianza.
    Llama a parlay_analyzer.py en modo silencioso y parsea la salida.
    """
    import subprocess, sys, re
    bot = BASE / "bots" / "parlay_analyzer.py"
    if not bot.exists():
        return ""
    try:
        result = subprocess.run(
            [sys.executable, str(bot), "--monto", "1000", "--min-prob", "0.65", "--max-legs", "3", "--top", "5"],
            capture_output=True, text=True, timeout=30,
            cwd=str(BASE)
        )
        lines = result.stdout.strip().split("\n") if result.stdout else []
    except Exception:
        return ""

    if not lines:
        return ""

    # Parsear picks individuales y parlays del output de texto
    in_picks = in_parlays = False
    picks_rows = ""
    parlay_rows = ""
    n_parlays_ev_pos = 0

    for line in lines:
        if "PICKS INDIVIDUALES" in line:
            in_picks = True; in_parlays = False; continue
        if "TOP PARLAYS" in line or "RESUMEN POR" in line:
            in_picks = False
            if "TOP PARLAYS" in line:
                in_parlays = True
                # Extraer cuántos tienen EV positivo
                m = re.search(r"(\d+) de \d+", line)
                if m: n_parlays_ev_pos = int(m.group(1))
            continue
        if "═" in line or "──" in line[:4]:
            in_parlays = False; continue

        if in_picks and "→" in line and "%" in line:
            # Extraer: pick | prob | cuota | ev | retorno
            parts = [p.strip() for p in line.strip().split("  ") if p.strip()]
            if len(parts) >= 3:
                # El pick es todo antes del primer número de prob
                m = re.search(r"(.+?)\s+(\d+\.\d+%)\s+([\d\.]+[~*])\s+([+\-]\d+\.\d+%)\s+\+?\$\s*([\d,]+)", line)
                if m:
                    pick_lbl = m.group(1).strip().replace("•", "").strip()
                    prob     = m.group(2)
                    cuota    = m.group(3)
                    ev       = m.group(4)
                    retorno  = m.group(5)
                    ev_color = "green" if "+" in ev and ev != "+0.0%" else "gray"
                    picks_rows += f"""
                    <tr>
                      <td>{pick_lbl}</td>
                      <td style="text-align:center"><b class="green">{prob}</b></td>
                      <td style="text-align:center">{cuota}</td>
                      <td style="text-align:center" class="{ev_color}">{ev}</td>
                      <td style="text-align:center" class="orange">+${retorno}</td>
                    </tr>"""

        if in_parlays and "selecciones" in line.lower() and "──" in line:
            # nueva entrada de parlay
            pass
        if in_parlays and "•" in line and "%" in line:
            pass  # líneas de picks dentro del parlay — no las parseamos individualmente
        if in_parlays and "Cuota combinada" in line:
            m = re.search(r"Cuota combinada:\s*([\d\.]+).+Prob combinada:\s*([\d\.]+%).+EV:\s*([+\-\d\.]+%)", line)
            if m:
                cuota_c = m.group(1); prob_c = m.group(2); ev_c = m.group(3)
                parlay_rows += f'<tr class="pick-row"><td colspan="3" style="color:#666;font-size:11px">Cuota: {cuota_c} | Prob: {prob_c} | EV: {ev_c}</td></tr>'
        if in_parlays and "Invertir" in line:
            m = re.search(r"Invertir \$([\d,]+).+ganarías \$([\d,]+)", line)
            if m:
                inv = m.group(1); gan = m.group(2)
                parlay_rows += f'<tr class="pick-alta"><td colspan="3"><b class="green">💰 $1,000 → ganas $+{gan}</b></td></tr>'

    if not picks_rows:
        return ""

    return f"""
    <div class="card" style="border-left:4px solid #6a1b9a">
      <div class="hdr" style="border-color:#6a1b9a">🎰 PARLAY ANALYZER — Combinadas de hoy</div>
      <div style="color:#555;font-size:11px;margin-bottom:10px">
        Picks con prob ≥65% · {n_parlays_ev_pos} parlays con EV positivo · Cuotas justas (~)</div>

      <div style="font-weight:bold;font-size:12px;margin-bottom:6px;color:#444">PICKS INDIVIDUALES</div>
      <table>
        <tr>
          <th>Pick</th><th style="text-align:center">Prob</th>
          <th style="text-align:center">Cuota</th><th style="text-align:center">EV</th>
          <th style="text-align:center">Retorno/$1000</th>
        </tr>
        {picks_rows}
      </table>

      <div style="font-weight:bold;font-size:12px;margin:12px 0 6px;color:#444">MEJORES PARLAYS</div>
      <table>{parlay_rows if parlay_rows else "<tr><td class='gray'>Sin parlays con EV positivo disponibles.</td></tr>"}</table>

      <div style="color:#888;font-size:10px;margin-top:8px">
        ~ cuota justa (1/prob). Para EV real necesitas cuota del bookmaker.
        Kelly 25% · max 3% bankroll · modelo en validación</div>
    </div>"""


def build_value_bets_section() -> str:
    """Incluye el HTML del email de value bets (generado por send_bets_email.py)."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(BASE / "scripts"))
        from send_bets_email import (
            get_ligamx_today, fetch_ligamx_odds, load_model_probs,
            build_ligamx_value_bets, load_intl_picks, build_parlays, build_html as _build_bets
        )
        partidos = get_ligamx_today()
        odds     = fetch_ligamx_odds()
        probs    = load_model_probs()
        vb       = build_ligamx_value_bets(partidos, odds, probs)
        intl     = load_intl_picks()
        parlays  = build_parlays(vb)
        inner    = _build_bets(vb, intl, parlays, partidos)
        # Extraer solo el body
        import re
        m = re.search(r'<body[^>]*>(.*)</body>', inner, re.DOTALL)
        content = m.group(1).strip() if m else inner
        return f'<div class="card" style="padding:0">{content}</div>'
    except Exception as e:
        return f'<div class="card"><div class="hdr" style="border-color:#E53935">🎰 Value Bets</div><p style="color:#888">Error cargando: {e}</p></div>'


def build_html(summary: dict, sections: dict[str, list[Path]]) -> str:
    tracker_section    = build_tracker_section()
    top_picks_section  = build_top_picks()
    ml_section         = build_betting_analysis()
    parlay_section     = build_parlay_section()
    value_bets_section = build_value_bets_section()

    # Ya no mostramos el HTML del Poisson betting bot (redundante con ml_section)
    betting_section = ""

    audit_html = load_audit_html()
    audit_section = ""
    if audit_html:
        audit_section = f"""
    <div class="card">
      <div class="hdr" style="border-color:#1565C0">🔍 Estado del sistema — Audit Bot</div>
      {audit_html}
    </div>"""

    discovery_html = load_discovery_html()
    discovery_section = ""
    if discovery_html:
        discovery_section = f"""
    <div class="card">
      <div class="hdr" style="border-color:#7b1fa2">🔬 Hallazgos automáticos — Discovery Bot</div>
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
        f"<tr><td class='gray'>{k}</td>"
        f"<td><b>{v}</b></td></tr>"
        for k, v in rows_stats
    )

    # Secciones de imágenes
    sections_html = ""
    for section_name, imgs in sections.items():
        if not imgs:
            continue
        color = SECTION_COLORS.get(section_name, "#e53935")
        items = "".join(
            f"<li style='padding:2px 0'>{i.name}</li>"
            for i in imgs
        )
        sections_html += f"""
        <div style='margin-bottom:18px;border-left:3px solid {color};padding-left:12px'>
          <div style='color:{color};font-weight:bold;font-size:14px;
                      text-transform:uppercase;margin-bottom:6px'>
            {section_name} &nbsp;<span class='gray' style='font-size:12px;font-weight:normal'>
            ({len(imgs)} imagen{'es' if len(imgs)!=1 else ''})</span>
          </div>
          <ul style='margin:0;padding-left:16px;color:#444;font-size:12px'>{items}</ul>
        </div>"""

    if not sections_html:
        sections_html = "<p class='gray'>Sin imágenes.</p>"

    # ── estilos globales para fondo blanco/legible ──────────────────────────
    CSS = """
      body{background:#f4f4f4;font-family:Arial,Helvetica,sans-serif;color:#222;margin:0;padding:20px}
      .wrap{max-width:640px;margin:0 auto}
      .card{background:#fff;border-radius:8px;padding:18px 22px;margin-bottom:14px;
            box-shadow:0 1px 4px rgba(0,0,0,.10)}
      .hdr{font-size:14px;font-weight:bold;color:#333;margin-bottom:10px;
           border-bottom:2px solid #e53935;padding-bottom:6px}
      table{width:100%;border-collapse:collapse}
      th{background:#f0f0f0;color:#555;font-size:11px;text-align:left;
         padding:5px 8px;border-bottom:1px solid #ddd}
      td{padding:5px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;color:#333}
      tr:last-child td{border-bottom:none}
      .green{color:#1a7a1a;font-weight:bold}
      .orange{color:#c07000;font-weight:bold}
      .red{color:#b00000}
      .gray{color:#888}
      .pill{display:inline-block;background:#f0f0f0;border-radius:12px;
            padding:3px 10px;font-size:12px;margin:2px}
      .pick-row{background:#e8f5e9}
      .pick-alta{background:#c8e6c9}
    """

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div class="wrap">

  <!-- HEADER -->
  <div style="background:#e53935;border-radius:8px;padding:16px 22px;margin-bottom:14px">
    <div style="color:#fff;font-size:22px;font-weight:bold;letter-spacing:1px">MAU-STATISTICS</div>
    <div style="color:#ffcdd2;font-size:13px;margin-top:2px">Resumen diario — {TODAY}</div>
  </div>

  <!-- PIPELINE STATS -->
  <div class="card">
    <div class="hdr">📊 Estado del pipeline</div>
    <table>{rows_html}</table>
  </div>

  <!-- IMÁGENES -->
  <div class="card">
    <div class="hdr">🖼️ Imágenes adjuntas — {total_imgs} en total</div>
    {sections_html}
  </div>

  {value_bets_section}

  {top_picks_section}

  {parlay_section}

  {tracker_section}

  {ml_section}

  {discovery_section}

  {audit_section}

  <div style="color:#aaa;font-size:11px;text-align:center;margin-top:10px">
    MAU-STATISTICS Bot · generado automáticamente · {TODAY}
  </div>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Enviar
# ─────────────────────────────────────────────────────────────────────────────
def send_email(app_password: str, dry_run: bool = False) -> bool:
    # Regenerar rankings si están desactualizados (antes de recolectar secciones)
    _regenerate_rankings_if_stale(max_age_days=7)

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
