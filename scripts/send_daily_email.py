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
TO_ADDR = "maucarvaz@gmail.com"
FROM_ADDR = "maucarvaz@gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

PRED_DIR   = BASE / "output/charts/predicciones"
PARTY_DIR  = BASE / "output/charts/partidos"
LOGS_DIR   = BASE / "logs"
SUMMARY_F  = LOGS_DIR / "daily_summary.json"


# ─────────────────────────────────────────────────────────────────────────────
# Colectar imágenes relevantes (no solo las de hoy)
# ─────────────────────────────────────────────────────────────────────────────
CHARTS   = BASE / "output/charts"
PRED_DIR = BASE / "output/charts/predicciones"


def collect_by_section() -> dict[str, list[Path]]:
    """
    Devuelve imágenes agrupadas por sección para el email.
    Estructura:
      "Liga MX"       → predicciones J{N} + postpartido J{N}
      "CCL"           → predicciones CCL
      "Internacional" → predicciones Intl + selecciones
      "ELO & Stats"   → ELO, montecarlo, rankings
    """
    sections: dict[str, list[Path]] = {
        "Liga MX": [],
        "CCL": [],
        "Internacional": [],
        "ELO & Stats": [],
    }

    # Liga MX — predicciones + postpartido en J{N}/
    liga_root = PRED_DIR / "LigaMX_Clausura_2026"
    if liga_root.exists():
        for jdir in sorted(liga_root.iterdir()):
            if not jdir.is_dir():
                continue
            # Predicciones directas en J{N}/
            sections["Liga MX"] += sorted(jdir.glob("pred_*.png"))
            # Post-partido en J{N}/postpartido/
            pp_dir = jdir / "postpartido"
            if pp_dir.exists():
                sections["Liga MX"] += sorted(pp_dir.glob("*.png"))

    # CCL
    ccl_root = PRED_DIR / "CCL_2025-26"
    if ccl_root.exists():
        sections["CCL"] += sorted(ccl_root.rglob("*.png"))

    # Internacional — predicciones
    intl_root = PRED_DIR / "Internacional"
    if intl_root.exists():
        sections["Internacional"] += sorted(intl_root.rglob("*.png"))

    # Internacional — selecciones charts
    for name in ["selecciones_ranking_elo.png", "selecciones_ultimos5.png",
                 "selecciones_prediccion.png"]:
        p = CHARTS / name
        if p.exists():
            sections["Internacional"].append(p)

    # Internacional — Mexico_vs_X predicciones antiguas
    for folder in sorted(PRED_DIR.glob("Mexico_vs_*")):
        sections["Internacional"] += sorted(folder.glob("*.png"))

    # ELO & Stats
    for name in ["elo_ranking.png", "elo_evolucion.png",
                 "montecarlo_clausura2026.png",
                 "ranking_portero.png", "ranking_defensa.png",
                 "ranking_mediocampista.png", "ranking_delantero.png"]:
        p = CHARTS / name
        if p.exists():
            sections["ELO & Stats"].append(p)

    # Resúmenes de jornada viejos (raíz)
    old_resumenes = sorted(CHARTS.glob("resumen_postjornada*.png"))
    if old_resumenes:
        sections["Liga MX"] += old_resumenes

    # Deduplicar por sección
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

def build_html(summary: dict, sections: dict[str, list[Path]]) -> str:
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
