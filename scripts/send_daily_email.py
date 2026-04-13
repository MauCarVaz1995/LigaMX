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
CHARTS = BASE / "output/charts"

def collect_all_relevant() -> list[Path]:
    """
    Devuelve todas las imágenes relevantes del proyecto, organizadas por sección.
    Excluye: pizza charts, paletas de prueba, predicciones_hoy/ (formato viejo).
    """
    imgs = []

    # 1. Predicciones (estructura nueva: LigaMX/J{N}/, CCL/Semis/, Internacional/)
    if PRED_DIR.exists():
        imgs += sorted(PRED_DIR.rglob("*.png"))

    # 2. Post-partido (ratings jugadores, porteros, team stats)
    if PARTY_DIR.exists():
        imgs += sorted(PARTY_DIR.glob("*.png"))

    # 3. ELO charts
    for name in ["elo_ranking.png", "elo_evolucion.png"]:
        p = CHARTS / name
        if p.exists():
            imgs.append(p)

    # 4. Selecciones
    for name in ["selecciones_ranking_elo.png", "selecciones_ultimos5.png",
                 "selecciones_prediccion.png"]:
        p = CHARTS / name
        if p.exists():
            imgs.append(p)

    # 5. Resúmenes de jornada (raíz y subcarpetas jornada*/)
    imgs += sorted(CHARTS.glob("resumen_postjornada*.png"))
    for jfolder in sorted(CHARTS.glob("jornada*/")):
        imgs += sorted(jfolder.glob("resumen_*.png"))
        imgs += sorted(jfolder.glob("prediccion_*.png"))

    # 6. Montecarlo / simulación del torneo
    for name in ["montecarlo_clausura2026.png", "montecarlo_clausura2026_rojo_fuego.png"]:
        p = CHARTS / name
        if p.exists():
            imgs.append(p)

    # 7. Rankings de jugadores
    imgs += sorted(CHARTS.glob("ranking_*.png"))

    # Deduplicar manteniendo orden
    seen = set()
    out = []
    for p in imgs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


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
def build_html(summary: dict, images: list[Path]) -> str:
    p = summary.get("pasos", {})
    nuevos_liga = p.get("paso1", {}).get("nuevos", 0)
    nuevos_intl = p.get("paso3", {}).get("nuevos", 0)
    equipos_elo = p.get("paso2", {}).get("equipos_afectados", 0)
    sels_elo    = p.get("paso4", {}).get("selecciones_afectadas", 0)
    tracker_upd = p.get("paso5", {}).get("actualizados", 0)
    aciertos    = p.get("paso5", {}).get("aciertos", 0)
    total_eval  = p.get("paso5", {}).get("total_evaluados", 0)
    pct_acierto = f"{100*aciertos/total_eval:.1f}%" if total_eval else "N/A"
    imgs_gen    = p.get("paso6", {}).get("predicciones_generadas", 0)
    git_pushed  = "✅" if p.get("paso7", {}).get("pushed") else "⚠️ No"
    elapsed     = summary.get("elapsed_s", "?")
    pipeline_ok = "✅ OK" if summary.get("success") else "❌ FALLO"

    rows_stats = [
        ("Liga MX — resultados nuevos",    nuevos_liga),
        ("Internacional — resultados nuevos", nuevos_intl),
        ("ELO Liga MX — equipos afectados",  equipos_elo),
        ("ELO Selecciones — actualizadas",   sels_elo),
        ("Tracker — predicciones resueltas", tracker_upd),
        (f"Tracker — aciertos ({pct_acierto})", f"{aciertos}/{total_eval}"),
        ("Imágenes generadas hoy",           len(images)),
        ("Git push",                         git_pushed),
        ("Pipeline",                         f"{pipeline_ok} ({elapsed}s)"),
    ]

    rows_html = "".join(
        f"<tr><td style='padding:6px 12px;color:#aaa'>{k}</td>"
        f"<td style='padding:6px 12px;color:#fff;font-weight:bold'>{v}</td></tr>"
        for k, v in rows_stats
    )

    # Agrupar imágenes por carpeta relativa
    by_folder: dict[str, list[Path]] = {}
    for img in images:
        folder = img.parent.relative_to(BASE / "output/charts")
        by_folder.setdefault(str(folder), []).append(img)

    img_sections = ""
    for folder, imgs in sorted(by_folder.items()):
        items = "".join(
            f"<li style='color:#ccc;font-size:13px'>{i.name}</li>" for i in imgs
        )
        img_sections += f"""
        <div style='margin-bottom:16px'>
            <div style='color:#e53935;font-weight:bold;font-size:14px;
                        text-transform:uppercase;margin-bottom:4px'>{folder}</div>
            <ul style='margin:0;padding-left:20px'>{items}</ul>
        </div>"""

    if not img_sections:
        img_sections = "<p style='color:#888'>Sin imágenes nuevas hoy.</p>"

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="background:#121212;font-family:Arial,sans-serif;padding:24px">

  <div style="max-width:600px;margin:0 auto">

    <div style="background:#1e1e1e;border-top:4px solid #e53935;
                border-radius:6px;padding:24px;margin-bottom:20px">
      <div style="color:#e53935;font-size:22px;font-weight:bold;
                  letter-spacing:2px">MAU-STATISTICS</div>
      <div style="color:#888;font-size:13px;margin-top:4px">
        Resumen diario — {TODAY}
      </div>
    </div>

    <div style="background:#1e1e1e;border-radius:6px;padding:20px;margin-bottom:20px">
      <div style="color:#fff;font-size:16px;font-weight:bold;
                  margin-bottom:12px">📊 Pipeline</div>
      <table style="width:100%;border-collapse:collapse">
        {rows_html}
      </table>
    </div>

    <div style="background:#1e1e1e;border-radius:6px;padding:20px;margin-bottom:20px">
      <div style="color:#fff;font-size:16px;font-weight:bold;
                  margin-bottom:12px">🖼️ Imágenes adjuntas ({len(images)})</div>
      {img_sections}
    </div>

    <div style="color:#555;font-size:11px;text-align:center;margin-top:16px">
      Generado automáticamente por MAU-STATISTICS Bot · @Miau_Stats_MX
    </div>

  </div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Enviar
# ─────────────────────────────────────────────────────────────────────────────
def send_email(app_password: str, dry_run: bool = False) -> bool:
    summary = load_summary()
    images  = collect_all_relevant()
    html    = build_html(summary, images)

    subject = f"MAU-STATISTICS · {TODAY} · {len(images)} imágenes"

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = FROM_ADDR
    msg["To"]      = TO_ADDR

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt)

    # Adjuntar imágenes
    attached = 0
    for img_path in images:
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
