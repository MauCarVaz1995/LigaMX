#!/usr/bin/env python3
"""
00_daily_pipeline.py — MAU-STATISTICS Daily Automation Pipeline
================================================================
Orquesta la actualización diaria completa sin intervención humana.

Pasos:
  1. Actualizar resultados Liga MX (historico_clausura_XXXX.json)
  2. Recalcular ELO Liga MX incremental
  3. Actualizar resultados internacionales
  4. Recalcular ELO selecciones incremental
  5. Actualizar tracker de predicciones con resultados reales
  6. Generar predicciones de partidos de HOY
  7. Git commit y push automático
  8. Log de ejecución en logs/pipeline_YYYYMMDD.log

Guardia: si ya corrió hoy (logs/daily_summary.json), no vuelve a correr.
"""

import json
import math
import subprocess
import sys
import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).resolve().parent.parent
SCRIPTS_DIR   = BASE / "scripts"
LOGS_DIR      = BASE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

TODAY         = date.today().isoformat()          # "2026-04-12"
TODAY_COMPACT = TODAY.replace("-", "")            # "20260412"

LOG_FILE      = LOGS_DIR / f"pipeline_{TODAY_COMPACT}.log"
SUMMARY_FILE  = LOGS_DIR / "daily_summary.json"

HIST_DIR      = BASE / "data/raw/historico"
ELO_CSV       = BASE / "data/processed/elo_historico.csv"
INTL_CSV      = BASE / "data/raw/internacional/results.csv"
PRED_LOG      = BASE / "data/processed/predicciones_log.csv"
RAW_FOTMOB    = BASE / "data/raw/fotmob"
RAW_FOTMOB.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES ELO
# ─────────────────────────────────────────────────────────────────────────────
K_LIGA    = 32
HOME_ADV  = 100
SCALE     = 400
K_INTL_MAP = {
    "FIFA World Cup": 60,
    "Copa América": 60,
    "CONCACAF Gold Cup": 35,
    "CONCACAF Nations League": 35,
    "UEFA Nations League": 35,
    "Copa MX": 20,
    "Friendly": 20,
}
K_INTL_DEFAULT = 25

FOTMOB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.fotmob.com/",
}

LIGA_NAME_MAP = {
    "CF America": "América", "Atletico de San Luis": "San Luis",
    "Queretaro FC": "Querétaro", "FC Juarez": "FC Juárez",
    "Mazatlan FC": "Mazatlán",
}
def norm_liga(n): return LIGA_NAME_MAP.get(str(n).strip(), str(n).strip())

INTL_NAME_FIXES = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
}
def norm_intl(n): return INTL_NAME_FIXES.get(str(n).strip(), str(n).strip())

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_with_retry(url: str, retries: int = 3, backoff: int = 30) -> dict | None:
    """GET JSON con reintentos y backoff exponencial."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=FOTMOB_HEADERS, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = backoff * attempt
            log.warning(f"  Intento {attempt}/{retries} fallido: {e}")
            if attempt < retries:
                log.info(f"  Reintentando en {wait}s…")
                time.sleep(wait)
    log.error(f"  Todos los intentos fallaron para {url}")
    return None


def run_script(script_path: str, args: list[str] = None) -> bool:
    """Ejecuta un script Python dentro del venv. Retorna True si exitoso."""
    python = BASE / ".venv/bin/python3"
    if not python.exists():
        python = Path(sys.executable)
    cmd = [str(python), str(SCRIPTS_DIR / script_path)] + (args or [])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE))
    if result.returncode != 0:
        log.error(f"  STDERR: {result.stderr[-500:]}")
        return False
    if result.stdout:
        log.info(result.stdout[-1000:])
    return True


def save_summary(summary: dict):
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def elo_expected(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / SCALE))


def goal_mult(gl: int, gv: int) -> float:
    diff = abs(gl - gv)
    return 1.0 if diff == 0 else 1.0 + math.log(diff + 1) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# GUARDIA: ya corrió hoy?
# ─────────────────────────────────────────────────────────────────────────────
def already_ran_today() -> bool:
    if not SUMMARY_FILE.exists():
        return False
    try:
        summary = json.loads(SUMMARY_FILE.read_text())
        return summary.get("date") == TODAY
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — Actualizar resultados Liga MX
# ─────────────────────────────────────────────────────────────────────────────
def step1_update_ligamx() -> dict:
    log.info("── PASO 1: Actualizar historico_clausura_2026.json ──")
    result = {"nuevos": 0, "terminados_antes": 0, "terminados_despues": 0, "error": None}

    hist_file = HIST_DIR / "historico_clausura_2026.json"
    if not hist_file.exists():
        result["error"] = "Archivo no encontrado"
        return result

    # Estado antes
    with open(hist_file) as f:
        d = json.load(f)
    antes = sum(1 for p in d["partidos"] if p.get("terminado"))
    result["terminados_antes"] = antes

    # Re-descargar con --force solo Clausura 2026
    import re
    script = SCRIPTS_DIR / "10_descargar_historico.py"
    python = BASE / ".venv/bin/python3"
    if not python.exists():
        python = Path(sys.executable)

    cmd = [str(python), str(script), "--download", "--force"]
    env_override = {}  # script descarga los 6 torneos en TARGETS; Clausura siempre primero
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE), timeout=120)
    if proc.returncode != 0:
        result["error"] = proc.stderr[-300:]
        log.error(f"  Error en descarga: {result['error']}")
        return result

    # Estado después
    with open(hist_file) as f:
        d = json.load(f)
    despues = sum(1 for p in d["partidos"] if p.get("terminado"))
    result["terminados_despues"] = despues
    result["nuevos"] = despues - antes
    log.info(f"  Terminados: {antes} → {despues} (+{result['nuevos']} nuevos)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — ELO Liga MX incremental
# ─────────────────────────────────────────────────────────────────────────────
def step2_elo_ligamx(nuevos_count: int) -> dict:
    log.info("── PASO 2: ELO Liga MX incremental ──")
    result = {"equipos_afectados": 0, "registros_nuevos": 0, "error": None}

    if nuevos_count == 0:
        log.info("  Sin partidos nuevos — ELO sin cambios")
        return result

    hist_file = HIST_DIR / "historico_clausura_2026.json"
    try:
        df_elo = pd.read_csv(ELO_CSV)
        elos = df_elo.groupby("equipo")["elo"].last().to_dict()

        # Fecha del último registro en el CSV
        ultima_fecha = df_elo["fecha"].max()

        with open(hist_file) as f:
            d = json.load(f)

        # Solo partidos posteriores al último registro en el CSV
        nuevos = [
            p for p in d["partidos"]
            if p.get("terminado") and p["fecha"][:10] > ultima_fecha
        ]
        nuevos.sort(key=lambda x: x["fecha"])

        if not nuevos:
            log.info("  ELO ya está al día")
            return result

        registros = []
        afectados = set()
        for p in nuevos:
            loc = norm_liga(p["local"])
            vis = norm_liga(p["visitante"])
            gl, gv = p["goles_local"], p["goles_visit"]
            fecha = p["fecha"][:10]

            elo_l = elos.get(loc, 1500)
            elo_v = elos.get(vis, 1500)

            exp_l = elo_expected(elo_l + HOME_ADV, elo_v)
            res_l = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
            res_v = 1.0 - res_l
            mult  = goal_mult(gl, gv)

            elo_l_new = elo_l + K_LIGA * mult * (res_l - exp_l)
            elo_v_new = elo_v + K_LIGA * mult * (res_v - (1 - exp_l))

            elos[loc] = elo_l_new
            elos[vis] = elo_v_new
            afectados.update([loc, vis])

            registros.append({"fecha": fecha, "equipo": loc, "elo": round(elo_l_new, 2), "torneo": "2025/2026 - Clausura"})
            registros.append({"fecha": fecha, "equipo": vis, "elo": round(elo_v_new, 2), "torneo": "2025/2026 - Clausura"})

        df_new = pd.DataFrame(registros)
        pd.concat([df_elo, df_new], ignore_index=True).to_csv(ELO_CSV, index=False)

        result["equipos_afectados"] = len(afectados)
        result["registros_nuevos"]  = len(registros)
        log.info(f"  {len(nuevos)} partidos procesados, {len(afectados)} equipos actualizados")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  Error ELO Liga MX: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — Resultados internacionales
# ─────────────────────────────────────────────────────────────────────────────
def step3_update_intl() -> dict:
    log.info("── PASO 3: Resultados internacionales ──")
    result = {"nuevos": 0, "fechas_procesadas": 0, "error": None}

    try:
        df = pd.read_csv(INTL_CSV)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        existing = set(zip(df["date"], df["home_team"], df["away_team"]))

        # Cubrir los últimos 3 días (por si alguno se escapó) + hoy
        fechas = [
            (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(3, -1, -1)
        ]

        total_nuevos = 0
        for date_str in fechas:
            url = f"https://www.fotmob.com/api/data/matches?date={date_str.replace('-','')}"
            data = fetch_with_retry(url)
            if not data:
                continue

            # Cachear raw
            raw_file = RAW_FOTMOB / f"intl_{date_str.replace('-','')}.json"
            raw_file.write_text(json.dumps(data), encoding="utf-8")

            # Parsear internacionales
            nuevos = []
            for league in data.get("leagues", []):
                if league.get("ccode", "") != "INT":
                    continue
                for m in league.get("matches", []):
                    if not m.get("status", {}).get("finished", False):
                        continue
                    home  = m.get("home", {})
                    away  = m.get("away", {})
                    hs, as_ = home.get("score"), away.get("score")
                    if hs is None or as_ is None:
                        continue
                    key = (date_str, home.get("name", ""), away.get("name", ""))
                    if key in existing:
                        continue
                    existing.add(key)
                    nuevos.append({
                        "date": date_str,
                        "home_team": home.get("name", ""),
                        "away_team": away.get("name", ""),
                        "home_score": int(hs),
                        "away_score": int(as_),
                        "tournament": league.get("name", "International"),
                        "city": "", "country": "", "neutral": True,
                    })

            if nuevos:
                df = pd.concat([df, pd.DataFrame(nuevos)], ignore_index=True)
                total_nuevos += len(nuevos)
                log.info(f"  {date_str}: {len(nuevos)} nuevos")
            else:
                log.info(f"  {date_str}: 0 nuevos")

            result["fechas_procesadas"] += 1
            time.sleep(1.2)

        if total_nuevos > 0:
            df.to_csv(INTL_CSV, index=False)

        result["nuevos"] = total_nuevos
        log.info(f"  Total nuevos: {total_nuevos} | CSV: {len(df)} partidos")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  Error internacionales: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — ELO selecciones incremental
# ─────────────────────────────────────────────────────────────────────────────
def step4_elo_selecciones() -> dict:
    log.info("── PASO 4: ELO selecciones incremental ──")
    result = {"selecciones_afectadas": 0, "nuevo_archivo": None, "error": None}

    try:
        # Encontrar el archivo ELO más reciente
        elo_files = sorted(BASE.glob("data/processed/elos_selecciones_*.json"))
        if not elo_files:
            result["error"] = "No se encontró archivo ELO selecciones"
            return result

        latest_file = elo_files[-1]
        # Fecha del archivo (nombre: elos_selecciones_YYYYMMDD.json)
        file_date_str = latest_file.stem.replace("elos_selecciones_", "")  # "20260412"
        file_date = date(int(file_date_str[:4]), int(file_date_str[4:6]), int(file_date_str[6:8]))

        # Si el archivo es de hoy, no hay nada que actualizar
        if file_date >= date.today():
            log.info(f"  ELO selecciones ya al día ({latest_file.name})")
            return result

        elos = json.loads(latest_file.read_text())

        # Partidos desde el día siguiente al archivo hasta hoy
        since = (file_date + timedelta(days=1)).strftime("%Y-%m-%d")
        df_intl = pd.read_csv(INTL_CSV, parse_dates=["date"])
        nuevos = df_intl[df_intl["date"] >= since].copy().sort_values("date")

        if nuevos.empty:
            log.info(f"  Sin partidos nuevos desde {since}")
            return result

        afectadas = set()
        for _, r in nuevos.iterrows():
            home = norm_intl(r["home_team"])
            away = norm_intl(r["away_team"])
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            neutral = bool(r.get("neutral", False))
            tourn   = str(r.get("tournament", ""))

            k = next((v for kw, v in K_INTL_MAP.items() if kw.lower() in tourn.lower()), K_INTL_DEFAULT)

            elo_h = elos.get(home, 1500.0)
            elo_a = elos.get(away, 1500.0)

            adv = 0 if neutral else HOME_ADV
            exp_h = elo_expected(elo_h + adv, elo_a)
            res_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            res_a = 1.0 - res_h
            mult  = goal_mult(hs, as_)

            elos[home] = elos.get(home, 1500.0) + k * mult * (res_h - exp_h)
            elos[away] = elos.get(away, 1500.0) + k * mult * (res_a - (1 - exp_h))
            afectadas.update([home, away])

        # Guardar nuevo estado con fecha de hoy
        out_file = BASE / f"data/processed/elos_selecciones_{TODAY_COMPACT}.json"
        out_file.write_text(json.dumps(elos, ensure_ascii=False, indent=2), encoding="utf-8")

        result["selecciones_afectadas"] = len(afectadas)
        result["nuevo_archivo"] = out_file.name
        log.info(f"  {len(afectadas)} selecciones actualizadas → {out_file.name}")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  Error ELO selecciones: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — Actualizar tracker de predicciones
# ─────────────────────────────────────────────────────────────────────────────
def step5_update_tracker() -> dict:
    log.info("── PASO 5: Actualizar tracker de predicciones ──")
    result = {"actualizados": 0, "pendientes": 0, "aciertos": 0, "total_evaluados": 0, "error": None}

    if not PRED_LOG.exists():
        log.info("  Sin tracker — saltando")
        return result

    try:
        df = pd.read_csv(PRED_LOG)
        pendientes_idx = df[df["resultado_real"].isna()].index.tolist()
        result["pendientes"] = len(pendientes_idx)

        if not pendientes_idx:
            log.info("  Sin predicciones pendientes")
            return result

        # Construir índice de resultados Liga MX
        ligamx_res = {}
        hist_file = HIST_DIR / "historico_clausura_2026.json"
        if hist_file.exists():
            with open(hist_file) as f:
                d = json.load(f)
            for p in d["partidos"]:
                if not p.get("terminado"):
                    continue
                key = (p["fecha"][:10], norm_liga(p["local"]), norm_liga(p["visitante"]))
                ligamx_res[key] = (p["goles_local"], p["goles_visit"])

        # Construir índice de resultados internacionales
        intl_res = {}
        df_intl = pd.read_csv(INTL_CSV)
        df_intl["date"] = pd.to_datetime(df_intl["date"]).dt.strftime("%Y-%m-%d")
        for _, r in df_intl.iterrows():
            key = (str(r["date"]), norm_intl(r["home_team"]), norm_intl(r["away_team"]))
            intl_res[key] = (int(r["home_score"]), int(r["away_score"]))

        actualizados = 0
        for idx in pendientes_idx:
            row   = df.loc[idx]
            fecha = str(row.get("fecha_partido", ""))[:10]
            loc   = str(row.get("equipo_local", "")).strip()
            vis   = str(row.get("equipo_visitante", "")).strip()

            # Buscar en Liga MX primero, luego en internacionales
            score = ligamx_res.get((fecha, norm_liga(loc), norm_liga(vis)))
            if score is None:
                score = intl_res.get((fecha, norm_intl(loc), norm_intl(vis)))
            if score is None:
                continue

            gl, gv = score
            resultado = "local" if gl > gv else ("empate" if gl == gv else "visitante")
            pred = str(row.get("ganador_predicho", "")).lower().strip()
            acierto = (pred == resultado)

            df.at[idx, "resultado_real"]       = resultado
            df.at[idx, "goles_local_real"]     = gl
            df.at[idx, "goles_visitante_real"] = gv
            df.at[idx, "acierto_ganador"]      = acierto
            actualizados += 1

        if actualizados > 0:
            df.to_csv(PRED_LOG, index=False)

        con_res = df[df["resultado_real"].notna()]
        aciertos = int(con_res["acierto_ganador"].sum()) if len(con_res) else 0

        result["actualizados"]    = actualizados
        result["total_evaluados"] = len(con_res)
        result["aciertos"]        = aciertos
        pct = f"{100*aciertos/len(con_res):.1f}%" if con_res.size else "N/A"
        log.info(f"  {actualizados} actualizados | {aciertos}/{len(con_res)} aciertos ({pct})")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  Error tracker: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — Predicciones de hoy
# ─────────────────────────────────────────────────────────────────────────────
def step6_predicciones_hoy() -> dict:
    log.info("── PASO 6: Predicciones de hoy ──")
    result = {"predicciones_generadas": 0, "saltado": False, "error": None}

    try:
        # Verificar si hay partidos de Liga MX hoy
        hist_file = HIST_DIR / "historico_clausura_2026.json"
        partidos_hoy = []
        if hist_file.exists():
            with open(hist_file) as f:
                d = json.load(f)
            partidos_hoy = [
                p for p in d["partidos"]
                if not p.get("terminado") and p["fecha"][:10] == TODAY
            ]

        # generar_prediccion.py maneja Liga MX + CCL + Internacional
        log.info("  Ejecutando generar_prediccion.py --competition all")
        ok = run_script("generar_prediccion.py", ["--competition", "all", "--date", TODAY])
        if ok:
            result["predicciones_generadas"] += max(len(partidos_hoy), 1)
        else:
            result["error"] = "Error en generar_prediccion.py"

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  Error predicciones: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7 — Git commit y push
# ─────────────────────────────────────────────────────────────────────────────
def step7_git_push(summary: dict) -> dict:
    log.info("── PASO 7: Git commit y push ──")
    result = {"committed": False, "pushed": False, "error": None}

    try:
        # Verificar si hay cambios reales
        status = subprocess.run(
            ["git", "diff", "--staged", "--quiet"],
            capture_output=True, cwd=str(BASE)
        )
        # Agregar archivos de datos y logs
        subprocess.run(
            ["git", "add",
             "data/processed/", "data/raw/historico/", "data/raw/internacional/",
             "data/processed/predicciones_log.csv",
             "logs/daily_summary.json",
             f"data/processed/elos_selecciones_{TODAY_COMPACT}.json"],
            capture_output=True, cwd=str(BASE)
        )

        # Verificar si hay algo staged
        diff = subprocess.run(
            ["git", "diff", "--staged", "--name-only"],
            capture_output=True, text=True, cwd=str(BASE)
        )
        files_changed = [l for l in diff.stdout.strip().splitlines() if l]

        if not files_changed:
            log.info("  Sin cambios para commitear")
            return result

        # Construir mensaje de commit
        s1 = summary.get("paso1", {})
        s3 = summary.get("paso3", {})
        nuevos_liga = s1.get("nuevos", 0)
        nuevos_intl = s3.get("nuevos", 0)
        msg_parts = [f"auto: daily update {TODAY}"]
        if nuevos_liga: msg_parts.append(f"Liga MX +{nuevos_liga}")
        if nuevos_intl: msg_parts.append(f"intl +{nuevos_intl}")
        msg = " · ".join(msg_parts)

        commit = subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True, text=True, cwd=str(BASE)
        )
        if commit.returncode == 0:
            result["committed"] = True
            log.info(f"  Commit: {msg}")
        else:
            result["error"] = commit.stderr
            log.error(f"  Error commit: {commit.stderr}")
            return result

        push = subprocess.run(
            ["git", "push"],
            capture_output=True, text=True, cwd=str(BASE)
        )
        if push.returncode == 0:
            result["pushed"] = True
            log.info("  Push exitoso")
        else:
            result["error"] = push.stderr
            log.error(f"  Error push: {push.stderr}")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  Error git: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info(f"MAU-STATISTICS Daily Pipeline — {TODAY}")
    log.info("=" * 70)

    # Guardia: no re-ejecutar si ya corrió hoy
    if already_ran_today():
        log.info("Pipeline ya ejecutado hoy. Saliendo.")
        sys.exit(0)

    summary = {"date": TODAY, "start": datetime.now().isoformat(), "pasos": {}}
    start_time = time.time()

    # ── Ejecutar pasos ──────────────────────────────────────────────────────
    steps = [
        ("paso1", "Liga MX update",         lambda: step1_update_ligamx()),
        ("paso2", "ELO Liga MX",             lambda s: step2_elo_ligamx(s["paso1"].get("nuevos", 0))),
        ("paso3", "Internacionales update",  lambda: step3_update_intl()),
        ("paso4", "ELO Selecciones",         lambda: step4_elo_selecciones()),
        ("paso5", "Tracker predicciones",    lambda: step5_update_tracker()),
        ("paso6", "Predicciones de hoy",     lambda: step6_predicciones_hoy()),
    ]

    for key, name, fn in steps:
        t0 = time.time()
        log.info("")
        try:
            # Pasos que dependen del resultado del anterior reciben summary
            import inspect
            sig = inspect.signature(fn)
            if sig.parameters:
                paso_result = fn(summary["pasos"])
            else:
                paso_result = fn()
        except Exception as e:
            paso_result = {"error": str(e)}
            log.error(f"  Paso {key} falló: {e}")

        paso_result["elapsed_s"] = round(time.time() - t0, 1)
        summary["pasos"][key] = paso_result
        status = "✓" if not paso_result.get("error") else "✗"
        log.info(f"  [{status}] {name} completado en {paso_result['elapsed_s']}s")

    # ── Post-partido (no-crítico) ───────────────────────────────────────────
    log.info("")
    log.info("── PASO 8 (no-crítico): Infografías post-partido ──")
    t0 = time.time()
    ok8 = run_script("gen_postpartido.py", ["--days", "2"])
    summary["pasos"]["paso8"] = {
        "ok": ok8,
        "elapsed_s": round(time.time() - t0, 1),
        "error": None if ok8 else "gen_postpartido.py falló",
    }
    log.info(f"  [{'✓' if ok8 else '✗'}] Post-partido completado en {summary['pasos']['paso8']['elapsed_s']}s")

    # ── Git commit ──────────────────────────────────────────────────────────
    log.info("")
    summary["pasos"]["paso7"] = step7_git_push(summary["pasos"])

    # ── Guardar summary antes del email (el email lo lee) ───────────────────
    summary["end"]       = datetime.now().isoformat()
    summary["elapsed_s"] = round(time.time() - start_time, 1)
    CRITICAL_PRE = {"paso1", "paso2", "paso3", "paso4"}
    summary["success"] = not any(summary["pasos"].get(k, {}).get("error") for k in CRITICAL_PRE)
    save_summary(summary)

    # ── Guardar summary ─────────────────────────────────────────────────────
    summary["end"]       = datetime.now().isoformat()
    summary["elapsed_s"] = round(time.time() - start_time, 1)

    # Pasos críticos vs no-críticos
    # Solo falla el pipeline si falla la adquisición de datos o el ELO
    CRITICAL = {"paso1", "paso2", "paso3", "paso4"}
    NON_CRITICAL = {"paso5", "paso6", "paso7"}
    critical_failed   = any(summary["pasos"].get(k, {}).get("error") for k in CRITICAL)
    noncritical_errors = [k for k in NON_CRITICAL if summary["pasos"].get(k, {}).get("error")]
    summary["success"] = not critical_failed
    summary["critical_failed"]    = critical_failed
    summary["noncritical_errors"] = noncritical_errors

    save_summary(summary)

    log.info("")
    log.info("=" * 70)
    status_str = "OK" if not critical_failed else "FALLO CRÍTICO"
    if noncritical_errors:
        status_str += f" (no-críticos con error: {', '.join(noncritical_errors)})"
    log.info(f"Pipeline finalizado en {summary['elapsed_s']}s — {status_str}")
    log.info(f"Log: {LOG_FILE}")
    log.info(f"Summary: {SUMMARY_FILE}")
    log.info("=" * 70)

    # Imprimir resumen compacto
    print("\n── RESUMEN ──────────────────────────────────────────")
    p = summary["pasos"]
    print(f"  Liga MX nuevos:      {p.get('paso1', {}).get('nuevos', 0)}")
    print(f"  Intl nuevos:         {p.get('paso3', {}).get('nuevos', 0)}")
    print(f"  ELO equipos:         {p.get('paso2', {}).get('equipos_afectados', 0)}")
    print(f"  ELO selecciones:     {p.get('paso4', {}).get('selecciones_afectadas', 0)}")
    print(f"  Tracker actualiz.:   {p.get('paso5', {}).get('actualizados', 0)}")
    aciertos = p.get("paso5", {}).get("aciertos", 0)
    total    = p.get("paso5", {}).get("total_evaluados", 0)
    print(f"  Aciertos modelo:     {aciertos}/{total}")
    print(f"  Predicciones hoy:    {p.get('paso6', {}).get('predicciones_generadas', 0)}")
    print(f"  Git push:            {'✓' if p.get('paso7', {}).get('pushed') else '✗'}")
    if noncritical_errors:
        print(f"  [WARN] Errores no-críticos: {', '.join(noncritical_errors)}")
    print("─────────────────────────────────────────────────────")

    # Exit 1 solo si fallaron pasos críticos de datos
    sys.exit(1 if critical_failed else 0)


if __name__ == "__main__":
    main()
