#!/usr/bin/env python3
"""
odds_scraper_bot.py — Bot de cuotas Caliente / BetCris / Codere
================================================================
Las tres principales casas accesibles desde México NO están en The Odds API.
Este bot las raspa directamente usando Playwright (navegador headless).

Instalación (una sola vez):
  pip install playwright
  playwright install chromium --with-deps

GitHub Actions: se puede agregar a hourly_bets.yml si se instala playwright.

Uso:
  python bots/odds_scraper_bot.py                     # las tres casas
  python bots/odds_scraper_bot.py --casa caliente
  python bots/odds_scraper_bot.py --casa betcris
  python bots/odds_scraper_bot.py --casa codere
  python bots/odds_scraper_bot.py --targets           # guía cuotas mínimas sin scraping

Output:
  data/processed/odds_mx_local.csv — cuotas unificadas de las 3 casas
  Columnas: casa, fecha, local, visitante, odd_1, odd_X, odd_2, odd_over25, odd_under25
"""

import argparse
import json
import re
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE     = Path(__file__).resolve().parent.parent
SCRIPTS  = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))

ODDS_MX_LOCAL = BASE / "data/processed/odds_mx_local.csv"
MX_TZ  = timezone(timedelta(hours=-6))
TODAY  = datetime.now(MX_TZ).strftime("%Y-%m-%d")
NOW_H  = datetime.now(MX_TZ).strftime("%H:%M")

# Normalización de nombres de equipos MX → nuestro sistema
TEAM_NORM = {
    "America":           "América",       "CF America": "América",
    "Club America":      "América",       "América":    "América",
    "Cruz Azul":         "Cruz Azul",
    "Chivas":            "Guadalajara",   "Guadalajara": "Guadalajara",
    "Monterrey":         "Monterrey",     "CF Monterrey": "Monterrey",
    "Tigres":            "Tigres",        "Tigres UANL":  "Tigres",
    "Pumas":             "Pumas",         "Pumas UNAM":   "Pumas",
    "Toluca":            "Toluca",
    "Atlas":             "Atlas",
    "Santos":            "Santos Laguna", "Santos Laguna": "Santos Laguna",
    "Necaxa":            "Necaxa",
    "Mazatlan":          "Mazatlán",      "Mazatlán FC":   "Mazatlán",
    "Juarez":            "FC Juárez",     "FC Juárez":     "FC Juárez",
    "San Luis":          "San Luis",      "Atletico San Luis": "San Luis",
    "Queretaro":         "Querétaro",
    "Tijuana":           "Tijuana",       "Xolos":          "Tijuana",
    "Leon":              "León",          "León":           "León",
    "Puebla":            "Puebla",
    "Pachuca":           "Pachuca",
}

def norm(name: str) -> str:
    return TEAM_NORM.get(name.strip(), name.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Scraper genérico con Playwright
# ─────────────────────────────────────────────────────────────────────────────
def _playwright_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa
        return True
    except ImportError:
        return False


def scrape_with_playwright(url: str, casa: str) -> list[dict]:
    """Scraper genérico — intercepts XHR + parsea DOM fallback."""
    from playwright.sync_api import sync_playwright

    partidos = []
    xhr_data = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
            locale="es-MX",
            timezone_id="America/Mexico_City",
        )
        page = ctx.new_page()

        # Interceptar respuestas JSON de odds/events/markets
        def on_response(resp):
            ct = resp.headers.get("content-type", "")
            if not any(x in resp.url for x in ["odds","event","market","sport","match","fixture"]):
                return
            if "json" not in ct:
                return
            try:
                xhr_data.append({"url": resp.url, "data": resp.json()})
            except Exception:
                pass

        page.on("response", on_response)

        try:
            page.goto(url, wait_until="networkidle", timeout=35000)
            page.wait_for_timeout(3000)
        except Exception as e:
            print(f"  [warn] {casa}: goto timeout — {e}")

        # ── Intentar parsear del DOM por patrones comunes ──
        for selector in [
            "[data-testid*='event']", "[class*='event-row']",
            "[class*='match-row']", "[class*='fixture']",
            "[data-event-id]", "[data-match-id]",
        ]:
            events = page.query_selector_all(selector)
            if events:
                print(f"  {casa}: {len(events)} eventos via selector {selector}")
                for ev in events:
                    try:
                        text = ev.inner_text()
                        # Extraer equipos y cuotas con regex
                        teams = re.findall(r'[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü\s]+(?:FC|MX)?', text)
                        odds  = re.findall(r'\b\d\.\d{2}\b', text)
                        if len(teams) >= 2 and len(odds) >= 3:
                            partidos.append({
                                "casa":       casa,
                                "fecha":      TODAY,
                                "local":      norm(teams[0]),
                                "visitante":  norm(teams[1]),
                                "odd_1":      float(odds[0]),
                                "odd_X":      float(odds[1]),
                                "odd_2":      float(odds[2]),
                                "odd_over25": None,
                                "odd_under25":None,
                            })
                    except Exception:
                        pass
                if partidos:
                    break

        # ── Fallback: analizar XHR capturadas ──
        if not partidos and xhr_data:
            print(f"  {casa}: analizando {len(xhr_data)} XHR interceptadas")
            for item in xhr_data[:10]:
                d = item["data"]
                # Intentar encontrar estructura de eventos
                events_list = []
                if isinstance(d, list):
                    events_list = d
                elif isinstance(d, dict):
                    for key in ["events","data","matches","fixtures","items"]:
                        if key in d and isinstance(d[key], list):
                            events_list = d[key]
                            break
                for event in events_list[:20]:
                    if not isinstance(event, dict):
                        continue
                    # Nombres de campos comunes en diferentes APIs
                    home  = (event.get("home") or event.get("homeTeam",{}) or
                             event.get("home_team",""))
                    away  = (event.get("away") or event.get("awayTeam",{}) or
                             event.get("away_team",""))
                    if isinstance(home, dict): home = home.get("name", home.get("shortName","?"))
                    if isinstance(away, dict): away = away.get("name", away.get("shortName","?"))
                    if not home or not away:
                        continue
                    # Odds en campos típicos
                    odds = (event.get("odds") or event.get("markets") or
                            event.get("selections",[]))
                    o1 = ox = o2 = None
                    if isinstance(odds, dict):
                        o1 = odds.get("1") or odds.get("home") or odds.get("W1")
                        ox = odds.get("X") or odds.get("draw") or odds.get("X2")
                        o2 = odds.get("2") or odds.get("away") or odds.get("W2")
                    partidos.append({
                        "casa": casa, "fecha": TODAY,
                        "local": norm(str(home)), "visitante": norm(str(away)),
                        "odd_1": float(o1) if o1 else None,
                        "odd_X": float(ox) if ox else None,
                        "odd_2": float(o2) if o2 else None,
                        "odd_over25": None, "odd_under25": None,
                    })

        browser.close()

    return partidos


# ─────────────────────────────────────────────────────────────────────────────
# Casas de apuestas
# ─────────────────────────────────────────────────────────────────────────────
CASAS = {
    "caliente": {
        "url":  "https://www.caliente.mx/deportes/futbol/mexico/liga-mx",
        "nombre": "Caliente.mx",
    },
    "betcris": {
        "url":  "https://www.betcris.mx/es/apuestas-deportivas/futbol/mexico/liga-mx",
        "nombre": "BetCris.mx",
    },
    "codere": {
        "url":  "https://www.codere.mx/apuestas-deportivas/futbol/mexico/liga-mx",
        "nombre": "Codere.mx",
    },
}


def scrape_casa(casa_key: str) -> list[dict]:
    cfg = CASAS[casa_key]
    print(f"\n  ── {cfg['nombre']} ──")
    if not _playwright_available():
        print("  [SKIP] Playwright no instalado.")
        print("  Instalar: pip install playwright && playwright install chromium --with-deps")
        return []
    partidos = scrape_with_playwright(cfg["url"], casa_key)
    print(f"  {len(partidos)} partidos encontrados")
    return partidos


def save_odds(all_partidos: list[dict]):
    if not all_partidos:
        return
    df_new = pd.DataFrame(all_partidos)
    df_new["descargado"] = TODAY

    if ODDS_MX_LOCAL.exists():
        df_old = pd.read_csv(ODDS_MX_LOCAL)
        df_out = pd.concat([df_old, df_new]).drop_duplicates(
            subset=["casa","fecha","local","visitante"], keep="last"
        ).reset_index(drop=True)
    else:
        df_out = df_new

    df_out.to_csv(ODDS_MX_LOCAL, index=False)
    print(f"\n  ✅ Guardado: {ODDS_MX_LOCAL} ({len(df_out)} registros)")


# ─────────────────────────────────────────────────────────────────────────────
# Guía de cuotas mínimas (sin scraping)
# ─────────────────────────────────────────────────────────────────────────────
def show_targets(bankroll: float = 1000.0):
    """
    Muestra la cuota mínima requerida en Caliente/BetCris/Codere para cada
    apuesta del portafolio. El usuario la revisa manualmente.
    """
    try:
        from send_bets_email import load_model_probs, get_ligamx_today, fetch_ligamx_odds
        partidos = get_ligamx_today()
        probs    = load_model_probs()
        odds_api = fetch_ligamx_odds()
    except Exception as e:
        print(f"  [warn] {e}")
        return

    MIN_EV = 0.04

    print(f"\n{'═'*72}")
    print(f"  DONDE APOSTAR — guía manual para CALIENTE / BETCRIS / CODERE")
    print(f"  EV mínimo requerido: {MIN_EV:.0%} · Bankroll: ${bankroll:.0f} MXN")
    print(f"{'═'*72}")
    print(f"\n  Mejor alternativa digital (sin VPN): 1xBet · Betway · BetOnline")
    print(f"  Con VPN (mejor precio): Betfair Exchange · Smarkets\n")

    headers = ("Partido", "Apuesta", "Prob", "CuotaMin", "MejorAPI", "Acción")
    print(f"  {headers[0]:<30} {headers[1]:<18} {headers[2]:>5} "
          f"{headers[3]:>9} {headers[4]:>9} {headers[5]}")
    print(f"  {'─'*75}")

    mercados = [
        ("1x2_local",          "1 (Local)",      lambda p: None),
        ("1x2_draw",           "X Empate",        lambda p: None),
        ("1x2_visita",         "2 (Visita)",      lambda p: None),
        ("over_2.5",           "Over 2.5",        lambda p: odds_api.get(
            (p["local"].lower(), p.get("visitante",p.get("visita","")).lower()),{}
        ).get("best_over25")),
        ("under_2.5",          "Under 2.5",       lambda p: odds_api.get(
            (p["local"].lower(), p.get("visitante",p.get("visita","")).lower()),{}
        ).get("best_under25")),
        ("tarjetas_over_4.5",  "Tarjetas O4.5",   lambda p: None),
        ("btts_si",            "Ambos Anotan",    lambda p: None),
    ]

    for p in partidos:
        lk = p["local"].lower()
        vk = p.get("visitante",p.get("visita","")).lower()
        ok = odds_api.get((lk, vk), {})
        partido = f"{p['local']} vs {p.get('visitante',p.get('visita',''))}"

        for key, label, get_api_odds in mercados:
            prob = probs.get((lk, vk, key))
            if prob is None or prob < 0.22:
                continue

            cuota_min = round(1 / prob * (1 + MIN_EV), 2)

            # Mejor cuota en The Odds API
            if key == "1x2_local":    best_api = ok.get("best_odd_1")
            elif key == "1x2_draw":   best_api = ok.get("best_odd_X")
            elif key == "1x2_visita": best_api = ok.get("best_odd_2")
            else:                     best_api = get_api_odds(p)

            ev_api = round(prob * best_api - 1, 3) if best_api else None
            bk_api = (ok.get("bk_odd_1") if key=="1x2_local" else
                      ok.get("bk_odd_X") if key=="1x2_draw" else
                      ok.get("bk_odd_2") if key=="1x2_visita" else "")

            if ev_api and ev_api >= MIN_EV:
                accion = f"✅ APUESTA en {bk_api or 'The Odds API'}"
                api_str = f"{best_api:.2f}"
            elif best_api:
                accion = f"sin valor en API ({best_api:.2f}<{cuota_min})"
                api_str = f"{best_api:.2f}❌"
            else:
                accion = f"busca ≥{cuota_min} en Caliente/BetCris/Codere"
                api_str = "—"

            print(f"  {partido[:29]:<30} {label[:17]:<18} {prob:>4.0%} "
                  f"{cuota_min:>9.2f} {api_str:>9}  {accion}")

    print(f"\n  Regla: si Caliente/BetCris/Codere ofrecen ≥ CuotaMin → apuesta con valor.")
    print(f"  Sin Playwright instalado, usa esta guía manualmente.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bot de cuotas Caliente / BetCris / Codere")
    parser.add_argument("--casa",    choices=["caliente","betcris","codere","all"], default="all")
    parser.add_argument("--targets", action="store_true",
                        help="Mostrar guía de cuotas mínimas (sin scraping)")
    parser.add_argument("--bankroll",type=float, default=1000.0)
    args = parser.parse_args()

    print(f"\n── odds_scraper_bot.py · {TODAY} {NOW_H} ──")

    if args.targets or not _playwright_available():
        if not _playwright_available():
            print("\n  [INFO] Playwright no disponible — mostrando guía manual")
            print("  Para scraping real: pip install playwright && playwright install chromium")
        show_targets(args.bankroll)
        return

    # Scraping con Playwright
    all_partidos = []
    casas = ["caliente","betcris","codere"] if args.casa == "all" else [args.casa]
    for casa in casas:
        partidos = scrape_casa(casa)
        all_partidos.extend(partidos)

    # Si Playwright corrió pero sin resultados, mostrar guía
    if not all_partidos:
        print("\n  Sin cuotas scraped — las estructuras HTML cambiaron o requieren login")
        print("  Mostrando guía manual:")
        show_targets(args.bankroll)
    else:
        save_odds(all_partidos)
        # Comparar con modelo
        try:
            from send_bets_email import load_model_probs, get_ligamx_today
            probs    = load_model_probs()
            partidos = get_ligamx_today()
            print("\n  ── Valor en casas mexicanas ──")
            for p in all_partidos:
                lk = p["local"].lower()
                vk = p["visitante"].lower()
                for key, label, odd_key in [
                    ("1x2_local", "Local", "odd_1"),
                    ("1x2_draw",  "Empate", "odd_X"),
                    ("1x2_visita","Visita", "odd_2"),
                ]:
                    prob  = probs.get((lk, vk, key))
                    cuota = p.get(odd_key)
                    if prob and cuota:
                        ev = round(prob * cuota - 1, 3)
                        if ev >= 0.04:
                            print(f"  ✅ VALUE {p['casa'].upper()}: {p['local']} vs {p['visitante']} "
                                  f"{label} @{cuota:.2f} EV={ev:+.1%}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
