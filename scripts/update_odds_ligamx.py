#!/usr/bin/env python3
"""
update_odds_ligamx.py — Cuotas en tiempo real para Liga MX via The Odds API
============================================================================
Descarga cuotas de Bet365, Betcris, Caliente y otras casas para los próximos
partidos de Liga MX. Guarda en data/processed/odds_ligamx.csv y actualiza
betting_log.csv con la cuota vista y el EV estimado.

The Odds API — free tier: 500 req/mes (suficiente para Liga MX completo)
  Liga MX sport key: "soccer_mexico_ligamx"
  Mercados: h2h (1X2), totals (Over/Under goles), btts

API key: variable de entorno ODDS_API_KEY
  Local:  export ODDS_API_KEY=xxxxxxxxxxxx
  GitHub: Settings → Secrets → Actions → ODDS_API_KEY

Uso:
  python scripts/update_odds_ligamx.py              # descarga + actualiza EV
  python scripts/update_odds_ligamx.py --dry-run    # muestra sin guardar
  python scripts/update_odds_ligamx.py --status     # cuántos requests quedan
"""

import argparse
import json
import os
import tempfile
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE         = Path(__file__).resolve().parent.parent
ODDS_LIGAMX  = BASE / "data/processed/odds_ligamx.csv"
BETTING_LOG  = BASE / "data/processed/betting_log.csv"

MX_TZ = timezone(timedelta(hours=-6))
TODAY = date.today().isoformat()

def utc_to_mx_date(utc_str: str) -> str:
    """Convierte timestamp UTC a fecha México City (UTC-6, sin DST desde 2023)."""
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        return dt.astimezone(MX_TZ).strftime("%Y-%m-%d")
    except Exception:
        return utc_str[:10]

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
# Cargar .env si existe (para uso local sin exportar variable)
_env_file = BASE / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

API_KEY      = os.environ.get("ODDS_API_KEY", "")
BASE_URL     = "https://api.the-odds-api.com/v4"
SPORT        = "soccer_mexico_ligamx"

# Casas de apuestas disponibles en The Odds API para Liga MX
# Orden: Pinnacle primero (menor margen, benchmark de eficiencia), luego por relevancia para MX
# Caliente no está disponible en The Odds API (sin API pública)
BOOKMAKERS   = [
    "pinnacle",       # benchmark eficiencia — margen ~2-3%
    "betsson",        # disponible MX — margen ~6-8%
    "onexbet",        # 1xBet — disponible en MX
    "betway",         # Betway — disponible
    "marathonbet",    # Marathon Bet
    "williamhill",    # William Hill
    "bet365",         # Bet365 — margen ~5%
    "draftkings",     # DraftKings — USA
    "betonlineag",    # BetOnline
    "fanduel",        # FanDuel
]

# Regiones a consultar (us2 incluye más casas americanas)
API_REGIONS  = "us,eu,uk"   # us2 cuesta requests extra — mantener en 3 regiones

# Mercados disponibles en free tier
MARKETS_FREE = ["h2h", "totals"]        # 1X2 + Over/Under goles
MARKETS_PAID = ["btts"]                 # requiere plan básico (~$10 USD/mes)

# Conversión de nombre The Odds API → nuestro sistema
TEAM_NAME_MAP = {
    "Club America":         "América",
    "Cruz Azul":            "Cruz Azul",
    "Chivas Guadalajara":   "Guadalajara",
    "CF Monterrey":         "Monterrey",
    "Tigres UANL":          "Tigres",
    "Pumas UNAM":           "Pumas",
    "Toluca":               "Toluca",
    "Atlas":                "Atlas",
    "Santos Laguna":        "Santos Laguna",
    "Necaxa":               "Necaxa",
    "Mazatlan FC":          "Mazatlán",
    "FC Juarez":            "FC Juárez",
    "Atletico San Luis":    "San Luis",
    "Queretaro":            "Querétaro",
    "Tijuana":              "Tijuana",
    "Leon":                 "León",
    "Puebla":               "Puebla",
    "Pachuca":              "Pachuca",
}

def norm_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


# Mapa para igualar nombres de odds_ligamx.csv → equipo_local/visita en betting_log.csv
# odds side → betting_log side
_ODDS_TO_LOG_MAP = {
    "querétaro":         "queretaro fc",
    "queretaro":         "queretaro fc",
    "atlético san luis": "atletico de san luis",
    "atletico san luis": "atletico de san luis",
    "guadalajara":       "chivas",
    "cf monterrey":      "monterrey",
    "pumas unam":        "pumas",
    "mazatlán":          "mazatlan fc",
    "mazatlan":          "mazatlan fc",
    "mazatlán fc":       "mazatlan fc",
    "fc juárez":         "fc juarez",
    "fc juarez":         "fc juarez",
    "América":           "cf america",
    "américa":           "cf america",
    "léon":              "leon",
    "león":              "leon",
    "santos laguna":     "santos laguna",
    "cruz azul":         "cruz azul",
    "tigres":            "tigres",
    "atlas":             "atlas",
    "necaxa":            "necaxa",
    "toluca":            "toluca",
    "puebla":            "puebla",
    "tijuana":           "tijuana",
    "pachuca":           "pachuca",
    "pumas":             "pumas",
}

def norm_for_match(name: str) -> str:
    """Normaliza para comparación cruzada odds ↔ betting_log."""
    n = name.strip().lower()
    return _ODDS_TO_LOG_MAP.get(n, n)


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get(endpoint: str, params: dict) -> dict | None:
    if not API_KEY:
        print("  [ERROR] ODDS_API_KEY no configurado. Exporta: export ODDS_API_KEY=xxxx")
        return None
    params["apiKey"] = API_KEY
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        print(f"  [ERROR] API: {e} — {r.text[:200]}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def check_status() -> dict:
    """Muestra los requests restantes del mes."""
    if not API_KEY:
        print("  ODDS_API_KEY no configurado")
        return {}
    r = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY}, timeout=10)
    remaining = r.headers.get("x-requests-remaining", "?")
    used      = r.headers.get("x-requests-used", "?")
    print(f"\n── The Odds API — uso del mes ──")
    print(f"  Requests usados:    {used}")
    print(f"  Requests restantes: {remaining}")
    return {"used": used, "remaining": remaining}


# ─────────────────────────────────────────────────────────────────────────────
# Descarga de cuotas
# ─────────────────────────────────────────────────────────────────────────────
def fetch_odds(markets: list[str] = None, verbose: bool = True) -> list[dict]:
    """
    Descarga cuotas de los próximos partidos de Liga MX.
    Retorna lista de dicts normalizados.
    """
    if markets is None:
        markets = MARKETS_FREE

    params = {
        "sport":      SPORT,
        "regions":    API_REGIONS,
        "markets":    ",".join(markets),
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    data = _get("sports/{sport}/odds".replace("{sport}", SPORT), params)
    if not data:
        return []

    partidos = []
    for event in data:
        local  = norm_team(event.get("home_team", ""))
        visita = norm_team(event.get("away_team", ""))
        fecha_raw = event.get("commence_time", "")
        fecha = utc_to_mx_date(fecha_raw)  # fecha en hora México City

        row = {
            "fecha":       fecha,
            "fecha_hora":  fecha_raw[:19].replace("T", " "),
            "partido":     f"{local} vs {visita}",
            "local":       local,
            "visitante":   visita,
            "descargado":  TODAY,
        }

        # Procesar cada bookmaker
        for bk in event.get("bookmakers", []):
            bk_key = bk.get("key", "")
            if bk_key not in BOOKMAKERS and bk_key not in [b.lower() for b in BOOKMAKERS]:
                continue

            for market in bk.get("markets", []):
                key = market.get("key", "")
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

                if key == "h2h":
                    row[f"{bk_key}_odd_1"] = outcomes.get(event.get("home_team", local))
                    row[f"{bk_key}_odd_X"] = outcomes.get("Draw")
                    row[f"{bk_key}_odd_2"] = outcomes.get(event.get("away_team", visita))

                elif key == "totals":
                    for o in market.get("outcomes", []):
                        if o["name"] == "Over" and o.get("point") == 2.5:
                            row[f"{bk_key}_odd_over25"] = o["price"]
                        elif o["name"] == "Under" and o.get("point") == 2.5:
                            row[f"{bk_key}_odd_under25"] = o["price"]

                elif key == "btts":
                    row[f"{bk_key}_odd_btts_si"] = outcomes.get("Yes")
                    row[f"{bk_key}_odd_btts_no"] = outcomes.get("No")

        partidos.append(row)
        if verbose:
            print(f"    {fecha} {local} vs {visita}")

    if verbose:
        print(f"  → {len(partidos)} partidos con cuotas")

    return partidos


# ─────────────────────────────────────────────────────────────────────────────
# Guardar y calcular EV
# ─────────────────────────────────────────────────────────────────────────────
def save_odds_ligamx(partidos: list[dict], dry_run: bool = False) -> int:
    """Guarda/actualiza odds_ligamx.csv con deduplicación."""
    if not partidos:
        return 0

    df_new = pd.DataFrame(partidos)

    if ODDS_LIGAMX.exists():
        df_old = pd.read_csv(ODDS_LIGAMX, low_memory=False)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new.copy()

    df_out = df_out.drop_duplicates(
        subset=["fecha", "local", "visitante", "descargado"], keep="last"
    ).sort_values(["fecha", "local"]).reset_index(drop=True)

    if not dry_run:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(ODDS_LIGAMX.parent), suffix=".tmp")
        try:
            df_out.to_csv(tmp_path, index=False)
            os.close(tmp_fd)
            os.replace(tmp_path, str(ODDS_LIGAMX))
        except Exception:
            os.close(tmp_fd)
            os.unlink(tmp_path)
            raise
        print(f"  → odds_ligamx.csv: {len(df_out)} registros")

    return len(df_new)


def update_betting_log_ev(partidos: list[dict], dry_run: bool = False) -> int:
    """
    Actualiza betting_log.csv con cuota_vista y ev_estimado
    para los partidos que tenemos cuotas.
    Usa la cuota más alta disponible entre los bookmakers (mejor EV para el apostador).
    """
    if not BETTING_LOG.exists() or not partidos:
        return 0

    df_log = pd.read_csv(BETTING_LOG, low_memory=False)
    if df_log.empty:
        return 0

    # Índice rápido: (fecha, local_norm, visita_norm) → cuotas
    # Doble índice: clave exacta + clave normalizada con _ODDS_TO_LOG_MAP
    odds_idx = {}
    for p in partidos:
        # clave exacta (por si los nombres ya coinciden)
        key_exact = (p["fecha"], p["local"].strip().lower(), p["visitante"].strip().lower())
        odds_idx[key_exact] = p
        # clave normalizada (para cruzar con betting_log)
        key_norm = (p["fecha"], norm_for_match(p["local"]), norm_for_match(p["visitante"]))
        odds_idx[key_norm] = p

    actualizados = 0
    for idx, row in df_log.iterrows():
        # Solo actualizar si no tiene cuota aún
        if pd.notna(row.get("cuota_vista")):
            continue

        local_lower  = str(row["equipo_local"]).strip().lower()
        visita_lower = str(row["equipo_visita"]).strip().lower()
        fecha_str    = str(row["fecha_partido"])[:10]

        key = (fecha_str, local_lower, visita_lower)
        p = odds_idx.get(key)
        if p is None:
            continue

        mercado   = str(row["mercado"])
        prob      = float(row["prob_modelo"])
        cuota_col = None

        # Encontrar la cuota del mercado correspondiente
        if mercado == "corners_over_8.5":
            pass  # corners no disponible en free tier — requiere plan pagado
        elif mercado == "goles_over_2.5":
            # Tomar la cuota más alta disponible entre los bookmakers
            candidates = [v for k, v in p.items()
                          if "_odd_over25" in k and pd.notna(v) and v]
            if candidates:
                cuota_col = max(float(c) for c in candidates)
        elif mercado == "goles_over_1.5":
            candidates = [v for k, v in p.items()
                          if "_odd_over15" in k and pd.notna(v) and v]
            if candidates:
                cuota_col = max(float(c) for c in candidates)
        elif mercado == "btts_si":
            candidates = [v for k, v in p.items()
                          if "_odd_btts_si" in k and pd.notna(v) and v]
            if candidates:
                cuota_col = max(float(c) for c in candidates)

        if cuota_col and cuota_col > 1.0:
            ev = round(prob * cuota_col - 1, 4)
            df_log.at[idx, "cuota_vista"]  = round(cuota_col, 3)
            df_log.at[idx, "ev_estimado"]  = ev
            actualizados += 1

    if actualizados > 0 and not dry_run:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(BETTING_LOG.parent), suffix=".tmp"
        )
        try:
            df_log.to_csv(tmp_path, index=False)
            os.close(tmp_fd)
            os.replace(tmp_path, str(BETTING_LOG))
        except Exception:
            os.close(tmp_fd)
            os.unlink(tmp_path)
            raise
        print(f"  → {actualizados} filas en betting_log con cuota y EV actualizados")

    return actualizados


# ─────────────────────────────────────────────────────────────────────────────
# Mostrar picks con EV
# ─────────────────────────────────────────────────────────────────────────────
def show_value_bets(partidos: list[dict]):
    """Imprime tabla de value bets detectados."""
    if not partidos:
        return

    # Leer model probs: primero betting_log, luego JSONs del betting bot (más frescos)
    model_probs = {}   # (fecha, local_norm, visita_norm, mercado) → prob_modelo

    if BETTING_LOG.exists():
        try:
            df_log = pd.read_csv(BETTING_LOG, low_memory=False)
            for _, row in df_log.iterrows():
                key = (
                    str(row["fecha_partido"])[:10],
                    norm_for_match(str(row["equipo_local"])),
                    norm_for_match(str(row["equipo_visita"])),
                    str(row["mercado"]),
                )
                model_probs[key] = float(row["prob_modelo"])
        except Exception:
            pass

    # Leer JSONs del betting bot (tienen probs frescas no escritas aún en betting_log)
    REPORTS_DIR = BASE / "output/reports"
    if REPORTS_DIR.exists():
        import glob as _glob
        for jf in sorted(_glob.glob(str(REPORTS_DIR / "betting_*.json")), reverse=True)[:5]:
            try:
                with open(jf, encoding="utf-8") as _f:
                    j_data = json.load(_f)
                if not isinstance(j_data, list):
                    continue
                for entry in j_data:
                    fecha_j = entry.get("fecha", "")[:10]
                    local_j = norm_for_match(entry.get("local", ""))
                    visita_j = norm_for_match(entry.get("visita", ""))
                    btts = entry.get("btts", {}) or {}
                    tarjetas = entry.get("tarjetas", {}) or {}
                    # goles mercados
                    _probe = {
                        "goles_over_2.5": btts.get("over_2.5"),
                        "goles_over_1.5": btts.get("over_1.5"),
                        "btts_si": btts.get("btts_si"),
                        "tarjetas_over_3.5": tarjetas.get("over_3.5"),
                        "tarjetas_over_4.5": tarjetas.get("over_4.5"),
                        "tarjetas_over_5.5": tarjetas.get("over_5.5"),
                    }
                    for mercado, prob in _probe.items():
                        if prob is not None:
                            key = (fecha_j, local_j, visita_j, mercado)
                            model_probs.setdefault(key, float(prob))  # newest file wins
                    # Calcular 1X2 con Dixon-Coles desde lambdas
                    lam_l = btts.get("lambda_local")
                    lam_v = btts.get("lambda_visita")
                    if lam_l and lam_v:
                        try:
                            from scipy.stats import poisson as _pois
                            _rho = -0.22
                            _w1 = _d = _w2 = 0.0
                            for _i in range(10):
                                for _j in range(10):
                                    _p = _pois.pmf(_i, lam_l) * _pois.pmf(_j, lam_v)
                                    if _i == 0 and _j == 0: _p *= (1 - lam_l * lam_v * _rho)
                                    elif _i == 1 and _j == 0: _p *= (1 + lam_v * _rho)
                                    elif _i == 0 and _j == 1: _p *= (1 + lam_l * _rho)
                                    elif _i == 1 and _j == 1: _p *= (1 - _rho)
                                    if _i > _j: _w1 += _p
                                    elif _i == _j: _d += _p
                                    else: _w2 += _p
                            for _mercado, _prob in [("1x2_local", _w1), ("1x2_draw", _d), ("1x2_visita", _w2)]:
                                model_probs.setdefault((fecha_j, local_j, visita_j, _mercado), round(_prob, 4))  # newest wins
                        except Exception:
                            pass
            except Exception:
                pass

    MIN_EV = 0.04   # solo mostrar si EV >= 4%

    print(f"\n── VALUE BETS Liga MX — {TODAY} (EV ≥ {MIN_EV:.0%}) ──\n")
    print(f"{'Partido':<34} {'Mercado':<20} {'ProbM':>6} {'Cuota':>6} {'EV':>7}")
    print("─" * 78)

    rows = []
    for p in partidos:
        local  = p["local"]
        visita = p["visitante"]
        fecha  = p["fecha"]
        local_n  = norm_for_match(local)
        visita_n = norm_for_match(visita)
        partido_str = f"{local} vs {visita}"

        # Para 1X2: buscar la cuota del equipo local/visita/draw por nombre
        # La API devuelve odds con nombre original del equipo
        local_raw  = p.get("_local_raw",  local)
        visita_raw = p.get("_visita_raw", visita)
        h2h_cuotas = {"1x2_local": None, "1x2_draw": None, "1x2_visita": None}
        h2h_bk     = {"1x2_local": None, "1x2_draw": None, "1x2_visita": None}
        for bk in BOOKMAKERS:
            for side, col in [("1x2_local", f"{bk}_odd_1"), ("1x2_draw", f"{bk}_odd_X"), ("1x2_visita", f"{bk}_odd_2")]:
                v = p.get(col)
                if v:
                    try:
                        fv = float(v)
                        if fv > 1.0 and (h2h_cuotas[side] is None or fv > h2h_cuotas[side]):
                            h2h_cuotas[side] = fv
                            h2h_bk[side] = bk
                    except (ValueError, TypeError):
                        pass

        mercado_map = {
            "goles_over_2.5": [f"{bk}_odd_over25" for bk in BOOKMAKERS],
            "goles_over_1.5": [f"{bk}_odd_over15" for bk in BOOKMAKERS],
            "btts_si":        [f"{bk}_odd_btts_si" for bk in BOOKMAKERS],
        }

        for mercado, cuota_cols in mercado_map.items():
            mkey = (fecha, local_n, visita_n, mercado)
            prob = model_probs.get(mkey)
            if prob is None:
                continue

            best_cuota = None
            best_bk    = None
            for col in cuota_cols:
                v = p.get(col)
                if v and float(v) > 1.0:
                    if best_cuota is None or float(v) > best_cuota:
                        best_cuota = float(v)
                        best_bk    = col.split("_odd_")[0]

            if best_cuota is None:
                continue

            ev = round(prob * best_cuota - 1, 4)
            if ev >= MIN_EV:
                rows.append((ev, partido_str, mercado, prob, best_cuota, best_bk))

        # 1X2 value bets
        for side in ["1x2_local", "1x2_draw", "1x2_visita"]:
            mkey = (fecha, local_n, visita_n, side)
            prob = model_probs.get(mkey)
            cuota = h2h_cuotas.get(side)
            if prob is None or cuota is None:
                continue
            ev = round(prob * cuota - 1, 4)
            if ev >= MIN_EV:
                label = side.replace("1x2_local", f"1X ({local})").replace("1x2_draw", "X (Empate)").replace("1x2_visita", f"2 ({visita})")
                rows.append((ev, partido_str, label, prob, cuota, h2h_bk.get(side, "")))

    rows.sort(key=lambda x: -x[0])
    for ev, partido_str, mercado, prob, cuota, bk in rows:
        print(f"  {partido_str:<32} {mercado:<20} {prob:>5.1%} {cuota:>6.2f} {ev:>+7.1%}  [{bk}]")

    if not rows:
        print("  Sin value bets con EV suficiente para los próximos partidos")
        # Mostrar resumen de todos los mercados con modelo
        print(f"\n  Todos los mercados con prob (EV < {MIN_EV:.0%}):")
        for p in partidos[:5]:
            local_n  = norm_for_match(p["local"])
            visita_n = norm_for_match(p["visitante"])
            for col in [f"{bk}_odd_over25" for bk in BOOKMAKERS]:
                v = p.get(col)
                if v and float(v) > 1.0:
                    mkey = (p["fecha"], local_n, visita_n, "goles_over_2.5")
                    prob = model_probs.get(mkey, None)
                    if prob:
                        ev = round(prob * float(v) - 1, 4)
                        print(f"    {p['local']} vs {p['visitante']}: Over2.5 prob={prob:.1%} cuota={v:.2f} EV={ev:+.1%}")
                    break
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Descarga cuotas Liga MX de The Odds API y calcula EV"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra sin guardar en CSV")
    parser.add_argument("--status",  action="store_true",
                        help="Muestra requests restantes del mes")
    parser.add_argument("--btts",    action="store_true",
                        help="Incluye mercado BTTS (requiere plan pagado)")
    args = parser.parse_args()

    if args.status:
        check_status()
        return

    markets = MARKETS_FREE + (MARKETS_PAID if args.btts else [])
    print(f"\n── Descargando cuotas Liga MX (mercados: {', '.join(markets)}) ──")

    partidos = fetch_odds(markets=markets, verbose=True)

    if partidos:
        save_odds_ligamx(partidos, dry_run=args.dry_run)
        update_betting_log_ev(partidos, dry_run=args.dry_run)
        show_value_bets(partidos)
    else:
        print("  Sin partidos con cuotas disponibles hoy")

    if not args.dry_run and not args.status:
        check_status()


if __name__ == "__main__":
    main()
