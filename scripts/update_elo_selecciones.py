#!/usr/bin/env python3
"""
update_elo_selecciones.py
Aplica los partidos nuevos (>= 2026-03-30) al ELO guardado en
data/processed/elos_selecciones_20260329.json y muestra tabla de cambios.
Guarda el nuevo estado en data/processed/elos_selecciones_YYYYMMDD.json
"""
import json, re
from pathlib import Path
from datetime import date

import pandas as pd

BASE     = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data/raw/internacional/results.csv"
ELO_IN   = BASE / "data/processed/elos_selecciones_20260329.json"
OUT_DATE = "20260412"
ELO_OUT  = BASE / f"data/processed/elos_selecciones_{OUT_DATE}.json"

# ─── Parámetros ELO (mismos que 18_prediccion_selecciones.py) ─────────────────
ELO_BASE = 1500
HOME_ADV = 100
SCALE    = 400

def k_base(tournament: str) -> int:
    t    = str(tournament).strip().lower()
    qual = 'qualifier' in t or 'qualifying' in t or 'qualification' in t
    if ('world cup' in t or 'fifa world cup' in t) and not qual:
        return 60
    main_35 = ['concacaf gold cup','copa america','afc asian cup',
                'africa cup of nations','confederations cup',
                'ofc nations cup','concacaf nations league']
    is_euro_main = (('euro ' in t or t.startswith('euro') or
                     'european championship' in t) and not qual)
    if is_euro_main:
        return 35
    if any(x in t for x in main_35) and not qual and 'uefa nations' not in t:
        return 35
    if qual or 'uefa nations league' in t:
        return 25
    return 20

def goal_mult(gf: int, ga: int) -> float:
    diff = abs(gf - ga)
    if diff <= 1:   return 1.0
    elif diff == 2: return 1.5
    else:           return 1.75 + (diff - 3) * 0.04

def expected_score(elo_a, elo_b, home_adv=0):
    return 1 / (1 + 10 ** ((elo_b - (elo_a + home_adv)) / SCALE))

def result_score(gl, gv):
    if gl > gv: return 1.0
    elif gl < gv: return 0.0
    return 0.5

# ─── Normalización de nombres (FotMob → clave ELO) ────────────────────────────
NAME_MAP = {
    "Czechia":              "Czech Republic",
    "Turkiye":              "Turkey",
    "USA":                  "United States",
    "Ireland":              "Republic of Ireland",
    "Curacao":              "Curaçao",
    "St. Kitts and Nevis":  "Saint Kitts and Nevis",
    "DR Congo":             "DR Congo",        # ya existe en el dict
    "Chinese Taipei":       "Taiwan",
    "Ivory Coast":          "Ivory Coast",     # ya existe
    "North Macedonia":      "North Macedonia", # ya existe
}

def norm(name: str) -> str:
    return NAME_MAP.get(name, name)

def is_a_team_match(row) -> bool:
    """Filtra U21, Sub-20, Femenino, y Ligas de clubes."""
    home = str(row["home_team"])
    away = str(row["away_team"])
    tourn = str(row["tournament"])

    # Excluir partidos Sub/U
    if re.search(r'\bU\d{2}\b', home) or re.search(r'\bU\d{2}\b', away):
        return False
    if re.search(r'\bSub-?\d{2}\b', home, re.I) or re.search(r'\bSub-?\d{2}\b', away, re.I):
        return False

    # Excluir competencias de U21/U20 etc en el torneo
    if re.search(r'\bU\d{2}\b', tourn) or 'U21' in tourn or 'U20' in tourn:
        return False

    # Excluir femenino
    if re.search(r"\(W\)|\bWomen\b|\bFemenin", home + " " + away + " " + tourn, re.I):
        return False
    if "Women's" in tourn or "Femenin" in tourn:
        return False

    return True

def main():
    # Cargar ELO base (2026-03-29)
    elos = json.loads(ELO_IN.read_text())
    print(f"ELO base cargado: {len(elos)} selecciones")
    print(f"México base: {elos.get('Mexico', ELO_BASE):.1f}")
    print(f"Portugal base: {elos.get('Portugal', ELO_BASE):.1f}")

    # Cargar partidos nuevos
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    new_matches = df[df["date"] >= "2026-03-30"].copy()
    new_matches = new_matches[new_matches.apply(is_a_team_match, axis=1)]
    new_matches = new_matches.sort_values("date")
    print(f"\nPartidos selecciones A (>= 2026-03-30): {len(new_matches)}")

    # Aplicar cada partido
    changes = []
    for _, r in new_matches.iterrows():
        home_raw = str(r["home_team"])
        away_raw = str(r["away_team"])
        home = norm(home_raw)
        away = norm(away_raw)
        gl   = int(r["home_score"])
        gv   = int(r["away_score"])
        neutral = str(r.get("neutral","TRUE")).upper() in ("TRUE","1")
        tourn   = str(r["tournament"])

        elo_h_before = elos.get(home, ELO_BASE)
        elo_a_before = elos.get(away, ELO_BASE)

        adv   = 0 if neutral else HOME_ADV
        ea    = expected_score(elo_h_before, elo_a_before, home_adv=adv)
        sa    = result_score(gl, gv)
        k_eff = k_base(tourn) * goal_mult(gl, gv)

        delta_h =  k_eff * (sa - ea)
        delta_a =  k_eff * ((1 - sa) - (1 - ea))

        elos[home] = elo_h_before + delta_h
        elos[away] = elo_a_before + delta_a

        res_str = f"{gl}-{gv}"
        if gl > gv:   outcome_h, outcome_a = "G", "P"
        elif gl < gv: outcome_h, outcome_a = "P", "G"
        else:         outcome_h, outcome_a = "E", "E"

        changes.append({
            "fecha":    str(r["date"].date()),
            "equipo":   home,
            "elo_antes": round(elo_h_before, 1),
            "elo_des":  round(elos[home], 1),
            "diff":     round(delta_h, 1),
            "rival":    away,
            "resultado": res_str,
            "res": outcome_h,
        })
        changes.append({
            "fecha":    str(r["date"].date()),
            "equipo":   away,
            "elo_antes": round(elo_a_before, 1),
            "elo_des":  round(elos[away], 1),
            "diff":     round(delta_a, 1),
            "rival":    home,
            "resultado": res_str,
            "res": outcome_a,
        })

    # Guardar nuevo estado
    ELO_OUT.write_text(json.dumps(elos, ensure_ascii=False, indent=2))
    print(f"\n✓ ELO guardado: {ELO_OUT}")

    # ─── Tabla de cambios ─────────────────────────────────────────────────────
    print("\n" + "═"*85)
    print(f"{'País':<28} {'ELO antes':>9} {'ELO después':>11} {'Diferencia':>11}  {'Rival':<28} {'Res'}")
    print("─"*85)

    df_ch = pd.DataFrame(changes)
    # Filtrar solo equipos con diferencia significativa (> 0.1 abs) o selecciones relevantes
    SHOW_TEAMS = {
        "Mexico","Portugal","Brazil","Argentina","France","Spain","England",
        "Germany","Italy","Netherlands","Colombia","Uruguay","Japan","South Korea",
        "Morocco","Senegal","Egypt","Australia","USA","Canada","Panama","Costa Rica",
        "Bolivia","Peru","Honduras","Ecuador","Chile","Paraguay","Venezuela",
        "Croatia","Serbia","Poland","Denmark","Sweden","Norway","Switzerland",
        "Austria","Belgium","Scotland","Wales","Turkey","Ukraine","Kazakhstan",
        "DR Congo","Jamaica","Iraq","Saudi Arabia","South Africa","Ivory Coast",
        "Ghana","Cameroon","Algeria","Tunisia","Kenya","Sierra Leone","Algeria",
        "Slovenia","Slovakia","Hungary","Romania","Greece","Finland","Ireland",
        "Republic of Ireland","North Macedonia","Albania","Montenegro","Luxembourg",
        "Latvia","Cape Verde","New Zealand","Rwanda","Azerbaijan",
    }

    shown = set()
    for _, r in df_ch.iterrows():
        team = r["equipo"]
        if team not in SHOW_TEAMS:
            continue
        key = (r["fecha"], team)
        if key in shown:
            continue
        shown.add(key)

        diff_str = f"{r['diff']:+.1f}"
        sign = "▲" if r["diff"] > 0 else ("▼" if r["diff"] < 0 else "─")
        print(f"{team:<28} {r['elo_antes']:>9.1f} {r['elo_des']:>11.1f} {sign}{diff_str:>10}  {r['rival']:<28} {r['resultado']} ({r['res']})")

    # ─── Top 10 ranking actual ─────────────────────────────────────────────────
    print("\n" + "═"*45)
    print("  TOP 10 RANKING ELO (actualizado al 2026-04-01)")
    print("─"*45)
    sorted_elos = sorted(elos.items(), key=lambda x: -x[1])
    major = [(k,v) for k,v in sorted_elos if v >= 1600][:10]
    for i,(k,v) in enumerate(major, 1):
        print(f"  {i:>2}. {k:<25} {v:.1f}")

    print("\n  México  :", round(elos.get("Mexico", ELO_BASE), 1))
    print("  Portugal:", round(elos.get("Portugal", ELO_BASE), 1))
    print("  Bélgica :", round(elos.get("Belgium", ELO_BASE), 1))
    print("  Brasil  :", round(elos.get("Brazil", ELO_BASE), 1))
    print("  Japón   :", round(elos.get("Japan", ELO_BASE), 1))
    print("  USA     :", round(elos.get("United States", ELO_BASE), 1))

if __name__ == "__main__":
    main()
