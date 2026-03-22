"""
03_radar_jugador.py
Lee los JSONs de data/raw/jugadores/, consolida en un DataFrame y genera
un radar chart con mplsoccer para un jugador específico.

Uso:
    python 03_radar_jugador.py                        # usa Paulinho por defecto
    python 03_radar_jugador.py --jugador 361377       # por ID
    python 03_radar_jugador.py --nombre "Paulinho"    # por nombre
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from mplsoccer import Radar, FontManager

# ── Rutas ─────────────────────────────────────────────────────────────────────
JUGADORES_DIR = Path(__file__).parent.parent / "data" / "raw" / "jugadores"
OUTPUT_DIR    = Path(__file__).parent.parent / "output" / "charts"

LIGA_MX_ID = 230         # leagueId de Liga MX en FotMob
CLAUSURA_START = "2026-01-01"  # fecha de inicio del Clausura 2025/2026

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-MX,es;q=0.9",
}

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)

# ── Métricas del radar ─────────────────────────────────────────────────────────
# (columna_df, etiqueta_display, invert)
# invert=True → valores bajos son "mejores" (tarjetas); se invierte solo para el rango
METRICAS = [
    ("goles",          "Goles",            False),
    ("asistencias",    "Asistencias",      False),
    ("rating",         "Rating",           False),
    ("minutos",        "Minutos jugados",  False),
    ("tarj_amarillas", "Tarj. amarillas",  True),
    ("valor_mercado",  "Valor mercado\n(M€)", False),
]

# ── Carga del DataFrame ────────────────────────────────────────────────────────

def cargar_dataframe() -> pd.DataFrame:
    registros = []
    for path in sorted(JUGADORES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        equipo = data["equipo"]
        equipo_id = data["equipo_id"]
        for j in data["jugadores"]:
            if j["rol_grupo"] == "coach":
                continue
            registros.append({
                "id":            j["id"],
                "nombre":        j["nombre"],
                "equipo":        equipo,
                "equipo_id":     equipo_id,
                "posicion":      j.get("posicion", ""),
                "edad":          j.get("edad") or np.nan,
                "goles":         j.get("goles") or 0,
                "asistencias":   j.get("asistencias") or 0,
                "rating":        j.get("rating") or np.nan,
                "tarj_amarillas":j.get("tarj_amarillas") or 0,
                "valor_mercado": (j.get("valor_mercado") or 0) / 1_000_000,  # → M€
                "minutos":       np.nan,  # se completa por separado
            })
    df = pd.DataFrame(registros)
    print(f"DataFrame: {len(df)} jugadores de {df['equipo'].nunique()} equipos")
    return df


# ── Minutos jugados desde FotMob ───────────────────────────────────────────────

def fetch_minutos_liga_mx(player_id: int) -> int:
    """Suma los minutos jugados en Liga MX Clausura 2025/26 para un jugador."""
    url = f"https://www.fotmob.com/es/players/{player_id}/overview/player"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    match = NEXT_DATA_RE.search(r.text)
    if not match:
        return 0

    page = json.loads(match.group(1))
    fallback = page["props"]["pageProps"].get("fallback", {})
    pdata = fallback.get(f"player:{player_id}", {})
    recent = pdata.get("recentMatches", [])

    minutos = sum(
        m.get("minutesPlayed", 0)
        for m in recent
        if (
            m.get("leagueId") == LIGA_MX_ID
            and m.get("matchDate", {}).get("utcTime", "") >= CLAUSURA_START
        )
    )
    return minutos


# ── Radar ──────────────────────────────────────────────────────────────────────

def build_radar(df: pd.DataFrame, jugador_row: pd.Series) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nombre  = jugador_row["nombre"]
    equipo  = jugador_row["equipo"]
    pos     = jugador_row.get("posicion", "")

    # Rangos: percentil 5 → percentil 95 (mplsoccer maneja lower_is_better internamente)
    params, low_vals, high_vals, player_vals = [], [], [], []

    for col, label, _invert in METRICAS:
        col_data = df[col].dropna()

        if col == "minutos":
            # Rango fijo: 0 – 990 (11 jornadas × 90 min)
            lo, hi = 0.0, 990.0
        else:
            lo = float(np.nanpercentile(col_data, 5))
            hi = float(np.nanpercentile(col_data, 99))  # p99 para no cortar outliers
            # Garantizar que lo != hi
            if lo == hi:
                lo = max(0.0, hi - 1.0)

        val = float(jugador_row[col]) if not pd.isna(jugador_row[col]) else lo
        # Clamp al rango para que el radar no se salga
        val = max(lo, min(hi, val))

        params.append(label)
        low_vals.append(round(lo, 2))
        high_vals.append(round(hi, 2))
        player_vals.append(round(val, 2))

    print("\nMétricas del radar:")
    print(f"{'Métrica':<22} {'Min':>8} {'Max':>8} {'Jugador':>10}")
    print("-" * 52)
    for p, lo, hi, v in zip(params, low_vals, high_vals, player_vals):
        print(f"{p:<22} {lo:>8.2f} {hi:>8.2f} {v:>10.2f}")

    # ── Construir radar ───────────────────────────────────────────────────────
    radar = Radar(
        params=params,
        min_range=low_vals,
        max_range=high_vals,
        lower_is_better=["Tarj. amarillas"],
        num_rings=4,
        ring_width=1,
        center_circle_radius=1,
    )

    fig, ax = radar.setup_axis(figsize=(8, 8), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Anillos y spokes
    rings_inner = radar.draw_circles(
        ax=ax,
        facecolor="#1a2233",
        edgecolor="#2e3d55",
        lw=1.5,
    )

    # Polígono del jugador
    radar_output = radar.draw_radar(
        player_vals,
        ax=ax,
        kwargs_radar={"facecolor": "#e63946", "alpha": 0.35},
        kwargs_rings={"facecolor": "#e63946", "alpha": 0.75},
    )

    # Etiquetas de parámetros
    radar.draw_param_labels(
        ax=ax,
        color="#e0e0e0",
        fontsize=10,
        fontweight="bold",
    )

    # Etiquetas de rango
    radar.draw_range_labels(
        ax=ax,
        color="#8899aa",
        fontsize=7.5,
    )

    # Título
    nombre_slug = nombre.lower().replace(" ", "_")
    fig.text(
        0.5, 0.96,
        nombre,
        ha="center", va="top",
        fontsize=18, fontweight="bold", color="#ffffff",
    )
    fig.text(
        0.5, 0.92,
        f"{equipo}  ·  {pos}  ·  Liga MX Clausura 2025/26",
        ha="center", va="top",
        fontsize=11, color="#aabbcc",
    )
    fig.text(
        0.5, 0.03,
        "Rango: percentil 5–95 del total de jugadores de Liga MX  |  Fuente: FotMob",
        ha="center", va="bottom",
        fontsize=7.5, color="#556677",
    )

    out_path = OUTPUT_DIR / f"radar_{nombre_slug}_{equipo.lower().replace(' ','_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nGráfica guardada en: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jugador", type=int, default=361377, help="ID del jugador en FotMob")
    parser.add_argument("--nombre",  type=str, default=None,   help="Nombre del jugador (búsqueda parcial)")
    args = parser.parse_args()

    df = cargar_dataframe()

    # Buscar jugador
    if args.nombre:
        mask = df["nombre"].str.contains(args.nombre, case=False, na=False)
        coincidencias = df[mask]
        if coincidencias.empty:
            print(f"No se encontró ningún jugador con '{args.nombre}'")
            sys.exit(1)
        if len(coincidencias) > 1:
            print(f"Varios jugadores encontrados:\n{coincidencias[['id','nombre','equipo']].to_string()}")
            print("\nUsa --jugador <ID> para seleccionar uno.")
            sys.exit(1)
        jugador_row = coincidencias.iloc[0]
    else:
        jugador_row = df[df["id"] == args.jugador]
        if jugador_row.empty:
            print(f"No se encontró jugador con ID {args.jugador}")
            sys.exit(1)
        jugador_row = jugador_row.iloc[0]

    print(f"\nJugador: {jugador_row['nombre']}  ({jugador_row['equipo']})")

    # Obtener minutos de Liga MX
    player_id = int(jugador_row["id"])
    print(f"Obteniendo minutos jugados en Liga MX para ID {player_id}...")
    minutos = fetch_minutos_liga_mx(player_id)
    print(f"  → {minutos} minutos en Liga MX")

    # Actualizar en el DataFrame y en la fila del jugador
    df.loc[df["id"] == player_id, "minutos"] = minutos
    jugador_row = jugador_row.copy()
    jugador_row["minutos"] = minutos

    # Para el rango de minutos usamos el máximo teórico de la temporada
    # (11 jornadas × 90 min = 990) y percentil 5 = 0
    # Se incorpora en el DataFrame para que los percentiles sean coherentes
    max_min = 11 * 90
    df["minutos"] = df["minutos"].fillna(max_min * 0.25)  # bench ≈ 25 % del máx

    build_radar(df, jugador_row)


if __name__ == "__main__":
    main()
