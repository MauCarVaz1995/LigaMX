"""
05_radar_p90.py  –  v4
Pizza chart estilo Statiskicks.  8×10 in · 150 DPI · fondo transparente (Figma-ready).
Render del jugador como fondo semitransparente a la derecha del radar.
Escudo vía TheSportsDB (alta calidad) con fallback a FotMob.

Uso:
    python 05_radar_p90.py                           # Paulinho por defecto
    python 05_radar_p90.py --nombre "Henry Martín"
    python 05_radar_p90.py --color "#1A47A0"
    python 05_radar_p90.py --todos                   # genera 3 ejemplos

Renders manuales (FootyRenders / otros):
    Descarga el render PNG y colócalo en  .cache/renders/{player_id}.png
    El script lo usará automáticamente con alpha=0.30.
"""

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_visual import PALETAS, PALETA_ACTIVA, get_paleta

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mplsoccer import PyPizza
from PIL import Image, ImageDraw
from scipy.stats import percentileofscore

# ── Rutas ──────────────────────────────────────────────────────────────────────
CSV_PATH   = Path("data/processed/jugadores_clausura2026.csv")
OUTPUT_DIR = Path("output/charts")
CACHE_DIR  = Path(".cache")

MIN_MINUTOS = 300
_pal        = get_paleta()
BG_COLOR    = _pal['bg_primary']
GRAY_C      = _pal['text_secondary']

FOTMOB_PLAYER  = "https://images.fotmob.com/image_resources/playerimages/{pid}.png"
FOTMOB_TEAM    = "https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png"
SPORTSDB_URL   = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={name}"
BEBAS_URL      = "https://fonts.gstatic.com/s/bebasneue/v16/JTUSjIg69CK48gW7PXooxW4.ttf"
RENDERS_DIR    = CACHE_DIR / "renders"

# ── Grupos de posición ─────────────────────────────────────────────────────────
GRUPOS_POS = {
    "ST": "delantero",  "CF":  "delantero",  "LW":  "delantero",
    "RW": "delantero",  "CAM": "delantero",
    "CM": "mediocampista", "CDM": "mediocampista",
    "LM": "mediocampista", "RM":  "mediocampista",
    "CB": "defensa", "LB": "defensa", "RB":  "defensa",
    "LWB":"defensa", "RWB":"defensa",
    "GK": "portero",
}

METRICAS = {
    "delantero": [
        ("goles_p90",                  "Goles\np90",              False),
        ("xG_p90",                     "xG\np90",                 False),
        ("tiros_p90",                  "Tiros\np90",              False),
        ("asistencias_p90",            "Asistencias\np90",        False),
        ("xA_p90",                     "xA\np90",                 False),
        ("grandes_chances_p90",        "Grandes\nchances p90",    False),
        ("duelos_tierra_ganados_p90",  "Duelos\nganados p90",     False),
        ("penales_ganados_p90",        "Penales\nganados p90",    False),
    ],
    "mediocampista": [
        ("pases_precisos_p90",          "Pases\nexactos p90",      False),
        ("pases_largos_p90",            "Pases\nlargos p90",       False),
        ("chances_creadas_p90",         "Chances\ncreadas p90",    False),
        ("asistencias_p90",             "Asistencias\np90",        False),
        ("xA_p90",                      "xA\np90",                 False),
        ("recuperaciones_campo_rival_p90", "Recup.\ncampo rival p90", False),
        ("intercepciones_p90",          "Intercep-\nciones p90",   False),
        ("duelos_tierra_ganados_p90",   "Duelos\nganados p90",     False),
    ],
    "defensa": [
        ("intercepciones_p90",             "Intercep-\nciones p90",   False),
        ("despejes_p90",                   "Despejes\np90",           False),
        ("recuperaciones_campo_rival_p90", "Recup.\ncampo rival p90", False),
        ("entradas_p90",                   "Entradas\np90",           False),
        ("tiros_bloqueados_p90",           "Tiros\nbloqueados p90",   False),
        ("pases_precisos_p90",             "Pases\nexactos p90",      False),
        ("pases_largos_p90",               "Pases\nlargos p90",       False),
        ("faltas_cometidas_p90",           "Faltas\ncometidas p90",   True),
    ],
    "portero": [
        ("paradas_p90",              "Paradas\np90",              False),
        ("porcentaje_paradas_p90",   "% Paradas\np90",            False),
        ("goles_evitados_p90",       "Goles\nevitados p90",       False),
        ("goles_recibidos_p90",      "Goles\nrecibidos p90",      True),
        ("porterias_cero_p90",       "Porterías\na cero p90",     False),
        ("pases_precisos_p90",       "Pases\nexactos p90",        False),
        ("pases_largos_p90",         "Pases\nlargos p90",         False),
        ("despejes_p90",             "Despejes\np90",             False),
    ],
}

COLORES_EQUIPO = {
    "Toluca":               "#D5001C", "Chivas":               "#CE142C",
    "Cruz Azul":            "#1A47A0", "CF America":           "#DAA520",
    "Tigres":               "#E8971A", "Monterrey":            "#0055A5",
    "Pumas":                "#003366", "Atlas":                "#8B1A1A",
    "Pachuca":              "#004C97", "Santos Laguna":        "#00A651",
    "Necaxa":               "#E30613", "Leon":                 "#006341",
    "Puebla":               "#5B2D8E", "Atletico de San Luis": "#C41E3A",
    "Tijuana":              "#2C3E50", "FC Juarez":            "#FF5722",
    "Mazatlan FC":          "#6A1B9A", "Queretaro FC":         "#1565C0",
}


# ── Fuente Bebas Neue ──────────────────────────────────────────────────────────

def get_bebas_neue() -> str | None:
    """Descarga y registra Bebas Neue. Retorna el nombre de la familia."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    font_path = CACHE_DIR / "BebasNeue-Regular.ttf"

    if not font_path.exists():
        print("  Descargando Bebas Neue...", end=" ", flush=True)
        try:
            r = requests.get(BEBAS_URL, timeout=15)
            r.raise_for_status()
            font_path.write_bytes(r.content)
            print(f"OK ({len(r.content)//1024} KB)")
        except Exception as e:
            print(f"ERROR: {e} — se usará fuente del sistema.")
            return None

    try:
        fm.fontManager.addfont(str(font_path))
    except Exception:
        return None
    prop = fm.FontProperties(fname=str(font_path))
    return prop.get_name()


# ── Descarga de imágenes ───────────────────────────────────────────────────────

def _fetch_pil(url: str, size: int | None = None) -> Image.Image | None:
    """Descarga una URL y retorna PIL Image RGBA (sin redimensionar si size=None)."""
    try:
        r = requests.get(url, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        if size:
            img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"  [IMG] {url[:60]}… → {e}")
        return None


def get_team_logo(team_id: int, team_name: str = "", size: int = 180) -> np.ndarray | None:
    """
    Intenta TheSportsDB (mayor calidad) → fallback FotMob.
    Retorna array RGBA redimensionado a `size`×`size`.
    """
    # 1. TheSportsDB (PNG alta resolución)
    if team_name:
        try:
            resp = requests.get(
                SPORTSDB_URL.format(name=requests.utils.quote(team_name)),
                timeout=8)
            resp.raise_for_status()
            teams = (resp.json().get("teams") or [])
            badge_url = teams[0].get("strTeamBadge") if teams else None
            if badge_url:
                img = _fetch_pil(badge_url + "/preview", size)   # TheSportsDB preview
                if img is None:
                    img = _fetch_pil(badge_url, size)
                if img:
                    print(f"  [Logo] TheSportsDB ✓  ({badge_url[-40:]})")
                    return np.array(img)
        except Exception as e:
            print(f"  [Logo] TheSportsDB falló: {e}")

    # 2. FotMob fallback
    img = _fetch_pil(FOTMOB_TEAM.format(tid=int(team_id)), size)
    if img:
        print(f"  [Logo] FotMob fallback ✓")
        return np.array(img)
    return None


def circular_crop(img: Image.Image, border_px: int = 0) -> np.ndarray:
    """Recorta imagen PIL en círculo. border_px agrega borde blanco exterior."""
    img = img.convert("RGBA")
    size = min(img.size)
    left = (img.width - size) // 2
    top  = (img.height - size) // 2
    img  = img.crop((left, top, left + size, top + size))

    # Máscara circular
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    img.putalpha(mask)

    if border_px <= 0:
        return np.array(img)

    # Añadir borde blanco exterior
    full = size + 2 * border_px
    bordered = Image.new("RGBA", (full, full), (0, 0, 0, 0))
    mask_full = Image.new("L", (full, full), 0)
    ImageDraw.Draw(mask_full).ellipse((0, 0, full - 1, full - 1), fill=255)
    white = Image.new("RGBA", (full, full), (255, 255, 255, 255))
    white.putalpha(mask_full)
    bordered.paste(white, (0, 0), white)
    bordered.paste(img, (border_px, border_px), img)
    return np.array(bordered)


def get_player_headshot(player_id: int) -> np.ndarray | None:
    """Descarga headshot, recorte circular con borde blanco — header del chart."""
    img = _fetch_pil(FOTMOB_PLAYER.format(pid=player_id), 220)
    return circular_crop(img, border_px=6) if img else None  # ~3px a 100px display


def get_player_render(player_id: int, nombre: str) -> tuple[Image.Image | None, bool]:
    """
    Busca render PNG del jugador.
    Prioridad:
      1. .cache/renders/{player_id}.png   ← coloca aquí renders de FootyRenders
      2. FotMob headshot (192 px, upscale)
    Retorna (imagen PIL RGBA, es_render_local).
    """
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    local = RENDERS_DIR / f"{player_id}.png"
    if local.exists():
        print(f"  [Render] Render local encontrado: {local}")
        return Image.open(local).convert("RGBA"), True

    # Aviso para el usuario
    print(f"  [Render] No hay render en {local}")
    print(f"           Descarga desde https://www.footyrenders.com y guárdalo como {local}")

    # Fallback: FotMob headshot a baja resolución
    img = _fetch_pil(FOTMOB_PLAYER.format(pid=player_id), 192)
    if img:
        print(f"  [Render] Usando headshot FotMob (alpha reducido)")
        return img, False
    return None, False


# ── Utilidades ─────────────────────────────────────────────────────────────────

def clasificar_pos(pos_str: str) -> str:
    if not pos_str:
        return "mediocampista"
    return GRUPOS_POS.get(pos_str.split(",")[0].strip(), "mediocampista")


def percentil(valor, serie: pd.Series, lower: bool = False) -> float:
    datos = serie.dropna().values
    if not len(datos) or pd.isna(valor):
        return 0.0
    p = percentileofscore(datos, float(valor), kind="rank")
    return round(100.0 - p if lower else p, 1)


def slice_color(pct: float, base_hex: str) -> str:
    br, bg, bb = mcolors.to_rgb(base_hex)
    dr, dg, db = mcolors.to_rgb("#1a1a2e")  # match gradient center
    t = 0.14 + 0.86 * (max(pct, 0.0) / 100.0)
    return mcolors.to_hex((dr + t*(br-dr), dg + t*(bg-dg), db + t*(bb-db)))


# ── Generación del gráfico ─────────────────────────────────────────────────────

def make_chart(
    jug: pd.Series,
    etiquetas: list[str],
    pcts: list[float],
    raws: list[float],
    grupo: str,
    base_color: str,
    bebas: str | None,
    player_headshot: np.ndarray | None,
    team_logo: np.ndarray | None,
    n_ref: int,
    dpi: int = 150,
) -> Path:
    nombre = jug["nombre"]
    equipo = jug["equipo"]
    rating = jug["rating"]

    pos_es = {"delantero": "Delantero", "mediocampista": "Mediocampista",
              "defensa": "Defensa", "portero": "Portero"}.get(grupo, grupo.title())
    rat_s  = f"{rating:.2f}" if pd.notna(rating) else "N/D"
    min_s  = f"{int(jug['minutos_stats'])}" if pd.notna(jug.get("minutos_stats")) else "?"

    ff_title = bebas or "DejaVu Sans"
    INNER    = 22   # hueco central del radar

    # ── Figura 7×9.5 in — más grande = más nitidez al agrandar ────────────────
    FIG_W, FIG_H = 7, 9.5
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="none")

    # ── Fondo: gradiente radial oscuro ────────────────────────────────────────
    bg_w, bg_h = int(FIG_W * dpi), int(FIG_H * dpi)
    _cx, _cy   = bg_w / 2, bg_h / 2
    _Yg, _Xg   = np.ogrid[:bg_h, :bg_w]
    _t = np.clip(
        np.sqrt((_Xg - _cx)**2 + (_Yg - _cy)**2) / np.sqrt(_cx**2 + _cy**2),
        0, 1)
    _c1 = np.array(mcolors.to_rgb("#1a1a2e"))
    _c2 = np.array(mcolors.to_rgb("#0d0d1a"))
    bg_rgb = (_c1 + _t[..., None] * (_c2 - _c1))   # (h, w, 3) float [0,1]

    ax_bg = fig.add_axes([0, 0, 1, 1], facecolor="none", label="bg")
    ax_bg.imshow(bg_rgb, aspect="auto", extent=[0, 1, 0, 1],
                 origin="lower", interpolation="bilinear")
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.axis("off")
    ax_bg.set_zorder(-100)

    # ── Layout ────────────────────────────────────────────────────────────────
    # Radar: eje cuadrado físico.  AX_W = AX_H * FIG_H/FIG_W
    # AX_H=0.58 → AX_W=0.773 → AX_L=0.113 (0.68in lateral)
    # Con param_location=106 el label lateral queda a 0.15in del borde → NO se corta.
    AX_B  = 0.07                          # margen inferior (footer)
    AX_H  = 0.58                          # altura del radar (fracción de figura)
    AX_W  = AX_H * FIG_H / FIG_W         # = 0.58 * 8/6 = 0.773 → cuadrado físico
    AX_L  = (1.0 - AX_W) / 2             # = 0.113 → 0.68in de margen lateral

    PIZZA_TOP = AX_B + AX_H              # 0.65
    # Espacio entre pizza y banner para que labels no se encimen con el header
    BANNER_Y  = PIZZA_TOP + 0.10         # 0.75
    BANNER_H  = 1.0 - BANNER_Y           # 0.28
    BANNER_CY = BANNER_Y + BANNER_H / 2  # 0.86

    # ── Polar axes ────────────────────────────────────────────────────────────
    ax = fig.add_axes([AX_L, AX_B, AX_W, AX_H], projection="polar",
                      facecolor="none")
    ax.set_zorder(1)

    # ── PyPizza ───────────────────────────────────────────────────────────────
    baker = PyPizza(
        params=etiquetas,
        background_color="#1a1a2e",
        straight_line_color="#252545",
        straight_line_lw=0.8,
        last_circle_color="#3a3f5e",
        last_circle_lw=2.0,
        other_circle_lw=0,
        inner_circle_size=INNER,
    )

    baker.make_pizza(
        values=pcts,
        ax=ax,
        color_blank_space="same",
        blank_alpha=0.07,
        slice_colors=[slice_color(p, base_color) for p in pcts],
        value_colors=["#ffffff"] * len(pcts),
        param_location=104,          # etiquetas justo fuera del outer ring (ylim=100)
        kwargs_slices={
            "edgecolor": "#1a1a2e",
            "linewidth": 2.0,
            "zorder": 2,
        },
        kwargs_params={
            "color":      "none",   # se ocultan; los redibujamos via fig.text()
            "fontsize":   0.01,
        },
        kwargs_values={
            "color":      "#ffffff",
            "fontsize":   15,
            "fontweight": "bold",
            "fontfamily": ff_title,
            "va":         "center",
            "zorder":     5,
        },
    )

    # ── Etiquetas: posicionadas analíticamente y dibujadas via fig.text() ────────
    # Los textos en axes polar son cortados por el clip path del axes aunque
    # clip_on=False.  fig.text() está en coordenadas de figura → sin clip.
    # Slice i centrado en θ (standard) = 90° − i*(360/n), clockwise desde top.
    ylim_max = ax.get_ylim()[1]               # = 100.0  (PyPizza lo fija)
    n_sl   = len(etiquetas)
    AX_CX  = AX_L + AX_W / 2                 # 0.500
    AX_CY  = AX_B + AX_H / 2                 # 0.360
    PLOC   = 102                              # 2% más allá del outer ring (ylim=100)
    RX = (AX_W / 2) * (PLOC / ylim_max)      # radio x en fracciones de figura
    RY = (AX_H / 2) * (PLOC / ylim_max)      # radio y en fracciones de figura

    for i, label in enumerate(etiquetas):
        theta = np.deg2rad(90.0 - i * 360.0 / n_sl)   # θ counterclockwise from right
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_f = AX_CX + RX * cos_t
        y_f = AX_CY + RY * sin_t

        # Alineación direccional: el texto siempre se extiende HACIA AFUERA del radar
        # El punto de anclaje queda en el borde exterior y nunca se encima con el radar
        THRESH = 0.35
        if cos_t > THRESH:
            ha = "left"
        elif cos_t < -THRESH:
            ha = "right"
        else:
            ha = "center"

        if sin_t > THRESH:
            va = "bottom"
        elif sin_t < -THRESH:
            va = "top"
        else:
            va = "center"

        fig.text(
            x_f, y_f, label,
            ha=ha, va=va,
            fontsize=11, fontweight="bold",
            color="#d8dff0", fontfamily=ff_title,
            clip_on=False,
        )

    # ── Percentiles: tamaño adaptado ─────────────────────────────────────────
    for txt, pct in zip(baker.get_value_texts(), pcts):
        txt.set_fontsize(15 if pct >= 15 else 10)
        txt.set_fontweight("bold")
        txt.set_clip_on(False)
        txt.set_clip_path(None)

    # Desactivar clip del axes polar para que textos externos no queden cortados
    ax.set_clip_on(False)

    # ── Anillo p50 ────────────────────────────────────────────────────────────
    p50_r = INNER + 50
    theta_ring = np.linspace(0, 2 * np.pi, 500)
    ax.plot(theta_ring, [p50_r] * 500,
            color="#3a4a6a", lw=1.0, ls="--", zorder=3, alpha=0.75)
    ax.text(0.12, p50_r + 2, "p50", color="#4a5a7a",
            fontsize=8, va="bottom", ha="left", zorder=4)

    # ── Rating en el centro del radar ─────────────────────────────────────────
    ax.text(
        0, 0, rat_s,
        transform=ax.transData,
        ha="center", va="center",
        fontsize=18, fontweight="bold", color="#ffffff",
        zorder=15,
        bbox=dict(boxstyle="round,pad=0.55",
                  facecolor=base_color, alpha=0.88, edgecolor="none"),
    )

    # ── Línea separadora entre radar y banner ─────────────────────────────────
    sep_y = BANNER_Y - 0.004
    fig.add_artist(plt.Line2D(
        [0.04, 0.96], [sep_y, sep_y],
        transform=fig.transFigure,
        color="#3a4a6a", linewidth=0.7, alpha=0.55, zorder=20,
    ))

    # ── BANNER ────────────────────────────────────────────────────────────────
    fig.patches.append(
        plt.Rectangle((0, BANNER_Y), 1, BANNER_H,
                       transform=fig.transFigure,
                       color=base_color, alpha=0.97, zorder=-1)
    )

    # Logo/headshot: 0.72 in de lado (constante física, independiente del DPI)
    ELEM_IN = 0.72
    ELEM_W  = ELEM_IN / FIG_W
    ELEM_H  = ELEM_IN / FIG_H

    # Escudo: center_x=0.10
    if team_logo is not None:
        ax_logo = fig.add_axes([
            0.10 - ELEM_W / 2,
            BANNER_CY - ELEM_H / 2,
            ELEM_W, ELEM_H,
        ], facecolor="none")
        ax_logo.imshow(team_logo, aspect="equal", interpolation="lanczos")
        ax_logo.axis("off")
        ax_logo.set_zorder(10)

    # Headshot circular con borde blanco: center_x=0.90
    if player_headshot is not None:
        ax_hs = fig.add_axes([
            0.90 - ELEM_W / 2,
            BANNER_CY - ELEM_H / 2,
            ELEM_W, ELEM_H,
        ], facecolor="none")
        ax_hs.imshow(player_headshot, aspect="equal")
        ax_hs.axis("off")
        ax_hs.set_zorder(10)

    # Nombre: parte alta del banner
    name_y = BANNER_Y + BANNER_H * 0.78
    fig.text(
        0.50, name_y, nombre.upper(),
        ha="center", va="center",
        fontsize=36, fontweight="bold",
        color="#ffffff", fontfamily=ff_title,
    )

    # Info: una línea debajo del nombre
    info_y = BANNER_Y + BANNER_H * 0.28
    fig.text(
        0.50, info_y,
        f"{equipo}   ·   {pos_es}   ·   Rating {rat_s}   ·   {min_s} min",
        ha="center", va="center",
        fontsize=10.5, color="#eef0fa", fontfamily=ff_title,
    )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    fig.text(
        0.03, 0.020, "Fuente: FotMob",
        ha="left", va="bottom",
        fontsize=10, color="#666666",
    )
    fig.text(
        0.50, 0.020,
        f"Percentiles vs {pos_es}s ≥{MIN_MINUTOS} min  (n={n_ref})  ·  LigaMX Clausura 2025/26",
        ha="center", va="bottom",
        fontsize=7.5, color="#444466", style="italic",
    )
    fig.text(
        0.97, 0.020, "MAU-STATISTICS",
        ha="right", va="bottom",
        fontsize=14, color=base_color,
        fontfamily=ff_title, fontweight="bold", alpha=0.95,
    )

    # ── Guardar ───────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = (f"pizza_{nombre.lower().replace(' ','_')}"
            f"_{equipo.lower().replace(' ','_')}.png")
    out = OUTPUT_DIR / slug

    # fig.text() es capturado correctamente por tight bbox (a diferencia de axes text)
    plt.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.20,
                transparent=True, edgecolor="none")
    plt.close()
    return out


# ── Genera una gráfica para un jugador dado su nombre ──────────────────────────

def _build_and_save(jug: pd.Series, df: pd.DataFrame, bebas: str | None,
                    color_override: str | None, dpi: int) -> None:
    """Núcleo compartido: calcula percentiles y llama a make_chart."""
    grupo  = jug["grupo"]
    equipo = jug["equipo"]

    ref = df[
        (df["grupo"] == grupo) &
        (df["minutos_stats"].fillna(0) >= MIN_MINUTOS)
    ]

    print(f"\n{'═'*60}")
    print(f"  Jugador : {jug['nombre']}  ·  {equipo}  ·  {jug['posicion']}")
    print(f"  Grupo   : {grupo}  |  Referencia: {len(ref)} jugadores")

    metricas = METRICAS.get(grupo, METRICAS["mediocampista"])
    etiquetas, pcts, raws = [], [], []
    for col, label, lower in metricas:
        rawf = float(jug.get(col)) if pd.notna(jug.get(col)) else np.nan
        p    = percentil(rawf, ref[col], lower=lower)
        etiquetas.append(label)
        pcts.append(p)
        raws.append(rawf)
        sym = "↓" if lower else " "
        print(f"  {sym} {label.replace(chr(10),' '):<42} "
              f"{rawf if not np.isnan(rawf) else 'N/D':>8}  →  {p:>5.1f}%")

    player_id = int(jug["id"])
    team_id   = int(jug["equipo_id"]) if "equipo_id" in jug.index else None

    print("\n  Imágenes...")
    player_headshot = get_player_headshot(player_id)
    team_logo = get_team_logo(team_id, equipo, size=180) if team_id else None
    print(f"  Headshot: {'OK' if player_headshot is not None else 'no disponible'}")
    print(f"  Escudo: {'OK' if team_logo is not None else 'no disponible'}")

    base_color = color_override or COLORES_EQUIPO.get(equipo, "#D5001C")

    out = make_chart(
        jug=jug,
        etiquetas=etiquetas,
        pcts=pcts,
        raws=raws,
        grupo=grupo,
        base_color=base_color,
        bebas=bebas,
        player_headshot=player_headshot,
        team_logo=team_logo,
        n_ref=len(ref),
        dpi=dpi,
    )
    print(f"  → Guardado: {out}  ({out.stat().st_size // 1024} KB)")


def generate_for(nombre_query: str, df: pd.DataFrame, bebas: str | None,
                 color_override: str | None, dpi: int) -> None:
    hits = df[df["nombre"].str.contains(nombre_query, case=False, na=False)]
    if hits.empty:
        print(f"[SKIP] No se encontró: '{nombre_query}'")
        return
    if len(hits) > 1:
        print(hits[["id","nombre","equipo","posicion"]].to_string(index=False))
        print(f"[SKIP] Varios resultados para '{nombre_query}' — sé más específico.")
        return
    _build_and_save(hits.iloc[0], df, bebas, color_override, dpi)


def generate_grupo(grupo: str, df: pd.DataFrame, bebas: str | None,
                   color_override: str | None, dpi: int) -> None:
    """Genera gráficas para todos los jugadores del grupo con ≥ MIN_MINUTOS."""
    candidatos = df[
        (df["grupo"] == grupo) &
        (df["minutos_stats"].fillna(0) >= MIN_MINUTOS)
    ].copy()

    if candidatos.empty:
        print(f"[SKIP] Sin jugadores de grupo '{grupo}' con ≥{MIN_MINUTOS} min.")
        return

    print(f"\nGenerando {len(candidatos)} gráficas para grupo '{grupo}'…")
    for _, jug in candidatos.iterrows():
        _build_and_save(jug, df, bebas, color_override, dpi)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nombre", type=str,  default=None)
    parser.add_argument("--grupo",  type=str,  default=None,
                        choices=["delantero","mediocampista","defensa","portero"],
                        help="Genera gráficas para todos los jugadores del grupo con ≥MIN_MINUTOS")
    parser.add_argument("--color",  type=str,  default=None)
    parser.add_argument("--dpi",    type=int,  default=220)
    parser.add_argument("--todos",  action="store_true",
                        help="Genera 3 ejemplos: Paulinho, Marcel Ruiz, delantero Cruz Azul")
    parser.add_argument("--paleta", type=str, default=None, choices=list(PALETAS.keys()),
                        help="Nombre de la paleta de colores")
    args = parser.parse_args()
    if args.paleta:
        global BG_COLOR, GRAY_C, _pal
        _pal     = get_paleta(args.paleta)
        BG_COLOR = _pal['bg_primary']
        GRAY_C   = _pal['text_secondary']
    if args.nombre is None and args.grupo is None and not args.todos:
        args.nombre = "Paulinho"   # default

    # ── Fuente ────────────────────────────────────────────────────────────────
    print("Configurando fuente...")
    bebas = get_bebas_neue()
    print(f"  Fuente: {bebas or 'sistema (fallback)'}")

    # ── Datos ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df["pos_prim"] = df["posicion"].str.split(",").str[0].str.strip()
    df["grupo"]    = df["pos_prim"].map(GRUPOS_POS).fillna("mediocampista")

    if args.grupo:
        generate_grupo(args.grupo, df, bebas, args.color, args.dpi)
    elif args.todos:
        ca = df[
            (df["equipo"].str.contains("Cruz Azul", case=False)) &
            (df["grupo"] == "delantero")
        ].sort_values("minutos_stats", ascending=False)
        ca_nombre = ca.iloc[0]["nombre"] if not ca.empty else "Gabriel Fernández"
        for nombre in ["Paulinho", "Marcel Ru", ca_nombre]:
            generate_for(nombre, df, bebas, args.color, args.dpi)
    else:
        generate_for(args.nombre, df, bebas, args.color, args.dpi)


if __name__ == "__main__":
    main()
