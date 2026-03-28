"""
config_visual.py  –  Paleta de identidad MAU-STATISTICS
Importar en todos los scripts de visualización:
    from config_visual import PALETTE, PALETAS, PALETA_ACTIVA, get_paleta, bebas, hex_rgba, hex_rgb
"""

from pathlib import Path
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# ─────────────────────────────────────────────────────────────────────────────
# PALETA BASE
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    # Fondos
    'bg_main':    '#0d1117',   # fondo principal
    'bg_secondary': '#161b22', # fondo secundario / filas alternas
    'bg_card':    '#0f151e',   # tarjetas / paneles

    # Texto
    'text_primary':   '#ffffff',
    'text_secondary': '#8b949e',

    # Acentos
    'accent':     '#D5001C',   # rojo MAU-STATISTICS
    'positive':   '#2ea043',   # verde (delta positivo)
    'negative':   '#f85149',   # rojo claro (delta negativo)

    # Bordes / líneas
    'border':     '#30363d',
    'divider':    '#21262d',
    'grid':       '#1e2530',

    # Barras / tracks
    'bar_track':  '#181c24',
    'bar_loser':  '#252525',
    'bar_loser_border': '#444444',
}

# ─────────────────────────────────────────────────────────────────────────────
# FUENTE
# ─────────────────────────────────────────────────────────────────────────────
BEBAS_TTF = Path.home() / '.fonts/BebasNeue.ttf'

if BEBAS_TTF.exists():
    fm.fontManager.addfont(str(BEBAS_TTF))

def bebas(size: float) -> dict:
    """Devuelve kwargs de Bebas Neue para matplotlib text."""
    if BEBAS_TTF.exists():
        return {'fontproperties': FontProperties(fname=str(BEBAS_TTF), size=size)}
    return {'fontsize': size, 'fontweight': 'bold'}

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE COLOR
# ─────────────────────────────────────────────────────────────────────────────
def hex_rgb(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' → (R, G, B) enteros 0-255."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def hex_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """'#RRGGBB' → (r, g, b, a) floats 0-1."""
    r, g, b = hex_rgb(hex_color)
    return (r/255, g/255, b/255, alpha)

def darken(hex_color: str, factor: float = 0.55) -> tuple[int, int, int]:
    """Oscurece un color RGB por un factor (0=negro, 1=original)."""
    r, g, b = hex_rgb(hex_color)
    return (int(r * factor), int(g * factor), int(b * factor))

def make_h_gradient(color_hex: str, width: int = 256,
                    dark_factor: float = 0.45) -> 'np.ndarray':
    """Array RGBA (1, width, 4) con gradiente horizontal saturado→oscuro."""
    import numpy as np
    r, g, b = hex_rgb(color_hex)
    arr = np.zeros((1, width, 4), dtype=np.uint8)
    for j in range(width):
        t = j / (width - 1)          # 0 = izquierda (vibrante), 1 = derecha (oscuro)
        arr[0, j] = [
            int(r * (1 - t * dark_factor)),
            int(g * (1 - t * dark_factor)),
            int(b * (1 - t * dark_factor)),
            230,
        ]
    return arr

# ─────────────────────────────────────────────────────────────────────────────
# SISTEMA DE PALETAS INTERCAMBIABLES
# ─────────────────────────────────────────────────────────────────────────────
# Cada paleta tiene exactamente estas 10 claves:
#   bg_primary, bg_secondary, cell_high, cell_mid, cell_low,
#   accent, accent2, text_primary, text_secondary, brand_color

PALETAS = {
    'cyberpunk_quetzal': dict(
        bg_primary   = '#000000',
        bg_secondary = '#0a0a15',
        cell_high    = '#00FF88',
        cell_mid     = '#00C853',
        cell_low     = '#050510',
        accent       = '#D5001C',
        accent2      = '#e040fb',
        text_primary = '#ffffff',
        text_secondary = '#9090c0',
        brand_color  = '#00FF88',
    ),
    'matrix_neon': dict(
        bg_primary   = '#050510',
        bg_secondary = '#0d0d20',
        cell_high    = '#76ff03',
        cell_mid     = '#4caf50',
        cell_low     = '#080808',
        accent       = '#D5001C',
        accent2      = '#00e5ff',
        text_primary = '#ffffff',
        text_secondary = '#80a080',
        brand_color  = '#76ff03',
    ),
    'negro_selva': dict(
        bg_primary   = '#000000',
        bg_secondary = '#071a10',
        cell_high    = '#00FF88',
        cell_mid     = '#00C853',
        cell_low     = '#040f09',
        accent       = '#D5001C',
        accent2      = '#b2dfdb',
        text_primary = '#b2dfdb',
        text_secondary = '#669977',
        brand_color  = '#00FF88',
    ),
    'medianoche_neon': dict(
        bg_primary   = '#08080f',
        bg_secondary = '#12121f',
        cell_high    = '#00ffaa',
        cell_mid     = '#00aa77',
        cell_low     = '#080810',
        accent       = '#ff2d7b',
        accent2      = '#ff2d7b',
        text_primary = '#ffffff',
        text_secondary = '#8888bb',
        brand_color  = '#00ffaa',
    ),
    'radioactivo': dict(
        bg_primary   = '#000000',
        bg_secondary = '#0f0f0f',
        cell_high    = '#39ff14',
        cell_mid     = '#2abf10',
        cell_low     = '#080808',
        accent       = '#D5001C',
        accent2      = '#b6ff00',
        text_primary = '#ffffff',
        text_secondary = '#708060',
        brand_color  = '#39ff14',
    ),
    'oceano_esmeralda': dict(
        bg_primary   = '#040812',
        bg_secondary = '#0c1420',
        cell_high    = '#00e676',
        cell_mid     = '#009955',
        cell_low     = '#060e18',
        accent       = '#D5001C',
        accent2      = '#69f0ae',
        text_primary = '#e0f2f1',
        text_secondary = '#669977',
        brand_color  = '#00e676',
    ),
    'rojo_fuego': dict(
        bg_primary   = '#0a0000',
        bg_secondary = '#1a0505',
        cell_high    = '#FF0000',
        cell_mid     = '#D5001C',
        cell_low     = '#2a0a0a',
        accent       = '#FF0000',
        accent2      = '#FFD700',
        text_primary = '#ffffff',
        text_secondary = '#cc8888',
        brand_color  = '#FF0000',
    ),
}

PALETA_ACTIVA = 'cyberpunk_quetzal'


def get_paleta(name: str = None) -> dict:
    """Devuelve el dict de paleta por nombre o la paleta activa si name es None."""
    key = name or PALETA_ACTIVA
    if key not in PALETAS:
        raise ValueError(f"Paleta '{key}' no existe. Disponibles: {list(PALETAS)}")
    return PALETAS[key]
