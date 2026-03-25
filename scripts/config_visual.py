"""
config_visual.py  –  Paleta de identidad MAU-STATISTICS
Importar en todos los scripts de visualización:
    from config_visual import PALETTE, bebas, hex_rgba, hex_rgb
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
    'bar_loser':  '#1a1a1a',
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
