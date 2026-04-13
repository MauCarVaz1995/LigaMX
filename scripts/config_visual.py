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
# FUENTE — Bebas Neue con fallback tolerante
# ─────────────────────────────────────────────────────────────────────────────
import os

# Buscar la fuente en múltiples ubicaciones posibles (local, CI, sistema)
_BEBAS_CANDIDATES = [
    Path(__file__).parent.parent / 'assets' / 'fonts' / 'BebasNeue-Regular.ttf',
    Path(__file__).parent / 'BebasNeue-Regular.ttf',
    Path.home() / '.fonts' / 'BebasNeue.ttf',
    Path.home() / '.fonts' / 'BebasNeue-Regular.ttf',
    Path('/usr/share/fonts/truetype/bebas-neue/BebasNeue-Regular.ttf'),
]
BEBAS_TTF = next((p for p in _BEBAS_CANDIDATES if p.exists()), None)

BEBAS_AVAILABLE = BEBAS_TTF is not None

def bebas(size: float, **kwargs) -> dict:
    """Devuelve kwargs de Bebas Neue para matplotlib text. Fallback a DejaVu Sans.

    Usa FontProperties(fname=...) directamente — no necesita addfont().
    """
    if BEBAS_AVAILABLE and BEBAS_TTF is not None:
        try:
            return {'fontproperties': FontProperties(fname=str(BEBAS_TTF), size=size), **kwargs}
        except Exception:
            pass
    return {'fontsize': size, 'fontfamily': 'DejaVu Sans', **kwargs}

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

PALETA_ACTIVA = 'rojo_fuego'


def get_paleta(name: str = None) -> dict:
    """Devuelve el dict de paleta por nombre o la paleta activa si name es None."""
    key = name or PALETA_ACTIVA
    if key not in PALETAS:
        raise ValueError(f"Paleta '{key}' no existe. Disponibles: {list(PALETAS)}")
    return PALETAS[key]


# ─────────────────────────────────────────────────────────────────────────────
# BANDERAS NACIONALES (flagcdn.com 80×60, cacheadas en data/raw/flags/)
# ─────────────────────────────────────────────────────────────────────────────
_FLAG_ISO = {
    # CONCACAF & América
    'Mexico': 'mx',             'United States': 'us',      'Canada': 'ca',
    'Costa Rica': 'cr',         'Honduras': 'hn',           'Panama': 'pa',
    'El Salvador': 'sv',        'Guatemala': 'gt',          'Jamaica': 'jm',
    'Cuba': 'cu',               'Haiti': 'ht',              'Trinidad and Tobago': 'tt',
    'Dominican Republic': 'do', 'Nicaragua': 'ni',          'Curaçao': 'cw',
    'Belize': 'bz',             'Guadeloupe': 'gp',         'Martinique': 'mq',
    'Puerto Rico': 'pr',        'Aruba': 'aw',              'Barbados': 'bb',
    'Saint Vincent and the Grenadines': 'vc', 'Anguilla': 'ai',
    'Cayman Islands': 'ky',     'Bahamas': 'bs',            'Virgin Islands': 'vi',
    'Bermuda': 'bm',            'Turks and Caicos Islands': 'tc',
    # Sudamérica
    'Brazil': 'br',             'Argentina': 'ar',          'Colombia': 'co',
    'Uruguay': 'uy',            'Chile': 'cl',              'Peru': 'pe',
    'Venezuela': 've',          'Ecuador': 'ec',            'Bolivia': 'bo',
    'Paraguay': 'py',
    # Europa Occidental
    'Portugal': 'pt',           'Spain': 'es',              'France': 'fr',
    'Germany': 'de',            'Italy': 'it',              'Netherlands': 'nl',
    'Belgium': 'be',            'England': 'gb-eng',        'Wales': 'gb-wls',
    'Scotland': 'gb-sct',       'Northern Ireland': 'gb-nir',
    'Republic of Ireland': 'ie','Ireland': 'ie',
    'Switzerland': 'ch',        'Austria': 'at',            'Sweden': 'se',
    'Norway': 'no',             'Denmark': 'dk',            'Finland': 'fi',
    'Iceland': 'is',            'Luxembourg': 'lu',
    # Europa del Este
    'Croatia': 'hr',            'Poland': 'pl',             'Czech Republic': 'cz',
    'Slovakia': 'sk',           'Hungary': 'hu',            'Romania': 'ro',
    'Serbia': 'rs',             'Ukraine': 'ua',            'Russia': 'ru',
    'Turkey': 'tr',             'Greece': 'gr',             'Albania': 'al',
    'Slovenia': 'si',           'Kosovo': 'xk',             'Georgia': 'ge',
    'Armenia': 'am',            'Israel': 'il',
    # Asia & Oceanía
    'Japan': 'jp',              'South Korea': 'kr',        'Iran': 'ir',
    'Saudi Arabia': 'sa',       'China PR': 'cn',           'Australia': 'au',
    'New Zealand': 'nz',        'Qatar': 'qa',              'United Arab Emirates': 'ae',
    'North Korea': 'kp',
    # África
    'Morocco': 'ma',            'Senegal': 'sn',            'Nigeria': 'ng',
    'Ghana': 'gh',              'Egypt': 'eg',              'Algeria': 'dz',
    'Ivory Coast': 'ci',        'Cameroon': 'cm',           'Tunisia': 'tn',
    'Mali': 'ml',               'Burkina Faso': 'bf',       'South Africa': 'za',
    'Zambia': 'zm',             'Cape Verde': 'cv',         'Liechtenstein': 'li',
    'Andorra': 'ad',            'San Marino': 'sm',         'Gibraltar': 'gi',
}


def get_escudo(pais: str, size: tuple = (40, 27)):
    """
    Descarga la bandera del país desde flagcdn.com (80×60 px),
    la cachea en data/raw/flags/{iso}.png y retorna un OffsetImage
    listo para usar con AnnotationBbox.
    Retorna None silenciosamente si el país no está en el diccionario
    o si la descarga falla.
    """
    iso = _FLAG_ISO.get(pais)
    if not iso:
        return None
    cache_dir = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'flags'
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    cache = cache_dir / f'{iso}.png'
    if not cache.exists():
        try:
            import urllib.request
            urllib.request.urlretrieve(
                f'https://flagcdn.com/80x60/{iso}.png', str(cache))
        except Exception:
            return None
    try:
        import numpy as np
        from PIL import Image as _PIL
        from matplotlib.offsetbox import OffsetImage
        resample = _PIL.Resampling.LANCZOS if hasattr(_PIL, 'Resampling') else _PIL.LANCZOS
        img = _PIL.open(cache).convert('RGBA').resize(size, resample)
        return OffsetImage(np.array(img), zoom=1.0)
    except Exception:
        return None
