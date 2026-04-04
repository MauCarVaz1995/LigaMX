# VISUAL_SYSTEM.md — Sistema visual MAU-STATISTICS

---

## Tipografía

| Elemento | Fuente | Tamaño |
|---|---|---|
| Títulos principales | **Bebas Neue** | 28–32pt |
| Nombres de equipos | Bebas Neue | 10–14pt |
| Firma MAU-STATISTICS | Bebas Neue | 14pt |
| Porcentaje favorito | Bebas Neue | 31pt (20% mayor que no-favorito) |
| Porcentaje no-favorito | Bebas Neue | 26pt |
| Datos / stats | System / bold | 7–11pt |
| Subtítulos | System | 8.5–9.5pt |
| IC 95% | System | 9pt, alpha=0.6 |

```python
# Uso en matplotlib
from config_visual import bebas
ax.text(x, y, "TEXTO", **bebas(32))
```

Fuente en: `~/.fonts/BebasNeue.ttf`

---

## Paletas de color

Todas tienen las mismas 10 claves. Paleta activa por defecto: `rojo_fuego`.

### `rojo_fuego` — Liga MX principal
```python
bg_primary   = '#0a0000'   # fondo principal
bg_secondary = '#1a0505'   # fondo secundario / paneles
cell_high    = '#FF0000'   # celdas más altas del heatmap
cell_mid     = '#D5001C'   # celdas intermedias
cell_low     = '#2a0a0a'   # celdas bajas
accent       = '#FF0000'   # acento / firma / líneas
accent2      = '#FFD700'   # acento secundario (dorado)
text_primary = '#ffffff'
text_secondary = '#cc8888'
brand_color  = '#FF0000'
```

### `medianoche_neon` — Liga MX preferida
```python
bg_primary   = '#08080f'
bg_secondary = '#12121f'
cell_high    = '#00ffaa'
cell_mid     = '#00aa77'
cell_low     = '#080810'
accent       = '#ff2d7b'   # magenta/rosa
accent2      = '#ff2d7b'
text_primary = '#ffffff'
text_secondary = '#8888bb'
brand_color  = '#00ffaa'
```

### `oceano_esmeralda` — Liga MX preferida
```python
bg_primary   = '#040812'
bg_secondary = '#0c1420'
cell_high    = '#00e676'
cell_mid     = '#009955'
cell_low     = '#060e18'
accent       = '#D5001C'
accent2      = '#69f0ae'
text_primary = '#e0f2f1'
text_secondary = '#669977'
brand_color  = '#00e676'
```

### `cyberpunk_quetzal` — Solo selecciones
```python
bg_primary   = '#000000'
bg_secondary = '#0a0a15'
cell_high    = '#00FF88'
cell_mid     = '#00C853'
cell_low     = '#050510'
accent       = '#D5001C'
accent2      = '#e040fb'   # púrpura
text_primary = '#ffffff'
text_secondary = '#9090c0'
brand_color  = '#00FF88'
```

### `matrix_neon` — Solo selecciones
```python
bg_primary   = '#050510'
bg_secondary = '#0d0d20'
cell_high    = '#76ff03'
cell_mid     = '#4caf50'
cell_low     = '#080808'
accent       = '#D5001C'
accent2      = '#00e5ff'
text_primary = '#ffffff'
text_secondary = '#80a080'
brand_color  = '#76ff03'
```

### `negro_selva` — Solo selecciones
```python
bg_primary   = '#000000'
bg_secondary = '#071a10'
cell_high    = '#00FF88'
cell_mid     = '#00C853'
cell_low     = '#040f09'
accent       = '#D5001C'
accent2      = '#b2dfdb'
text_primary = '#b2dfdb'
text_secondary = '#669977'
brand_color  = '#00FF88'
```

### `radioactivo` — Solo selecciones
```python
bg_primary   = '#000000'
bg_secondary = '#0f0f0f'
cell_high    = '#39ff14'
cell_mid     = '#2abf10'
cell_low     = '#080808'
accent       = '#D5001C'
accent2      = '#b6ff00'
text_primary = '#ffffff'
text_secondary = '#708060'
brand_color  = '#39ff14'
```

---

## Reglas de uso de paletas

| Contexto | Paletas permitidas | Paletas prohibidas |
|---|---|---|
| **Predicciones Liga MX** | `medianoche_neon`, `oceano_esmeralda`, `rojo_fuego` | Las otras 4 |
| **Predicciones Selecciones** | Todas las 7 | Ninguna |
| **Infografías post-partido** | Aleatoria entre todas | Ninguna |
| **Resumen de jornada** | `rojo_fuego` (por defecto) | Ninguna |
| **Monte Carlo** | Cualquiera | Ninguna |

Regla de no-repetición consecutiva: nunca usar la misma paleta dos veces seguidas.

---

## Tamaños de imagen por tipo

| Tipo de contenido | Dimensiones | DPI | Pulgadas |
|---|---|---|---|
| Predicción de partido | 1080×1080px | 150 | 7.2×7.2 |
| Ratings post-partido | 1080×1500px | 150 | 7.2×10.0 |
| Team stats comparativa | 1080×1080px | 150 | 7.2×7.2 |
| Ranking ELO selecciones | 1080×variable | 150 | — |
| Últimos 5 partidos | 1080×1350px | 150 | 7.2×9.0 |
| Pizza chart P90 | 1080×1080px | 150 | 7.2×7.2 |
| Resumen de jornada | 1080×variable | 150 | — |

---

## Layout estándar de predicción 1080×1080

```
┌─────────────────────────────────────────────┐  ← HEADER (24.5%)
│  ¿QUIÉN GANA HOY?                           │
│  [escudo] EQUIPO LOCAL    VS   VISITANTE [esc]│
│  ELO, λ info                                │
├─────────────────────────────────────────────┤  ← línea accent lw=3
│                                             │
│         GOLES  VISITANTE                    │
│    0    1    2    3    4    5               │
│ 5 │    │    │    │    │    │    │           │  ← HEATMAP GRADIENTE
│ 4 │    │    │    │    │    │    │    GOLES  │     (55% de la imagen)
│ 3 │    │    │    │    │    │    │    LOCAL  │
│ 2 │    │    │    │    │    │    │           │
│ 1 │    │    │    │    │    │    │           │
│ 0 │    │    │    │    │    │    │           │
├─────────────────────────────────────────────┤
│  GANA LOCAL  │  EMPATE  │  GANA VISITANTE  │  ← BLOQUES % (15.5%)
│    41.9%     │  25.8%   │     32.3%        │
│ IC 95%: [...] IC 95%: [...] IC 95%: [...]  │
├─────────────────────────────────────────────┤
│  Marcador más probable: EQUIPO 1-1 EQUIPO  │  ← FACT BAR (5.2%)
├─────────────────────────────────────────────┤
│  Modelo ELO + Poisson | @LigaMX_Stats  MAU │  ← FOOTER (5.5%)
└─────────────────────────────────────────────┘
```

### Heatmap degradado
```python
cmap = LinearSegmentedColormap.from_list(
    'ligamx', [BG, BG2, CMID, CHIGH], N=256)
# Normalizar: t = prob / prob_max → color = cmap(t)
```
- Diagonal marcada con borde dorado `#FFD700` (empates)
- Texto blanco si brightness < 0.55, negro si ≥ 0.55

### Bloques de probabilidad
- Favorito: `FAV_FS = 31` (Bebas), borde del color del equipo
- No-favoritos: `BASE_FS = 26` (Bebas), sin borde
- IC 95% debajo: `fontsize=9`, `alpha=0.6`, color `text_primary`

---

## Escudos y logos

### Liga MX
- Fuente: FotMob CDN `https://images.fotmob.com/image_resources/logo/teamlogo/{id}.png`
- Cache local: `data/raw/logos/ligamx/{nombre_equipo}.png`
- Tamaño nativo: 192×192px modo `P` (palette)
- Inserción: `fig.add_axes([x, y, LOGO_SZ, LOGO_SZ])` + `ax.imshow(logo_arr)`
- `LOGO_SZ = 0.067` (figure fraction) ≈ 72px en output 1080px
- Posición en header: local a la izquierda del panel, visitante a la derecha

### IDs FotMob por equipo
```python
TEAM_IDS = {
    'Monterrey': 7849, 'San Luis': 6358, 'Querétaro': 1943,
    'Toluca': 6618,    'Cruz Azul': 6578, 'Pachuca': 7848,
    'León': 1841,      'Atlas': 6577,     'Santos': 7857,
    'América': 6576,   'Chivas': 7807,    'Tigres': 8561,
    'Pumas': 1946,     'Necaxa': 1842,    'Tijuana': 162418,
    'Mazatlán': 1170234, 'FC Juárez': 649424, 'Puebla': 7847,
}
```
**Nota**: sufijo `_xh` da 403 Forbidden — usar URL sin sufijo.

### Selecciones nacionales
- Fuente: flagcdn.com `https://flagcdn.com/80x60/{iso}.png`
- Cache local: `data/raw/flags/{iso}.png`
- Función: `get_escudo(pais, size)` en `config_visual.py`

---

## Colores de equipo Liga MX

```python
TEAM_COLORS = {
    'Monterrey': '#003DA5', 'San Luis':  '#D52B1E',
    'Querétaro': '#1A7FCB', 'Toluca':    '#D5001C',
    'Cruz Azul': '#0047AB', 'Pachuca':   '#A8B8C8',
    'León':      '#2D8C3C', 'Atlas':     '#B22222',
    'Santos':    '#2E8B57', 'América':   '#FFD700',
    'Chivas':    '#CD1F2D', 'Tigres':    '#F5A623',
    'Pumas':     '#C8A84B', 'Necaxa':    '#D62828',
    'Tijuana':   '#C62828', 'FC Juárez': '#4CAF50',
    'Mazatlán':  '#9B59B6', 'Puebla':    '#2563EB',
}
```

---

## Scripts base por tipo de gráfica

| Tipo de gráfica | Script de referencia |
|---|---|
| Predicción Liga MX (diseño actual) | `gen_predicciones_ligamx_20260404.py` |
| Predicción Liga MX (círculos %) | `11_modelo_prediccion.py` |
| Predicción selecciones | `19_predicciones_hoy.py` |
| Selecciones 3 imágenes | `gen_predicciones_intl_3img.py` |
| Ratings post-partido | `05_viz_player_performance.py` |
| Pizza P90 | `05_radar_p90.py` |
| Ranking barras | `07_ranking_posicion.py` |
| Comparativo 1v1 | `08_comparativo_1v1.py` |
| Monte Carlo heatmap | `14_simulacion_montecarlo.py` |
| ELO evolución + ranking | `12_modelo_elo.py` |

---

## Estructura de carpetas de salida

```
output/charts/
├── predicciones/
│   ├── LigaMX_Clausura_2026/     ← estándar: {Liga}_{Torneo}_{Año}
│   └── (otras temporadas...)
├── predicciones_hoy/             ← selecciones del día
├── partidos/                     ← infografías post-partido
├── jornada{N}/                   ← resumen de jornada
├── paletas/                      ← comparativa de paletas
├── pizza_*.png
├── ranking_*.png
├── elo_*.png
└── montecarlo_*.png
```
